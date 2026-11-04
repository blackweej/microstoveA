/* weights.c — model.fcv 해석 (§7.1) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "internal.h"

#define FILE_VERSION 2
#define HEADER_BYTES 68     /* 매직 4, 버전 4, K·입력·단 12, 채널 20, μσ 24, τ 4 */

fcv_model fcv_m;
int fcv_state = FCV_STATE_NONE;
const int fcv_channels[FCV_STAGES + 1] = { 3, 16, 32, 64, 64 };

static float *params;       /* 단별 W′, b′, FC A, b */

static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

/* §7.1-9 FNV-1a 32비트 */
static uint32_t fnv1a(const uint8_t *p, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

static size_t param_count(int K) {
    size_t n = 0;
    for (int s = 0; s < FCV_STAGES; s++) {
        size_t ci = (size_t)fcv_channels[s], co = (size_t)fcv_channels[s + 1];
        n += co * ci * 9 + co;
    }
    return n + (size_t)K * FCV_GAP_C + (size_t)K;
}

/* 파일 전체를 읽는다. 호출자가 free */
static int read_all(const char *path, uint8_t **data, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return FCV_E_FILE;
    long n = -1;
    if (fseek(f, 0, SEEK_END) == 0) n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return FCV_E_FILE; }
    uint8_t *buf = FCV_MALLOC(n ? (size_t)n : 1);
    if (!buf) { fclose(f); return FCV_E_NOMEM; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return FCV_E_FILE; }
    fclose(f);
    *data = buf; *size = (size_t)n;
    return FCV_OK;
}

static int parse(const uint8_t *d, size_t size) {
    if (size < HEADER_BYTES) return FCV_E_FORMAT;
    if (memcmp(d, "FCV1", 4) != 0) return FCV_E_FORMAT;
    if (rd_u32(d + 4) != FILE_VERSION) return FCV_E_VERSION;
    uint32_t K = rd_u32(d + 8);
    if (K < 1 || K > FCV_MAX_CLASSES) return FCV_E_FORMAT;
    if (rd_u32(d + 12) != FCV_IN || rd_u32(d + 16) != FCV_STAGES) return FCV_E_FORMAT;
    for (int i = 0; i <= FCV_STAGES; i++)
        if (rd_u32(d + 20 + 4 * i) != (uint32_t)fcv_channels[i]) return FCV_E_FORMAT;

    size_t names = (size_t)K * FCV_NAME_LEN, np = param_count((int)K);
    size_t body = HEADER_BYTES + names + np * sizeof(float);
    if (size != body + 4) return FCV_E_FORMAT;
    if (fnv1a(d, body) != rd_u32(d + body)) return FCV_E_FORMAT;

    memset(&fcv_m, 0, sizeof fcv_m);
    fcv_m.K = (int)K;
    memcpy(fcv_m.mu, d + 40, sizeof fcv_m.mu);
    memcpy(fcv_m.sigma, d + 52, sizeof fcv_m.sigma);
    memcpy(&fcv_m.tau, d + 64, sizeof fcv_m.tau);
    for (int c = 0; c < 3; c++) if (!(fcv_m.sigma[c] > 0)) return FCV_E_FORMAT;
    for (uint32_t k = 0; k < K; k++) {
        memcpy(fcv_m.names[k], d + HEADER_BYTES + (size_t)k * FCV_NAME_LEN, FCV_NAME_LEN);
        fcv_m.names[k][FCV_NAME_LEN - 1] = 0;
    }
    params = FCV_MALLOC(np * sizeof(float));
    if (!params) return FCV_E_NOMEM;
    memcpy(params, d + HEADER_BYTES + names, np * sizeof(float));

    const float *p = params;
    for (int s = 0; s < FCV_STAGES; s++) {
        int ci = fcv_channels[s], co = fcv_channels[s + 1];
        fcv_m.w[s] = p; p += (size_t)co * ci * 9;
        fcv_m.b[s] = p; p += co;
    }
    fcv_m.fc_w = p; p += (size_t)K * FCV_GAP_C;
    fcv_m.fc_b = p;
    fcv_m.hash = rd_u32(d + body);
    return FCV_OK;
}

void fcv_free(void) {
    cnn_free();
    free(params);
    params = NULL;
    memset(&fcv_m, 0, sizeof fcv_m);
    fcv_state = FCV_STATE_NONE;
}

int fcv_init(const char *weight_path) {
    fcv_free();
    if (!weight_path) {                 /* §5.4 전처리 전용 상태 */
        for (int c = 0; c < 3; c++) fcv_m.sigma[c] = 1.0f;
        fcv_state = FCV_STATE_PREPROCESS;
        return FCV_OK;
    }
    uint8_t *data; size_t size;
    int rc = read_all(weight_path, &data, &size);
    if (rc) return rc;
    rc = parse(data, size);
    free(data);
    if (!rc) rc = cnn_init();
    if (rc) { fcv_free(); return rc; }
    fcv_state = FCV_STATE_READY;
    return FCV_OK;
}

int fcv_num_classes(void) {
    if (fcv_state == FCV_STATE_NONE) return FCV_E_STATE;
    return fcv_m.K;
}

const char *fcv_class_name(int k) {
    return fcv_state == FCV_STATE_READY && k >= 0 && k < fcv_m.K ? fcv_m.names[k] : NULL;
}

uint32_t fcv_model_hash(void) { return fcv_m.hash; }
