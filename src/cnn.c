/* cnn.c — 합성곱 엔진 (§6) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "internal.h"

/* §6.4 버퍼: 패딩 입력 40,000 + 풀링 출력 36,864 + 평면 9,216 float */
#define PAD_FLOATS   (16 * 50 * 50)
#define OUT_FLOATS   (16 * 48 * 48)
#define PLANE_FLOATS (FCV_IN * FCV_IN)

static float *buf;
static float *xp, *y, *plane;

int cnn_init(void) {
    buf = FCV_MALLOC((PAD_FLOATS + OUT_FLOATS + PLANE_FLOATS) * sizeof(float));
    if (!buf) return FCV_E_NOMEM;
    xp = buf; y = buf + PAD_FLOATS; plane = y + OUT_FLOATS;
    return FCV_OK;
}

void cnn_free(void) { free(buf); buf = xp = y = plane = NULL; }

/* 입력에 테두리 0을 붙인다 (§6.3 패딩 입력 x~) */
static void pad_input(const float *x, int C, int H, int W, float *out) {
    memset(out, 0, (size_t)C * (H + 2) * (W + 2) * sizeof(float));
    for (int c = 0; c < C; c++) for (int i = 0; i < H; i++)
        memcpy(out + ((size_t)c * (H + 2) + i + 1) * (W + 2) + 1, x + ((size_t)c * H + i) * W, (size_t)W * sizeof(float));
}

/* §6.3 출력 채널 하나의 평면. w: [Cin][3][3] */
void cnn_conv_plane(const float *xpad, int Cin, int H, int W, const float *w, float bias, float *P) {
    for (int k = 0; k < H * W; k++) P[k] = bias;                                /* b'_o */
    for (int c = 0; c < Cin; c++) for (int u = 0; u < 3; u++) for (int v = 0; v < 3; v++) {
        const float wv = w[(c * 3 + u) * 3 + v];                                /* W'_{o,c,u,v} */
        for (int i = 0; i < H; i++) {
            const float *xr = xpad + ((size_t)c * (H + 2) + i + u) * (W + 2) + v;
            float *pr = P + (size_t)i * W;
            for (int j = 0; j < W; j++) pr[j] += wv * xr[j];
        }
    }
}

/* §6.2  y[i][j] = ReLU(max(P[2i][2j], P[2i][2j+1], P[2i+1][2j], P[2i+1][2j+1])) */
void cnn_pool_relu(const float *P, int H, int W, float *out) {
    for (int i = 0; i < H / 2; i++) for (int j = 0; j < W / 2; j++) {
        const float *a = P + (size_t)(2 * i) * W + 2 * j, *b = a + W;
        float m = a[0];
        if (a[1] > m) m = a[1];
        if (b[0] > m) m = b[0];
        if (b[1] > m) m = b[1];
        out[(size_t)i * (W / 2) + j] = m > 0 ? m : 0;
    }
}

#ifdef FCV_DEBUG_DUMP
static void dump_stage(int s, const float *data, size_t n) {
    char name[32];
    snprintf(name, sizeof name, "dump_stage%d.bin", s);
    FILE *f = fopen(name, "wb");
    if (f) { fwrite(data, sizeof(float), n, f); fclose(f); }
}
#endif

void cnn_logits(const float *x, float *z) {
    int H = FCV_IN, W = FCV_IN;
    const float *src = x;
    for (int s = 0; s < FCV_STAGES; s++) {
        int Cin = fcv_channels[s], Cout = fcv_channels[s + 1];
        pad_input(src, Cin, H, W, xp);
        for (int o = 0; o < Cout; o++) {
            cnn_conv_plane(xp, Cin, H, W, fcv_m.w[s] + (size_t)o * Cin * 9, fcv_m.b[s][o], plane);
            cnn_pool_relu(plane, H, W, y + (size_t)o * (H / 2) * (W / 2));
        }
        src = y; H /= 2; W /= 2;
#ifdef FCV_DEBUG_DUMP
        dump_stage(s + 1, y, (size_t)Cout * H * W);
#endif
    }
    /* §6.5  g_c = (1/36) Σ y[c],  z_k = b_k + Σ_c A[k][c]·g_c */
    float g[FCV_GAP_C];
    for (int c = 0; c < FCV_GAP_C; c++) {
        float sum = 0;
        for (int k = 0; k < H * W; k++) sum += y[c * H * W + k];
        g[c] = sum / (float)(H * W);
    }
    for (int k = 0; k < fcv_m.K; k++) {
        float acc = fcv_m.fc_b[k];
        for (int c = 0; c < FCV_GAP_C; c++) acc += fcv_m.fc_w[k * FCV_GAP_C + c] * g[c];
        z[k] = acc;
    }
}

/* §6.5 m = max z를 빼서 계산한다 */
void cnn_softmax(const float *z, int n, float *p) {
    float m = z[0], sum = 0;
    for (int k = 1; k < n; k++) if (z[k] > m) m = z[k];
    for (int k = 0; k < n; k++) { p[k] = expf(z[k] - m); sum += p[k]; }
    for (int k = 0; k < n; k++) p[k] /= sum;
}

int fcv_classify(const uint8_t *img, int w, int h, int stride, int fmt, const int roi[4],
                 float *probs, int n_probs, fcv_result *out) {
    if (fcv_state != FCV_STATE_READY) return FCV_E_STATE;
    if (!out || (probs && n_probs < 0)) return FCV_E_ARG;
    /* 전처리 결과는 3×96×96 = 27,648 float라 풀링 출력 버퍼에 들어간다.
       cnn_logits가 먼저 패딩 버퍼로 옮기므로 같은 버퍼를 출력으로 써도 된다 */
    int rc = fcv_preprocess(img, w, h, stride, fmt, roi, y);
    if (rc) return rc;
    float z[FCV_MAX_CLASSES], p[FCV_MAX_CLASSES];
    cnn_logits(y, z);
    cnn_softmax(z, fcv_m.K, p);
    int best = 0;
    for (int k = 1; k < fcv_m.K; k++) if (p[k] > p[best]) best = k;
    out->class_id = best;
    out->prob = p[best];
    out->rejected = p[best] < fcv_m.tau;        /* §6.5 rejected = (p_max < τ) */
    if (probs) {
        int n = n_probs < fcv_m.K ? n_probs : fcv_m.K;
        memcpy(probs, p, (size_t)n * sizeof(float));
    }
    return FCV_OK;
}
