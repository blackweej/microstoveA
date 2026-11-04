/* test_cnn.c — T-P1, T-P2, T-C1~C4 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/internal.h"
#include "minitest.h"

#define GOLD_N 20
#define NX (3 * FCV_IN * FCV_IN)

static float x[NX], x2[NX];

static void test_preprocess_only(void) {
    static uint8_t img[480 * 640 * 3], flip[480 * 640 * 3];
    fcv_result r;
    int roi[4] = { 80, 0, 480, 480 };

    fcv_free();
    CHECK(fcv_preprocess(img, 640, 480, 640 * 3, FCV_PIX_BGR24, roi, x) == FCV_E_STATE);
    CHECK(fcv_init(NULL) == FCV_OK);
    CHECK(fcv_num_classes() == 0);
    CHECK(fcv_classify(img, 640, 480, 640 * 3, FCV_PIX_BGR24, roi, NULL, 0, &r) == FCV_E_STATE);

    /* T-P1 상수 영상 */
    memset(img, 100, sizeof img);
    CHECK(fcv_preprocess(img, 640, 480, 640 * 3, FCV_PIX_BGR24, roi, x) == FCV_OK);
    for (int i = 0; i < NX; i++) if (fabsf(x[i] - 100 / 255.0f) > 1e-6f) { CHECK(0); break; }

    /* T-P1 BGR과 RGB가 같은 내부 표현이 된다 */
    srand(1);
    for (int i = 0; i < 480 * 640; i++) {
        uint8_t r8 = (uint8_t)rand(), g8 = (uint8_t)rand(), b8 = (uint8_t)rand();
        img[i * 3] = r8; img[i * 3 + 1] = g8; img[i * 3 + 2] = b8;
        flip[i * 3] = b8; flip[i * 3 + 1] = g8; flip[i * 3 + 2] = r8;
    }
    CHECK(fcv_preprocess(img, 640, 480, 640 * 3, FCV_PIX_RGB24, roi, x) == FCV_OK);
    CHECK(fcv_preprocess(flip, 640, 480, 640 * 3, FCV_PIX_BGR24, roi, x2) == FCV_OK);
    CHECK(memcmp(x, x2, sizeof x) == 0);

    /* §5.2 면적 평균: 한 칸을 직접 평균한 값과 같다 */
    double sum = 0;
    for (int p = 0; p < 5; p++) for (int q = 0; q < 5; q++)
        sum += img[((10 * 5 + p) * 640 + 80 + 20 * 5 + q) * 3 + 1];
    CHECK(fabs(x[(1 * FCV_IN + 10) * FCV_IN + 20] - sum / 25 / 255) < 1e-6);

    /* §5.3 GRAY8은 세 채널에 같은 값 */
    CHECK(fcv_preprocess(img, 480, 480, 480, FCV_PIX_GRAY8, NULL, x) == FCV_OK);
    CHECK(memcmp(x, x + FCV_IN * FCV_IN, FCV_IN * FCV_IN * sizeof(float)) == 0);
    CHECK(memcmp(x, x + 2 * FCV_IN * FCV_IN, FCV_IN * FCV_IN * sizeof(float)) == 0);

    /* T-P2 */
    int not96[4] = { 0, 0, 100, 100 }, rect[4] = { 0, 0, 192, 96 }, outside[4] = { 200, 0, 480, 480 };
    int neg[4] = { -1, 0, 96, 96 }, tall[4] = { 0, 96, 480, 480 };
    CHECK(fcv_preprocess(img, 640, 480, 640 * 3, FCV_PIX_BGR24, not96, x) == FCV_E_ARG);
    CHECK(fcv_preprocess(img, 640, 480, 640 * 3, FCV_PIX_BGR24, rect, x) == FCV_E_ARG);
    CHECK(fcv_preprocess(img, 640, 480, 640 * 3, FCV_PIX_BGR24, outside, x) == FCV_E_ARG);
    CHECK(fcv_preprocess(img, 640, 480, 640 * 3, FCV_PIX_BGR24, neg, x) == FCV_E_ARG);
    CHECK(fcv_preprocess(img, 640, 480, 640 * 3, FCV_PIX_BGR24, tall, x) == FCV_E_ARG);
    CHECK(fcv_preprocess(img, 640, 480, 640 * 3, FCV_PIX_BGR24, NULL, x) == FCV_E_ARG);   /* §5.1 정사각형 아님 */
    fcv_free();
}

static long read_file(const char *path, void *dst, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    long got = (long)fread(dst, 1, n, f);
    fclose(f);
    return got;
}

static void test_golden(void) {
    static uint8_t images[GOLD_N * FCV_IN * FCV_IN * 3];
    static double logits[GOLD_N * 7];
    CHECK(read_file("tests/golden/images.bin", images, sizeof images) == (long)sizeof images);
    CHECK(read_file("tests/golden/logits.bin", logits, sizeof logits) == (long)sizeof logits);
    CHECK(fcv_init("tests/golden/model.fcv") == FCV_OK);
    CHECK(fcv_num_classes() == 7);
    CHECK(strcmp(fcv_class_name(0), "samgak_gimbap") == 0);
    CHECK(strcmp(fcv_class_name(6), "fried_rice") == 0);
    CHECK(fcv_class_name(7) == NULL && fcv_class_name(-1) == NULL);

    /* T-P1 정규화: 상수 영상 → (v/255 − μ)/σ */
    {
        static uint8_t flat[FCV_IN * FCV_IN * 3];
        memset(flat, 200, sizeof flat);
        CHECK(fcv_preprocess(flat, FCV_IN, FCV_IN, FCV_IN * 3, FCV_PIX_RGB24, NULL, x) == FCV_OK);
        for (int c = 0; c < 3; c++)
            CHECK(fabsf(x[c * FCV_IN * FCV_IN] - (200 / 255.0f - fcv_m.mu[c]) / fcv_m.sigma[c]) < 1e-5f);
    }

    /* T-C1 §6.6  |z_C − z_ref| ≤ 1e-4 · max(1, |z_ref|) */
    double worst = 0;
    for (int n = 0; n < GOLD_N; n++) {
        const uint8_t *img = images + (size_t)n * FCV_IN * FCV_IN * 3;
        float z[7], p[7], probs[7];
        CHECK(fcv_preprocess(img, FCV_IN, FCV_IN, FCV_IN * 3, FCV_PIX_RGB24, NULL, x) == FCV_OK);
        cnn_logits(x, z);
        for (int k = 0; k < 7; k++) {
            double ref = logits[n * 7 + k], lim = fabs(ref) > 1 ? fabs(ref) : 1;
            double err = fabs(z[k] - ref) / lim;
            if (err > worst) worst = err;
            CHECK(err <= 1e-4);
        }
        /* fcv_classify도 같은 결과를 낸다. 확률은 logit의 소프트맥스 */
        fcv_result r;
        cnn_softmax(z, 7, p);
        CHECK(fcv_classify(img, FCV_IN, FCV_IN, FCV_IN * 3, FCV_PIX_RGB24, NULL, probs, 7, &r) == FCV_OK);
        CHECK(memcmp(p, probs, sizeof p) == 0);
        CHECK(r.prob == probs[r.class_id]);
        CHECK(r.rejected == (r.prob < 0.5f));
    }
    printf("T-C1 worst relative logit error %.3g\n", worst);

    /* T-C4 fcv_classify 중 할당 0 */
    {
        fcv_result r;
        long before = fcv_malloc_count;
        for (int n = 0; n < 3; n++)
            CHECK(fcv_classify(images, FCV_IN, FCV_IN, FCV_IN * 3, FCV_PIX_RGB24, NULL, NULL, 0, &r) == FCV_OK);
        CHECK(fcv_malloc_count == before);
        CHECK(fcv_classify(images, FCV_IN, FCV_IN, FCV_IN * 3, FCV_PIX_RGB24, NULL, NULL, 0, NULL) == FCV_E_ARG);
        float one;
        CHECK(fcv_classify(images, FCV_IN, FCV_IN, FCV_IN * 3, FCV_PIX_RGB24, NULL, &one, 1, &r) == FCV_OK);
    }
    fcv_free();
}

/* T-C2 max→ReLU와 ReLU→max의 차이는 정확히 0 */
static void test_fusion(void) {
    enum { S = 24 };
    static float xpad[2 * (S + 2) * (S + 2)], w[2 * 9], P[S * S], fused[(S / 2) * (S / 2)];
    srand(2);
    for (size_t i = 0; i < sizeof xpad / sizeof *xpad; i++) xpad[i] = rand() / (float)RAND_MAX - 0.5f;
    for (int i = 0; i < 18; i++) w[i] = rand() / (float)RAND_MAX - 0.5f;
    cnn_conv_plane(xpad, 2, S, S, w, -0.1f, P);
    cnn_pool_relu(P, S, S, fused);
    int any_zero = 0, any_pos = 0;
    for (int i = 0; i < S / 2; i++) for (int j = 0; j < S / 2; j++) {
        float m = 0;                                    /* ReLU를 먼저 하고 최댓값 */
        for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) {
            float v = P[(2 * i + a) * S + 2 * j + b];
            v = v > 0 ? v : 0;
            if (v > m) m = v;
        }
        CHECK(m == fused[i * (S / 2) + j]);
        any_zero |= m == 0; any_pos |= m > 0;
    }
    CHECK(any_zero && any_pos);
}

/* T-C3 */
static void test_softmax(void) {
    float z[3] = { 1000, 1001, 1002 }, p[3];
    cnn_softmax(z, 3, p);
    CHECK(fabsf(p[0] - 0.090f) < 1e-3f && fabsf(p[1] - 0.245f) < 1e-3f && fabsf(p[2] - 0.665f) < 1e-3f);
    CHECK(!isnan(p[0]) && !isnan(p[1]) && !isnan(p[2]));
}

void test_cnn(void) {
    test_preprocess_only();
    test_golden();
    test_fusion();
    test_softmax();
}
