/* internal.h — 모듈 사이 공유 상태와 테스트가 부르는 내부 함수 (§2). DLL로 나가지 않는다 */
#ifndef FCV_INTERNAL_H
#define FCV_INTERNAL_H
#include <stddef.h>
#include <stdint.h>
#include "foodcv.h"

#define FCV_IN 96
#define FCV_STAGES 4
#define FCV_GAP_C 64
#define FCV_MAX_CLASSES 64
#define FCV_NAME_LEN 32

/* T-C4: FCV_DEBUG_DUMP 빌드에서는 할당을 센다 */
#ifdef FCV_DEBUG_DUMP
void *fcv_counted_malloc(size_t n);
extern long fcv_malloc_count;
#  define FCV_MALLOC(n) fcv_counted_malloc(n)
#else
#  include <stdlib.h>
#  define FCV_MALLOC(n) malloc(n)
#endif

enum { FCV_STATE_NONE = 0, FCV_STATE_PREPROCESS = 1, FCV_STATE_READY = 2 };

typedef struct {
    int K;
    float mu[3], sigma[3], tau;
    char names[FCV_MAX_CLASSES][FCV_NAME_LEN];
    const float *w[FCV_STAGES], *b[FCV_STAGES];   /* W′, b′ */
    const float *fc_w, *fc_b;                     /* A, b */
    uint32_t hash;
} fcv_model;

extern fcv_model fcv_m;
extern int fcv_state;
extern const int fcv_channels[FCV_STAGES + 1];

/* cnn.c */
int  cnn_init(void);
void cnn_free(void);
void cnn_conv_plane(const float *xp, int Cin, int H, int W, const float *w, float bias, float *P);
void cnn_pool_relu(const float *P, int H, int W, float *y);
void cnn_logits(const float *x, float *z);        /* x: 3×96×96 CHW, z: K개 */
void cnn_softmax(const float *z, int n, float *p);

/* seg.c */
int  seg_solve_homography(const double quad[8], double canon_w, double canon_h, double h[8]);
int  seg_digit_code(int digit);
int  seg_otsu(const float *v, int n, float *threshold, float *contrast, float *margin);

#endif
