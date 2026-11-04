/* preprocess.c — 크롭·면적평균 축소·정규화 (§5) */
#include "internal.h"

int fcv_preprocess(const uint8_t *img, int w, int h, int stride, int fmt, const int roi[4], float *out) {
    if (fcv_state == FCV_STATE_NONE) return FCV_E_STATE;
    if (!img || !out || w <= 0 || h <= 0) return FCV_E_ARG;
    if (fmt != FCV_PIX_BGR24 && fmt != FCV_PIX_RGB24 && fmt != FCV_PIX_GRAY8) return FCV_E_ARG;
    int bpp = fmt == FCV_PIX_GRAY8 ? 1 : 3;
    if (stride < bpp * w) return FCV_E_ARG;

    /* §5.1 정사각형, 한 변은 96의 정수배, 영상 안 */
    int x0 = roi ? roi[0] : 0, y0 = roi ? roi[1] : 0, rw = roi ? roi[2] : w, rh = roi ? roi[3] : h;
    if (rw != rh || rw <= 0 || rw % FCV_IN) return FCV_E_ARG;
    if (x0 < 0 || y0 < 0 || rw > w - x0 || rh > h - y0) return FCV_E_ARG;

    /* §5.3 내부는 항상 RGB. BGR이면 읽는 순서만 바꾼다. GRAY8이면 세 채널에 같은 값 */
    const int off[3] = { fmt == FCV_PIX_BGR24 ? 2 : 0, fmt == FCV_PIX_GRAY8 ? 0 : 1,
                         fmt == FCV_PIX_BGR24 ? 0 : (fmt == FCV_PIX_GRAY8 ? 0 : 2) };
    const int s = rw / FCV_IN;
    const float inv = 1.0f / ((float)s * (float)s);
    for (int c = 0; c < 3; c++) {
        float mu = fcv_m.mu[c], sigma = fcv_m.sigma[c];
        for (int i = 0; i < FCV_IN; i++) for (int j = 0; j < FCV_IN; j++) {
            /* §5.2  x̂[c][i][j] = (1/s²) Σ_{p<s} Σ_{q<s} I[c][y0 + s·i + p][x0 + s·j + q] */
            uint32_t sum = 0;
            for (int p = 0; p < s; p++) {
                const uint8_t *row = img + (size_t)(y0 + s * i + p) * stride + (size_t)(x0 + s * j) * bpp + off[c];
                for (int q = 0; q < s; q++) sum += row[q * bpp];
            }
            /* §5.3  x = (x̂/255 − μ_c) / σ_c */
            out[(c * FCV_IN + i) * FCV_IN + j] = ((float)sum * inv / 255.0f - mu) / sigma;
        }
    }
    return FCV_OK;
}
