/* seg.c — 7-세그먼트 표시창 판독 (§8) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "internal.h"

#define MAX_DIGITS 4
#define MAX_VALUES (7 * MAX_DIGITS)

typedef struct {
    int digits, blank_lead, mss, use_max, dark;
    double canon_w, canon_h;
    float cmin;
    double seg[MAX_DIGITS][7][4];       /* x0 y0 x1 y1 */
    int code[10];
    int *grid, n_grid;
} layout_t;

static layout_t L;
static int ready = 0, has_quad = 0;
static double Hm[8];                    /* h11 h12 h13 h21 h22 h23 h31 h32 */

/* §8.4 비트 순서 a=bit0 … g=bit6 */
static const int default_code[10] = { 0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F };

int seg_digit_code(int digit) { return digit >= 0 && digit <= 9 ? default_code[digit] : -1; }

/* ---------------- §8.6 레이아웃 파서 ---------------- */

static int next_int(int *out) {
    const char *t = strtok(NULL, " \t\r");
    char *end;
    if (!t) return 0;
    long v = strtol(t, &end, 0);
    if (*end) return 0;
    *out = (int)v;
    return 1;
}

static int next_double(double *out) {
    const char *t = strtok(NULL, " \t\r");
    char *end;
    if (!t) return 0;
    *out = strtod(t, &end);
    return *end == 0;
}

static int parse_line(char *line, int *seen, unsigned *seg_seen, int cap) {
    char *hash = strchr(line, '#');
    if (hash) *hash = 0;
    const char *key = strtok(line, " \t\r");
    if (!key) return FCV_OK;
    if (!strcmp(key, "digits")) {
        if (!next_int(&L.digits) || L.digits < 1 || L.digits > MAX_DIGITS) return FCV_E_FORMAT;
        *seen |= 1;
    } else if (!strcmp(key, "canon")) {
        if (!next_double(&L.canon_w) || !next_double(&L.canon_h) || L.canon_w <= 0 || L.canon_h <= 0) return FCV_E_FORMAT;
        *seen |= 2;
    } else if (!strcmp(key, "channel")) {
        const char *v = strtok(NULL, " \t\r");
        if (!v || (strcmp(v, "R") && strcmp(v, "MAX"))) return FCV_E_FORMAT;
        L.use_max = !strcmp(v, "MAX");
        *seen |= 4;
    } else if (!strcmp(key, "polarity")) {
        const char *v = strtok(NULL, " \t\r");
        if (!v || (strcmp(v, "bright") && strcmp(v, "dark"))) return FCV_E_FORMAT;
        L.dark = !strcmp(v, "dark");
        *seen |= 8;
    } else if (!strcmp(key, "cmin")) {
        double v;
        if (!next_double(&v) || v < 0) return FCV_E_FORMAT;
        L.cmin = (float)v;
        *seen |= 16;
    } else if (!strcmp(key, "blank_lead")) {
        if (!next_int(&L.blank_lead) || L.blank_lead < 0 || L.blank_lead > MAX_DIGITS) return FCV_E_FORMAT;
        *seen |= 32;
    } else if (!strcmp(key, "format")) {
        const char *v = strtok(NULL, " \t\r");
        if (!v || (strcmp(v, "MMSS") && strcmp(v, "MSS"))) return FCV_E_FORMAT;
        L.mss = !strcmp(v, "MSS");
        *seen |= 64;
    } else if (!strcmp(key, "seg")) {
        int d; double r[4];
        const char *name;
        if (!next_int(&d) || d < 0 || d >= MAX_DIGITS) return FCV_E_FORMAT;
        name = strtok(NULL, " \t\r");
        if (!name || name[1] || name[0] < 'a' || name[0] > 'g') return FCV_E_FORMAT;
        for (int i = 0; i < 4; i++) if (!next_double(&r[i])) return FCV_E_FORMAT;
        if (r[2] <= r[0] || r[3] <= r[1]) return FCV_E_FORMAT;
        memcpy(L.seg[d][name[0] - 'a'], r, sizeof r);
        seg_seen[d] |= 1u << (name[0] - 'a');
    } else if (!strcmp(key, "glyph_variant")) {
        int d, code;
        if (!next_int(&d) || !next_int(&code) || d < 0 || d > 9 || code < 0 || code > 0x7F) return FCV_E_FORMAT;
        L.code[d] = code;
    } else if (!strcmp(key, "grid")) {
        int v;
        while (next_int(&v)) {
            if (L.n_grid >= cap || v < 0 || (L.n_grid && v <= L.grid[L.n_grid - 1])) return FCV_E_FORMAT;
            L.grid[L.n_grid++] = v;
        }
    } else if (strcmp(key, "fuse")) {   /* fuse는 래퍼가 읽는다 (§8.6) */
        return FCV_E_FORMAT;
    }
    return FCV_OK;
}

static int parse_layout(char *text, size_t size) {
    int seen = 0, cap = (int)(size / 2 + 1);
    unsigned seg_seen[MAX_DIGITS] = {0};
    L.grid = FCV_MALLOC((size_t)cap * sizeof(int));
    if (!L.grid) return FCV_E_NOMEM;
    memcpy(L.code, default_code, sizeof L.code);
    for (char *line = text, *nl; line; line = nl) {
        nl = strchr(line, '\n');
        if (nl) *nl++ = 0;
        int rc = parse_line(line, &seen, seg_seen, cap);
        if (rc) return rc;
    }
    if (seen != 127 || L.n_grid == 0) return FCV_E_FORMAT;      /* §8.6 격자 없이 구동 금지 */
    if (L.digits != (L.mss ? 3 : 4)) return FCV_E_FORMAT;
    for (int d = 0; d < L.digits; d++) {
        if (seg_seen[d] != 0x7F) return FCV_E_FORMAT;
        for (int s = 0; s < 7; s++)
            if (L.seg[d][s][0] < 0 || L.seg[d][s][1] < 0 || L.seg[d][s][2] > L.canon_w || L.seg[d][s][3] > L.canon_h)
                return FCV_E_FORMAT;
    }
    return FCV_OK;
}

void seg_free(void) {
    free(L.grid);
    memset(&L, 0, sizeof L);
    ready = has_quad = 0;
}

int seg_init(const char *layout_path) {
    seg_free();
    if (!layout_path) return FCV_E_ARG;
    FILE *f = fopen(layout_path, "rb");
    if (!f) return FCV_E_FILE;
    long n = -1;
    if (fseek(f, 0, SEEK_END) == 0) n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return FCV_E_FILE; }
    char *text = FCV_MALLOC((size_t)n + 1);
    if (!text) { fclose(f); return FCV_E_NOMEM; }
    size_t got = fread(text, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) { free(text); return FCV_E_FILE; }
    text[n] = 0;
    int rc = parse_layout(text, (size_t)n);
    free(text);
    if (rc) { seg_free(); return rc; }
    ready = 1;
    return FCV_OK;
}

/* ---------------- §8.1 호모그래피 ---------------- */

/* 정규 (0,0),(W,0),(W,H),(0,H) ↔ quad TL,TR,BR,BL.
   §8.1  u = (h11 x + h12 y + h13) / (h31 x + h32 y + 1). 분모를 곱한 일차식을 부분 피벗 가우스 소거로 푼다.
   T-S1이 1e-9를 재므로 double을 받는다 */
int seg_solve_homography(const double quad[8], double canon_w, double canon_h, double h[8]) {
    const double cx[4] = { 0, canon_w, canon_w, 0 }, cy[4] = { 0, 0, canon_h, canon_h };
    double A[8][9];
    for (int i = 0; i < 4; i++) {
        double x = cx[i], y = cy[i], u = quad[2 * i], v = quad[2 * i + 1];
        double r0[9] = { x, y, 1, 0, 0, 0, -u * x, -u * y, u };
        double r1[9] = { 0, 0, 0, x, y, 1, -v * x, -v * y, v };
        memcpy(A[2 * i], r0, sizeof r0);
        memcpy(A[2 * i + 1], r1, sizeof r1);
    }
    for (int c = 0; c < 8; c++) {
        int p = c;
        for (int r = c + 1; r < 8; r++) if (fabs(A[r][c]) > fabs(A[p][c])) p = r;
        if (!(fabs(A[p][c]) >= 1e-12)) return FCV_E_ARG;
        if (p != c) {
            double t[9];
            memcpy(t, A[c], sizeof t); memcpy(A[c], A[p], sizeof t); memcpy(A[p], t, sizeof t);
        }
        for (int r = c + 1; r < 8; r++) {
            double f = A[r][c] / A[c][c];
            for (int k = c; k < 9; k++) A[r][k] -= f * A[c][k];
        }
    }
    for (int r = 7; r >= 0; r--) {
        double s = A[r][8];
        for (int k = r + 1; k < 8; k++) s -= A[r][k] * h[k];
        h[r] = s / A[r][r];
    }
    return FCV_OK;
}

int seg_set_quad(const float quad[8]) {
    if (!ready) return FCV_E_STATE;
    if (!quad) return FCV_E_ARG;
    double q[8], h[8];
    for (int i = 0; i < 8; i++) q[i] = quad[i];
    int rc = seg_solve_homography(q, L.canon_w, L.canon_h, h);
    if (rc) return rc;
    memcpy(Hm, h, sizeof Hm);
    has_quad = 1;
    return FCV_OK;
}

/* ---------------- §8.2 획 밝기 ---------------- */

static inline float pixel(const uint8_t *img, int stride, int fmt, int x, int y) {
    if (fmt == FCV_PIX_GRAY8) return img[(size_t)y * stride + x];
    const uint8_t *q = img + (size_t)y * stride + 3 * (size_t)x;
    int R = fmt == FCV_PIX_BGR24 ? q[2] : q[0];
    if (!L.use_max) return (float)R;
    int m = R > q[1] ? R : q[1], o = fmt == FCV_PIX_BGR24 ? q[0] : q[2];
    return (float)(m > o ? m : o);
}

/* 쌍선형 보간. 픽셀 중심이 정수 좌표다. 영상 밖은 가장자리 값 */
static float sample(const uint8_t *img, int w, int h, int stride, int fmt, double u, double v) {
    if (u < 0) u = 0;
    if (v < 0) v = 0;
    if (u > w - 1) u = w - 1;
    if (v > h - 1) v = h - 1;
    int x0 = (int)u, y0 = (int)v, x1 = x0 + 1 < w ? x0 + 1 : x0, y1 = y0 + 1 < h ? y0 + 1 : y0;
    float fx = (float)(u - x0), fy = (float)(v - y0);
    float a = pixel(img, stride, fmt, x0, y0), b = pixel(img, stride, fmt, x1, y0);
    float c = pixel(img, stride, fmt, x0, y1), d = pixel(img, stride, fmt, x1, y1);
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

static float stroke(const uint8_t *img, int w, int h, int stride, int fmt, const double *r) {
    double sum = 0;
    int n = 0;
    for (double y = r[1]; y <= r[3]; y += 1) for (double x = r[0]; x <= r[2]; x += 1) {
        double den = Hm[6] * x + Hm[7] * y + 1;
        double u = (Hm[0] * x + Hm[1] * y + Hm[2]) / den, v = (Hm[3] * x + Hm[4] * y + Hm[5]) / den;
        sum += sample(img, w, h, stride, fmt, u, v);
        n++;
    }
    float m = (float)(sum / n);
    return L.dark ? 255 - m : m;        /* §8.2 polarity = dark면 255 − 값 */
}

/* §8.3 모든 경계에서 w0·w1·(μ1 − μ0)² 최대인 문턱. margin = 경계에 가장 가까운 값과 문턱의 거리 / 대비 */
int seg_otsu(const float *v, int n, float *threshold, float *contrast, float *margin) {
    float s[MAX_VALUES];
    if (!v || n < 2 || n > MAX_VALUES) return FCV_E_ARG;
    for (int i = 0; i < n; i++) {
        int k = i;
        while (k > 0 && s[k - 1] > v[i]) { s[k] = s[k - 1]; k--; }
        s[k] = v[i];
    }
    double total = 0, left = 0, best = -1;
    for (int i = 0; i < n; i++) total += s[i];
    int cut = 1;
    double mu0b = s[0], mu1b = s[n - 1];
    for (int k = 1; k < n; k++) {
        left += s[k - 1];
        double mu0 = left / k, mu1 = (total - left) / (n - k);
        double score = (double)k / n * (double)(n - k) / n * (mu1 - mu0) * (mu1 - mu0);
        if (score > best) { best = score; cut = k; mu0b = mu0; mu1b = mu1; }
    }
    float t = 0.5f * (s[cut - 1] + s[cut]), c = (float)(mu1b - mu0b);
    *threshold = t;
    *contrast = c;
    *margin = c > 0 ? (s[cut] - t) / c : 0;
    return FCV_OK;
}

/* ---------------- §8.4 디코딩 ---------------- */

int seg_read(const uint8_t *img, int w, int h, int stride, int fmt, seg_result *out) {
    if (!ready || !has_quad) return FCV_E_STATE;
    if (!img || !out || w <= 0 || h <= 0) return FCV_E_ARG;
    if (fmt != FCV_PIX_BGR24 && fmt != FCV_PIX_RGB24 && fmt != FCV_PIX_GRAY8) return FCV_E_ARG;
    if (stride < (fmt == FCV_PIX_GRAY8 ? w : 3 * w)) return FCV_E_ARG;

    float v[MAX_VALUES], thr, contrast;
    int n = 7 * L.digits;
    for (int d = 0; d < L.digits; d++) for (int s = 0; s < 7; s++)
        v[d * 7 + s] = stroke(img, w, h, stride, fmt, L.seg[d][s]);
    memset(out, 0, sizeof *out);
    out->seconds = -1;
    seg_otsu(v, n, &thr, &contrast, &out->margin);
    if (contrast < L.cmin) {            /* §8.3 */
        out->blank = 1;
        for (int d = 0; d < MAX_DIGITS; d++) out->digits[d] = -1;
        return FCV_OK;
    }
    int valid = 1;
    for (int d = 0; d < L.digits; d++) {
        int code = 0, digit = -1;
        for (int s = 0; s < 7; s++) code |= (v[d * 7 + s] > thr) << s;
        for (int k = 0; k < 10; k++) if (code == L.code[k]) digit = k;
        if (code == 0 && d < L.blank_lead) digit = 0;   /* 앞쪽 빈 자리만 값 0 */
        out->digits[d] = digit;
        valid &= digit >= 0;
    }
    if (valid) {
        const int *g = out->digits;
        out->seconds = L.mss ? g[0] * 60 + g[1] * 10 + g[2] : (g[0] * 10 + g[1]) * 60 + g[2] * 10 + g[3];
    }
    return FCV_OK;
}

/* ---------------- §8.5 판정 ---------------- */

int seg_grid_index(int seconds, int *index) {
    if (!ready) return FCV_E_STATE;
    if (!index) return FCV_E_ARG;
    *index = -1;
    for (int i = 0; i < L.n_grid; i++) if (L.grid[i] == seconds) *index = i;
    return FCV_OK;
}

static int usable(const seg_result *r) { return !r->blank && r->seconds >= 0; }

int seg_vote(const seg_result *r, int n, seg_result *out) {
    if (!ready) return FCV_E_STATE;
    if (!r || !out || n <= 0) return FCV_E_ARG;
    int best = -1;
    float best_margin = 0;
    for (int i = 0; i < n; i++) {
        if (!usable(&r[i])) continue;
        int count = 0;
        float margin = 0;
        for (int j = 0; j < n; j++) if (usable(&r[j]) && r[j].seconds == r[i].seconds) { count++; margin += r[j].margin; }
        if (count >= 3 && (best < 0 || margin > best_margin)) { best = i; best_margin = margin; }
    }
    memset(out, 0, sizeof *out);
    out->seconds = -1;
    if (best < 0) return SEG_UNSTABLE;

    /* §8.5-3 (a) 01:70과 02:10은 초가 같으므로 묶인 결과를 모두 본다 */
    int tens = L.mss ? 1 : 2;
    for (int j = 0; j < n; j++)
        if (usable(&r[j]) && r[j].seconds == r[best].seconds && r[j].digits[tens] > 5) return SEG_INVALID;
    int index;
    seg_grid_index(r[best].seconds, &index);
    if (index < 0) return SEG_INVALID;
    *out = r[best];
    return SEG_STABLE;
}
