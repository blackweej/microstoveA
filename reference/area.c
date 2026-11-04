/* area.c — 접시 위 기준 도형으로 음식 면적을 cm² 단위로 측정 (C11, 외부 라이브러리 없음)
 *
 * 수식 대응 (설계 문서 2–3절)
 *   빨강 판정   R >= r_min,  R >= k·G,  R >= k·B                          (2.1)
 *   스케일      s = A_ref / N_ref   [cm²/px]                                (2.2)
 *   충실도      F = N / (4π √det Σ),  Σ = 2차 중심 모멘트 + I/12           (2.3)
 *   배경 판정   |c − c̄| < τ_c  그리고  β_lo <= S / S_bg(x,y) <= β_hi        (3.1)
 *   면적        A_i = s · N_i,  A = Σ_i A_i                                 (3.3)
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define AREA_PI 3.14159265358979323846   /* M_PI는 C11 표준이 아니다 */
#ifdef _WIN32
#  define AREA_API __declspec(dllexport)
#else
#  define AREA_API
#endif

enum { AREA_OK = 0, AREA_REF_MISSING = 1, AREA_REF_CHANGED = 2, AREA_REF_MERGED = 3,
       AREA_EMPTY = 4, AREA_E_NOMEM = -4, AREA_E_ARG = -5, AREA_E_STATE = -6 };
enum { PIX_BGR24 = 0, PIX_RGB24 = 1 };
#define AREA_MAX_PARTS 16

typedef struct {
    float ref_cm2;              /* 기준 도형 실제 면적 A_ref (cm²) */
    float cx, cy, r;            /* 접시 ROI 원 (px) */
    float red_k;                /* 빨강 판정 비율 k */
    int   red_min;              /* 빨강 판정 R 하한 */
    float chroma_tol;           /* τ_c */
    float shade_lo, shade_hi;   /* β 허용 범위 (그림자 포함) */
    int   min_part_px;          /* 조각 최소 픽셀 수 */
    float fill_tol;             /* |F − 1| 허용치 */
    float count_tol;            /* |N_ref/N_cal − 1| 허용치 */
    int   ref_dilate;           /* 기준 도형 주변 제외 폭 (px) */
} area_params;

typedef struct {
    int   status;
    float food_cm2;             /* A */
    float scale;                /* s */
    int   ref_px;               /* N_ref */
    float ref_fill;             /* F */
    int   n_parts;
    float part_cm2[AREA_MAX_PARTS];
    float perim_px;             /* 음식 경계 길이 P ≈ (π/4)·경계 변 수 */
    int   touches_edge;         /* ROI 경계에 닿은 조각 수 */
} area_result;

typedef struct { double n, sx, sy, sxx, syy, sxy; int edge; } stat_t;

/* ---------------- 전역 상태: area_init에서만 할당 ---------------- */
static area_params P;
static int W, H, ready = 0, calibrated = 0;
static float *Sbg;              /* S_bg(x,y) */
static double *I1, *I2;         /* 보정용 적분 영상 */
static float cbar_r, cbar_g;    /* c̄ */
static int Ncal;
static uint8_t *m1, *m2, *roi, *edge;
static int32_t *lab, *par, *cid;
static stat_t *st;

static void free_all(void) {
    free(Sbg); free(I1); free(I2); free(m1); free(m2); free(roi); free(edge);
    free(lab); free(par); free(cid); free(st);
    Sbg = 0; I1 = I2 = 0; m1 = m2 = roi = edge = 0; lab = par = cid = 0; st = 0;
    ready = calibrated = 0;
}

AREA_API void area_free(void) { free_all(); }

AREA_API int area_init(const area_params *p, int w, int h) {
    if (!p || w <= 0 || h <= 0 || p->ref_cm2 <= 0) return AREA_E_ARG;
    free_all();
    P = *p; W = w; H = h;
    size_t n = (size_t)w * h, nl = n / 2 + 2;
    Sbg = malloc(n * sizeof *Sbg);
    I1 = malloc((size_t)(w + 1) * (h + 1) * sizeof *I1);
    I2 = malloc((size_t)(w + 1) * (h + 1) * sizeof *I2);
    m1 = malloc(n); m2 = malloc(n); roi = malloc(n); edge = malloc(n);
    lab = malloc(n * sizeof *lab); par = malloc(nl * sizeof *par); cid = malloc(nl * sizeof *cid);
    st = malloc(nl * sizeof *st);
    if (!Sbg || !I1 || !I2 || !m1 || !m2 || !roi || !edge || !lab || !par || !cid || !st) {
        free_all(); return AREA_E_NOMEM;
    }
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        float dx = x + 0.5f - P.cx, dy = y + 0.5f - P.cy, d2 = dx * dx + dy * dy;
        roi[y * w + x] = d2 <= P.r * P.r;
        edge[y * w + x] = d2 <= P.r * P.r && d2 > (P.r - 1.5f) * (P.r - 1.5f);  /* ROI 테두리 띠 */
    }
    ready = 1;
    return 0;
}

static inline void rgb(const uint8_t *img, int stride, int fmt, int x, int y, int *R, int *G, int *B) {
    const uint8_t *q = img + (size_t)y * stride + 3 * x;
    if (fmt == PIX_BGR24) { *B = q[0]; *G = q[1]; *R = q[2]; } else { *R = q[0]; *G = q[1]; *B = q[2]; }
}

/* (2.1) 빨강 판정.
 * 보정 전: 거친 비율 판정 R >= k·max(G,B) 로 기준 도형을 찾는다.
 * 보정 후: 붉은 정도 ρ = R − (G+B)/2 를 조명 지도 L = S_bg / S̄_bg 로 나눈 값이
 *          ρ_t = (ρ_plate + ρ_ref)/2 이상이면 빨강. ρ는 RGB에 선형이라, 경계 픽셀에서
 *          빨강 비율 f가 0.5인 곳에 문턱이 놓인다 → 경계가 안팎으로 치우치지 않는다. */
static float rho_t, Sbar;   /* 보정에서 정함 */
static void red_mask(const uint8_t *img, int stride, int fmt, uint8_t *m) {
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int R, G, B, i = y * W + x; rgb(img, stride, fmt, x, y, &R, &G, &B);
        if (!roi[i]) { m[i] = 0; continue; }
        if (!calibrated) m[i] = R >= P.red_min && R >= P.red_k * G && R >= P.red_k * B;
        else m[i] = (R - 0.5f * (G + B)) * Sbar / Sbg[i] >= rho_t;
    }
}

/* 8-연결 레이블링 (union-find 2패스). lab[i] = 1..n, st[1..n] = 모멘트 */
static int findr(int a) { while (par[a] != a) { par[a] = par[par[a]]; a = par[a]; } return a; }
static void unite(int a, int b) { a = findr(a); b = findr(b); if (a != b) { if (a < b) par[b] = a; else par[a] = b; } }

static int label(const uint8_t *m) {
    int next = 1;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int i = y * W + x; lab[i] = 0;
        if (!m[i]) continue;
        int l = 0;
        int nb[4] = { x > 0 && m[i - 1] ? lab[i - 1] : 0,
                      y > 0 && x > 0 && m[i - W - 1] ? lab[i - W - 1] : 0,
                      y > 0 && m[i - W] ? lab[i - W] : 0,
                      y > 0 && x < W - 1 && m[i - W + 1] ? lab[i - W + 1] : 0 };
        for (int k = 0; k < 4; k++) if (nb[k]) { if (!l) l = nb[k]; else unite(l, nb[k]); }
        if (!l) { par[next] = next; l = next++; }
        lab[i] = l;
    }
    int n = 0;
    for (int a = 1; a < next; a++) cid[a] = 0;
    for (int a = 1; a < next; a++) { int r = findr(a); if (!cid[r]) cid[r] = ++n; cid[a] = cid[r]; }
    memset(st, 0, (size_t)(n + 1) * sizeof *st);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int i = y * W + x; if (!lab[i]) continue;
        int c = lab[i] = cid[lab[i]];
        double fx = x + 0.5, fy = y + 0.5;
        st[c].n += 1; st[c].sx += fx; st[c].sy += fy;
        st[c].sxx += fx * fx; st[c].syy += fy * fy; st[c].sxy += fx * fy;
        if (edge[i]) st[c].edge = 1;
    }
    return n;
}

/* (2.3) 모멘트 타원 충실도 F = N / (4π √det Σ) */
static double fill_ratio(const stat_t *s) {
    double n = s->n, mx = s->sx / n, my = s->sy / n;
    double vxx = s->sxx / n - mx * mx + 1.0 / 12, vyy = s->syy / n - my * my + 1.0 / 12;
    double vxy = s->sxy / n - mx * my, det = vxx * vyy - vxy * vxy;
    return det > 0 ? n / (4 * AREA_PI * sqrt(det)) : 0;
}

/* 기준 도형 후보 선택: 크기가 N_cal 근처이고 F가 1에 가장 가까운 빨간 성분 */
static int pick_ref(int n, int *best_out, double *F_out) {
    int best = 0; double bestcost = 1e9, bestF = 0;
    for (int c = 1; c <= n; c++) {
        double N = st[c].n;
        if (calibrated ? (N < 0.4 * Ncal || N > 2.5 * Ncal) : N < 50) continue;
        double F = fill_ratio(&st[c]);
        double cost = fabs(F - 1) + (calibrated ? fabs(N / Ncal - 1) : -N * 1e-9);
        if (cost < bestcost) { bestcost = cost; best = c; bestF = F; }
    }
    *best_out = best; *F_out = bestF;
    return best != 0;
}

/* 빈 접시로 보정: N_cal, c̄, S_bg 저장 */
AREA_API int area_calibrate(const uint8_t *img, int w, int h, int stride, int fmt) {
    if (!ready) return AREA_E_STATE;
    if (!img || w != W || h != H || stride < 3 * w) return AREA_E_ARG;
    calibrated = 0;
    red_mask(img, stride, fmt, m1);
    int n = label(m1), ref; double F;
    if (!pick_ref(n, &ref, &F)) return AREA_REF_MISSING;
    Ncal = (int)st[ref].n;
    /* 접시 픽셀 = ROI 안, 빨강 아님. 색도 평균과 밝기 지도 */
    double sr = 0, sg = 0, cnt = 0;
    memset(I1, 0, (size_t)(W + 1) * (H + 1) * sizeof *I1);
    memset(I2, 0, (size_t)(W + 1) * (H + 1) * sizeof *I2);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int i = y * W + x, R, G, B; rgb(img, stride, fmt, x, y, &R, &G, &B);
        double S = R + G + B, wgt = (roi[i] && !m1[i] && S > 0) ? 1 : 0;
        if (wgt) { sr += R / S; sg += G / S; cnt += 1; }
        size_t j = (size_t)(y + 1) * (W + 1) + (x + 1);
        I1[j] = I1[j - 1] + I1[j - (W + 1)] - I1[j - (W + 1) - 1] + wgt * S;
        I2[j] = I2[j - 1] + I2[j - (W + 1)] - I2[j - (W + 1) - 1] + wgt;
    }
    if (cnt < 1000) return AREA_E_ARG;
    cbar_r = (float)(sr / cnt); cbar_g = (float)(sg / cnt);
    /* 가중 상자 평균: 기준 도형 자리는 주변 접시로 채워진다.
     * 창 반지름이 도형 등가 반지름보다 작으면 도형 중심에 접시 픽셀이 하나도 없어 지도가 비므로,
     * rad = max(15, √(N/π) + 8). */
    int rad = (int)(sqrt(Ncal / AREA_PI) + 8); if (rad < 15) rad = 15;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int x0 = x - rad < 0 ? 0 : x - rad, x1 = x + rad + 1 > W ? W : x + rad + 1;
        int y0 = y - rad < 0 ? 0 : y - rad, y1 = y + rad + 1 > H ? H : y + rad + 1;
#define BOX(I) (I[(size_t)y1*(W+1)+x1] - I[(size_t)y0*(W+1)+x1] - I[(size_t)y1*(W+1)+x0] + I[(size_t)y0*(W+1)+x0])
        double sw = BOX(I2);
        Sbg[y * W + x] = sw > 0 ? (float)(BOX(I1) / sw) : 1.0f;
#undef BOX
    }
    /* 붉은 정도 문턱: 접시 평균과 기준 도형 안쪽(경계 2 px 떨어진 곳) 평균의 중간 */
    double sS = 0, nS = 0, sp = 0, np_ = 0, sq = 0, nq = 0;
    for (int i = 0; i < W * H; i++) if (roi[i] && !m1[i]) { sS += Sbg[i]; nS += 1; }
    Sbar = (float)(sS / nS);
    const int g = 4;   /* 경계에서 g px 이상 떨어진 픽셀만 평균 (흐림 영향 제외) */
    for (int y = g; y < H - g; y++) for (int x = g; x < W - g; x++) {
        int i = y * W + x, R, G, B; if (!roi[i]) continue;
        int core = 1, far = 1;
        for (int d = -g; d <= g; d++) {
            core &= lab[i + d] == ref && lab[i + d * W] == ref;
            far  &= !m1[i + d] && !m1[i + d * W];
        }
        if (!core && !far) continue;
        rgb(img, stride, fmt, x, y, &R, &G, &B);
        double rho = (R - 0.5 * (G + B)) * Sbar / Sbg[i];
        if (core) { sq += rho; nq += 1; } else { sp += rho; np_ += 1; }
    }
    if (nq < 20 || np_ < 1000) return AREA_REF_MISSING;
    rho_t = (float)(0.5 * (sp / np_ + sq / nq));
    calibrated = 1;
    /* 문턱이 바뀌었으므로 N_cal을 새 판정으로 다시 센다 */
    red_mask(img, stride, fmt, m1);
    n = label(m1);
    if (!pick_ref(n, &ref, &F)) { calibrated = 0; return AREA_REF_MISSING; }
    Ncal = (int)st[ref].n;
    return AREA_OK;
}

static void morph(uint8_t *src, uint8_t *tmp, int dilate) {   /* 3×3 침식(0)/팽창(1), ROI 안에서만 */
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int i = y * W + x, v = dilate ? 0 : 1;
        if (!roi[i]) { tmp[i] = 0; continue; }
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
            int xx = x + dx, yy = y + dy;
            int s = (xx >= 0 && xx < W && yy >= 0 && yy < H) ? src[yy * W + xx] : 0;
            if (dilate) v |= s; else v &= s;
        }
        tmp[i] = (uint8_t)v;
    }
    memcpy(src, tmp, (size_t)W * H);
}

static int cmp_desc(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b; return (x < y) - (x > y);
}

AREA_API int area_measure(const uint8_t *img, int w, int h, int stride, int fmt, area_result *out) {
    if (!calibrated) return AREA_E_STATE;
    if (!img || !out || w != W || h != H || stride < 3 * w) return AREA_E_ARG;
    memset(out, 0, sizeof *out);

    /* 1. 기준 도형 → 스케일 */
    red_mask(img, stride, fmt, m1);
    int n = label(m1), ref; double F;
    if (!pick_ref(n, &ref, &F)) { out->status = AREA_REF_MISSING; return 0; }
    out->ref_px = (int)st[ref].n; out->ref_fill = (float)F;
    out->scale = P.ref_cm2 / (float)st[ref].n;
    /* (2.3) 판정 순서: 모양이 깨졌으면 붙은 빨간 음식, 개수만 다르면 가림 또는 카메라 이동 */
    if (fabs(F - 1) > P.fill_tol) { out->status = AREA_REF_MERGED; return 0; }
    if (fabs(st[ref].n / (double)Ncal - 1) > P.count_tol) { out->status = AREA_REF_CHANGED; return 0; }

    /* 2. 기준 도형 제외 영역: 성분을 ref_dilate 만큼 넓힌다 */
    memset(m2, 0, (size_t)W * H);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if (lab[y * W + x] == ref) {
        for (int dy = -P.ref_dilate; dy <= P.ref_dilate; dy++)
            for (int dx = -P.ref_dilate; dx <= P.ref_dilate; dx++) {
                int xx = x + dx, yy = y + dy;
                if (xx >= 0 && xx < W && yy >= 0 && yy < H) m2[yy * W + xx] = 1;
            }
    }

    /* 3. 음식 마스크 (3.1): 접시 색도와 다르거나, 밝기 비가 범위 밖이면 음식 */
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int i = y * W + x, R, G, B;
        if (!roi[i] || m2[i]) { m1[i] = 0; continue; }
        rgb(img, stride, fmt, x, y, &R, &G, &B);
        float S = (float)(R + G + B);
        if (S < 1) { m1[i] = 1; continue; }
        float dr = R / S - cbar_r, dg = G / S - cbar_g, beta = S / Sbg[i];
        int bg = (dr * dr + dg * dg < P.chroma_tol * P.chroma_tol) && beta >= P.shade_lo && beta <= P.shade_hi;
        m1[i] = !bg;
    }
    morph(m1, m2, 0); morph(m1, m2, 1);   /* 열림: 잡음 점 제거 */
    morph(m1, m2, 1); morph(m1, m2, 0);   /* 닫힘: 작은 구멍 메움 */

    /* 4. 조각별 면적 */
    n = label(m1);
    float parts[AREA_MAX_PARTS]; int np = 0;
    float tmp[4096]; int nt = 0;
    for (int c = 1; c <= n && nt < 4096; c++) {
        if (st[c].n < P.min_part_px) continue;
        tmp[nt++] = (float)(st[c].n * out->scale);
        out->food_cm2 += (float)(st[c].n * out->scale);
        out->touches_edge += st[c].edge;
    }
    qsort(tmp, (size_t)nt, sizeof *tmp, cmp_desc);
    for (int k = 0; k < nt && np < AREA_MAX_PARTS; k++) parts[np++] = tmp[k];
    out->n_parts = nt;
    memcpy(out->part_cm2, parts, (size_t)np * sizeof *parts);
    /* 경계 길이: 남긴 조각의 픽셀과 다른 레이블 이웃 사이의 변 수 × π/4
     * (격자 변 길이는 무작위 방향 경계에서 실제 길이의 평균 4/π 배) */
    long cracks = 0;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int c = lab[y * W + x]; if (!c || st[c].n < P.min_part_px) continue;
        cracks += (x == 0 || lab[y * W + x - 1] != c) + (x == W - 1 || lab[y * W + x + 1] != c)
                + (y == 0 || lab[(y - 1) * W + x] != c) + (y == H - 1 || lab[(y + 1) * W + x] != c);
    }
    out->perim_px = (float)(cracks * AREA_PI / 4);
    out->status = nt == 0 ? AREA_EMPTY : AREA_OK;
    return 0;
}

/* 디버그: 마지막 음식 레이블을 8비트로 내보낸다 (0 = 배경) */
AREA_API int area_debug_labels(uint8_t *outmask) {
    if (!ready || !outmask) return AREA_E_ARG;
    for (int i = 0; i < W * H; i++) outmask[i] = (uint8_t)(lab[i] > 255 ? 255 : lab[i]);
    return 0;
}
