/* area.c — 접시 위 기준 원으로 음식 면적을 cm² 단위로 측정 (§4)
 *
 * reference/area.c에서 이식했다. 시제품과 다른 점은 §4.9에 있다.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "internal.h"

#define FCV_PI 3.14159265358979323846
#define BETA_BINS 512               /* §4.3-5a 히스토그램: β ∈ [0, 2) */
#define MIN_PLATE_PX 1000           /* §4.2-4, §4.3-5a */

typedef struct { double n, sx, sy, sxx, syy, sxy; int edge; } stat_t;

/* ---------------- 전역 상태: area_init에서만 할당 ---------------- */
static area_params P;
static int W, H, ready = 0, calibrated = 0;
static float *Sbg;                  /* S_bg(x,y) */
static double *I1, *I2;             /* 보정용 적분 영상 */
static float cbar_r, cbar_g;        /* c̄ */
static float rho_t, Sbar;           /* §4.2-5, §4.2-6 */
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

void area_free(void) { free_all(); }

int area_init(const area_params *p, int w, int h) {
    if (!p || w <= 0 || h <= 0 || p->ref_cm2 <= 0) return FCV_E_ARG;
    free_all();
    P = *p; W = w; H = h;
    size_t n = (size_t)w * h, nl = n / 2 + 2;
    Sbg = FCV_MALLOC(n * sizeof *Sbg);
    I1 = FCV_MALLOC((size_t)(w + 1) * (h + 1) * sizeof *I1);
    I2 = FCV_MALLOC((size_t)(w + 1) * (h + 1) * sizeof *I2);
    m1 = FCV_MALLOC(n); m2 = FCV_MALLOC(n); roi = FCV_MALLOC(n); edge = FCV_MALLOC(n);
    lab = FCV_MALLOC(n * sizeof *lab); par = FCV_MALLOC(nl * sizeof *par); cid = FCV_MALLOC(nl * sizeof *cid);
    st = FCV_MALLOC(nl * sizeof *st);
    if (!Sbg || !I1 || !I2 || !m1 || !m2 || !roi || !edge || !lab || !par || !cid || !st) {
        free_all(); return FCV_E_NOMEM;
    }
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        float dx = x + 0.5f - P.cx, dy = y + 0.5f - P.cy, d2 = dx * dx + dy * dy;
        roi[y * w + x] = d2 <= P.r * P.r;
        edge[y * w + x] = d2 <= P.r * P.r && d2 > (P.r - 1.5f) * (P.r - 1.5f);  /* §4.2-2 ROI 테두리 띠 */
    }
    ready = 1;
    return FCV_OK;
}

static inline void rgb(const uint8_t *img, int stride, int fmt, int x, int y, int *R, int *G, int *B) {
    const uint8_t *q = img + (size_t)y * stride + 3 * x;
    if (fmt == FCV_PIX_BGR24) { *B = q[0]; *G = q[1]; *R = q[2]; } else { *R = q[0]; *G = q[1]; *B = q[2]; }
}

/* §10 흰 접시로 바뀌면 이 함수 하나만 일반화한다. §4.2-6 ρ의 조명 보정 전 값 R − (G+B)/2 */
static inline float marker_score(int R, int G, int B) { return R - 0.5f * (G + B); }

static void red_mask(const uint8_t *img, int stride, int fmt, uint8_t *m) {
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int R, G, B, i = y * W + x; rgb(img, stride, fmt, x, y, &R, &G, &B);
        if (!roi[i]) { m[i] = 0; continue; }
        /* §4.2-1  R ≥ red_min ∧ R ≥ k·G ∧ R ≥ k·B */
        if (!calibrated) m[i] = R >= P.red_min && R >= P.red_k * G && R >= P.red_k * B;
        /* §4.2-6  ρ = (R − (G+B)/2) · S̄_bg / S_bg ;  ρ_t = (ρ_q + ρ_p)/2 */
        /* §4.3-1  빨강 = ρ ≥ ρ_t ∧ R ≥ k·G */
        else m[i] = marker_score(R, G, B) * Sbar / Sbg[i] >= rho_t && R >= P.red_k * G;
    }
}

/* §4.2-2 8-연결 레이블링 (union-find 2패스). lab[i] = 1..n, st[1..n] = 모멘트 */
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

/* §4.2-3  F = N / (4π √det Σ),  Σ_xx += 1/12, Σ_yy += 1/12 */
static double fill_ratio(const stat_t *s) {
    double n = s->n, mx = s->sx / n, my = s->sy / n;
    double vxx = s->sxx / n - mx * mx + 1.0 / 12, vyy = s->syy / n - my * my + 1.0 / 12;
    double vxy = s->sxy / n - mx * my, det = vxx * vyy - vxy * vxy;
    return det > 0 ? n / (4 * FCV_PI * sqrt(det)) : 0;
}

/* §4.2-3, §4.3-1 기준 원 후보 선택 */
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

/* ‖c − c̄‖ < τ_c */
static inline int plate_chroma(int R, int G, int B) {
    float S = (float)(R + G + B);
    if (S < 1) return 0;
    float dr = R / S - cbar_r, dg = G / S - cbar_g;
    return dr * dr + dg * dg < P.chroma_tol * P.chroma_tol;
}

/* §4.3-6  ‖c − c̄‖ < τ_c  ∧  β_lo ≤ S/S_bg ≤ β_hi  → 접시 */
static inline int is_plate(int R, int G, int B, int i) {
    float beta = (float)(R + G + B) / Sbg[i];
    return plate_chroma(R, G, B) && beta >= P.shade_lo && beta <= P.shade_hi;
}

/* §4.3-5 성분 ref를 d px 넓힌 영역을 out에 쓴다 */
static void dilate_component(int ref, int d, uint8_t *out) {
    memset(out, 0, (size_t)W * H);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if (lab[y * W + x] == ref) {
        for (int dy = -d; dy <= d; dy++) for (int dx = -d; dx <= d; dx++) {
            int xx = x + dx, yy = y + dy;
            if (xx >= 0 && xx < W && yy >= 0 && yy < H) out[yy * W + xx] = 1;
        }
    }
}

static int fmt_ok(int fmt) { return fmt == FCV_PIX_BGR24 || fmt == FCV_PIX_RGB24; }

/* §4.2 빈 접시로 보정: N_cal, c̄, S_bg, ρ_t 저장 */
int area_calibrate(const uint8_t *img, int w, int h, int stride, int fmt) {
    if (!ready) return FCV_E_STATE;
    if (!img || w != W || h != H || stride < 3 * w || !fmt_ok(fmt)) return FCV_E_ARG;
    calibrated = 0;
    red_mask(img, stride, fmt, m1);
    int n = label(m1), ref; double F;
    if (!pick_ref(n, &ref, &F)) return AREA_REF_MISSING;
    Ncal = (int)st[ref].n;
    int big_red = 0;                        /* §4.2-8 (a) */
    for (int c = 1; c <= n; c++) big_red += st[c].n >= 50;

    /* §4.2-4 접시 픽셀 = ROI 안, 빨강 아님. 색도 평균과 밝기 지도 */
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
    if (cnt < MIN_PLATE_PX) return FCV_E_ARG;
    cbar_r = (float)(sr / cnt); cbar_g = (float)(sg / cnt);

    /* §4.2-5 가중 상자 평균. rad = max(15, √(N_cal/π) + 8) */
    int rad = (int)(sqrt(Ncal / FCV_PI) + 8); if (rad < 15) rad = 15;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int x0 = x - rad < 0 ? 0 : x - rad, x1 = x + rad + 1 > W ? W : x + rad + 1;
        int y0 = y - rad < 0 ? 0 : y - rad, y1 = y + rad + 1 > H ? H : y + rad + 1;
#define BOX(I) (I[(size_t)y1*(W+1)+x1] - I[(size_t)y0*(W+1)+x1] - I[(size_t)y1*(W+1)+x0] + I[(size_t)y0*(W+1)+x0])
        double sw = BOX(I2);
        Sbg[y * W + x] = sw > 0 ? (float)(BOX(I1) / sw) : 1.0f;
#undef BOX
    }

    /* §4.2-6 빨강 문턱: 경계에서 g px 이상 떨어진 원 내부·접시 평균의 중간 */
    double sS = 0, nS = 0, sp = 0, np_ = 0, sq = 0, nq = 0;
    for (int i = 0; i < W * H; i++) if (roi[i] && !m1[i]) { sS += Sbg[i]; nS += 1; }
    Sbar = (float)(sS / nS);
    const int g = 4;
    for (int y = g; y < H - g; y++) for (int x = g; x < W - g; x++) {
        int i = y * W + x, R, G, B; if (!roi[i]) continue;
        int core = 1, far = 1;
        for (int d = -g; d <= g; d++) {
            core &= lab[i + d] == ref && lab[i + d * W] == ref;
            far  &= !m1[i + d] && !m1[i + d * W];
        }
        if (!core && !far) continue;
        rgb(img, stride, fmt, x, y, &R, &G, &B);
        double rho = (double)marker_score(R, G, B) * Sbar / Sbg[i];
        if (core) { sq += rho; nq += 1; } else { sp += rho; np_ += 1; }
    }
    if (nq < 20 || np_ < MIN_PLATE_PX) return AREA_REF_MISSING;
    rho_t = (float)(0.5 * (sp / np_ + sq / nq));

    /* §4.2-7 문턱이 바뀌었으므로 N_cal을 새 판정으로 다시 센다 */
    calibrated = 1;
    red_mask(img, stride, fmt, m1);
    n = label(m1);
    if (!pick_ref(n, &ref, &F)) { calibrated = 0; return AREA_REF_MISSING; }
    Ncal = (int)st[ref].n;

    /* §4.2-8 빈 접시 확인 */
    if (big_red >= 2) { calibrated = 0; return AREA_NOT_EMPTY; }
    dilate_component(ref, 3, m2);
    double outside = 0, total = 0;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int i = y * W + x, R, G, B;
        if (!roi[i] || m2[i]) continue;
        rgb(img, stride, fmt, x, y, &R, &G, &B);
        total += 1; outside += !is_plate(R, G, B, i);
    }
    if (outside > P.empty_tol * total) { calibrated = 0; return AREA_NOT_EMPTY; }
    return AREA_OK;
}

static void morph(uint8_t *src, uint8_t *tmp, int dilate) {   /* §4.3-7 3×3 침식(0)/팽창(1), ROI 안에서만 */
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

/* §4.3-5a 접시 색 픽셀의 β 중앙값이 범위 안인가. m2 = 기준 원 제외 영역 */
static int light_ok(const uint8_t *img, int stride, int fmt) {
    int hist[BETA_BINS] = {0}, cnt = 0;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int i = y * W + x, R, G, B;
        if (!roi[i] || m2[i]) continue;
        rgb(img, stride, fmt, x, y, &R, &G, &B);
        if (!plate_chroma(R, G, B)) continue;
        int b = (int)((float)(R + G + B) / Sbg[i] * (BETA_BINS / 2));
        hist[b >= BETA_BINS ? BETA_BINS - 1 : b]++;
        cnt++;
    }
    if (cnt < MIN_PLATE_PX) return 0;
    int acc = 0, b = 0;
    while (b < BETA_BINS - 1 && (acc += hist[b]) * 2 < cnt) b++;
    float median = (b + 0.5f) / (BETA_BINS / 2);
    return median >= P.light_lo && median <= P.light_hi;
}

/* §4.3-5b ROI 테두리 띠의 비접시 비율이 허용치 안인가 */
static int edge_ok(const uint8_t *img, int stride, int fmt) {
    int off = 0, total = 0;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int i = y * W + x, R, G, B;
        if (!edge[i]) continue;
        rgb(img, stride, fmt, x, y, &R, &G, &B);
        total++; off += !plate_chroma(R, G, B);
    }
    return off <= P.edge_tol * total;
}

int area_measure(const uint8_t *img, int w, int h, int stride, int fmt, area_result *out) {
    if (!calibrated) return FCV_E_STATE;
    if (!img || !out || w != W || h != H || stride < 3 * w || !fmt_ok(fmt)) return FCV_E_ARG;
    memset(out, 0, sizeof *out);

    /* §4.3-1 기준 원 → 스케일 */
    red_mask(img, stride, fmt, m1);
    int n = label(m1), ref; double F;
    if (!pick_ref(n, &ref, &F)) {
        out->status = AREA_REF_MISSING;
        for (int c = 1; c <= n; c++) if (st[c].n > 2.5 * Ncal) out->status = AREA_REF_MERGED;
        return FCV_OK;
    }
    /* §4.3-2 */
    out->ref_px = (int)st[ref].n; out->ref_fill = (float)F;
    out->scale = P.ref_cm2 / (float)st[ref].n;
    /* §4.3-3, §4.3-4 순서를 바꾸지 말 것: F만으로는 가림을 못 잡는다 */
    if (fabs(F - 1) > P.fill_tol) { out->status = AREA_REF_MERGED; return FCV_OK; }
    if (fabs(st[ref].n / (double)Ncal - 1) > P.count_tol) { out->status = AREA_REF_CHANGED; return FCV_OK; }

    /* §4.3-5 기준 원 제외 영역 */
    dilate_component(ref, P.ref_dilate, m2);
    if (!light_ok(img, stride, fmt)) { out->status = AREA_LIGHT_CHANGED; return FCV_OK; }
    if (!edge_ok(img, stride, fmt)) { out->status = AREA_PLATE_MOVED; return FCV_OK; }

    /* §4.3-6 음식 마스크 */
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int i = y * W + x, R, G, B;
        if (!roi[i] || m2[i]) { m1[i] = 0; continue; }
        rgb(img, stride, fmt, x, y, &R, &G, &B);
        m1[i] = !is_plate(R, G, B, i);
    }
    morph(m1, m2, 0); morph(m1, m2, 1);   /* §4.3-7 열림 */
    morph(m1, m2, 1); morph(m1, m2, 0);   /* §4.3-7 닫힘 */

    /* §4.3-8  A_i = s · N_i,  s = A_ref / N_ref. 내림차순 앞 16개를 삽입으로 유지한다 */
    n = label(m1);
    int kept = 0;
    for (int c = 1; c <= n; c++) {
        if (st[c].n < P.min_part_px) continue;
        float a = (float)(st[c].n * out->scale), f = (float)fill_ratio(&st[c]);
        out->food_cm2 += a;
        out->touches_edge += st[c].edge;
        int k = kept < AREA_MAX_PARTS ? kept : AREA_MAX_PARTS - 1;
        if (kept >= AREA_MAX_PARTS && a <= out->part_cm2[k]) { kept++; continue; }
        while (k > 0 && out->part_cm2[k - 1] < a) {
            out->part_cm2[k] = out->part_cm2[k - 1];
            out->part_fill[k] = out->part_fill[k - 1];
            k--;
        }
        out->part_cm2[k] = a; out->part_fill[k] = f;
        kept++;
    }
    out->n_parts = kept;

    /* §4.3-9 경계 길이: 남긴 조각의 픽셀과 다른 레이블 이웃 사이의 변 수 × π/4 */
    long cracks = 0;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int c = lab[y * W + x]; if (!c || st[c].n < P.min_part_px) continue;
        cracks += (x == 0 || lab[y * W + x - 1] != c) + (x == W - 1 || lab[y * W + x + 1] != c)
                + (y == 0 || lab[(y - 1) * W + x] != c) + (y == H - 1 || lab[(y + 1) * W + x] != c);
    }
    out->perim_px = (float)(cracks * FCV_PI / 4);
    out->status = kept == 0 ? AREA_EMPTY : AREA_OK;
    return FCV_OK;
}

/* 디버그: 마지막 레이블을 8비트로 내보낸다 (0 = 배경) */
int area_debug_labels(uint8_t *mask) {
    if (!ready) return FCV_E_STATE;
    if (!mask) return FCV_E_ARG;
    for (int i = 0; i < W * H; i++) mask[i] = (uint8_t)(lab[i] > 255 ? 255 : lab[i]);
    return FCV_OK;
}
