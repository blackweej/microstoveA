#ifndef FOODCV_H
#define FOODCV_H
#include <stdint.h>
#ifdef _WIN32
#  define FCV_API __declspec(dllexport)
#else
#  define FCV_API
#endif

#define FCV_API_VERSION 3

/* ---- 공통 ---- */
enum { FCV_OK = 0, FCV_E_FILE = -1, FCV_E_FORMAT = -2, FCV_E_VERSION = -3,
       FCV_E_NOMEM = -4, FCV_E_ARG = -5, FCV_E_STATE = -6 };
enum { FCV_PIX_BGR24 = 0, FCV_PIX_RGB24 = 1, FCV_PIX_GRAY8 = 2 };
FCV_API int         fcv_api_version(void);          /* FCV_API_VERSION 반환 */
FCV_API const char *fcv_strerror(int code);         /* 정적 문자열 */

/* ---- 음식 분류 (§5–7) ---- */
typedef struct { int class_id; float prob; int rejected; } fcv_result;

FCV_API int  fcv_init(const char *weight_path);     /* NULL = 전처리 전용 상태 (§5.4) */
FCV_API int  fcv_num_classes(void);
FCV_API const char *fcv_class_name(int k);          /* 범위 밖이면 NULL */
FCV_API uint32_t    fcv_model_hash(void);           /* 실행 기록용 */
FCV_API int  fcv_preprocess(const uint8_t *img, int w, int h, int stride, int fmt,
                            const int roi[4], float *out);   /* out: 3×96×96 CHW */
FCV_API int  fcv_classify(const uint8_t *img, int w, int h, int stride, int fmt,
                          const int roi[4],                   /* x,y,w,h. NULL = 전체 (정사각형 영상만, §5.1) */
                          float *probs, int n_probs,          /* NULL 허용 */
                          fcv_result *out);
FCV_API void fcv_free(void);

/* ---- 면적 측정 (§4) ---- */
enum { AREA_OK = 0, AREA_REF_MISSING = 1, AREA_REF_CHANGED = 2,
       AREA_REF_MERGED = 3, AREA_EMPTY = 4, AREA_LIGHT_CHANGED = 5,
       AREA_PLATE_MOVED = 6, AREA_NOT_EMPTY = 7 };
#define AREA_MAX_PARTS 16

typedef struct {
    float ref_cm2;              /* 기준 원 실제 면적 A_ref (cm²) — 자로 잰 지름에서 계산 */
    float cx, cy, r;            /* 접시 ROI 원 (px) */
    float red_k; int red_min;   /* 빨강 판정. red_k는 보정 후에도 R ≥ k·G로 쓴다 (§4.3-1) */
    float chroma_tol;           /* τ_c */
    float shade_lo, shade_hi;   /* β 허용 범위 */
    int   min_part_px;          /* 조각 최소 픽셀 수 */
    float fill_tol;             /* |F − 1| 허용치 → AREA_REF_MERGED */
    float count_tol;            /* |N_ref/N_cal − 1| 허용치 → AREA_REF_CHANGED */
    int   ref_dilate;           /* 기준 원 주변 제외 폭 (px) */
    float light_lo, light_hi;   /* 접시 β 중앙값 허용 범위 → AREA_LIGHT_CHANGED */
    float edge_tol;             /* ROI 테두리 띠의 비접시 비율 상한 → AREA_PLATE_MOVED */
    float empty_tol;            /* 보정 영상의 비접시 비율 상한 → AREA_NOT_EMPTY */
} area_params;

typedef struct {
    int   status;               /* AREA_* (함수 반환값과 별개) */
    float food_cm2;             /* A = s · Σ N_i */
    float scale;                /* s = A_ref / N_ref  (cm²/px) */
    int   ref_px;               /* N_ref */
    float ref_fill;             /* F */
    int   n_parts;              /* 조각 수 (AREA_MAX_PARTS 초과 가능) */
    float part_cm2[AREA_MAX_PARTS];   /* 조각별 면적, 내림차순, 앞 16개 */
    float part_fill[AREA_MAX_PARTS];  /* 조각별 F, part_cm2와 같은 순서 (§10 조각 합침) */
    float perim_px;             /* 음식 경계 길이 P (px) */
    int   touches_edge;         /* 접시 ROI 테두리에 닿은 조각 수 */
} area_result;

FCV_API int  area_init(const area_params *p, int w, int h);
FCV_API int  area_calibrate(const uint8_t *img, int w, int h, int stride, int fmt); /* 빈 접시 1장 */
FCV_API int  area_measure(const uint8_t *img, int w, int h, int stride, int fmt,
                          area_result *out);
FCV_API int  area_debug_labels(uint8_t *mask);      /* w×h, 0 = 배경, 1..255 = 조각 */
FCV_API void area_free(void);

/* ---- 표시창 판독 (§8) ---- */
enum { SEG_STABLE = 0, SEG_UNSTABLE = 1, SEG_INVALID = 2 };
typedef struct { int seconds; int digits[4]; int blank; float margin; } seg_result;

FCV_API int  seg_init(const char *layout_path);
FCV_API int  seg_set_quad(const float quad[8]);     /* TL,TR,BR,BL (u,v) 사진 좌표 */
FCV_API int  seg_read(const uint8_t *img, int w, int h, int stride, int fmt, seg_result *out);
FCV_API int  seg_vote(const seg_result *r, int n, seg_result *out);   /* SEG_* 반환 */
FCV_API int  seg_grid_index(int seconds, int *index); /* *index = 격자 위치, 없으면 *index = -1 (반환은 0) */
FCV_API void seg_free(void);
#endif
