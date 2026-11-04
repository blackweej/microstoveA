# foodcv — A파트 구현 명세 (Claude Code용)

전자레인지 타임세터 프로젝트의 A파트(음식 분류 · 음식 면적 측정 · 표시창 판독) 구현 명세다.
이 파일만 읽고 구현할 수 있게 쓴다. 근거와 유도는 팀 설계 문서(2026-11-03 최종 설계, 2026-10-23 기술 명세)에 있고,
여기에는 **구현 계약**(무엇을, 어떤 시그니처로, 어떤 수치 기준으로)만 둔다.

- 마감: DLL을 C파트에 넘기는 날 2026-11-13, 최종 2026-12-04
- 언어: 라이브러리 C11(외부 라이브러리 없음), 도구·학습 Python 3.11(numpy, opencv-python, torch, scipy, pytest 허용. scipy는 시뮬레이터용)
- 산출물: `foodcv.dll`(Windows, MinGW-w64 x86_64) / `libfoodcv.so`(Linux, 개발·테스트용), `model.fcv`, `tools/*.py`
- 개정: API 버전 3 (2026-11-04). 바뀐 내용은 `CHANGELOG.md`, 면적 쪽 근거 수치는 §4.9.

---

## 0. 절대 규칙

1. `src/`의 C 코드는 C11 표준 라이브러리(`stdint.h`, `stdlib.h`, `string.h`, `math.h`, `stdio.h`)만 쓴다. OpenCV·BLAS·스레드 라이브러리 금지.
2. 메모리 할당은 `*_init`에서만 한다. `fcv_classify`, `area_measure`, `seg_read`, `seg_vote` 안에서 `malloc` 금지.
3. 모든 공개 함수는 `int` 상태 코드를 반환한다(0 = 성공, 음수 = 오류). 결과는 출력 인자로 돌려준다. 예외: `*_free`, 이름 조회 함수.
4. 입력 픽셀 배열은 호출자 소유다. 라이브러리는 포인터를 보관하지 않는다. 한 스레드에서만 호출된다고 가정한다.
5. `M_PI` 금지(C11 표준 아님). `#define FCV_PI 3.14159265358979323846`을 쓴다.
6. `-march=native` 금지. 배포 빌드는 `-O3`.
7. `include/foodcv.h`의 공개 API를 바꾸면 `FCV_API_VERSION`을 올리고 `CHANGELOG.md`에 적는다. C파트가 이 헤더에 맞춰 코드를 쓴다.
8. **모르는 값은 지어내지 않는다.** 10절의 미확정 항목은 설정 파일 값으로 받고, 기본값을 코드에 박지 않는다.
9. 수식을 구현한 곳에는 이 명세의 절 번호와 수식을 주석으로 단다(예: `/* §4.2 s = A_ref / N_ref */`).

---

## 1. 시스템 안에서 A파트의 위치

```
측정대 웹캠 ──사진──▶ fcv_classify ──클래스 k──▶ C파트: 프로파일 조회
                       │                            │ uses_area?
                       └──같은 사진──▶ area_measure ─┘ (참일 때만) ──면적──▶ C파트: 시간 계산
표시창 웹캠 ──프레임──▶ seg_read ×5 ──▶ seg_vote ──초──▶ C파트: 다이얼 루프
```

- 카메라 캡처, IR 온도, 시간 계산, 다이얼 모터 루프, UI는 **C파트/B파트 몫**이다. A는 픽셀을 받아 숫자를 돌려준다.
- 그래서 A는 하드웨어 없이 사진 파일·합성 영상만으로 개발·검증한다.
- **A는 질량을 계산하지 않는다.** 면적(cm²)을 돌려주면 C파트가 그대로 시간식에 넣는다(§9 참고용).

---

## 2. 저장소 구조

```
foodcv/
  include/foodcv.h          공개 헤더 (§3)
  src/
    internal.h              테스트가 부르는 내부 함수 선언 (§2 끝)
    preprocess.c            §5 크롭·면적평균 축소·정규화
    cnn.c                   §6 합성곱 엔진
    weights.c               §7 model.fcv 해석
    area.c                  §4 면적 측정  ← reference/area.c(시제품)에서 이식
    seg.c                   §8 표시창 판독 (호모그래피 포함)
    common.c                상태 코드 문자열, 버전
  tests/
    test_main.c             의존성 없는 미니 테스트 러너 (assert 매크로 + 실패 개수 반환)
    test_area.c test_cnn.c test_seg.c test_weights.c
  tools/
    foodcv.py               ctypes 래퍼 (§11) — C파트도 이것을 import
    sim_area.py             합성 접시 영상 생성기 ← reference/sim.py에서 이식
    sim_seg.py              합성 7-세그먼트 표시창 생성기 (§8.7)
    make_golden.py          NumPy 참조 구현으로 골든 데이터 생성 (§6.6)
    train.py export.py      학습·가중치 내보내기 (§7)
    capture.py              고정 설정 웹캠 촬영 (§12)
    calib_tool.py           설치 보정 GUI: 접시 원, 표시창 네 꼭짓점, CNN ROI (§12)
    eval_area.py            손 라벨 마스크와 IoU 비교 (실험 X6)
  config/
    area.cfg display.layout display.quad roi.cfg   (§4.6, §8.6, §12, §5.1)
  reference/                시제품 원본 (수정하지 말 것)
    area.c sim.py
  Makefile CHANGELOG.md
```

`reference/area.c`는 2026-11-03에 합성 영상으로 검증한 시제품이다(§4.8 결과). `src/area.c`는 이것을 이식하되
공통 오류 코드·헤더에 맞추고, **§4.9에 적은 변경만** 넣는다. 그 밖의 동작(판정 순서, 문턱 계산)은 바꾸지 않는다. 더 바꿔야 하면 `tests/test_area.c` 기준을 먼저 통과시킨다.
시제품의 가림 수치는 `render()` 기본값인 흰 접시 조건이다. 파란 접시에서는 시제품 그대로면 가림을 못 잡는다(§4.9).

### 빌드

```make
CFLAGS = -std=c11 -O3 -Wall -Wextra -Wpedantic -Iinclude
# Linux 개발용
libfoodcv.so: src/*.c ; gcc $(CFLAGS) -fPIC -shared -o $@ $^ -lm
# Windows 배포용 (Python이 64비트이므로 DLL도 64비트)
foodcv.dll:   src/*.c ; x86_64-w64-mingw32-gcc $(CFLAGS) -shared -o $@ $^ -lm
test: ; gcc $(CFLAGS) -DFCV_DEBUG_DUMP -o run_tests tests/*.c src/*.c -lm && ./run_tests
debug: ; gcc -std=c11 -O0 -g -Iinclude -DFCV_DEBUG_DUMP -fsanitize=address,undefined ...
```

경고 0개가 기준이다.

`src/internal.h`는 공개 API로 닿지 않는 함수를 테스트에 내준다: logit 계산(T-C1), 두 순서의 합성곱 단(T-C2), 소프트맥스(T-C3), 호모그래피 풀이(T-S1), 7-세그먼트 표(T-S2), Otsu(T-S4). `FCV_API`를 붙이지 않으므로 DLL에는 나가지 않는다.

---

## 3. 공개 헤더 `include/foodcv.h`

```c
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

FCV_API int  fcv_init(const char *weight_path);
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
```

반환 규칙: `area_calibrate`는 성공 0, 기준 원을 못 찾으면 `AREA_REF_MISSING`, 빈 접시가 아니면 `AREA_NOT_EMPTY`(둘 다 양수), 인자 오류는 음수.
`area_measure`는 계산이 끝나면 0을 반환하고 판정은 `out->status`에 넣는다. `seg_vote`만 `SEG_*`를 반환한다.
`AREA_*` 판정은 BGR24/RGB24만 지원하고 GRAY8이면 `FCV_E_ARG`.

---

## 4. 면적 측정 `src/area.c`

### 4.1 하는 일

접시(광택 없는 **파랑 단색**) 가장자리에 그린 **빨간 꽉 찬 원** 하나(지름 ≥ 4 cm)로 픽셀 하나의 실제 넓이를 구하고,
접시로 보이지 않는 픽셀을 음식으로 세어 cm²로 돌려준다. 캘리브레이션 도형이 따로 있지 않다. 이 원 하나가 스케일 기준이자 가림 검사 대상이다.
"캘리브레이션"은 시작할 때 **빈 접시 사진 한 장**으로 원의 픽셀 수·접시 색·조명 지도를 저장하는 절차다.

### 4.2 보정 `area_calibrate(빈 접시)`

1. **거친 빨강:** ROI 안에서 `R ≥ red_min && R ≥ red_k·G && R ≥ red_k·B`.
2. **8-연결 레이블링**(union-find 2회 훑기). 각 성분의 픽셀 수 N, 좌표 합·제곱합·곱합을 누적한다(픽셀 중심 = 정수 + 0.5).
   ROI 테두리 띠(반지름 r − 1.5 ~ r)에 닿았는지도 기록.
3. **기준 원 선택:** N ≥ 50인 성분 중 |F − 1|이 가장 작은 것. 충실도
   `F = N / (4π √det Σ)`, Σ = 좌표 공분산 + 대각에 1/12.
   없으면 `AREA_REF_MISSING`. 일단 `N_cal = N`.
4. **접시 평균 색도** c̄ = 평균(R/S, G/S), S = R+G+B. ROI 안·빨강 아닌 픽셀. 1,000개 미만이면 `FCV_E_ARG`.
5. **조명 지도** S_bg(x,y) = 빨강 아닌 ROI 픽셀만 넣은 가중 상자 평균(적분 영상).
   상자 반지름 `rad = max(15, √(N_cal/π) + 8)`. **기준 원보다 작으면 원 중심에 접시 픽셀이 없어 지도가 빈다**(시제품 버그). S̄_bg = 접시 픽셀의 S_bg 평균.
6. **빨강 문턱(혼합 50%):** 붉은 정도 ρ = (R − (G+B)/2) · S̄_bg / S_bg(x,y).
   경계에서 4 px 이상 떨어진 원 내부 픽셀 평균 ρ_q, 4 px 이상 떨어진 접시 픽셀 평균 ρ_p로
   `ρ_t = (ρ_q + ρ_p) / 2`. (내부 < 20개 또는 접시 < 1,000개 → `AREA_REF_MISSING`)
   ρ는 RGB의 일차식이라 경계 섞임 비율 50%에서 판정이 바뀌어 원 넓이를 치우침 없이 센다.
7. `calibrated = 1`로 두고 **새 문턱으로 다시** 빨강 판정·레이블링·선택을 해서 N_cal을 갱신한다.
8. **빈 접시 확인:** 다음 중 하나면 `calibrated = 0`으로 되돌리고 `AREA_NOT_EMPTY`.
   (a) 1번 거친 빨강에서 N ≥ 50인 성분이 2개 이상(양념치킨은 빨강이라 (b)에 안 걸린다).
   (b) 기준 원을 7×7로 넓힌 영역 밖 ROI 픽셀 중 §4.3-6 접시 판정에 실패한 비율 > `empty_tol`.

### 4.3 측정 `area_measure`

판정은 이 순서로 하고, 앞에서 걸리면 즉시 반환한다.

| 순서 | 계산 | 실패 시 status |
|---|---|---|
| 1 | 빨강 = `ρ ≥ ρ_t` **그리고** `R ≥ red_k·G` (보정 후 판정). 레이블링. 후보 = N ∈ [0.4, 2.5]·N_cal, 비용 = \|F−1\| + \|N/N_cal−1\| 최소 | 후보 없음: 가장 큰 빨간 성분 N > 2.5·N_cal이면 `AREA_REF_MERGED`, 아니면 `AREA_REF_MISSING` |
| 2 | `ref_px = N`, `ref_fill = F`, `scale = ref_cm2 / N` 을 먼저 채운다 | — |
| 3 | \|F − 1\| > fill_tol | `AREA_REF_MERGED` (빨간 음식이 붙음) |
| 4 | \|N/N_cal − 1\| > count_tol | `AREA_REF_CHANGED` (가림 또는 카메라 이동) |
| 5 | 기준 원 성분을 `ref_dilate` px 넓힌 영역 제외 | — |
| 5a | 제외 영역 밖 ROI에서 `‖c − c̄‖ < τ_c`인 픽셀의 β = S/S_bg 중앙값(고정 크기 히스토그램, malloc 없음) | 그런 픽셀 < 1,000개, 또는 중앙값이 [light_lo, light_hi] 밖 → `AREA_LIGHT_CHANGED` |
| 5b | ROI 테두리 띠에서 `‖c − c̄‖ ≥ τ_c`인 픽셀 비율 | > edge_tol → `AREA_PLATE_MOVED` |
| 6 | 접시 판정: `‖c − c̄‖ < τ_c` **그리고** `shade_lo ≤ S/S_bg ≤ shade_hi`. 아니면 음식. S < 1이면 음식 | — |
| 7 | 3×3 열림(침식→팽창) 후 3×3 닫힘(팽창→침식), ROI 안에서만 | — |
| 8 | 레이블링, N < min_part_px 조각 버림. 조각 면적 `A_i = scale · N_i`, 합 `food_cm2`, 내림차순 정렬 후 앞 16개 | 남은 조각 0 → `AREA_EMPTY` |
| 9 | `perim_px = (조각 픽셀과 다른 레이블 이웃 사이 변 개수) · π/4`, `touches_edge` 합산 | — |
| 10 | — | `AREA_OK` |

**모양 검사(F)만으로는 가림을 못 잡는다.** 원의 한쪽을 직선으로 20% 가려도 F = 0.979라 fill_tol 0.04를 통과한다.
가림은 4번(개수 비교)이 잡는다. 이 순서를 바꾸지 말 것.

**1번의 `R ≥ red_k·G`를 빼지 말 것.** 파란 접시는 ρ_p가 음수라 ρ_t가 약 45까지 내려가고, 치킨(ρ 83)·크러스트(90)·치즈(93)가 ρ만으로는 전부 빨강이 된다. 그러면 원을 덮어도 픽셀 수가 줄지 않아 4번이 가림을 못 잡는다. 파랑–빨강 경계에서 이 조건은 혼합 약 48%에서 바뀌어 ρ의 50%보다 바깥쪽이므로 원 넓이 판정은 그대로 ρ가 정한다.

**5a·5b가 없으면 조용히 틀린다.** 밝기가 shade_hi를 넘으면 접시 전체가 음식이 되고, 접시가 밀리면 접시 밖 배경이 음식이 되는데, 둘 다 기준 원은 멀쩡해서 `AREA_OK`로 나온다(§4.9).

그림자는 색도를 바꾸지 않고 밝기만 낮추므로 6번의 shade_lo(0.55)로 접시에 포함된다.

### 4.4 C파트가 하는 일 (A는 구현하지 않음, 테스트 문서용)

| status | UI 안내 |
|---|---|
| `AREA_OK` | 면적으로 시간 계산. 면적이 프로파일 `area_range_cm2` 밖이면 계산 금지. `touches_edge > 0`이면 "접시 밖으로 나간 음식" 경고만 |
| `AREA_REF_MISSING` | "접시를 안내선에 맞춰 주세요" |
| `AREA_REF_CHANGED` | "빨간 표시 위의 음식을 치워 주세요". 음식이 없는데 나면 카메라 이동 → 재보정 |
| `AREA_REF_MERGED` | 위와 같음 |
| `AREA_LIGHT_CHANGED` | 시간 계산 금지. "조명이 바뀌었습니다" → 재보정 |
| `AREA_PLATE_MOVED` | 시간 계산 금지. "접시를 안내선에 맞춰 주세요" |
| `AREA_NOT_EMPTY` (보정) | "접시를 비우고 다시 보정해 주세요" |
| `AREA_EMPTY` | "음식이 보이지 않습니다" (흰 접시 + 흰 음식 가능성) |

### 4.5 수식–코드 대응 (필수 주석 위치)

```c
/* §4.2-6  ρ = (R − (G+B)/2) · S̄_bg / S_bg ;  ρ_t = (ρ_q + ρ_p)/2 */
/* §4.3-1  빨강 = ρ ≥ ρ_t ∧ R ≥ k·G */
m[i] = marker_score(R, G, B) * Sbar / Sbg[i] >= rho_t && R >= P.red_k * G;
/* §4.2-3  F = N / (4π √det Σ),  Σ_xx += 1/12, Σ_yy += 1/12 */
/* §4.3-6  ‖c − c̄‖ < τ_c  ∧  β_lo ≤ S/S_bg ≤ β_hi  → 접시 */
/* §4.3-8  A_i = s · N_i,  s = A_ref / N_ref */
```

### 4.6 설정 `config/area.cfg` (키=값 텍스트, `#` 주석)

C 라이브러리는 파일을 읽지 않는다. `tools/foodcv.py`가 읽어 `area_params`를 채운다.

```ini
ref_diameter_cm = ?        # 필수. 자로 잰 실제 지름. ref_cm2 = π(d/2)²
plate_cx = ?  plate_cy = ?  plate_r = ?   # calib_tool.py가 기록. r는 접시 반지름 − 6 px
red_k = 1.8
red_min = 90
chroma_tol = 0.035
shade_lo = 0.55
shade_hi = 1.12
min_part_px = 40
fill_tol = 0.04
count_tol = 0.02
ref_dilate = 3
light_lo = 0.80
light_hi = 1.08
edge_tol = 0.10
empty_tol = 0.01
```

`?` 값이 비어 있으면 래퍼가 예외를 낸다(기본값 금지). 뒤의 네 값은 합성 영상으로 정했다(§4.9). 실물 X1·X2 후 다시 본다.

### 4.7 버퍼

w×h 기준: S_bg float, 적분 영상 2개 double (w+1)(h+1), 마스크 uint8 4개(m1, m2, roi, edge), 레이블 int32,
union-find 부모·압축 번호 int32 (w·h/2 + 2), 성분 통계 (w·h/2 + 2). 640×480에서 약 8 MB. 모두 `area_init`에서.

### 4.8 시제품 검증 결과 (회귀 기준)

합성 조건: 640×480, 16 px/cm, 접시 지름 24 cm, 4배 초표본 후 픽셀 평균, 가우스 흐림 0.7 px, 잡음 σ=2,
가장자리 15% 비네팅, 음식 그림자 4 px·0.72배. `reference/sim.py`의 `render()`.

| 항목 | 시제품 값 |
|---|---|
| 기준 원 지름 4 cm, 접시 회전 12회 | 픽셀 수 오차 평균 −0.11%, 최대 0.28% |
| 지름 2 cm | 최대 1.27% (→ 4 cm 이상 권장) |
| 가림 5%·10% | 6/6 `AREA_REF_CHANGED` |
| 가림 3% | 2/6만 검출, 미검출 시 스케일 오차 ≤ 2% |
| 양념치킨이 원에 맞닿음 | 4/4 `AREA_REF_MERGED`, F ≈ 0.88 |
| 같은 음식 10회 반복(파란 접시) | 면적 표준편차 0.18–0.60% |
| 파란 접시 평균 치우침 | 피자 +4.2, 치킨 +5.9, 양념치킨 +6.6, 볶음밥 +4.5, 만두 +11.2 % |
| 흰 접시 + 흰 만두 | 10/10 `AREA_EMPTY` |
| 처리 시간 (서버, -O3, 단일 스레드) | 10–17 ms / 장 |

평균 치우침은 경계가 바깥으로 δ ≈ 1 px 밀린 결과(ΔA ≈ δ·P)이고 클래스마다 일정해서 C파트 시간 계수가 흡수한다.
그래서 **면적을 보정하지 않는다.** 대신 `perim_px`를 돌려 나중에 `A' = s(N − δ̂·P)` 보정을 할 수 있게 둔다.
단 ΔA/A = δ·P/A이므로 "일정"은 개체 크기가 비슷할 때만 맞다. 크기 범위가 넓으면 X3에서 잔차를 확인한다.

### 4.9 시제품 대비 변경과 근거 (2026-11-04)

`reference/area.c`의 판정을 NumPy로 옮겨 `reference/sim.py` 영상으로 잰 값이다(컴파일한 C가 아니다). 옮긴 코드는 §4.8의 치우침을 재현했다. 음식은 기준 원에서 1 cm 이상 떼어 놓았다.

| 변경 | 시제품 | 변경 후 |
|---|---|---|
| §4.3-1 `R ≥ red_k·G` 추가 | 파란 접시 가림 5·10·20%: 18/18 미검출 | 18/18 `AREA_REF_CHANGED`. k = 1.6–1.8 동일, k = 1.5는 5·10% 미검출, k = 2.0은 N_cal이 0.4% 줄어 기존 red_k 1.8을 씀 |
| §4.3-1 큰 성분 → MERGED | 원과 합쳐진 피자(N ≈ 5.1·N_cal)가 `AREA_REF_MISSING` | `AREA_REF_MERGED` |
| §4.3-5a 조명 검사 | 밝기 ×1.15: 피자 55 → 388.7 cm², `AREA_OK` | `AREA_LIGHT_CHANGED`. ×0.8–×1.12 면적은 54.9–55.7 cm²로 정상, 중앙값 ×1.10 = 1.094 |
| §4.3-5b 이동 검사 | 빈 접시 16 px 이동: 9.8 cm², `AREA_OK` | `AREA_PLATE_MOVED`. 띠 비율: 이동 없음 0, 6 px 0.02, 8 px 0.21, 16 px 0.40 |
| §4.2-8 빈 접시 확인 | 음식 위에서도 보정 성공 | `AREA_NOT_EMPTY`. 비접시 비율: 빈 접시 0, 만두 0.09, 피자 0.19, 치킨·볶음밥 0.99. 양념치킨은 0이라 (a)로 잡음 |

변경 후 파란 접시 값: T-A1 최대 0.16%, 양념치킨 맞닿음 4/4 MERGED, 5종 10/10 `AREA_OK`, 치우침 피자 +4.2, 치킨 +6.0, 양념치킨 +6.7, 볶음밥 +4.5, 만두 +11.2 %, 정상 음식 5a 중앙값 0.998–0.999, 5b 비율 0.

합성 음식은 테두리에 닿지 않는다. 음식이 접시 가장자리를 넓게 덮으면 5b가 `AREA_PLATE_MOVED`로 오탐할 수 있다(틀린 시간이 나가지는 않는 쪽의 실패).

---

## 5. 분류 전처리 `src/preprocess.c`

### 5.1 ROI

- 정사각형, 한 변은 96의 정수배(예: 480 = 96×5). 아니면 `FCV_E_ARG`. `roi == NULL`이면 영상 전체를 같은 조건으로 검사하므로, 640×480처럼 정사각형이 아닌 영상은 `FCV_E_ARG`다. 운용에서는 항상 roi.cfg 값을 넘긴다.
- ROI가 영상 밖으로 나가면 `FCV_E_ARG`. `config/roi.cfg`(`x y size`)에 calib_tool.py가 기록한다.
- 면적 마스크로 잘라 쓰지 않는다(분류 입력은 항상 고정 ROI).

### 5.2 면적 평균 축소 (쌍선형 금지)

s = size/96 (정수). `x̂[c][i][j] = (1/s²) Σ_{p<s} Σ_{q<s} I[c][y0 + s·i + p][x0 + s·j + q]`.
쌍선형은 s > 2에서 고주파를 거르지 않아, 1 px 흔들림에 출력이 5배 민감했다(10/23 검증).

### 5.3 정규화와 채널 순서

`x = (x̂/255 − μ_c) / σ_c`. μ, σ는 `model.fcv`에서 읽는다. 내부는 항상 **RGB, CHW**.
입력이 BGR24면 축소할 때 읽는 순서만 바꾼다. GRAY8이면 세 채널에 같은 값.

### 5.4 학습–추론 전처리 단일화

학습 스크립트도 ctypes로 `fcv_preprocess`를 불러 텐서를 만든다. Python에 같은 전처리를 따로 구현하지 않는다.
순서: 원본 PNG → 증강(Python, 원본 해상도) → `fcv_preprocess` → 텐서.
단, `fcv_preprocess`는 μ, σ가 필요하므로 학습 전 단계용으로 `fcv_init` 없이 μ=0, σ=1을 쓰는 모드가 필요하다:
`weight_path == NULL`로 `fcv_init`을 부르면 "전처리 전용 상태"(μ=0, σ=1, 분류 불가 → `fcv_classify`는 `FCV_E_STATE`).
train.py가 전처리된 학습셋에서 μ, σ를 계산해 export.py가 파일에 쓴다.

---

## 6. CNN 추론 엔진 `src/cnn.c`

### 6.1 구조 (K = 7)

| 단 | 입력 → 출력 | 파라미터 |
|---|---|---|
| 1 | 3×96×96 → 16×48×48 | 448 |
| 2 | 16×48×48 → 32×24×24 | 4,640 |
| 3 | 32×24×24 → 64×12×12 | 18,496 |
| 4 | 64×12×12 → 64×6×6 | 36,928 |
| GAP + FC | 64×6×6 → 64 → 7 | 455 |
| 합계 | | 60,967 (≈ 238 KB) |

각 단 = 3×3 합성곱(패딩 1, 편향 있음) → ReLU → 2×2 최대 풀링(stride 2). C에는 BN이 없다(내보낼 때 접음, §7.2).

### 6.2 융합: max 먼저, ReLU 나중

ReLU가 단조이므로 `max_k ReLU(a_k) = ReLU(max_k a_k)`. 출력 채널 o 하나의 96×96 평면 P를 계산한 즉시 풀링하고 ReLU 한 번.
전체 해상도 결과를 저장하지 않는다. 두 순서의 차이는 정확히 0이어야 한다(테스트).

### 6.3 핵심 루프 (가장 안쪽 = 연속 열 j, 분기 없음)

```c
for (o = 0; o < Cout; o++) {
  for (k = 0; k < H*W; k++) P[k] = b[o];                                  /* b'_o */
  for (c = 0; c < Cin; c++) for (u = 0; u < 3; u++) for (v = 0; v < 3; v++) {
    const float wv = w[((o*Cin + c)*3 + u)*3 + v];                         /* W'_{o,c,u,v} */
    for (i = 0; i < H; i++) {
      const float *xr = xp + ((size_t)c*(H+2) + i + u)*(W+2) + v;          /* 패딩 입력 x~ */
      for (j = 0; j < W; j++) P[i*W + j] += wv * xr[j];
    }
  }
  /* y[o][i][j] = ReLU(max(P[2i][2j], P[2i][2j+1], P[2i+1][2j], P[2i+1][2j+1])) */
}
```

입력은 테두리 0을 미리 붙인 패딩 버퍼(인덱스 i+u, i+u−1 아님).

### 6.4 버퍼 (`fcv_init`에서 1회)

패딩 입력 최대 16×50×50 = 40,000 float, 풀링 출력 최대 16×48×48 = 36,864 float, 평면 96×96 = 9,216 float. 합 86,080 float = 344,320 바이트.
단마다 입력·출력을 번갈아 쓴다(다음 단 입력은 출력을 패딩 버퍼로 옮겨 테두리 0을 붙임).

### 6.5 출력

GAP `g_c = (1/36) Σ y[c]`, `z_k = b_k + Σ_c A[k][c]·g_c`, 소프트맥스는 `m = max z`를 빼서 계산
(z = (1000, 1001, 1002)에서 NaN이 나지 않아야 한다 → (0.090, 0.245, 0.665)).
`rejected = (p_max < τ)`. `probs`가 NULL이 아니면 `min(n_probs, K)`개를 채운다.

### 6.6 수치 일치 기준

`|z_C − z_ref| ≤ 1e-4 · max(1, |z_ref|)` (모든 logit). 반올림만으로는 이 값의 1/100도 나오지 않는다(시제품 실측 5.7e-6).
`tools/make_golden.py`가 float64 NumPy 참조 구현(ReLU 후 풀링 순서, BN 접기 전 모델)으로 무작위 가중치 모델과
입력 20개, 기대 logit을 `tests/golden/`에 쓴다. `FCV_DEBUG_DUMP` 빌드에서는 단마다 출력을 `dump_stageN.bin`(float32 CHW)으로 쓴다.

### 6.7 속도

`-O3` 단일 스레드 7.5 ms(서버 Xeon 2.1 GHz). 기준은 **팀 노트북에서 640×480 입력부터 전처리 포함 최대 200 ms 미만**.
루프 펼치기·멀티스레드 같은 추가 최적화는 하지 않는다.

---

## 7. 가중치 파일 `model.fcv` (`src/weights.c`, `tools/export.py`)

### 7.1 형식 (리틀엔디언, 패딩 없음)

| 순서 | 항목 | 형식 |
|---|---|---|
| 1 | 매직 `FCV1`, 버전 2 | char[4], uint32 |
| 2 | K, 입력 크기 96, 단 수 4 | uint32 ×3 |
| 3 | 채널 (3, 16, 32, 64, 64) | uint32 ×5 |
| 4 | μ_R μ_G μ_B σ_R σ_G σ_B | float32 ×6 |
| 5 | 거부 기준 τ | float32 |
| 6 | 클래스 이름 | char[32] ×K, UTF-8, NUL 채움 |
| 7 | 단별 W′ [Cout][Cin][3][3], b′ [Cout] | float32 |
| 8 | FC A [K][64], b [K] | float32 |
| 9 | 1–8번 항목 전체 바이트의 FNV-1a 32비트 해시 (offset 2166136261, prime 16777619) | uint32 |

`fcv_init`은 매직, 버전(다르면 `FCV_E_VERSION`), 기대 파일 크기 정확히 일치, 해시를 모두 검사하고 하나라도 틀리면 `FCV_E_FORMAT`.
해시는 μ, σ, τ, 클래스 이름까지 덮는다. 이 값들이 훼손되면 분류가 조용히 틀어지기 때문이다.

### 7.2 내보내기 (BN 접기)

```python
model.eval()                                             # 반드시 eval
for conv, bn in stages:
    s  = bn.weight / torch.sqrt(bn.running_var + bn.eps) # s_o = γ/√(σ²+ε)
    W  = conv.weight * s[:, None, None, None]            # W'_o = s_o W_o
    b0 = conv.bias if conv.bias is not None else torch.zeros_like(s)
    b  = (b0 - bn.running_mean) * s + bn.bias            # b'_o = s_o(b_o − μ_o) + β_o
    f.write(W.detach().contiguous().numpy().astype('<f4').tobytes())
    f.write(b.detach().numpy().astype('<f4').tobytes())
```

PyTorch 합성곱 가중치는 이미 [Cout, Cin, kh, kw], `nn.Linear`는 [출력, 입력]이라 그대로 맞는다.

### 7.3 클래스 목록 (순서 고정, C파트 프로파일과 이름 대조)

| k | class_id | 이름 | 면적 사용 |
|---|---|---|---|
| 0 | `samgak_gimbap` | 삼각김밥 | 안 함 |
| 1 | `dosirak` | 도시락 | 안 함 |
| 2 | `hotbar` | 핫바 | 안 함 |
| 3 | `chicken` | 치킨 | 함 (γ = 3/2) |
| 4 | `pizza` | 피자 | 함 (γ = 1) |
| 5 | `dumpling` | 만두 | 함 (γ = 1) |
| 6 | `fried_rice` | 볶음밥 | **미정** (§10) |

냉동/냉장은 CNN이 판단하지 않는다(B파트 IR: −2 °C 미만 냉동, +2 °C 초과 냉장, 사이는 사용자 확인).
C파트는 시작할 때 `fcv_class_name(k)`와 프로파일 이름이 모두 같은지 확인하고 다르면 구동을 막는다. 이름 불일치는 안전 문제다.

### 7.4 학습 (`tools/train.py`)

- 사진: **최종 파란 접시 + 빨간 원 위에서**, 운용과 같은 웹캠·같은 고정 설정(`capture.py`)으로 PNG 저장. 휴대폰·JPEG 금지.
- 개체 단위 수집: 클래스당 개체 ≥ 10, 개체당 20–30장(위치·방향·용기·조명 변화). 학습:검증:평가 = **개체 기준** 6:2:2.
  `data/images/metadata.csv` 열: `file, class_id, item_id, angle, lighting, captured_at`.
- 정량 품목은 기준 원을 가린 사진과 안 가린 사진을 모두 넣는다(빨간 원 가림이 클래스 단서가 되지 않게).
- 증강: 0–360° 회전, 좌우 반전, ±15 px 이동, 밝기·대비 ±15%, 약한 색온도. **크기 변화 증강 금지**(카메라 높이 고정).
  회전은 ROI 모서리에 빈 영역을 만들 수 있으니, 접시 중심 기준으로 돌리고 빈 모서리는 원본 배경으로 채운다.
- 학습: 처음부터, Adam lr 1e-3, 배치 32, 검증 손실 10에폭 무개선 시 정지. 합성곱 뒤 BN.
- τ: 검증셋 p_max 하위 5% 값(정상 음식의 약 5%가 거부되는 값이다). 미등록 음식(다른 음식·빈 접시·손) ≥ 50장으로 거부율을 따로 보고. 합격선은 §10.
- 개체 10개면 검증 개체가 클래스당 2개뿐이라 τ와 조기 정지가 그 2개에 좌우된다. `train_report.json`에 검증 개체 수를 적는다.
- 출력: `model.fcv`, `train_report.json`(혼동 행렬, 클래스별 재현율, τ, 미등록 거부율, 모델 해시).

---

## 8. 표시창 판독 `src/seg.c`

### 8.1 원근 보정

정규 좌표(`canon_w × canon_h`, 예: 200×60) → 사진 좌표 호모그래피 H.
`u = (h11 x + h12 y + h13) / (h31 x + h32 y + 1)`, `v` 동일 꼴. 분모를 곱하면 미지수 8개의 일차식,
네 꼭짓점(정규 (0,0),(W,0),(W,H),(0,H) ↔ quad TL,TR,BR,BL)으로 8×8 연립방정식 → 부분 피벗 가우스 소거.
피벗 절댓값 < 1e-12면 `FCV_E_ARG`. 역행렬은 쓰지 않는다(정규 점을 사진으로 보내 쌍선형 보간).

### 8.2 획 밝기

layout의 자리별 7획(a 위, b 오른쪽 위, c 오른쪽 아래, d 아래, e 왼쪽 아래, f 왼쪽 위, g 가운데) 사각형마다
정규 좌표 1단위 간격 격자점을 H로 사진에 보내 쌍선형 보간한 밝기 평균. 밝기 채널은 layout `channel`(R 또는 MAX = max(R,G,B)).
`polarity = dark`(LCD)면 255 − 값.

### 8.3 이진화 (Otsu, 프레임마다)

7·digits개 값을 정렬하고 모든 경계를 시도해 `w0·w1·(μ1 − μ0)²` 최대인 문턱을 고른다.
대비 μ1 − μ0 < `cmin`이면 `blank = 1`. `margin = (경계에 가장 가까운 값과 문턱의 거리) / 대비`.

### 8.4 디코딩

자리별 7비트를 0–9 표와 정확히 일치시킨다. 모두 꺼진 자리는 **앞쪽 `blank_lead`개 자리에서만** 빈 자리(값 0)로 허용.
표에 없으면 `digits[i] = -1`이고 `seconds = -1`이다. 모든 자리가 유효할 때만 4자리 MM:SS면 `seconds = (d0·10 + d1)·60 + d2·10 + d3`.
`digits` 배열 의미와 자릿수는 layout에서 정한다(3자리 M:SS도 지원).

비트 순서 a=bit0 … g=bit6. 표(bit6..0 = gfedcba): 0=0x3F 1=0x06 2=0x5B 3=0x4F 4=0x66 5=0x6D 6=0x7D 7=0x07 8=0x7F 9=0x6F.
**주의:** 6·7·9는 기종마다 a·f 획 유무가 다르다(6=0x7C, 7=0x27, 9=0x67 변형). layout `glyph_variant` 줄로 덮어쓸 수 있게 한다.

### 8.5 판정 `seg_vote` (기본 n = 5)

1. `blank`인 결과와 `seconds < 0`인 결과를 뺀다.
2. 남은 것 중 `seconds`가 같은 결과가 **3개 이상**인 값을 찾는다(동률이면 margin 합이 큰 쪽).
3. 그 값이 (a) 초의 십의 자리 0–5, (b) 격자 위(`seg_grid_index`의 `*index ≥ 0`)이면 `SEG_STABLE`, `out`에 대표 결과.
4. 3개 이상인 값이 없으면 `SEG_UNSTABLE`, 있는데 검사 실패면 `SEG_INVALID`.

"한 클릭에 격자 한 칸" 연속성 검사는 이전 값을 아는 C파트 루프가 `seg_grid_index` 차이로 한다.
7-세그먼트 부호는 최소 해밍 거리 1이라(거리 1 쌍: 0-8, 1-7, 3-9, 5-6, 5-9, 6-8, 8-9) 획 하나 오류를 부호로는 못 막는다. 위 제약이 그 방어선이다.
**분 자리는 이 제약으로 보호되지 않는다**(1:30 → 7:30은 둘 다 격자 위일 수 있다). 분 자리 오독은 C파트의 연속성 검사가 유일한 방어선이다.

다중화 표시창은 자리가 통째로 꺼진 프레임이 나온다. 자리별 꺼짐 확률 0.25에서 다수결만 쓰면 정답 STABLE이 35.6%뿐이다(앞자리 0 기준). 이때는 래퍼가 프레임을 합성한 뒤 판독한다(§11 `fuse`). 10:00 이상이 표시되면 앞자리 누락이 유효한 값으로 읽힐 수 있어(같은 조건에서 오답 STABLE 1.0%, 정답 18.5%) 합성이 필수다.

### 8.6 레이아웃 `config/display.layout` (텍스트, `#` 주석, 줄 단위)

```
digits 4
canon 200 60
channel R            # R | MAX
polarity bright      # bright(LED) | dark(LCD)
cmin 20
blank_lead 1
format MMSS          # MMSS | MSS
fuse 1               # 판독 1회에 합성할 프레임 수. 1 = 합성 안 함. 래퍼가 읽고 seg_init은 무시
seg 0 a 10 2 40 8    # 자리 획 x0 y0 x1 y1 (정규 좌표)
...                  # 자리 × 7줄
glyph_variant 9 0x6F # 선택
grid 0 10 20 30 40 50 60 ...   # 다이얼로 만들 수 있는 초 값, 오름차순, 한 줄 또는 여러 줄
```

`grid`가 비어 있으면 `seg_init`은 `FCV_E_FORMAT`(격자 없이 구동 금지). 이 파일은 실물 관찰 후 calib_tool.py로 만든다.

### 8.7 합성 표시창 `tools/sim_seg.py`

임의 숫자·기울기(±15°)·조명·잡음·**다중화 깜빡임**(자리별 켜짐 비율 d, 주기 T, 노출 e 지정)으로 프레임을 만든다.
획이 사라질 확률의 기대값 `P_dark = max(0, (1 − d) − e/T)`를 시뮬레이션이 재현하는지 테스트에 쓴다.
이 식은 프레임마다 위상이 무작위라고 가정한다. 프레임 주기가 표시창 주기의 정수배에 가까우면 같은 자리가 계속 꺼져 합성으로도 복구되지 않으므로, 실물에서 `fuse`를 정할 때 확인한다.

---

## 9. C파트 인터페이스 참고 (A는 구현하지 않음)

시간식: `x = (Σ_i A_i^γ) · (T_f − T₀)`, `t = a′_k · x + b_k`, 학습은 순방향 실험마다
`a′ = Cov(x, t)/Var(x)`, `b = t̄ − a′ x̄`. 정량 품목(`uses_area: false`)은 기존 식.
γ는 클래스별 1과 3/2 둘 다 학습해 교차검증 오차가 작은 쪽. 프로파일 키는 (class_id, 상태).

```yaml
- class_id: pizza
  state: frozen
  uses_area: true
  gamma: 1.0
  a: null          # s/(cm²·K)
  b: null          # s
  t_max: 360
  area_range_cm2: [40, 300]
```

A가 줄 것: `food_cm2`, `part_cm2[]`, `part_fill[]`, `n_parts`, `status`, `fcv_class_name`. `n_parts > 16`이면 C파트는 γ ≠ 1에서 Σ A_i^γ를 계산할 수 없으므로
래퍼(`foodcv.py`)가 경고를 낸다.

---

## 10. 미확정 항목 (코드에 박지 말 것 — 설정으로 받기)

| 항목 | 받는 곳 | 결정 주체 |
|---|---|---|
| 기준 원 실제 지름 | area.cfg `ref_diameter_cm` | B (자로 측정) |
| 접시 색 | **파랑으로 진행(2026-11-04).** 바뀔 경우: (흰 접시면 기준 원을 파랑으로 바꾸고 문턱 식의 ρ를 B − (R+G)/2로 일반화해야 함 → 그때 API 변경. §4.3-1의 `R ≥ red_k·G`도 흰 접시에서는 경계를 안쪽으로 밀어 N_cal을 1% 줄이므로 다시 정해야 함) | B |
| 표시창 종류·자릿수·형식·빈 자리 규칙 | display.layout | 실물 관찰 |
| 다이얼 격자 | display.layout `grid` | 실물 관찰 |
| 6·7·9 글리프 모양 | display.layout `glyph_variant` | 실물 관찰 |
| 볶음밥 `uses_area` | C파트 프로파일 | A·C, 실험 X3 후 |
| 웹캠 수동 초점·노출·WB 지원 | capture.py가 확인 후 기록 | A |
| 카메라 높이 Z₀ | 문서용 (코드 불필요) | B |
| 조각 합침 처리 (아래) | C파트 프로파일·시간 계산 | A·C |
| `count_tol` 실물 값 | area.cfg. X1 측정 후 정함 (아래) | A |
| 음식 높이에 따른 원근 확대 | X3 잔차로 확인 (아래) | A |
| 미등록 음식 거부율 합격선 | 값 없음. 실측 후 정함 | A·C |
| 표시창 `fuse` 값 | display.layout. 깜빡임 관찰 후 정함 | 실물 관찰 |

- **조각 합침:** 조각이 맞닿거나 2 px 이내면 닫힘 연산으로 한 조각이 되어 γ = 3/2에서 같은 음식의 ΣA^γ가 최대 1.41배가 된다(A 두 개 → 2A 하나). `part_fill`을 돌려주지만 F만으로는 가를 수 없다(겹친 두 타원 0.969, 맞닿은 두 타원 0.89, 반달 만두 한 개 0.959). 치킨 시간 계산에서 이 오차를 어떻게 다룰지는 아직 정해지지 않았다.
- **`count_tol`:** 기본 0.02는 X1 합격선 2%와 같다. 스케일 균일성이 합격선 근처면 접시를 돌리기만 해도 `AREA_REF_CHANGED`가 뜰 수 있다. 시제품의 0.28%는 왜곡 없는 합성 카메라 값이다.
- **원근 확대:** 스케일은 접시 면 기준이고 음식 윗면은 카메라에 더 가까워서 높은 음식일수록 크게 잰다. Z₀가 정해지면 크기를 추정한다.

`ρ` 일반화 대비: `red_mask`의 붉은 정도 계산을 `static inline float marker_score(int R, int G, int B)` 하나로 모아 둔다.

---

## 11. Python 래퍼 `tools/foodcv.py`

C파트가 import해서 쓴다. ctypes `argtypes`/`restype`를 **모든 함수에** 지정한다
(안 하면 64비트 Windows에서 포인터 상위 비트가 잘린다).

```python
class FoodCV:
    def __init__(self, lib_path, weight_path, area_cfg, display_layout, display_quad, roi_cfg): ...
    def classify(self, frame_bgr) -> tuple[int, str, float, bool, np.ndarray]   # k, 이름, p, rejected, probs
    def calibrate_plate(self, empty_frame_bgr) -> int                          # AREA_* 또는 예외. 첫 호출 때 프레임 크기로 area_init
    def measure_area(self, frame_bgr) -> AreaResult                            # dataclass, status 이름 포함
    def read_display(self, frames_bgr: list) -> tuple[int, int]                 # SEG_*, seconds (STABLE이 아니면 -1)
    def grid_index(self, seconds) -> int                                       # 격자 밖이면 -1 (예외 아님)
    def model_hash(self) -> int
```

- 입력은 OpenCV BGR `uint8` H×W×3. `np.ascontiguousarray` 후 `frame.ctypes.data`, `stride = frame.strides[0]`, `FCV_PIX_BGR24`.
- 음수 반환은 `FoodCVError(code, fcv_strerror(code))` 예외.
- 생성자에서 `fcv_api_version() == 3` 확인. `display.quad`를 읽어 `seg_set_quad`를 부른다.
- `area_init`은 영상 크기가 필요하므로 `calibrate_plate`의 첫 프레임 크기로 부른다. 크기가 바뀌면 다시 init하고 재보정을 요구한다. 보정 전 `measure_area`는 `FCV_E_STATE` 예외.
- `read_display`는 `5 × fuse`장을 받는다(아니면 `ValueError`). `fuse`장씩 픽셀별 합성(`polarity bright`는 최댓값, `dark`는 최솟값)해 5장을 만들고 `seg_read` ×5 → `seg_vote`. `fuse`와 `polarity`는 래퍼가 `display.layout`에서 직접 읽는다.

---

## 12. 도구

- **capture.py:** 웹캠을 열어 자동 초점·자동 노출·자동 화이트밸런스를 끄고(`CAP_PROP_AUTOFOCUS=0`, `CAP_PROP_AUTO_EXPOSURE`, `CAP_PROP_AUTO_WB=0`),
  설정이 실제로 먹었는지 두 번 읽어 밝기 변화로 확인해 `camera_check.json`에 기록. 백엔드마다 값이 달라 실패할 수 있으니 실패를 숨기지 말 것.
  버퍼 프레임 버리기(`grab()` n회) 옵션. 저장은 PNG.
- **calib_tool.py:** OpenCV 창 하나. 모드 1: 접시 원 3점 클릭 → 중심·반지름(area.cfg). 모드 2: 표시창 네 꼭짓점 클릭(TL,TR,BR,BL) → `config/display.quad`(공백으로 나눈 float 8개).
  모드 3: CNN ROI 정사각형(96 배수로 스냅) → roi.cfg. 모드 4: 정규화된 표시창 이미지 위에 획 사각형 드래그 → display.layout `seg` 줄.
- **eval_area.py:** 사진과 손 라벨 마스크(PNG, 255 = 음식)를 받아 `area_debug_labels` 결과와 IoU, 면적 비를 표로 출력.

---

## 13. 테스트 (`make test`, 모두 통과해야 완료)

### 13.1 C 단위 테스트

| ID | 대상 | 기준 |
|---|---|---|
| T-W1 | weights | 정상 파일 로드, 매직·버전·크기·해시 각각 1바이트 훼손 시 정확한 오류 코드 |
| T-P1 | preprocess | 상수 영상 → 모든 출력이 (v/255 − μ)/σ, BGR/RGB 입력 결과 동일 |
| T-P2 | preprocess | 96 배수 아닌 ROI, 영상 밖 ROI → `FCV_E_ARG` |
| T-C1 | cnn | golden 20개 logit §6.6 기준 |
| T-C2 | cnn | max→ReLU와 ReLU→max 순서 결과 차이 0 |
| T-C3 | cnn | 소프트맥스 (1000,1001,1002) → (0.090, 0.245, 0.665) ±1e-3, NaN 없음 |
| T-C4 | cnn | `fcv_classify` 중 malloc 호출 0 (`FCV_DEBUG_DUMP` 빌드의 할당 카운터) |
| T-S1 | seg | 알려진 H로 만든 네 점에서 H 복원 오차 < 1e-9 |
| T-S2 | seg | 0–9 부호 해밍 거리 1 쌍이 정확히 7개 (위 목록), 초 십의 자리 0–5에서 획 1개 뒤집기 42가지 모두 검출 |
| T-S3 | seg_vote | 5개 중 3 일치 → STABLE, 2 일치 → UNSTABLE, 3 일치지만 격자 밖 → INVALID, blank 3개 제외 후 판정, `seconds = -1` 3개 → UNSTABLE |
| T-S4 | seg | Otsu: 두 무리 값(30±3, 200±3)에서 문턱이 두 무리 사이, 대비 < cmin → blank |

T-C1~C3, T-S1·S2·S4는 `src/internal.h`를 통해 부른다. T-C4의 할당 카운터는 `make test`가 `FCV_DEBUG_DUMP`로 빌드해서 켜진다.

### 13.2 Python 통합 테스트 (`pytest tools/tests`, 합성 영상)

접시를 따로 적지 않은 시험은 **파란 접시**다. `sim_area.py`의 `render()`는 접시 색을 필수 인자로 받고, 음식은 기준 원에서 1 cm 이상 떼어 놓는다(`food_touch_ref`, `occl`, T-A12 제외).

| ID | 기준 (시제품보다 나빠지면 실패) |
|---|---|
| T-A1 | 기준 원 4 cm, 접시 회전 12회: 픽셀 수 오차 최대 ≤ 0.5% |
| T-A2 | 가림 5%, 10% 각 6회: 전부 `AREA_REF_CHANGED` |
| T-A3 | 양념치킨 맞닿음 4회: 전부 `AREA_REF_MERGED` |
| T-A4 | 파란 접시, 5종 × 10회: 전부 `AREA_OK`, 음식별 면적 표준편차 ≤ 1% |
| T-A5 | 파란 접시 평균 치우침이 §4.8 값 ±1.5%p 안 (알고리즘이 바뀌었는지 감시) |
| T-A6 | 흰 접시 + 흰 만두 → `AREA_EMPTY` (알려진 실패가 조용히 OK로 바뀌지 않는지 감시) |
| T-A7 | 빈 접시 측정 → `AREA_EMPTY`, 기준 원 없는 영상 → `AREA_REF_MISSING` |
| T-A9 | 밝기 ×1.15, ×1.2 → `AREA_LIGHT_CHANGED`. ×0.9, ×1.05 → `AREA_OK` |
| T-A10 | 빈 접시·음식 접시를 16 px 이동 → `AREA_PLATE_MOVED` 또는 `AREA_REF_CHANGED`. `AREA_OK`·`AREA_EMPTY`는 실패 |
| T-A11 | 5종 음식이 있는 영상으로 보정 → 전부 `AREA_NOT_EMPTY`. 빈 접시 보정 → 성공 |
| T-A12 | 피자가 기준 원과 합쳐져 N > 2.5·N_cal → `AREA_REF_MERGED` |
| T-A8 | 640×480 한 장 처리 ≤ 50 ms (CI 서버) |
| T-D1 | sim_seg 기울기 ±15°, 잡음 σ=3, 전 격자 값 200회: 틀린 값 STABLE 0회, UNSTABLE < 5% |
| T-D2 | 깜빡임 d=1/4, e=1 ms, T=2 ms: 단일 프레임 획 누락률 ≈ 0.25 (±0.05), `fuse 5`로 `read_display`(25장) 시 판정 성공 |
| T-E2E | 래퍼로 classify → measure_area → read_display 한 번씩, 예외 없이 dataclass 반환 |

---

## 14. 구현 순서 (각 단계 끝에 `make test` 통과 + 커밋)

1. **M0 골격:** 헤더, Makefile, 미니 테스트 러너, `fcv_api_version`, `fcv_strerror`, foodcv.py 뼈대.
2. **M1 면적:** `reference/area.c` → `src/area.c` 이식(오류 코드 통일, §4.9 변경), `sim_area.py` 이식, T-A1~A12. 시제품이 있으므로 가장 먼저 끝낸다.
3. **M2 전처리:** T-P1, T-P2. 전처리 전용 모드(§5.4).
4. **M3 CNN·가중치:** make_golden.py, export.py, T-W1, T-C1~C4.
5. **M4 표시창:** 호모그래피, 획 샘플링, Otsu, 디코딩, vote, layout 파서, sim_seg.py, T-S1~S4, T-D1~D2.
6. **M5 도구:** capture.py, calib_tool.py, eval_area.py, train.py.
7. **M6 래퍼 완성:** T-E2E, Windows DLL 빌드 확인, C파트에 전달할 `README_C_PART.md`(래퍼 사용 예 10줄).

11/13 전달 기준: M0–M4, M6 필수. M5의 train.py는 실제 사진이 모이는 대로.

---

## 15. 실물 실험에서 이 코드가 쓰이는 곳 (참고)

| 실험 | 내용 | 필요한 도구 |
|---|---|---|
| X1 | 1 cm 모눈종이를 접시 다섯 곳에 놓고 스케일 균일성(차이 ≤ 2%). 결과로 `count_tol` 결정(§10) | measure_area + eval_area |
| X2 | 같은 음식 10회·접시 회전 10회, 면적 표준편차 ≤ 2% | capture + measure_area |
| X3 | 면적 사용 클래스마다 개체 ≥ 15개 면적·**질량(저울)** 동시 측정, m–A R² ≥ 0.9 (치킨은 m–ΣA_i^1.5). 크기별 치우침·높이 영향·조각 합침 빈도도 기록(§10) | measure_area, part_cm2, part_fill |
| X6 | 7종 × 3개 손 라벨, `AREA_EMPTY` 0회, IoU ≥ 0.9 | eval_area |
| X7 | 가림 5%·20%, 빨간 음식 붙임, 지그 1 cm 들어올림 각 10회 → 전부 OK 아님 | measure_area |
| E1–E8 | 분류 일치·속도·정확도·거부·표시창 (10/23 명세) | 전체 |

질량은 X3 검증에만 쓴다. 운용 장치에는 저울이 없다.
