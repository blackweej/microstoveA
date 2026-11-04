/* test_area.c — 면적 모듈의 인자·상태 검사와 잡음 없는 소형 장면 (합성 회귀는 tools/tests) */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "foodcv.h"
#include "minitest.h"

#define TW 320
#define TH 240

static void paint(uint8_t *img, int cx, int cy, int r, int R, int G, int B) {
    for (int y = 0; y < TH; y++) for (int x = 0; x < TW; x++) {
        int dx = x - cx, dy = y - cy;
        if (dx * dx + dy * dy <= r * r) {
            uint8_t *q = img + (y * TW + x) * 3;
            q[0] = (uint8_t)R; q[1] = (uint8_t)G; q[2] = (uint8_t)B;
        }
    }
}

static void scene(uint8_t *img) {
    memset(img, 90, TW * TH * 3);
    paint(img, 160, 120, 115, 60, 105, 165);   /* 파란 접시 */
    paint(img, 235, 120, 20, 200, 32, 36);     /* 기준 원 */
}

static area_params params(void) {
    area_params p;
    memset(&p, 0, sizeof p);
    p.ref_cm2 = 12.566f; p.cx = 160; p.cy = 120; p.r = 109;
    p.red_k = 1.8f; p.red_min = 90; p.chroma_tol = 0.035f; p.shade_lo = 0.55f; p.shade_hi = 1.12f;
    p.min_part_px = 40; p.fill_tol = 0.04f; p.count_tol = 0.02f; p.ref_dilate = 3;
    p.light_lo = 0.80f; p.light_hi = 1.08f; p.edge_tol = 0.10f; p.empty_tol = 0.01f;
    return p;
}

void test_area(void) {
    static uint8_t img[TW * TH * 3], food[TW * TH * 3];
    area_params p = params();
    area_result r;

    area_free();
    CHECK(area_calibrate(img, TW, TH, TW * 3, FCV_PIX_RGB24) == FCV_E_STATE);
    CHECK(area_measure(img, TW, TH, TW * 3, FCV_PIX_RGB24, &r) == FCV_E_STATE);
    CHECK(area_init(NULL, TW, TH) == FCV_E_ARG);
    CHECK(area_init(&p, 0, TH) == FCV_E_ARG);
    CHECK(area_init(&p, TW, TH) == FCV_OK);

    scene(img);
    CHECK(area_calibrate(img, TW, TH, TW * 3, FCV_PIX_GRAY8) == FCV_E_ARG);
    CHECK(area_calibrate(img, TW + 1, TH, TW * 3, FCV_PIX_RGB24) == FCV_E_ARG);
    CHECK(area_calibrate(img, TW, TH, TW * 2, FCV_PIX_RGB24) == FCV_E_ARG);
    CHECK(area_measure(img, TW, TH, TW * 3, FCV_PIX_RGB24, &r) == FCV_E_STATE);
    CHECK(area_calibrate(img, TW, TH, TW * 3, FCV_PIX_RGB24) == AREA_OK);

    CHECK(area_measure(img, TW, TH, TW * 3, FCV_PIX_GRAY8, &r) == FCV_E_ARG);
    CHECK(area_measure(img, TW, TH, TW * 3, FCV_PIX_RGB24, NULL) == FCV_E_ARG);
    CHECK(area_measure(img, TW, TH, TW * 3, FCV_PIX_RGB24, &r) == FCV_OK);
    CHECK(r.status == AREA_EMPTY);
    CHECK(r.ref_px > 1200 && r.ref_px < 1320);

    /* 음식 두 조각: 면적 내림차순, F는 원이라 1 근처 */
    memcpy(food, img, sizeof food);
    paint(food, 120, 120, 12, 232, 226, 210);
    paint(food, 140, 70, 25, 232, 226, 210);
    CHECK(area_measure(food, TW, TH, TW * 3, FCV_PIX_RGB24, &r) == FCV_OK);
    CHECK(r.status == AREA_OK);
    CHECK(r.n_parts == 2);
    CHECK(r.part_cm2[0] > r.part_cm2[1]);
    CHECK(fabsf(r.part_cm2[0] + r.part_cm2[1] - r.food_cm2) < 1e-3f);
    CHECK(fabsf(r.part_fill[0] - 1) < 0.03f && fabsf(r.part_fill[1] - 1) < 0.03f);

    /* BGR로 뒤집어도 결과가 같다 */
    {
        static uint8_t bgr[TW * TH * 3];
        area_result r2;
        for (int i = 0; i < TW * TH; i++) {
            bgr[i * 3] = food[i * 3 + 2]; bgr[i * 3 + 1] = food[i * 3 + 1]; bgr[i * 3 + 2] = food[i * 3];
        }
        CHECK(area_measure(bgr, TW, TH, TW * 3, FCV_PIX_BGR24, &r2) == FCV_OK);
        CHECK(r2.food_cm2 == r.food_cm2);
    }

    /* 조각이 16개를 넘어도 합계는 전부 세고 앞 16개는 내림차순이다 */
    {
        memcpy(food, img, sizeof food);
        int n = 0;
        for (int yy = 45; yy <= 195; yy += 25) for (int xx = 75; xx <= 195; xx += 24) {
            paint(food, xx, yy, 6 + n % 4, 232, 226, 210); n++;
        }
        CHECK(area_measure(food, TW, TH, TW * 3, FCV_PIX_RGB24, &r) == FCV_OK);
        CHECK(r.n_parts > AREA_MAX_PARTS);
        float sum16 = 0;
        for (int k = 0; k < AREA_MAX_PARTS; k++) {
            sum16 += r.part_cm2[k];
            if (k) CHECK(r.part_cm2[k - 1] >= r.part_cm2[k]);
        }
        CHECK(sum16 < r.food_cm2);
    }

    /* §4.2-8 음식이 있으면 보정을 거부하고 보정 전 상태로 돌아간다 */
    memcpy(food, img, sizeof food);
    paint(food, 120, 120, 30, 232, 226, 210);
    CHECK(area_calibrate(food, TW, TH, TW * 3, FCV_PIX_RGB24) == AREA_NOT_EMPTY);
    CHECK(area_measure(img, TW, TH, TW * 3, FCV_PIX_RGB24, &r) == FCV_E_STATE);

    /* 기준 원이 없으면 보정 실패 */
    memset(food, 90, sizeof food);
    paint(food, 160, 120, 115, 60, 105, 165);
    CHECK(area_calibrate(food, TW, TH, TW * 3, FCV_PIX_RGB24) == AREA_REF_MISSING);

    uint8_t *mask = malloc(TW * TH);
    CHECK(area_debug_labels(NULL) == FCV_E_ARG);
    CHECK(area_debug_labels(mask) == FCV_OK);
    free(mask);
    area_free();
    CHECK(area_debug_labels(img) == FCV_E_STATE);
}
