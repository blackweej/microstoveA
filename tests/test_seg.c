/* test_seg.c — T-S1~S4와 레이아웃 파서 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../src/internal.h"
#include "minitest.h"

int seg_solve_homography(const double quad[8], double canon_w, double canon_h, double h[8]);
int seg_digit_code(int digit);
int seg_otsu(const float *v, int n, float *threshold, float *contrast, float *margin);

#define TMP "t_layout.tmp"

static void write_segs(FILE *f, int digits) {
    for (int d = 0; d < digits; d++) for (char s = 'a'; s <= 'g'; s++)
        fprintf(f, "seg %d %c %d 2 %d 8\n", d, s, 10 + 40 * d, 40 + 40 * d);
}

static int layout_with(const char *head, const char *tail, int digits) {
    FILE *f = fopen(TMP, "wb");
    if (!f) return 1;
    fputs(head, f);
    write_segs(f, digits);
    fputs(tail, f);
    fclose(f);
    int rc = seg_init(TMP);
    remove(TMP);
    return rc;
}

#define HEAD "digits 4\ncanon 200 60\nchannel R  # 주석\npolarity bright\ncmin 20\nblank_lead 1\nformat MMSS\nfuse 5\n"

static void test_parser(void) {
    CHECK(seg_init(NULL) == FCV_E_ARG);
    CHECK(seg_init("no_such_layout") == FCV_E_FILE);
    CHECK(layout_with(HEAD, "", 4) == FCV_E_FORMAT);                    /* 격자 없음 */
    CHECK(layout_with(HEAD, "grid 10 5\n", 4) == FCV_E_FORMAT);         /* 오름차순 아님 */
    CHECK(layout_with(HEAD, "bogus 1\ngrid 10\n", 4) == FCV_E_FORMAT);  /* 모르는 키 */
    CHECK(layout_with(HEAD, "grid 10\n", 3) == FCV_E_FORMAT);           /* 자리 하나 누락 */
    CHECK(layout_with("digits 4\ncanon 200 60\n", "grid 10\n", 4) == FCV_E_FORMAT);   /* 필수 줄 누락 */
    CHECK(layout_with(HEAD, "grid 0 10 20\ngrid 30 40\nglyph_variant 9 0x67\n", 4) == FCV_OK);
    int idx;
    CHECK(seg_grid_index(30, &idx) == FCV_OK && idx == 3);
    CHECK(seg_grid_index(35, &idx) == FCV_OK && idx == -1);
    CHECK(seg_grid_index(30, NULL) == FCV_E_ARG);
    seg_free();
    CHECK(seg_grid_index(30, &idx) == FCV_E_STATE);
}

/* T-S1 */
static void test_homography(void) {
    const double Wc = 200, Hc = 60;
    const double t[8] = { 1.3, 0.2, 40, -0.1, 1.1, 25, 0.0009, -0.0007 };
    const double cx[4] = { 0, Wc, Wc, 0 }, cy[4] = { 0, 0, Hc, Hc };
    double quad[8], h[8];
    for (int i = 0; i < 4; i++) {
        double den = t[6] * cx[i] + t[7] * cy[i] + 1;
        quad[2 * i] = (t[0] * cx[i] + t[1] * cy[i] + t[2]) / den;
        quad[2 * i + 1] = (t[3] * cx[i] + t[4] * cy[i] + t[5]) / den;
    }
    CHECK(seg_solve_homography(quad, Wc, Hc, h) == FCV_OK);
    for (int i = 0; i < 8; i++) CHECK(fabs(h[i] - t[i]) < 1e-9);
    const double same[8] = { 5, 5, 5, 5, 5, 5, 5, 5 };
    CHECK(seg_solve_homography(same, Wc, Hc, h) == FCV_E_ARG);
}

static int popcount7(int v) { int n = 0; for (int i = 0; i < 7; i++) n += v >> i & 1; return n; }

/* T-S2 */
static void test_codes(void) {
    static const int pairs[7][2] = { {0,8},{1,7},{3,9},{5,6},{5,9},{6,8},{8,9} };
    int count = 0;
    for (int a = 0; a < 10; a++) for (int b = a + 1; b < 10; b++) {
        if (popcount7(seg_digit_code(a) ^ seg_digit_code(b)) != 1) continue;
        int listed = 0;
        for (int i = 0; i < 7; i++) listed |= pairs[i][0] == a && pairs[i][1] == b;
        CHECK(listed);
        count++;
    }
    CHECK(count == 7);
    int detected = 0;
    for (int d = 0; d <= 5; d++) for (int bit = 0; bit < 7; bit++) {
        int flipped = seg_digit_code(d) ^ 1 << bit, as = -1;
        for (int k = 0; k < 10; k++) if (seg_digit_code(k) == flipped) as = k;
        detected += as < 0 || as > 5;       /* 표에 없거나 십의 자리 제약에 걸린다 */
    }
    CHECK(detected == 42);
}

static seg_result val(int seconds) {
    seg_result r;
    memset(&r, 0, sizeof r);
    r.seconds = seconds;
    r.digits[0] = seconds / 600; r.digits[1] = seconds / 60 % 10;
    r.digits[2] = seconds % 60 / 10; r.digits[3] = seconds % 10;
    r.margin = 0.3f;
    return r;
}

/* T-S3 */
static void test_vote(void) {
    seg_result r[5], out;
    CHECK(seg_vote(r, 5, &out) == FCV_E_STATE);
    CHECK(layout_with(HEAD, "grid 10 20 30 90\n", 4) == FCV_OK);

    r[0] = val(30); r[1] = val(30); r[2] = val(30); r[3] = val(20); r[4] = val(10);
    CHECK(seg_vote(r, 5, &out) == SEG_STABLE && out.seconds == 30);

    r[2] = val(20); r[3] = val(10);
    CHECK(seg_vote(r, 5, &out) == SEG_UNSTABLE);

    r[0] = val(40); r[1] = val(40); r[2] = val(40); r[3] = val(10); r[4] = val(20);
    CHECK(seg_vote(r, 5, &out) == SEG_INVALID);

    /* blank 3개를 빼면 2개만 남는다 */
    r[0] = val(30); r[1] = val(30); r[2] = val(30); r[3] = val(30); r[4] = val(30);
    r[0].blank = r[1].blank = r[2].blank = 1;
    CHECK(seg_vote(r, 5, &out) == SEG_UNSTABLE);

    /* 판독 실패 3개는 표가 되지 않는다 */
    r[0] = val(30); r[1] = val(10); r[2] = val(20); r[3] = val(20); r[4] = val(10);
    r[2].seconds = r[3].seconds = r[4].seconds = -1;
    CHECK(seg_vote(r, 5, &out) == SEG_UNSTABLE);

    /* 01:30 셋 중 하나가 00:90으로 읽혔다: 초는 같지만 십의 자리가 틀렸다 */
    r[0] = val(90); r[1] = val(90); r[2] = val(90); r[3] = val(10); r[4] = val(20);
    r[1].digits[1] = 0; r[1].digits[2] = 9;
    CHECK(seg_vote(r, 5, &out) == SEG_INVALID);

    CHECK(seg_vote(NULL, 5, &out) == FCV_E_ARG);
    CHECK(seg_vote(r, 0, &out) == FCV_E_ARG);
    seg_free();
}

/* T-S4 */
static void test_otsu(void) {
    float v[28], t, c, m;
    for (int i = 0; i < 28; i++) v[i] = i % 2 ? 200.0f + (float)(i % 7) - 3 : 30.0f + (float)(i % 7) - 3;
    CHECK(seg_otsu(v, 28, &t, &c, &m) == FCV_OK);
    CHECK(t > 33 && t < 197);
    CHECK(c > 160);
    CHECK(m > 0.3f && m <= 0.5f);

    /* 대비 < cmin → blank: 영상으로 확인한다 */
    static uint8_t img[60 * 200];
    float quad[8] = { 0, 0, 200, 0, 200, 60, 0, 60 };
    seg_result r;
    memset(img, 90, sizeof img);
    CHECK(layout_with(HEAD, "grid 10\n", 4) == FCV_OK);
    CHECK(seg_read(img, 200, 60, 200, FCV_PIX_GRAY8, &r) == FCV_E_STATE);     /* quad 전 */
    CHECK(seg_set_quad(quad) == FCV_OK);
    CHECK(seg_read(img, 200, 60, 200, FCV_PIX_GRAY8, &r) == FCV_OK);
    CHECK(r.blank == 1 && r.seconds == -1);
    seg_free();
}

void test_seg(void) {
    test_parser();
    test_homography();
    test_codes();
    test_vote();
    test_otsu();
}
