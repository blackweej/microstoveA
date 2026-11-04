/* test_weights.c — T-W1 정상 로드와 훼손별 오류 코드 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/internal.h"
#include "minitest.h"

#define TMP "t_w1.fcv.tmp"

static uint8_t *orig;
static long size;

static int load_with(long offset, int xor_mask, long new_size) {
    FILE *f = fopen(TMP, "wb");
    if (!f) return 1;
    uint8_t saved = 0;
    if (offset >= 0) { saved = orig[offset]; orig[offset] ^= (uint8_t)xor_mask; }
    long n = new_size < size ? new_size : size;
    fwrite(orig, 1, (size_t)n, f);
    for (long i = size; i < new_size; i++) fputc(0, f);
    fclose(f);
    if (offset >= 0) orig[offset] = saved;
    int rc = fcv_init(TMP);
    fcv_free();
    remove(TMP);
    return rc;
}

void test_weights(void) {
    FILE *f = fopen("tests/golden/model.fcv", "rb");
    CHECK(f != NULL);
    if (!f) return;
    fseek(f, 0, SEEK_END); size = ftell(f); fseek(f, 0, SEEK_SET);
    orig = malloc((size_t)size);
    CHECK(fread(orig, 1, (size_t)size, f) == (size_t)size);
    fclose(f);

    CHECK(fcv_init("no_such_file.fcv") == FCV_E_FILE);
    CHECK(load_with(-1, 0, size) == FCV_OK);
    CHECK(load_with(0, 1, size) == FCV_E_FORMAT);            /* 매직 */
    CHECK(load_with(4, 1, size) == FCV_E_VERSION);           /* 버전 */
    CHECK(load_with(-1, 0, size - 1) == FCV_E_FORMAT);       /* 크기: 짧음 */
    CHECK(load_with(-1, 0, size + 1) == FCV_E_FORMAT);       /* 크기: 김 */
    CHECK(load_with(-1, 0, 10) == FCV_E_FORMAT);             /* 헤더도 안 됨 */
    CHECK(load_with(size - 1, 1, size) == FCV_E_FORMAT);     /* 해시 */
    CHECK(load_with(size - 100, 1, size) == FCV_E_FORMAT);   /* 가중치 */
    CHECK(load_with(8, 1, size) == FCV_E_FORMAT);            /* K */
    CHECK(load_with(24, 1, size) == FCV_E_FORMAT);           /* 채널 */
    CHECK(load_with(40, 1, size) == FCV_E_FORMAT);           /* μ: 해시 범위 §7.1 */
    CHECK(load_with(64, 1, size) == FCV_E_FORMAT);           /* τ */
    CHECK(load_with(68, 1, size) == FCV_E_FORMAT);           /* 클래스 이름 */
    CHECK(fcv_state == FCV_STATE_NONE);
    free(orig);
}
