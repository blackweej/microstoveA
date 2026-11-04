/* test_main.c — 미니 테스트 러너. 실패가 있으면 1을 돌려준다 (§2) */
#include <string.h>
#include "foodcv.h"
#include "minitest.h"

int mt_checks = 0, mt_failures = 0;

static void test_common(void) {
    CHECK(fcv_api_version() == FCV_API_VERSION);
    CHECK(strcmp(fcv_strerror(FCV_OK), "ok") == 0);
    CHECK(strcmp(fcv_strerror(FCV_E_ARG), "bad argument") == 0);
    CHECK(strcmp(fcv_strerror(12345), "unknown error") == 0);
}

int main(void) {
    test_common();
    printf("%d checks, %d failures\n", mt_checks, mt_failures);
    return mt_failures ? 1 : 0;
}
