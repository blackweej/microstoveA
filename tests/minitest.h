/* minitest.h — 의존성 없는 미니 테스트 매크로 (§2) */
#ifndef MINITEST_H
#define MINITEST_H
#include <stdio.h>
extern int mt_checks, mt_failures;
#define CHECK(cond) do { mt_checks++; if (!(cond)) { mt_failures++; \
    printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)
#endif
