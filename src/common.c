/* common.c — 상태 코드 문자열, 버전 (§3) */
#include "foodcv.h"

int fcv_api_version(void) { return FCV_API_VERSION; }

const char *fcv_strerror(int code) {
    switch (code) {
    case FCV_OK:        return "ok";
    case FCV_E_FILE:    return "file error";
    case FCV_E_FORMAT:  return "bad format";
    case FCV_E_VERSION: return "version mismatch";
    case FCV_E_NOMEM:   return "out of memory";
    case FCV_E_ARG:     return "bad argument";
    case FCV_E_STATE:   return "bad state";
    default:            return "unknown error";
    }
}
