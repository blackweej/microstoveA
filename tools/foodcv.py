"""foodcv ctypes 래퍼 (§11). C파트가 import해서 쓴다."""
import ctypes as C

API_VERSION = 3


class FoodCVError(Exception):
    """음수 상태 코드."""

    def __init__(self, code, message):
        super().__init__(f"{message} ({code})")
        self.code = code


def load_library(lib_path):
    """라이브러리를 열고 API 버전을 확인한다."""
    lib = C.CDLL(lib_path)
    lib.fcv_api_version.argtypes = []
    lib.fcv_api_version.restype = C.c_int
    lib.fcv_strerror.argtypes = [C.c_int]
    lib.fcv_strerror.restype = C.c_char_p
    version = lib.fcv_api_version()
    if version != API_VERSION:
        raise RuntimeError(f"foodcv API {version}, 래퍼는 {API_VERSION}")
    return lib
