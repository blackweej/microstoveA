"""foodcv ctypes 래퍼 (§11). C파트가 import해서 쓴다."""
import ctypes as C
import math
import re
import warnings
from dataclasses import dataclass

import numpy as np

API_VERSION = 3
PIX_BGR24, PIX_RGB24, PIX_GRAY8 = 0, 1, 2
AREA_MAX_PARTS = 16
AREA_STATUS = {0: "OK", 1: "REF_MISSING", 2: "REF_CHANGED", 3: "REF_MERGED", 4: "EMPTY",
               5: "LIGHT_CHANGED", 6: "PLATE_MOVED", 7: "NOT_EMPTY"}
AREA_OK, AREA_REF_MISSING, AREA_REF_CHANGED, AREA_REF_MERGED, AREA_EMPTY = range(5)
AREA_LIGHT_CHANGED, AREA_PLATE_MOVED, AREA_NOT_EMPTY = 5, 6, 7

AREA_KEYS = ("ref_diameter_cm", "plate_cx", "plate_cy", "plate_r", "red_k", "red_min", "chroma_tol",
             "shade_lo", "shade_hi", "min_part_px", "fill_tol", "count_tol", "ref_dilate",
             "light_lo", "light_hi", "edge_tol", "empty_tol")


class FoodCVError(Exception):
    """음수 상태 코드."""

    def __init__(self, code, message):
        super().__init__(f"{message} ({code})")
        self.code = code


class _AreaParams(C.Structure):
    _fields_ = [("ref_cm2", C.c_float), ("cx", C.c_float), ("cy", C.c_float), ("r", C.c_float),
                ("red_k", C.c_float), ("red_min", C.c_int), ("chroma_tol", C.c_float),
                ("shade_lo", C.c_float), ("shade_hi", C.c_float), ("min_part_px", C.c_int),
                ("fill_tol", C.c_float), ("count_tol", C.c_float), ("ref_dilate", C.c_int),
                ("light_lo", C.c_float), ("light_hi", C.c_float), ("edge_tol", C.c_float),
                ("empty_tol", C.c_float)]


class _AreaResult(C.Structure):
    _fields_ = [("status", C.c_int), ("food_cm2", C.c_float), ("scale", C.c_float), ("ref_px", C.c_int),
                ("ref_fill", C.c_float), ("n_parts", C.c_int),
                ("part_cm2", C.c_float * AREA_MAX_PARTS), ("part_fill", C.c_float * AREA_MAX_PARTS),
                ("perim_px", C.c_float), ("touches_edge", C.c_int)]


@dataclass
class AreaResult:
    status: int
    status_name: str
    food_cm2: float
    scale: float
    ref_px: int
    ref_fill: float
    n_parts: int
    part_cm2: list
    part_fill: list
    perim_px: float
    touches_edge: int


def load_library(lib_path):
    """라이브러리를 열고 함수 형식을 지정한 뒤 API 버전을 확인한다."""
    lib = C.CDLL(lib_path)
    img = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int]
    for name, res, args in (
        ("fcv_api_version", C.c_int, []),
        ("fcv_strerror", C.c_char_p, [C.c_int]),
        ("area_init", C.c_int, [C.POINTER(_AreaParams), C.c_int, C.c_int]),
        ("area_calibrate", C.c_int, img),
        ("area_measure", C.c_int, img + [C.POINTER(_AreaResult)]),
        ("area_debug_labels", C.c_int, [C.c_void_p]),
        ("area_free", None, []),
    ):
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    version = lib.fcv_api_version()
    if version != API_VERSION:
        raise RuntimeError(f"foodcv API {version}, 래퍼는 {API_VERSION}")
    return lib


def read_area_cfg(path):
    """§4.6 키=값 파일을 읽는다. 값이 없거나 ?이면 예외(기본값 금지)."""
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            for key, value in re.findall(r"(\w+)\s*=\s*(\S+)", line.split("#", 1)[0]):
                cfg[key] = value
    missing = [k for k in AREA_KEYS if cfg.get(k, "?") == "?"]
    if missing:
        raise ValueError(f"{path}: 값이 없는 키: {', '.join(missing)}")
    return {k: float(cfg[k]) for k in AREA_KEYS}


def _area_params(cfg):
    p = _AreaParams()
    p.ref_cm2 = math.pi * (cfg["ref_diameter_cm"] / 2) ** 2     # §4.6 ref_cm2 = π(d/2)²
    p.cx, p.cy, p.r = cfg["plate_cx"], cfg["plate_cy"], cfg["plate_r"]
    for name in ("red_k", "chroma_tol", "shade_lo", "shade_hi", "fill_tol", "count_tol",
                 "light_lo", "light_hi", "edge_tol", "empty_tol"):
        setattr(p, name, cfg[name])
    for name in ("red_min", "min_part_px", "ref_dilate"):
        setattr(p, name, int(cfg[name]))
    return p


def _frame(frame):
    """H×W×3 uint8을 C 연속 배열로 만든다."""
    frame = np.ascontiguousarray(frame)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("입력은 H×W×3 uint8이어야 한다")
    return frame


class AreaMeter:
    """면적 측정. area_init은 영상 크기가 필요해서 첫 보정 때 부른다 (§11)."""

    def __init__(self, lib, cfg):
        self._lib, self._cfg, self._size = lib, cfg, None

    def _check(self, rc):
        if rc < 0:
            raise FoodCVError(rc, self._lib.fcv_strerror(rc).decode())
        return rc

    def calibrate(self, frame_bgr):
        frame = _frame(frame_bgr)
        h, w = frame.shape[:2]
        if self._size != (w, h):
            p = _area_params(self._cfg)
            self._check(self._lib.area_init(C.byref(p), w, h))
            self._size = (w, h)
        return self._check(self._lib.area_calibrate(frame.ctypes.data, w, h, frame.strides[0], PIX_BGR24))

    def measure(self, frame_bgr):
        if self._size is None:
            raise FoodCVError(-6, "calibrate first")
        frame = _frame(frame_bgr)
        h, w = frame.shape[:2]
        r = _AreaResult()
        self._check(self._lib.area_measure(frame.ctypes.data, w, h, frame.strides[0], PIX_BGR24, C.byref(r)))
        n = min(r.n_parts, AREA_MAX_PARTS)
        if r.n_parts > AREA_MAX_PARTS:
            warnings.warn(f"조각 {r.n_parts}개: 앞 {AREA_MAX_PARTS}개만 있어 γ ≠ 1이면 ΣA^γ를 계산할 수 없다")
        return AreaResult(r.status, AREA_STATUS[r.status], r.food_cm2, r.scale, r.ref_px, r.ref_fill, r.n_parts,
                          list(r.part_cm2[:n]), list(r.part_fill[:n]), r.perim_px, r.touches_edge)

    def debug_labels(self):
        w, h = self._size
        mask = np.empty((h, w), np.uint8)
        self._check(self._lib.area_debug_labels(mask.ctypes.data))
        return mask
