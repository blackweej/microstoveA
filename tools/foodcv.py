"""foodcv ctypes 래퍼 (§11). C파트가 import해서 쓴다."""
import ctypes as C
import math
import re
import warnings
from dataclasses import dataclass

import numpy as np

API_VERSION = 3
PIX_BGR24, PIX_RGB24, PIX_GRAY8 = 0, 1, 2
FCV_E_STATE = -6
AREA_MAX_PARTS = 16
AREA_STATUS = {0: "OK", 1: "REF_MISSING", 2: "REF_CHANGED", 3: "REF_MERGED", 4: "EMPTY",
               5: "LIGHT_CHANGED", 6: "PLATE_MOVED", 7: "NOT_EMPTY"}
AREA_OK, AREA_REF_MISSING, AREA_REF_CHANGED, AREA_REF_MERGED, AREA_EMPTY = range(5)
AREA_LIGHT_CHANGED, AREA_PLATE_MOVED, AREA_NOT_EMPTY = 5, 6, 7
SEG_STABLE, SEG_UNSTABLE, SEG_INVALID = 0, 1, 2
VOTE_FRAMES = 5

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


class _FcvResult(C.Structure):
    _fields_ = [("class_id", C.c_int), ("prob", C.c_float), ("rejected", C.c_int)]


class _SegResult(C.Structure):
    _fields_ = [("seconds", C.c_int), ("digits", C.c_int * 4), ("blank", C.c_int), ("margin", C.c_float)]


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
    """라이브러리를 열고 모든 함수의 형식을 지정한 뒤 API 버전을 확인한다."""
    lib = C.CDLL(lib_path)
    img = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int]
    float_p, int_p = C.POINTER(C.c_float), C.POINTER(C.c_int)
    for name, res, args in (
        ("fcv_api_version", C.c_int, []),
        ("fcv_strerror", C.c_char_p, [C.c_int]),
        ("fcv_init", C.c_int, [C.c_char_p]),
        ("fcv_num_classes", C.c_int, []),
        ("fcv_class_name", C.c_char_p, [C.c_int]),
        ("fcv_model_hash", C.c_uint32, []),
        ("fcv_preprocess", C.c_int, img + [int_p, float_p]),
        ("fcv_classify", C.c_int, img + [int_p, float_p, C.c_int, C.POINTER(_FcvResult)]),
        ("fcv_free", None, []),
        ("area_init", C.c_int, [C.POINTER(_AreaParams), C.c_int, C.c_int]),
        ("area_calibrate", C.c_int, img),
        ("area_measure", C.c_int, img + [C.POINTER(_AreaResult)]),
        ("area_debug_labels", C.c_int, [C.c_void_p]),
        ("area_free", None, []),
        ("seg_init", C.c_int, [C.c_char_p]),
        ("seg_set_quad", C.c_int, [float_p]),
        ("seg_read", C.c_int, img + [C.POINTER(_SegResult)]),
        ("seg_vote", C.c_int, [C.POINTER(_SegResult), C.c_int, C.POINTER(_SegResult)]),
        ("seg_grid_index", C.c_int, [C.c_int, int_p]),
        ("seg_free", None, []),
    ):
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    version = lib.fcv_api_version()
    if version != API_VERSION:
        raise RuntimeError(f"foodcv API {version}, 래퍼는 {API_VERSION}")
    return lib


def _check(lib, rc):
    if rc < 0:
        raise FoodCVError(rc, lib.fcv_strerror(rc).decode())
    return rc


def _config_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.split("#", 1)[0].strip() for line in f]


def read_area_cfg(path):
    """§4.6 키=값 파일을 읽는다. 값이 없거나 ?이면 예외(기본값 금지)."""
    cfg = {}
    for line in _config_lines(path):
        cfg.update(re.findall(r"(\w+)\s*=\s*(\S+)", line))
    missing = [k for k in AREA_KEYS if cfg.get(k, "?") == "?"]
    if missing:
        raise ValueError(f"{path}: 값이 없는 키: {', '.join(missing)}")
    return {k: float(cfg[k]) for k in AREA_KEYS}


def read_roi_cfg(path):
    """§5.1 roi.cfg: x y size → C가 받는 [x, y, w, h]."""
    x, y, size = (int(v) for v in " ".join(_config_lines(path)).split())
    return x, y, size, size


def read_quad(path):
    """§12 display.quad: float 8개."""
    quad = [float(v) for v in " ".join(_config_lines(path)).split()]
    if len(quad) != 8:
        raise ValueError(f"{path}: float 8개가 필요하다")
    return quad


def read_layout_options(path):
    """래퍼가 직접 쓰는 display.layout 줄 (§11)."""
    opts = {"fuse": 1, "polarity": "bright"}
    for line in _config_lines(path):
        parts = line.split()
        if len(parts) == 2 and parts[0] == "fuse":
            opts["fuse"] = int(parts[1])
        elif len(parts) == 2 and parts[0] == "polarity":
            opts["polarity"] = parts[1]
    if opts["fuse"] < 1:
        raise ValueError(f"{path}: fuse는 1 이상이어야 한다")
    return opts


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


def preprocess(lib, frame, roi=None, fmt=PIX_BGR24):
    """fcv_preprocess를 불러 3×96×96 float32를 돌려준다 (§5.4)."""
    frame = _frame(frame)
    h, w = frame.shape[:2]
    out = np.empty((3, 96, 96), np.float32)
    roi_c = (C.c_int * 4)(*roi) if roi is not None else None
    _check(lib, lib.fcv_preprocess(frame.ctypes.data, w, h, frame.strides[0], fmt, roi_c,
                                   out.ctypes.data_as(C.POINTER(C.c_float))))
    return out


class AreaMeter:
    """면적 측정. area_init은 영상 크기가 필요해서 보정 때 부른다 (§11)."""

    def __init__(self, lib, cfg):
        self._lib, self._cfg, self._size = lib, cfg, None

    def calibrate(self, frame_bgr):
        frame = _frame(frame_bgr)
        h, w = frame.shape[:2]
        if self._size != (w, h):
            p = _area_params(self._cfg)
            _check(self._lib, self._lib.area_init(C.byref(p), w, h))
            self._size = (w, h)
        return _check(self._lib, self._lib.area_calibrate(frame.ctypes.data, w, h, frame.strides[0], PIX_BGR24))

    def measure(self, frame_bgr):
        if self._size is None:
            raise FoodCVError(FCV_E_STATE, "calibrate first")
        frame = _frame(frame_bgr)
        h, w = frame.shape[:2]
        r = _AreaResult()
        _check(self._lib, self._lib.area_measure(frame.ctypes.data, w, h, frame.strides[0], PIX_BGR24, C.byref(r)))
        n = min(r.n_parts, AREA_MAX_PARTS)
        if r.n_parts > AREA_MAX_PARTS:
            warnings.warn(f"조각 {r.n_parts}개: 앞 {AREA_MAX_PARTS}개만 있어 γ ≠ 1이면 ΣA^γ를 계산할 수 없다")
        return AreaResult(r.status, AREA_STATUS[r.status], r.food_cm2, r.scale, r.ref_px, r.ref_fill, r.n_parts,
                          list(r.part_cm2[:n]), list(r.part_fill[:n]), r.perim_px, r.touches_edge)

    def debug_labels(self):
        if self._size is None:
            raise FoodCVError(FCV_E_STATE, "calibrate first")
        w, h = self._size
        mask = np.empty((h, w), np.uint8)
        _check(self._lib, self._lib.area_debug_labels(mask.ctypes.data))
        return mask


class FoodCV:
    def __init__(self, lib_path, weight_path, area_cfg, display_layout, display_quad, roi_cfg):
        self._lib = lib = load_library(lib_path)
        _check(lib, lib.fcv_init(weight_path.encode()))
        _check(lib, lib.seg_init(display_layout.encode()))
        self._area = AreaMeter(lib, read_area_cfg(area_cfg))
        self._roi = (C.c_int * 4)(*read_roi_cfg(roi_cfg))
        self._layout = read_layout_options(display_layout)
        self.set_quad(read_quad(display_quad))
        self._k = _check(lib, lib.fcv_num_classes())

    def set_quad(self, quad):
        """표시창 네 꼭짓점 TL,TR,BR,BL을 바꾼다."""
        q = (C.c_float * 8)(*(float(v) for v in np.ravel(quad)))
        _check(self._lib, self._lib.seg_set_quad(q))

    def classify(self, frame_bgr):
        """(k, 이름, p, rejected, probs)"""
        frame = _frame(frame_bgr)
        h, w = frame.shape[:2]
        probs = np.empty(self._k, np.float32)
        r = _FcvResult()
        _check(self._lib, self._lib.fcv_classify(frame.ctypes.data, w, h, frame.strides[0], PIX_BGR24, self._roi,
                                                 probs.ctypes.data_as(C.POINTER(C.c_float)), self._k, C.byref(r)))
        name = self._lib.fcv_class_name(r.class_id).decode("utf-8")
        return r.class_id, name, r.prob, bool(r.rejected), probs

    def calibrate_plate(self, empty_frame_bgr):
        return self._area.calibrate(empty_frame_bgr)

    def measure_area(self, frame_bgr):
        return self._area.measure(frame_bgr)

    def _fuse(self, frames):
        """§11 fuse장씩 픽셀별로 합성한다. 켜진 획은 bright면 최댓값, dark면 최솟값에 남는다."""
        fuse = self._layout["fuse"]
        if len(frames) != VOTE_FRAMES * fuse:
            raise ValueError(f"프레임 {VOTE_FRAMES * fuse}장이 필요하다 ({len(frames)}장 받음)")
        reduce = np.maximum if self._layout["polarity"] == "bright" else np.minimum
        return [reduce.reduce(np.stack(frames[i:i + fuse])) for i in range(0, len(frames), fuse)]

    def read_display(self, frames_bgr):
        """(SEG_*, seconds). STABLE이 아니면 seconds는 -1."""
        results = (_SegResult * VOTE_FRAMES)()
        for i, f in enumerate(self._fuse(frames_bgr)):
            frame = _frame(f)
            h, w = frame.shape[:2]
            _check(self._lib, self._lib.seg_read(frame.ctypes.data, w, h, frame.strides[0], PIX_BGR24,
                                                 C.byref(results[i])))
        out = _SegResult()
        status = _check(self._lib, self._lib.seg_vote(results, VOTE_FRAMES, C.byref(out)))
        return status, out.seconds if status == SEG_STABLE else -1

    def grid_index(self, seconds):
        """격자 위치. 격자 밖이면 -1."""
        index = C.c_int()
        _check(self._lib, self._lib.seg_grid_index(seconds, C.byref(index)))
        return index.value

    def model_hash(self):
        return self._lib.fcv_model_hash()

    def close(self):
        self._lib.fcv_free()
        self._lib.area_free()
        self._lib.seg_free()
