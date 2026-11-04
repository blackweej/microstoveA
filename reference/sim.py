"""합성 측정대 영상 생성 + C 면적 모듈 호출 (ctypes)."""
import ctypes as C, numpy as np, os
from scipy.ndimage import gaussian_filter, shift as ndshift

HERE = os.path.dirname(os.path.abspath(__file__))
lib = C.CDLL(os.path.join(HERE, os.environ.get("AREA_LIB", "libarea.so")))

class Params(C.Structure):
    _fields_ = [("ref_cm2", C.c_float), ("cx", C.c_float), ("cy", C.c_float), ("r", C.c_float),
                ("red_k", C.c_float), ("red_min", C.c_int), ("chroma_tol", C.c_float),
                ("shade_lo", C.c_float), ("shade_hi", C.c_float), ("min_part_px", C.c_int),
                ("fill_tol", C.c_float), ("count_tol", C.c_float), ("ref_dilate", C.c_int)]

class Result(C.Structure):
    _fields_ = [("status", C.c_int), ("food_cm2", C.c_float), ("scale", C.c_float), ("ref_px", C.c_int),
                ("ref_fill", C.c_float), ("n_parts", C.c_int), ("part_cm2", C.c_float * 16),
                ("perim_px", C.c_float), ("touches_edge", C.c_int)]

lib.area_init.argtypes = [C.POINTER(Params), C.c_int, C.c_int]
for f in (lib.area_calibrate,):
    f.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int]
lib.area_measure.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(Result)]

W, H, SS = 640, 480, 4          # 출력 해상도, 초표본 배율
K = 16.0                        # px/cm (참값)
PCX, PCY, PR = 320.0, 240.0, 12 * K   # 접시 중심·반지름 (px)
STATUS = {0: "OK", 1: "REF_MISSING", 2: "REF_CHANGED", 3: "REF_MERGED", 4: "EMPTY"}

# 초표본 좌표 (px 단위, 픽셀 중심)
yy, xx = np.mgrid[0:H * SS, 0:W * SS].astype(np.float32)
xx = (xx + 0.5) / SS; yy = (yy + 0.5) / SS

def disk(cx, cy, r):
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r

def ellipse(cx, cy, a, b, th):
    c, s = np.cos(th), np.sin(th); dx, dy = xx - cx, yy - cy
    u, v = c * dx + s * dy, -s * dx + c * dy
    return (u / a) ** 2 + (v / b) ** 2 <= 1

PLATES = {"white": (235, 233, 228), "blue": (60, 105, 165)}
RED = (200, 32, 36)

def ref_center(D_cm, ang):
    rr = PR - (D_cm / 2 + 1.0) * K
    return PCX + rr * np.cos(ang), PCY + rr * np.sin(ang)

def make_food(kind, rng):
    """음식 마스크(초표본)와 색 지도를 반환."""
    layers = []      # (mask, color array or tuple)
    if kind == "pizza":
        ang0 = rng.uniform(0, 2 * np.pi); R_ = 11 * K; half = np.deg2rad(25)
        ax, ay = PCX - 4 * K * np.cos(ang0), PCY - 4 * K * np.sin(ang0)
        dx, dy = xx - ax, yy - ay; rr = np.hypot(dx, dy)
        dang = np.angle(np.exp(1j * (np.arctan2(dy, dx) - ang0)))
        m = (rr <= R_) & (np.abs(dang) <= half)
        layers.append((m & (rr > R_ - 1.0 * K), (205, 150, 80)))
        layers.append((m & (rr <= R_ - 1.0 * K), (238, 195, 95)))
        for t in np.linspace(0.3, 0.8, 4):
            c = (ax + t * R_ * np.cos(ang0 + rng.uniform(-0.15, 0.15)), ay + t * R_ * np.sin(ang0 + rng.uniform(-0.15, 0.15)))
            layers.append((m & disk(*c, 1.1 * K), (172, 40, 32)))
        return m, layers
    if kind in ("chicken", "yangnyeom"):
        col = (150, 92, 42) if kind == "chicken" else (168, 48, 34)
        m = np.zeros_like(xx, bool)
        for _ in range(5):
            for _try in range(50):
                rr_, aa = rng.uniform(0, 6.5 * K), rng.uniform(0, 2 * np.pi)
                e = ellipse(PCX + rr_ * np.cos(aa), PCY + rr_ * np.sin(aa), rng.uniform(2.2, 3.0) * K,
                            rng.uniform(1.6, 2.2) * K, rng.uniform(0, np.pi))
                if not (e & m).any(): break
            m |= e
        layers.append((m, col)); return m, layers
    if kind == "dumpling":
        m = np.zeros_like(xx, bool)
        for _ in range(6):
            for _try in range(80):
                rr_, aa, th = rng.uniform(0, 6.5 * K), rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
                cx, cy = PCX + rr_ * np.cos(aa), PCY + rr_ * np.sin(aa)
                e = ellipse(cx, cy, 2.0 * K, 1.4 * K, th)
                e &= (np.cos(th) * (yy - cy) - np.sin(th) * (xx - cx)) >= -0.3 * K    # 반달
                if not (e & m).any(): break
            m |= e
        layers.append((m, (232, 226, 210))); return m, layers
    if kind == "rice":
        m = np.zeros_like(xx, bool)
        for _ in range(25):
            rr_, aa = rng.uniform(0, 6 * K), rng.uniform(0, 2 * np.pi)
            m |= disk(PCX + rr_ * np.cos(aa), PCY + rr_ * np.sin(aa), rng.uniform(1.0, 2.0) * K)
        layers.append((m, "speckle")); return m, layers
    raise ValueError(kind)

def render(plate="white", D_cm=4.0, ref_ang=0.0, food=None, rng=None, occl=0.0, food_touch_ref=False,
           noise=2.0, blur=0.7):
    rng = rng or np.random.default_rng(0)
    img = np.empty((H * SS, W * SS, 3), np.float32); img[:] = (90, 90, 95)
    pm = disk(PCX, PCY, PR); img[pm] = PLATES[plate]
    rcx, rcy = ref_center(D_cm, ref_ang); rm = disk(rcx, rcy, D_cm / 2 * K)
    img[rm] = RED
    fm = np.zeros_like(pm)
    if food:
        fm, layers = make_food(food, rng)
        fm &= pm
        # 그림자: 왼쪽 위 조명 → 오른쪽 아래로 4 px, 접시 위에서 0.72배
        sh = ndshift(fm.astype(np.float32), (4 * SS, 4 * SS), order=0) > 0.5
        sh = gaussian_filter(sh.astype(np.float32), 2 * SS) * (pm & ~fm)
        img *= (1 - 0.28 * sh)[..., None]
        for msk, col in layers:
            msk = msk & pm
            if isinstance(col, str):
                pal = np.array([(200, 150, 90), (240, 220, 150), (120, 160, 60), (200, 70, 45), (170, 120, 70)], np.float32)
                blk = (np.floor(xx / 0.35) * 7919 + np.floor(yy / 0.35) * 104729).astype(np.int64) % 5
                img[msk] = pal[blk[msk]]
            else:
                tex = rng.normal(0, 10, msk.sum())[:, None]
                img[msk] = np.clip(np.array(col, np.float32) + tex, 0, 255)
    if occl > 0:   # 기준 도형의 바깥쪽 일부를 음식 조각(갈색)이 덮음: 덮인 비율 occl
        r = D_cm / 2 * K; best = None
        for d in np.linspace(-r, r, 80):    # 반평면 경계 위치를 찾아 덮인 비율 맞춤
            ux, uy = np.cos(ref_ang), np.sin(ref_ang)
            cov = rm & (((xx - rcx) * ux + (yy - rcy) * uy) >= d)
            frac = cov.sum() / rm.sum()
            if best is None or abs(frac - occl) < abs(best[0] - occl): best = (frac, cov)
        img[best[1]] = (150, 92, 42)
        fm = fm | best[1]
    if food_touch_ref:   # 양념치킨 한 조각이 기준 도형에 붙어 있음
        e = ellipse(rcx - np.cos(ref_ang) * (D_cm / 2 + 1.2) * K, rcy - np.sin(ref_ang) * (D_cm / 2 + 1.2) * K,
                    1.4 * K, 1.0 * K, ref_ang) & pm & ~rm
        img[e] = (168, 48, 34); fm = fm | e
    # 조명 비네팅
    vig = 1 - 0.15 * ((xx - PCX) ** 2 + (yy - PCY) ** 2) / 300 ** 2
    img *= vig[..., None]
    # 센서 적분(초표본 평균) → 렌즈 흐림 → 잡음
    out = img.reshape(H, SS, W, SS, 3).mean(axis=(1, 3))
    out = gaussian_filter(out, (blur, blur, 0))
    out = np.clip(out + rng.normal(0, noise, out.shape), 0, 255).astype(np.uint8)
    true_food = (fm & ~rm).sum() / SS ** 2 / K ** 2           # cm² (기준 도형 위 가림 부분 제외)
    render.overlap = float((fm & rm).sum() / rm.sum())
    return out, true_food

def params(D_cm, **kw):
    p = dict(ref_cm2=np.pi * (D_cm / 2) ** 2, cx=PCX, cy=PCY, r=PR - 6, red_k=1.8, red_min=90,
             chroma_tol=0.035, shade_lo=0.55, shade_hi=1.12, min_part_px=40, fill_tol=0.04,
             count_tol=0.02, ref_dilate=3)
    p.update(kw); return Params(**p)

def run(img, p, calib_img):
    assert lib.area_init(C.byref(p), W, H) == 0
    ci = np.ascontiguousarray(calib_img[..., ::-1])            # BGR, OpenCV 순서
    rc = lib.area_calibrate(ci.ctypes.data, W, H, ci.strides[0], 0)
    assert rc == 0, rc
    im = np.ascontiguousarray(img[..., ::-1]); r = Result()
    assert lib.area_measure(im.ctypes.data, W, H, im.strides[0], 0, C.byref(r)) == 0
    return r
