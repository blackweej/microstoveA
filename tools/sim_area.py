"""합성 측정대 영상 생성기. reference/sim.py에서 이식했다.

시제품과 다른 점 (§13.2):
- 접시 색은 필수 인자다. 시제품은 흰 접시가 기본이라 가림 수치가 운용 조건과 달랐다.
- 음식은 기준 원에서 food_gap_cm 이상 떼어 놓는다. None이면 시제품처럼 아무 데나 놓는다.
- rng는 음식 모양을, noise_rng는 질감과 센서 잡음을 정한다. 같은 음식을 반복 촬영할 수 있다.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndshift

W, H, SS = 640, 480, 4          # 출력 해상도, 초표본 배율
K = 16.0                        # px/cm (참값)
PCX, PCY, PR = 320.0, 240.0, 12 * K   # 접시 중심·반지름 (px)

# 초표본 좌표 (px 단위, 픽셀 중심)
yy, xx = np.mgrid[0:H * SS, 0:W * SS].astype(np.float32)
xx = (xx + 0.5) / SS
yy = (yy + 0.5) / SS

PLATES = {"white": (235, 233, 228), "blue": (60, 105, 165)}
RED = (200, 32, 36)
FOODS = ("pizza", "chicken", "yangnyeom", "dumpling", "rice")


def disk(cx, cy, r):
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def ellipse(cx, cy, a, b, th):
    c, s = np.cos(th), np.sin(th)
    dx, dy = xx - cx, yy - cy
    u, v = c * dx + s * dy, -s * dx + c * dy
    return (u / a) ** 2 + (v / b) ** 2 <= 1


def ref_center(D_cm, ang):
    rr = PR - (D_cm / 2 + 1.0) * K
    return PCX + rr * np.cos(ang), PCY + rr * np.sin(ang)


def make_food(kind, rng):
    """음식 마스크(초표본)와 색 층을 돌려준다."""
    layers = []      # (mask, 색 또는 "speckle")
    if kind == "pizza":
        ang0 = rng.uniform(0, 2 * np.pi)
        R_ = 11 * K
        half = np.deg2rad(25)
        ax, ay = PCX - 4 * K * np.cos(ang0), PCY - 4 * K * np.sin(ang0)
        dx, dy = xx - ax, yy - ay
        rr = np.hypot(dx, dy)
        dang = np.angle(np.exp(1j * (np.arctan2(dy, dx) - ang0)))
        m = (rr <= R_) & (np.abs(dang) <= half)
        layers.append((m & (rr > R_ - 1.0 * K), (205, 150, 80)))
        layers.append((m & (rr <= R_ - 1.0 * K), (238, 195, 95)))
        for t in np.linspace(0.3, 0.8, 4):
            c = (ax + t * R_ * np.cos(ang0 + rng.uniform(-0.15, 0.15)),
                 ay + t * R_ * np.sin(ang0 + rng.uniform(-0.15, 0.15)))
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
                if not (e & m).any():
                    break
            m |= e
        layers.append((m, col))
        return m, layers
    if kind == "dumpling":
        m = np.zeros_like(xx, bool)
        for _ in range(6):
            for _try in range(80):
                rr_, aa, th = rng.uniform(0, 6.5 * K), rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
                cx, cy = PCX + rr_ * np.cos(aa), PCY + rr_ * np.sin(aa)
                e = ellipse(cx, cy, 2.0 * K, 1.4 * K, th)
                e &= (np.cos(th) * (yy - cy) - np.sin(th) * (xx - cx)) >= -0.3 * K    # 반달
                if not (e & m).any():
                    break
            m |= e
        layers.append((m, (232, 226, 210)))
        return m, layers
    if kind == "rice":
        m = np.zeros_like(xx, bool)
        for _ in range(25):
            rr_, aa = rng.uniform(0, 6 * K), rng.uniform(0, 2 * np.pi)
            m |= disk(PCX + rr_ * np.cos(aa), PCY + rr_ * np.sin(aa), rng.uniform(1.0, 2.0) * K)
        layers.append((m, "speckle"))
        return m, layers
    raise ValueError(kind)


def placed_food(kind, rng, pm, rcx, rcy, D_cm, gap_cm):
    """기준 원에서 gap_cm 이상 떨어진 음식이 나올 때까지 다시 뽑는다."""
    if gap_cm is None:
        fm, layers = make_food(kind, rng)
        return fm & pm, layers
    keep_out = disk(rcx, rcy, (D_cm / 2 + gap_cm) * K)
    for _ in range(200):
        fm, layers = make_food(kind, rng)
        fm = fm & pm
        if not (fm & keep_out).any():
            return fm, layers
    raise RuntimeError(f"{kind}: 기준 원을 피한 배치를 찾지 못함")


def render(plate, D_cm=4.0, ref_ang=0.0, food=None, rng=None, noise_rng=None, occl=0.0,
           food_touch_ref=False, touch_scale=1.0, noise=2.0, blur=0.7, food_gap_cm=1.0):
    """RGB uint8 영상과 음식 참 면적(cm²)을 돌려준다."""
    rng = rng or np.random.default_rng(0)
    noise_rng = noise_rng or rng
    img = np.empty((H * SS, W * SS, 3), np.float32)
    img[:] = (90, 90, 95)
    pm = disk(PCX, PCY, PR)
    img[pm] = PLATES[plate]
    rcx, rcy = ref_center(D_cm, ref_ang)
    rm = disk(rcx, rcy, D_cm / 2 * K)
    img[rm] = RED
    fm = np.zeros_like(pm)
    if food:
        fm, layers = placed_food(food, rng, pm, rcx, rcy, D_cm, food_gap_cm)
        # 그림자: 왼쪽 위 조명 → 오른쪽 아래로 4 px, 접시 위에서 0.72배
        sh = ndshift(fm.astype(np.float32), (4 * SS, 4 * SS), order=0) > 0.5
        sh = gaussian_filter(sh.astype(np.float32), 2 * SS) * (pm & ~fm)
        img *= (1 - 0.28 * sh)[..., None]
        for msk, col in layers:
            msk = msk & pm
            if isinstance(col, str):
                pal = np.array([(200, 150, 90), (240, 220, 150), (120, 160, 60), (200, 70, 45), (170, 120, 70)],
                               np.float32)
                blk = (np.floor(xx / 0.35) * 7919 + np.floor(yy / 0.35) * 104729).astype(np.int64) % 5
                img[msk] = pal[blk[msk]]
            else:
                tex = noise_rng.normal(0, 10, msk.sum())[:, None]
                img[msk] = np.clip(np.array(col, np.float32) + tex, 0, 255)
    if occl > 0:   # 기준 원의 바깥쪽 일부를 음식 조각(갈색)이 덮음: 덮인 비율 occl
        r = D_cm / 2 * K
        best = None
        for d in np.linspace(-r, r, 80):    # 반평면 경계 위치를 찾아 덮인 비율 맞춤
            ux, uy = np.cos(ref_ang), np.sin(ref_ang)
            cov = rm & (((xx - rcx) * ux + (yy - rcy) * uy) >= d)
            frac = cov.sum() / rm.sum()
            if best is None or abs(frac - occl) < abs(best[0] - occl):
                best = (frac, cov)
        img[best[1]] = (150, 92, 42)
        fm = fm | best[1]
    if food_touch_ref:   # 양념치킨 한 조각이 기준 원에 붙어 있음. touch_scale로 조각 크기 조절
        off = D_cm / 2 + 1.2 * touch_scale
        e = ellipse(rcx - np.cos(ref_ang) * off * K, rcy - np.sin(ref_ang) * off * K,
                    1.4 * touch_scale * K, 1.0 * touch_scale * K, ref_ang) & pm & ~rm
        img[e] = (168, 48, 34)
        fm = fm | e
    # 조명 비네팅
    vig = 1 - 0.15 * ((xx - PCX) ** 2 + (yy - PCY) ** 2) / 300 ** 2
    img *= vig[..., None]
    # 센서 적분(초표본 평균) → 렌즈 흐림 → 잡음
    out = img.reshape(H, SS, W, SS, 3).mean(axis=(1, 3))
    out = gaussian_filter(out, (blur, blur, 0))
    out = np.clip(out + noise_rng.normal(0, noise, out.shape), 0, 255).astype(np.uint8)
    true_food = (fm & ~rm).sum() / SS ** 2 / K ** 2           # cm² (기준 원 위 가림 부분 제외)
    return out, true_food


def area_config(D_cm=4.0, **kw):
    """config/area.cfg에 해당하는 합성 장면용 값."""
    cfg = dict(ref_diameter_cm=D_cm, plate_cx=PCX, plate_cy=PCY, plate_r=PR - 6, red_k=1.8, red_min=90,
               chroma_tol=0.035, shade_lo=0.55, shade_hi=1.12, min_part_px=40, fill_tol=0.04,
               count_tol=0.02, ref_dilate=3, light_lo=0.80, light_hi=1.08, edge_tol=0.10, empty_tol=0.01)
    cfg.update(kw)
    return cfg
