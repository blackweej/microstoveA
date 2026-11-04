"""합성 7-세그먼트 표시창 생성기 (§8.7).

정규 좌표 200×60에 표시창을 그리고, 호모그래피로 사진 좌표에 옮긴다.
다중화: 자리 i는 주기 T 안에서 자기 시간창(길이 d·T)에만 켜지고, 프레임의 밝기는 노출 e와 겹친 비율이다.
"""
import numpy as np
from scipy.ndimage import map_coordinates

CANON_W, CANON_H = 200, 60
IMG_W, IMG_H = 320, 240
SS = 4                                  # 정규 영상 초표본 배율
CODE = (0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F)
DIGIT_X = (10, 52, 110, 152)            # 자리 왼쪽 끝. 가운데 틈은 콜론 자리
DW, TOP, MID, BOT, TH = 36, 5, 30, 55, 6


def strokes(x):
    """자리 왼쪽 끝 x에서 a..g 획 사각형 (x0, y0, x1, y1)."""
    r = x + DW
    return ((x + TH, TOP, r - TH, TOP + TH),                    # a 위
            (r - TH, TOP + TH, r, MID - TH // 2),               # b 오른쪽 위
            (r - TH, MID + TH // 2, r, BOT - TH),               # c 오른쪽 아래
            (x + TH, BOT - TH, r - TH, BOT),                    # d 아래
            (x, MID + TH // 2, x + TH, BOT - TH),               # e 왼쪽 아래
            (x, TOP + TH, x + TH, MID - TH // 2),               # f 왼쪽 위
            (x + TH, MID - TH // 2, r - TH, MID + TH // 2))     # g 가운데


def layout_text(grid, fuse=1, cmin=20):
    """display.layout 내용. 획 사각형은 가장자리 흐림을 피해 1 단위 안쪽으로 잡는다."""
    lines = ["digits 4", f"canon {CANON_W} {CANON_H}", "channel R", "polarity bright",
             f"cmin {cmin}", "blank_lead 1", "format MMSS", f"fuse {fuse}"]
    for d, x in enumerate(DIGIT_X):
        for name, (x0, y0, x1, y1) in zip("abcdefg", strokes(x)):
            lines.append(f"seg {d} {name} {x0 + 1} {y0 + 1} {x1 - 1} {y1 - 1}")
    lines.append("grid " + " ".join(str(g) for g in grid))
    return "\n".join(lines) + "\n"


def digits_of(seconds):
    m, s = divmod(seconds, 60)
    return m // 10, m % 10, s // 10, s % 10


def canon_image(seconds, levels, bg=18.0, lit=225.0, lead_blank=True):
    """자리별 밝기 levels(0..1)로 정규 영상을 그린다."""
    img = np.full((CANON_H * SS, CANON_W * SS), bg, np.float32)
    d = digits_of(seconds)
    for i, x in enumerate(DIGIT_X):
        if i == 0 and d[0] == 0 and lead_blank:
            continue
        for bit, (x0, y0, x1, y1) in enumerate(strokes(x)):
            if CODE[d[i]] >> bit & 1:
                img[y0 * SS:y1 * SS, x0 * SS:x1 * SS] = bg + (lit - bg) * levels[i]
    return img


def random_quad(rng, tilt_deg=15.0):
    """표시창 네 꼭짓점 TL,TR,BR,BL. ±tilt_deg 회전과 약한 원근."""
    a = np.deg2rad(rng.uniform(-tilt_deg, tilt_deg))
    w, h = 220 * rng.uniform(0.95, 1.05), 66 * rng.uniform(0.95, 1.05)
    corners = np.array([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)])
    corners *= 1 + np.array([[-1], [1], [1], [-1]]) * rng.uniform(-0.05, 0.05)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    centre = (IMG_W / 2 + rng.uniform(-10, 10), IMG_H / 2 + rng.uniform(-10, 10))
    return (corners @ rot.T + centre).astype(np.float32)


def homography_photo_to_canon(quad):
    src = np.asarray(quad, np.float64)
    dst = np.array([(0, 0), (CANON_W, 0), (CANON_W, CANON_H), (0, CANON_H)], np.float64)
    rows, rhs = [], []
    for (u, v), (x, y) in zip(src, dst):
        rows.append((u, v, 1, 0, 0, 0, -x * u, -x * v))
        rows.append((0, 0, 0, u, v, 1, -y * u, -y * v))
        rhs += [x, y]
    return np.append(np.linalg.solve(np.array(rows), np.array(rhs)), 1).reshape(3, 3)


def warp(canon, quad):
    """정규 영상을 사진에 옮긴다. 표시창 밖은 어두운 패널."""
    hm = homography_photo_to_canon(quad)
    v, u = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)
    p = hm @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    x, y = p[0] / p[2], p[1] / p[2]
    out = map_coordinates(canon, [y * SS, x * SS], order=1, mode="constant", cval=35.0)
    return out.reshape(IMG_H, IMG_W)


def multiplex_levels(rng, n_digits=4, d=0.25, T=2.0, e=1.0):
    """자리별 밝기 = 노출과 켜짐 시간창의 겹침 / 켜짐 길이. 노출 시작 위상은 무작위."""
    t0 = rng.uniform(0, T)
    levels = np.zeros(n_digits)
    on = d * T
    for i in range(n_digits):
        start = i * T / n_digits
        for k in (-1, 0, 1, 2):                             # 주기 경계를 넘는 겹침
            a = max(t0, start + k * T)
            b = min(t0 + e, start + k * T + on)
            levels[i] += max(0.0, b - a)
    return np.minimum(levels / on, 1.0)


def frame(seconds, quad, rng, levels=None, noise=3.0, gain=1.0):
    """BGR uint8 프레임. LED는 빨강이라 R 채널에 표시가 있고 나머지는 배경광이다."""
    levels = np.ones(4) if levels is None else levels
    r = warp(canon_image(seconds, levels), quad) * gain
    g = np.full_like(r, 30.0) * gain
    out = np.stack([g, g, r], axis=-1)                      # B, G, R
    out = out + rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)
