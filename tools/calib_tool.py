"""설치 보정 GUI (§12). OpenCV 창 하나.

    python tools/calib_tool.py plate   --image plate.png --cfg config/area.cfg
    python tools/calib_tool.py quad    --image panel.png --out config/display.quad
    python tools/calib_tool.py roi     --image plate.png --out config/roi.cfg
    python tools/calib_tool.py strokes --image panel.png --quad config/display.quad --out strokes.txt

plate: 접시 가장자리 3점 클릭. quad: 표시창 꼭짓점 TL,TR,BR,BL 클릭. roi: 정사각형 대각 두 점 클릭.
strokes: 정규화된 표시창에서 획 사각형을 드래그한다. 순서는 자리 0의 a..g, 자리 1의 a..g, ...
Enter 저장, Esc 취소, Backspace 마지막 점 삭제. GUI가 있는 opencv-python이 필요하다.
"""
import argparse
import re
import sys

import numpy as np

PLATE_MARGIN_PX = 6       # §4.6 r는 접시 반지름 − 6 px
ROI_UNIT = 96             # §5.1


def circle_from_points(p1, p2, p3):
    """세 점을 지나는 원의 (cx, cy, r). 세 점이 한 직선이면 ValueError."""
    (ax, ay), (bx, by), (cx, cy) = p1, p2, p3
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        raise ValueError("세 점이 한 직선 위에 있다")
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return ux, uy, float(np.hypot(ax - ux, ay - uy))


def snap_roi(p1, p2, width, height):
    """두 점으로 고른 정사각형을 96 배수로 맞추고 영상 안에 넣는다. (x, y, size)"""
    span = max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))
    limit = min(width, height) // ROI_UNIT * ROI_UNIT
    size = min(max(int(round(span / ROI_UNIT)) * ROI_UNIT, ROI_UNIT), limit)
    if size <= 0:
        raise ValueError("영상이 96 px보다 작다")
    cx, cy = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    x = min(max(int(round(cx - size / 2)), 0), width - size)
    y = min(max(int(round(cy - size / 2)), 0), height - size)
    return x, y, size


def update_area_cfg(text, cx, cy, plate_radius):
    """area.cfg의 접시 값을 바꾼다. 다른 줄은 그대로 둔다."""
    values = {"plate_cx": f"{cx:.1f}", "plate_cy": f"{cy:.1f}", "plate_r": f"{plate_radius - PLATE_MARGIN_PX:.1f}"}
    for key, value in values.items():
        text, n = re.subn(rf"\b{key}\s*=\s*\S+", f"{key} = {value}", text)
        if n != 1:
            raise ValueError(f"area.cfg에 {key}가 {n}번 있다")
    return text


def stroke_lines(rects):
    """드래그한 사각형을 display.layout `seg` 줄로 바꾼다. 7개씩 한 자리."""
    if len(rects) % 7:
        raise ValueError("획은 자리마다 7개여야 한다")
    lines = []
    for i, (x0, y0, x1, y1) in enumerate(rects):
        lines.append(f"seg {i // 7} {'abcdefg'[i % 7]} {min(x0, x1)} {min(y0, y1)} {max(x0, x1)} {max(y0, y1)}")
    return lines


def rectify(cv2, image, quad, canon_w, canon_h, zoom=4):
    src = np.array(quad, np.float32).reshape(4, 2)
    dst = np.array([(0, 0), (canon_w, 0), (canon_w, canon_h), (0, canon_h)], np.float32) * zoom
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (canon_w * zoom, canon_h * zoom))


def collect_clicks(cv2, image, want, title):
    """점을 want개 받는다. want가 None이면 Enter까지. Esc면 None."""
    points = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and (want is None or len(points) < want):
            points.append((x, y))

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        view = image.copy()
        for i, p in enumerate(points):
            cv2.circle(view, p, 3, (0, 255, 0), -1)
            cv2.putText(view, str(i + 1), (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow(title, view)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(title)
            return None
        if key == 8 and points:
            points.pop()
        if key == 13 and (want is None or len(points) == want):
            cv2.destroyWindow(title)
            return points


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("plate", "quad", "roi", "strokes"))
    ap.add_argument("--image", required=True)
    ap.add_argument("--cfg", default="config/area.cfg")
    ap.add_argument("--quad", default="config/display.quad")
    ap.add_argument("--out")
    ap.add_argument("--canon", type=int, nargs=2, default=(200, 60))
    args = ap.parse_args()

    import cv2
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        print(f"{args.image}을(를) 읽지 못했다", file=sys.stderr)
        return 1
    h, w = image.shape[:2]

    if args.mode == "plate":
        pts = collect_clicks(cv2, image, 3, "plate: 3 points")
        if pts is None:
            return 1
        cx, cy, r = circle_from_points(*pts)
        with open(args.cfg, encoding="utf-8") as f:
            text = update_area_cfg(f.read(), cx, cy, r)
        with open(args.cfg, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"접시 중심 ({cx:.1f}, {cy:.1f}), 반지름 {r:.1f}")
    elif args.mode == "quad":
        pts = collect_clicks(cv2, image, 4, "quad: TL TR BR BL")
        if pts is None:
            return 1
        with open(args.out or "config/display.quad", "w", encoding="utf-8") as f:
            f.write(" ".join(f"{x} {y}" for x, y in pts) + "\n")
    elif args.mode == "roi":
        pts = collect_clicks(cv2, image, 2, "roi: 2 corners")
        if pts is None:
            return 1
        x, y, size = snap_roi(pts[0], pts[1], w, h)
        with open(args.out or "config/roi.cfg", "w", encoding="utf-8") as f:
            f.write(f"{x} {y} {size}\n")
        print(f"ROI {x} {y} {size}")
    else:
        with open(args.quad, encoding="utf-8") as f:
            quad = [float(v) for v in f.read().split()]
        zoom = 4
        flat = rectify(cv2, image, quad, args.canon[0], args.canon[1], zoom)
        pts = collect_clicks(cv2, flat, None, "strokes: 2 clicks per stroke")
        if pts is None:
            return 1
        if len(pts) % 2:
            print("점 수가 홀수다", file=sys.stderr)
            return 1
        rects = [(a[0] // zoom, a[1] // zoom, b[0] // zoom, b[1] // zoom) for a, b in zip(pts[::2], pts[1::2])]
        with open(args.out or "strokes.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(stroke_lines(rects)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
