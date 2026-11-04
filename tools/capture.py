"""고정 설정 웹캠 촬영 (§12).

    python tools/capture.py --camera 0 --out data/images --count 20 --prefix pizza_01

자동 초점·노출·화이트밸런스를 끄고, 실제로 먹었는지 확인해 camera_check.json에 기록한다.
백엔드마다 값이 달라 실패할 수 있다. 실패는 숨기지 않고 기록하고 종료 코드 1로 알린다.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

EXPOSURE_MANUAL = {"dshow": 0.25, "msmf": 0.25, "v4l2": 1}     # 백엔드별 CAP_PROP_AUTO_EXPOSURE 수동 값
BRIGHTNESS_MIN_CHANGE = 3.0


def mean_brightness(cap, drop):
    for _ in range(drop):
        cap.grab()
    ok, frame = cap.read()
    return float(frame.mean()) if ok else None


def exposure_responds(before, after):
    """노출을 바꿨을 때 밝기가 실제로 달라졌는가."""
    return before is not None and after is not None and abs(after - before) >= BRIGHTNESS_MIN_CHANGE


def check_camera(cap, backend, drop):
    """설정이 먹었는지 확인한 결과를 돌려준다."""
    result = {"backend": backend, "requested": {}, "readback": {}}
    wanted = {"CAP_PROP_AUTOFOCUS": (cv2.CAP_PROP_AUTOFOCUS, 0),
              "CAP_PROP_AUTO_WB": (cv2.CAP_PROP_AUTO_WB, 0),
              "CAP_PROP_AUTO_EXPOSURE": (cv2.CAP_PROP_AUTO_EXPOSURE, EXPOSURE_MANUAL.get(backend, 1))}
    for name, (prop, value) in wanted.items():
        result["requested"][name] = {"value": value, "accepted": bool(cap.set(prop, value))}
        result["readback"][name] = cap.get(prop)

    base = cap.get(cv2.CAP_PROP_EXPOSURE)
    before = mean_brightness(cap, drop)
    cap.set(cv2.CAP_PROP_EXPOSURE, base - 2)
    after = mean_brightness(cap, drop)
    cap.set(cv2.CAP_PROP_EXPOSURE, base)
    result["exposure"] = {"base": base, "brightness_before": before, "brightness_after": after,
                          "responds": exposure_responds(before, after)}
    result["ok"] = result["exposure"]["responds"] and all(v["accepted"] for v in result["requested"].values())
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--backend", choices=("dshow", "msmf", "v4l2"), default="dshow" if os.name == "nt" else "v4l2")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--drop", type=int, default=5, help="촬영 전에 버릴 버퍼 프레임 수")
    ap.add_argument("--out", default=".")
    ap.add_argument("--prefix", default="shot")
    ap.add_argument("--count", type=int, default=0, help="0이면 설정 확인만 한다")
    ap.add_argument("--report", default="camera_check.json")
    args = ap.parse_args()

    api = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2}[args.backend]
    cap = cv2.VideoCapture(args.camera, api)
    if not cap.isOpened():
        print(f"카메라 {args.camera}을(를) 열지 못했다", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    result = check_camera(cap, args.backend, args.drop)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    if not result["ok"]:
        print(f"경고: 고정 설정이 확인되지 않았다. {args.report}를 볼 것", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    for i in range(args.count):
        for _ in range(args.drop):
            cap.grab()
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했다", file=sys.stderr)
            return 1
        name = os.path.join(args.out, f"{args.prefix}_{int(time.time() * 1000)}_{i:03d}.png")
        if not cv2.imwrite(name, frame):
            print(f"{name} 저장 실패", file=sys.stderr)
            return 1
        print(name, f"평균 밝기 {np.mean(frame):.1f}")
    cap.release()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
