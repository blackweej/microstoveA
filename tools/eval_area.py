"""손 라벨 마스크와 면적 결과 비교 (§12, 실험 X6).

    python tools/eval_area.py --lib foodcv.dll --cfg config/area.cfg --empty empty.png \
        --photos data/x6 --masks data/x6_masks

사진과 같은 이름의 마스크 PNG(255 = 음식)를 찾아 IoU와 면적 비를 표로 낸다.
"""
import argparse
import os
import sys

import numpy as np

import foodcv as fc


def compare(pred, truth):
    """음식 마스크 둘의 (IoU, 면적 비 예측/정답)."""
    pred, truth = pred > 0, truth > 0
    union = (pred | truth).sum()
    iou = (pred & truth).sum() / union if union else 1.0
    ratio = pred.sum() / truth.sum() if truth.any() else float("inf")
    return float(iou), float(ratio)


def evaluate(meter, frame, truth):
    r = meter.measure(frame)
    if r.status != fc.AREA_OK:
        return r, None, None
    return (r, *compare(meter.debug_labels(), truth))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--empty", required=True)
    ap.add_argument("--photos", required=True)
    ap.add_argument("--masks", required=True)
    args = ap.parse_args()

    import cv2
    lib = fc.load_library(os.path.abspath(args.lib))
    meter = fc.AreaMeter(lib, fc.read_area_cfg(args.cfg))
    rc = meter.calibrate(cv2.imread(args.empty, cv2.IMREAD_COLOR))
    if rc != fc.AREA_OK:
        print(f"보정 실패: {fc.AREA_STATUS[rc]}", file=sys.stderr)
        return 1

    print(f"{'파일':30s} {'상태':14s} {'면적 cm²':>9s} {'IoU':>6s} {'면적비':>6s}")
    ious = []
    for name in sorted(os.listdir(args.photos)):
        mask_path = os.path.join(args.masks, name)
        if not name.lower().endswith(".png") or not os.path.exists(mask_path):
            continue
        frame = cv2.imread(os.path.join(args.photos, name), cv2.IMREAD_COLOR)
        truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        r, iou, ratio = evaluate(meter, frame, truth)
        if iou is None:
            print(f"{name:30s} {r.status_name:14s}")
            continue
        ious.append(iou)
        print(f"{name:30s} {r.status_name:14s} {r.food_cm2:9.1f} {iou:6.3f} {ratio:6.3f}")
    if ious:
        print(f"평균 IoU {np.mean(ious):.3f}, 최소 {np.min(ious):.3f}, {len(ious)}장")
    return 0


if __name__ == "__main__":
    sys.exit(main())
