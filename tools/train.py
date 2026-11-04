"""분류기 학습 (§7.4).

    python tools/train.py --images data/images --lib foodcv.dll --roi config/roi.cfg --out model.fcv

data/images/metadata.csv 열: file, class_id, item_id, angle, lighting, captured_at
전처리는 Python에 다시 구현하지 않고 fcv_preprocess를 ctypes로 부른다 (§5.4).
"""
import argparse
import copy
import csv
import hashlib
import json
import os

import cv2
import numpy as np
import torch
from torch import nn

import export
import foodcv

SPLIT = (6, 2, 2)                # 학습:검증:평가, 개체 기준 (§7.4)
BATCH, LR, PATIENCE, MAX_EPOCHS = 32, 1e-3, 10, 500
TAU_QUANTILE = 0.05              # §7.4 τ = 검증셋 p_max 하위 5%


class FoodNet(nn.Module):
    """§6.1 구조. 합성곱 뒤 BN."""

    def __init__(self, n_classes=len(export.CLASSES)):
        super().__init__()
        chans = export.CHANNELS
        self.convs = nn.ModuleList(nn.Conv2d(a, b, 3, padding=1) for a, b in zip(chans, chans[1:]))
        self.bns = nn.ModuleList(nn.BatchNorm2d(b) for b in chans[1:])
        self.fc = nn.Linear(chans[-1], n_classes)

    @property
    def stages(self):
        return list(zip(self.convs, self.bns))

    def forward(self, x):
        for conv, bn in self.stages:
            x = nn.functional.max_pool2d(torch.relu(bn(conv(x))), 2)
        return self.fc(x.mean(dim=(2, 3)))


def split_items(rows):
    """개체 단위로 나눈다. 같은 개체의 사진이 학습과 평가에 같이 들어가면 안 된다 (§7.4)."""
    parts = {"train": [], "val": [], "test": []}
    by_class = {}
    for r in rows:
        by_class.setdefault(r["class_id"], set()).add(r["item_id"])
    assign = {}
    for cls, items in by_class.items():
        order = sorted(items, key=lambda s: hashlib.sha1(s.encode()).hexdigest())
        n_val = max(1, round(len(order) * SPLIT[1] / sum(SPLIT)))
        n_test = max(1, round(len(order) * SPLIT[2] / sum(SPLIT)))
        if len(order) < 3:
            raise ValueError(f"{cls}: 개체가 {len(order)}개뿐이다")
        for i, item in enumerate(order):
            assign[cls, item] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    for r in rows:
        parts[assign[r["class_id"], r["item_id"]]].append(r)
    return parts


def augment(img, rng):
    """§7.4 원본 해상도에서 증강한다. 크기 변화 금지. 빈 모서리는 원본 배경으로 채운다."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(0, 360), 1.0)
    m[:, 2] += rng.uniform(-15, 15, 2)
    out = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.5:
        out = out[:, ::-1]
    gain = rng.uniform(0.85, 1.15)
    contrast = rng.uniform(0.85, 1.15)
    tint = 1 + rng.uniform(-0.03, 0.03, 3)
    f = (out.astype(np.float32) - 127.5) * contrast + 127.5
    return np.clip(f * gain * tint, 0, 255).astype(np.uint8)


class Loader:
    def __init__(self, lib, folder, roi):
        self.lib, self.folder, self.roi, self.cache = lib, folder, roi, {}

    def image(self, row):
        name = row["file"]
        if name not in self.cache:
            img = cv2.imread(os.path.join(self.folder, name), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(name)
            self.cache[name] = img
        return self.cache[name]

    def tensor(self, row, rng=None):
        img = self.image(row)
        return foodcv.preprocess(self.lib, augment(img, rng) if rng is not None else img, self.roi)


def batches(rows, loader, mu, sigma, rng=None, shuffle=False):
    order = rng.permutation(len(rows)) if shuffle else np.arange(len(rows))
    labels = {c: i for i, c in enumerate(export.CLASSES)}
    for i in range(0, len(rows), BATCH):
        part = [rows[j] for j in order[i:i + BATCH]]
        x = np.stack([loader.tensor(r, rng if shuffle else None) for r in part])
        x = (x - mu[None, :, None, None]) / sigma[None, :, None, None]
        y = np.array([labels[r["class_id"]] for r in part])
        yield torch.from_numpy(x), torch.from_numpy(y)


def evaluate(model, rows, loader, mu, sigma):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in batches(rows, loader, mu, sigma):
            probs.append(torch.softmax(model(x), 1).numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train(rows, loader, seed=0, max_epochs=MAX_EPOCHS):
    parts = split_items(rows)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # μ, σ는 전처리된 학습셋에서 계산한다 (§5.4)
    stack = np.stack([loader.tensor(r) for r in parts["train"]])
    mu, sigma = stack.mean(axis=(0, 2, 3)), stack.std(axis=(0, 2, 3))
    del stack

    model = FoodNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    best, best_loss, stale = None, float("inf"), 0
    for epoch in range(max_epochs):
        model.train()
        for x, y in batches(parts["train"], loader, mu, sigma, rng, shuffle=True):
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        probs, labels = evaluate(model, parts["val"], loader, mu, sigma)
        val_loss = float(-np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean())
        print(f"epoch {epoch}: 검증 손실 {val_loss:.4f}")
        if val_loss < best_loss:
            best, best_loss, stale = copy.deepcopy(model.state_dict()), val_loss, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best)
    probs, _ = evaluate(model, parts["val"], loader, mu, sigma)
    tau = float(np.quantile(probs.max(axis=1), TAU_QUANTILE))
    return model, mu, sigma, tau, parts


def report(model, parts, loader, mu, sigma, tau, unknown_rows, model_path):
    k = len(export.CLASSES)
    probs, labels = evaluate(model, parts["test"], loader, mu, sigma)
    pred = probs.argmax(axis=1)
    confusion = np.zeros((k, k), int)
    np.add.at(confusion, (labels, pred), 1)
    recall = confusion.diagonal() / np.maximum(confusion.sum(axis=1), 1)
    out = {"classes": list(export.CLASSES), "confusion": confusion.tolist(), "recall": recall.tolist(), "tau": tau,
           "val_items_per_class": {c: len({r["item_id"] for r in parts["val"] if r["class_id"] == c})
                                   for c in export.CLASSES},
           "unknown_reject_rate": None}
    if unknown_rows:
        x = np.stack([(loader.tensor(r) - mu[:, None, None]) / sigma[:, None, None] for r in unknown_rows])
        model.eval()
        with torch.no_grad():
            pmax = torch.softmax(model(torch.from_numpy(x)), 1).max(dim=1).values.numpy()
        out["unknown_reject_rate"] = float((pmax < tau).mean())
    with open(model_path, "rb") as f:
        out["model_hash"] = int.from_bytes(f.read()[-4:], "little")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--out", default="model.fcv")
    ap.add_argument("--report", default="train_report.json")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = ap.parse_args()

    lib = foodcv.load_library(os.path.abspath(args.lib))
    foodcv._check(lib, lib.fcv_init(None))         # §5.4 전처리 전용 상태
    with open(os.path.join(args.images, "metadata.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    known = [r for r in rows if r["class_id"] in export.CLASSES]
    unknown = [r for r in rows if r["class_id"] == "unknown"]     # 미등록 음식 (§7.4)
    other = {r["class_id"] for r in rows} - set(export.CLASSES) - {"unknown"}
    if other:
        raise ValueError(f"모르는 class_id: {sorted(other)}")
    loader = Loader(lib, args.images, foodcv.read_roi_cfg(args.roi))
    model, mu, sigma, tau, parts = train(known, loader, max_epochs=args.max_epochs)
    export.export_model(model, args.out, mu, sigma, tau)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report(model, parts, loader, mu, sigma, tau, unknown, args.out), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
