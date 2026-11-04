"""NumPy float64 참조 구현으로 골든 데이터를 만든다 (§6.6).

BN을 접지 않은 모델로, ReLU 후 풀링 순서로 계산한다. C 구현과 겹치는 코드가 없어야 검증이 된다.

    python tools/make_golden.py        tests/golden/ 에 쓴다
"""
import os

import numpy as np

import export

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "golden")
N_INPUTS = 20
EPS = 1e-5


def random_model(rng):
    stages = []
    for cin, cout in zip(export.CHANNELS, export.CHANNELS[1:]):
        stages.append(dict(
            w=rng.normal(0, np.sqrt(2 / (cin * 9)), (cout, cin, 3, 3)), b=rng.normal(0, 0.05, cout),
            gamma=rng.uniform(0.5, 1.5, cout), beta=rng.normal(0, 0.1, cout),
            mean=rng.normal(0, 0.1, cout), var=rng.uniform(0.5, 1.5, cout)))
    return stages, rng.normal(0, 0.25, (len(export.CLASSES), 64)), rng.normal(0, 0.1, len(export.CLASSES))


def conv3x3(x, w, b):
    c, h, wd = x.shape
    xp = np.zeros((c, h + 2, wd + 2))
    xp[:, 1:-1, 1:-1] = x
    win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(1, 2))
    return np.einsum("cijuv,ocuv->oij", win, w) + b[:, None, None]


def forward(x, stages, fc_w, fc_b):
    for s in stages:
        a = conv3x3(x, s["w"], s["b"])
        a = (a - s["mean"][:, None, None]) / np.sqrt(s["var"] + EPS)[:, None, None]
        a = a * s["gamma"][:, None, None] + s["beta"][:, None, None]
        a = np.maximum(a, 0)                                       # ReLU 먼저
        c, h, w = a.shape
        x = a.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))    # 그다음 풀링
    return fc_w @ x.mean(axis=(1, 2)) + fc_b


def main():
    rng = np.random.default_rng(20261104)
    stages, fc_w, fc_b = random_model(rng)
    mu, sigma, tau = (0.45, 0.42, 0.40), (0.25, 0.24, 0.26), 0.5
    os.makedirs(OUT, exist_ok=True)
    folded = [export.fold_bn(s["w"], s["b"], s["gamma"], s["beta"], s["mean"], s["var"], EPS) for s in stages]
    export.write_fcv(os.path.join(OUT, "model.fcv"), export.CLASSES, mu, sigma, tau, folded, fc_w, fc_b)

    images = rng.integers(0, 256, (N_INPUTS, 96, 96, 3), dtype=np.uint8)
    logits = np.empty((N_INPUTS, len(export.CLASSES)))
    m = np.array(mu, np.float64)[:, None, None]
    d = np.array(sigma, np.float64)[:, None, None]
    for i, img in enumerate(images):
        x = (img.transpose(2, 0, 1).astype(np.float64) / 255 - m) / d
        logits[i] = forward(x, stages, fc_w, fc_b)
    images.tofile(os.path.join(OUT, "images.bin"))                 # RGB HWC uint8
    logits.astype("<f8").tofile(os.path.join(OUT, "logits.bin"))
    print("logit 범위", logits.min(), logits.max())


if __name__ == "__main__":
    main()
