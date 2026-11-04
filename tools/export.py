"""model.fcv 쓰기와 BN 접기 (§7)."""
import struct

import numpy as np

FILE_VERSION = 2
INPUT_SIZE = 96
CHANNELS = (3, 16, 32, 64, 64)
CLASSES = ("samgak_gimbap", "dosirak", "hotbar", "chicken", "pizza", "dumpling", "fried_rice")   # §7.3


def fnv1a(data):
    h = 2166136261
    for byte in data:
        h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
    return h


def fold_bn(weight, bias, gamma, beta, mean, var, eps):
    """§7.2  s_o = γ/√(σ²+ε),  W'_o = s_o W_o,  b'_o = s_o(b_o − μ_o) + β_o"""
    s = gamma / np.sqrt(var + eps)
    return weight * s[:, None, None, None], (bias - mean) * s + beta


def write_fcv(path, classes, mu, sigma, tau, stages, fc_w, fc_b):
    """stages: 단별 (W′, b′). §7.1 형식으로 쓴다."""
    out = bytearray(b"FCV1")
    out += struct.pack("<I", FILE_VERSION)
    out += struct.pack("<3I", len(classes), INPUT_SIZE, len(stages))
    out += struct.pack("<5I", *CHANNELS)
    out += np.asarray(mu, "<f4").tobytes() + np.asarray(sigma, "<f4").tobytes()
    out += struct.pack("<f", tau)
    for name in classes:
        raw = name.encode("utf-8")
        if len(raw) >= 32:
            raise ValueError(f"클래스 이름이 너무 길다: {name}")
        out += raw.ljust(32, b"\0")
    for w, b in stages:
        out += np.ascontiguousarray(w).astype("<f4").tobytes()
        out += np.ascontiguousarray(b).astype("<f4").tobytes()
    out += np.ascontiguousarray(fc_w).astype("<f4").tobytes()
    out += np.ascontiguousarray(fc_b).astype("<f4").tobytes()
    out += struct.pack("<I", fnv1a(out))
    with open(path, "wb") as f:
        f.write(out)


def export_model(model, path, mu, sigma, tau, classes=CLASSES):
    """학습한 모델을 BN을 접어 내보낸다. model.stages = [(conv, bn), ...], model.fc = Linear."""
    model.eval()                                             # 반드시 eval
    stages = []
    for conv, bn in model.stages:
        as_np = lambda t: t.detach().double().numpy()
        bias = as_np(conv.bias) if conv.bias is not None else np.zeros(conv.out_channels)
        stages.append(fold_bn(as_np(conv.weight), bias, as_np(bn.weight), as_np(bn.bias),
                              as_np(bn.running_mean), as_np(bn.running_var), bn.eps))
    write_fcv(path, classes, mu, sigma, tau, stages,
              model.fc.weight.detach().numpy(), model.fc.bias.detach().numpy())
