"""GUI·카메라 없이 돌릴 수 있는 도구 계산과 train.py 연기 시험."""
import csv
import os

import numpy as np
import pytest

import calib_tool
import eval_area
import foodcv as fc
import sim_area


def test_circle_from_points():
    cx, cy, r = 320.0, 240.0, 192.0
    pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in (0.1, 2.0, 4.0)]
    got = calib_tool.circle_from_points(*pts)
    assert np.allclose(got, (cx, cy, r))
    with pytest.raises(ValueError):
        calib_tool.circle_from_points((0, 0), (1, 1), (2, 2))


def test_snap_roi():
    assert calib_tool.snap_roi((80, 0), (560, 480), 640, 480) == (80, 0, 480)
    x, y, size = calib_tool.snap_roi((100, 50), (310, 240), 640, 480)
    assert size % 96 == 0 and size == 192
    x, y, size = calib_tool.snap_roi((0, 0), (900, 900), 640, 480)      # 영상 안으로 잘린다
    assert (size, x + size <= 640, y + size <= 480) == (480, True, True)


def test_update_area_cfg(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(root, "config", "area.cfg"), encoding="utf-8") as f:
        text = calib_tool.update_area_cfg(f.read(), 320.04, 240.0, 192.0)
    assert "plate_r = 186.0" in text and "plate_cx = 320.0" in text
    p = tmp_path / "area.cfg"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="ref_diameter_cm"):            # 지름은 아직 ?다
        fc.read_area_cfg(str(p))
    p.write_text(text.replace("ref_diameter_cm = ?", "ref_diameter_cm = 4.0"), encoding="utf-8")
    assert fc.read_area_cfg(str(p))["plate_r"] == 186.0


def test_stroke_lines():
    lines = calib_tool.stroke_lines([(10, 8, 2, 4)] * 7)
    assert lines[0] == "seg 0 a 2 4 10 8" and lines[6].startswith("seg 0 g")
    with pytest.raises(ValueError):
        calib_tool.stroke_lines([(0, 0, 1, 1)] * 6)


def test_eval_area_on_synthetic_truth(lib):
    m = fc.AreaMeter(lib, sim_area.area_config())
    rng = np.random.default_rng(1)
    assert m.calibrate(sim_area.render("blue", ref_ang=1.0, rng=rng)[0][..., ::-1]) == fc.AREA_OK
    frame = sim_area.render("blue", ref_ang=1.0, food="chicken", rng=rng)[0][..., ::-1]
    r = m.measure(frame)
    truth = m.debug_labels().copy()
    assert eval_area.compare(truth, truth) == (1.0, 1.0)
    empty = np.zeros_like(truth)
    assert eval_area.compare(empty, truth)[0] == 0.0
    r2, iou, ratio = eval_area.evaluate(m, frame, truth)
    assert r2.status == r.status == fc.AREA_OK and iou == 1.0 and ratio == 1.0


def test_train_smoke(lib, tmp_path):
    """합성 사진으로 학습→내보내기→C 로드까지 지나가는지만 본다. 정확도 시험이 아니다."""
    pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    import export
    import train

    fc._check(lib, lib.fcv_init(None))
    rows = []
    colours = np.linspace(40, 220, len(export.CLASSES))
    for k, name in enumerate(export.CLASSES):
        for item in range(3):
            file = f"{name}_{item}.png"
            img = np.full((96, 96, 3), colours[k], np.uint8)
            cv2.imwrite(str(tmp_path / file), img)
            rows.append(dict(file=file, class_id=name, item_id=f"{name}{item}", angle="0", lighting="a",
                             captured_at=""))
    with open(tmp_path / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    parts = train.split_items(rows)
    items = {p: {r["item_id"] for r in parts[p]} for p in parts}
    assert not (items["train"] & items["val"]) and not (items["train"] & items["test"])

    loader = train.Loader(lib, str(tmp_path), (0, 0, 96, 96))
    model, mu, sigma, tau, parts = train.train(rows, loader, max_epochs=1)
    assert mu.shape == (3,) and (sigma >= 0).all() and 0 < tau <= 1
    path = str(tmp_path / "m.fcv")
    export.export_model(model, path, mu, np.maximum(sigma, 1e-3), tau)
    report = train.report(model, parts, loader, mu, np.maximum(sigma, 1e-3), tau, [], path)
    assert len(report["confusion"]) == len(export.CLASSES)
    assert fc._check(lib, lib.fcv_init(path.encode())) == 0
    assert lib.fcv_model_hash() == report["model_hash"]
    lib.fcv_free()
