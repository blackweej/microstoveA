"""§13.2 T-D1, T-D2, T-E2E와 내보내기 일치 시험."""
import os

import numpy as np
import pytest

import export
import foodcv as fc
import sim_area
import sim_seg as sim

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRID = list(range(0, 361, 10))


def write_config(folder, fuse=1, model=None):
    cfg = sim_area.area_config()
    area = folder / "area.cfg"
    area.write_text("\n".join(f"{k} = {v}" for k, v in cfg.items()) + "\n", encoding="utf-8")
    layout = folder / "display.layout"
    layout.write_text(sim.layout_text(GRID, fuse=fuse), encoding="utf-8")
    quad = folder / "display.quad"
    quad.write_text("50 90 270 90 270 156 50 156\n", encoding="utf-8")
    roi = folder / "roi.cfg"
    roi.write_text("80 0 480\n", encoding="utf-8")
    return (model or os.path.join(ROOT, "tests", "golden", "model.fcv"), str(area), str(layout), str(quad), str(roi))


@pytest.fixture
def make_cv(lib, tmp_path):
    path = lib._name
    made = []

    def make(fuse=1, model=None):
        cv = fc.FoodCV(path, *write_config(tmp_path, fuse, model))
        made.append(cv)
        return cv
    yield make
    for cv in made:
        cv.close()


def test_d1_all_grid_values(make_cv):
    cv = make_cv()
    rng = np.random.default_rng(1)
    trials = wrong = unstable = 0
    for seconds in GRID:
        for _ in range(200):
            quad = sim.random_quad(rng)
            cv.set_quad(quad)
            clean = sim.clean_frame(seconds, quad, gain=rng.uniform(0.85, 1.1))
            status, value = cv.read_display([sim.add_noise(clean, rng) for _ in range(5)])
            trials += 1
            wrong += status == fc.SEG_STABLE and value != seconds
            unstable += status != fc.SEG_STABLE
    print("T-D1", trials, "wrong", wrong, "unstable", unstable)
    assert wrong == 0
    assert unstable / trials < 0.05


def test_d2_multiplex(make_cv):
    rng = np.random.default_rng(2)
    dark = np.mean([sim.multiplex_levels(rng) == 0 for _ in range(20000)])
    assert abs(dark - 0.25) <= 0.05

    cv = make_cv(fuse=5)
    quad = sim.random_quad(rng)
    cv.set_quad(quad)
    stable = 0
    trials = 40
    for i in range(trials):
        seconds = GRID[1 + i % (len(GRID) - 1)]
        frames = [sim.frame(seconds, quad, rng, levels=sim.multiplex_levels(rng)) for _ in range(25)]
        status, value = cv.read_display(frames)
        if status == fc.SEG_STABLE:
            assert value == seconds
            stable += 1
    print("T-D2 fused stable", stable, "/", trials)
    assert stable >= 0.9 * trials


def test_read_display_needs_exact_frames(make_cv):
    cv = make_cv(fuse=5)
    with pytest.raises(ValueError):
        cv.read_display([np.zeros((240, 320, 3), np.uint8)] * 5)


def test_e2e(make_cv):
    cv = make_cv()
    rng = np.random.default_rng(3)
    empty = sim_area.render("blue", ref_ang=1.0, rng=rng)[0][..., ::-1]
    food = sim_area.render("blue", ref_ang=1.0, food="pizza", rng=rng)[0][..., ::-1]
    k, name, p, rejected, probs = cv.classify(food)
    assert name == export.CLASSES[k] and abs(probs.sum() - 1) < 1e-5 and probs[k] == p
    assert isinstance(rejected, bool)
    assert cv.calibrate_plate(empty) == fc.AREA_OK
    r = cv.measure_area(food)
    assert isinstance(r, fc.AreaResult) and r.status_name == "OK" and r.food_cm2 > 0
    quad = sim.random_quad(rng)
    cv.set_quad(quad)
    assert cv.read_display([sim.frame(150, quad, rng) for _ in range(5)]) == (fc.SEG_STABLE, 150)
    assert cv.grid_index(150) == 15 and cv.grid_index(155) == -1
    assert cv.model_hash() != 0


def test_export_matches_torch(lib, make_cv, tmp_path):
    """학습 코드의 모델을 내보낸 파일로 C가 계산한 확률이 torch와 같다."""
    torch = pytest.importorskip("torch")
    from train import FoodNet
    torch.manual_seed(0)
    model = FoodNet()
    for _, bn in model.stages:                  # 기본값이면 BN 접기가 시험되지 않는다
        bn.running_mean.normal_(0, 0.2)
        bn.running_var.uniform_(0.5, 2)
        bn.weight.data.uniform_(0.5, 1.5)
        bn.bias.data.normal_(0, 0.2)
    mu, sigma = np.array([0.4, 0.5, 0.6]), np.array([0.2, 0.25, 0.3])
    path = str(tmp_path / "m.fcv")
    export.export_model(model, path, mu, sigma, 0.3)
    cv = make_cv(model=path)

    rng = np.random.default_rng(4)
    frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    _, _, _, _, probs = cv.classify(frame)
    crop = frame[:, 80:560, ::-1].astype(np.float64)                         # RGB
    small = crop.reshape(96, 5, 96, 5, 3).mean(axis=(1, 3)).transpose(2, 0, 1) / 255
    x = (small - mu[:, None, None]) / sigma[:, None, None]
    model.eval()
    with torch.no_grad():
        ref = torch.softmax(model(torch.from_numpy(x[None]).float()), 1)[0].numpy()
    assert np.abs(ref - probs).max() < 1e-4


def test_classify_speed(make_cv):
    import time
    cv = make_cv()
    frame = np.zeros((480, 640, 3), np.uint8)
    cv.classify(frame)
    t0 = time.perf_counter()
    for _ in range(10):
        cv.classify(frame)
    ms = (time.perf_counter() - t0) * 100
    print(f"classify {ms:.1f} ms")
    assert ms < 200
