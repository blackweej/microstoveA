"""§13.2 T-A1~A12: 합성 영상으로 면적 모듈 회귀를 감시한다. 접시를 안 적으면 파랑."""
import time

import numpy as np
import pytest

import foodcv as fc
import sim_area as sim

BIAS_48 = dict(pizza=4.2, chicken=5.9, yangnyeom=6.6, rice=4.5, dumpling=11.2)   # §4.8 파란 접시 %
ANGLES = np.linspace(0, 2 * np.pi, 6, endpoint=False) + 0.3


def bgr(img):
    return img[..., ::-1]


def rng(seed):
    return np.random.default_rng(seed)


@pytest.fixture(scope="module")
def empty_blue():
    return sim.render("blue", ref_ang=0.0, rng=rng(1))[0]


@pytest.fixture
def meter(lib, empty_blue):
    m = fc.AreaMeter(lib, sim.area_config())
    assert m.calibrate(bgr(empty_blue)) == fc.AREA_OK
    return m


def test_a1_ref_count(meter):
    true = np.pi * (2 * sim.K) ** 2
    errs = []
    for i, a in enumerate(np.linspace(0, 2 * np.pi, 12, endpoint=False)):
        r = meter.measure(bgr(sim.render("blue", ref_ang=a, rng=rng(40 + i))[0]))
        assert r.status == fc.AREA_EMPTY
        errs.append(abs(r.ref_px / true - 1))
    assert max(errs) <= 0.005


@pytest.mark.parametrize("occl", [0.05, 0.10])
def test_a2_occlusion(meter, occl):
    for i, a in enumerate(ANGLES):
        r = meter.measure(bgr(sim.render("blue", ref_ang=a, occl=occl, rng=rng(10 + i))[0]))
        assert r.status == fc.AREA_REF_CHANGED, (occl, i, r.status_name)


def test_a3_touching_red_food(meter):
    for i, a in enumerate(ANGLES[:4]):
        r = meter.measure(bgr(sim.render("blue", ref_ang=a, food_touch_ref=True, rng=rng(20 + i))[0]))
        assert r.status == fc.AREA_REF_MERGED, (i, r.status_name)


@pytest.mark.parametrize("food", sim.FOODS)
def test_a4_a5_foods(meter, food):
    """T-A4: 전부 OK, 같은 음식 반복 표준편차 ≤ 1%. T-A5: 평균 치우침이 §4.8 ±1.5%p."""
    bias = []
    for i in range(10):
        a = (i * 0.7) % (2 * np.pi)
        img, true = sim.render("blue", ref_ang=a, food=food, rng=rng(100 + i))
        r = meter.measure(bgr(img))
        assert r.status == fc.AREA_OK, (food, i, r.status_name)
        bias.append((r.food_cm2 / true - 1) * 100)
    assert abs(np.mean(bias) - BIAS_48[food]) <= 1.5, np.mean(bias)

    areas = []
    for i in range(10):
        img, _ = sim.render("blue", ref_ang=1.0, food=food, rng=rng(7), noise_rng=rng(200 + i))
        r = meter.measure(bgr(img))
        assert r.status == fc.AREA_OK
        areas.append(r.food_cm2)
    assert np.std(areas) / np.mean(areas) <= 0.01


def test_a6_white_on_white(lib):
    m = fc.AreaMeter(lib, sim.area_config())
    assert m.calibrate(bgr(sim.render("white", rng=rng(1))[0])) == fc.AREA_OK
    for i in range(10):
        img, _ = sim.render("white", ref_ang=i * 0.6, food="dumpling", rng=rng(60 + i))
        assert m.measure(bgr(img)).status == fc.AREA_EMPTY


def test_a7_empty_and_no_ref(meter):
    assert meter.measure(bgr(sim.render("blue", ref_ang=1.0, rng=rng(6))[0])).status == fc.AREA_EMPTY
    img = sim.render("blue", ref_ang=1.0, rng=rng(8))[0].copy()
    cx, cy = sim.ref_center(4.0, 1.0)
    gy, gx = np.mgrid[0:sim.H, 0:sim.W]
    img[(gx - cx) ** 2 + (gy - cy) ** 2 <= 40 ** 2] = (56, 98, 155)
    assert meter.measure(bgr(img)).status == fc.AREA_REF_MISSING


def test_a8_speed(meter):
    img = np.ascontiguousarray(bgr(sim.render("blue", ref_ang=1.0, food="chicken", rng=rng(9))[0]))
    meter.measure(img)
    t0 = time.perf_counter()
    for _ in range(10):
        meter.measure(img)
    assert (time.perf_counter() - t0) / 10 <= 0.050


def test_a9_lighting(meter):
    img = sim.render("blue", ref_ang=1.0, food="pizza", rng=rng(5))[0]
    scaled = lambda k: bgr(np.clip(img * k, 0, 255).astype(np.uint8))
    for k in (1.15, 1.2):
        assert meter.measure(scaled(k)).status == fc.AREA_LIGHT_CHANGED, k
    for k in (0.9, 1.05):
        assert meter.measure(scaled(k)).status == fc.AREA_OK, k


@pytest.mark.parametrize("food", [None, "chicken"])
def test_a10_plate_moved(meter, food):
    img = sim.render("blue", ref_ang=1.0, food=food, rng=rng(7))[0]
    for dx, dy in ((16, 0), (-16, 0), (0, 16), (0, -16)):
        r = meter.measure(bgr(np.roll(np.roll(img, dx, 1), dy, 0)))
        assert r.status in (fc.AREA_PLATE_MOVED, fc.AREA_REF_CHANGED), (dx, dy, r.status_name)


@pytest.mark.parametrize("food", sim.FOODS)
def test_a11_calibrate_with_food(lib, food):
    m = fc.AreaMeter(lib, sim.area_config())
    img = sim.render("blue", ref_ang=1.0, food=food, rng=rng(100))[0]
    assert m.calibrate(bgr(img)) == fc.AREA_NOT_EMPTY
    with pytest.raises(fc.FoodCVError):
        m.measure(bgr(img))


def test_a12_big_red_food_merged(meter):
    img = sim.render("blue", ref_ang=1.0, food_touch_ref=True, touch_scale=3.0, rng=rng(3))[0]
    r = meter.measure(bgr(img))
    assert r.status == fc.AREA_REF_MERGED and r.ref_px == 0, (r.status_name, r.ref_px)


def test_cfg_rejects_unset_values(tmp_path):
    p = tmp_path / "area.cfg"
    p.write_text("ref_diameter_cm = ?\nred_k = 1.8\n", encoding="utf-8")
    with pytest.raises(ValueError):
        fc.read_area_cfg(str(p))
