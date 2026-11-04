# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

`foodcv` is Part A of a microwave time-setter project: food classification, food area measurement, and display reading. It takes pixels and returns numbers. Camera capture, IR temperature, time calculation, the dial motor loop, and UI belong to Parts B and C, and A never computes mass.

**Most of this is not built yet.** The repo holds only:

- `A_PART_SPEC.md`: the implementation contract, written in Korean. It is authoritative, so read the relevant section before implementing anything. Section references below (§N) point to it.
- `reference/area.c`, `reference/sim.py`: the validated prototype. **Do not modify `reference/`.**

The layout in §2 (`include/`, `src/`, `tests/`, `tools/`, `config/`, `Makefile`, `CHANGELOG.md`) is the target. Work proceeds through milestones M0–M6 in §14, with `make test` passing and a commit at the end of each. The directory is not a git repository yet.

Deadlines: DLL handoff to Part C on 2026-11-13 (M0–M4 and M6 required), final on 2026-12-04.

## Commands

These are defined by the spec (§2, §13). The Makefile does not exist until M0.

```sh
make libfoodcv.so     # Linux dev build
make foodcv.dll       # Windows release build (x86_64-w64-mingw32-gcc; 64-bit to match Python)
make test             # C unit tests: builds run_tests from tests/*.c + src/*.c
make debug            # -O0 -g -DFCV_DEBUG_DUMP -fsanitize=address,undefined
pytest tools/tests    # Python integration tests on synthetic images
```

The bar is zero warnings under `-std=c11 -O3 -Wall -Wextra -Wpedantic`. Never use `-march=native`.

This Windows machine currently has no `gcc`, `make`, MinGW, `pytest`, or `opencv-python`. It has Python 3.12 (the spec says 3.11) with numpy and scipy. C code cannot be built or tested here until a toolchain is installed.

## Hard rules (§0)

- `src/` uses only the C11 standard library (`stdint.h`, `stdlib.h`, `string.h`, `math.h`, `stdio.h`). No OpenCV, BLAS, or threads.
- Allocate only in `*_init`. `fcv_classify`, `area_measure`, `seg_read`, and `seg_vote` must not call `malloc` (T-C4 checks this for `fcv_classify`).
- Public functions return an `int` status (0 = success, negative = error) and give results through output arguments. Exceptions: `*_free` and name lookups.
- Input pixels are caller-owned and pointers are never retained. Assume single-threaded calls.
- Use `FCV_PI`, not `M_PI`.
- Changing `include/foodcv.h` requires bumping `FCV_API_VERSION` and adding a `CHANGELOG.md` entry, because Part C codes against that header. The wrapper asserts the version in its constructor.
- **Never invent unknown values.** Undecided items (§10) come from config files with no defaults in code. `?` values in `area.cfg` must make the wrapper raise.
- Code implementing a formula gets a comment with the spec section and formula, e.g. `/* §4.2 s = A_ref / N_ref */`.

## Architecture

```
measurement webcam ─photo─▶ fcv_classify ─class k─▶ Part C: profile lookup
                     └─same photo─▶ area_measure ─▶ (only if uses_area) Part C: time calc
display webcam ─frames─▶ seg_read ×5 ─▶ seg_vote ─seconds─▶ Part C: dial loop
```

There are three independent C subsystems behind one header. `tools/foodcv.py` is the ctypes wrapper Part C imports. Every function needs `argtypes`/`restype`, or 64-bit Windows truncates pointers.

**Area (`src/area.c`, §4).** A red filled circle on a matte blue plate is both the scale reference and the occlusion check. `area_calibrate` takes one empty-plate photo and stores the circle's pixel count, plate chromaticity, and an illumination map. `area_measure` then counts non-plate pixels as food.
- Port from `reference/area.c` with unified error codes, and change only what §4.9 lists. Any other behaviour change needs the `tests/test_area.c` criteria first.
- The port must also add what the prototype lacks: reject `GRAY8` with `FCV_E_ARG` (§3), and move the redness calculation into `static inline float marker_score(int R, int G, int B)` (§10).
- The prototype's occlusion numbers came from a white plate (`render()` defaults to white), but the operating plate is blue. On blue, ρ alone marks brown food as red, which is why §4.3-1 adds `R ≥ red_k·G`. Do not remove it.
- The check order in §4.3 is load-bearing. Shape (F) alone misses occlusion: a straight 20% cover still gives F = 0.979. The pixel-count check catches that, so do not reorder them.
- The light and plate-move checks (§4.3-5a, 5b) exist because both failures leave the reference circle intact, so without them a wrong area comes back as `AREA_OK`.
- Do not correct the +4–11% area bias (§4.8). It is constant per class and Part C's time coefficients absorb it. `perim_px` is returned for a possible later correction.
- `area_measure` returns 0 once computation finishes, and the verdict goes in `out->status`. `AREA_*` verdicts are positive, so they are not errors.

**Classifier (`src/preprocess.c`, `cnn.c`, `weights.c`, §5–7).** A 4-stage CNN (3→16→32→64→64, then GAP and FC to 7 classes, 60,967 parameters).
- Downscaling is area averaging, never bilinear. ROI is square with a side that is a multiple of 96.
- Each stage fuses conv → max-pool → ReLU per output channel, without storing full-resolution results. The result must be exactly equal to ReLU-then-pool.
- BatchNorm exists only in training and is folded into the weights at export (§7.2), so C has no BN.
- Training calls `fcv_preprocess` through ctypes. Never reimplement preprocessing in Python. `fcv_init(NULL)` gives a preprocess-only mode (μ=0, σ=1, and `fcv_classify` returns `FCV_E_STATE`).
- Class order in §7.3 is fixed. Part C refuses to run if names mismatch its profiles, which is a safety check.
- `model.fcv` is an exact binary format with an FNV-1a hash (§7.1). Numerical correctness is checked against float64 NumPy golden data from `tools/make_golden.py` (§6.6).
- No extra optimization such as unrolling or threading (§6.7).

**Display reader (`src/seg.c`, §8).** Homography from four corners (8×8 Gaussian elimination, no matrix inverse), per-segment brightness sampling, per-frame Otsu, exact 7-segment table lookup, then a 5-frame vote. Because the code's minimum Hamming distance is 1, the vote's validity checks (seconds tens digit 0–5, value on the dial `grid`) are the real defence against a single bad segment. `seg_init` must fail if `grid` is empty. Multiplexed displays need frame fusing before reading, which the wrapper does (`fuse` in `display.layout`, §11).

**Config boundary.** The C library reads `display.layout` (via `seg_init`) and `model.fcv`. `tools/foodcv.py` reads `area.cfg` to fill `area_params`, reads `display.quad` for `seg_set_quad`, and calls `area_init` with the first calibration frame's size.

## Testing notes

Everything is developed without hardware, using image files and synthetic generators (`tools/sim_area.py`, ported from `reference/sim.py`, and `tools/sim_seg.py`). Test IDs and pass criteria are in §13. §4.8 holds the prototype numbers used as the regression baseline, and §4.9 holds the evidence for each change from the prototype. Some tests pin known failures on purpose: T-A6 requires white dumplings on a white plate to stay `AREA_EMPTY`.

T-A tests default to the blue plate and keep food at least 1 cm from the reference circle, so `tools/sim_area.py` must differ from `reference/sim.py` on both points (§13.2). `reference/sim.py` loads `libarea.so` by default (override with the `AREA_LIB` env var, resolved relative to that file).

## Conventions

The spec and the reference code comments are in Korean. Keep the Korean terminology and comment style when porting.
