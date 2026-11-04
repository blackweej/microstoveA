# microstoveA

전자레인지 타임세터 프로젝트의 A파트입니다. 웹캠 사진에서 **음식 종류**를 분류하고, **음식 면적**(cm²)을 재고, **표시창의 남은 시간**을 읽습니다. 픽셀을 받아 숫자를 돌려주는 C11 라이브러리이고, 외부 라이브러리를 쓰지 않습니다.

> **현재 상태:** 구현과 시험은 합성 영상으로만 했습니다. 실제 카메라와 사진으로는 아직 검증하지 않았고, 학습된 `model.fcv`도 없습니다. `tests/golden/model.fcv`는 시험용 무작위 가중치라 분류 결과에 의미가 없습니다.

## 하는 일

```
측정대 웹캠 ──사진──▶ fcv_classify ──클래스──▶ 프로파일 조회
                       └──같은 사진──▶ area_measure ──면적──▶ 시간 계산
표시창 웹캠 ──프레임──▶ seg_read ×5 ──▶ seg_vote ──초──▶ 다이얼 제어
```

| 모듈 | 방법 |
|---|---|
| 음식 분류 | 4단 CNN(60,967 파라미터)을 C로 직접 구현. 7개 클래스 |
| 면적 측정 | 파란 접시에 그린 빨간 원 하나로 스케일을 구하고, 접시로 보이지 않는 픽셀을 음식으로 셈 |
| 표시창 판독 | 호모그래피로 펴고 획별 밝기를 Otsu로 이진화한 뒤 5프레임 투표 |

면적 측정은 틀린 값을 조용히 내지 않는 쪽으로 설계했습니다. 기준 원이 가려지거나, 조명이 바뀌거나, 접시가 밀리면 면적 대신 상태 코드를 돌려줍니다.

카메라 캡처, 온도 측정, 시간 계산, 모터 제어, UI는 이 저장소의 범위가 아닙니다.

## 빌드

`gcc`와 `make`가 있으면:

```sh
make libfoodcv.so     # Linux
make foodcv.dll       # Windows (x86_64-w64-mingw32-gcc)
make test             # C 단위 테스트
```

없으면 pip로 받은 `zig cc`로 같은 빌드를 돌릴 수 있습니다:

```sh
python -m venv .venv
.venv/Scripts/python -m pip install -r tools/requirements.txt    # Linux는 .venv/bin/python
.venv/Scripts/python tools/build.py lib      # foodcv.dll 또는 libfoodcv.so
.venv/Scripts/python tools/build.py test
```

## 시험

```sh
.venv/Scripts/python -m pytest tools/tests                    # 전체. 약 13분
.venv/Scripts/python -m pytest tools/tests/test_area.py -k a2 # 하나만
```

C 단위 검사 540개와 Python 시험 34개가 있습니다. 시험 항목과 합격 기준은 `A_PART_SPEC.md` §13에 있습니다.

지금까지 `zig cc`의 `x86_64-windows-gnu` 타깃으로만 빌드해 시험했습니다. `make`, MinGW `gcc`, Linux 빌드는 아직 실행해 보지 않았습니다.

## 사용법

```python
import cv2
from foodcv import FoodCV, AREA_OK, SEG_STABLE

cv = FoodCV("foodcv.dll", "model.fcv", "config/area.cfg",
            "config/display.layout", "config/display.quad", "config/roi.cfg")

cv.calibrate_plate(cv2.imread("empty_plate.png"))   # 시작할 때 빈 접시 1장

k, name, p, rejected, probs = cv.classify(frame)    # frame: OpenCV BGR uint8
area = cv.measure_area(frame)
if area.status == AREA_OK:
    print(area.food_cm2, area.part_cm2)
status, seconds = cv.read_display(frames)           # 5 × fuse장
```

`config/`의 값은 장치마다 다릅니다. 설치할 때 `tools/calib_tool.py`로 만듭니다. `area.cfg`에 `?`가 남아 있으면 래퍼가 예외를 냅니다.

## 학습시작

```python
  .venv/Scripts/python tools/build.py lib
  .venv/Scripts/python tools/train.py --images data/images --lib foodcv.dll --roi config/roi.cfg --out model.fcv
```

## 구조

```
include/foodcv.h   공개 API
src/               C 구현 (area, cnn, preprocess, weights, seg)
tests/             C 단위 테스트와 골든 데이터
tools/             ctypes 래퍼, 시뮬레이터, 학습·보정 도구, Python 시험
config/            설정 템플릿
reference/         시제품 원본 (수정하지 않음)
```

## 문서

- [`A_PART_SPEC.md`](A_PART_SPEC.md): 구현 명세. 수식, 판정 순서, 시험 기준
- [`CHANGELOG.md`](CHANGELOG.md): API 변경 내역
- [`README_C_PART.md`](README_C_PART.md): 라이브러리를 쓰는 쪽을 위한 짧은 안내
