# foodcv 사용법 (C파트용)

`foodcv.dll`, `tools/foodcv.py`, `model.fcv`, `config/*`를 받아서 쓴다. Python은 64비트여야 한다.

```python
import cv2
from foodcv import FoodCV, AREA_OK, SEG_STABLE

cv = FoodCV("foodcv.dll", "model.fcv", "config/area.cfg",
            "config/display.layout", "config/display.quad", "config/roi.cfg")

cv.calibrate_plate(cv2.imread("empty_plate.png"))     # 시작할 때 빈 접시 1장. 0이 아니면 다시 찍는다

k, name, p, rejected, probs = cv.classify(frame)      # frame: OpenCV BGR uint8
area = cv.measure_area(frame)                         # uses_area인 클래스만
if area.status == AREA_OK:
    x = sum(a ** gamma for a in area.part_cm2)        # area.n_parts > 16이면 γ ≠ 1에서 계산 불가
status, seconds = cv.read_display(frames)             # 5 × fuse장. STABLE이 아니면 seconds = -1
```

- 음수 상태는 `FoodCVError` 예외로 나온다. `area.status`의 뜻과 UI 안내는 `A_PART_SPEC.md` §4.4에 있다.
- 시작할 때 `fcv_class_name`과 프로파일 이름이 모두 같은지 확인한다(§7.3).
- API 3의 변경 사항과 C파트 요청은 `CHANGELOG.md`에 있다.
