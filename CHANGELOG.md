# CHANGELOG

## API 3 (2026-11-04)

C파트가 영향을 받는 변경이다. 근거 수치는 `A_PART_SPEC.md` §4.9.

### 공개 헤더

- `FCV_API_VERSION` 2 → 3.
- `area_*` status 추가: `AREA_LIGHT_CHANGED = 5`, `AREA_PLATE_MOVED = 6`, `AREA_NOT_EMPTY = 7`(보정 전용).
- `area_params` 뒤에 필드 추가: `light_lo`, `light_hi`, `edge_tol`, `empty_tol`.
- `area_result`에 `part_fill[AREA_MAX_PARTS]` 추가. `part_cm2` 바로 뒤에 들어가므로 뒤 필드의 오프셋이 바뀐다.
- `seg_grid_index(int seconds)` → `seg_grid_index(int seconds, int *index)`. 기존 "없음 = -1"이 `FCV_E_FILE`과 같았다.

### 동작

- 보정 후 빨강 판정에 `R ≥ red_k·G`를 추가했다. 파란 접시에서 가림을 못 잡던 문제를 고친다.
- 후보 없음에서 가장 큰 빨간 성분이 2.5·N_cal을 넘으면 `AREA_REF_MISSING` 대신 `AREA_REF_MERGED`를 낸다.
- `area_measure`에 조명 검사(§4.3-5a)와 접시 이동 검사(§4.3-5b)를 추가했다.
- `area_calibrate`가 빈 접시인지 확인한다(§4.2-8).
- 자리가 무효면 `seg_read`의 `seconds`는 -1이고, `seg_vote`는 이 결과를 집계에서 뺀다.

### 파일 형식

- `model.fcv` 버전 1 → 2. 해시 범위를 파일 앞부분 전체로 넓혔다.
- `config/display.quad` 추가, `display.layout`에 `fuse` 줄 추가, `area.cfg`에 키 4개 추가.

### Python 래퍼

- `FoodCV.__init__`에 `display_quad` 인자 추가.
- `read_display`는 `5 × fuse`장을 받고, STABLE이 아니면 seconds로 -1을 돌려준다.
- `grid_index`는 격자 밖이면 예외 없이 -1을 돌려준다.

### C파트 요청

- 새 status 세 개의 UI 안내를 넣는다(§4.4).
- 면적이 프로파일 `area_range_cm2` 밖이면 시간 계산을 막는다.
- 치킨 조각 합침 처리 방침을 A파트와 정한다(§10).
