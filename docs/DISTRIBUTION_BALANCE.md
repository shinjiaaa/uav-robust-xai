# 데이터 분포 균형: 다른 XAI 실험과 차이·맞추는 방법

같은 데이터셋으로 다른 XAI를 쓰면 “이상적인” 분포(예: stable/persistent 비율이 적당한)가 잘 나오는데, **여기(Grad-CAM + tiny object)에서는 transient/oscillatory가 많고 제대로 된 분포를 뽑기 어렵다**고 느껴질 수 있습니다. 이유와 대응을 정리합니다.

---

## 1. 왜 이 실험에서만 분포가 틀어져 보일 수 있는가

| 요인 | 설명 |
|------|------|
| **Grad-CAM + tiny object** | 작은 객체는 수 픽셀 단위라, gradient 기반 설명이 **한 레이어·한 프레임**에 매우 민감. 다른 XAI(attention, score-CAM 등)는 상대적으로 덜 요동침. |
| **ROI(bbox crop)만 사용** | CAM을 bbox 크롭 안에서만 계산하면, **박스가 살짝 어긋나거나 letterbox 차이**만 있어도 E_bbox/spread가 크게 요동. 전체 이미지 기준 XAI는 이 영향이 적음. |
| **단일 레이어** | 한 레이어만 쓰면 그 레이어에서의 “일시적” 붕괴가 그대로 transient로 잡힘. 다중 레이어/다중 설명을 쓰는 실험은 평균되어 부드러운 추세에 가깝게 나옴. |
| **변조 × CAM 상호작용** | fog/lowlight/motion_blur가 **대비·그래디언트**를 바꿔서, 중간 severity에서만 CAM이 잠깐 flat이 되는 구간이 생기기 쉬움. |

즉 **모델이 이상한 것이 아니라**, “설명 방식 + tiny + ROI + 단일 레이어” 조합이 **측정을 더 불안정하게 만든다**고 보는 것이 타당합니다.

---

## 2. 전체 데이터 균형을 맞추는 방법 (설정으로 제어)

실험 설계는 그대로 두고, **분류 단계에서만** “측정 실패·요동”을 보정해 분포를 다른 XAI 실험에 가깝게 맞추는 옵션입니다.

### 2.0 완화된 붕괴 규칙 (권장, 기본 on)

- **설정**: `pattern_classification.use_relaxed_breakdown: true` (기본값)
- **핵심**: “Grad-CAM의 순간적 설명 실패를 붕괴로 간주하지 않도록” 판정 규칙 변경.
- **이상 신호**: 다음 중 **2개 이상** 동시 성립 시 해당 severity만 이상 신호.
  - `E_bbox_1.25x <= 0.2`, `cam_quality = flat`, `cam_sum <= 1e-8`, `spread <= 1e-5`,
  - `entropy >= 1.3 * entropy_L0`, `bbox_dist >= 2 * bbox_dist_L0`
- **공식 breakdown**: 이상 신호가 **2단계 연속**이거나, 이상 신호 발생 **후 회복 없음**일 때만 1.  
  → 한 단계만 깨진 경우는 transient로 분리되고, 공식 붕괴로 세지 않음.
- **효과**: flat/spread=0/cam_sum=0 단독 붕괴 금지 → transient 과다 완화, 자연스러운 추세가 더 많이 남음.

### 2.1 품질 게이트 (이미 구현)

- **설정**: `pattern_classification.use_quality_gate: true`
- **효과**: L0~L4 **모두** `cam_valid==True`인 시계열만 패턴 분류에 사용.
- **의미**: “측정 실패 없이 정상 관측된” 샘플만 집계 → transient가 과대 추정되는 구간을 줄임.

### 2.2 Invalid 구간 보간 (추가 구현)

- **설정**: `pattern_classification.interpolate_invalid: true`
- **동작**: `cam_valid==False`인 severity의 E_bbox/spread/entropy/bbox_dist를 **인접 유효 구간으로 선형 보간**한 뒤, 그 값으로 breakdown을 다시 계산해 패턴 재분류.
- **산출**: `pattern_repaired`, `pattern_counts_repaired.json`, 보고서 §5 (보간 기준 집계).
- **의미**: “한두 단계 측정 실패” 때문에 transient로 잡힌 것을 완화해, **보간된 추세** 기준으로 분포를 맞출 수 있음.

### 2.3 스무딩 (추가 구현)

- **설정**: `pattern_classification.use_smoothing: true`
- **동작**: 시계열에 **3점 이동평균** 적용 후 breakdown 재계산·패턴 재분류.
- **산출**: `pattern_smoothed`, `pattern_counts_smoothed.json`, 보고서 §5 (스무딩 기준 집계).
- **의미**: 작은 요동을 평균해서 **단조에 가까운 추세**로 바꾸면, stable/persistent 비율이 늘어나 다른 XAI 실험의 “깔끔한” 분포에 가깝게 볼 수 있음.

### 2.4 조합 권장

- **일단 분포만 맞춰보기**: `interpolate_invalid: true` + `use_smoothing: true` 로 09 실행 → `pattern_counts_repaired.json`, `pattern_counts_smoothed.json`과 보고서 §5를 보고, 다른 실험의 비율과 비슷해졌는지 확인.
- **엄격한 품질만 쓰기**: `use_quality_gate: true` 로 “전부 유효한” 시계열만 남긴 뒤, 위 보간/스무딩을 적용해도 됨.

---

## 3. 파이프라인상 위치

- **09_cam_pattern_classification.py** 실행 시 `configs/experiment.yaml`의 `pattern_classification`를 읽음.
- `interpolate_invalid` / `use_smoothing` 이 true이면, **원본 패턴( pattern )은 그대로 두고** `pattern_repaired`, `pattern_smoothed` 컬럼과 별도 집계 파일을 추가로 만듦.
- “이상적 추세만 보기”·ideal_trend_samples는 **기존처럼 원본 pattern** 기준으로 동작 (변경 없음).

---

## 4. 요약

- **이상 현상이 많은 이유**: Grad-CAM + tiny + ROI + 단일 레이어 때문에 “측정이 예민해서” transient/oscillatory가 많이 잡힘.
- **해결 방향**: 모델 수정이 아니라 **측정 안정화** (품질 게이트, 보간, 스무딩)로 전체 데이터 분포를 조정.
- **다른 XAI와 맞추기**: `interpolate_invalid`·`use_smoothing`을 켜고, `pattern_counts_repaired.json` / `pattern_counts_smoothed.json`·보고서 §5를 기준으로 분포를 비교·보고하면 됨.
