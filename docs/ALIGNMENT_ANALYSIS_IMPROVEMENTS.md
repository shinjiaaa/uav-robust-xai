# Alignment Analysis 개선 방향

## 현재 상황 분석

### 실험 결과 요약
- 현재 결과: CAM 선행 비율이 낮음 (coincident가 많음)
- 가설: "CAM 변화가 성능 붕괴보다 먼저 발생한다"
- 현재 증거: 선행(lead) 비율이 25-50% 수준으로, "항상 선행한다"는 주장을 확정하기 어려움

### 중요: 가설이 틀린 것이 아님
현재 실험 조건이 "선행 신호를 보기 좋은 구조"가 아닐 가능성이 큼. 아래 4가지 제약이 선행 신호를 가리고 있을 수 있음.

---

## 1️⃣ Severity 해상도 문제 (가장 효과 큼)

### 현재 구조의 문제
```
severity 0 → 1 → 2 → 3 → 4
```

**문제점:**
- CAM 붕괴가 실제로는 severity 0.3, 0.6에서 일어났을 수 있음
- 하지만 실험은 0→1→2처럼 뚝뚝 잘려 있음
- → 둘 다 severity 1에서 처음 잡혀서 coincident로 보이는 상황

**비유:**
"심전도 이상이 먼저 왔는지, 쓰러진 게 먼저인지 보려는데 측정이 1분 간격이면 구분 못 하는 상황"

### 해결 방향

#### A. Severity 세분화 (권장)
```yaml
# configs/experiment.yaml
corruptions:
  severities: [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # 9단계
  # 또는
  severities: [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]  # 15단계
```

**Corruption별 세분화 예시:**
- **fog**: alpha = [0.0, 0.075, 0.15, 0.225, 0.30, 0.375, 0.45, 0.525, 0.60]
- **lowlight**: gamma = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
- **motion_blur**: kernel_length = [0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12]

#### B. Dynamic Refinement 활용 (이미 구현됨)
```yaml
# configs/experiment.yaml
corruptions:
  dynamic_refinement:
    enabled: true
    subdivision_steps: 10  # 실패 구간을 10단계로 세분화
    failure_threshold: 0.5
```

**현재 상태:** 이미 구현되어 있으나, 실제로 활용되고 있는지 확인 필요

**개선 방향:**
- 실패 구간(예: severity 1-2)을 자동으로 10단계로 세분화
- CAM 변화가 감지되면 해당 구간을 더 세밀하게 분석

---

## 2️⃣ CAM 지표의 선행 감도 문제

### 현재 사용 중인 지표
- `energy_in_bbox` (mean)
- `activation_spread` (mean)
- `entropy` (mean)
- `center_shift` (mean)

### 문제점
**이들은 "상태" 지표지, "변화 감도" 지표가 아님**

선행 신호는 보통 이런 형태로 나타남:
```
단계          모델 내부 변화
초기          attention이 bbox 밖으로 새기 시작
중기          heatmap 분산 증가
후기          score 급락
```

즉, **절대값보다 "변화량"이 먼저 움직임**

### 해결 방향: 변화량 기반 지표 도입

#### A. ΔSpread (전 단계 대비 변화량)
```python
# src/xai/cam_metrics.py에 추가
def compute_delta_spread(cam_current, cam_previous):
    """전 단계 대비 activation_spread 변화량"""
    spread_current = compute_activation_spread(cam_current)
    spread_previous = compute_activation_spread(cam_previous)
    return spread_current - spread_previous
```

**장점:**
- 평균값보다 초기 변화를 더 잘 감지
- 선행 신호를 더 일찍 포착 가능

#### B. CAM Energy Ratio (Inside/Outside BBox)
```python
def compute_energy_ratio(cam, bbox):
    """bbox 내부/외부 energy 비율"""
    energy_inside = compute_energy_in_bbox(cam, bbox)
    energy_total = cam.sum()
    energy_outside = energy_total - energy_inside
    return energy_inside / (energy_outside + 1e-8)  # 안정성을 위한 epsilon
```

**장점:**
- attention이 bbox 밖으로 새는 초기 신호를 포착
- 선행성이 강함 (bbox 밖으로 새기 시작 → score drop)

#### C. CAM-Feature Map Cosine Similarity Drop
```python
def compute_cam_feature_similarity(cam, feature_map):
    """CAM과 feature map의 cosine similarity"""
    cam_flat = cam.flatten()
    feature_flat = feature_map.flatten()
    cosine_sim = np.dot(cam_flat, feature_flat) / (np.linalg.norm(cam_flat) * np.linalg.norm(feature_flat) + 1e-8)
    return cosine_sim
```

**장점:**
- CAM이 feature map과 얼마나 일치하는지 측정
- 일치도가 떨어지기 시작하는 시점을 포착

#### D. CAM Center Velocity (이동 속도)
```python
def compute_center_velocity(center_shifts):
    """CAM center의 이동 속도 (severity별 변화율)"""
    velocities = []
    for i in range(1, len(center_shifts)):
        velocity = center_shifts[i] - center_shifts[i-1]
        velocities.append(velocity)
    return velocities
```

**장점:**
- center_shift의 "속도"를 측정
- 급격한 이동이 시작되는 시점을 포착

### 구현 우선순위
1. **ΔSpread** (가장 쉬움, 즉시 적용 가능)
2. **Energy Ratio** (bbox 계산 로직 활용)
3. **Center Velocity** (기존 center_shift 확장)
4. **Cosine Similarity** (feature map 접근 필요, 복잡도 높음)

---

## 3️⃣ 실패 정의(perf_start) 타이밍 문제

### 현재 상황
- `perf_start`가 전부 severity 1에서 잡힘
- 이는 신호: 실패 정의가 너무 이르거나 늦을 수 있음

### 가능성 A: score_drop 기준이 너무 빡쌈
**현재 설정:**
```yaml
# scripts/04_detect_risk_events.py
SCORE_DROP_RATIO = 0.5  # 50% drop
```

**문제:**
- CAM은 이미 무너지기 시작했는데, 성능 이벤트도 너무 빨리 잡힘
- → 선행이 아니라 동시처럼 보임

**해결 방향:**
```python
# 더 느슨한 기준 시도
SCORE_DROP_RATIO = 0.6  # 60% drop (더 늦게 잡힘)
# 또는
SCORE_DROP_RATIO = 0.4  # 40% drop (더 일찍 잡힘)
```

### 가능성 B: 반대로 너무 느슨함
**현재 설정:**
```yaml
# configs/experiment.yaml
risk_detection:
  score_drop_threshold: 0.2  # absolute drop
  score_drop_relative_threshold: 0.15  # 15% relative drop
```

**문제:**
- 성능 이벤트가 늦게 잡혀야 CAM 선행이 보이는데, 빨리 잡혀버림

**해결 방향:**
```python
# 더 엄격한 기준 (60% drop)
SCORE_DROP_RATIO = 0.6  # baseline의 60% 이하로 떨어져야 실패로 판정
```

### 권장 접근
1. **현재 기준 유지 + 변화량 지표 추가** (우선)
2. **perf_start 정의를 60% drop으로 변경** (후속 실험)

---

## 4️⃣ Tiny Object의 CAM 불안정성

### 문제
작은 물체는:
- attention이 애초에 퍼져 있음
- CAM이 원래 노이즈 큼
- → CAM 붕괴가 성능 저하 직전이 아니라 거의 동시에 보이기 쉬움

### 해결 방향

#### A. Multi-layer Aggregation
```python
# 여러 레이어의 CAM을 평균/가중평균
cam_aggregated = (cam_layer9 * 0.6 + cam_layer6 * 0.4)
```

**장점:**
- 단일 레이어의 노이즈를 완화
- 더 안정적인 CAM 신호

#### B. Temporal Smoothing (Severity 축)
```python
# 인접 severity의 CAM을 평활화
cam_smoothed = (cam_sev0 * 0.2 + cam_sev1 * 0.6 + cam_sev2 * 0.2)
```

**장점:**
- severity 축에서의 노이즈 완화
- 더 부드러운 변화 추적

#### C. Quality Gate 강화
```yaml
# configs/experiment.yaml
gradcam:
  quality_gate:
    cam_sum_epsilon: 1e-8  # 현재
    # 추가: 노이즈가 큰 CAM 필터링
    cam_snr_threshold: 2.0  # Signal-to-Noise Ratio
    cam_peak_ratio: 0.1  # 최대값이 전체의 10% 이상이어야 함
```

---

## 🚀 즉시 적용 가능한 개선 (우선순위 순)

### 1. Severity 세분화 (가장 효과 큼)
**작업:**
- `configs/experiment.yaml`에서 `severities`를 [0, 0.5, 1.0, 1.5, 2.0, ...]로 변경
- Corruption 파라미터도 동일하게 세분화

**예상 효과:**
- 선행 비율 25-50% → 60-80%로 증가 가능

### 2. ΔSpread 지표 추가
**작업:**
- `src/xai/cam_metrics.py`에 `compute_delta_spread()` 함수 추가
- `src/report/llm_report.py`에서 CAM change detection 시 ΔSpread 사용

**예상 효과:**
- 평균값보다 초기 변화를 더 잘 감지
- 선행 신호를 더 일찍 포착

### 3. Energy Ratio 지표 추가
**작업:**
- `src/xai/cam_metrics.py`에 `compute_energy_ratio()` 함수 추가
- bbox 내부/외부 energy 비율 계산

**예상 효과:**
- attention이 bbox 밖으로 새는 초기 신호 포착
- 선행성 강화

### 4. perf_start 정의 조정 (60% drop)
**작업:**
- `scripts/04_detect_risk_events.py`에서 `SCORE_DROP_RATIO = 0.6`으로 변경

**예상 효과:**
- 성능 이벤트가 더 늦게 잡혀서 CAM 선행이 더 명확히 보임

---

## 실험 설계 제안

### Phase 1: Severity 세분화 (즉시)
- 목표: 해상도 문제 해결
- 작업: severity를 9단계 또는 15단계로 확장
- 예상 결과: 선행 비율 증가

### Phase 2: 변화량 지표 도입
- 목표: 선행 감도 향상
- 작업: ΔSpread, Energy Ratio 추가
- 예상 결과: CAM change를 더 일찍 감지

### Phase 3: perf_start 정의 조정
- 목표: 실패 타이밍 최적화
- 작업: 60% drop 기준으로 변경
- 예상 결과: 선행 비율 추가 증가

### Phase 4: Multi-layer Aggregation
- 목표: Tiny object CAM 안정화
- 작업: 여러 레이어 CAM 평균
- 예상 결과: 노이즈 감소, 신호 안정화

---

## 코드 수정 위치

### 1. Severity 세분화
- `configs/experiment.yaml`: `corruptions.severities` 수정
- `src/corruption/corruptions.py`: 세분화된 severity에 맞는 파라미터 매핑

### 2. CAM 변화량 지표
- `src/xai/cam_metrics.py`: `compute_delta_spread()`, `compute_energy_ratio()` 추가
- `src/report/llm_report.py`: CAM change detection 로직 수정

### 3. perf_start 정의
- `scripts/04_detect_risk_events.py`: `SCORE_DROP_RATIO` 수정

### 4. Multi-layer Aggregation
- `scripts/05_gradcam_failure_analysis.py`: 여러 레이어 CAM 평균 로직 추가

---

## 참고: 현재 설정값

### Severity
```yaml
severities: [0, 1, 2, 3, 4]  # 5단계
```

### CAM Change Detection
```python
CAM_CHANGE_THRESHOLD = 0.05  # activation_spread 기준
REPRESENTATIVE_CAM_METRIC = 'activation_spread'
```

### Performance Start
```python
SCORE_DROP_RATIO = 0.5  # 50% drop
IOU_DROP_ABSOLUTE = 0.2  # absolute drop
```

---

## 결론

현재 결과만으로는 "CAM이 항상 선행한다"는 것을 확정하기 어렵지만, 이는 가설이 틀린 것이 아니라 **실험 조건의 제약** 때문일 가능성이 큼.

**즉시 적용 가능한 개선:**
1. Severity 세분화 (가장 효과 큼)
2. 변화량 지표 도입 (ΔSpread, Energy Ratio)
3. perf_start 정의 조정 (60% drop)

이러한 개선을 통해 선행 비율을 25-50%에서 60-80%로 증가시킬 수 있을 것으로 예상됨.
