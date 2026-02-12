# RQ1 Implementation Plan

## 목표 (RQ1)

**(RQ1-a)** 변조 severity 증가에 따라 **탐지 성능 저하(미스/스코어/IoU 급락)**가 발생할 때, Grad-CAM 분포 지표가 일관된 변화 패턴(단조 변화/확산/이동/붕괴)을 보이는가?

**(RQ1-b)** 그 패턴은 변조 유형(fog/lowlight/motion_blur) 또는 **모델(yolo_generic / FT / RT-DETR …)**이 바뀌어도 **동일한 템플릿(방향/형태)**로 재현되는가?

## 핵심 구현 3가지 (우선순위)

### 1. Miss에서도 CAM을 뽑는 타겟 정의 확정 ⚠️ CRITICAL

**문제:** 현재 matched=0일 때 CAM이 생성되지 않거나 실패 → 실패 구간 CAM이 비어서 RQ1 주장 불가

**해결책:**
- **matched=1일 때**: 매칭된 prediction의 class_id와 score를 target으로 사용
- **matched=0일 때 (miss)**: 다음 중 하나로 정의
  1. **GT class target logit** (권장): GT class_id에 대한 logit을 target으로 사용
  2. **가장 근접한 후보**: IoU가 가장 높은 prediction (IoU < threshold여도)의 class_id 사용

**구현 위치:**
- `scripts/05_gradcam_failure_analysis.py`: detection_records에서 matched 정보 로드
- `src/xai/gradcam_yolo.py`: `generate_cam()` 함수에 target 선택 로직 추가

**코드 변경:**
```python
# detection_records에서 matched prediction 정보 로드
if matched == 1:
    target_class_id = pred_class_id  # 매칭된 prediction의 class
    target_score = pred_score
else:
    # Miss: GT class target logit 사용
    target_class_id = gt_class_id
    target_score = None  # GT에는 score가 없으므로 logit만 사용
```

### 2. Event-window CAM 생성으로 커버리지 확보 ⚠️ CRITICAL

**문제:** 현재 n_cam_frames가 0~5로 너무 적음 → 시계열 분석 불가

**해결책:**
- **Severity window**: 현재 `cam_sev_from ~ cam_sev_to` 유지
- **Frame window 추가**: `perf_start_frame ± k` (k=10 권장)
- **결합**: severity window AND frame window 교집합

**구현 위치:**
- `scripts/04_detect_risk_events.py`: `cam_sev_from`, `cam_sev_to`에 추가로 `cam_frame_from`, `cam_frame_to` 계산
- `scripts/05_gradcam_failure_analysis.py`: frame window도 고려하여 CAM 생성

**코드 변경:**
```python
# risk_events.csv에 추가 컬럼
cam_frame_from = max(0, perf_start_frame - k)  # k=10
cam_frame_to = perf_start_frame + k

# CAM 생성 시
for severity in range(cam_sev_from, cam_sev_to + 1):
    for frame_idx in range(cam_frame_from, cam_frame_to + 1):
        # CAM 생성
```

### 3. cam_change_sev 검출기 고정 (임계치 기반) ⚠️ CRITICAL

**문제:** cam_change_sev가 수동/불일치 → 통계적 신뢰도 낮음

**해결책:**
- **임계치 기반 검출 (메인)**: z-score >= k (k=2 또는 3)
- **Baseline 정규화**: severity 0의 metric을 baseline으로 z-score 계산
- **PELT 변화점 탐지 (보조)**: 통계적 변화점 검출

**구현 위치:**
- `src/report/llm_report.py`: `alignment_analysis` 섹션에서 cam_change_sev 계산 로직 수정
- 새 파일: `src/eval/cam_change_detection.py` (검출 알고리즘 분리)

**알고리즘:**
```python
def detect_cam_change(metric_series, baseline_metric, threshold_k=2.0):
    """
    metric_series: dict {severity: metric_value}
    baseline_metric: severity 0의 metric 값
    threshold_k: z-score threshold (default 2.0 = 2σ)
    
    Returns:
        cam_change_severity: 최초 변화점 severity
    """
    # Baseline 정규화 (severity 0의 여러 프레임 평균/std 사용)
    baseline_mean = baseline_metric['mean']
    baseline_std = baseline_metric['std'] if baseline_metric['std'] > 0 else 1e-6
    
    # 각 severity에서 z-score 계산
    for sev in sorted(metric_series.keys()):
        if sev == 0:
            continue
        metric_val = metric_series[sev]
        z_score = abs(metric_val - baseline_mean) / baseline_std
        
        if z_score >= threshold_k:
            return sev  # 최초 변화점
    
    return None  # 변화점 없음
```

## 전체 파이프라인 실행 순서

### Phase 1: 탐지 로그 생성 (전량)
- ✅ `scripts/03_detect_tiny_objects_timeseries.py`: detection_records.csv 생성
- ✅ `scripts/04_detect_risk_events.py`: risk_events.csv 생성 (perf_start_sev 포함)

### Phase 2: CAM 생성 (Event-window 방식)
- 🔄 `scripts/05_gradcam_failure_analysis.py`: 
  - Miss에서도 CAM 생성 (타겟 정의 수정)
  - Event-window (severity + frame) 적용
  - cam_records.csv 생성

### Phase 3: cam_change 검출 (알고리즘 고정)
- 🔄 `src/eval/cam_change_detection.py`: 새 파일 생성
  - 임계치 기반 검출 (메인)
  - PELT 변화점 탐지 (보조)
- 🔄 `src/report/llm_report.py`: cam_change_sev 계산 로직 수정

### Phase 4: Alignment + 패턴 일치도
- 🔄 `src/report/llm_report.py`: 
  - alignment 계산 (lead_steps = perf_start_sev - cam_change_sev)
  - 패턴 signature 생성
  - 패턴 일치도 계산 (direction agreement, Spearman, DTW)

## 실험 설계 파라미터 (초기 권장값)

```yaml
# configs/experiment.yaml에 추가
rq1_analysis:
  event_window:
    frame_k: 10  # ±10 프레임
    severity_range: [s-1, s, s+1]  # perf_start_sev 기준
  cam_change_detection:
    method: "threshold"  # "threshold" or "pelt"
    threshold_k: 2.0  # z-score threshold (2σ)
    baseline_normalization: "z_score"  # "z_score" or "ratio"
  pattern_agreement:
    metrics: ["energy_in_bbox", "activation_spread", "entropy", "center_shift"]
    similarity_metrics: ["direction_agreement", "spearman", "dtw"]
  min_events_per_condition: 30  # corruption×model당 최소 이벤트 수
```

## 최종 산출물 (리뷰어용)

1. **Event-level evidence table**: 
   - `alignment_detail.csv`: event_id, perf_start_sev, cam_change_sev, lead_steps, metric, pattern_signature

2. **Corruption×Model summary**: 
   - `alignment_summary.csv`: lead% / coincident% / lag% + 평균 lead step + 유의성

3. **패턴 일치도 matrix**: 
   - `pattern_agreement_matrix.csv`: (모델 간, 변조 간) direction_agreement, Spearman, DTW

4. **대표 시계열 플롯 (부록)**: 
   - 성능축 vs CAM축을 severity/time로 정렬한 그림

## 다음 단계

1. ✅ Miss에서도 CAM 타겟 정의 구현
2. ✅ Event-window CAM 생성 구현
3. ✅ cam_change_sev 검출기 고정 구현
4. 🔄 패턴 일치도 계산 구현
5. 🔄 시각화 (시계열 플롯)
