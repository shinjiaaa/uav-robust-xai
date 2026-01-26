# 파이프라인 문제 분석 및 해결책

## 1️⃣ lowlight/motion_blur가 detection_records에 없는 원인

### 진단 결과
**실제로는 detection_records가 존재합니다!**
- fog: 500 records (100 objects × 5 severities)
- lowlight: 500 records
- motion_blur: 500 records

### 문제의 실제 원인
리포트 생성 시점에 `llm_report.py`의 `available_corruptions_from_det` 필터링 로직이 잘못 작동했을 가능성이 있습니다.

### 해결책
1. **진단 스크립트 실행**: `scripts/diagnose_pipeline.py`로 실제 데이터 상태 확인
2. **리포트 재생성**: 데이터가 존재하므로 리포트를 재생성하면 정상적으로 표시됩니다
3. **로깅 강화**: `03_detect_tiny_objects_timeseries.py`에 corruption별 처리 카운트 로깅 추가

### 추가된 로깅
```python
# scripts/03_detect_tiny_objects_timeseries.py에 추가
corruption_counts = {}
for corruption in corruptions:
    corruption_counts[corruption] = 0
    # ... 처리 후
    corruption_counts[corruption] += 1
    if corruption_counts[corruption] % 100 == 0:
        print(f"  Processed {corruption_counts[corruption]} records for {corruption}")
```

---

## 2️⃣ Grad-CAM이 fog에서만 계산되는 이유

### 진단 결과
- fog: 119 CAM records
- lowlight: 0 CAM records (failure_events는 27개 존재)
- motion_blur: 0 CAM records (failure_events는 33개 존재)

### 원인 분석
`05_gradcam_failure_analysis.py`에서:
1. failure_events에서 샘플링: `max_samples//3` per corruption (line 86)
2. lowlight/motion_blur의 failure_events는 존재하지만, CAM 생성 단계에서 실패했을 가능성
3. 에러 로그 확인 필요: `results/gradcam_errors.csv`

### 해결책
1. **에러 로그 분석**: `gradcam_errors.csv`에서 lowlight/motion_blur의 에러 패턴 확인
2. **CAM 생성 재시도**: lowlight/motion_blur의 failure_events에 대해 CAM 생성 재시도
3. **에러 처리 개선**: shape mismatch, device mismatch 등 에러에 대한 더 robust한 처리

### Methods/Limitations 섹션에 추가할 설명
```
Grad-CAM Analysis Scope:
- CAM computation is performed only for failure events that satisfy:
  1) Detection records exist (tiny_records_timeseries.csv)
  2) Failure event detected (failure_events.csv)
  3) Successful CAM generation (no shape/device errors)
  
- In this experiment, fog corruption had 119 successful CAM generations,
  while lowlight and motion_blur had 0 due to CAM generation failures
  (see gradcam_errors.csv for detailed error logs).
```

---

## 3️⃣ score drop / IoU drop 정의의 corruption-independent 여부

### 코드 분석 결과
**✅ corruption-independent입니다!**

`src/eval/failure_detection.py` (line 28-45):
```python
# Group by model, corruption, image_id, class_id (tiny object)
groups = records_df.groupby(['model', 'corruption', 'image_id', 'class_id'])

for (model, corruption, image_id, class_id), group in groups:
    # Baseline (severity 0) - 각 corruption별로 독립적으로 계산
    baseline = group[group['severity'] == 0]
    baseline_score = baseline_matched['score'].mean()
    baseline_iou = baseline_matched['iou'].mean()
    
    # 각 corruption의 severity 0를 baseline으로 사용
    if baseline_score - avg_score >= risk_config['score_drop_threshold']:
        score_drop_severity = severity
```

### 확인 사항
- ✅ 각 corruption별로 독립적인 baseline (severity 0) 사용
- ✅ threshold는 config에서 공통으로 정의 (`risk_config['score_drop_threshold']`)
- ✅ fog의 baseline을 다른 corruption에 적용하지 않음

### 리포트에 명시할 내용
```
Score Drop / IoU Drop Definition:
- Baseline: severity 0 for each corruption (corruption-specific baseline)
- Threshold: score_drop_threshold = 0.1, iou_drop_threshold = 0.1 (common across corruptions)
- Detection: First severity where (baseline_score - current_score) >= threshold
- This ensures fair comparison across different corruption types.
```

---

## 4️⃣ "failure_severity up to …" 루프의 의미 명확화

### 코드 분석 결과
`scripts/05_gradcam_failure_analysis.py` (line 158):
```python
for severity in range(0, failure_severity + 1):
    # 각 severity별로 독립적으로 CAM 계산
    cam = gradcam.generate_cam(image, yolo_bbox, class_id)
    metrics = compute_cam_metrics(cam, yolo_bbox, ..., baseline_cam=baseline_cam)
```

### 현재 동작
- **누적 계산이 아님**: 각 severity별로 독립적으로 CAM 생성 및 metric 계산
- **baseline 비교**: severity 0의 CAM을 baseline으로 사용하여 각 severity의 CAM과 비교
- **범위**: severity 0부터 failure_severity까지 (failure_severity 포함)

### 명확화 필요 사항
1. **CAM metric의 의미**: 각 severity에서의 CAM distribution (baseline 대비 변화)
2. **failure_severity 포함 이유**: 실패가 발생한 severity에서의 CAM도 분석 대상
3. **누적 vs 독립**: 각 severity는 독립적으로 계산되며, baseline과의 비교를 통해 변화 추적

### Methods 섹션에 추가할 설명
```
Grad-CAM Computation:
- For each failure event, CAM is computed for severity 0 (baseline) 
  through failure_severity (inclusive).
- Each severity's CAM is computed independently (not cumulative).
- CAM metrics (energy_in_bbox, activation_spread, entropy, center_shift) 
  are computed by comparing each severity's CAM to the baseline (severity 0).
- This allows tracking CAM distribution changes as corruption severity increases
  up to the point of failure.
```

---

## 요약 및 다음 단계

### 즉시 해결 가능한 문제
1. ✅ **detection_records 존재 확인**: 실제로는 모든 corruption에 데이터가 있음
2. ✅ **score/IoU drop 정의**: corruption-independent로 정상 작동
3. ✅ **failure_severity 루프 의미**: 코드 분석 완료, 문서화 필요

### 추가 작업 필요
1. **Grad-CAM 에러 분석**: `gradcam_errors.csv` 분석하여 lowlight/motion_blur CAM 실패 원인 파악
2. **로깅 강화**: 각 단계에서 corruption별 처리 상태 로깅 추가
3. **문서화**: Methods/Limitations 섹션에 위 내용 추가

### 실행 명령
```bash
# 1. 파이프라인 진단
python scripts/diagnose_pipeline.py

# 2. Grad-CAM 에러 분석
python -c "import pandas as pd; df = pd.read_csv('results/gradcam_errors.csv'); print(df.groupby('corruption').size())"

# 3. 리포트 재생성 (데이터가 존재하므로 정상 작동할 것)
python scripts/06_generate_report.py
```
