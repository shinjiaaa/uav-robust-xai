# Pilot Experiment Guide

## 개요

사전 실험(Pilot Experiment)은 극단적인 변조 단계(severity 50)를 사용하여 파이프라인을 테스트하고, 적절한 변조 단계를 결정하기 위한 소규모 실험입니다.

## 실험 설정

### 현재 설정 (configs/experiment.yaml)

```yaml
experiment:
  pilot_mode: true
  sample_size: 100  # 100개 tiny object 샘플링
  one_per_image: true  # 이미지당 1개만 (중복 이미지 방지)

corruptions:
  severities: [0, 50]  # 극단적 severity 테스트
```

### 극단적 Severity 값

- **Fog (severity 50)**: alpha = 0.95
- **Low-light (severity 50)**: gamma = 5.0, brightness_scale = 0.1
- **Motion Blur (severity 50)**: kernel_length = 100

## 실행 방법

```bash
# 사전 실험 실행
python scripts/00_pilot_experiment.py
```

또는 단계별 실행:

```bash
# 1. 단일 이미지에서 tiny object 샘플링 (100개 샘플, 이미지당 1개)
python scripts/01_sample_tiny_objects.py

# 2. 극단적 변조 생성 (severity 0, 50)
python scripts/02_generate_corruption_sequences.py

# 3. 탐지 수행
python scripts/03_detect_tiny_objects_timeseries.py

# 4. 실패 이벤트 감지
python scripts/04_detect_failure_events.py

# 5. Grad-CAM 분석 (동적 세분화 포함)
python scripts/05_gradcam_failure_analysis.py
```

## 동적 세분화 (Dynamic Refinement)

Grad-CAM 분석에서 "펑 터지는" 구간을 자동으로 감지하고, 그 구간을 10단계로 세분화하여 상세 분석합니다.

### 작동 방식

1. **실패 구간 감지**: CAM 지표의 급격한 변화 감지 (threshold: 0.3)
2. **세분화**: 실패 구간을 10단계로 나눔
   - 예: severity 2-3 구간이 실패 구간으로 감지되면
   - 2.1, 2.2, 2.3, ..., 2.9, 3.0으로 세분화
3. **재분석**: 세분화된 각 단계에 대해 Grad-CAM 재계산

## 결과 확인

### 주요 출력 파일

- `results/tiny_records_timeseries.csv`: 탐지 기록 (miss, score, IoU)
- `results/failure_events.csv`: 실패 이벤트
- `results/gradcam_metrics_timeseries.csv`: CAM 지표 (기본 + 세분화)
- `results/risk_regions.csv`: 위험 구간

### 다음 단계

1. **결과 분석**: severity 50에서의 성능 저하 확인
2. **Severity 조정**: 
   - severity 50이 너무 극단적이면 → 낮춤 (예: 30, 40)
   - severity 50에서도 성능이 유지되면 → 더 높임 (예: 60, 70)
3. **전체 실험**: 적절한 severity 범위 결정 후 전체 실험 수행

## 전체 실험으로 전환

`configs/experiment.yaml` 수정:

```yaml
experiment:
  pilot_mode: false
  sample_size: 500  # 전체 샘플 크기
  one_per_image: true  # 유지

corruptions:
  severities: [0, 1, 2, 3, 4]  # 표준 severity
```
