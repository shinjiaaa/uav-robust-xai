# DASC 실험 종합 정리

## 1. 연구 목적 및 실험 구조

### 1.1 연구 질문 (RQ)
**객체 탐지 모델의 성능 저하가 일어나기 전에, XAI(Grad-CAM)가 성능 저하 전조를 잡을 수 있는가?**

- **정량**: IoU/mAP 곡선, 모델 저하 단계 vs Grad-CAM 붕괴 단계 비교, 변조별 일관성
- **정성**: Heatmap 시각화를 통한 인지적 도움 평가

### 1.2 실험 축
| 축 | 데이터 소스 | 내용 |
|----|-------------|------|
| **Performance Axis** | detection_records.csv, risk_events.csv | 언제 성능이 저하하는지 (miss, score drop, IoU drop) |
| **Cognition Axis** | cam_records.csv | 내부 주의/인지(CAM)가 어떻게 변하는지 (energy_in_bbox, entropy 등) |

**정렬(Alignment)**: 두 축을 severity(시간) 기준으로 맞춰 **Lead**(CAM이 먼저 변함) / **Coincident**(동시) / **Lag**(성능이 먼저 변함)를 산출.

### 1.3 변조 및 단계
- **변조**: fog, lowlight, motion_blur (3종)
- **Severity**: 0(원본) ~ 4 (5단계, 표준)
- **모델**: yolo_generic (YOLOv8s, COCO pretrained, DASC 기본)

---

## 2. 실험에서 사용한 정량적 기준

### 2.1 Tiny Object 선정 기준 (bbox 선정의 전제)

**목적**: “한 개의 작은 객체”만 분석 단위로 두기 위해, **어떤 GT 박스를 tiny object로 쓸지** 정하는 기준.

| 기준 | 설정값 | 설명 |
|------|--------|------|
| **area_threshold** | 2500 px² | 최소 넓이. 50×50 픽셀 수준, YOLO 탐지율 약 63.8% 목표 |
| **width_threshold** | 50 px | 가로 **또는** 세로 중 최소 하나가 50px 이상 |
| **height_threshold** | 50 px | (동일) |
| **max_area_threshold** | 20000 px² | 최대 넓이. 이보다 크면 “작은 객체”에서 제외 |

**판정식** (01_sample_tiny_objects.py, VisDrone 어노테이션 기준):
- `area_ok` = (bbox_width × bbox_height) ≥ 2500  
- `size_ok` = (bbox_width ≥ 50 **or** bbox_height ≥ 50)  
- `area_not_too_large` = 넓이 ≤ 20000  
- **is_tiny** = area_ok **and** size_ok **and** area_not_too_large  
- object_category == 0(ignored) 인 것은 제외  

→ **넓이 2500~20000 px² 이고, 한 변이 50px 이상인 GT 박스**만 tiny object로 사용.

---

### 2.2 GT–예측 매칭 기준 (탐지 성능·이벤트 정의)

| 기준 | 설정값 | 설명 |
|------|--------|------|
| **tiny_match_iou_threshold** | 0.3 | GT와 예측 bbox **IoU ≥ 0.3**이면 매칭(탐지 성공). 소형 객체라 0.5보다 완화 |
| **tiny_match_same_class** | false | COCO pretrained와 VisDrone 클래스 불일치 허용, 클래스 무관 매칭 |
| **inference.conf_thres** | 0.01 | 탐지 confidence 임계값 (소형 객체 검출을 위해 낮춤) |
| **inference.iou_thres** | 0.45 | NMS용 IoU 임계값 |

→ **IoU 0.3 이상**이면 “해당 tiny object를 맞춤”으로 간주.

---

### 2.3 성능 저하·리스크 구간 정의

| 기준 | 설정값 | 설명 |
|------|--------|------|
| **model_degradation_threshold** | 0.25 | miss_rate ≥ 25%인 severity를 “성능 저하 단계”로 봄 |
| **miss_rate_threshold** | 0.25 | 리스크 구간 판정 시 사용하는 miss rate 임계값 |
| **map_drop_threshold** | 0.15 | mAP가 severity 0 대비 15%p 이상 떨어지면 저하로 봄 |
| **score_drop_threshold** | 0.2 | 점수 절대 하락 0.2 이상 |
| **score_drop_ratio** | 0.9 | score(sev) / score0 < 0.9 이면 상대적 score drop |
| **iou_drop_threshold** | 0.2 | IoU 절대 하락 0.2 이상 |
| **score_drop_relative_threshold** | 0.15 | 기준 대비 15% 점수 하락 |
| **iou_drop_relative_threshold** | 0.15 | 기준 대비 15% IoU 하락 |
| **instability_threshold** | 0.3 | severity 간 분산(불안정성) 임계값 |

**Z-score 기반 성능 붕괴** (risk_detection):
- **use_z_score_performance**: true  
- **perf_collapse_z_threshold**: -2.0  
- severity 0 대비 score_drop_z 또는 iou_drop_z **< -2.0**이면 “성능 붕괴”로 정의  

---

### 2.4 CAM 붕괴 정의 (선행 분석용)

| 기준 | 설정값 | 설명 |
|------|--------|------|
| **gradcam_breakdown_threshold** | 0.3 | energy_in_bbox가 baseline 대비 30% 이상 감소 시 “패턴 붕괴” (레거시) |
| **cam_z_threshold** | 2.0 | CAM 메트릭 z-score \|z\| ≥ 2.0 이면 “CAM 붕괴” |
| **perf_z_threshold** | -2.0 | 성능 쪽 z-score < -2.0 이면 “성능 붕괴” |
| **cam_metrics** | energy_in_bbox, activation_spread, entropy, center_shift | 붕괴 판정에 쓰는 메트릭 |
| **min_metrics_for_ensemble** | 2 | 위 메트릭 중 2개 이상이 임계값을 넘어야 앙상블 붕괴로 인정 |

---

### 2.5 Grad-CAM·품질 관련

| 기준 | 설정값 | 설명 |
|------|--------|------|
| **use_bbox_roi** | true | Grad-CAM을 **해당 tiny object bbox 크롭**에만 적용, bbox 밖은 0 |
| **quality_gate.finite_ratio_threshold** | 0.99 | CAM 값의 99% 이상이 유한수여야 통과 |
| **quality_gate.cam_sum_epsilon** | 1e-8 | CAM 합이 이보다 크면 “빈 CAM”이 아님 |
| **quality_gate.cam_var_epsilon** | 1e-10 | 분산 하한 |

**Explanation quality gate (CAM failure ≠ model behavior)**  
- 각 레코드에 **cam_valid** 저장: `cam_sum >= 1e-8` 이고 `activation_spread > 0`일 때만 True.  
- 패턴 분류(09)에서 `pattern_classification.use_quality_gate: true`면 cam_valid==True인 행만 사용해, “설명 측정 실패”를 주 분석에서 제외할 수 있음.  
- 주 분석·뷰어에서는 **E_bbox_1.25x**를 기본 사용(ROI miss 감소). Raw CAM 저장은 `gradcam.save_raw_cam: true` 시 `results/raw_cam/`에 .npy 및 전체 오버레이 저장.

---

## 3. 산출 가능한 지표 전체 목록

### 3.1 파일별 산출물

| 파일 | 산출 주체 | 포함 지표/내용 |
|------|------------|----------------|
| **tiny_objects_samples.json** | 01_sample_tiny_objects.py | 샘플된 tiny object 목록 (image_id, bbox, class_id, frame_path 등) |
| **tiny_curves.csv** | 03_detect_tiny_objects_timeseries.py | 변조·severity별 tiny 곡선 (miss_rate 등) |
| **tiny_records.csv** | 03_detect_tiny_objects_timeseries.py | 객체·프레임별 탐지 기록 |
| **detection_records.csv** | 03_detect_tiny_objects_timeseries.py | 객체·severity별 매칭, score, IoU, miss |
| **risk_events.csv** | 04_detect_risk_events.py | 리스크 이벤트 (object_uid, cam_sev_from, cam_sev_to, start_severity 등) |
| **failure_events.csv** | 04_detect_failure_events.py | (레거시) 실패 이벤트 |
| **cam_records.csv** | 05_gradcam_failure_analysis.py | CAM 메트릭 (energy_in_bbox, energy_in_bbox_1_25x, activation_spread, entropy, cam_quality, cam_valid 등) |
| **gradcam_metrics.csv** | 05_gradcam_failure_analysis.py | (요약) Grad-CAM 관련 메트릭 |
| **lead_table.csv** | 07_lead_analysis.py | 객체별 t_cam, t_perf, lead (프레임/단계) |
| **lead_stats.json** | 07_lead_analysis.py | 선행 통계 (n_lead, n_coincident, n_lag, mean_lead, sign_test, permutation_test) |
| **dasc_summary.json** | 08_dasc_deliverables.py | iou_curve, miss_rate_curve, model_degradation_stage, gradcam_breakdown_stage, consistency 등 |
| **report.md** | 06_llm_report.py | 성능 테이블, CAM 테이블, 정렬 요약, 샘플 이벤트 |
| **heatmap_samples/** | 05_gradcam_failure_analysis.py | L0~L4별 heatmap PNG (전체 이미지 + bbox 안에만 heatmap) |

### 3.2 주요 지표 (해석·보고용)

| 분류 | 지표 | 의미 |
|------|------|------|
| **선행 분석** | n_lead, n_coincident, n_lag, n_total | CAM이 먼저 변한 건수 / 동시 / 나중 / 전체 |
| | 선행률 (Lead %) | n_lead / (n_lead + n_coincident + n_lag) |
| | 선행+동시 비율 | (n_lead + n_coincident) / n_total |
| | mean_lead | 선행 시 평균 lead(단계) |
| | Sign test p-value, Permutation test p-value | 선행 효과의 통계적 유의성 |
| **성능** | Miss rate (변조·severity별) | 해당 단계에서의 미탐지 비율 |
| | Mean IoU, n_matched | 매칭된 객체 평균 IoU, 매칭 수 |
| | model_degradation_stage | 성능이 임계값을 넘어 저하되는 severity |
| **CAM** | Energy in Bbox, Activation spread, Entropy, Center shift | 내부 주의/분포 특성 |
| | activation_fragmentation, bbox_center_activation_distance | 소형 객체용 구조 메트릭 |
| **정렬** | alignment (lead/coincident/lag), lead_steps | 이벤트별 선행/동시/지연 및 단계 차이 |

---

## 4. 의미 있는 결과 해석 (예시)

### 4.1 Lead 통계 (lead_stats.json 기준)

- **n_lead=485, n_coincident=428, n_lag=0**  
  CAM이 성능보다 **먼저** 변한 경우가 약 53%, **동시**가 나머지에 가깝고, **지연**은 0에 가깝다.  
  → “성능 저하 전에 CAM이 먼저 또는 동시에 변한다”는 가설과 부합.

- **선행+동시 비율 ≈ 91%**  
  대부분의 이벤트에서 CAM이 성능보다 늦게만 변하는 구간은 거의 없다.

- **Sign test p ≈ 0.032, Permutation p = 0.0**  
  선행 효과가 우연으로 보기 어렵고, 순열검정에서도 유의.  
  → **CAM이 성능 저하 전조 지표로 쓰일 가능성**을 정량적으로 뒷받침.

### 4.2 변조별 성능 (dasc_summary / report)

- **fog**: L4에서 miss rate 80.6%, mean IoU 0.76 수준. 변조가 강해 고심도에서 급격한 저하.
- **lowlight**: L4 miss 59.8%. fog보다는 완만하지만 저조도에서도 저하 뚜렷.
- **motion_blur**: L4 miss 21%. 상대적으로 변조 영향이 작고, 소형 객체도 일부 유지.

→ 변조 유형별로 “어느 severity부터 문제인지”를 정리하면, 실무 적용 시 변조 강도·한계 설정에 활용 가능.

### 4.3 CAM 메트릭 (report.md Table 2)

- **Energy in Bbox, Activation spread, Entropy**가 severity가 올라가며 변하는 패턴을 보임.  
  → “어디를 보는가”(에너지 분포), “얼마나 퍼져 있는가”(spread), “무질서도”(entropy)가 변조에 따라 변함을 보여줌.
- **cam_valid_ratio**가 고심도에서 떨어지는 구간이 있음.  
  → 해당 구간에서는 CAM 품질/해석에 주의가 필요함을 의미.

### 4.4 Heatmap 시각화

- **전체 이미지 + bbox 안에만 heatmap**:  
  “어느 장면의 어느 객체에 대한 설명인지” 맥락이 유지됨.  
- **L0~L4가 모두 있는 샘플만 UI에 표시**:  
  동일 객체에 대해 변조 단계별로 일관 비교 가능.

---

## 5. 화면(UI)에서 확인 가능한 것

### 5.1 Heatmap 뷰어 (python app.py → http://127.0.0.1:5000)

| 구역 | 확인 가능 내용 |
|------|----------------|
| **상단 지표** | 선행률(Lead %), 선행+동시 비율, 선행/동시/총 이벤트 건수, 평균 Lead, Sign test / Permutation p-value, 변조별 Miss rate L4 |
| **좌측** | 모델·Corruption 선택, **샘플 목록**(L0~L4 전부 존재하는 샘플만) |
| **본문** | 선택한 **한 샘플**에 대해 **L0, L1, L2, L3, L4** 이미지 한 줄 표시 |
| **이미지** | **전체 이미지** + **해당 tiny object bbox 안에만** Grad-CAM heatmap |
| **클릭** | 셀 클릭 시 해당 이미지 모달 확대 |

### 5.2 샘플 수 (L0~L4 전부 있는 경우)

- fog: 338  
- lowlight: 175  
- motion_blur: 24  
- **합계: 537** (객체 기준. 이미지 장수는 537×5)

**이 숫자의 의미 (오해 방지)**  
이 수치는 **「탐지된 샘플 수」가 아니라**, **「해당 변조에 대해 L0, L1, L2, L3, L4 heatmap이 모두 저장된 tiny-object 샘플 수」**입니다.  
UI는 L0~L4가 **전부 존재하는** 샘플만 목록에 넣고, 그 개수를 변조별로 세어 보여줍니다.

- **fog 338**: fog 변조에 대해 05 단계에서 L0~L4 heatmap 저장이 모두 성공한 샘플이 338개.
- **lowlight 175**: 동일 조건으로 175개.
- **motion_blur 24**: 동일 조건으로 24개.

따라서 **motion_blur가 24인 이유는 “모션 블러에서 탐지가 덜 돼서”가 아닙니다.**  
보고서의 “fog가 가장 심하다”는 **성능 저하**(L4에서 miss rate가 가장 높다, IoU 하락이 가장 크다)를 말하는 것이고,  
338/175/24는 **파이프라인 완결도**(그 변조·severity에 대해 heatmap이 다 저장된 샘플이 몇 개인지)를 말합니다.  
즉, **서로 다른 지표**이므로 모순이 없습니다.  
**motion_blur가 24인 이유 (실제 원인)**  
변조 이미지 자체는 **02 단계에서 모든 기본 이미지에 대해 fog/lowlight/motion_blur를 동일 조건으로** 생성합니다(동일 이미지 집합 × 3변조 × L1~L4).  
차이는 **05 단계 입력** 때문입니다. 05는 **risk_events.csv에 있는 이벤트에만** Grad-CAM을 돌리고 heatmap을 저장합니다.  
risk_events는 **04 단계**에서 “해당 (객체, 변조) 시계열에서 성능 저하(miss / score_drop / iou_drop)가 **한 번이라도 관찰된** 경우”만 넣습니다.  
motion_blur는 성능 저하가 상대적으로 적어서(보고서에서 “가장 덜 심함”) 이 기준을 만족하는 객체 수가 적고, 따라서 risk_events에 motion_blur 행이 적습니다. 그 결과 05에서 motion_blur에 대해 heatmap이 저장되는 (객체, L0~L4) 샘플이 24개뿐입니다.  
**정리**: “모든 기본 이미지에 대해 각 변조를 동일 조건으로 적용”은 **02에서 지켜짐**. **Heatmap 저장**은 설계상 “전체가 아니라 **risk 이벤트만**” 대상이라, 변조별로 risk 발생 수가 다르기 때문에 fog 338 / lowlight 175 / motion_blur 24처럼 개수 차이가 난 것입니다.  
**동일 개수로 맞추려면**: 05를 “모든 (tiny_object, corruption) 쌍”에 대해 실행하도록 바꾸거나(risk_events 무관), 별도 옵션으로 “full coverage” 모드를 두면 됩니다.

---

## 6. 실험 기준 요약 표

| 목적 | 기준 | 정량값 |
|------|------|--------|
| Tiny object 선정 | 넓이 하한 | ≥ 2500 px² |
| | 한 변 하한 | ≥ 50 px (가로 또는 세로) |
| | 넓이 상한 | ≤ 20000 px² |
| GT–예측 매칭 | IoU | ≥ 0.3 (tiny_match_iou_threshold) |
| 성능 저하 | Miss rate | ≥ 0.25 (model_degradation_threshold 등) |
| 성능 붕괴 (Z-score) | score_drop_z / iou_drop_z | < -2.0 |
| CAM 붕괴 | CAM 메트릭 z-score | \|z\| ≥ 2.0 |
| CAM 앙상블 붕괴 | 해당 메트릭 수 | ≥ 2개 |
| Grad-CAM 적용 범위 | bbox ROI | 해당 tiny object bbox 크롭에만 적용 |

---

## 7. 참고: 파이프라인 순서

1. **01_sample_tiny_objects.py** — Tiny object 샘플링 (위 2.1 기준)  
2. **02_generate_corruption_sequences.py** — 변조 이미지 L1~L4 생성  
3. **03_detect_tiny_objects_timeseries.py** — 탐지·매칭 (2.2 기준), tiny_curves/tiny_records/detection_records  
4. **04_detect_risk_events.py** — 리스크 이벤트 (2.3 기준), risk_events.csv  
5. **05_gradcam_failure_analysis.py** — CAM 추출·저장 (2.4, 2.5 기준), cam_records, heatmap_samples  
6. **06_llm_report.py** — report.md 생성  
7. **07_lead_analysis.py** — lead_table.csv, lead_stats.json  
8. **08_dasc_deliverables.py** — dasc_summary.json, dasc_curves 등  
9. **09_cam_pattern_classification.py** — CAM 시계열 패턴 분류 (stable / persistent_collapse / transient_instability / oscillatory), pattern_summary.csv, pattern_counts.json, cam_pattern_report.md, lead_comparison_with_without_transient.json  
10. **app.py** — Heatmap 뷰어 UI (5.1 내용)

---

## 8. CAM 시계열 패턴 분류 (논문용)

**목적**: severity 증가에 따라 “중간에만 CAM이 붕괴했다가 회복하는” 비단조 패턴을 정량화하고, persistent collapse와 구분해 해석하기 위함.

**Breakdown 정의 (1차 규칙)**  
- `E_bbox <= 0.05` 또는 `activation_spread <= 1e-6` 또는 `cam_quality != 'high'`

**패턴**  
- **Stable**: 모든 severity에서 breakdown 없음  
- **Persistent collapse**: 첫 breakdown 이후 더 높은 severity에서 회복 없음  
- **Transient instability**: severity 1~3에서 한 덩어리 breakdown 발생 후, 이후 단계에서 전부 회복  
- **Oscillatory**: breakdown과 회복이 두 번 이상 번갈아 나타남  

**이상적 추세 샘플 (ideal_trend_samples.json)**  
- 패턴이 stable 또는 persistent_collapse인 것 중, **L0→L4 구간에서 단조 추세**만 허용.  
- **E_bbox**: 매 단계 비증가(연속 하강). **entropy, bbox_dist, spread**: 매 단계 비감소(연속 상승).  
- 중간에 오락가락(반대로 튀는 구간)이 있으면 제외 → "측정 실패 없이 정상적으로 관측된" 연속·단조 열화만 이상적 추세로 분류.

**산출물**  
- `results/pattern_summary.csv`, `results/pattern_counts.json`  
- `results/pattern_transient_instability.csv` 등 패턴별 CSV  
- `results/ideal_trend_samples.json`: 단조 이상적 추세 샘플 ID (뷰어 "이상적 추세만 보기"에 사용)  
- `results/cam_pattern_report.md`  
- `results/lead_comparison_with_without_transient.json`: transient_instability 샘플을 제외한 lead 통계. “transient가 있어도 주 결론(선행 효과)이 유지되는지” 확인용.

**논문 권장 프레이밍**  
- 주 결과: CAM 변화가 성능 저하보다 선행하거나 동시인 경우가 많다.  
- 부가 관찰: 일부 샘플에서는 severity가 단조 증가함에도 CAM이 중간 단계에서만 일시 붕괴했다가 회복하는 비단조적 불안정성(transient instability)이 관찰된다.  
- 해석: Grad-CAM·ROI·품질 한계에서 비롯된 설명 불안정성일 가능성을 인정하고, persistent collapse와 transient instability를 구분해 해석한다.

이 문서는 위 파이프라인과 설정(configs/experiment.yaml)을 기준으로 작성되었으며, 실제 수치는 실험 실행 결과(예: results/lead_stats.json, results/report.md)에 따라 달라질 수 있습니다.
