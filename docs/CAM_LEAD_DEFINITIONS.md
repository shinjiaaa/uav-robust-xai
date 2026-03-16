# CAM 선행 분석: 개념 분리 및 기준 정의

## 0. 분석 전환: population 평균 → object-level dynamics

- **기존**: severity별 평균 CAM 곡선 → tiny object에서는 변화 시점이 객체마다 달라 평균이 평평해짐.
- **권장**: **객체별 CAM 변화 시점 분포** (onset distribution), **lead 분포**, **survival curve**.
- 연구 질문 "CAM이 성능 저하 전에 변화하는가?"는 **change timing** 문제이므로, 평균 변화량이 아니라 **객체 단위 시점 분포**로 분석하는 것이 적합함.

---

## 1. 개념 분리

다음 세 가지를 **분리**해서 정의한다.

| 개념 | 정의 |
|------|------|
| **1) CAM 변화 시작 시점 (t_cam_warn)** | 모델의 주의 구조가 baseline(L0)에서 **의미 있게 벗어나기 시작한** 시점. 민감하게 잡음 (early warning). |
| **2) CAM 붕괴 시점 (t_cam_collapse)** | 주의 구조가 단순 변화를 넘어 **객체 중심성을 잃고 지속적으로 불안정**해진 시점. 보수적으로 잡음 (특이도 중시). |
| **3) 선행 (Lead)** | CAM 변화 시작 시점 또는 CAM 붕괴 시점이 **성능 저하(t_perf)보다 앞서는가**. 선행은 붕괴/경고 정의에 종속됨. |

- **lead_warn** = t_perf - t_cam_warn (경고 선행)
- **lead_collapse** = t_perf - t_cam_collapse (붕괴 선행)

---

## 2. 4개 지표와 2축

| 축 | 지표 | 의미 |
|----|------|------|
| **A. 위치 이탈** | bbox_dist, peak_dist | 주의가 객체 중심에서 얼마나 벗어났는가 (attention displacement) |
| **B. 확산/객체 중심성** | spread, E_ring_ratio | 주의가 얼마나 퍼지는가(spread), 객체보다 주변 문맥에 얼마나 기대는가(ring_ratio) |

- **bbox_dist**: bbox_center_activation_distance  
- **peak_dist**: peak_bbox_distance  
- **spread**: activation_spread  
- **E_ring_ratio**: ring_energy_ratio (낮을수록 context 쪽)

연구 질문과의 대응:
- ① 객체 중심에서 벗어나는가 → bbox_dist, peak_dist  
- ② 더 넓게 퍼지는가 → spread  
- ③ 객체보다 주변 문맥에 더 의존하는가 → E_ring_ratio  

---

## 3. Baseline: L0 대비 변화량(Δ)

- **Baseline**: 객체별·corruption별 **L0**.
- **Δ** = value(severity) - value(L0). 절대 z-score가 아니라 **within-object change**.
- Δbbox_dist, Δpeak_dist, Δspread: 증가 = 악화.  
- Δring = E_ring_ratio(sev) - E_ring_ratio(L0): **감소(Δring < 0)** = 객체 중심성 약화(악화).

---

## 4. Threshold: 분포 기반 (분위수)

- 전체 객체의 (object, severity) 쌍에 대해 Δ·(-Δring) 분포를 구한 뒤 **분위수** 사용.
- **Warning (변화 시작)**: Q75 (상위 25%) 수준 → 민감.
- **Collapse (붕괴)**: Q90 (상위 10%) 수준 → 보수적.

설정: `collapse_detection.warning_percentile` (기본 0.75), `collapse_detection.collapse_percentile` (기본 0.90).

---

## 5. t_cam_warn (CAM 경고 시점)

**정의**: 다음 **두 축 각각에서 최소 1개씩**, 총 **2개 이상** 조건이 warning threshold(Q75)를 넘는 **첫 severity**.

- **축 A (위치 이탈)**: Δbbox_dist > T_d_warn **또는** Δpeak_dist > T_p_warn  
- **축 B (확산/객체 중심성)**: Δspread > T_s_warn **또는** Δring < -T_r_warn  

→ “의미 있는 구조 변화” = 한 축만이 아니라 **위치 이탈 + (퍼짐 또는 객체 중심성 약화)** 조합.

---

## 6. t_cam_collapse (CAM 붕괴 시점)

**정의**: 다음을 **동시에** 만족하는 **첫 severity**.

1. **4개 조건 중 3개 이상**이 collapse threshold(Q90) 초과:  
   Δbbox_dist > T_d_collapse, Δpeak_dist > T_p_collapse, Δspread > T_s_collapse, Δring < -T_r_collapse  
2. **지속성**: 그 다음 severity에서도 3개 이상 유지되거나, 다음 severity가 없음(마지막 단계).  
   → 1단계만 튀는 **transient**는 붕괴로 인정하지 않음.

---

## 7. 설정 요약 (configs/experiment.yaml)

```yaml
collapse_detection:
  use_delta_based: true
  axis_a_metrics: ["bbox_center_activation_distance", "peak_bbox_distance"]
  axis_b_metrics: ["activation_spread", "ring_energy_ratio"]
  warning_percentile: 0.75   # Q75 → t_cam_warn
  collapse_percentile: 0.90 # Q90 → t_cam_collapse
  min_conditions_collapse: 3
  collapse_require_persistence: true
```

---

## 8. 산출물

- **lead_table.csv**: object_uid, t_perf, t_cam_warn, t_cam_collapse, **lead_warn**, **lead_collapse**, t_cam(=t_cam_warn), lead(=lead_warn), alignment.
- **lead_stats.json**: lead_warn 기준 n_lead/n_coincident/n_lag, sign_test, permutation_test; optionally **lead_collapse_stats**.

선행률(Lead %) 등은 lead(또는 lead_warn) 기준으로 계산하며, 필요 시 lead_collapse 기준 통계는 `lead_collapse_stats`에서 사용.

---

## 9. Composite CAM change score (object-level 권장)

- **CAM_change_score** = z(Δbbox_dist) + z(Δpeak_dist) + z(Δspread) + z(-Δring).  
  각 z는 전체 (object, severity) 분포로 계산; 스케일 통일.
- **t_cam_change** = 해당 객체에서 `CAM_change_score ≥ threshold`가 **처음** 나오는 severity.  
  threshold = 전체 점수 분포의 **상위 25%** (예: 75th percentile).
- **lead** = t_perf - t_cam_change.

설정: `collapse_detection.cam_change_method: "composite_score"`, `score_threshold_percentile: 0.75`.

---

## 10. 핵심 시각화 (object-level)

1. **CAM change onset distribution**: x = severity (L1~L4, no_change), y = object count.  
   → CAM 변화가 어느 단계에서 가장 많이 시작되는가.
2. **Lead distribution**: x = lead, y = object count.  
   → lead > 0 비율이 바로 보임.
3. **Survival curve**: P(CAM change not yet occurred) vs severity.  
   → CAM이 성능보다 먼저 흔들리기 시작함을 시각적으로 보여줌.
4. **Per-corruption mean lead**: corruption별 mean lead.  
   → 어떤 변조에서 CAM이 가장 빨리 흔들리는지.

산출: `results/cam_onset_lead_survival.json`, `scripts/10_lead_visualization.py` → `results/figures/*.png`.
