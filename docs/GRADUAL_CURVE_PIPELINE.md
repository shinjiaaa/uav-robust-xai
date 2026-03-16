# Gradual degradation curve pipeline (tiny object)

## Goal

Grad-CAM-based metrics should change **continuously** with severity (smooth curve), not bimodal 0/1.

---

## CAM 적용 방식 (권장 vs 비권장)

### 권장: 전체 이미지 CAM + 후처리로 bbox 지표

1. **전체 이미지**에 대해 CAM 생성  
2. CAM 품질 체크 (flat 여부, cam_sum, spread 등)  
3. **후처리**로 bbox/ROI 지표 계산:
   - bbox_center_activation_distance, peak_bbox_distance  
   - energy_in_bbox, energy_in_bbox_1.25x, soft ROI energy (필요 시 ring energy)

이렇게 해야 **attention이 bbox 밖으로 이동하는 현상**도 보이고, distance·E_bbox 해석이 의미 있게 유지됩니다.

### 비권장

- **처음부터 bbox에만 CAM을 씌우는 방식**  
- **bbox 밖을 미리 0으로 잘라놓고 CAM을 해석하는 방식**

이유: “severity가 커질수록 attention이 어떻게 바뀌는가”를 보려면, bbox로 먼저 잘라버리면 **관측 자유도가 사라짐** → E_bbox≈1, distance 변화 미약은 모델이 잘 봐서가 아니라 **측정 구조** 때문일 수 있음.

### 지표 역할 정리

| 구분 | 역할 |
|------|------|
| **전체 CAM 기반** | 어디를 보는가, 얼마나 이동했는가, 얼마나 퍼졌는가 |
| **bbox 기반 지표** | 객체에 얼마나 집중했는가, ROI 민감도, corruption이 객체 내부 집중도를 약화시키는가 |

→ CAM은 **전체 이미지**에서 만들고, **bbox는 측정 창**으로만 사용.

### 최종 추천 구조

- **CAM 생성**: full-image CAM (`use_bbox_roi: false`)
- **주지표**: bbox_center_activation_distance, peak_bbox_distance, entropy, activation_spread
- **보조지표**: energy_in_bbox, energy_in_bbox_1.25x, **ring_energy_ratio** (bbox vs ring; >0.5 객체 중심, <0.5 context)
- **참고용**: invalid CAM ratio

---

## Cause of flat/bimodal results

- **Tiny object** + **low-resolution CAM** + **bbox energy ratio (E_bbox)**  
  → blob inside bbox ≈ 1, outside ≈ 0 → **bimodal distribution** → mean does not give a smooth curve.

## Changes applied

### 1. CAM target layer (most impactful)

- **model.9**: deep, ~20×20 FM → bimodal 0/1 for tiny objects.
- **model.6**: too shallow → CAM spreads over whole map → E_bbox=1.0, distance flat (CoM ≈ center).
- **model.8** (current): sweet spot between 9 and 6; aim for gradual distance/entropy curves.

Config: `configs/experiment.yaml` → `gradcam.layers.primary.name` = `model.8.cv2.conv`.

### 2. Primary metrics: distance instead of E_bbox

- **E_bbox_1.25x** remains as **auxiliary** (structurally 0/1 for tiny objects).
- **Primary metrics** for analysis and UI:
  - **bbox_center_activation_distance** (recommended): distance(CAM center of mass, bbox center) — always continuous.
  - **peak_bbox_distance**: distance(CAM peak, bbox center) — also continuous and stable.
  - **entropy**, **activation_spread** — continuous.

### 3. New metric: peak_bbox_distance

- Implemented in `src/xai/cam_metrics.py` (`compute_peak_bbox_distance`).
- Stored in `cam_records.csv` as `peak_bbox_distance`.
- Shown in app aggregate views.

### 4. Ring Energy Ratio (E_ring_ratio)

- **정의**: ring = (1.25× bbox) \\ bbox (확장 영역에서 bbox 제외).  
  **E_ring_ratio = energy_bbox / (energy_bbox + energy_ring)**.
- **해석**: **>0.5** → 객체 중심 집중, **≈0.5** → 주변과 비슷, **<0.5** → context 중심.
- tiny object에서 E_bbox는 전체 대비라 값이 거의 항상 작음; bbox vs **주변(ring)** 비교로 0~1에서 잘 분포하고 context shift 포착에 유리.
- `src/xai/cam_metrics.py` → `compute_ring_energy_ratio`, CSV 컬럼 `ring_energy_ratio`, 앱 집계/샘플 메트릭에 포함.

### 5. Optional (already available)

- **Grad-CAM++ (FastCAM)** in `configs/experiment.yaml`: `gradcam.xai_methods: ["gradcam", "fastcam"]` — broader activation, less extreme blob.
- **Multi-scale CAM** (e.g. primary + secondary layers and average) can be added later for further smoothing.

## Why E_bbox = 1.0 when bbox-only CAM is used

With **use_bbox_roi: true**, Grad-CAM is run on the **bbox crop** only. The "bbox" in CAM coordinates is then the **entire map**, so:

- `E_bbox = sum(CAM in bbox) / sum(CAM)` = **1.0** always.
- Distance metrics are computed on a map that has no outside region → **attention leakage** and real movement are invisible.

Hence **full-image CAM** is recommended; bbox is used only as a measurement window for post-processing metrics.

## Expected result after re-run (model.8)

- **distance ↑** with severity  
- **entropy ↑**, **spread** may increase  
- With **use_bbox_roi: false**, E_bbox reflects real object focus; bbox_dist / peak_dist remain primary for gradual curves.

## Re-running the pipeline

1. (Optional) Clear or backup `results/cam_records.csv` and `results/heatmap_samples/`.
2. Run: `python scripts/05_gradcam_failure_analysis.py` (config: `model.8.cv2.conv`).
3. Check **변조별 전체 통계** in the app: bbox_dist / peak_dist should show gradual trends across L0–L4.

---

## 실행 결과 예상 (Expected run outcomes)

### 1. 05 스크립트 실행 시

| 상황 | 예상 |
|------|------|
| **정상** | `model.8.cv2.conv`가 존재하면 CAM 추출/저장/메트릭 완료. model.9보다 완만, model.6보다 blob 구조 유지 기대. |
| **레이어 오류** | `Activations not captured` 등 → 해당 경로 없음. 후보: `model.9.cv2.conv`, `model.8.cv2.conv`, `model.7.cv2.conv`. |
| **출력** | `cam_records.csv`에 **peak_bbox_distance** 포함. bbox_dist/peak_dist가 severity에 따라 변화하는지 확인. |

### 2. cam_records.csv / 지표 분포

- **bbox_center_activation_distance (bbox_dist)**  
  - 이전: 0 또는 1에 가까운 값 많음 (깊은 레이어 + tiny bbox → blob이 bbox 안/밖만).  
  - 변경 후 예상: 0.002, 0.005, 0.01, 0.02 … 처럼 **0~0.1 구간에서 연속적으로 퍼진 값** 증가. severity가 올라갈수록 **평균/중앙값이 서서히 증가**하는 경향 가능.
- **peak_bbox_distance (peak_dist)**  
  - 새 지표. peak 한 점과 bbox 중심 거리라 **항상 연속값**. severity 증가에 따라 **조금씩 증가**하는 곡선이 나올 가능성 높음.
- **energy_in_bbox_1_25x (E_bbox)**  
  - 구조적 한계(blob inside/outside)는 그대로라 **여전히 0/1에 가까운 bimodal** 가능성 있음. 주 지표를 distance로 바꿨으므로 **참고용**으로만 해석.

### 3. 앱 변조별 전체 통계 (그래프)

- **bbox_dist, peak_dist**  
  - L0 → L1 → … → L4로 갈수록 **우상향**하는 곡선이 나올 가능성이 큼.  
  - 변조(fog, motion_blur, lowlight)마다 기울기나 절편은 다를 수 있음.
- **entropy**  
  - 노이즈/붕괴가 심해질수록 **전반적으로 증가**하는 추세 가능.
- **activation_spread**  
  - 활성화가 퍼지면 **증가**하는 방향일 가능성.
- **E_bbox_1.25x**  
  - 이전처럼 **평탄하거나 0/1 사이에서만 움직이는** 형태일 수 있음. 주 지표가 아니므로 무시해도 됨.

### 4. 실패 시 체크리스트

- 05 실행 중 **레이어 관련 에러** → `gradcam.layers.primary.name`을 YOLO에 맞게 수정 (후보: model.9, model.8, model.7).
- **peak_bbox_distance**가 CSV/앱에 안 보임 → 05를 변경 후 **한 번이라도 다시 실행**했는지 확인 (기존 CSV에는 컬럼이 없음).
- **bbox_dist가 여전히 평평** → 레이어를 **model.7** 또는 **model.9**로 바꿔 비교 (model.8이 과도하면 9, 아직 평평하면 7).
- **E_bbox = 1.0 전부** → bbox-only CAM 사용 시 정상; 권장은 full-image CAM(`use_bbox_roi: false`) + bbox_dist / peak_dist 주 지표.
