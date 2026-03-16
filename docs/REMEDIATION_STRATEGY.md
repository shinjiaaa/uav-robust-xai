# 비단조 CAM 패턴 보완 전략

**목표**: transient instability를 줄이고, 남는 불안정성을 생성·측정·해석 세 층위에서 분리해 주 결과를 오염시키지 않는다.

**세 층위**: 생성 안정화 → 측정 안정화 → 해석 분리

---

## 1. 전체 보완 전략

| 층위 | 내용 |
|------|------|
| **1.1 설명 생성** | Grad-CAM이 입력에 따라 갑자기 0이 되는 현상 감소 |
| **1.2 ROI/매칭/좌표** | CAM은 있는데 잘라내기·박스 어긋남으로 0처럼 보이는 현상 감소 |
| **1.3 분석** | 남는 불안정성을 별도 패턴으로 분리해 주 결과 보호 |

---

## 2. 설명 생성 단계 보완

### 2.1 다중 설명 평균

- **원인**: 기울기 기반 설명의 순간적 불안정성.
- **권장**: Grad-CAM + Grad-CAM++ + Eigen-CAM (보조), Score-CAM 일부 검증.
- **적용안**:
  - 기본 분석은 Grad-CAM 유지.
  - 동일 샘플에 Grad-CAM++·Eigen-CAM 추가 계산.
  - **2개 이상**에서 breakdown → “강한 붕괴”.
  - Grad-CAM에서만 flat → “설명 기법 특이 불안정성”으로 분류.

### 2.2 TTA 기반 CAM 평균화

- 원본 + 밝기 ±5%, 아주 약한 가우시안 블러, 1~2px 이동 등 4~8개 버전에서 CAM 구한 뒤 **평균 CAM** 사용.
- 주의: 매우 약한 변형만 사용.

### 2.3 기울기 0 방지용 스무딩

- noise tunnel: 입력에 아주 작은 노이즈 여러 번 → 평균 CAM.
- gradient averaging, feature map 평균화 등.

### 2.4 레이어 선택 재검토

- 작은 객체는 **공간 해상도가 높은(얕은) 레이어** 우선.
- 동일 샘플 100개로 layer A/B/C별 transient 비율 비교 후 채택.

---

## 3. ROI 처리 보완

### 3.1 Soft ROI weighting

- **현재**: bbox 밖 = 0 (hard mask).
- **개선**: bbox 내 1.0, 주변 margin 10~20%는 0.3~0.7, 그 밖 0.
- 경계 근처 활성화가 완전히 사라지지 않도록 함.

### 3.2 Bbox margin 확장 실험

- 원본 bbox / 1.1배 / 1.25배 세 버전으로 E_bbox 등 계산.
- 확장 시 transient가 크게 줄면 → ROI 민감성 문제 근거.

### 3.3 Raw CAM과 ROI CAM 동시 저장

- **저장**: full-image CAM sum, ROI-masked sum, full/ROI entropy, peak location.
- **구분**: “진짜 CAM 없음”(전체 0) vs “ROI 안에만 없음”(전체는 있음).

### 3.4 GT 기준 vs 예측 박스 기준 동시 비교

- GT bbox 기준 E_bbox, matched prediction bbox 기준 E_bbox 둘 다 계산.
- GT만 0이고 prediction 기준 높음 → attention은 있었으나 위치 밀림.
- 둘 다 0 → 진짜 설명 붕괴 가능성.

---

## 4. 품질 게이트 보완

### 4.0 Explanation quality gate (CAM failure ≠ model behavior)

- **원칙**: CAM failure를 필터링해야 한다. "설명 측정 실패"와 "모델 행동"을 구분.
- **Invalid CAM 정의**: `cam_sum < 1e-8` 또는 `spread == 0` → 해당 레코드는 **invalid**로 표시.
- **구현**:
  - `cam_valid`: 레코드에 `cam_valid = (cam_sum >= 1e-8 and spread > 0)` 저장 (05, cam_records).
  - 패턴 분류(09) 시 옵션 `pattern_classification.use_quality_gate: true`면 `cam_valid==True`인 행만 사용.
- **논문 관례**: cam_sum > threshold, spread > threshold, finite_ratio > threshold를 만족하는 CAM만 분석하는 **explanation quality gate**를 두는 경우가 많음.

### 4.1 Flat 재시도

- flat 나오면: 같은 입력 재계산, 다른 레이어 1회, soft ROI 버전 계산.
- **세 번 모두 flat**일 때만 최종 breakdown; 그 외는 transient explanation failure 후보.

### 4.2 Breakdown 정의 다중 지표화

- **2개 이상**일 때만 breakdown:
  - ROI E_bbox ≤ 0.05
  - spread ≤ 매우 작은 값
  - cam_sum ≤ epsilon
  - cam_quality = flat
  - full-image CAM도 거의 0
  - 다른 설명 방식에서도 flat

### 4.3 연속성 제약 (weak vs strong)

- **weak breakdown**: 한 단계에서만 flat.
- **strong breakdown**: 두 단계 이상 연속 또는 이후 유지.
- 논문 주 결과는 strong 위주; weak는 부록/한계.

---

## 5. 작은 객체 전용 보완

### 5.1 Tiny object subgroup

- bbox 짧은 변 50~60 / 60~80 / 80 이상 별 subgroup.
- 가장 작은 구간에 transient가 몰리면 → 설명 기법이 극소 객체에 약함 근거.

### 5.2 문맥 포함 지표

- inner box energy, expanded box energy, **ring energy**(박스 주변 띠) 저장.
- “bbox 안은 0인데 주변에 반응” 케이스 포착.

---

## 6. 분석 단계 보완

### 6.1 4분류 공식화

- stable / transient_instability / persistent_collapse / oscillatory.
- stable·persistent = 주 분석; transient·oscillatory = 설명 신뢰성 분석.

### 6.2 주 결과 두 버전 제시

- **Version A**: 전체 샘플.
- **Version B**: transient_instability 제외.
- 비슷하면 주 결론 견고; 크게 다르면 CAM 조기 경고가 설명 불안정성 영향 큼.

### 6.3 Transient 비율 보고 지표화

- 전체·corruption별·객체 크기별·layer별·CAM 방식별 transient 비율.
- GT ROI vs expanded ROI 시 transient 감소율.

---

## 7. 구현 우선순위

| 단계 | 내용 |
|------|------|
| **1단계: 원인 분리** | raw CAM 저장, GT/expanded/prediction ROI 비교, transient 재집계 → “ROI vs 진짜 flat” 확인 |
| **2단계: 계산 안정화** | TTA 평균 CAM, flat 재시도, 레이어 비교 → transient 비율 감소 확인 |
| **3단계: 분석 체계화** | strong/weak breakdown, transient 제외/포함 lead, 논문용 표·그림 |

---

## 8. 추천 실험 매트릭스

| 실험 | 변인 | 비교 지표 |
|------|------|-----------|
| **A. ROI 민감도** | hard ROI / soft ROI / bbox 1.1x / 1.25x | transient·persistent 비율, lead rate |
| **B. 설명 방식** | Grad-CAM / Grad-CAM++ / Eigen-CAM | flat 비율, transient 비율 |
| **C. 레이어** | 얕은 / 중간 / 깊은 층 | tiny E_bbox 안정성, transient 비율 |
| **D. 재시도/평균화** | single / 4-view TTA / 8-view TTA | flat 감소율, 계산 비용 |

---

## 9. 구현 상태 (코드 반영)

| 항목 | 상태 | 비고 |
|------|------|------|
| **4.0 Explanation quality gate** | ✅ | `cam_valid` (cam_sum≥1e-8, spread>0) 레코드 저장. 09에서 `use_quality_gate` 옵션으로 필터 |
| **3.2 Bbox margin 확장** | ✅ | `energy_in_bbox_1_1x`, `energy_in_bbox_1_25x` 추가. **기본 사용**: 09·뷰어는 E_bbox_1.25x 기준 (ROI miss 감소) |
| **3.3 Raw CAM 저장** | ✅ | `gradcam.save_raw_cam: true` 시 full CAM `.npy` + 전체 이미지 오버레이 `_full.png` 저장 (진짜 CAM 없음 vs ROI 구분) |
| **3.3 Raw vs ROI 지표** | ✅ 부분 | `full_cam_sum`, `full_cam_entropy` 저장 (전용 full-image run 시 차이 기록 가능) |
| **6.1 4분류** | ✅ | 09에서 stable / transient_instability / persistent_collapse / oscillatory |
| **6.2 주 결과 두 버전** | ✅ | lead_comparison_with_without_transient.json (transient 제외 lead) |
| **6.3 Transient 비율** | ✅ | pattern_counts.json, expansion_comparison(원본 vs 1.1x vs 1.25x) |

**권장 수정 조합 (측정 안정화)**  
1. CAM quality filter: `cam_valid` 사용 (이미 저장됨). 09에서 `use_quality_gate: true` 시 품질 통과만 분석.  
2. E_bbox_1.25x 사용: 09·Heatmap 뷰어 기본 적용됨.  
3. Raw CAM 저장: `configs/experiment.yaml`에서 `gradcam.save_raw_cam: true` 후 05 재실행.  
4. Transient 필터: 이상적 추세만 보기·lead 비교에서 transient 제외 옵션 있음.

*위 전략의 나머지(TTA, 다중 CAM, flat 재시도 등)는 단계별 확장 시 참고.*
