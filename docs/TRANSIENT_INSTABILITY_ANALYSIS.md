# Severity 증가에 따라 “중간에만 CAM이 붕괴했다가 회복하는” 비단조 패턴 원인 분석

## 1. 현상 요약

실험에서 **transient instability** 패턴(약 297건)이 관찰된다.  
즉, severity가 0→1→2→3→4로 단조 증가하는데, **어느 중간 단계(주로 L1 또는 L2)에서만** CAM이 breakdown(E_bbox≈0, spread≈0, cam_quality=flat)했다가, **그 다음 단계에서 다시 정상**으로 돌아온다.

예시 (cam_records.csv 기준):

| object_id | corruption | L0 | L1 | L2 | L3 | L4 |
|-----------|------------|----|----|----|----|-----|
| 0000001_02999_d_0000005_obj_8 | fog | high, E_bbox=1 | high | **flat, E_bbox=0, spread=0** | high | high |
| 0000213_05745_d_0000247_obj_4 | fog | high | **flat, E_bbox=0** | high | high | high |

breakdown 단계에서는 **cam_sum=0, activation_spread=0, energy_in_bbox=0**이며, **cam_quality가 'flat'**으로 기록된다. 즉 “CAM이 전혀 활성화되지 않은 상태”가 **그 severity에서만** 발생한다.

---

## 2. 원인 분석 (네 가지 관점)

### 2.1 설명 기법(Grad-CAM) 자체의 불안정성

- Grad-CAM은 **기울기 기반** 설명이다.  
  특정 입력(해당 severity 이미지)에서  
  - 기울기가 거의 **0**이 되거나  
  - 해당 레이어의 feature map이 **포화/절단**되면  
  CAM이 전부 0에 가깝게 나올 수 있다.
- **중간 severity에서만** 그런 이유는,  
  - L0는 원본이라 gradient가 잘 흐르고,  
  - L3/L4는 변조가 강해 다른 경로로 gradient가 흐르거나,  
  - **L1/L2** 구간에서만 **gradient가 소실되거나 채널이 포화**하는 조합이 생기기 때문이다.
- 즉 “모델이 그 단계에서 객체를 안 본다”기보다, **그 입력에서만 기울기/정규화 때문에 설명 맵이 사라진다**고 보는 것이 타당하다.

### 2.2 ROI(bbox) 마스킹/크롭 방식의 영향

- 현재 설정은 **use_bbox_roi = true**: bbox 밖을 0으로 잘라낸 CAM만 사용한다.
- 이때:
  - 원래 활성화가 **bbox 경계 근처**에만 있으면, ROI 자르면서 대부분이 잘려 나가 **거의 0**처럼 보일 수 있고,
  - 작은 객체는 bbox가 조금만 어긋나도 “bbox 안 에너지”가 급격히 줄어들 수 있으며,
  - 단계별로 letterbox/해상도가 같아도, **그 단계에서만** 예측이 약간 틀어져 reference bbox와 활성화 위치가 어긋나면, ROI 안에서는 0에 가깝게 나올 수 있다.
- 따라서 “실제로 주의가 사라졌다”기보다 **“ROI 처리 후 수치가 그 단계에서만 급락했다”** 가능성이 있다.

### 2.3 작은 객체 + 변조의 비단조적 효과

- 작은 객체는 원래 **시각적 신호가 약해** CAM이 불안정하기 쉽다.
- fog/lowlight/motion_blur는 **severity가 올라간다고 해서 난이도가 선형으로만 올라가지는 않는다**.
  - 어떤 중간 단계에서는 **대비나 edge가 일시적으로 더 뚜렷**해질 수 있고,
  - 다른 단계에서는 **다른 시각 단서(배경 구조물, 윤곽선 등)**가 더 강하게 잡혀서,  
    그 단계에서만 “객체 bbox 안” 활성화가 줄어들 수 있다.
- 그래서 **중간 severity에서만** CAM이 일시적으로 붕괴했다가, 더 높은 단계에서 다시 객체 쪽으로 활성화가 모이는 **비단조**가 나올 수 있다.

### 2.4 품질/수치 이슈 (데이터로 확인된 부분)

- cam_records 상 breakdown 구간은 거의 항상 **cam_quality = 'flat'** 이며, **cam_sum = 0, spread = 0, E_bbox = 0**이다.
- 즉 “일부 채널만 약해진다”가 아니라 **전체 CAM이 0**이다.  
  → **gradient vanishing**, **ReLU/정규화로 인한 0 출력**, 또는 **해당 severity forward pass에서의 수치적 예외**가 한 번에 발생한 상황으로 해석할 수 있다.
- 같은 (object_id, corruption) 시계열에서 **다른 severity에서는 high**가 나오므로,  
  “모델이 해당 객체를 아예 못 보는 지속적 붕괴”라기보다 **그 입력(이미지)에서만 설명이 생성에 실패한 경우**에 가깝다.

---

## 3. 정리 및 논문에서의 해석

- **현상**: severity가 올라가는데 **중간 단계에서만** CAM이 flat(0)이 되었다가, 이후 단계에서 다시 회복한다.
- **가능한 원인** (상호 배타적이 아니라 공존 가능):
  1. **Grad-CAM의 기울기 기반 불안정성**: 특정 severity 입력에서만 gradient 소실/포화.
  2. **ROI 처리**: bbox 크롭/마스킹 때문에 “그 단계에서만” bbox 안 에너지가 0에 가깝게 나옴.
  3. **작은 객체 + 변조의 비단조성**: 중간 severity에서만 시각 단서가 달라져서 bbox 내 활성화가 일시적으로 사라짐.
  4. **설명 품질/수치 이슈**: 해당 severity에서만 CAM 생성이 실패(flat, cam_sum=0).

이를 **“모델 주의의 진짜 붕괴”만으로 해석하기보다**,  
**persistent collapse(끝까지 회복 없음)** 와 **transient instability(중간에만 붕괴 후 회복)** 를 구분하고,  
후자는 “설명 기법·ROI·품질·작은 객체 특성에 따른 설명 불안정성”일 가능성을 Discussion에서 명시하는 것이 좋다.  
(실험 요약 문서의 권장 프레이밍과 동일.)

---

## 4. 데이터 상의 근거 (요약)

- **Breakdown 구간**: `cam_quality = 'flat'`, `cam_sum = 0`, `activation_spread = 0`, `energy_in_bbox = 0`.
- **회복 구간**: 동일 object_id × corruption에서 다른 severity에서는 `cam_quality = 'high'`, E_bbox ≈ 1, spread > 0.
- **패턴**: transient_instability는 세 변조(fog, lowlight, motion_blur)에 고르게 분포하며, 특정 변조에만 몰려 있지 않다.
- 따라서 “그 severity의 입력/경로에서만 설명이 실패한다”는 가설이 데이터와 부합한다.
