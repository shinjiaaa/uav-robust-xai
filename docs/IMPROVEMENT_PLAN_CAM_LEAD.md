# CAM 선행(Lead) 증명 개선 계획

## 현재 결과 한 문장

- **실험 질문:** "모델이 완전히 망가지기 전에, 내부 주의(CAM)가 먼저 이상해지지 않을까?"
- **현재 결론:** "CAM이 변하긴 하는데, 성능이 망가지는 시점과 거의 동시에 변함."
  - ❌ "CAM이 먼저 붕괴한다"는 증거 없음
  - ✔ "CAM과 성능이 같이 나빠진다"는 것은 관찰됨

---

## 왜 "먼저"를 증명 못 했는가

- 실험 구조: **severity 0,1,2,3,4** 5단계만 존재.
- severity 1에서 이미 성능 저하 + CAM 변화 동시 발생 → **CAM 변화 시점 ≈ 성능 저하 시점** → "선행"이 아니라 "동시".

---

## 핵심 문제 3가지

| 문제 | 설명 |
|------|------|
| **(1) Threshold 과민** | severity 1에서 거의 모든 object가 이벤트 발생 → 선행 분석 불가 |
| **(2) Collapse 정의 없음** | 현재는 "CAM이 조금 변하면 change". 가설은 "CAM **붕괴**"인데, 붕괴의 수학적 정의 없음 |
| **(3) 시간 해상도 거침** | severity 단위만 사용. 프레임 단위 time-series 없음 |

---

## 개선 방향 (현재 상태)

- 구조(Performance Axis, Cognition Axis, Alignment 테이블)는 잘 잡혀 있음.
- **정밀도** 부족 → 아래 5단계로 보강.

---

# 🔥 단계 1: Collapse 정의 (CAM)

**현재:** "변화(change)"만 있음.

**할 일:**

1. **Baseline:** severity 0에서 각 CAM metric의 **평균(μ)** 과 **표준편차(σ)** 계산 (object별 또는 frame별).
2. **Collapse 기준 (z-score):**
   - `entropy_z = (entropy - μ) / σ`
   - `energy_z = (energy_in_bbox - μ) / σ` 등 동일
   - **예시 규칙:** `entropy_z > 2` (또는 `< -2` 등 방향 정의) → **collapse**
   - 즉, baseline 대비 **2σ 이상 벗어나야** 붕괴로 인정.

**산출물:**  
- `cam_collapse_threshold`: z_threshold (e.g. 2.0)  
- per-object 또는 per-frame `cam_collapse_severity` / `cam_collapse_frame`

---

# 🔥 단계 2: Performance 붕괴 정의 통일

**현재:** threshold가 애매함 (고정 비율/절대값).

**할 일:**

1. **Baseline:** severity 0에서 score, IoU의 평균·표준편차.
2. **붕괴 기준 (z-score):**
   - `score_drop_z = (score_sev - score_0) / std_0` (또는 Δscore 기준)
   - `IoU_drop_z` 동일
   - **예시:** `score_drop_z < -2` 또는 `IoU_drop_z < -2` → performance collapse

**산출물:**  
- config: `performance_collapse_z_threshold` (e.g. -2)  
- per-object `perf_collapse_severity` / `perf_collapse_frame`

---

# 🔥 단계 3: Frame 단위 분석 (Time-series)

**현재:** severity 1,2,3 단위만 사용.

**할 일:**

1. **Frame index 기준** object별 time series 구성 (이미지/프레임 순서 = severity 또는 실제 시퀀스).
2. **시점 정의:**
   - `t_cam`: CAM collapse가 처음 발생한 시점 (frame 또는 severity index)
   - `t_perf`: Performance collapse가 처음 발생한 시점
3. **Lead 계산:** `lead = t_perf - t_cam`  
   - lead > 0 → CAM이 먼저 붕괴 (선행)  
   - lead = 0 → 동시  
   - lead < 0 → 성능이 먼저

**산출물:**  
- object별 `t_cam`, `t_perf`, `lead`  
- 시각화: object별 time-series (CAM metric + score/IoU), lead 분포

---

# 🔥 단계 4: 통계 검정

**현재:** 단순 카운트 (Lead count, %).

**할 일:**

1. **Lead 분포:** object별 `lead` 수치 확보 후,
2. **가설:** 평균 lead > 0 (CAM이 평균적으로 먼저 붕괴).
3. **검정:**
   - **Sign test:** lead > 0인 객체 비율이 0.5보다 유의하게 큰지
   - **Permutation test:** lead 레이블 셔플 후 평균 lead 분포 구하고, 실제 평균 lead의 p-value 산출
4. **산출:** p-value, 신뢰구간 → 논문용 서술 가능하도록

**산출물:**  
- `lead_stats.json` 또는 리포트 섹션: mean_lead, std, p_value (sign / permutation), n_lead, n_coincident, n_lag

---

# 🔥 단계 5: CAM metric 정제 (Tiny object 맞춤)

**현재:** metric이 전반적으로 약하거나 tiny object에 맞지 않을 수 있음.

**할 일:**

1. **구조적 지표 강화:**
   - **bbox 내부 에너지 비율:** (bbox 내 activation 합) / (전체 activation 합)
   - **bbox 중심 ↔ activation 중심 거리:** 이미 center_shift 유사, 정규화/정의 명확화
   - **Activation fragmentation:** bbox 내에서 activation이 얼마나 쪼개져 있는지 (예: connected components, entropy within bbox)
2. Tiny object용으로 **작은 영역**에서 안정적으로 나오는 metric 선택/가중.

**산출물:**  
- 새 metric 정의 (config + 코드)  
- 기존 energy_in_bbox, activation_spread, entropy와 함께 테이블/시각화에 포함

---

# 구현 우선순위 제안

| 순서 | 단계 | 이유 |
|------|------|------|
| 1 | 단계 1 (CAM collapse 정의) | 선행 분석의 "시작점" 정의 없으면 나머지 불가 |
| 2 | 단계 2 (Performance collapse 정의) | CAM과 동일한 논리로 "붕괴" 정의 필요 |
| 3 | 단계 3 (Frame 단위 lead) | 실제 lead 수치 확보 |
| 4 | 단계 4 (통계 검정) | 논문용 정량 근거 |
| 5 | 단계 5 (CAM metric 정제) | 해석 강화, 선택 사항 |

---

# Config/코드 터치 포인트 (참고)

- **Baseline 통계:** `detection_records.csv` severity=0, `cam_records.csv` severity=0
- **z-score collapse:** `src/eval/failure_detection.py` 또는 `risk_detection.py`, 새 모듈 `collapse_detection.py`
- **Lead 계산:** `src/report/llm_report.py` alignment 부분, 또는 `scripts/04_*` / `05_*` 하류
- **통계 검정:** `src/eval/` 또는 `scripts/` 내 새 스크립트
- **CAM metric:** `src/xai/cam_metrics.py`, `cam_records.py`
