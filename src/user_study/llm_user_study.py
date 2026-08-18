"""Structured LLM prompts for user study (Grad-CAM vs FastCAV), strict I/O format."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from openai import OpenAI
from dotenv import load_dotenv

# Full instructions for the model (human evaluation; no hallucination).
SYSTEM_PROMPT_STRICT = """You are generating structured explanations for a user study.

IMPORTANT:

* You must strictly follow the output format.
* Do NOT add extra sentences.
* Do NOT hallucinate or infer unseen information.
* Only use the provided inputs.
* If uncertain, explicitly say "uncertain".

---

## TASK

Generate a structured explanation based on:

* visualization summary (Grad-CAM or FastCAV pseudo-heatmap)
* concept scores (if provided)
* performance signals (confidence, miss trend, corruption, severity, predicted class, etc.)
* when provided: detection record metrics (match / IoU / deltas) and Grad-CAM record metrics (spread, ring ratio, distances) — use these numbers; do not invent others

The explanation supports human evaluation and **pre-deployment risk awareness** (validation before field use, not real-time piloting).

---

## OUTPUT FORMAT (STRICT)

You MUST follow this exact format:

[Assessment] <one sentence: overall detection reliability or trust in this condition>

[Key Issue] <one sentence: how detection is failing or weakening—e.g. weak target evidence, diffuse attention. Do NOT use a separate location-only line; if helpful, weave spatial hints here>

[Reason] <one sentence: connect corruption/environment to weakened visual cues, based ONLY on given signals>

[Pre-Deployment Warning] <one sentence: operational risk before deployment and whether additional validation is needed for similar conditions>

---

## RULES

1. Do NOT use a standalone [Location] section. Any “where” detail belongs inside [Key Issue] if needed.

2. DO NOT assume precise spatial localization for FastCAV
   * Use cautious language like "around highlighted regions", "diffuse support", "weak spatial concentration"

3. DO NOT invent causes not present in input

4. Keep each field to ONE sentence

5. Use simple and clear language

6. No bullet points, no extra formatting in your output

7. For Grad-CAM, do not claim finer detail than the visualization summary provides.

8. [Pre-Deployment Warning] must end on a **pre-field / validation** framing (missed detection risk, false trust, recheck similar weather or corruption)—not vague generic harm.

---

Respond only with the four labeled lines [Assessment] through [Pre-Deployment Warning], nothing else."""


def build_strict_user_study_user_message(
    *,
    method: str,
    visualization_summary: str,
    concept_signals: Optional[str],
    performance_signals: str,
) -> str:
    """
    Variable part of the prompt: [Method], [Visualization Summary], [Concept Signals], [Performance Signals].
    method must be exactly "Grad-CAM" or "FastCAV" for the study arms.
    """
    concept_block = (
        concept_signals.strip()
        if concept_signals and str(concept_signals).strip()
        else "(none — not applicable or not provided)"
    )
    return (
        f"[Method]\n{method}\n\n"
        f"[Visualization Summary]\n{visualization_summary.strip()}\n\n"
        f"[Concept Signals]\n{concept_block}\n\n"
        f"[Performance Signals]\n{performance_signals.strip()}\n\n"
        "Now generate the explanation for the input above."
    )


def format_performance_signals_block(
    *,
    confidence_trend: str,
    miss_trend: str,
    confidence_l0: Optional[float] = None,
    confidence_current: Optional[float] = None,
) -> str:
    """One compact block; numbers only if provided (from data, not invented)."""
    lines = [f"confidence trend: {confidence_trend}"]
    if confidence_l0 is not None and confidence_current is not None:
        lines.append(f"confidence values (severity 0 → current): {confidence_l0:.4f} → {confidence_current:.4f}")
    lines.append(f"miss / detection-failure trend: {miss_trend}")
    return "\n".join(lines)


def _as_opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        if isinstance(v, str) and not v.strip():
            return None
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    return x


def _as_opt_bool(v: Any) -> Optional[bool]:
    if v is None or (isinstance(v, str) and not str(v).strip()):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        if isinstance(v, float) and v != v:
            return None
        if v == 0:
            return False
        if v == 1:
            return True
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _fmt_float(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def format_detection_quant_block(
    row: Mapping[str, Any],
    row_l0: Optional[Mapping[str, Any]] = None,
) -> str:
    """Per-frame detection metrics from detection_records (numeric only, no invention)."""
    lines = [
        "detection record (same frame as visualization):",
    ]
    matched = _as_opt_bool(row.get("matched"))
    if matched is not None:
        lines.append(f"  matched to GT tiny box: {matched}")
    is_miss = _as_opt_bool(row.get("is_miss"))
    if is_miss is None:
        is_miss = _as_opt_bool(row.get("is_missed"))
    if is_miss is not None:
        lines.append(f"  is_miss (no usable match): {is_miss}")
    miou = _as_opt_float(row.get("match_iou"))
    if miou is None:
        miou = _as_opt_float(row.get("best_iou"))
    lines.append(f"  match IoU (best / matched): {_fmt_float(miou)}")
    ps = _as_opt_float(row.get("pred_score"))
    if ps is None:
        ps = _as_opt_float(row.get("score"))
    lines.append(f"  pred confidence: {_fmt_float(ps)}")
    for label, key in (
        ("delta_score vs severity-0 baseline", "delta_score"),
        ("delta_iou vs severity-0 baseline", "delta_iou"),
    ):
        dv = _as_opt_float(row.get(key))
        if dv is not None:
            lines.append(f"  {label}: {_fmt_float(dv)}")
    ft = row.get("failure_type")
    if ft is not None and str(ft).strip() and str(ft).lower() not in ("nan", "none"):
        lines.append(f"  failure_type (pipeline label): {ft}")
    if row_l0 is not None:
        lines.append("severity-0 reference (same object × corruption):")
        m0 = _as_opt_bool(row_l0.get("matched"))
        if m0 is not None:
            lines.append(f"  matched@L0: {m0}")
        miss0 = _as_opt_bool(row_l0.get("is_miss"))
        if miss0 is None:
            miss0 = _as_opt_bool(row_l0.get("is_missed"))
        if miss0 is not None:
            lines.append(f"  is_miss@L0: {miss0}")
        i0 = _as_opt_float(row_l0.get("match_iou"))
        if i0 is None:
            i0 = _as_opt_float(row_l0.get("best_iou"))
        lines.append(f"  match IoU@L0: {_fmt_float(i0)}")
        p0 = _as_opt_float(row_l0.get("pred_score"))
        if p0 is None:
            p0 = _as_opt_float(row_l0.get("score"))
        lines.append(f"  pred confidence@L0: {_fmt_float(p0)}")
    lines.append(
        "  note: corruption severity is a discrete pipeline level (0=reference, higher=stronger corruption in this study), not a physical weather SI unit."
    )
    return "\n".join(lines)


def format_cam_record_quant_block(
    cam: Optional[Mapping[str, Any]],
    cam_l0: Optional[Mapping[str, Any]] = None,
) -> str:
    """Grad-CAM pipeline metrics from cam_records.csv (numeric; omit if file/row missing)."""
    if not cam:
        return "gradcam metrics record: (none — cam_records row not found or not merged)"
    lines = ["gradcam metrics record (cam_records.csv, primary layer if filtered upstream):"]
    st = cam.get("cam_status")
    if st is not None and str(st).strip():
        lines.append(f"  cam_status: {st}")
    cv = cam.get("cam_valid")
    if cv is not None and str(cv).strip():
        lines.append(f"  cam_valid: {cv}")
    keys = [
        ("bbox_center_activation_distance", "bbox_center_activation_distance"),
        ("peak_bbox_distance", "peak_bbox_distance"),
        ("activation_spread", "activation_spread"),
        ("ring_energy_ratio", "ring_energy_ratio"),
        ("entropy", "entropy"),
        ("energy_in_bbox_1_25x", "energy_in_bbox_1_25x"),
    ]
    for label, k in keys:
        v = _as_opt_float(cam.get(k))
        if v is not None:
            lines.append(f"  {label}: {_fmt_float(v)}")
    if cam_l0:
        lines.append("  vs severity-0 CAM record (same keys):")
        for label, k in keys:
            v0 = _as_opt_float(cam_l0.get(k))
            v1 = _as_opt_float(cam.get(k))
            if v0 is not None and v1 is not None:
                lines.append(f"    {label}: L0 {_fmt_float(v0)} → current {_fmt_float(v1)} (Δ {_fmt_float(v1 - v0)})")
    return "\n".join(lines)


def format_overlay_numeric_block(
    grad_summary: Mapping[str, Any],
    grad_summary_l0: Optional[Mapping[str, Any]] = None,
) -> str:
    """Numbers derived from saved Grad-CAM overlay PNG (proxy; same as visualization summary source)."""
    t10 = _as_opt_float(grad_summary.get("top10_mass_fraction"))
    lines = [
        "gradcam overlay PNG proxy metrics:",
        f"  peak_quadrant: {grad_summary.get('peak_quadrant', 'unknown')}",
        f"  concentration label: {grad_summary.get('concentration', 'unknown')}",
        f"  top10_mass_fraction: {_fmt_float(t10)}",
    ]
    if grad_summary_l0:
        t0 = _as_opt_float(grad_summary_l0.get("top10_mass_fraction"))
        t1 = _as_opt_float(grad_summary.get("top10_mass_fraction"))
        if t0 is not None and t1 is not None:
            lines.append(
                f"  vs L0 overlay: top10_mass_fraction L0 {_fmt_float(t0)} → current {_fmt_float(t1)} "
                f"(Δ {_fmt_float(t1 - t0)})"
            )
    return "\n".join(lines)


def _trend_numeric(v0: Optional[float], v1: Optional[float], *, eps: float = 1e-5) -> str:
    if v0 is None or v1 is None:
        return "uncertain"
    try:
        a, b = float(v0), float(v1)
    except (TypeError, ValueError):
        return "uncertain"
    d = b - a
    if abs(d) < eps * max(1.0, abs(a)):
        return "stable"
    return "increasing" if d > 0 else "decreasing"


def _trend_miss(m0: Optional[bool], m1: Optional[bool]) -> str:
    if m0 is None or m1 is None:
        return "uncertain"
    if m0 == m1:
        return "stable"
    if m1 and not m0:
        return "increasing (miss or failure at current severity, not at severity 0)"
    return "decreasing (failure at severity 0 resolved at current severity)"


def format_concept_signals_with_trends(
    concepts_current: Dict[str, float],
    concepts_l0: Optional[Dict[str, float]],
) -> str:
    """Lines like: name: 0.05 (decreasing vs severity 0)."""
    if not concepts_current:
        return "(no concept scores in input)"
    lines = []
    keys = sorted(concepts_current.keys())
    for k in keys:
        v = concepts_current[k]
        if concepts_l0 is None or k not in concepts_l0:
            lines.append(f"{k}: {v:.4f} (trend vs severity 0: uncertain — no baseline)")
            continue
        tr = _trend_numeric(concepts_l0[k], v)
        lines.append(f"{k}: {v:.4f} ({tr} vs severity 0)")
    return "\n".join(lines)


def narrate_gradcam_visualization_summary(
    summary_current: Dict[str, Any],
    summary_l0: Optional[Dict[str, Any]],
    *,
    severity: int = 0,
) -> str:
    """Single short paragraph from grid summaries only (no raw CAM)."""
    peak = summary_current.get("peak_quadrant", "unknown")
    conc = summary_current.get("concentration", "unknown")
    t10 = summary_current.get("top10_mass_fraction", "n/a")
    s1 = (
        f"Overlay-based saliency emphasizes the {peak} area with {conc} concentration "
        f"(top-10% mass fraction {t10})."
    )
    if int(severity) == 0:
        return s1
    if summary_l0 is None:
        return s1 + " Change vs severity 0 overlay: uncertain (L0 overlay missing or unreadable)."
    p0 = summary_l0.get("peak_quadrant", "unknown")
    c0 = summary_l0.get("concentration", "unknown")
    if p0 == peak and c0 == conc:
        return s1 + " Compared to severity 0, the highlighted region pattern is unchanged."
    return (
        s1
        + f" Compared to severity 0, emphasis shifted from {p0} ({c0} concentration) "
        f"to {peak} ({conc} concentration)."
    )


def narrate_fastcav_visualization_summary(
    summary_current: Dict[str, Any],
    summary_l0: Optional[Dict[str, Any]],
    stress_g: float,
    stress_l0: Optional[float],
    *,
    severity: int = 0,
) -> str:
    """Explicit pseudo-heatmap disclaimer + optional change vs L0."""
    peak = summary_current.get("peak_quadrant", "unknown")
    conc = summary_current.get("concentration", "unknown")
    t10 = summary_current.get("top10_mass_fraction", "n/a")
    s1 = (
        f"FastCAV pseudo-heatmap (not precise spatial localization) shows relative emphasis "
        f"toward the {peak} area with {conc} concentration (top-10% mass fraction {t10}); "
        f"global stress scalar g={float(stress_g):.4f}."
    )
    if int(severity) == 0:
        return s1
    if summary_l0 is None and stress_l0 is None:
        return s1 + " Change vs severity 0: uncertain (no L0 pseudo baseline computed)."
    parts = [s1]
    if stress_l0 is not None:
        tr = _trend_numeric(stress_l0, stress_g)
        parts.append(f"Stress g vs severity 0: {float(stress_l0):.4f} → {float(stress_g):.4f} ({tr}).")
    if summary_l0 is not None:
        p0 = summary_l0.get("peak_quadrant", "unknown")
        c0 = summary_l0.get("concentration", "unknown")
        if p0 != peak or c0 != conc:
            parts.append(
                f"Pseudo spatial emphasis shifted vs severity 0 from {p0} ({c0}) to {peak} ({conc})."
            )
        else:
            parts.append("Pseudo spatial emphasis pattern matches severity 0.")
    return " ".join(parts)


def generate_explanation_openai(
    user_message: str,
    *,
    model: str = "gpt-4o-mini",
    system_prompt: str = SYSTEM_PROMPT_STRICT,
    temperature: float = 0.1,
    client: Optional[OpenAI] = None,
) -> str:
    load_dotenv()
    if client is None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def write_full_prompt_for_disk(system_prompt: str, user_message: str) -> str:
    """Plain-text bundle for a unit folder (audit / reproducibility)."""
    return (
        "=== SYSTEM ===\n"
        + system_prompt.strip()
        + "\n\n=== USER ===\n"
        + user_message.strip()
        + "\n"
    )


# --- Legacy names (older exports / docs); thin wrappers ---

def build_gradcam_method_block(heatmap_summary: Dict[str, object]) -> str:
    return (
        "Heatmap summary (class-discriminative saliency; approximate spatial cue):\n"
        f"- peak_quadrant: {heatmap_summary.get('peak_quadrant', 'unknown')}\n"
        f"- concentration: {heatmap_summary.get('concentration', 'unknown')}\n"
        f"- top10_mass_fraction: {heatmap_summary.get('top10_mass_fraction', 'n/a')}\n"
    )


def build_fastcav_method_block(
    concept_scores: Dict[str, float],
    pseudo_summary: Dict[str, object],
    stress_g: float,
) -> str:
    lines = [
        "Concept scores (global, not pixel-level):",
    ]
    for k, v in sorted(concept_scores.items()):
        if isinstance(v, (int, float)):
            lines.append(f"- {k}: {float(v):.4f}")
    lines.append(
        "Pseudo-heatmap note: visualization = fixed spatial prior (image/Sobel) × global stress g; "
        "NOT FastCAV localization."
    )
    lines.append(f"- stress_g (scalar applied to prior): {stress_g:.4f}")
    lines.append(f"- pseudo_peak_quadrant: {pseudo_summary.get('peak_quadrant', 'unknown')}")
    lines.append(f"- pseudo_concentration: {pseudo_summary.get('concentration', 'unknown')}")
    return "\n".join(lines) + "\n"


def build_explanation_prompt(
    *,
    method_label: str,
    corruption: str,
    severity: int,
    pred_class: str,
    pred_conf: float,
    method_block: str,
) -> str:
    """Deprecated layout; prefer build_strict_user_study_user_message + SYSTEM_PROMPT_STRICT."""
    return f"""Task: Explain the model behavior for ONE image.

Inputs:
- Corruption: {corruption}, Severity: {severity}
- Detection (metadata only, no boxes): class={pred_class}, confidence={pred_conf:.4f}
- Method: {method_label}

Signals:
{method_block}

Output EXACTLY in this format (English):

[Assessment]
...

[Key Issue]
...

[Reason]
...

[Pre-Deployment Warning]
...
"""


SYSTEM_PROMPT_TRAJECTORY_KO = """역할: UAV 영상 검증 결과를 운영자에게 보고하는 분석가
독자: 코드·통계 배경이 없는 항공 인증 검증자

규칙:
1. 변수명, 코드명, 기술 약어(CAM, IoU, severity, alpha, gamma, kernel, ring_energy 등) 사용 금지.
2. 숫자·소수점 값은 절대 출력하지 않는다. 입력에 이미 변환된 등급 표현과 단계 번호만 사용.
3. 객체 ID, 파일명 언급 금지.
4. 입력에 제공된 "성능 붕괴 단계", "시각적 주의 변화 시작 단계", "선행 관계" 값을 반드시 인용하여 서술한다.
5. 출력 형식은 정확히 아래 세 섹션, 총 3~4문장:

[성능 붕괴] 변조가 몇 단계에서 탐지 신뢰도가 어떻게 떨어졌는지 한 문장. 단계 번호와 탐지 등급을 반드시 포함.
[조기 신호] 모델의 시각적 주의가 몇 단계에서 변화하기 시작했는지 한 문장. 단계 번호를 반드시 포함.
[해석] 설명이 성능보다 몇 단계 먼저/나중에/동시에 변화했는지 한 문장, 그리고 이 객체에서 시각적 설명이 조기 경보로 활용 가능한지 한 문장. (총 두 문장)

예시 입력:
- 변조 종류: 안개
- 정상 영상 탐지: 안정적으로 인식
- 변조 최대 영상 탐지: 거의 인식 못함
- 성능 붕괴 단계: 3단계 (짙은 안개)
- 시각적 주의 변화 시작 단계: 1단계 (옅은 안개)
- 선행 관계: 설명이 성능 저하보다 2단계 먼저 변화
- 조기 경보 활용 가능성: 가능

예시 출력:
[성능 붕괴] 안개가 3단계에 도달했을 때 탐지 신뢰도가 안정 수준에서 거의 인식 못함으로 떨어졌습니다.
[조기 신호] 모델의 시각적 주의는 1단계에서 이미 객체에서 벗어나기 시작했습니다.
[해석] 설명이 성능 저하보다 두 단계 먼저 변화를 보였습니다. 이 객체에 대해서는 시각적 설명이 조기 경보로 활용 가능합니다.

다른 출력은 금지. 설명·머리말·부연 없음. 위 세 섹션만."""


SYSTEM_PROMPT_TRAJECTORY_EN = """Role: An analyst reporting UAV imagery validation results to operators.
Audience: Aviation certification validators without a coding or statistics background.

Rules:
1. Do NOT use variable names, code identifiers, or technical abbreviations (CAM, IoU, severity, alpha, gamma, kernel, ring_energy, etc.).
2. NEVER output raw numeric or decimal values. Use only the pre-converted grade labels and stage numbers provided in the input.
3. Do NOT mention object IDs or filenames.
4. You MUST cite the "Performance collapse stage", "Visual attention change onset stage", and "Lead relationship" values from the input.
5. Output must follow exactly the three sections below, in 3-4 total sentences:

[Performance Collapse] One sentence on the stage at which detection confidence degraded and how. Must include the stage number and the detection grade.
[Early Signal] One sentence on the stage at which the model's visual attention began to change. Must include the stage number.
[Interpretation] One sentence on whether the explanation changed before, after, or simultaneously with performance, and one sentence on whether visual explanation can serve as an early warning for this object. (Two sentences total.)

Example input:
- Corruption type: fog
- Detection on clean image: stably detected
- Detection at maximum corruption: barely detected
- Performance collapse stage: stage 3 (dense fog)
- Visual attention change onset stage: stage 1 (light fog)
- Lead relationship: explanation changed two stages earlier than performance degradation
- Early warning usability: possible

Example output:
[Performance Collapse] When fog reached stage 3, detection confidence dropped from a stable level to barely detected.
[Early Signal] The model's visual attention had already begun drifting away from the object at stage 1.
[Interpretation] The explanation changed two stages earlier than performance degradation. For this object, visual explanation can be used as an early warning.

No other output. No headings. No preamble or commentary. Only the three sections above."""


# --- humanization helpers (value → grade string, lang-aware) ---

_NA_LABEL = {"ko": "값 없음", "en": "no value"}
_UNK_LABEL = {"ko": "알 수 없음", "en": "unknown"}

_SCORE_GRADES = {
    "ko": ("거의 인식 못함", "약하게 인식", "안정적으로 인식", "확실히 인식"),
    "en": ("barely detected", "weakly detected", "stably detected", "confidently detected"),
}
_FOG_GRADES = {
    "ko": ("정상 (안개 없음)", "옅은 안개", "중간 농도 안개", "짙은 안개", "매우 짙은 안개"),
    "en": ("clear (no fog)", "light fog", "moderate fog", "dense fog", "very dense fog"),
}
_LOWLIGHT_GRADES = {
    "ko": ("정상 조도", "약간 어두움", "어두움", "매우 어두움", "거의 암흑"),
    "en": ("normal lighting", "slightly dark", "dark", "very dark", "near darkness"),
}
_MOTION_GRADES = {
    "ko": ("정상 (흔들림 없음)", "약간 흔들림", "보통 흔들림", "심한 흔들림", "매우 심한 흔들림"),
    "en": ("normal (no blur)", "slight motion blur", "moderate motion blur", "severe motion blur", "very severe motion blur"),
}
_CHANGE_PCT_GRADES = {
    "ko": ("변화 알 수 없음", "변화 없음", "약간 변화", "크게 변화"),
    "en": ("change unknown", "no change", "slight change", "large change"),
}
_ATTENTION_SHIFT = {
    "ko": {
        "unknown": "주의 변화 알 수 없음",
        "none": "변화 없음",
        "diffuse_strong": "객체 밖으로 크게 분산",
        "diffuse_weak": "객체 주변으로 약간 분산",
        "focus_more": "객체 쪽으로 더 집중",
        "slight": "약간 변화",
    },
    "en": {
        "unknown": "attention change unknown",
        "none": "no change",
        "diffuse_strong": "strongly diffused outside the object",
        "diffuse_weak": "slightly diffused around the object",
        "focus_more": "more focused on the object",
        "slight": "slight change",
    },
}


def humanize_score(score, lang: str = "ko") -> str:
    """Detection confidence grade."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return _NA_LABEL[lang]
    if s != s:  # NaN
        return _NA_LABEL[lang]
    g = _SCORE_GRADES[lang]
    if s < 0.2:
        return g[0]
    if s < 0.5:
        return g[1]
    if s < 0.8:
        return g[2]
    return g[3]


def humanize_fog(alpha, lang: str = "ko") -> str:
    try:
        a = float(alpha)
    except (TypeError, ValueError):
        return _UNK_LABEL[lang]
    g = _FOG_GRADES[lang]
    if a <= 0.01:
        return g[0]
    if a <= 0.25:
        return g[1]
    if a <= 0.50:
        return g[2]
    if a <= 0.75:
        return g[3]
    return g[4]


def humanize_lowlight(brightness, lang: str = "ko") -> str:
    """brightness = 1.0 (normal) → 0.2 (very dark)."""
    try:
        b = float(brightness)
    except (TypeError, ValueError):
        return _UNK_LABEL[lang]
    g = _LOWLIGHT_GRADES[lang]
    if b >= 0.95:
        return g[0]
    if b >= 0.70:
        return g[1]
    if b >= 0.45:
        return g[2]
    if b >= 0.25:
        return g[3]
    return g[4]


def humanize_motion(kernel, lang: str = "ko") -> str:
    try:
        k = float(kernel)
    except (TypeError, ValueError):
        return _UNK_LABEL[lang]
    g = _MOTION_GRADES[lang]
    if k <= 0.5:
        return g[0]
    if k <= 5:
        return g[1]
    if k <= 10:
        return g[2]
    if k <= 15:
        return g[3]
    return g[4]


def humanize_corruption(corruption: str, value, lang: str = "ko") -> str:
    c = str(corruption).lower()
    if c == "fog":
        return humanize_fog(value, lang=lang)
    if c == "lowlight":
        return humanize_lowlight(value, lang=lang)
    if c == "motion_blur":
        return humanize_motion(value, lang=lang)
    return str(value)


def humanize_change_pct(v_new, v_old, lang: str = "ko") -> str:
    """Percent-change classifier. Uses absolute relative change."""
    g = _CHANGE_PCT_GRADES[lang]
    try:
        a, b = float(v_new), float(v_old)
    except (TypeError, ValueError):
        return g[0]
    if a != a or b != b:
        return g[0]
    denom = max(abs(b), 1e-6)
    pct = abs(a - b) / denom
    if pct < 0.10:
        return g[1]
    if pct < 0.30:
        return g[2]
    return g[3]


def humanize_attention_shift(ring_l0, ring_l4, lang: str = "ko") -> str:
    """Ring energy ratio measures object-centric focus. Drop = attention diffused away from object."""
    s = _ATTENTION_SHIFT[lang]
    try:
        r0, r4 = float(ring_l0), float(ring_l4)
    except (TypeError, ValueError):
        return s["unknown"]
    if r0 != r0 or r4 != r4:
        return s["unknown"]
    drop = r0 - r4
    if abs(drop) < 0.05:
        return s["none"]
    if drop > 0.30:
        return s["diffuse_strong"]
    if drop > 0.10:
        return s["diffuse_weak"]
    if drop < -0.10:
        return s["focus_more"]
    return s["slight"]


def _first_last_numeric(rows, key):
    """Return (baseline_value, worst_value) where baseline = first row's key, worst = last non-nan."""
    base = None
    last = None
    for r in rows:
        if r is None:
            continue
        v = r.get(key)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv != fv:
            continue
        if base is None:
            base = fv
        last = fv
    return base, last


def _detect_performance_collapse_stage(traj_rows, collapse_score_threshold: float = 0.2):
    """First severity index where detection is no longer acceptable.

    Criteria (whichever fires first):
    - is_miss == 1 (no usable GT match)
    - pred_score < collapse_score_threshold (below "거의 인식 못함" grade)

    Returns int severity index 1..4, or None if detection holds across all stages.
    L0 (severity 0) is skipped since it's the baseline.
    """
    for r in traj_rows:
        if r is None:
            continue
        try:
            sev = int(r.get("severity", -1))
        except (TypeError, ValueError):
            continue
        if sev <= 0:
            continue
        is_miss = r.get("is_miss")
        try:
            if int(float(is_miss)) == 1:
                return sev
        except (TypeError, ValueError):
            pass
        ps = r.get("pred_score")
        try:
            if float(ps) < collapse_score_threshold:
                return sev
        except (TypeError, ValueError):
            pass
    return None


def _detect_cam_change_stage(cam_rows, min_relative_change: float = 0.10):
    """First severity index where ring_energy_ratio deviates materially from L0 baseline.

    min_relative_change: threshold on |ring(sev) - ring(L0)| / max(|ring(L0)|, ε).
    Returns int severity index 1..4, or None if CAM attention stays stable.
    """
    base_ring = None
    for r in cam_rows:
        if r is None:
            continue
        try:
            sev = int(r.get("severity", -1))
        except (TypeError, ValueError):
            continue
        if sev != 0:
            continue
        try:
            base_ring = float(r.get("ring_energy_ratio"))
            break
        except (TypeError, ValueError):
            continue
    if base_ring is None or base_ring != base_ring:
        return None

    for r in cam_rows:
        if r is None:
            continue
        try:
            sev = int(r.get("severity", -1))
        except (TypeError, ValueError):
            continue
        if sev <= 0:
            continue
        try:
            cur = float(r.get("ring_energy_ratio"))
        except (TypeError, ValueError):
            continue
        if cur != cur:
            continue
        rel = abs(cur - base_ring) / max(abs(base_ring), 1e-6)
        if rel >= min_relative_change:
            return sev
    return None


_NO_CHANGE_LABEL = {"ko": "변화 없음", "en": "no change"}


def _stage_label(sev, lang: str = "ko") -> str:
    if sev is None:
        return _NO_CHANGE_LABEL[lang]
    try:
        s = int(sev)
    except (TypeError, ValueError):
        return _NO_CHANGE_LABEL[lang]
    return f"{s}단계" if lang == "ko" else f"stage {s}"


def _corruption_stage_grade(corruption: str, perturbation_values, sev, lang: str = "ko") -> str:
    """For a given severity index, return humanized grade label of the corruption parameter."""
    if sev is None or not perturbation_values:
        return ""
    try:
        idx = int(sev)
        if idx < 0 or idx >= len(perturbation_values):
            return ""
        return humanize_corruption(corruption, perturbation_values[idx], lang=lang)
    except (TypeError, ValueError):
        return ""


def _lead_relation(t_perf, t_cam) -> tuple:
    """Return (relation_label, lead_steps_or_None).

    relation: 'lead' (CAM ahead), 'coincident', 'lag' (CAM behind), 'unknown'.
    """
    if t_perf is None and t_cam is None:
        return ("no_change", None)
    if t_perf is None:
        return ("perf_stable", None)
    if t_cam is None:
        return ("cam_stable", None)
    lead = int(t_perf) - int(t_cam)
    if lead > 0:
        return ("lead", lead)
    if lead == 0:
        return ("coincident", 0)
    return ("lag", lead)


_KO_STEP_WORD = {1: "한", 2: "두", 3: "세", 4: "네"}
_EN_STEP_WORD = {1: "one", 2: "two", 3: "three", 4: "four"}


def _step_count_word(n: int, lang: str = "ko") -> str:
    if n is None:
        return ""
    try:
        i = int(n)
    except (TypeError, ValueError):
        return ""
    table = _KO_STEP_WORD if lang == "ko" else _EN_STEP_WORD
    return table.get(i, f"{i}")


# Back-compat alias for legacy callers.
def _korean_step_count(n: int) -> str:
    return _step_count_word(n, lang="ko")


_USABILITY = {
    "ko": {"possible": "가능", "limited": "제한적", "not_possible": "불가", "n_a": "해당 없음"},
    "en": {"possible": "possible", "limited": "limited", "not_possible": "not possible", "n_a": "not applicable"},
}


def _humanize_lead_relation(relation: str, steps, lang: str = "ko") -> tuple:
    """Return (relation_phrase, early_warning_usability)."""
    u = _USABILITY[lang]
    if relation == "lead":
        k = _step_count_word(steps, lang=lang)
        if lang == "ko":
            phrase = f"설명이 성능 저하보다 {k} 단계 먼저 변화"
        else:
            stage_word = "stage" if str(k) == "1" or k == "one" else "stages"
            phrase = f"explanation changed {k} {stage_word} earlier than performance degradation"
        return (phrase, u["possible"])
    if relation == "coincident":
        if lang == "ko":
            return ("설명과 성능이 같은 단계에서 변화", u["limited"])
        return ("explanation and performance changed at the same stage", u["limited"])
    if relation == "lag":
        k = _step_count_word(abs(steps), lang=lang) if steps is not None else ""
        if lang == "ko":
            phrase = f"설명이 성능 저하보다 {k} 단계 늦게 변화"
        else:
            stage_word = "stage" if str(k) == "1" or k == "one" else "stages"
            phrase = f"explanation changed {k} {stage_word} later than performance degradation"
        return (phrase, u["not_possible"])
    if relation == "perf_stable":
        if lang == "ko":
            return ("변조 최대까지 탐지 성능이 유지됨", u["n_a"])
        return ("detection performance held through maximum corruption", u["n_a"])
    if relation == "cam_stable":
        if lang == "ko":
            return ("시각적 주의에 의미 있는 변화가 없음", u["not_possible"])
        return ("no meaningful change in visual attention", u["not_possible"])
    if lang == "ko":
        return ("판단 불가", u["not_possible"])
    return ("cannot determine", u["not_possible"])


_TRAJ_LABELS = {
    "ko": {
        "corruption_type": "변조 종류",
        "clean_detection": "정상 영상 탐지",
        "max_corruption_detection": "변조 최대 영상 탐지",
        "perf_collapse_stage": "성능 붕괴 단계",
        "cam_onset_stage": "시각적 주의 변화 시작 단계",
        "lead_relationship": "선행 관계",
        "early_warning_usability": "조기 경보 활용 가능성",
        "perf_held": "변조 최대까지 탐지 성능 유지",
        "cam_no_change": "변조 최대까지 시각적 주의 변화 없음",
        "input_header": "[입력]",
        "instruction": "위 정보만 사용해 정확히 3섹션([성능 붕괴] / [조기 신호] / [해석])으로 요약.",
    },
    "en": {
        "corruption_type": "Corruption type",
        "clean_detection": "Detection on clean image",
        "max_corruption_detection": "Detection at maximum corruption",
        "perf_collapse_stage": "Performance collapse stage",
        "cam_onset_stage": "Visual attention change onset stage",
        "lead_relationship": "Lead relationship",
        "early_warning_usability": "Early warning usability",
        "perf_held": "detection performance held through maximum corruption",
        "cam_no_change": "no visual attention change through maximum corruption",
        "input_header": "[Input]",
        "instruction": "Using only the information above, summarize in exactly three sections ([Performance Collapse] / [Early Signal] / [Interpretation]).",
    },
}


def format_trajectory_block(
    traj_rows: list,
    cam_rows: list,
    *,
    corruption: str,
    perturbation: dict,
    lang: str = "ko",
) -> str:
    """Humanized trajectory summary with pre-computed collapse/early-warning stages.

    The LLM receives pre-analyzed stage numbers + lead relation so it only has to
    narrate, never infer from raw numerics.
    """
    p_values = perturbation.get("values", [])
    L = _TRAJ_LABELS[lang]
    na = _NA_LABEL[lang]

    # 1. detection confidence grade (clean vs max)
    score_base, score_last = _first_last_numeric(traj_rows, "pred_score")
    score_base_grade = humanize_score(score_base, lang=lang) if score_base is not None else na
    score_last_grade = humanize_score(score_last, lang=lang) if score_last is not None else na

    # 2. performance collapse stage
    t_perf = _detect_performance_collapse_stage(traj_rows)
    t_perf_label = _stage_label(t_perf, lang=lang) if t_perf is not None else (
        "성능 유지됨" if lang == "ko" else "performance held"
    )
    t_perf_grade = _corruption_stage_grade(corruption, p_values, t_perf, lang=lang)
    if t_perf is not None and t_perf_grade:
        t_perf_full = f"{t_perf_label} ({t_perf_grade})"
    elif t_perf is not None:
        t_perf_full = t_perf_label
    else:
        t_perf_full = L["perf_held"]

    # 3. visual attention change onset stage
    t_cam = _detect_cam_change_stage(cam_rows)
    t_cam_label = _stage_label(t_cam, lang=lang) if t_cam is not None else _NO_CHANGE_LABEL[lang]
    t_cam_grade = _corruption_stage_grade(corruption, p_values, t_cam, lang=lang)
    if t_cam is not None and t_cam_grade:
        t_cam_full = f"{t_cam_label} ({t_cam_grade})"
    elif t_cam is not None:
        t_cam_full = t_cam_label
    else:
        t_cam_full = L["cam_no_change"]

    # 4. lead relationship
    relation, steps = _lead_relation(t_perf, t_cam)
    relation_phrase, usability = _humanize_lead_relation(relation, steps, lang=lang)

    lines = [
        f"- {L['corruption_type']}: {humanize_corruption_name(corruption, lang=lang)}",
        f"- {L['clean_detection']}: {score_base_grade}",
        f"- {L['max_corruption_detection']}: {score_last_grade}",
        f"- {L['perf_collapse_stage']}: {t_perf_full}",
        f"- {L['cam_onset_stage']}: {t_cam_full}",
        f"- {L['lead_relationship']}: {relation_phrase}",
        f"- {L['early_warning_usability']}: {usability}",
    ]
    return "\n".join(lines)


_CORRUPTION_NAMES = {
    "ko": {"fog": "안개", "lowlight": "저조도", "motion_blur": "카메라 흔들림"},
    "en": {"fog": "fog", "lowlight": "low light", "motion_blur": "camera motion blur"},
}


def humanize_corruption_name(corruption: str, lang: str = "ko") -> str:
    c = str(corruption).lower()
    return _CORRUPTION_NAMES[lang].get(c, c)


def build_trajectory_user_message(
    *,
    image_id: str,
    object_uid: str,
    gt_class: str,
    corruption: str,
    trajectory_block: str,
    lang: str = "ko",
) -> str:
    """Humanized USER message: no IDs, no codes. Only grade labels."""
    L = _TRAJ_LABELS[lang]
    return (
        f"{L['input_header']}\n{trajectory_block}\n\n"
        f"{L['instruction']}"
    )


def write_evaluation_questionnaire(path: Path) -> None:
    """Likert template for paper appendix / IRB packet."""
    text = """User study — Likert (1=Strongly disagree … 5=Strongly agree)

Spatial alignment
Q1. The explanation’s described region matches where the visualization emphasizes activation.
Q2. The explanation does not contradict what I see in the visualization.

Interpretability
Q3. The explanation is intuitive and easy to understand.
Q4. The explanation uses clear language (not overly technical).

Usefulness
Q5. The explanation helps me understand why detection may be unreliable under this corruption.
Q6. The explanation would help me decide whether to trust the detector in this situation.

Manipulation check (FastCAV pseudo condition)
Q7. I understand the second visualization is not claimed to show exact pixel-level causes of concept scores.

(Optional) Open: What felt mismatched between text and image?
"""
    path.write_text(text, encoding="utf-8")
