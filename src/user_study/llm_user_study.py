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


SYSTEM_PROMPT_TRAJECTORY_KO = """You summarize how Grad-CAM attention and YOLO detection jointly degrade for ONE object as a perturbation severity sweeps from L0 to L4.

Rules:
* You will receive numeric trajectory tables (severity 0..4) for perturbation parameter, detection, and Grad-CAM.
* Every sentence MUST cite at least one numeric value AND the severity index (or physical parameter value) from the tables. No generic prose.
* Do NOT invent values. If a row is missing (e.g., cam_valid=0, NaN), say "cam invalid" or "결측" briefly.
* Judge CAM–performance alignment as one of: 선행(lead) / 동시(coincident) / 지연(lag). Use the severity at which ring_energy_ratio crosses 0.5 vs the severity at which is_miss first becomes 1 or score drops >0.2 from L0.
* Output EXACTLY 3 lines in Korean, in this exact format:

[판정] <one sentence: the severity at which detection breaks, with physical parameter + score/IoU numbers>
[CAM] <one sentence: CAM alignment label + 2 CAM numeric changes (L0→L4 or L0→break-point)>
[경고] <one sentence: pre-deployment operational warning framed on the physical parameter threshold>

No other text. No headings. No bullet points. No English unless quoting a variable name or unit."""


def format_trajectory_block(
    traj_rows: list,
    cam_rows: list,
    *,
    corruption: str,
    perturbation: dict,
) -> str:
    """Assemble 5-severity numeric tables for one (image, object, corruption) unit.

    traj_rows: list of detection_records dicts sorted by severity (len 1..5).
    cam_rows:  list of cam_records dicts sorted by severity (may contain None entries when missing).
    perturbation: {"param_name": str, "values": [v0..v4]} from experiment.yaml.
    """
    def _col(rows, key, n=2):
        out = []
        for r in rows:
            v = None if r is None else r.get(key)
            if v is None or (isinstance(v, float) and v != v):
                out.append("n/a")
            else:
                try:
                    out.append(f"{float(v):.{n}f}")
                except (TypeError, ValueError):
                    out.append(str(v))
        return ", ".join(out)

    def _col_int(rows, key):
        out = []
        for r in rows:
            v = None if r is None else r.get(key)
            if v is None or (isinstance(v, float) and v != v):
                out.append("n/a")
            else:
                try:
                    out.append(str(int(float(v))))
                except (TypeError, ValueError):
                    out.append(str(v))
        return ", ".join(out)

    def _col_str(rows, key):
        out = []
        for r in rows:
            v = None if r is None else r.get(key)
            if v is None or (isinstance(v, float) and v != v) or str(v).strip() == "":
                out.append("n/a")
            else:
                out.append(str(v))
        return ", ".join(out)

    severities = ", ".join(str(int(r.get("severity", i))) for i, r in enumerate(traj_rows))
    p_name = perturbation.get("param_name", "param")
    p_vals = ", ".join(
        f"{float(v):.2f}" if isinstance(v, (int, float)) else str(v)
        for v in perturbation.get("values", [])
    )

    lines = [
        f"[Perturbation] corruption={corruption}",
        f"severity:   {severities}",
        f"{p_name}:   {p_vals}",
        "",
        "[Detection trajectory] (detection_records.csv)",
        f"pred_score:    {_col(traj_rows, 'pred_score', 4)}",
        f"match_iou:     {_col(traj_rows, 'match_iou', 4)}",
        f"delta_score:   {_col(traj_rows, 'delta_score', 4)}",
        f"delta_iou:     {_col(traj_rows, 'delta_iou', 4)}",
        f"is_miss:       {_col_int(traj_rows, 'is_miss')}",
        f"failure_type:  {_col_str(traj_rows, 'failure_type')}",
        "",
        "[Grad-CAM trajectory] (cam_records.csv, primary layer)",
        f"bbox_center_dist:   {_col(cam_rows, 'bbox_center_activation_distance', 2)}",
        f"peak_bbox_dist:     {_col(cam_rows, 'peak_bbox_distance', 2)}",
        f"activation_spread:  {_col(cam_rows, 'activation_spread', 2)}",
        f"ring_energy_ratio:  {_col(cam_rows, 'ring_energy_ratio', 4)}",
        f"energy_in_bbox:     {_col(cam_rows, 'energy_in_bbox', 4)}",
        f"entropy:            {_col(cam_rows, 'entropy', 2)}",
        f"cam_status:         {_col_str(cam_rows, 'cam_status')}",
        "",
        "[Definitions]",
        "- ring_energy_ratio > 0.5 = object-centric attention; < 0.5 = diffuse/off-object",
        "- bbox_center_dist: pixel distance from CAM center-of-mass to GT bbox center",
        "- is_miss=1 when YOLO fails a usable match to GT",
    ]
    return "\n".join(lines)


def build_trajectory_user_message(
    *,
    image_id: str,
    object_uid: str,
    gt_class: str,
    corruption: str,
    trajectory_block: str,
) -> str:
    """Assemble the USER message for the trajectory prompt."""
    return (
        f"[Unit]\n"
        f"image_id: {image_id}\n"
        f"object_uid: {object_uid}\n"
        f"gt_class: {gt_class}\n"
        f"corruption: {corruption}\n\n"
        f"{trajectory_block}\n\n"
        "위 테이블만 사용해 정확히 3줄([판정]/[CAM]/[경고])로 요약."
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
