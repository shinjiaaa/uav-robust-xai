"""
FastCAV / concept-score visualization on bounding boxes (object-level, not heatmaps).

Concept scores are semantic; this module draws boxes and text only—no CAM heatmaps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


def _to_bgr_uint8(image: Union[np.ndarray, "Image.Image"]) -> np.ndarray:
    """Return HxWx3 BGR uint8 copy for OpenCV drawing."""
    if Image is not None and isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif arr.shape[2] == 3:
        # assume RGB (common from PIL/numpy pipelines)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr.copy()


def _score_to_bgr(
    concept_value: float,
    threshold: float,
    *,
    low_is_bad: bool = True,
) -> Tuple[int, int, int]:
    """
    Map scalar concept to BGR color: green = healthy, red = degraded.

    If low_is_bad: values below threshold trend red; above trend green.
    """
    if low_is_bad:
        t = np.clip((concept_value - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    else:
        t = np.clip((threshold - concept_value) / max(1e-6, threshold), 0.0, 1.0)
    # t=0 -> red (0,0,255 BGR), t=1 -> green (0,255,0)
    b = int(255 * (1.0 - t))
    g = int(255 * t)
    r = int(255 * (1.0 - t))
    return (b, g, r)


def _format_concept_line(
    concepts: Dict[str, float],
    primary_key: Optional[str],
    top_k: int,
) -> str:
    if not concepts:
        return ""
    items = [(k, float(v)) for k, v in concepts.items() if isinstance(v, (int, float, np.floating))]
    if not items:
        return ""
    if primary_key and primary_key in dict(items):
        rest = [(k, v) for k, v in items if k != primary_key]
        rest.sort(key=lambda x: -abs(x[1]))
        chosen = [(primary_key, dict(items)[primary_key])] + rest[: max(0, top_k - 1)]
    else:
        rest = sorted(items, key=lambda x: -abs(x[1]))
        chosen = rest[:top_k]
    parts = [f"{k}={v:.3f}" for k, v in chosen]
    return " | ".join(parts)


def _collect_annotation_lines(
    det: Dict[str, Any],
    concepts: Dict[str, float],
    *,
    primary_concept_key: str,
    threshold: float,
    low_is_bad: bool,
    baseline: Optional[float],
    show_delta: bool,
    warn_below_threshold: bool,
    top_k_concepts: int,
) -> List[str]:
    label = str(det.get("label") or det.get("class_name") or det.get("class", "?"))
    conf = det.get("confidence")
    if conf is None:
        conf = det.get("score", det.get("pred_score", float("nan")))
    try:
        conf_f = float(conf)
    except (TypeError, ValueError):
        conf_f = float("nan")
    primary = concepts.get(primary_concept_key)
    if primary is None and concepts:
        primary = next(iter(concepts.values()))
    elif primary is None:
        primary = 0.0

    lines: List[str] = []
    conf_s = f"{conf_f:.3f}" if np.isfinite(conf_f) else "N/A"
    lines.append(f"{label}  conf={conf_s}")

    ctext = _format_concept_line(concepts, primary_concept_key, top_k_concepts)
    if ctext:
        lines.append(ctext)

    if show_delta and baseline is not None and np.isfinite(baseline):
        b0 = float(baseline)
        lines.append(f"{primary_concept_key}: {b0:.3f} -> {float(primary):.3f}")

    if warn_below_threshold and low_is_bad and float(primary) < threshold:
        lines.append("WARN: below threshold")
    return lines


def visualize_fastcav_bbox(
    image: Union[np.ndarray, "Image.Image"],
    detections: Sequence[Dict[str, Any]],
    concept_scores: Union[Sequence[Dict[str, float]], Dict[str, float]],
    *,
    primary_concept_key: str = "tiny_object_visibility",
    threshold: float = 0.3,
    low_is_bad: bool = True,
    baseline_scores: Optional[Sequence[Optional[float]]] = None,
    show_delta: bool = True,
    warn_below_threshold: bool = True,
    top_k_concepts: int = 3,
    line_thickness: int = 2,
    font_scale: float = 0.45,
    margin: int = 4,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Draw colored bounding boxes on the image; class, confidence, and concept scores in a panel below.

    This does **not** overlay heatmaps. Text is not drawn on top of image pixels.

    Args:
        image: RGB numpy (H,W,3) uint8 or PIL Image.
        detections: One dict per box, keys:
            - ``bbox``: (x1, y1, x2, y2) in image pixels (float or int).
            - ``label`` or ``class_name``: class string.
            - ``confidence`` or ``score`` or ``pred_score``: detector confidence.
        concept_scores: Either:
            - A list of dicts (same length as ``detections``), concept name -> float; or
            - A single dict applied to every box (e.g. global fog proxy).
        primary_concept_key: Used for color and first line of concept text.
        threshold: Below (if low_is_bad) => more red; above => more green.
        low_is_bad: If False, high values are treated as bad (invert mapping).
        baseline_scores: Optional per-box baseline (e.g. severity-0) for "0.22 -> 0.05".
        show_delta: If True and baseline given, append delta in label.
        warn_below_threshold: Draw "WARN" when primary concept < threshold (if low_is_bad).
        top_k_concepts: Max concept key=value segments on second text line.
        output_path: If set, write BGR image to this path (png recommended).

    Returns:
        BGR uint8 image: image region with colored boxes only; annotations in a panel below.

    Notes:
        Text is drawn in a dark strip under the image (not overlaid on pixels).
    """
    det_list = list(detections)
    n = len(det_list)
    if n == 0:
        out = _to_bgr_uint8(image)
        if output_path is not None:
            cv2.imwrite(str(output_path), out)
        return out

    if isinstance(concept_scores, dict) and not isinstance(
        concept_scores, (list, tuple)
    ):
        per_box_concepts: List[Dict[str, float]] = [
            {k: float(v) for k, v in concept_scores.items() if isinstance(v, (int, float, np.floating))}
            for _ in range(n)
        ]
    else:
        cs = list(concept_scores)
        if len(cs) != n:
            raise ValueError(
                f"concept_scores length {len(cs)} must match detections length {n} "
                "unless concept_scores is a single dict (broadcast)."
            )
        per_box_concepts = []
        for c in cs:
            if not isinstance(c, dict):
                raise TypeError("Each concept_scores entry must be a dict of str -> float")
            per_box_concepts.append(
                {k: float(v) for k, v in c.items() if isinstance(v, (int, float, np.floating))}
            )

    baselines: List[Optional[float]]
    if baseline_scores is None:
        baselines = [None] * n
    else:
        baselines = list(baseline_scores)
        if len(baselines) != n:
            raise ValueError("baseline_scores length must match detections length")

    img_layer = _to_bgr_uint8(image)
    h_img, w_img = img_layer.shape[:2]

    # (box_color, lines, valid bbox for optional future use)
    panel_blocks: List[Tuple[Tuple[int, int, int], List[str]]] = []

    for i, det in enumerate(det_list):
        bbox = det.get("bbox") or det.get("box")
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img))
        if x2 <= x1 or y2 <= y1:
            continue

        concepts = per_box_concepts[i]
        primary = concepts.get(primary_concept_key)
        if primary is None and concepts:
            primary = next(iter(concepts.values()))
        elif primary is None:
            primary = 0.0

        color = _score_to_bgr(float(primary), threshold, low_is_bad=low_is_bad)
        cv2.rectangle(img_layer, (x1, y1), (x2, y2), color, line_thickness)

        lines = _collect_annotation_lines(
            det,
            concepts,
            primary_concept_key=primary_concept_key,
            threshold=threshold,
            low_is_bad=low_is_bad,
            baseline=baselines[i],
            show_delta=show_delta,
            warn_below_threshold=warn_below_threshold,
            top_k_concepts=top_k_concepts,
        )
        panel_blocks.append((color, lines))

    if not panel_blocks:
        if output_path is not None:
            cv2.imwrite(str(output_path), img_layer)
        return img_layer

    # --- Bottom panel: stack text blocks ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    line_gap = 4
    block_gap = 14
    panel_pad_x = 12
    panel_pad_top = 12
    panel_pad_bottom = 14
    accent_w = 5

    def text_block_height(text: str) -> int:
        (_, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
        return th + bl

    total_text_h = 0
    for bi, (_, lines) in enumerate(panel_blocks):
        if bi > 0:
            total_text_h += block_gap
        total_text_h += text_block_height(f"#{bi + 1}") + line_gap
        for ln in lines:
            total_text_h += text_block_height(ln) + line_gap

    panel_h = panel_pad_top + total_text_h + panel_pad_bottom
    panel_h = max(panel_h, 48)

    out = np.zeros((h_img + panel_h, w_img, 3), dtype=np.uint8)
    out[:] = (26, 26, 26)
    out[0:h_img, :] = img_layer
    cv2.line(out, (0, h_img), (w_img, h_img), (60, 60, 60), 1)

    y_top = h_img + panel_pad_top
    x_text = panel_pad_x + accent_w + 8

    for bi, (bgr, lines) in enumerate(panel_blocks):
        if bi > 0:
            y_top += block_gap
        header = f"#{bi + 1}"
        (_, th0), bl0 = cv2.getTextSize(header, font, font_scale, thickness)
        x0 = panel_pad_x
        bar_top = y_top
        bar_bottom = y_top + th0 + bl0 + 2
        cv2.rectangle(out, (x0, bar_top), (x0 + accent_w - 1, bar_bottom), bgr, -1)
        cv2.putText(
            out,
            header,
            (x_text, y_top + th0 + bl0),
            font,
            font_scale,
            (220, 220, 220),
            thickness,
            cv2.LINE_AA,
        )
        y_top += th0 + bl0 + line_gap
        for ln in lines:
            (_, th), bl = cv2.getTextSize(ln, font, font_scale, thickness)
            cv2.putText(
                out,
                ln,
                (x_text, y_top + th + bl),
                font,
                font_scale,
                (235, 235, 235),
                thickness,
                cv2.LINE_AA,
            )
            y_top += th + bl + line_gap

    if output_path is not None:
        cv2.imwrite(str(output_path), out)
    return out


def visualize_fastcav_bbox_from_row(
    image: Union[np.ndarray, "Image.Image"],
    *,
    bbox_xyxy: Tuple[float, float, float, float],
    label: str,
    confidence: float,
    concepts: Dict[str, float],
    **kwargs: Any,
) -> np.ndarray:
    """Convenience wrapper for a single detection dict built from one CSV row."""
    det = {
        "bbox": bbox_xyxy,
        "label": label,
        "confidence": confidence,
    }
    return visualize_fastcav_bbox(image, [det], [concepts], **kwargs)
