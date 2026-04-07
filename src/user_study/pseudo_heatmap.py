"""
FastCAV pseudo-heatmap: non-spatial concept scores × fixed image-only spatial prior.

This is NOT FastCAV localization. For user studies / figures, disclose explicitly.

Note: ``gradcam.xai_methods`` uses ``gradcampp`` for Grad-CAM++ (YOLOGradCAMPP); legacy folder ``fastcam`` may still exist.
(see scripts/05_gradcam_failure_analysis.py), not this module. FastCAV overlays are
``fastcav_pseudo_overlay.png`` from generate_fastcav_pseudo_heatmap / scripts/12.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


def _minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + eps)


def scalar_stress_from_concepts(
    concept_scores: Dict[str, float],
    *,
    visibility_key: str = "tiny_object_visibility",
    visibility_weight: float = 0.7,
    floor: float = 0.12,
    ceiling: float = 1.0,
) -> float:
    """
    Map global concept scores to a scalar g in (0, 1] scaling the surrogate map.

    Default: higher stress when tiny_object_visibility is lower (1 - v).
    """
    v = None
    for k in (visibility_key, f"concept_{visibility_key}"):
        if k in concept_scores:
            v = float(concept_scores[k])
            break
    if v is None:
        g = 0.5
    else:
        g = visibility_weight * float(np.clip(1.0 - v, 0.0, 1.0)) + (1.0 - visibility_weight) * 0.5
    g = float(np.clip(floor + (ceiling - floor) * g, floor, ceiling))
    return g


def generate_fastcav_pseudo_heatmap(
    image_bgr: np.ndarray,
    concept_scores: Dict[str, float],
    base_map: Optional[np.ndarray] = None,
    *,
    gaussian_sigma: float = 4.0,
    stress_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Pseudo-heatmap: H = normalize(B * g(c)) where B is a fixed surrogate prior.

    Args:
        image_bgr: H×W×3 uint8 BGR (corrupted frame used for fair comparison).
        concept_scores: e.g. tiny_object_visibility, other floats.
        base_map: Optional H×W float surrogate; if None, Sobel magnitude + Gaussian blur.
        gaussian_sigma: Blur sigma for default B (pixels).
        stress_fn: Optional callable(concept_scores) -> g in (0,1]; default scalar_stress_from_concepts.

    Returns:
        dict with keys: heatmap_01 (H,W), surrogate_b_01, stress_g, overlay_bgr, jet_bgr
    """
    if stress_fn is None:
        stress_fn = scalar_stress_from_concepts
    g = float(stress_fn(concept_scores))

    if base_map is None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        b = np.sqrt(gx * gx + gy * gy)
        k = int(max(3, round(6 * gaussian_sigma))) | 1
        b = cv2.GaussianBlur(b, (k, k), gaussian_sigma)
        b_01 = _minmax(b)
    else:
        b_01 = _minmax(np.asarray(base_map, dtype=np.float32))

    h_01 = _minmax(b_01 * g)
    jet = cv2.applyColorMap((np.clip(h_01, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = 0.5
    overlay = np.clip(
        alpha * jet.astype(np.float32) + (1 - alpha) * image_bgr.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    return {
        "heatmap_01": h_01,
        "surrogate_b_01": b_01,
        "stress_g": g,
        "overlay_bgr": overlay,
        "jet_bgr": jet,
        "disclaimer": "Pseudo-heatmap: fixed spatial prior (image/Sobel) scaled by global FastCAV concepts; not localization.",
    }


def save_pseudo_heatmap_bundle(
    out_dir: Path,
    image_bgr: np.ndarray,
    concept_scores: Dict[str, float],
    base_map: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Dict[str, Path]:
    """Write jet, overlay, and a small meta json next to images."""
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = generate_fastcav_pseudo_heatmap(image_bgr, concept_scores, base_map, **kwargs)
    p_jet = out_dir / "fastcav_pseudo_jet.png"
    p_ol = out_dir / "fastcav_pseudo_overlay.png"
    cv2.imwrite(str(p_jet), res["jet_bgr"])
    cv2.imwrite(str(p_ol), res["overlay_bgr"])
    meta = {
        "stress_g": res["stress_g"],
        "disclaimer": res["disclaimer"],
        "concept_scores": {k: float(v) for k, v in concept_scores.items() if isinstance(v, (int, float))},
    }
    p_meta = out_dir / "fastcav_pseudo_meta.json"
    p_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"jet": p_jet, "overlay": p_ol, "meta": p_meta}
