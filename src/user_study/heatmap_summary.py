"""Coarse spatial summary from a 2D saliency (no bounding boxes)."""

from __future__ import annotations

from typing import Dict

import numpy as np


def summarize_heatmap_regions(heatmap_01: np.ndarray) -> Dict[str, object]:
    """
    3×3 grid mass → peak quadrant label + concentration heuristic.

    heatmap_01: H×W float in [0,1] (or any nonnegative).
    """
    h, w = heatmap_01.shape[:2]
    if h < 3 or w < 3:
        return {
            "peak_quadrant": "diffuse",
            "concentration": "low",
            "top10_mass_fraction": 0.0,
        }
    cell_h, cell_w = h // 3, w // 3
    masses = []
    labels = [
        "top-left",
        "top-center",
        "top-right",
        "middle-left",
        "middle-center",
        "middle-right",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    ]
    for gi in range(3):
        for gj in range(3):
            y1, y2 = gi * cell_h, (gi + 1) * cell_h if gi < 2 else h
            x1, x2 = gj * cell_w, (gj + 1) * cell_w if gj < 2 else w
            patch = heatmap_01[y1:y2, x1:x2]
            masses.append(float(np.sum(patch)))
    total = sum(masses) + 1e-12
    frac = [m / total for m in masses]
    peak_i = int(np.argmax(masses))
    top10 = float(np.sum(np.sort(np.asarray(heatmap_01).ravel())[-max(1, int(0.1 * h * w)) :]))
    tot = float(np.sum(heatmap_01)) + 1e-12
    top10_mass_fraction = top10 / tot
    if top10_mass_fraction > 0.45:
        conc = "high"
    elif top10_mass_fraction > 0.25:
        conc = "medium"
    else:
        conc = "low"
    peak = labels[peak_i]
    if max(frac) < 1.0 / 9.0 + 0.05:
        peak = "diffuse"
    return {
        "peak_quadrant": peak,
        "concentration": conc,
        "top10_mass_fraction": round(top10_mass_fraction, 4),
        "cell_mass_fractions": {labels[i]: round(frac[i], 4) for i in range(9)},
    }
