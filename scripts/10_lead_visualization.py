"""
Object-level CAM lead visualization.

Reads cam_onset_lead_survival.json (from 07_lead_analysis.py) and produces:
1. CAM change onset distribution (x=severity, y=object count)
2. Lead distribution (x=lead, y=object count)
3. Survival curve P(CAM change not yet occurred) vs severity
4. Per-corruption mean lead (bar or table)

Usage: python scripts/10_lead_visualization.py
Output: results/figures/cam_onset_distribution.png, lead_distribution.png, cam_survival_curve.png, mean_lead_by_corruption.png
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    # Use project root so results/ is found whether run from root or scripts/
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output figures: {figures_dir.resolve()}")

    json_path = results_dir / "cam_onset_lead_survival.json"
    if not json_path.exists():
        print(f"Run 07_lead_analysis.py first. Missing {json_path}")
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skipping figures. Install with: pip install matplotlib")
        sys.exit(0)

    # 1) CAM change onset distribution
    onset = data.get("cam_change_onset_distribution") or {}
    if onset:
        labels = [k for k in (data.get("cam_change_onset_severities") or onset.keys()) if k in onset]
        counts = [onset[k] for k in labels]
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(labels))
        ax.bar(x, counts, color="steelblue", edgecolor="navy", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Object count")
        ax.set_xlabel("CAM change onset (severity)")
        ax.set_title("CAM change onset distribution (object-level)")
        fig.tight_layout()
        p = figures_dir / "cam_onset_distribution.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"Saved {p.resolve()}")

    # 2) Lead distribution
    lead_dist = data.get("lead_distribution") or {}
    if lead_dist:
        leads = sorted([int(k) for k in lead_dist.keys()])
        counts = [lead_dist[str(v)] for v in leads]
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(leads))
        colors = ["#2ecc71" if v > 0 else "#e74c3c" if v < 0 else "#95a5a6" for v in leads]
        ax.bar(x, counts, color=colors, edgecolor="black", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(leads)
        ax.set_ylabel("Object count")
        ax.set_xlabel("Lead (t_perf - t_cam_change)")
        ax.set_title("Lead distribution (lead > 0: CAM precedes performance)")
        fig.tight_layout()
        p = figures_dir / "lead_distribution.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"Saved {p.resolve()}")

    # 3) Survival curve
    survival = data.get("cam_survival_curve") or {}
    if survival:
        sevs = data.get("cam_survival_severities") or sorted(survival.keys(), key=lambda x: int(x[1:]))
        sevs = [s for s in sevs if s in survival]
        probs = [survival[s] for s in sevs]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sevs, probs, "o-", color="steelblue", linewidth=2, markersize=8)
        ax.set_ylabel("P(CAM change not yet occurred)")
        ax.set_xlabel("Severity")
        ax.set_title("CAM change survival curve")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = figures_dir / "cam_survival_curve.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"Saved {p.resolve()}")

    # 4) Per-corruption mean lead
    mean_by_c = data.get("mean_lead_by_corruption") or {}
    if mean_by_c:
        corruptions = data.get("corruptions") or list(mean_by_c.keys())
        corruptions = [c for c in corruptions if c in mean_by_c]
        means = [float(mean_by_c[c]) for c in corruptions]
        fig, ax = plt.subplots(figsize=(max(6, len(corruptions) * 1.2), 4))
        x = np.arange(len(corruptions))
        colors = ["#2ecc71" if m > 0 else "#e74c3c" if m < 0 else "#95a5a6" for m in means]
        ax.bar(x, means, color=colors, edgecolor="black", alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(corruptions, rotation=45, ha="right")
        ax.set_ylabel("Mean lead")
        ax.set_xlabel("Corruption")
        ax.set_title("Mean lead by corruption (positive = CAM precedes performance)")
        fig.tight_layout()
        p = figures_dir / "mean_lead_by_corruption.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"Saved {p.resolve()}")

    print("Done.")


if __name__ == "__main__":
    main()
