"""
Analyze E_bbox_1.25x attention patterns from cam_records.csv.

For each corruption (fog, motion_blur, lowlight) and severity (0~4), compute:
  - high-attention:   E_bbox_1.25x >= 0.8
  - low-attention:    E_bbox_1.25x <= 0.2
  - middle-attention: 0.2 < E_bbox_1.25x < 0.8

Then:
  - Plot stacked bar charts of proportions (high / middle / low) vs severity.
  - Plot line plots of proportions vs severity.

Also, for each corruption, analyze object-level transitions across severity:
  - always_high: attention >= 0.8 at all severities where the object appears
  - always_low:  attention <= 0.2 at all severities where the object appears
  - high_to_low: at least one high (>=0.8) at an earlier severity and one low (<=0.2) at a later severity
  - oscillatory: mixed patterns not covered above

Usage:
  python scripts/ebbox_attention_summary.py

Outputs:
  results/ebbox_attention_proportions_[corruption].png
  results/ebbox_attention_transitions_[corruption].csv
"""

from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CSV_PATH = RESULTS_DIR / "cam_records.csv"


HIGH_THR = 0.8
LOW_THR = 0.2


def load_cam_records() -> pd.DataFrame:
    if not CSV_PATH.exists():
        print(f"cam_records.csv not found: {CSV_PATH}")
        return None
    df = pd.read_csv(CSV_PATH)
    if "energy_in_bbox_1_25x" not in df.columns or "severity" not in df.columns or "corruption" not in df.columns:
        print("Required columns missing in cam_records.csv")
        return None
    # Filter to valid CAMs, primary layer (align with app)
    if "cam_status" in df.columns:
        df = df[df["cam_status"] == "ok"]
    if "layer_role" in df.columns:
        df = df[df["layer_role"] == "primary"]
    # Cast types
    df = df.copy()
    df["severity"] = df["severity"].astype(int)
    df["e125"] = pd.to_numeric(df["energy_in_bbox_1_25x"], errors="coerce")
    df = df.dropna(subset=["e125"])
    # Object key: model + corruption + image_id + object_id (if exists)
    if "object_id" in df.columns:
        df["_obj_key"] = (
            df["model"].astype(str)
            + "|"
            + df["corruption"].astype(str)
            + "|"
            + df["image_id"].astype(str)
            + "|"
            + df["object_id"].astype(str)
        )
    else:
        df["_obj_key"] = (
            df["model"].astype(str)
            + "|"
            + df["corruption"].astype(str)
            + "|"
            + df["image_id"].astype(str)
        )
    return df


def compute_proportions(df: pd.DataFrame, corruption: str) -> pd.DataFrame:
    sub = df[df["corruption"] == corruption].copy()
    if sub.empty:
        return pd.DataFrame()
    rows: List[Dict] = []
    for sev in range(5):
        s = sub[sub["severity"] == sev]["e125"]
        n = len(s)
        if n == 0:
            rows.append(
                {
                    "corruption": corruption,
                    "severity": sev,
                    "n": 0,
                    "prop_high": np.nan,
                    "prop_low": np.nan,
                    "prop_mid": np.nan,
                }
            )
            continue
        vals = s.values
        high = np.sum(vals >= HIGH_THR)
        low = np.sum(vals <= LOW_THR)
        mid = np.sum((vals > LOW_THR) & (vals < HIGH_THR))
        rows.append(
            {
                "corruption": corruption,
                "severity": sev,
                "n": int(n),
                "prop_high": high / n,
                "prop_low": low / n,
                "prop_mid": mid / n,
            }
        )
    return pd.DataFrame(rows)


def plot_proportions(df_prop: pd.DataFrame, corruption: str) -> None:
    if df_prop.empty:
        return
    sev = df_prop["severity"].values
    prop_high = df_prop["prop_high"].values
    prop_mid = df_prop["prop_mid"].values
    prop_low = df_prop["prop_low"].values

    x = np.arange(len(sev))
    labels = [f"L{s}" for s in sev]

    # Stacked bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.bar(x, prop_low, label="low (<=0.2)", color="#f97373")
    ax.bar(x, prop_mid, bottom=prop_low, label="mid (0.2~0.8)", color="#facc6b")
    ax.bar(x, prop_high, bottom=prop_low + prop_mid, label="high (>=0.8)", color="#3fb983")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("proportion")
    ax.set_title(f"{corruption} - stacked proportions")
    ax.legend(loc="upper right", fontsize=8)

    # Line plot
    ax2 = axes[1]
    ax2.plot(x, prop_low, "-o", label="low (<=0.2)", color="#f97373")
    ax2.plot(x, prop_mid, "-o", label="mid (0.2~0.8)", color="#facc6b")
    ax2.plot(x, prop_high, "-o", label="high (>=0.8)", color="#3fb983")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("proportion")
    ax2.set_title(f"{corruption} - proportion vs severity")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path = RESULTS_DIR / f"ebbox_attention_{corruption}_proportions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved stacked/line plots for {corruption}: {out_path}")


def classify_transition(vals_by_sev: Dict[int, float]) -> str:
    """Classify object-level attention pattern across severities."""
    if not vals_by_sev:
        return "unknown"
    sevs = sorted(vals_by_sev.keys())
    vals = np.array([vals_by_sev[s] for s in sevs], dtype=float)
    high_mask = vals >= HIGH_THR
    low_mask = vals <= LOW_THR

    if np.all(high_mask):
        return "always_high"
    if np.all(low_mask):
        return "always_low"

    # Any high at earlier sev and low at later sev
    high_indices = np.where(high_mask)[0]
    low_indices = np.where(low_mask)[0]
    if len(high_indices) > 0 and len(low_indices) > 0:
        if high_indices[0] < low_indices[-1]:
            return "high_to_low"

    return "oscillatory"


def analyze_transitions(df: pd.DataFrame, corruption: str) -> pd.DataFrame:
    sub = df[df["corruption"] == corruption].copy()
    if sub.empty:
        return pd.DataFrame()

    pattern_counts: Dict[str, int] = {"always_high": 0, "always_low": 0, "high_to_low": 0, "oscillatory": 0}
    # Group by object key
    for obj_key, g in sub.groupby("_obj_key"):
        vals_by_sev: Dict[int, float] = {}
        for sev, row in g.groupby("severity"):
            # If multiple rows for same sev/key, take mean
            vals_by_sev[int(sev)] = float(row["e125"].mean())
        cat = classify_transition(vals_by_sev)
        if cat not in pattern_counts:
            pattern_counts[cat] = 0
        pattern_counts[cat] += 1

    total = sum(pattern_counts.values())
    rows = []
    for cat, cnt in pattern_counts.items():
        prop = cnt / total if total > 0 else np.nan
        rows.append({"corruption": corruption, "pattern": cat, "count": cnt, "proportion": prop})
    df_out = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / f"ebbox_attention_transitions_{corruption}.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved transition summary for {corruption}: {out_csv}")
    return df_out


def main() -> None:
    df = load_cam_records()
    if df is None or df.empty:
        sys.exit(1)

    corruptions = ["fog", "motion_blur", "lowlight"]
    for corr in corruptions:
        print(f"\n=== {corr} ===")
        df_prop = compute_proportions(df, corr)
        if df_prop.empty:
            print("  No data for this corruption.")
        else:
            print(df_prop[["corruption", "severity", "n", "prop_low", "prop_mid", "prop_high"]])
            plot_proportions(df_prop, corr)
        analyze_transitions(df, corr)


if __name__ == "__main__":
    main()

