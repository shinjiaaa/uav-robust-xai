"""
Plot distribution of E_bbox_1.25x across all objects per severity (L0–L4).
Check for bimodality (peaks near 0 and 1) and report median + quartiles per severity.

Usage: python scripts/plot_ebbox_distribution.py
Output: results/ebbox_1_25x_distribution.png and summary to stdout.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CSV_PATH = RESULTS_DIR / "cam_records.csv"
OUT_PATH = RESULTS_DIR / "ebbox_1_25x_distribution.png"


def load_and_filter():
    if not CSV_PATH.exists():
        print(f"Not found: {CSV_PATH}")
        return None
    df = pd.read_csv(CSV_PATH)
    if "energy_in_bbox_1_25x" not in df.columns or "severity" not in df.columns:
        print("Missing columns: energy_in_bbox_1_25x or severity")
        return None
    # Same filters as app: ok, primary
    if "cam_status" in df.columns:
        df = df[df["cam_status"] == "ok"]
    if "layer_role" in df.columns:
        df = df[df["layer_role"] == "primary"]
    df["severity"] = df["severity"].astype(int)
    df["e125"] = pd.to_numeric(df["energy_in_bbox_1_25x"], errors="coerce")
    df = df.dropna(subset=["e125"])
    return df


def quartiles_and_median(series):
    s = series.dropna()
    if len(s) == 0:
        return None, None, None
    return (
        float(s.quantile(0.25)),
        float(s.median()),
        float(s.quantile(0.75)),
    )


def check_bimodal(values, bins=50):
    """Bimodality check: significant mass near 0 and near 1 (peaks at 0 and 1)."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return False, "too_few"
    n = len(values)
    # Fraction of mass in [0, 0.2] and [0.8, 1]
    low = np.sum((values >= 0) & (values <= 0.2)) / n
    high = np.sum((values >= 0.8) & (values <= 1.0)) / n
    # Bimodal if both tails have substantial mass (e.g. > 15% each)
    bimodal = low >= 0.15 and high >= 0.15
    hist, bin_edges = np.histogram(values, bins=bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] > 0:
            peaks.append((bin_centers[i], hist[i]))
    peaks.sort(key=lambda p: p[1], reverse=True)
    if len(peaks) >= 2:
        p0, p1 = peaks[0][0], peaks[1][0]
        note = f"pct_0-0.2={low:.2f} pct_0.8-1={high:.2f} peaks@{p0:.2f},{p1:.2f}"
    else:
        note = f"pct_0-0.2={low:.2f} pct_0.8-1={high:.2f}"
    return bimodal, note


def main():
    df = load_and_filter()
    if df is None or len(df) == 0:
        sys.exit(1)

    severities = sorted(df["severity"].unique())
    severities = [s for s in severities if 0 <= s <= 4]
    if not severities:
        print("No severity 0–4 in data")
        sys.exit(1)

    # Summary stats: median and quartiles per severity
    print("E_bbox_1.25x - Median and quartiles by severity")
    print("-" * 60)
    stats_rows = []
    for sev in severities:
        s = df.loc[df["severity"] == sev, "e125"]
        q1, med, q3 = quartiles_and_median(s)
        bimodal, note = check_bimodal(s.values)
        n = len(s)
        stats_rows.append((sev, n, q1, med, q3, bimodal, note))
        print(f"  L{sev}:  n={n:5d}   Q1={q1:.4f}   median={med:.4f}   Q3={q3:.4f}   bimodal={bimodal}  ({note})")
    print("-" * 60)

    # Plot: one subplot per severity (distribution)
    n_sev = len(severities)
    n_cols = min(3, n_sev)
    n_rows = (n_sev + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_sev == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, sev in enumerate(severities):
        ax = axes[idx]
        vals = df.loc[df["severity"] == sev, "e125"].values
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=50, range=(0, 1), color="steelblue", alpha=0.7, edgecolor="white", density=True, label="histogram")
        # Add KDE for smooth shape
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals, bw_method=0.1)
            x = np.linspace(0, 1, 200)
            ax.plot(x, kde(x), color="darkblue", linewidth=2, label="KDE")
        except Exception:
            pass
        q1, med, q3 = quartiles_and_median(pd.Series(vals))
        ax.axvline(med, color="red", linestyle="--", linewidth=1.5, label=f"median={med:.3f}")
        ax.axvline(q1, color="gray", linestyle=":", linewidth=1, alpha=0.8)
        ax.axvline(q3, color="gray", linestyle=":", linewidth=1, alpha=0.8)
        bimodal, note = check_bimodal(vals)
        ax.set_title(f"L{sev}  (n={len(vals)})  bimodal={bimodal}")
        ax.set_xlabel("E_bbox_1.25x")
        ax.set_ylabel("density")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper right", fontsize=8)

    for j in range(len(severities), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {OUT_PATH}")

    # Optional: summary table to CSV
    summary_path = RESULTS_DIR / "ebbox_1_25x_summary_by_severity.csv"
    sum_df = pd.DataFrame(
        [
            {"severity": f"L{sev}", "n": n, "Q1": q1, "median": med, "Q3": q3, "bimodal": bimodal, "note": note}
            for (sev, n, q1, med, q3, bimodal, note) in stats_rows
        ]
    )
    sum_df.to_csv(summary_path, index=False)
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
