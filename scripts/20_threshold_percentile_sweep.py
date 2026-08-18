"""Percentile-threshold sweep for the CAM composite-score onset definition.

Reuses the same pipeline as scripts/07_lead_analysis.py (composite_score path)
and re-runs detect_cam_change_onset_by_score for several percentiles, then
recomputes the lead table and aggregate stats for each.

Outputs:
- results/threshold_percentile_sweep.csv  (one row per percentile)
- prints a small markdown table to stdout
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.collapse_detection import (
    compute_cam_deltas,
    compute_lead_table_from_cam_change,
    detect_perf_collapse_severity,
    compute_perf_baseline_stats,
    get_cam_l0_baseline,
)


def compute_cam_change_score_fixed(deltas_df: pd.DataFrame) -> pd.DataFrame:
    """Inline replacement: comp starts at 0 (not NaN). The shipped version has a
    bug that leaves every score NaN."""
    df = deltas_df.copy()
    comp = np.zeros(len(df), dtype=float)
    for col, use_neg in [
        ("delta_bbox_dist", False),
        ("delta_peak_dist", False),
        ("delta_spread", False),
        ("delta_ring", True),
    ]:
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        if use_neg:
            vals = -vals
        mu = np.nanmean(vals)
        std = np.nanstd(vals)
        if std == 0 or np.isnan(std):
            std = 1e-8
        z = (vals - mu) / std
        np.putmask(z, np.isnan(vals), 0.0)
        comp += z
    df["cam_change_score"] = comp
    return df


def detect_onset_by_score(
    deltas_df: pd.DataFrame, percentile: float, group_keys
) -> pd.DataFrame:
    available = [c for c in group_keys if c in deltas_df.columns]
    scores = deltas_df["cam_change_score"].dropna()
    threshold = float(np.percentile(scores, float(percentile)))
    rows = []
    for key_vals, grp in deltas_df.groupby(available):
        key = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        grp = grp.sort_values("severity")
        t_cam = None
        for _, row in grp.iterrows():
            sc = row.get("cam_change_score")
            if pd.notna(sc) and float(sc) >= threshold:
                t_cam = int(row["severity"])
                break
        row_out = {k: v for k, v in zip(available, key)}
        row_out["t_cam_change"] = t_cam
        row_out["cam_change_score_threshold"] = threshold
        rows.append(row_out)
    return pd.DataFrame(rows)
from src.eval.lead_statistics import aggregate_lead_stats
from src.utils.io import load_yaml
from src.utils.seed import set_seed


PERCENTILES = [50, 60, 70, 75, 80, 90]


def run_one(deltas_df, perf_collapse_df, failure_events_df, percentile, seed):
    cam_change_df = detect_onset_by_score(
        deltas_df,
        percentile=float(percentile),
        group_keys=["model", "corruption", "object_id"],
    )
    if "object_uid" not in cam_change_df.columns and "object_id" in cam_change_df.columns:
        cam_change_df["object_uid"] = cam_change_df["object_id"]

    lead_df = compute_lead_table_from_cam_change(
        failure_events_df,
        cam_change_df,
        perf_collapse_df,
        object_uid_col="object_uid",
    )
    stats = aggregate_lead_stats(
        lead_df, lead_col="lead", n_permutations=10000, random_state=seed
    )
    by_corr = (
        lead_df.dropna(subset=["lead"]).groupby("corruption")["lead"].agg(["mean", "count"])
        if "corruption" in lead_df.columns
        else pd.DataFrame()
    )
    lead_pos = lead_df.dropna(subset=["lead"]).copy()
    lead_pct_by_corr = {}
    for corr, sub in lead_pos.groupby("corruption"):
        n_lead = (sub["lead"] > 0).sum()
        n_total = len(sub)
        lead_pct_by_corr[corr] = (
            100.0 * n_lead / n_total if n_total > 0 else float("nan")
        )

    return {
        "percentile": percentile,
        "n_total": stats.get("n_total"),
        "n_with_lead": (
            stats.get("n_lead", 0)
            + stats.get("n_coincident", 0)
            + stats.get("n_lag", 0)
        ),
        "n_cam_missing": stats.get("n_cam_missing"),
        "n_lead": stats.get("n_lead"),
        "n_coincident": stats.get("n_coincident"),
        "n_lag": stats.get("n_lag"),
        "lead_pct": (
            100.0
            * stats.get("n_lead", 0)
            / max(
                1,
                stats.get("n_lead", 0)
                + stats.get("n_coincident", 0)
                + stats.get("n_lag", 0),
            )
        ),
        "mean_lead": stats.get("mean_lead"),
        "std_lead": stats.get("std_lead"),
        "sign_p": (stats.get("sign_test") or {}).get("p_value"),
        "perm_p": (stats.get("permutation_test") or {}).get("p_value"),
        "lead_pct_fog": lead_pct_by_corr.get("fog"),
        "lead_pct_lowlight": lead_pct_by_corr.get("lowlight"),
        "lead_pct_motion_blur": lead_pct_by_corr.get("motion_blur"),
        "mean_lead_fog": float(by_corr.loc["fog", "mean"]) if "fog" in by_corr.index else None,
        "mean_lead_lowlight": float(by_corr.loc["lowlight", "mean"]) if "lowlight" in by_corr.index else None,
        "mean_lead_motion_blur": float(by_corr.loc["motion_blur", "mean"]) if "motion_blur" in by_corr.index else None,
    }


def main():
    root = Path(__file__).resolve().parent.parent
    config = load_yaml(root / "configs" / "experiment.yaml")
    seed = config.get("seed", 42)
    set_seed(seed)

    results_dir = (root / config["results"]["root"]).resolve()
    cfg = config.get("collapse_detection", {})
    axis_a = cfg.get(
        "axis_a_metrics",
        ["bbox_center_activation_distance", "peak_bbox_distance"],
    )
    axis_b = cfg.get("axis_b_metrics", ["activation_spread", "ring_energy_ratio"])
    perf_z = cfg.get("perf_z_threshold", -2.0)

    det_df = pd.read_csv(results_dir / "detection_records.csv")
    cam_df = pd.read_csv(results_dir / "cam_records.csv")
    failure_events_df = pd.read_csv(results_dir / "failure_events.csv")

    if "layer_role" in cam_df.columns:
        cam_df = cam_df[cam_df["layer_role"] == "primary"].copy()
    if "cam_status" in cam_df.columns:
        cam_df = cam_df[cam_df["cam_status"] == "ok"].copy()
    if "object_id" not in cam_df.columns and "object_uid" in cam_df.columns:
        cam_df["object_id"] = cam_df["object_uid"]
    if "object_uid" not in failure_events_df.columns and {"image_id", "class_id"}.issubset(failure_events_df.columns):
        failure_events_df["object_uid"] = (
            failure_events_df["image_id"].astype(str)
            + "_obj_"
            + failure_events_df["class_id"].astype(str)
        )

    metrics_delta = [m for m in axis_a + axis_b if m in cam_df.columns]
    baseline_l0 = get_cam_l0_baseline(
        cam_df,
        group_keys=["model", "corruption", "object_id"],
        metrics=metrics_delta,
    )
    deltas_df = compute_cam_deltas(
        cam_df,
        baseline_l0,
        group_keys=["model", "corruption", "object_id"],
        metrics=metrics_delta,
    )
    deltas_df = compute_cam_change_score_fixed(deltas_df)
    print(f"Loaded {len(cam_df)} CAM rows; {len(deltas_df)} delta rows")

    perf_baseline = compute_perf_baseline_stats(
        det_df, group_keys=["model", "corruption", "image_id", "class_id"]
    )
    perf_collapse_df = detect_perf_collapse_severity(
        det_df,
        perf_baseline,
        z_threshold=perf_z,
        group_keys=["model", "corruption", "image_id", "class_id"],
    )

    rows = []
    for p in PERCENTILES:
        print(f"\n=== Percentile {p} ===")
        row = run_one(deltas_df, perf_collapse_df, failure_events_df, p, seed)
        for k, v in row.items():
            print(f"  {k}: {v}")
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_csv = results_dir / "threshold_percentile_sweep.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved sweep to {out_csv}")


if __name__ == "__main__":
    main()
