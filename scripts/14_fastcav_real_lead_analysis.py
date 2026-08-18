"""Real FastCAV lead/coincident/lag analysis (mirrors RQ3 with proper FastCAV CAV scores).

Uses fastcav_real_concept_scores.csv (per image, corruption, severity) joined with
failure_events.csv (per object, corruption) to define:
    - t_perf  = first severity where the object's performance event fires
    - t_cam_real = first severity where the *matching* concept score's delta
                   exceeds Q75 of the per-corruption delta distribution

Lead = t_perf - t_cam_real, classified as lead / coincident / lag / unavailable.

Outputs:
    results/fastcav_real_lead_table.csv          (per object × corruption row)
    results/fastcav_real_corruption_summary.csv  (aggregate per corruption)
    results/fastcav_real_lead_stats.json         (overall + sign/permutation tests)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.lead_statistics import sign_test_lead, permutation_test_lead


CONCEPT_FOR_CORRUPTION = {
    "fog": "score_fog_present",
    "lowlight": "score_lowlight_present",
    "motion_blur": "score_motion_blur_present",
}


def compute_t_cam_real(
    scores_df: pd.DataFrame,
    *,
    corruption: str,
    score_col: str,
    threshold_percentile: float = 75.0,
) -> pd.DataFrame:
    """Per image_id × corruption: find first severity (>=1) where delta vs L0 >= Q75 threshold.

    Returns DataFrame with columns: image_id, corruption, t_cam_real, delta_threshold.
    """
    sub = scores_df[scores_df["corruption"] == corruption].copy()
    sub["severity"] = sub["severity"].astype(int)

    # baseline (L0) per image
    base = sub[sub["severity"] == 0][["image_id", score_col]].rename(
        columns={score_col: "baseline_score"}
    )
    sub = sub.merge(base, on="image_id", how="left")
    sub["delta"] = sub[score_col] - sub["baseline_score"]

    # Q75 of POSITIVE deltas across (image, severity≥1) for this corruption
    pos_deltas = sub.loc[sub["severity"] >= 1, "delta"].dropna()
    if len(pos_deltas) == 0:
        threshold = 0.0
    else:
        threshold = float(np.percentile(pos_deltas, threshold_percentile))

    onsets = []
    for iid, grp in sub.groupby("image_id"):
        g = grp.sort_values("severity")
        t = None
        for _, r in g.iterrows():
            if int(r["severity"]) == 0:
                continue
            if pd.isna(r["delta"]):
                continue
            if float(r["delta"]) >= threshold:
                t = int(r["severity"])
                break
        onsets.append({
            "image_id": iid,
            "corruption": corruption,
            "t_cam_real": t,
            "delta_threshold": threshold,
        })
    return pd.DataFrame(onsets)


def main():
    results_root = ROOT / "results"
    scores_path = results_root / "fastcav_real_concept_scores.csv"
    fe_path = results_root / "failure_events.csv"

    scores = pd.read_csv(scores_path)
    fe = pd.read_csv(fe_path)
    print(f"[load] scores: {len(scores)} rows, failure_events: {len(fe)} rows")

    # ---- Per-corruption t_cam_real ----
    onset_frames = []
    for corr, score_col in CONCEPT_FOR_CORRUPTION.items():
        df = compute_t_cam_real(scores, corruption=corr, score_col=score_col)
        print(f"[onset] {corr}: {len(df)} images, threshold={df['delta_threshold'].iloc[0]:.4f}, "
              f"detected_onsets={int(df['t_cam_real'].notna().sum())}")
        onset_frames.append(df)
    onset_df = pd.concat(onset_frames, ignore_index=True)

    # ---- Merge with failure events ----
    fe_keep = fe[["image_id", "corruption", "class_id", "failure_severity", "failure_type"]].copy()
    fe_keep["image_id"] = fe_keep["image_id"].astype(str)
    fe_keep["corruption"] = fe_keep["corruption"].astype(str)
    fe_keep["failure_severity"] = pd.to_numeric(fe_keep["failure_severity"], errors="coerce")
    onset_df["image_id"] = onset_df["image_id"].astype(str)
    onset_df["corruption"] = onset_df["corruption"].astype(str)

    merged = fe_keep.merge(onset_df, on=["image_id", "corruption"], how="left")

    # ---- Compute lead ----
    def classify(row):
        t_perf = row["failure_severity"]
        t_cam = row["t_cam_real"]
        if pd.isna(t_perf) or pd.isna(t_cam):
            return pd.Series({"lead": np.nan, "alignment": "unavailable"})
        lead = int(t_perf) - int(t_cam)
        if lead > 0:
            cat = "lead"
        elif lead == 0:
            cat = "coincident"
        else:
            cat = "lag"
        return pd.Series({"lead": float(lead), "alignment": cat})

    merged[["lead", "alignment"]] = merged.apply(classify, axis=1)
    merged.to_csv(results_root / "fastcav_real_lead_table.csv", index=False)
    print(f"[save] fastcav_real_lead_table.csv  ({len(merged)} rows)")

    # ---- Corruption-wise summary ----
    summary_rows = []
    for corr, grp in merged.groupby("corruption"):
        n_total = len(grp)
        n_avail = int((grp["alignment"] != "unavailable").sum())
        n_lead = int((grp["alignment"] == "lead").sum())
        n_coin = int((grp["alignment"] == "coincident").sum())
        n_lag = int((grp["alignment"] == "lag").sum())
        n_unav = int((grp["alignment"] == "unavailable").sum())
        valid_leads = grp.loc[grp["alignment"] != "unavailable", "lead"].dropna()
        mean_lead = float(valid_leads.mean()) if len(valid_leads) else np.nan
        lead_pct = (n_lead / n_avail * 100.0) if n_avail else np.nan
        summary_rows.append({
            "corruption": corr,
            "n_total": n_total,
            "n_available": n_avail,
            "n_lead": n_lead,
            "n_coincident": n_coin,
            "n_lag": n_lag,
            "n_unavailable": n_unav,
            "lead_ratio_pct": lead_pct,
            "mean_lead_steps": mean_lead,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_root / "fastcav_real_corruption_summary.csv", index=False)
    print(f"[save] fastcav_real_corruption_summary.csv")
    print(summary_df.to_string(index=False))

    # ---- Statistical tests on overall lead distribution ----
    leads_all = merged.loc[merged["alignment"] != "unavailable", "lead"].dropna()
    stats_out = {
        "n_total_events": int(len(merged)),
        "n_available": int(len(leads_all)),
        "n_lead": int((leads_all > 0).sum()),
        "n_coincident": int((leads_all == 0).sum()),
        "n_lag": int((leads_all < 0).sum()),
        "mean_lead": float(leads_all.mean()) if len(leads_all) else None,
        "std_lead": float(leads_all.std()) if len(leads_all) else None,
    }
    if len(leads_all) > 0:
        stats_out["sign_test"] = sign_test_lead(leads_all, alternative="greater")
        stats_out["permutation_test"] = permutation_test_lead(
            leads_all, n_permutations=10000, random_state=42, alternative="greater"
        )
    with open(results_root / "fastcav_real_lead_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2, default=str)
    print(f"[save] fastcav_real_lead_stats.json")
    if "sign_test" in stats_out:
        print(f"  sign_test p = {stats_out['sign_test'].get('p_value')}")
    if "permutation_test" in stats_out:
        print(f"  permutation_test p = {stats_out['permutation_test'].get('p_value')}")


if __name__ == "__main__":
    main()
