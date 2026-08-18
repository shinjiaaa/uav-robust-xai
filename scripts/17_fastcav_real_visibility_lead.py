"""Lead/coincident/lag analysis for the visibility CAV (option E).

Visibility score DROPS as corruption increases. Onset is the first severity where
the score has dropped sufficiently from the L0 baseline.

Onset criterion: drop = baseline(L0) - score(L) >= Q75 of all positive drops in that
corruption (matches the heuristic threshold convention).

Outputs:
    results/fastcav_real_visibility_lead_table.csv
    results/fastcav_real_visibility_corruption_summary.csv
    results/fastcav_real_visibility_lead_stats.json
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
    "fog": "score_visible_under_fog",
    "lowlight": "score_visible_under_lowlight",
    "motion_blur": "score_visible_under_motion_blur",
}


def compute_t_cam_visibility(
    scores_df: pd.DataFrame,
    *,
    corruption: str,
    score_col: str,
    threshold_percentile: float = 75.0,
) -> pd.DataFrame:
    """First severity where (baseline - score) >= Q75-of-drops threshold for this corruption."""
    sub = scores_df[scores_df["corruption"] == corruption].copy()
    sub["severity"] = sub["severity"].astype(int)

    base = sub[sub["severity"] == 0][["object_uid", score_col]].rename(
        columns={score_col: "baseline_score"}
    )
    sub = sub.merge(base, on="object_uid", how="left")
    sub["drop"] = sub["baseline_score"] - sub[score_col]

    pos_drops = sub.loc[(sub["severity"] >= 1) & (sub["drop"] > 0), "drop"].dropna()
    threshold = float(np.percentile(pos_drops, threshold_percentile)) if len(pos_drops) else 0.0

    onsets = []
    for ouid, grp in sub.groupby("object_uid"):
        g = grp.sort_values("severity")
        t = None
        for _, r in g.iterrows():
            if int(r["severity"]) == 0:
                continue
            if pd.isna(r["drop"]):
                continue
            if float(r["drop"]) >= threshold:
                t = int(r["severity"])
                break
        onsets.append({
            "object_uid": ouid,
            "corruption": corruption,
            "t_cam_visibility": t,
            "drop_threshold": threshold,
        })
    return pd.DataFrame(onsets)


def main():
    results_root = ROOT / "results"
    scores = pd.read_csv(results_root / "fastcav_real_visibility_concept_scores.csv")
    fe = pd.read_csv(results_root / "failure_events.csv")
    print(f"[load] visibility scores: {len(scores)}, failure_events: {len(fe)}")

    onset_frames = []
    for corr, sc in CONCEPT_FOR_CORRUPTION.items():
        df = compute_t_cam_visibility(scores, corruption=corr, score_col=sc)
        n_onset = int(df["t_cam_visibility"].notna().sum())
        thr = df["drop_threshold"].iloc[0]
        print(f"[onset] {corr}: {len(df)} objects, drop_threshold={thr:.4f}, detected_onsets={n_onset}")
        onset_frames.append(df)
    onset_df = pd.concat(onset_frames, ignore_index=True)

    # ---- Merge with failure_events ----
    fe_keep = fe[["image_id", "corruption", "class_id", "failure_severity", "failure_type"]].copy()
    fe_keep["image_id"] = fe_keep["image_id"].astype(str)
    fe_keep["corruption"] = fe_keep["corruption"].astype(str)
    fe_keep["class_id"] = fe_keep["class_id"].astype(int)
    fe_keep["failure_severity"] = pd.to_numeric(fe_keep["failure_severity"], errors="coerce")

    # Need (image_id, class_id, corruption) → onset. But onset is per (object_uid, corruption).
    # Get image_id and class_id per object_uid from detection_records.
    det = pd.read_csv(results_root / "detection_records.csv", usecols=["object_uid", "image_id", "gt_class_id"])
    det = det.drop_duplicates(subset=["object_uid"])
    det["image_id"] = det["image_id"].astype(str)
    det["class_id"] = det["gt_class_id"].astype(int)

    onset_df = onset_df.merge(det[["object_uid", "image_id", "class_id"]], on="object_uid", how="left")

    # Aggregate object-level onsets to (image_id, corruption, class_id) median for fair join with failure_events
    onset_agg = (
        onset_df.dropna(subset=["t_cam_visibility"])
        .groupby(["image_id", "class_id", "corruption"])["t_cam_visibility"]
        .min()  # earliest onset across instances of same class in same image
        .reset_index()
    )

    merged = fe_keep.merge(onset_agg, on=["image_id", "class_id", "corruption"], how="left")

    def classify(row):
        t_perf = row["failure_severity"]
        t_cam = row["t_cam_visibility"]
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
    merged.to_csv(results_root / "fastcav_real_visibility_lead_table.csv", index=False)
    print(f"[save] fastcav_real_visibility_lead_table.csv  ({len(merged)} rows)")

    # corruption summary
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
    summary_df.to_csv(results_root / "fastcav_real_visibility_corruption_summary.csv", index=False)
    print("[save] fastcav_real_visibility_corruption_summary.csv")
    print(summary_df.to_string(index=False))

    # statistical tests
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
    with open(results_root / "fastcav_real_visibility_lead_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2, default=str)
    print("[save] fastcav_real_visibility_lead_stats.json")
    if "sign_test" in stats_out:
        print(f"  sign_test p = {stats_out['sign_test'].get('p_value')}")
    if "permutation_test" in stats_out:
        print(f"  permutation_test p = {stats_out['permutation_test'].get('p_value')}")


if __name__ == "__main__":
    main()
