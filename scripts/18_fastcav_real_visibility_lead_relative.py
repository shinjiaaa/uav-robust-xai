"""Lead analysis with relative-15% onset (matches heuristic threshold convention).

Same as scripts/17 but uses baseline-relative drop ≥ 15% as the onset criterion,
making the result directly comparable to the heuristic CAM-derived score in script 11.

Outputs (separate files; do NOT overwrite the Q75 versions):
    results/fastcav_real_visibility_lead_table_rel15.csv
    results/fastcav_real_visibility_corruption_summary_rel15.csv
    results/fastcav_real_visibility_lead_stats_rel15.json
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
RELATIVE_DROP_THRESHOLD = 0.15  # matches heuristic CHANGE_THRESHOLD in scripts/11


def compute_t_cam_visibility_relative(scores_df, *, corruption, score_col, threshold=RELATIVE_DROP_THRESHOLD):
    sub = scores_df[scores_df["corruption"] == corruption].copy()
    sub["severity"] = sub["severity"].astype(int)
    base = sub[sub["severity"] == 0][["object_uid", score_col]].rename(columns={score_col: "baseline_score"})
    sub = sub.merge(base, on="object_uid", how="left")
    sub["rel_drop"] = (sub["baseline_score"] - sub[score_col]) / sub["baseline_score"].abs().clip(lower=1e-6)
    onsets = []
    for ouid, grp in sub.groupby("object_uid"):
        g = grp.sort_values("severity")
        t = None
        for _, r in g.iterrows():
            if int(r["severity"]) == 0:
                continue
            if pd.notna(r["rel_drop"]) and float(r["rel_drop"]) >= threshold:
                t = int(r["severity"])
                break
        onsets.append({
            "object_uid": ouid, "corruption": corruption, "t_cam_visibility": t,
            "rel_drop_threshold": threshold,
        })
    return pd.DataFrame(onsets)


def main():
    results_root = ROOT / "results"
    scores = pd.read_csv(results_root / "fastcav_real_visibility_concept_scores.csv")
    fe = pd.read_csv(results_root / "failure_events.csv")
    det = pd.read_csv(results_root / "detection_records.csv", usecols=["object_uid", "image_id", "gt_class_id"]).drop_duplicates(subset=["object_uid"])
    det["class_id"] = det["gt_class_id"].astype(int)
    det["image_id"] = det["image_id"].astype(str)
    fe["image_id"] = fe["image_id"].astype(str)
    fe["class_id"] = fe["class_id"].astype(int)
    print(f"[load] visibility scores: {len(scores)}, failure_events: {len(fe)}")

    onset_frames = []
    for corr, sc in CONCEPT_FOR_CORRUPTION.items():
        df = compute_t_cam_visibility_relative(scores, corruption=corr, score_col=sc)
        n = int(df["t_cam_visibility"].notna().sum())
        print(f"[onset] {corr}: {len(df)} objects, threshold=15% relative, detected_onsets={n}")
        onset_frames.append(df)
    onset_df = pd.concat(onset_frames, ignore_index=True)
    onset_df = onset_df.merge(det[["object_uid", "image_id", "class_id"]], on="object_uid", how="left")
    onset_agg = onset_df.dropna(subset=["t_cam_visibility"]).groupby(["image_id", "class_id", "corruption"])["t_cam_visibility"].min().reset_index()

    fe_keep = fe[["image_id", "corruption", "class_id", "failure_severity", "failure_type"]].copy()
    fe_keep["failure_severity"] = pd.to_numeric(fe_keep["failure_severity"], errors="coerce")
    merged = fe_keep.merge(onset_agg, on=["image_id", "class_id", "corruption"], how="left")

    def classify(row):
        t_perf, t_cam = row["failure_severity"], row["t_cam_visibility"]
        if pd.isna(t_perf) or pd.isna(t_cam):
            return pd.Series({"lead": np.nan, "alignment": "unavailable"})
        lead = int(t_perf) - int(t_cam)
        cat = "lead" if lead > 0 else ("coincident" if lead == 0 else "lag")
        return pd.Series({"lead": float(lead), "alignment": cat})

    merged[["lead", "alignment"]] = merged.apply(classify, axis=1)
    merged.to_csv(results_root / "fastcav_real_visibility_lead_table_rel15.csv", index=False)
    print(f"[save] fastcav_real_visibility_lead_table_rel15.csv  ({len(merged)} rows)")

    summary_rows = []
    for corr, grp in merged.groupby("corruption"):
        n_total = len(grp)
        n_avail = int((grp["alignment"] != "unavailable").sum())
        n_lead = int((grp["alignment"] == "lead").sum())
        n_coin = int((grp["alignment"] == "coincident").sum())
        n_lag = int((grp["alignment"] == "lag").sum())
        valid_leads = grp.loc[grp["alignment"] != "unavailable", "lead"].dropna()
        summary_rows.append({
            "corruption": corr, "n_total": n_total, "n_available": n_avail,
            "n_lead": n_lead, "n_coincident": n_coin, "n_lag": n_lag,
            "n_unavailable": int((grp["alignment"] == "unavailable").sum()),
            "lead_ratio_pct": (n_lead / n_avail * 100.0) if n_avail else np.nan,
            "mean_lead_steps": float(valid_leads.mean()) if len(valid_leads) else np.nan,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_root / "fastcav_real_visibility_corruption_summary_rel15.csv", index=False)
    print("[save] fastcav_real_visibility_corruption_summary_rel15.csv")
    print(summary_df.to_string(index=False))

    leads_all = merged.loc[merged["alignment"] != "unavailable", "lead"].dropna()
    stats_out = {
        "threshold_definition": "baseline-relative drop >= 0.15",
        "n_available": int(len(leads_all)),
        "n_lead": int((leads_all > 0).sum()),
        "n_coincident": int((leads_all == 0).sum()),
        "n_lag": int((leads_all < 0).sum()),
        "mean_lead": float(leads_all.mean()),
        "std_lead": float(leads_all.std()),
        "sign_test": sign_test_lead(leads_all, alternative="greater"),
        "permutation_test": permutation_test_lead(leads_all, n_permutations=10000, random_state=42, alternative="greater"),
    }
    with open(results_root / "fastcav_real_visibility_lead_stats_rel15.json", "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2, default=str)
    print(f"[save] fastcav_real_visibility_lead_stats_rel15.json")
    print(f"  sign_test p = {stats_out['sign_test'].get('p_value')}")
    print(f"  permutation_test p = {stats_out['permutation_test'].get('p_value')}")


if __name__ == "__main__":
    main()
