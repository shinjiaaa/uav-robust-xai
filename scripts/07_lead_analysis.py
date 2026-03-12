"""
Lead analysis: CAM collapse vs Performance collapse (z-score definition), frame/severity-level lead, and statistics.

- Step 1–2: Collapse definitions (z-score) in collapse_detection.py
- Step 3: Object-level t_cam, t_perf, lead = t_perf - t_cam
- Step 4: Sign test and permutation test (lead_statistics.py)
Outputs: results/lead_table.csv, results/lead_stats.json
"""

import sys
import json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.collapse_detection import (
    compute_cam_baseline_stats,
    detect_cam_collapse_severity,
    compute_perf_baseline_stats,
    detect_perf_collapse_severity,
    compute_lead_table,
)
from src.eval.lead_statistics import aggregate_lead_stats
from src.utils.io import load_yaml
from src.utils.seed import set_seed


def main():
    config_path = Path("configs/experiment.yaml")
    if not config_path.exists():
        print("Error: config not found")
        sys.exit(1)
    config = load_yaml(config_path)
    set_seed(config.get("seed", 42))

    results_dir = Path(config["results"]["root"])
    collapse_cfg = config.get("collapse_detection", {})
    cam_z = collapse_cfg.get("cam_z_threshold", 2.0)
    perf_z = collapse_cfg.get("perf_z_threshold", -2.0)
    cam_metrics_list = collapse_cfg.get("cam_metrics", ["energy_in_bbox", "activation_spread", "entropy", "center_shift"])
    min_ensemble = collapse_cfg.get("min_metrics_for_ensemble", 2)

    # Load data
    det_csv = results_dir / "detection_records.csv"
    cam_csv = results_dir / "cam_records.csv"
    fail_csv = results_dir / "failure_events.csv"

    if not det_csv.exists() or det_csv.stat().st_size == 0:
        print("Error: detection_records.csv not found or empty. Run 03 and 04 first.")
        sys.exit(1)
    if not fail_csv.exists() or fail_csv.stat().st_size == 0:
        print("Error: failure_events.csv not found or empty. Run 04 first.")
        sys.exit(1)

    det_df = pd.read_csv(det_csv)
    failure_events_df = pd.read_csv(fail_csv)

    # object_uid for failure_events (04 may not write it)
    if "object_uid" not in failure_events_df.columns and "image_id" in failure_events_df.columns and "class_id" in failure_events_df.columns:
        failure_events_df = failure_events_df.copy()
        failure_events_df["object_uid"] = (
            failure_events_df["image_id"].astype(str) + "_obj_" + failure_events_df["class_id"].astype(str)
        )

    # CAM collapse (if cam_records exist)
    cam_collapse_df = pd.DataFrame()
    if cam_csv.exists() and cam_csv.stat().st_size > 0:
        cam_df = pd.read_csv(cam_csv)
        # Use primary layer only
        if "layer_role" in cam_df.columns:
            cam_df = cam_df[cam_df["layer_role"] == "primary"].copy()
        if "cam_status" in cam_df.columns:
            cam_df = cam_df[cam_df["cam_status"] == "ok"].copy()
        if len(cam_df) > 0:
            baseline_cam = compute_cam_baseline_stats(
                cam_df,
                baseline_severity=0,
                group_keys=["model", "corruption", "object_id"],
                metrics=[m for m in cam_metrics_list if m in cam_df.columns],
            )
            cam_collapse_df = detect_cam_collapse_severity(
                cam_df,
                baseline_cam,
                z_threshold=cam_z,
                group_keys=["model", "corruption", "object_id"],
                metrics=[m for m in cam_metrics_list if m in cam_df.columns],
                min_metrics_ensemble=min_ensemble,
                use_ensemble=True,
            )
            if "object_uid" not in cam_collapse_df.columns and "object_id" in cam_collapse_df.columns:
                cam_collapse_df["object_uid"] = cam_collapse_df["object_id"]
            print(f"CAM collapse: {len(cam_collapse_df)} objects with collapse severity")
    else:
        print("No cam_records.csv; skipping CAM collapse. Lead table will have t_cam=None.")

    # Performance collapse (z-score from detection_records)
    perf_baseline = compute_perf_baseline_stats(det_df, group_keys=["model", "corruption", "image_id", "class_id"])
    perf_collapse_df = detect_perf_collapse_severity(
        det_df,
        perf_baseline,
        z_threshold=perf_z,
        group_keys=["model", "corruption", "image_id", "class_id"],
    )
    print(f"Performance collapse (z<{perf_z}): {len(perf_collapse_df)} objects with collapse severity")

    # Lead table: t_perf from failure_events.failure_severity, t_cam from cam_collapse
    lead_df = compute_lead_table(
        failure_events_df,
        cam_collapse_df,
        perf_collapse_df,
        object_uid_col="object_uid",
        severity_as_time=True,
    )
    if len(lead_df) == 0:
        print("No lead table rows (failure_events may have no matching object_uid).")
    else:
        lead_csv = results_dir / "lead_table.csv"
        lead_df.to_csv(lead_csv, index=False)
        print(f"Saved {len(lead_df)} rows to {lead_csv}")

    # Statistics (only on rows with numeric lead)
    lead_series = lead_df["lead"] if "lead" in lead_df.columns else pd.Series()
    stats = aggregate_lead_stats(lead_df, lead_col="lead", n_permutations=10000, random_state=config.get("seed", 42))

    # JSON-serializable
    def _nan_default(obj):
        if isinstance(obj, dict):
            return {k: _nan_default(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_nan_default(x) for x in obj]
        if isinstance(obj, float) and (obj != obj or abs(obj) == float("inf")):
            return None
        return obj

    stats_ser = _nan_default(stats)
    stats_json = results_dir / "lead_stats.json"
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats_ser, f, indent=2)
    print(f"Saved lead_stats to {stats_json}")
    print(f"  mean_lead = {stats.get('mean_lead')}")
    print(f"  n_lead / n_coincident / n_lag = {stats.get('n_lead')} / {stats.get('n_coincident')} / {stats.get('n_lag')}")
    if stats.get("sign_test"):
        print(f"  sign_test p_value = {stats['sign_test'].get('p_value')}")
    if stats.get("permutation_test"):
        print(f"  permutation_test p_value = {stats['permutation_test'].get('p_value')}")


if __name__ == "__main__":
    main()
