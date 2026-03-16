"""
Lead analysis: CAM change onset (warning) vs CAM collapse vs Performance collapse.

When use_delta_based: true (recommended):
- t_cam_warn = first severity with ≥1 from axis A and ≥1 from axis B over Q75 of deltas.
- t_cam_collapse = first severity with ≥3 of 4 over Q90 and persistence at next sev.
- lead_warn = t_perf - t_cam_warn, lead_collapse = t_perf - t_cam_collapse.

Otherwise: legacy z-score CAM collapse, lead = t_perf - t_cam.

Outputs: results/lead_table.csv (with lead_warn, lead_collapse), results/lead_stats.json
"""

import sys
import json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.collapse_detection import (
    compute_cam_baseline_stats,
    detect_cam_collapse_severity,
    get_cam_l0_baseline,
    compute_cam_deltas,
    compute_cam_change_score,
    detect_cam_change_onset_by_score,
    compute_delta_thresholds,
    detect_cam_warning_onset,
    detect_cam_collapse_onset,
    compute_lead_table,
    compute_lead_table_with_warn_collapse,
    compute_lead_table_from_cam_change,
    aggregate_onset_lead_survival,
    compute_perf_baseline_stats,
    detect_perf_collapse_severity,
)
from src.eval.lead_statistics import aggregate_lead_stats
from src.utils.io import load_yaml
from src.utils.seed import set_seed


def main():
    root = Path(__file__).resolve().parent.parent
    config_path = root / "configs" / "experiment.yaml"
    if not config_path.exists():
        print("Error: config not found at", config_path)
        sys.exit(1)
    config = load_yaml(config_path)
    set_seed(config.get("seed", 42))

    results_dir = (root / config["results"]["root"]).resolve()
    print(f"Results dir: {results_dir}")
    collapse_cfg = config.get("collapse_detection", {})
    use_delta_based = collapse_cfg.get("use_delta_based", True)
    perf_z = collapse_cfg.get("perf_z_threshold", -2.0)
    cam_z = collapse_cfg.get("cam_z_threshold", 2.0)
    cam_metrics_list = collapse_cfg.get("cam_metrics", ["energy_in_bbox", "activation_spread", "entropy", "center_shift"])
    min_ensemble = collapse_cfg.get("min_metrics_for_ensemble", 2)

    # Delta-based config
    cam_change_method = collapse_cfg.get("cam_change_method", "composite_score")  # composite_score | two_axes
    score_threshold_pct = collapse_cfg.get("score_threshold_percentile", 0.75)  # top 25% for t_cam_change
    if 0 < score_threshold_pct < 1:
        score_threshold_pct *= 100.0
    axis_a = collapse_cfg.get("axis_a_metrics", ["bbox_center_activation_distance", "peak_bbox_distance"])
    axis_b = collapse_cfg.get("axis_b_metrics", ["activation_spread", "ring_energy_ratio"])
    warning_pct = collapse_cfg.get("warning_percentile", 0.75)
    collapse_pct = collapse_cfg.get("collapse_percentile", 0.90)
    min_conditions_collapse = collapse_cfg.get("min_conditions_collapse", 3)
    require_persistence = collapse_cfg.get("collapse_require_persistence", True)
    if 0 < warning_pct < 1:
        warning_pct *= 100.0
    if 0 < collapse_pct < 1:
        collapse_pct *= 100.0

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

    if "object_uid" not in failure_events_df.columns and "image_id" in failure_events_df.columns and "class_id" in failure_events_df.columns:
        failure_events_df = failure_events_df.copy()
        failure_events_df["object_uid"] = (
            failure_events_df["image_id"].astype(str) + "_obj_" + failure_events_df["class_id"].astype(str)
        )

    cam_collapse_df = pd.DataFrame()
    cam_warn_collapse_df = pd.DataFrame()
    cam_change_df = pd.DataFrame()  # composite score onset (t_cam_change)

    if cam_csv.exists() and cam_csv.stat().st_size > 0:
        cam_df = pd.read_csv(cam_csv)
        if "layer_role" in cam_df.columns:
            cam_df = cam_df[cam_df["layer_role"] == "primary"].copy()
        if "cam_status" in cam_df.columns:
            cam_df = cam_df[cam_df["cam_status"] == "ok"].copy()
        if "object_id" not in cam_df.columns and "object_uid" in cam_df.columns:
            cam_df = cam_df.copy()
            cam_df["object_id"] = cam_df["object_uid"]

        if len(cam_df) > 0:
            if use_delta_based:
                metrics_delta = [m for m in axis_a + axis_b if m in cam_df.columns]
                if len(metrics_delta) < 2:
                    print("[WARN] Delta-based: need at least 2 of axis_a+axis_b in cam_records; falling back to legacy.")
                    use_delta_based = False
                else:
                    baseline_l0 = get_cam_l0_baseline(cam_df, group_keys=["model", "corruption", "object_id"], metrics=metrics_delta)
                    if len(baseline_l0) == 0:
                        print("[WARN] No L0 baseline; falling back to legacy.")
                        use_delta_based = False
                    else:
                        deltas_df = compute_cam_deltas(cam_df, baseline_l0, group_keys=["model", "corruption", "object_id"], metrics=metrics_delta)
                        if len(deltas_df) == 0:
                            print("[WARN] No deltas; falling back to legacy.")
                            use_delta_based = False
                        else:
                            if cam_change_method == "composite_score":
                                deltas_df = compute_cam_change_score(deltas_df)
                                cam_change_df = detect_cam_change_onset_by_score(
                                    deltas_df,
                                    score_threshold_percentile=score_threshold_pct,
                                    group_keys=["model", "corruption", "object_id"],
                                )
                                if "object_uid" not in cam_change_df.columns and "object_id" in cam_change_df.columns:
                                    cam_change_df["object_uid"] = cam_change_df["object_id"]
                                print(f"CAM (composite score): t_cam_change for {len(cam_change_df)} objects (threshold top {100 - score_threshold_pct:.0f}%)")

                            thresholds = compute_delta_thresholds(
                                deltas_df,
                                warning_percentile=warning_pct,
                                collapse_percentile=collapse_pct,
                            )
                            warn_df = detect_cam_warning_onset(deltas_df, thresholds, group_keys=["model", "corruption", "object_id"])
                            collapse_df = detect_cam_collapse_onset(
                                deltas_df,
                                thresholds,
                                group_keys=["model", "corruption", "object_id"],
                                min_conditions=min_conditions_collapse,
                                require_persistence=require_persistence,
                            )
                            cam_warn_collapse_df = warn_df.merge(
                                collapse_df,
                                on=["model", "corruption", "object_id"],
                                how="outer",
                            )
                            if "object_uid" not in cam_warn_collapse_df.columns:
                                cam_warn_collapse_df["object_uid"] = cam_warn_collapse_df["object_id"]
                            print(f"CAM (delta-based): t_cam_warn/t_cam_collapse for {len(cam_warn_collapse_df)} objects")

            if not use_delta_based:
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
                print(f"CAM collapse (legacy z-score): {len(cam_collapse_df)} objects")
    else:
        print("No cam_records.csv; skipping CAM. Lead table will have t_cam=None.")

    perf_baseline = compute_perf_baseline_stats(det_df, group_keys=["model", "corruption", "image_id", "class_id"])
    perf_collapse_df = detect_perf_collapse_severity(
        det_df,
        perf_baseline,
        z_threshold=perf_z,
        group_keys=["model", "corruption", "image_id", "class_id"],
    )
    print(f"Performance collapse (z<{perf_z}): {len(perf_collapse_df)} objects")

    if use_delta_based and len(cam_warn_collapse_df) > 0:
        if cam_change_method == "composite_score" and len(cam_change_df) > 0:
            lead_df = compute_lead_table_from_cam_change(
                failure_events_df,
                cam_change_df,
                perf_collapse_df,
                object_uid_col="object_uid",
            )
            # Add t_cam_warn, lead_warn from two-axes for reference
            if "t_cam_warn" in cam_warn_collapse_df.columns:
                lead_df = lead_df.merge(
                    cam_warn_collapse_df[["model", "corruption", "object_uid", "t_cam_warn"]],
                    on=["model", "corruption", "object_uid"],
                    how="left",
                )
                lead_df["lead_warn"] = lead_df.apply(
                    lambda r: (int(r["t_perf"]) - int(r["t_cam_warn"])) if pd.notna(r["t_perf"]) and pd.notna(r["t_cam_warn"]) else None,
                    axis=1,
                )
        else:
            lead_df = compute_lead_table_with_warn_collapse(
                failure_events_df,
                cam_warn_collapse_df,
                perf_collapse_df,
                object_uid_col="object_uid",
            )
    else:
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

        # Object-level dynamics: onset distribution, lead distribution, survival curve
        df_for_onset = cam_change_df if len(cam_change_df) > 0 else None
        if df_for_onset is None and len(cam_warn_collapse_df) > 0 and "t_cam_warn" in cam_warn_collapse_df.columns:
            df_for_onset = cam_warn_collapse_df[["model", "corruption", "object_id", "object_uid"] if "object_uid" in cam_warn_collapse_df.columns else ["model", "corruption", "object_id"]].copy()
            df_for_onset["t_cam_change"] = cam_warn_collapse_df["t_cam_warn"].values
        lead_for_onset = lead_df.copy()
        if "t_cam_change" not in lead_for_onset.columns and "t_cam_warn" in lead_for_onset.columns:
            lead_for_onset["t_cam_change"] = lead_for_onset["t_cam_warn"]
        if "lead" not in lead_for_onset.columns and "lead_warn" in lead_for_onset.columns:
            lead_for_onset["lead"] = lead_for_onset["lead_warn"]
        if df_for_onset is not None and ("t_cam_change" in lead_for_onset.columns or "t_cam_warn" in lead_for_onset.columns):
            onset_lead_survival = aggregate_onset_lead_survival(lead_for_onset, df_for_onset, severities=[0, 1, 2, 3, 4])
            from datetime import datetime
            onset_lead_survival["last_updated"] = datetime.now().isoformat()
            out_path = results_dir / "cam_onset_lead_survival.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(onset_lead_survival, f, indent=2)
            print(f"Saved object-level dynamics to {out_path}")
        if len(cam_change_df) > 0:
            cam_change_csv = results_dir / "cam_change_onset.csv"
            cam_change_df.to_csv(cam_change_csv, index=False)
            print(f"Saved cam_change_onset to {cam_change_csv}")

    # Statistics: primary on lead (t_cam_change) if present, else lead_warn, else lead
    lead_col = "lead"
    if lead_col not in lead_df.columns or lead_df[lead_col].notna().sum() == 0:
        lead_col = "lead_warn" if "lead_warn" in lead_df.columns else "lead"
    lead_series = lead_df[lead_col] if lead_col in lead_df.columns else pd.Series()
    stats = aggregate_lead_stats(lead_df, lead_col=lead_col, n_permutations=10000, random_state=config.get("seed", 42))
    if "lead_collapse" in lead_df.columns:
        stats_collapse = aggregate_lead_stats(lead_df, lead_col="lead_collapse", n_permutations=10000, random_state=config.get("seed", 42))
        stats["lead_collapse_stats"] = stats_collapse

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
    print(f"  mean_lead ({lead_col}) = {stats.get('mean_lead')}")
    print(f"  n_lead / n_coincident / n_lag = {stats.get('n_lead')} / {stats.get('n_coincident')} / {stats.get('n_lag')}")
    if stats.get("sign_test"):
        print(f"  sign_test p_value = {stats['sign_test'].get('p_value')}")
    if stats.get("permutation_test"):
        print(f"  permutation_test p_value = {stats['permutation_test'].get('p_value')}")


if __name__ == "__main__":
    main()
