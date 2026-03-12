"""
Collapse detection: z-score based definition for CAM and Performance.

- CAM collapse: first severity where |z(metric)| >= threshold vs severity-0 baseline.
- Performance collapse: first severity where score_drop_z < -2 or iou_drop_z < -2 vs baseline.
- Lead: t_perf - t_cam (frame/severity index); lead > 0 => CAM collapses first.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any


def compute_cam_baseline_stats(
    cam_df: pd.DataFrame,
    baseline_severity: int = 0,
    group_keys: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
) -> Dict[Tuple, Dict[str, Tuple[float, float]]]:
    """
    Compute per-group (model, corruption, object) baseline mean and std for CAM metrics at severity 0.

    Returns:
        key -> { metric_name: (mean, std) }
    """
    if group_keys is None:
        group_keys = ["model", "corruption", "object_id"]
    if metrics is None:
        metrics = ["energy_in_bbox", "activation_spread", "entropy", "center_shift"]
    # Allow object_uid as fallback
    if "object_id" not in cam_df.columns and "object_uid" in cam_df.columns:
        cam_df = cam_df.rename(columns={"object_uid": "object_id"})
    available = [c for c in group_keys if c in cam_df.columns]
    if not available:
        return {}

    baseline = cam_df[cam_df["severity"] == baseline_severity].copy()
    if len(baseline) == 0:
        return {}

    out: Dict[Tuple, Dict[str, Tuple[float, float]]] = {}
    for key_vals, grp in baseline.groupby(available):
        key = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        out[key] = {}
        for m in metrics:
            if m not in grp.columns:
                continue
            vals = grp[m].dropna()
            if len(vals) < 1:
                continue
            mu, std = float(vals.mean()), float(vals.std())
            if np.isnan(std) or std == 0:
                std = 1e-8
            out[key][m] = (mu, std)
    return out


def detect_cam_collapse_severity(
    cam_df: pd.DataFrame,
    baseline_stats: Dict[Tuple, Dict[str, Tuple[float, float]]],
    z_threshold: float = 2.0,
    group_keys: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    min_metrics_ensemble: int = 2,
    use_ensemble: bool = True,
) -> pd.DataFrame:
    """
    For each (model, corruption, object) time series, find first severity where CAM collapse occurs.
    Collapse: |z| >= z_threshold for at least one metric (or min_metrics_ensemble for ensemble).

    Returns:
        DataFrame with columns: model, corruption, object_id, cam_collapse_severity, cam_collapse_metric.
    """
    if group_keys is None:
        group_keys = ["model", "corruption", "object_id"]
    if metrics is None:
        metrics = ["energy_in_bbox", "activation_spread", "entropy", "center_shift"]
    if "object_id" not in cam_df.columns and "object_uid" in cam_df.columns:
        cam_df = cam_df.copy()
        cam_df["object_id"] = cam_df["object_uid"]

    available = [c for c in group_keys if c in cam_df.columns]
    if not available:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for key_vals, grp in cam_df.groupby(available):
        key = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        base = baseline_stats.get(key, {})
        if not base:
            continue
        grp = grp.sort_values("severity")
        severities = grp["severity"].unique()
        cam_collapse_sev = None
        cam_collapse_metric = None
        for sev in sorted(severities):
            if sev == 0:
                continue
            sev_rows = grp[grp["severity"] == sev]
            if len(sev_rows) == 0:
                continue
            if use_ensemble:
                n_collapsed = 0
                for m in metrics:
                    if m not in base or m not in sev_rows.columns:
                        continue
                    mu, std = base[m]
                    vals = sev_rows[m].dropna()
                    if len(vals) == 0:
                        continue
                    z = (float(vals.mean()) - mu) / std
                    if abs(z) >= z_threshold:
                        n_collapsed += 1
                        if cam_collapse_metric is None:
                            cam_collapse_metric = m
                if n_collapsed >= min_metrics_ensemble:
                    cam_collapse_sev = int(sev)
                    if cam_collapse_metric is None:
                        cam_collapse_metric = "ensemble"
                    break
            else:
                for m in metrics:
                    if m not in base or m not in sev_rows.columns:
                        continue
                    mu, std = base[m]
                    vals = sev_rows[m].dropna()
                    if len(vals) == 0:
                        continue
                    z = (float(vals.mean()) - mu) / std
                    if abs(z) >= z_threshold:
                        cam_collapse_sev = int(sev)
                        cam_collapse_metric = m
                        break
                if cam_collapse_sev is not None:
                    break
        row = {k: v for k, v in zip(available, key)}
        row["cam_collapse_severity"] = cam_collapse_sev
        row["cam_collapse_metric"] = cam_collapse_metric
        rows.append(row)
    return pd.DataFrame(rows)


def compute_perf_baseline_stats(
    det_df: pd.DataFrame,
    group_keys: Optional[List[str]] = None,
) -> Dict[Tuple, Dict[str, Tuple[float, float]]]:
    """Baseline (severity 0) mean and std for score and iou per (model, corruption, object)."""
    if group_keys is None:
        group_keys = ["model", "corruption", "image_id", "class_id"]
    available = [c for c in group_keys if c in det_df.columns]
    if not available:
        return {}

    baseline = det_df[det_df["severity"] == 0].copy()
    matched = baseline[baseline["miss"] == 0]
    if len(matched) == 0:
        return {}

    out: Dict[Tuple, Dict[str, Tuple[float, float]]] = {}
    for key_vals, grp in matched.groupby(available):
        key = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        out[key] = {}
        for col in ["score", "iou"]:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna()
            if len(vals) < 1:
                continue
            mu, std = float(vals.mean()), float(vals.std())
            if np.isnan(std) or std == 0:
                std = 1e-8
            out[key][col] = (mu, std)
    return out


def detect_perf_collapse_severity(
    det_df: pd.DataFrame,
    baseline_stats: Dict[Tuple, Dict[str, Tuple[float, float]]],
    z_threshold: float = -2.0,
    group_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    First severity where score_drop_z < z_threshold or iou_drop_z < z_threshold.
    z = (value_sev - baseline_mean) / baseline_std (drop => negative z).

    Returns:
        DataFrame: model, corruption, image_id, class_id, perf_collapse_severity, perf_collapse_type (score|iou).
    """
    if group_keys is None:
        group_keys = ["model", "corruption", "image_id", "class_id"]
    available = [c for c in group_keys if c in det_df.columns]
    if not available:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for key_vals, grp in det_df.groupby(available):
        key = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        base = baseline_stats.get(key, {})
        if not base:
            continue
        grp = grp.sort_values("severity")
        perf_collapse_sev = None
        perf_collapse_type = None
        for sev in sorted(grp["severity"].unique()):
            if sev == 0:
                continue
            sev_rows = grp[grp["severity"] == sev]
            matched = sev_rows[sev_rows["miss"] == 0]
            if len(matched) == 0:
                continue
            for col, label in [("score", "score"), ("iou", "iou")]:
                if col not in base or col not in matched.columns:
                    continue
                mu, std = base[col]
                vals = matched[col].dropna()
                if len(vals) == 0:
                    continue
                z = (float(vals.mean()) - mu) / std
                if z <= z_threshold:
                    perf_collapse_sev = int(sev)
                    perf_collapse_type = label
                    break
            if perf_collapse_sev is not None:
                break
        row = {k: v for k, v in zip(available, key)}
        row["perf_collapse_severity"] = perf_collapse_sev
        row["perf_collapse_type"] = perf_collapse_type
        rows.append(row)
    return pd.DataFrame(rows)


def compute_lead_table(
    failure_events_df: pd.DataFrame,
    cam_collapse_df: pd.DataFrame,
    perf_collapse_df: pd.DataFrame,
    object_uid_col: str = "object_uid",
    severity_as_time: bool = True,
) -> pd.DataFrame:
    """
    Object-level lead = t_perf - t_cam (severity or frame index).
    lead > 0 => CAM collapsed first (lead); lead == 0 => coincident; lead < 0 => lag.

    failure_events_df must have: model, corruption, and some object identifier (image_id+class_id or object_uid).
    cam_collapse_df: from detect_cam_collapse_severity (object_id or object_uid).
    perf_collapse_df: from detect_perf_collapse_severity (image_id, class_id).
    """
    # Build join key: (model, corruption, object)
    # failure_events: model, corruption, image_id, class_id
    # cam: model, corruption, object_id
    # perf: model, corruption, image_id, class_id
    if failure_events_df is None or len(failure_events_df) == 0:
        return pd.DataFrame()

    fail = failure_events_df.copy()
    if "object_uid" not in fail.columns and "image_id" in fail.columns and "class_id" in fail.columns:
        fail["object_uid"] = fail["image_id"].astype(str) + "_obj_" + fail["class_id"].astype(str)

    # Per-event: we have failure_severity from failure_events as t_perf; need t_cam from cam_collapse
    # Join cam_collapse: by (model, corruption, object_id). object_id in cam may be object_uid.
    cam = cam_collapse_df.copy()
    if "object_uid" not in cam.columns and "object_id" in cam.columns:
        cam["object_uid"] = cam["object_id"]

    perf = perf_collapse_df.copy()

    rows: List[Dict[str, Any]] = []
    for _, ev in fail.iterrows():
        model = ev.get("model")
        corruption = ev.get("corruption")
        obj_uid = ev.get("object_uid")
        image_id = ev.get("image_id")
        class_id = ev.get("class_id")
        t_perf = ev.get("failure_severity")  # performance collapse severity
        failure_type = ev.get("failure_type", "unknown")

        # Resolve t_cam from cam_collapse
        cam_row = cam[(cam["model"] == model) & (cam["corruption"] == corruption) & (cam["object_uid"] == obj_uid)]
        if len(cam_row) == 0 and image_id is not None and class_id is not None:
            cam_row = cam[(cam["model"] == model) & (cam["corruption"] == corruption) &
                         (cam.get("image_id", "") == image_id) & (cam.get("class_id", "").astype(str) == str(class_id))]
        t_cam = None
        cam_metric = None
        if len(cam_row) > 0:
            _t_cam = cam_row["cam_collapse_severity"].iloc[0]
            if pd.notna(_t_cam):
                t_cam = int(_t_cam)
            cam_metric = cam_row["cam_collapse_metric"].iloc[0] if "cam_collapse_metric" in cam_row.columns else None

        # If we use failure_events failure_severity as t_perf, we have it; else get from perf_collapse
        if t_perf is None and perf is not None and len(perf) > 0 and image_id is not None and class_id is not None:
            perf_row = perf[(perf["model"] == model) & (perf["corruption"] == corruption) &
                           (perf["image_id"] == image_id) & (perf["class_id"] == class_id)]
            if len(perf_row) > 0:
                _t = perf_row["perf_collapse_severity"].iloc[0]
                if pd.notna(_t):
                    t_perf = int(_t)
        elif t_perf is not None and pd.isna(t_perf):
            t_perf = None
        else:
            try:
                t_perf = int(t_perf) if t_perf is not None and pd.notna(t_perf) else None
            except (TypeError, ValueError):
                t_perf = None

        lead = None
        alignment = None
        if t_perf is not None and t_cam is not None:
            lead = t_perf - t_cam
            if lead > 0:
                alignment = "lead"
            elif lead == 0:
                alignment = "coincident"
            else:
                alignment = "lag"
        elif t_perf is not None:
            alignment = "cam_missing"

        rows.append({
            "model": model,
            "corruption": corruption,
            "object_uid": obj_uid,
            "failure_type": failure_type,
            "t_perf": t_perf,
            "t_cam": t_cam,
            "lead": lead,
            "alignment": alignment,
            "cam_collapse_metric": cam_metric,
        })
    return pd.DataFrame(rows)
