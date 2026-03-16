"""
Collapse detection: CAM change onset (warning) vs CAM collapse, and Performance collapse.

Concepts (separated):
- CAM change onset (t_cam_warn): attention meaningfully deviates from baseline (sensitive, for lead_warn).
- CAM collapse (t_cam_collapse): object-centeredness lost and persistent (conservative, for lead_collapse).
- Lead: lead_warn = t_perf - t_cam_warn, lead_collapse = t_perf - t_cam_collapse.

Delta-based definition (recommended):
- Baseline: per-object L0; use deltas Δ = value(sev) - value(L0).
- Two axes: A = position displacement (bbox_dist, peak_dist), B = diffusion/object-centeredness (spread, ring_ratio).
- Warning: first sev where ≥1 from A and ≥1 from B exceed Q3 of respective delta distributions.
- Collapse: first sev where ≥3 of 4 exceed Q90 and state persists at next sev.

Legacy: z-score CAM collapse (use_delta_based: false) and performance collapse (z < perf_z_threshold).
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


# ---------------------------------------------------------------------------
# Delta-based CAM warning (change onset) and collapse (strict)
# ---------------------------------------------------------------------------

# Short names for the four metrics (columns in cam_df)
_DELTA_AXIS_A = ["bbox_center_activation_distance", "peak_bbox_distance"]
_DELTA_AXIS_B = ["activation_spread", "ring_energy_ratio"]
_DELTA_METRICS = _DELTA_AXIS_A + _DELTA_AXIS_B
# For ring_energy_ratio, "worse" = drop (negative delta); threshold on -delta
_RING_METRIC = "ring_energy_ratio"


def get_cam_l0_baseline(
    cam_df: pd.DataFrame,
    group_keys: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Per (model, corruption, object_id) baseline: mean of each metric at severity 0.
    Returns one row per group with columns model, corruption, object_id, and {m}_l0 for each metric.
    """
    if group_keys is None:
        group_keys = ["model", "corruption", "object_id"]
    if metrics is None:
        metrics = _DELTA_METRICS.copy()
    df = cam_df.copy()
    if "object_id" not in df.columns and "object_uid" in df.columns:
        df["object_id"] = df["object_uid"]
    available = [c for c in group_keys if c in df.columns]
    if not available:
        return pd.DataFrame()
    baseline = df[df["severity"] == 0]
    if len(baseline) == 0:
        return pd.DataFrame()
    agg = {m: "mean" for m in metrics if m in baseline.columns}
    if not agg:
        return pd.DataFrame()
    out = baseline.groupby(available, as_index=False).agg(agg)
    out = out.rename(columns={m: f"{m}_l0" for m in metrics if m in out.columns})
    return out


def compute_cam_deltas(
    cam_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    group_keys: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    For each (object, severity) with severity > 0, compute delta = value(sev) - value(L0).
    Returns DataFrame with model, corruption, object_id, severity, and delta_* columns.
    """
    if group_keys is None:
        group_keys = ["model", "corruption", "object_id"]
    if metrics is None:
        metrics = _DELTA_METRICS.copy()
    df = cam_df.copy()
    if "object_id" not in df.columns and "object_uid" in df.columns:
        df["object_id"] = df["object_uid"]
    available = [c for c in group_keys if c in df.columns]
    if not available:
        return pd.DataFrame()
    # One value per (group, severity): aggregate if multiple rows (e.g. xai_method)
    agg = {m: "mean" for m in metrics if m in df.columns}
    if not agg:
        return pd.DataFrame()
    by_sev = df.groupby(available + ["severity"], as_index=False).agg(agg)
    # Merge L0
    base_cols = [c for c in baseline_df.columns if c.endswith("_l0")]
    if not base_cols:
        return pd.DataFrame()
    merge_df = baseline_df[[c for c in available if c in baseline_df.columns] + base_cols].copy()
    merged = by_sev.merge(merge_df, on=available, how="inner")
    # Compute deltas only for severity > 0
    merged = merged[merged["severity"] > 0].copy()
    # Short names for delta columns
    def _delta_col(m: str) -> str:
        if m == _RING_METRIC:
            return "delta_ring"
        if m == "bbox_center_activation_distance":
            return "delta_bbox_dist"
        if m == "peak_bbox_distance":
            return "delta_peak_dist"
        if m == "activation_spread":
            return "delta_spread"
        return f"delta_{m}"
    for m in metrics:
        l0_col = f"{m}_l0"
        if m not in merged.columns or l0_col not in merged.columns:
            continue
        merged[_delta_col(m)] = merged[m] - merged[l0_col]
    if "ring_energy_ratio" in merged.columns and "delta_ring" not in merged.columns and "ring_energy_ratio_l0" in merged.columns:
        merged["delta_ring"] = merged["ring_energy_ratio"] - merged["ring_energy_ratio_l0"]
    return merged


def compute_delta_thresholds(
    deltas_df: pd.DataFrame,
    axis_a: Optional[List[str]] = None,
    axis_b: Optional[List[str]] = None,
    warning_percentile: float = 75.0,
    collapse_percentile: float = 90.0,
) -> Dict[str, float]:
    """
    Build thresholds from delta distributions. For bbox_dist, peak_dist, spread: higher delta = worse.
    For ring_ratio: lower = worse, so we use percentile of (-delta_ring) as T_r; condition is delta_ring < -T_r.
    Returns T_*_warn (e.g. Q75) and T_*_collapse (e.g. Q90) for each metric.
    """
    out: Dict[str, float] = {}
    for col, key in [
        ("delta_bbox_dist", "bbox_dist"),
        ("delta_peak_dist", "peak_dist"),
        ("delta_spread", "spread"),
    ]:
        if col not in deltas_df.columns:
            continue
        vals = deltas_df[col].dropna()
        if len(vals) < 2:
            continue
        out[f"T_{key}_warn"] = float(np.percentile(vals, warning_percentile))
        out[f"T_{key}_collapse"] = float(np.percentile(vals, collapse_percentile))
    if "delta_ring" in deltas_df.columns:
        vals = deltas_df["delta_ring"].dropna()
        if len(vals) >= 2:
            neg = -vals  # drop amount
            out["T_ring_warn"] = float(np.percentile(neg, warning_percentile))
            out["T_ring_collapse"] = float(np.percentile(neg, collapse_percentile))
    return out


def compute_cam_change_score(deltas_df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite CAM change score = z(Δbbox_dist) + z(Δpeak_dist) + z(Δspread) + z(-Δring).
    Each z is computed over all (object, severity) rows; higher score = more attention collapse.
    Returns deltas_df with added column 'cam_change_score'.
    """
    df = deltas_df.copy()
    comp = np.zeros(len(df), dtype=float)
    comp[:] = np.nan
    n = 0
    for col, use_neg in [
        ("delta_bbox_dist", False),
        ("delta_peak_dist", False),
        ("delta_spread", False),
        ("delta_ring", True),  # z(-Δring): drop = worse
    ]:
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        if use_neg:
            vals = -vals
        mu, std = np.nanmean(vals), np.nanstd(vals)
        if std == 0 or np.isnan(std):
            std = 1e-8
        z = (vals - mu) / std
        np.putmask(z, np.isnan(vals), 0.0)
        comp += z
        n += 1
    df["cam_change_score"] = comp
    return df


def detect_cam_change_onset_by_score(
    deltas_df: pd.DataFrame,
    score_threshold_percentile: float = 75.0,
    group_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Per object: first severity where cam_change_score >= threshold.
    Threshold = percentile (e.g. 75 = top 25%) of the score distribution across all rows.
    Returns DataFrame with model, corruption, object_id, t_cam_change.
    """
    if "cam_change_score" not in deltas_df.columns:
        deltas_df = compute_cam_change_score(deltas_df)
    if group_keys is None:
        group_keys = ["model", "corruption", "object_id"]
    available = [c for c in group_keys if c in deltas_df.columns]
    if not available:
        return pd.DataFrame()
    scores = deltas_df["cam_change_score"].dropna()
    if len(scores) < 2:
        return pd.DataFrame()
    if 0 < score_threshold_percentile < 1:
        score_threshold_percentile *= 100.0
    threshold = float(np.percentile(scores, score_threshold_percentile))
    rows: List[Dict[str, Any]] = []
    for key_vals, grp in deltas_df.groupby(available):
        key = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        grp = grp.sort_values("severity")
        t_cam_change = None
        for _, row in grp.iterrows():
            sc = row.get("cam_change_score")
            if pd.notna(sc) and float(sc) >= threshold:
                t_cam_change = int(row["severity"])
                break
        row_out = {k: v for k, v in zip(available, key)}
        row_out["t_cam_change"] = t_cam_change
        row_out["cam_change_score_threshold"] = threshold
        rows.append(row_out)
    return pd.DataFrame(rows)


def _count_conditions_collapse(
    row: pd.Series,
    thresholds: Dict[str, float],
) -> int:
    """Return count (0--4) of conditions over collapse threshold."""
    n = 0
    for key in ["T_bbox_dist_collapse", "T_peak_dist_collapse", "T_spread_collapse", "T_ring_collapse"]:
        T = thresholds.get(key)
        if T is None:
            continue
        if "ring" in key:
            val = row.get("delta_ring")
            if val is not None and not np.isnan(val) and val < -T:
                n += 1
        else:
            dkey = "delta_bbox_dist" if "bbox" in key else "delta_peak_dist" if "peak" in key else "delta_spread"
            val = row.get(dkey)
            if val is not None and not np.isnan(val) and val > T:
                n += 1
    return n


def detect_cam_warning_onset(
    deltas_df: pd.DataFrame,
    thresholds: Dict[str, float],
    group_keys: Optional[List[str]] = None,
    axis_a: Optional[List[str]] = None,
    axis_b: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Per object: first severity where at least 1 condition from axis A and 1 from axis B
    exceed warning threshold (Q3). Returns DataFrame with model, corruption, object_id, t_cam_warn.
    """
    if group_keys is None:
        group_keys = ["model", "corruption", "object_id"]
    available = [c for c in group_keys if c in deltas_df.columns]
    if not available:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for key_vals, grp in deltas_df.groupby(available):
        key = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        grp = grp.sort_values("severity")
        t_cam_warn = None
        for _, row in grp.iterrows():
            na = 0
            if "delta_bbox_dist" in row.index:
                T = thresholds.get("T_bbox_dist_warn")
                if T is not None and pd.notna(row["delta_bbox_dist"]) and float(row["delta_bbox_dist"]) > T:
                    na += 1
            if "delta_peak_dist" in row.index:
                T = thresholds.get("T_peak_dist_warn")
                if T is not None and pd.notna(row["delta_peak_dist"]) and float(row["delta_peak_dist"]) > T:
                    na += 1
            nb = 0
            if "delta_spread" in row.index:
                T = thresholds.get("T_spread_warn")
                if T is not None and pd.notna(row["delta_spread"]) and float(row["delta_spread"]) > T:
                    nb += 1
            if "delta_ring" in row.index:
                T = thresholds.get("T_ring_warn")
                if T is not None and pd.notna(row["delta_ring"]) and float(row["delta_ring"]) < -T:
                    nb += 1
            if na >= 1 and nb >= 1:
                t_cam_warn = int(row["severity"])
                break
        row_out = {k: v for k, v in zip(available, key)}
        row_out["t_cam_warn"] = t_cam_warn
        rows.append(row_out)
    return pd.DataFrame(rows)


def detect_cam_collapse_onset(
    deltas_df: pd.DataFrame,
    thresholds: Dict[str, float],
    group_keys: Optional[List[str]] = None,
    min_conditions: int = 3,
    require_persistence: bool = True,
) -> pd.DataFrame:
    """
    Per object: first severity where at least min_conditions of 4 exceed collapse threshold (Q90),
    and (if require_persistence) the next severity also meets or there is no next severity.
    Returns DataFrame with model, corruption, object_id, t_cam_collapse.
    """
    if group_keys is None:
        group_keys = ["model", "corruption", "object_id"]
    available = [c for c in group_keys if c in deltas_df.columns]
    if not available:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for key_vals, grp in deltas_df.groupby(available):
        key = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        grp = grp.sort_values("severity")
        sevs = grp["severity"].tolist()
        counts = [_count_conditions_collapse(grp.iloc[i], thresholds) for i in range(len(grp))]
        t_cam_collapse = None
        for i, sev in enumerate(sevs):
            if counts[i] < min_conditions:
                continue
            if require_persistence:
                if i + 1 < len(sevs):
                    if counts[i + 1] < min_conditions:
                        continue
            t_cam_collapse = int(sev)
            break
        row_out = {k: v for k, v in zip(available, key)}
        row_out["t_cam_collapse"] = t_cam_collapse
        rows.append(row_out)
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


def compute_lead_table_with_warn_collapse(
    failure_events_df: pd.DataFrame,
    cam_warn_collapse_df: pd.DataFrame,
    perf_collapse_df: pd.DataFrame,
    object_uid_col: str = "object_uid",
) -> pd.DataFrame:
    """
    Like compute_lead_table but cam_warn_collapse_df has t_cam_warn and t_cam_collapse.
    Returns lead_warn = t_perf - t_cam_warn, lead_collapse = t_perf - t_cam_collapse.
    Also sets t_cam = t_cam_warn, lead = lead_warn, alignment from lead_warn for backward compat.
    """
    if failure_events_df is None or len(failure_events_df) == 0:
        return pd.DataFrame()
    fail = failure_events_df.copy()
    if "object_uid" not in fail.columns and "image_id" in fail.columns and "class_id" in fail.columns:
        fail["object_uid"] = fail["image_id"].astype(str) + "_obj_" + fail["class_id"].astype(str)
    cam = cam_warn_collapse_df.copy()
    if "object_uid" not in cam.columns and "object_id" in cam.columns:
        cam["object_uid"] = cam["object_id"]
    perf = perf_collapse_df.copy() if perf_collapse_df is not None else pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, ev in fail.iterrows():
        model = ev.get("model")
        corruption = ev.get("corruption")
        obj_uid = ev.get("object_uid")
        image_id = ev.get("image_id")
        class_id = ev.get("class_id")
        t_perf = ev.get("failure_severity")
        failure_type = ev.get("failure_type", "unknown")
        if t_perf is not None and pd.isna(t_perf):
            t_perf = None
        else:
            try:
                t_perf = int(t_perf) if t_perf is not None and pd.notna(t_perf) else None
            except (TypeError, ValueError):
                t_perf = None
        if t_perf is None and len(perf) > 0 and image_id is not None and class_id is not None:
            perf_row = perf[(perf["model"] == model) & (perf["corruption"] == corruption) &
                           (perf["image_id"] == image_id) & (perf["class_id"] == class_id)]
            if len(perf_row) > 0:
                _t = perf_row["perf_collapse_severity"].iloc[0]
                if pd.notna(_t):
                    t_perf = int(_t)

        cam_row = cam[(cam["model"] == model) & (cam["corruption"] == corruption) & (cam["object_uid"] == obj_uid)]
        if len(cam_row) == 0 and image_id is not None and class_id is not None and "image_id" in cam.columns and "class_id" in cam.columns:
            cam_row = cam[(cam["model"] == model) & (cam["corruption"] == corruption) &
                         (cam["image_id"] == image_id) & (cam["class_id"].astype(str) == str(class_id))]
        t_cam_warn = None
        t_cam_collapse = None
        if len(cam_row) > 0:
            if "t_cam_warn" in cam_row.columns:
                v = cam_row["t_cam_warn"].iloc[0]
                if pd.notna(v):
                    t_cam_warn = int(v)
            if "t_cam_collapse" in cam_row.columns:
                v = cam_row["t_cam_collapse"].iloc[0]
                if pd.notna(v):
                    t_cam_collapse = int(v)

        lead_warn = (t_perf - t_cam_warn) if (t_perf is not None and t_cam_warn is not None) else None
        lead_collapse = (t_perf - t_cam_collapse) if (t_perf is not None and t_cam_collapse is not None) else None
        t_cam = t_cam_warn  # backward compat
        lead = lead_warn
        alignment = None
        if t_perf is not None and t_cam_warn is not None:
            if lead_warn > 0:
                alignment = "lead"
            elif lead_warn == 0:
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
            "t_cam_warn": t_cam_warn,
            "t_cam_collapse": t_cam_collapse,
            "lead_warn": lead_warn,
            "lead_collapse": lead_collapse,
            "t_cam": t_cam,
            "lead": lead,
            "alignment": alignment,
        })
    return pd.DataFrame(rows)


def compute_lead_table_from_cam_change(
    failure_events_df: pd.DataFrame,
    cam_change_df: pd.DataFrame,
    perf_collapse_df: pd.DataFrame,
    object_uid_col: str = "object_uid",
) -> pd.DataFrame:
    """
    Lead table using t_cam_change (composite score onset). lead = t_perf - t_cam_change.
    cam_change_df must have model, corruption, object_id (or object_uid), t_cam_change.
    """
    if failure_events_df is None or len(failure_events_df) == 0:
        return pd.DataFrame()
    fail = failure_events_df.copy()
    if "object_uid" not in fail.columns and "image_id" in fail.columns and "class_id" in fail.columns:
        fail["object_uid"] = fail["image_id"].astype(str) + "_obj_" + fail["class_id"].astype(str)
    cam = cam_change_df.copy()
    if "object_uid" not in cam.columns and "object_id" in cam.columns:
        cam["object_uid"] = cam["object_id"]
    perf = perf_collapse_df.copy() if perf_collapse_df is not None else pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, ev in fail.iterrows():
        model = ev.get("model")
        corruption = ev.get("corruption")
        obj_uid = ev.get("object_uid")
        image_id = ev.get("image_id")
        class_id = ev.get("class_id")
        t_perf = ev.get("failure_severity")
        failure_type = ev.get("failure_type", "unknown")
        if t_perf is not None and pd.isna(t_perf):
            t_perf = None
        else:
            try:
                t_perf = int(t_perf) if t_perf is not None and pd.notna(t_perf) else None
            except (TypeError, ValueError):
                t_perf = None
        if t_perf is None and len(perf) > 0 and image_id is not None and class_id is not None:
            perf_row = perf[(perf["model"] == model) & (perf["corruption"] == corruption) &
                           (perf["image_id"] == image_id) & (perf["class_id"] == class_id)]
            if len(perf_row) > 0:
                _t = perf_row["perf_collapse_severity"].iloc[0]
                if pd.notna(_t):
                    t_perf = int(_t)

        cam_row = cam[(cam["model"] == model) & (cam["corruption"] == corruption) & (cam["object_uid"] == obj_uid)]
        t_cam_change = None
        if len(cam_row) > 0 and "t_cam_change" in cam_row.columns:
            v = cam_row["t_cam_change"].iloc[0]
            if pd.notna(v):
                t_cam_change = int(v)

        lead = (t_perf - t_cam_change) if (t_perf is not None and t_cam_change is not None) else None
        alignment = None
        if t_perf is not None and t_cam_change is not None:
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
            "t_cam_change": t_cam_change,
            "lead": lead,
            "alignment": alignment,
        })
    return pd.DataFrame(rows)


def aggregate_onset_lead_survival(
    lead_df: pd.DataFrame,
    cam_change_df: pd.DataFrame,
    severities: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    From lead table (with t_cam_change, lead) and cam_change_df (with t_cam_change per object),
    compute: (1) CAM change onset distribution (count per severity),
             (2) Lead distribution (count per lead value),
             (3) Survival curve P(CAM change not yet occurred) by severity,
             (4) Per-corruption mean lead.
    Returns dict for JSON export.
    """
    if severities is None:
        severities = [0, 1, 2, 3, 4]
    out: Dict[str, Any] = {}

    # Onset distribution: among objects with t_cam_change, count per severity
    if "t_cam_change" in lead_df.columns:
        tc = lead_df["t_cam_change"].dropna()
        tc = tc[tc >= 0].astype(int)
        onset_counts = {f"L{sev}": int((tc == sev).sum()) for sev in severities if sev > 0}
        onset_counts["no_change"] = int(lead_df["t_cam_change"].isna().sum())
        out["cam_change_onset_distribution"] = onset_counts
        out["cam_change_onset_severities"] = [f"L{s}" for s in severities if s > 0] + ["no_change"]

    # Lead distribution
    if "lead" in lead_df.columns:
        lead_vals = lead_df["lead"].dropna()
        lead_vals = lead_vals.astype(int)
        lead_counts: Dict[str, int] = {}
        for v in sorted(lead_vals.unique()):
            lead_counts[str(v)] = int((lead_vals == v).sum())
        out["lead_distribution"] = lead_counts
        out["n_lead"] = int((lead_vals > 0).sum())
        out["n_coincident"] = int((lead_vals == 0).sum())
        out["n_lag"] = int((lead_vals < 0).sum())
        out["n_with_lead"] = int(len(lead_vals))
        if len(lead_vals) > 0:
            out["mean_lead"] = float(lead_vals.mean())

    # Survival: P(t_cam_change > sev) = proportion of objects (with any t_cam_change) that have not yet changed by sev
    if "t_cam_change" in lead_df.columns and cam_change_df is not None and len(cam_change_df) > 0:
        n_objects = len(cam_change_df)
        tc_col = "t_cam_change" if "t_cam_change" in cam_change_df.columns else None
        if tc_col:
            survival = {}
            for sev in severities:
                # proportion with t_cam_change > sev (or no change, i.e. NaN)
                no_yet = (cam_change_df[tc_col].isna()) | (cam_change_df[tc_col] > sev)
                survival[f"L{sev}"] = float(no_yet.sum() / n_objects) if n_objects else 0.0
            out["cam_survival_curve"] = survival
            out["cam_survival_severities"] = [f"L{s}" for s in severities]

    # Per-corruption mean lead
    if "corruption" in lead_df.columns and "lead" in lead_df.columns:
        by_c = lead_df.groupby("corruption")["lead"].apply(lambda s: s.dropna().mean())
        out["mean_lead_by_corruption"] = by_c.dropna().astype(float).to_dict()
        out["corruptions"] = list(by_c.dropna().index.astype(str))

    return out
