"""
Diagnose alignment analysis: why lead 0%, coincident 0%, lag 100%.
Traces: risk_events ↔ cam_records join, cam_change_severity, lead_steps, alignment.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    results_dir = ROOT / "results"
    risk_path = results_dir / "risk_events.csv"
    fallback_risk = results_dir / "failure_events.csv"
    cam_path = results_dir / "cam_records.csv"

    if risk_path.exists():
        risk_df = pd.read_csv(risk_path)
        if "start_severity" not in risk_df.columns and "failure_severity" in risk_df.columns:
            risk_df["start_severity"] = risk_df["failure_severity"]
    elif fallback_risk.exists():
        print(f"[WARN] Using {fallback_risk} (no risk_events.csv). Column start_severity = failure_severity.")
        risk_df = pd.read_csv(fallback_risk)
        risk_df["start_severity"] = risk_df.get("failure_severity", risk_df.get("first_miss_severity", None))
        if risk_df["start_severity"].isna().all():
            print("[ERROR] failure_events has no failure_severity/first_miss_severity.")
            return
    else:
        print(f"[ERROR] Neither {risk_path} nor {fallback_risk} found. Run 04_detect_risk_events.py first.")
        return
    if not cam_path.exists():
        print(f"[ERROR] {cam_path} not found. Run 05_gradcam_failure_analysis.py first.")
        return

    cam_df = pd.read_csv(cam_path)

    print("=== 1. risk/failure events columns & sample ===")
    print(risk_df.columns.tolist())
    sample_cols = [c for c in ["corruption", "object_uid", "failure_type", "start_severity", "failure_event_id", "image_id", "class_id", "failure_severity"] if c in risk_df.columns]
    if sample_cols:
        print(risk_df[sample_cols].head(10).to_string())
    print()

    print("=== 2. cam_records columns & join keys ===")
    print("Columns:", [c for c in cam_df.columns if c in ("severity", "failure_severity", "object_id", "failure_event_id", "corruption", "layer_role")])
    print("failure_event_id sample (first 5 non-null):", cam_df["failure_event_id"].dropna().head(5).tolist())
    print("failure_event_id null count:", cam_df["failure_event_id"].isna().sum(), "of", len(cam_df))
    print("severity dtype:", cam_df["severity"].dtype)
    print("severity unique:", sorted(cam_df["severity"].dropna().unique().tolist()))
    if "failure_severity" in cam_df.columns:
        print("failure_severity unique (sample):", cam_df["failure_severity"].dropna().unique()[:10].tolist())
    print()

    # Deduplicate like llm_report (need object_uid for join; if missing, build from image_id+class_id)
    if "object_uid" not in risk_df.columns and "image_id" in risk_df.columns and "class_id" in risk_df.columns:
        risk_df["object_uid"] = risk_df["image_id"].astype(str) + "_obj_" + risk_df["class_id"].astype(str)
    if "failure_event_id" not in risk_df.columns:
        risk_df["failure_event_id"] = risk_df.get("object_uid", "").astype(str) + "|" + risk_df.get("failure_type", "").astype(str)
    if "corruption" in risk_df.columns and "object_uid" in risk_df.columns and "failure_type" in risk_df.columns:
        risk_df = risk_df.drop_duplicates(subset=["corruption", "object_uid", "failure_type"], keep="first").copy()
    print("=== 3. After dedup risk_events rows:", len(risk_df))

    # Reproduce join + filter + cam_change_sev for first N events
    CAM_METRICS = ["bbox_center_activation_distance", "peak_bbox_distance", "activation_spread", "ring_energy_ratio"]
    Z_THRESHOLD = 2.0
    MIN_METRICS_CHANGED = 2
    n_sample = min(15, len(risk_df))
    rows = []

    for idx, (_, risk_event) in enumerate(risk_df.head(n_sample).iterrows()):
        event_id = str(risk_event.get("failure_event_id", ""))
        corruption = risk_event.get("corruption", "")
        object_uid = str(risk_event.get("object_uid", ""))
        start_severity = int(pd.to_numeric(risk_event.get("start_severity", risk_event.get("failure_severity", -1)), errors="coerce") or -1)
        failure_type = risk_event.get("failure_type", "unknown")

        if start_severity < 0:
            rows.append({"event_id": event_id, "join": "skip", "reason": "start_severity<0"})
            continue

        event_cam = pd.DataFrame()
        if "failure_event_id" in cam_df.columns:
            event_cam = cam_df[cam_df["failure_event_id"] == event_id].copy()
        if len(event_cam) == 0 and "object_id" in cam_df.columns:
            event_cam = cam_df[cam_df["object_id"] == object_uid].copy()
        if len(event_cam) == 0:
            try:
                if "_cls" in object_uid:
                    image_id_from_uid = object_uid.split("_cls")[0].rsplit("_obj", 1)[0] if "_obj" in object_uid else object_uid.split("_cls")[0]
                    class_id_from_uid = int(object_uid.split("_cls")[1])
                else:
                    image_id_from_uid = object_uid
                    class_id_from_uid = None
                if image_id_from_uid is not None and class_id_from_uid is not None:
                    event_cam = cam_df[
                        (cam_df["image_id"].astype(str) == str(image_id_from_uid))
                        & (cam_df["class_id"].astype(int) == int(class_id_from_uid))
                        & (cam_df["corruption"] == corruption)
                    ].copy()
            except Exception as e:
                pass

        n_before = len(event_cam)
        # Coerce severity to int (same as filter)
        if "severity" in event_cam.columns:
            event_cam["severity"] = pd.to_numeric(event_cam["severity"], errors="coerce").fillna(-1).astype(int)
        event_cam = event_cam[
            (event_cam["corruption"] == corruption) & (event_cam["severity"] <= start_severity)
        ].copy()
        n_after = len(event_cam)

        severity_order = []
        cam_change_sev = None
        if len(event_cam) > 0:
            primary = event_cam[event_cam.get("layer_role", "primary") == "primary"]
            if len(primary) > 0:
                severity_order = sorted(primary["severity"].unique().tolist())
                baseline = primary[primary["severity"] == 0]
                if len(baseline) >= 1:
                    for sev in severity_order:
                        if sev == 0:
                            continue
                        sev_cam = primary[primary["severity"] == sev]
                        if len(sev_cam) == 0:
                            continue
                        n_changed = 0
                        for m in CAM_METRICS:
                            if m not in baseline.columns or m not in sev_cam.columns:
                                continue
                            m0 = baseline[m].dropna()
                            ms = sev_cam[m].dropna()
                            if len(m0) < 1 or len(ms) < 1:
                                continue
                            mean0, std0 = m0.mean(), m0.std()
                            if pd.isna(std0) or std0 == 0:
                                std0 = 1e-8
                            z = (ms.mean() - mean0) / std0
                            if abs(z) >= Z_THRESHOLD:
                                n_changed += 1
                        if n_changed >= MIN_METRICS_CHANGED:
                            cam_change_sev = int(sev)
                            break

        lead_steps = (start_severity - cam_change_sev) if cam_change_sev is not None else None
        if lead_steps is not None:
            alignment = "lead" if lead_steps > 0 else ("coincident" if lead_steps == 0 else "lag")
        else:
            alignment = "N/A"

        max_sev_in_data = max(severity_order) if severity_order else None
        rows.append({
            "event_id": str(event_id)[:40],
            "object_uid": str(object_uid)[:35],
            "failure_type": failure_type,
            "start_severity": start_severity,
            "n_cam_before": n_before,
            "n_cam_after": n_after,
            "severity_order": severity_order,
            "max_sev_in_data": max_sev_in_data,
            "cam_change_sev": cam_change_sev,
            "lead_steps": lead_steps,
            "alignment": alignment,
            "has_new_metrics": all(m in event_cam.columns for m in CAM_METRICS) if len(event_cam) > 0 else False,
        })

    print("=== 4. Per-event trace (first", n_sample, "events) ===")
    trace_df = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(trace_df.to_string())
    print()

    # Summary
    with_cam = trace_df[trace_df["cam_change_sev"].notna()]
    print("=== 5. Summary ===")
    print("Events with CAM and computed cam_change_sev:", len(with_cam))
    if len(with_cam) > 0:
        print("Alignment counts:", with_cam["alignment"].value_counts().to_dict())
        print("lead_steps min/max:", with_cam["lead_steps"].min(), with_cam["lead_steps"].max())
        # Check for impossible lag (cam_change_sev > start_severity)
        impossible = with_cam[with_cam["cam_change_sev"] > with_cam["start_severity"]]
        if len(impossible) > 0:
            print("[BUG] Impossible: cam_change_sev > start_severity for", len(impossible), "events:")
            print(impossible[["start_severity", "cam_change_sev", "max_sev_in_data", "alignment"]].to_string())
        else:
            print("No events with cam_change_sev > start_severity (filter is correct).")
    print()
    print("Done. If alignment is always 'lag', check: (1) severity dtype/coercion, (2) wrong column used, (3) join returns wrong object's CAM.")


if __name__ == "__main__":
    main()
