"""
Export user-study bundles: original image, Grad-CAM overlay (copy), FastCAV pseudo-heatmap,
structured LLM prompts (+ optional OpenAI explanations).

Does not modify training/inference pipelines. Reads results/*.csv and existing heatmap PNGs.

Usage:
  python scripts/12_user_study_export.py --max-units 20
  python scripts/12_user_study_export.py --skip-llm --max-units 50
  python scripts/12_user_study_export.py --image-id 0000022_00500_d_0000005 --out-dir results/user_study_test_one
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.io import load_yaml
from src.user_study.llm_user_study import (
    SYSTEM_PROMPT_TRAJECTORY_KO,
    build_trajectory_user_message,
    format_trajectory_block,
    generate_explanation_openai,
    write_evaluation_questionnaire,
    write_full_prompt_for_disk,
)


def _resolve_image_path(root: Path, rel) -> Path | None:
    if rel is None or (isinstance(rel, float) and np.isnan(rel)):
        return None
    s = str(rel).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute() and p.exists():
        return p
    cand = (root / p).resolve()
    if cand.exists():
        return cand
    cand2 = (root / s.replace("\\", "/")).resolve()
    if cand2.exists():
        return cand2
    return None


def _confidence_from_row(row: pd.Series) -> float | None:
    for k in ("pred_score", "score", "conf"):
        if k not in row.index:
            continue
        v = row.get(k)
        try:
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _miss_from_row(row: pd.Series) -> bool | None:
    if "is_miss" in row.index:
        v = row.get("is_miss")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return bool(v)
    if "matched" in row.index:
        v = row.get("matched")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return not bool(v)
    return None


def _miss_trend_label(m0: bool | None, m1: bool | None) -> str:
    if m0 is None or m1 is None:
        return "uncertain"
    if m0 == m1:
        return "stable"
    if m1 and not m0:
        return "increasing (failure or miss at current severity, not at severity 0)"
    return "decreasing (failure at severity 0 not present at current severity)"


def _conf_trend_label(c0: float | None, c1: float | None) -> str:
    if c0 is None or c1 is None:
        return "uncertain"
    d = float(c1) - float(c0)
    if abs(d) < 1e-5 * max(1.0, abs(float(c0))):
        return "stable"
    return "increasing" if d > 0 else "decreasing"


def _prepare_cam_df_for_export(cam_full: pd.DataFrame | None, gradcam_xai: str) -> pd.DataFrame | None:
    """Restrict cam_records to the same XAI arm as heatmap_samples + primary layer when possible."""
    if cam_full is None or len(cam_full) == 0:
        return None
    df = cam_full
    if "xai_method" in df.columns:
        sub = df[df["xai_method"].astype(str) == str(gradcam_xai)]
        if len(sub) > 0:
            df = sub
    if "layer_role" in df.columns:
        sub = df[df["layer_role"].astype(str) == "primary"]
        if len(sub) > 0:
            df = sub
    return df if len(df) > 0 else None


_OBJECT_UID_TAIL_IDX = re.compile(r"^(.*_obj_\d+)_\d+$")


def _cam_object_key(object_uid: str) -> str:
    """detection_records.object_uid is `..._obj_{class}_{idx}`; cam_records.object_id drops the trailing `_{idx}`."""
    s = str(object_uid)
    m = _OBJECT_UID_TAIL_IDX.match(s)
    return m.group(1) if m else s


def _cam_row_lookup(
    cam_df: pd.DataFrame | None,
    *,
    image_id: str,
    object_uid: str,
    corruption: str,
    severity: int,
    model_name: str,
) -> pd.Series | None:
    if cam_df is None or len(cam_df) == 0:
        return None
    oid_col = "object_id" if "object_id" in cam_df.columns else None
    if oid_col is None:
        return None
    mod_col = "model" if "model" in cam_df.columns else ("model_id" if "model_id" in cam_df.columns else None)
    if mod_col is None:
        return None
    cam_oid = _cam_object_key(object_uid)
    m = (
        (cam_df["image_id"].astype(str) == str(image_id))
        & (cam_df[oid_col].astype(str) == cam_oid)
        & (cam_df["corruption"].astype(str) == str(corruption))
        & (cam_df["severity"].astype(int) == int(severity))
        & (cam_df[mod_col].astype(str) == str(model_name))
    )
    sub = cam_df.loc[m]
    if len(sub) == 0:
        # Fallback: object_uid format already matches cam (no trailing idx).
        m2 = (
            (cam_df["image_id"].astype(str) == str(image_id))
            & (cam_df[oid_col].astype(str) == str(object_uid))
            & (cam_df["corruption"].astype(str) == str(corruption))
            & (cam_df["severity"].astype(int) == int(severity))
            & (cam_df[mod_col].astype(str) == str(model_name))
        )
        sub = cam_df.loc[m2]
        if len(sub) == 0:
            return None
    return sub.iloc[0]


def _row_for_severity(
    full: pd.DataFrame,
    image_id: str,
    object_uid: str,
    corruption: str,
    severity: int,
) -> pd.Series | None:
    m = (
        (full["image_id"].astype(str) == str(image_id))
        & (full["object_uid"].astype(str) == str(object_uid))
        & (full["corruption"].astype(str) == str(corruption))
        & (full["severity"].astype(int) == int(severity))
    )
    sub = full.loc[m]
    if len(sub) == 0:
        return None
    return sub.iloc[0]


def _gradcam_overlay_path(
    heatmap_dir: Path,
    gradcam_xai: str,
    model_name: str,
    corruption: str,
    severity: int,
    image_id: str,
    safe_uid: str,
) -> Path:
    return (
        Path(heatmap_dir)
        / gradcam_xai
        / model_name
        / corruption
        / f"L{severity}"
        / f"{image_id}_{safe_uid}.png"
    )


def _perturbation_for_corruption(cfg: dict, corruption: str) -> dict:
    """Pick the physical parameter list for one corruption from experiment.yaml."""
    cor_cfg = (cfg.get("corruptions", {}) or {}).get(corruption, {}) or {}
    if corruption == "fog":
        return {"param_name": "fog_alpha", "values": list(cor_cfg.get("alpha", []))}
    if corruption == "lowlight":
        gamma = list(cor_cfg.get("gamma", []))
        bright = list(cor_cfg.get("brightness_scale", []))
        if bright:
            return {"param_name": "lowlight_brightness_scale", "values": bright}
        return {"param_name": "lowlight_gamma", "values": gamma}
    if corruption == "motion_blur":
        return {"param_name": "motion_blur_kernel_length", "values": list(cor_cfg.get("kernel_length", []))}
    return {"param_name": "severity_index", "values": [0, 1, 2, 3, 4]}


def main():
    ap = argparse.ArgumentParser(description="User study export bundles")
    ap.add_argument("--max-units", type=int, default=None, help="Max (image,object,corruption) units to export")
    ap.add_argument("--max-images", type=int, default=None, help="Max distinct image_ids to export (each image expands to all its objects × corruptions)")
    ap.add_argument("--out-dir", type=str, default="results/user_study_bundles")
    ap.add_argument(
        "--gradcam-xai",
        type=str,
        default=None,
        help="Subfolder under heatmap_samples (gradcam | gradcampp=Grad-CAM++ | layercam). Default: user_study.gradcam_overlay_method in experiment.yaml",
    )
    ap.add_argument("--skip-llm", action="store_true", help="Only write prompts, no API calls")
    ap.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    ap.add_argument(
        "--image-id",
        type=str,
        default=None,
        help="Only export rows with this image_id (all corruptions × severities × objects in merge). Folder names: {corruption}_L{sev}_{object_uid}.",
    )
    args = ap.parse_args()

    cfg = load_yaml(ROOT / "configs" / "experiment.yaml")
    results_root = ROOT / cfg.get("results", {}).get("root", "results")
    heatmap_dir = ROOT / cfg.get("results", {}).get("heatmap_samples_dir", "results/heatmap_samples")
    us_cfg = cfg.get("user_study", {}) or {}
    llm_model_cfg = us_cfg.get("llm", {}).get("model", args.llm_model)
    gradcam_xai = args.gradcam_xai or us_cfg.get("gradcam_overlay_method", "gradcam")

    cam_path = results_root / "cam_records.csv"
    cam_df: pd.DataFrame | None = None
    if cam_path.exists():
        cam_raw = pd.read_csv(cam_path)
        cam_df = _prepare_cam_df_for_export(cam_raw, gradcam_xai)
        n_cam = len(cam_df) if cam_df is not None else 0
        print(
            f"[export] cam_records.csv: {len(cam_raw)} rows → {n_cam} after xai_method={gradcam_xai!r} / primary layer",
            flush=True,
        )
    else:
        print("[export] cam_records.csv not found — LLM prompts will show CAM rows as n/a", flush=True)

    det_path = results_root / "detection_records.csv"
    if not det_path.exists():
        print(f"Missing {det_path}")
        sys.exit(1)
    det = pd.read_csv(det_path)
    keys = ["image_id", "object_uid", "corruption", "severity"]
    for k in keys:
        if k not in det.columns:
            print(f"Missing column {k} in detection_records.csv")
            sys.exit(1)

    if args.image_id:
        iid = str(args.image_id).strip()
        det = det[det["image_id"].astype(str) == iid]
        if len(det) == 0:
            print(f"No detection rows for image_id={iid!r}")
            sys.exit(1)
        print(f"[export] filter image_id={iid!r} → {len(det)} rows", flush=True)

    if args.max_images:
        keep_ids = list(dict.fromkeys(det["image_id"].astype(str).tolist()))[: int(args.max_images)]
        det = det[det["image_id"].astype(str).isin(keep_ids)]
        print(f"[export] --max-images {args.max_images} → {len(keep_ids)} image_ids, {len(det)} rows", flush=True)

    mcol = "model_id" if "model_id" in det.columns else "model"

    # One unit = (image_id, object_uid, corruption); each unit covers severities 0..4 in a single LLM call.
    group_keys = ["image_id", "object_uid", "corruption"]
    grouped = det.groupby(group_keys, sort=False)
    units = list(grouped)
    if args.max_units:
        units = units[: int(args.max_units)]

    out_root = ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []

    total_units = len(units)
    print(
        f"[export] units (image,object,corruption): {total_units}, skip_llm={args.skip_llm}, "
        f"~{total_units} Grad-CAM LLM calls if not skipping",
        flush=True,
    )

    completed = 0
    for uidx, ((image_id, object_uid, corruption), grp) in enumerate(units):
        image_id = str(image_id)
        object_uid = str(object_uid)
        corruption = str(corruption)
        grp_sorted = grp.sort_values("severity")
        rows_by_sev: dict[int, pd.Series] = {
            int(r["severity"]): r for _, r in grp_sorted.iterrows()
        }
        severities_present = sorted(rows_by_sev.keys())
        if not severities_present:
            continue

        safe_o = object_uid.replace("/", "_").replace("\\", "_").replace(":", "_")[:72]
        if args.image_id:
            unit_id = f"{corruption}_{safe_o}"
        else:
            unit_id = f"unit_{uidx:05d}"
        udir = out_root / unit_id
        udir.mkdir(parents=True, exist_ok=True)

        ref_row = rows_by_sev[severities_present[0]]
        model_name = str(ref_row.get(mcol, "yolo_generic"))
        gt_class = str(ref_row.get("gt_class_name", ref_row.get("pred_class_name", "?")))

        # Build trajectory rows (detection + cam) in fixed severity order 0..4.
        sev_order = [0, 1, 2, 3, 4]
        traj_rows: list = []
        cam_rows: list = []
        for sev in sev_order:
            det_row = rows_by_sev.get(sev)
            traj_rows.append(det_row.to_dict() if det_row is not None else {"severity": sev})
            cam_row = _cam_row_lookup(
                cam_df,
                image_id=image_id,
                object_uid=object_uid,
                corruption=corruption,
                severity=sev,
                model_name=model_name,
            )
            cam_rows.append(cam_row.to_dict() if cam_row is not None else None)

            # Save the corrupted frame and overlay for this severity, for visual context.
            if det_row is not None:
                rel = det_row.get("corrupted_image_path") or det_row.get("image_path")
                img_path = _resolve_image_path(ROOT, rel)
                if img_path is not None and img_path.exists():
                    shutil.copy2(img_path, udir / f"original_L{sev}.png")
            safe_uid = object_uid.replace("/", "_").replace("\\", "_")[:80]
            grad_path = _gradcam_overlay_path(
                Path(heatmap_dir), gradcam_xai, model_name, corruption, sev, image_id, safe_uid,
            )
            if grad_path.exists():
                shutil.copy2(grad_path, udir / f"gradcam_overlay_L{sev}.png")

        perturbation = _perturbation_for_corruption(cfg, corruption)
        traj_block = format_trajectory_block(
            traj_rows, cam_rows, corruption=corruption, perturbation=perturbation,
        )
        user_msg = build_trajectory_user_message(
            image_id=image_id,
            object_uid=object_uid,
            gt_class=gt_class,
            corruption=corruption,
            trajectory_block=traj_block,
        )
        (udir / "prompt_gradcam.txt").write_text(
            write_full_prompt_for_disk(SYSTEM_PROMPT_TRAJECTORY_KO, user_msg),
            encoding="utf-8",
        )
        (udir / "trajectory_table.json").write_text(
            json.dumps(
                {
                    "image_id": image_id,
                    "object_uid": object_uid,
                    "corruption": corruption,
                    "gt_class": gt_class,
                    "model_id": model_name,
                    "perturbation": perturbation,
                    "severities": sev_order,
                    "detection": traj_rows,
                    "cam": cam_rows,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        expl = ""
        if not args.skip_llm:
            try:
                expl = generate_explanation_openai(
                    user_msg,
                    model=llm_model_cfg,
                    system_prompt=SYSTEM_PROMPT_TRAJECTORY_KO,
                )
            except Exception as e:
                expl = f"[LLM error] {e}"
        (udir / "explanation_gradcam.txt").write_text(
            expl or "(empty — use --skip-llm or set OPENAI_API_KEY)",
            encoding="utf-8",
        )

        manifest.append(
            {
                "unit_id": unit_id,
                "image_id": image_id,
                "object_uid": object_uid,
                "corruption": corruption,
                "model_id": model_name,
                "gradcam_heatmap_subdir": gradcam_xai,
                "n_severities_present": len(severities_present),
            }
        )
        completed += 1
        if completed % 10 == 0 or completed == 1:
            print(f"[export] completed {completed}/{total_units} (last {unit_id})", flush=True)

    pd.DataFrame(manifest).to_csv(out_root / "manifest.csv", index=False)
    write_evaluation_questionnaire(out_root / "evaluation_questions_likert.txt")
    readme = f"""User study bundles (Grad-CAM trajectory, severity 0..4 in one LLM call)

Per unit folder = ONE (image_id, object_uid, corruption) combo:
- original_L{{0..4}}.png        — corrupted frames at each severity
- gradcam_overlay_L{{0..4}}.png — Grad-CAM family overlay from heatmap_samples/{gradcam_xai}/...
- trajectory_table.json         — exact numeric tables fed to the LLM
- prompt_gradcam.txt            — full SYSTEM+USER prompt (Korean 3-line trajectory format)
- explanation_gradcam.txt       — LLM output (3 lines: [판정] / [CAM] / [경고])

Notes:
- One LLM call per unit (was per-severity × 2 methods previously). FastCAV arm removed.
- See evaluation_questions_likert.txt for Likert items.
"""
    (out_root / "README_user_study.txt").write_text(readme, encoding="utf-8")
    print(f"[OK] Wrote {len(manifest)} units under {out_root}")


if __name__ == "__main__":
    main()
