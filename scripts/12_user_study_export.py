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
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.io import load_yaml
from src.user_study.pseudo_heatmap import generate_fastcav_pseudo_heatmap
from src.user_study.heatmap_summary import summarize_heatmap_regions
from src.user_study.llm_user_study import (
    SYSTEM_PROMPT_STRICT,
    build_strict_user_study_user_message,
    format_cam_record_quant_block,
    format_concept_signals_with_trends,
    format_detection_quant_block,
    format_overlay_numeric_block,
    format_performance_signals_block,
    generate_explanation_openai,
    narrate_fastcav_visualization_summary,
    narrate_gradcam_visualization_summary,
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


def _concept_dict_from_row(row: pd.Series) -> dict:
    out = {}
    for c in row.index:
        cs = str(c)
        if cs.startswith("concept_"):
            short = cs.replace("concept_", "", 1)
            v = row[c]
            if isinstance(v, (int, float, np.floating)) and pd.notna(v):
                out[short] = float(v)
    return out


def _heatmap_summary_from_overlay_png(path: Path) -> dict:
    """Approximate saliency emphasis from saved Grad-CAM overlay (no raw CAM array)."""
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        return {"peak_quadrant": "unknown", "concentration": "low", "top10_mass_fraction": 0.0}
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    return summarize_heatmap_regions(g)


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
    m = (
        (cam_df["image_id"].astype(str) == str(image_id))
        & (cam_df[oid_col].astype(str) == str(object_uid))
        & (cam_df["corruption"].astype(str) == str(corruption))
        & (cam_df["severity"].astype(int) == int(severity))
        & (cam_df[mod_col].astype(str) == str(model_name))
    )
    sub = cam_df.loc[m]
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


def _l0_pseudo_summary_and_stress(
    *,
    cache: dict,
    full_merged: pd.DataFrame,
    root: Path,
    image_id: str,
    object_uid: str,
    corruption: str,
    gaussian_sigma: float,
) -> tuple[dict | None, float | None]:
    key = (image_id, object_uid, corruption, gaussian_sigma)
    if key in cache:
        return cache[key]
    row0 = _row_for_severity(full_merged, image_id, object_uid, corruption, 0)
    if row0 is None:
        cache[key] = (None, None)
        return cache[key]
    rel = row0.get("corrupted_image_path") or row0.get("image_path")
    p = _resolve_image_path(root, rel)
    if p is None:
        cache[key] = (None, None)
        return cache[key]
    im_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im_bgr is None:
        cache[key] = (None, None)
        return cache[key]
    concepts0 = _concept_dict_from_row(row0)
    ph0 = generate_fastcav_pseudo_heatmap(
        im_bgr, concepts0, gaussian_sigma=gaussian_sigma
    )
    sm0 = summarize_heatmap_regions(ph0["heatmap_01"])
    g0 = float(ph0["stress_g"])
    cache[key] = (sm0, g0)
    return cache[key]


def main():
    ap = argparse.ArgumentParser(description="User study export bundles")
    ap.add_argument("--max-units", type=int, default=None, help="Max rows to export")
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
    gaussian_sigma = float(us_cfg.get("pseudo_heatmap", {}).get("gaussian_sigma", 4.0))
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
        print("[export] cam_records.csv not found — LLM prompts omit CAM-record numerics", flush=True)

    det_path = results_root / "detection_records.csv"
    fc_path = results_root / "fastcav_tiny_concept_scores.csv"
    if not det_path.exists():
        print(f"Missing {det_path}")
        sys.exit(1)
    if not fc_path.exists():
        print(f"Missing {fc_path}")
        sys.exit(1)

    det = pd.read_csv(det_path)
    fc = pd.read_csv(fc_path)
    keys = ["image_id", "object_uid", "corruption", "severity"]
    for k in keys:
        if k not in det.columns or k not in fc.columns:
            print(f"Missing column {k} for merge")
            sys.exit(1)
    concept_cols = [c for c in fc.columns if str(c).startswith("concept_")]
    fc_min = fc[keys + concept_cols].drop_duplicates(subset=keys)
    full_merged = det.merge(fc_min, on=keys, how="inner")
    if len(full_merged) == 0:
        print("No rows after merge(det, fastcav_tiny_concept_scores)")
        sys.exit(1)

    if args.image_id:
        iid = str(args.image_id).strip()
        full_merged = full_merged[full_merged["image_id"].astype(str) == iid]
        if len(full_merged) == 0:
            print(f"No merged rows for image_id={iid!r} (check detection_records + fastcav CSVs)")
            sys.exit(1)
        print(f"[export] filter image_id={iid!r} → {len(full_merged)} rows", flush=True)

    mcol = "model_id" if "model_id" in full_merged.columns else "model"
    export_df = (
        full_merged.head(int(args.max_units))
        if args.max_units
        else full_merged
    )
    if args.image_id and len(export_df) > 0:
        cols = [c for c in ("corruption", "severity", "object_uid") if c in export_df.columns]
        if cols:
            export_df = export_df.sort_values(cols).reset_index(drop=True)

    out_root = ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    l0_pseudo_cache: dict = {}

    total_rows = len(export_df)
    completed_units = 0
    print(
        f"[export] rows in batch: {total_rows}, skip_llm={args.skip_llm}, "
        f"~{2 * total_rows} LLM calls if not skipping",
        flush=True,
    )

    for uidx, (_, row) in enumerate(export_df.iterrows()):
        image_id = str(row["image_id"])
        object_uid = str(row["object_uid"])
        corruption = str(row["corruption"])
        severity = int(row["severity"])
        safe_o = object_uid.replace("/", "_").replace("\\", "_").replace(":", "_")[:72]
        if args.image_id:
            unit_id = f"{corruption}_L{severity}_{safe_o}"
        else:
            unit_id = f"unit_{uidx:05d}"
        udir = out_root / unit_id
        udir.mkdir(parents=True, exist_ok=True)
        model_name = str(row.get(mcol, "yolo_generic"))
        pred_class = str(row.get("pred_class_name", row.get("gt_class_name", "?")))
        try:
            pred_conf = float(row.get("pred_score", row.get("score", 0.0)))
        except (TypeError, ValueError):
            pred_conf = 0.0

        rel = row.get("corrupted_image_path") or row.get("image_path")
        img_path = _resolve_image_path(ROOT, rel)
        if img_path is None:
            continue
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

        concepts = _concept_dict_from_row(row)
        ph = generate_fastcav_pseudo_heatmap(
            image_bgr,
            concepts,
            gaussian_sigma=gaussian_sigma,
        )
        pseudo_sum = summarize_heatmap_regions(ph["heatmap_01"])
        cv2.imwrite(str(udir / "fastcav_pseudo_jet.png"), ph["jet_bgr"])
        cv2.imwrite(str(udir / "fastcav_pseudo_overlay.png"), ph["overlay_bgr"])
        (udir / "fastcav_pseudo_meta.json").write_text(
            json.dumps(
                {
                    "stress_g": ph["stress_g"],
                    "disclaimer": ph["disclaimer"],
                    "concepts": concepts,
                    "pseudo_spatial_summary": pseudo_sum,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        shutil.copy2(img_path, udir / "original.png")

        safe_uid = object_uid.replace("/", "_").replace("\\", "_")[:80]
        grad_path = _gradcam_overlay_path(
            Path(heatmap_dir),
            gradcam_xai,
            model_name,
            corruption,
            severity,
            image_id,
            safe_uid,
        )
        grad_summary = {"peak_quadrant": "unknown", "concentration": "low", "top10_mass_fraction": 0.0}
        if grad_path.exists():
            shutil.copy2(grad_path, udir / "gradcam_overlay.png")
            grad_summary = _heatmap_summary_from_overlay_png(grad_path)
        else:
            (udir / "gradcam_overlay_MISSING.txt").write_text(
                f"Expected path:\n{grad_path}\n", encoding="utf-8"
            )

        grad_path_l0 = _gradcam_overlay_path(
            Path(heatmap_dir),
            gradcam_xai,
            model_name,
            corruption,
            0,
            image_id,
            safe_uid,
        )
        grad_summary_l0 = (
            _heatmap_summary_from_overlay_png(grad_path_l0)
            if grad_path_l0.exists()
            else None
        )

        row_l0 = _row_for_severity(
            full_merged, image_id, object_uid, corruption, 0
        )
        conf0 = _confidence_from_row(row_l0) if row_l0 is not None else None
        conf_cur = _confidence_from_row(row)
        miss0 = _miss_from_row(row_l0) if row_l0 is not None else None
        miss_cur = _miss_from_row(row)
        conf_tr = _conf_trend_label(conf0, conf_cur)
        miss_tr = _miss_trend_label(miss0, miss_cur)
        performance_block = format_performance_signals_block(
            confidence_trend=conf_tr,
            miss_trend=miss_tr,
            confidence_l0=conf0,
            confidence_current=conf_cur,
        )
        performance_block += (
            f"\ncorruption type: {corruption}\n"
            f"severity level index: {severity}\n"
            f"predicted class (metadata only): {pred_class}"
        )
        performance_block += "\n\n" + format_detection_quant_block(
            row.to_dict(),
            row_l0.to_dict() if row_l0 is not None else None,
        )
        performance_block += "\n\n" + format_overlay_numeric_block(
            grad_summary,
            grad_summary_l0,
        )
        cam_cur = _cam_row_lookup(
            cam_df,
            image_id=image_id,
            object_uid=object_uid,
            corruption=corruption,
            severity=severity,
            model_name=model_name,
        )
        cam_l0_row = None
        if int(severity) != 0:
            cam_l0_row = _cam_row_lookup(
                cam_df,
                image_id=image_id,
                object_uid=object_uid,
                corruption=corruption,
                severity=0,
                model_name=model_name,
            )
        performance_block += "\n\n" + format_cam_record_quant_block(
            cam_cur.to_dict() if cam_cur is not None else None,
            cam_l0_row.to_dict() if cam_l0_row is not None else None,
        )

        concepts_l0 = _concept_dict_from_row(row_l0) if row_l0 is not None else None
        pseudo_l0, stress_l0 = _l0_pseudo_summary_and_stress(
            cache=l0_pseudo_cache,
            full_merged=full_merged,
            root=ROOT,
            image_id=image_id,
            object_uid=object_uid,
            corruption=corruption,
            gaussian_sigma=gaussian_sigma,
        )

        viz_gradcam = narrate_gradcam_visualization_summary(
            grad_summary, grad_summary_l0, severity=severity
        )
        user_gradcam = build_strict_user_study_user_message(
            method="Grad-CAM",
            visualization_summary=viz_gradcam,
            concept_signals=None,
            performance_signals=performance_block,
        )
        viz_fastcav = narrate_fastcav_visualization_summary(
            pseudo_sum, pseudo_l0, float(ph["stress_g"]), stress_l0, severity=severity
        )
        concept_block_f = format_concept_signals_with_trends(concepts, concepts_l0)
        user_fastcav = build_strict_user_study_user_message(
            method="FastCAV",
            visualization_summary=viz_fastcav,
            concept_signals=concept_block_f,
            performance_signals=performance_block,
        )
        (udir / "prompt_gradcam.txt").write_text(
            write_full_prompt_for_disk(SYSTEM_PROMPT_STRICT, user_gradcam),
            encoding="utf-8",
        )
        (udir / "prompt_fastcav.txt").write_text(
            write_full_prompt_for_disk(SYSTEM_PROMPT_STRICT, user_fastcav),
            encoding="utf-8",
        )

        expl_g = expl_f = ""
        if not args.skip_llm:
            try:
                expl_g = generate_explanation_openai(user_gradcam, model=llm_model_cfg)
                expl_f = generate_explanation_openai(user_fastcav, model=llm_model_cfg)
            except Exception as e:
                expl_g = f"[LLM error] {e}"
                expl_f = expl_g
        (udir / "explanation_gradcam.txt").write_text(expl_g or "(empty — use --skip-llm or set OPENAI_API_KEY)", encoding="utf-8")
        (udir / "explanation_fastcav.txt").write_text(expl_f or "(empty)", encoding="utf-8")

        manifest.append(
            {
                "unit_id": unit_id,
                "image_id": image_id,
                "object_uid": object_uid,
                "corruption": corruption,
                "severity": severity,
                "model_id": model_name,
                "gradcam_heatmap_subdir": gradcam_xai,
                "gradcam_overlay_exists": grad_path.exists(),
            }
        )
        completed_units += 1
        if completed_units % 10 == 0 or completed_units == 1:
            print(
                f"[export] completed {completed_units}/{total_rows} (last {unit_id})",
                flush=True,
            )

    pd.DataFrame(manifest).to_csv(out_root / "manifest.csv", index=False)
    write_evaluation_questionnaire(out_root / "evaluation_questions_likert.txt")
    readme = f"""User study bundles (exported)

Terminology
- fastcav_pseudo_overlay.png = FastCAV pseudo-heatmap (concept stress × fixed Sobel prior). NOT pixel-level FastCAV localization.
- heatmap_samples subfolder "{gradcam_xai}" = Grad-CAM family overlay copied to gradcam_overlay.png. Use gradcampp for Grad-CAM++, not FastCAV.

Per unit folder:
- original.png (same corrupted frame as detection / CAM)
- gradcam_overlay.png (from heatmap_samples/{gradcam_xai}/...; missing if file absent)
- fastcav_pseudo_jet.png, fastcav_pseudo_overlay.png — FastCAV arm visual (pseudo-heatmap)
- prompt_gradcam.txt / prompt_fastcav.txt — full SYSTEM+USER prompt (strict user-study format; viz vs L0 + performance trends)
- explanation_*.txt — LLM output (omit for Condition C)
- fastcav_pseudo_meta.json — stress_g, disclaimer, concepts

Conditions
- A: original + gradcam_overlay + explanation_gradcam
- B: original + fastcav_pseudo_overlay + explanation_fastcav
- C: same as A or B but visualization only (do not show explanation_*.txt)

Fair comparison: inner merge on (image_id, object_uid, corruption, severity). See docs/USER_STUDY_ACADEMIC_PIPELINE.md.

See evaluation_questions_likert.txt for Likert items (spatial alignment, interpretability, usefulness, manipulation check).
"""
    (out_root / "README_user_study.txt").write_text(readme, encoding="utf-8")
    print(f"[OK] Wrote {len(manifest)} units under {out_root}")


if __name__ == "__main__":
    main()
