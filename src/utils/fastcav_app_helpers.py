"""
Data loading and on-the-fly bbox rendering for FastCAV in app.py.

Joins detection_records.csv with fastcav_tiny_concept_scores.csv (or fastcav_concept_scores.csv).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.user_study.pseudo_heatmap import generate_fastcav_pseudo_heatmap
from src.utils.fastcav_bbox_viz import visualize_fastcav_bbox
from src.utils.io import load_yaml

# Cached DataFrames per process
_DET_DF = None
_FC_DF = None
_FC_KIND: Optional[str] = None  # "tiny" | "legacy"


def _results_paths(root: Path) -> Tuple[Path, Path]:
    r = root / "results"
    return r / "fastcav_tiny_concept_scores.csv", r / "fastcav_concept_scores.csv"


def load_fastcav_tables(root: Path):
    """Load detection_records and FastCAV scores; set module cache."""
    global _DET_DF, _FC_DF, _FC_KIND
    import pandas as pd

    det_path = root / "results" / "detection_records.csv"
    tiny_p, leg_p = _results_paths(root)
    if not det_path.exists() or det_path.stat().st_size == 0:
        _DET_DF = None
        _FC_DF = None
        _FC_KIND = None
        return
    _DET_DF = pd.read_csv(det_path)
    if tiny_p.exists() and tiny_p.stat().st_size > 0:
        _FC_DF = pd.read_csv(tiny_p)
        _FC_KIND = "tiny"
    elif leg_p.exists() and leg_p.stat().st_size > 0:
        _FC_DF = pd.read_csv(leg_p)
        _FC_KIND = "legacy"
    else:
        _FC_DF = None
        _FC_KIND = None


def fastcav_available(root: Path) -> bool:
    if _DET_DF is None or _FC_DF is None:
        load_fastcav_tables(root)
    return _DET_DF is not None and _FC_DF is not None and len(_DET_DF) > 0 and len(_FC_DF) > 0


def _model_col(df) -> str:
    if "model_id" in df.columns:
        return "model_id"
    return "model"


def _resolve_image_path(root: Path, rel: Any) -> Optional[Path]:
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


def _row_image_path(root: Path, row: Any) -> Optional[Path]:
    for key in ("corrupted_image_path", "image_path", "frame_path"):
        if hasattr(row, "index") and key in row.index:
            p = _resolve_image_path(root, row.get(key))
            if p is not None:
                return p
    return None


def _concept_dict_from_fc_row(row: Any, kind: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if hasattr(row, "index"):
        for k in row.index:
            if not isinstance(k, str):
                continue
            if not k.startswith("concept_"):
                continue
            v = row[k]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            try:
                short = k.replace("concept_", "", 1)
                out[short] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def _primary_concept_key(kind: str, sample_keys: List[str]) -> str:
    if kind == "tiny":
        if "tiny_object_visibility" in sample_keys:
            return "tiny_object_visibility"
        for k in sample_keys:
            if "visibility" in k:
                return k
    else:
        for k in ("Diffused", "Focused", "Background", "Collapse"):
            if k in sample_keys:
                return k
    if sample_keys:
        return sample_keys[0]
    return "tiny_object_visibility"


def list_models(root: Path) -> List[str]:
    if not fastcav_available(root):
        return []
    df = _DET_DF
    col = _model_col(df)
    if col not in df.columns:
        return []
    return sorted(df[col].astype(str).unique().tolist())


def list_corruptions(root: Path, model_id: str) -> List[str]:
    if not fastcav_available(root):
        return []
    df = _DET_DF
    col = _model_col(df)
    sub = df[df[col].astype(str) == str(model_id)]
    if "corruption" not in sub.columns:
        return []
    return sorted(sub["corruption"].astype(str).unique().tolist())


def _severity_int(s: Any) -> int:
    try:
        return int(s)
    except (TypeError, ValueError):
        return -1


def list_samples_intersection(root: Path, model_id: str, corruption: str) -> List[str]:
    """image_id values that have both detection and FastCAV rows for severities 0..4."""
    if not fastcav_available(root):
        return []
    det = _DET_DF
    fc = _FC_DF
    kind = _FC_KIND
    mcol = _model_col(det)
    dsub = det[
        (det[mcol].astype(str) == str(model_id))
        & (det["corruption"].astype(str) == str(corruption))
    ]
    if len(dsub) == 0:
        return []
    fsub = fc[fc["corruption"].astype(str) == str(corruption)].copy()
    if kind == "legacy" and "model" in fsub.columns:
        fsub = fsub[fsub["model"].astype(str) == str(model_id)]
    dsub = dsub.copy()
    dsub["_sev"] = dsub["severity"].map(_severity_int)
    fsub = fsub.copy()
    fsub["_sev"] = fsub["severity"].map(_severity_int)

    by_img: Dict[str, set] = {}
    for _, row in dsub.iterrows():
        sev = row["_sev"]
        if sev < 0 or sev > 4:
            continue
        iid = str(row.get("image_id", ""))
        if not iid:
            continue
        by_img.setdefault(iid, set()).add(sev)

    fc_by_img: Dict[str, set] = {}
    for _, row in fsub.iterrows():
        sev = row["_sev"]
        if sev < 0 or sev > 4:
            continue
        iid = str(row.get("image_id", ""))
        if not iid:
            continue
        fc_by_img.setdefault(iid, set()).add(sev)

    required = {0, 1, 2, 3, 4}
    out = []
    for iid in sorted(by_img.keys()):
        if required <= by_img.get(iid, set()) and required <= fc_by_img.get(iid, set()):
            out.append(iid)
    return out


def _fc_lookup_tiny(
    fc_sub,
) -> Dict[Tuple[str, int, str], Dict[str, float]]:
    """(image_id, severity, object_uid) -> concepts."""
    lut: Dict[Tuple[str, int, str], Dict[str, float]] = {}
    for _, row in fc_sub.iterrows():
        sev = _severity_int(row.get("severity"))
        iid = str(row.get("image_id", ""))
        ouid = str(row.get("object_uid", ""))
        if not iid or not ouid:
            continue
        lut[(iid, sev, ouid)] = _concept_dict_from_fc_row(row, "tiny")
    return lut


def _fc_lookup_legacy(fc_sub) -> Dict[Tuple[str, int, str], Dict[str, float]]:
    """(image_id, severity, object_id) -> concepts (mean if duplicates)."""
    buckets: Dict[Tuple[str, int, str], List[Dict[str, float]]] = {}
    for _, row in fc_sub.iterrows():
        sev = _severity_int(row.get("severity"))
        iid = str(row.get("image_id", ""))
        oid = str(row.get("object_id", ""))
        if not iid or not oid:
            continue
        key = (iid, sev, oid)
        buckets.setdefault(key, []).append(_concept_dict_from_fc_row(row, "legacy"))
    lut: Dict[Tuple[str, int, str], Dict[str, float]] = {}
    for key, dicts in buckets.items():
        keys = set()
        for d in dicts:
            keys |= set(d.keys())
        merged = {}
        for k in keys:
            vals = [d[k] for d in dicts if k in d]
            if vals:
                merged[k] = float(np.mean(vals))
        lut[key] = merged
    return lut


def _bbox_from_detection(row: Any) -> Optional[Tuple[float, float, float, float]]:
    for xs, ys, xe, ye in (
        ("pred_bbox_x1", "pred_bbox_y1", "pred_bbox_x2", "pred_bbox_y2"),
        ("pred_x1", "pred_y1", "pred_x2", "pred_y2"),
    ):
        if not all(k in row.index for k in (xs, ys, xe, ye)):
            continue
        try:
            a, b, c, d = float(row[xs]), float(row[ys]), float(row[xe]), float(row[ye])
        except (TypeError, ValueError):
            continue
        if any(np.isnan(x) for x in (a, b, c, d)):
            continue
        if c <= a or d <= b:
            continue
        return (a, b, c, d)
    return None


def get_fastcav_pseudo_gaussian_sigma(root: Path) -> float:
    """Sobel prior blur for pseudo-heatmap; matches user_study.pseudo_heatmap in experiment.yaml."""
    try:
        cfg = load_yaml(root / "configs" / "experiment.yaml")
        return float(cfg.get("user_study", {}).get("pseudo_heatmap", {}).get("gaussian_sigma", 4.0))
    except Exception:
        return 4.0


def _frame_bgr_and_concepts_for_pseudo(
    root: Path,
    model_id: str,
    corruption: str,
    severity: int,
    image_id: str,
    object_uid: Optional[str] = None,
) -> Optional[Tuple[Any, Dict[str, float]]]:
    """
    One corrupted frame (BGR) and concept dict for pseudo-heatmap (first object with concepts, or requested uid).
    """
    if not fastcav_available(root):
        return None
    det = _DET_DF
    fc = _FC_DF
    kind = _FC_KIND
    mcol = _model_col(det)
    sev = int(severity)
    drows = det[
        (det[mcol].astype(str) == str(model_id))
        & (det["corruption"].astype(str) == str(corruption))
        & (det["severity"].map(_severity_int) == sev)
        & (det["image_id"].astype(str) == str(image_id))
    ]
    if len(drows) == 0:
        return None

    fsub = fc[fc["corruption"].astype(str) == str(corruption)].copy()
    fsub["_sev"] = fsub["severity"].map(_severity_int)
    fsub = fsub[fsub["_sev"] == sev]
    if kind == "legacy" and "model" in fsub.columns:
        fsub = fsub[fsub["model"].astype(str) == str(model_id)]

    if kind == "tiny":
        lut = _fc_lookup_tiny(fsub)
        id_key = "object_uid"
    else:
        lut = _fc_lookup_legacy(fsub)
        id_key = "object_id"

    first_path = None
    candidates: List[Tuple[Any, str, Dict[str, float]]] = []
    for _, row in drows.iterrows():
        if first_path is None:
            first_path = _row_image_path(root, row)
        oid = str(row.get(id_key, row.get("object_uid", row.get("object_id", ""))))
        concepts = lut.get((str(image_id), sev, oid), {}) or {}
        candidates.append((row, oid, concepts))

    if first_path is None or not first_path.exists():
        return None
    chosen: Optional[Tuple[Any, str, Dict[str, float]]] = None
    if object_uid is not None and str(object_uid).strip():
        ou = str(object_uid).strip()
        for row, oid, concepts in candidates:
            if oid == ou:
                chosen = (row, oid, concepts)
                break
    if chosen is None:
        for row, oid, concepts in candidates:
            if concepts:
                chosen = (row, oid, concepts)
                break
    if chosen is None and candidates:
        chosen = candidates[0]
    if chosen is None:
        return None
    _, _, concepts = chosen

    im_bgr = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
    if im_bgr is None:
        return None
    return (im_bgr, dict(concepts))


def render_fastcav_pseudo_png_bytes(
    root: Path,
    model_id: str,
    corruption: str,
    severity: int,
    image_id: str,
    *,
    object_uid: Optional[str] = None,
    gaussian_sigma: Optional[float] = None,
) -> Optional[bytes]:
    """
    FastCAV pseudo-heatmap overlay (Sobel prior × concept stress); NOT localization.
    """
    tup = _frame_bgr_and_concepts_for_pseudo(
        root, model_id, corruption, severity, image_id, object_uid=object_uid
    )
    if tup is None:
        return None
    im_bgr, concepts = tup
    sigma = float(gaussian_sigma) if gaussian_sigma is not None else get_fastcav_pseudo_gaussian_sigma(root)
    ph = generate_fastcav_pseudo_heatmap(im_bgr, concepts, gaussian_sigma=sigma)
    ok, buf = cv2.imencode(".png", ph["overlay_bgr"])
    if not ok:
        return None
    return buf.tobytes()


def render_fastcav_png_bytes(
    root: Path,
    model_id: str,
    corruption: str,
    severity: int,
    image_id: str,
    *,
    threshold: float = 0.3,
    primary_concept_key: Optional[str] = None,
) -> Optional[bytes]:
    """
    Build bbox visualization PNG bytes for one frame, or None if data/image missing.
    """
    if not fastcav_available(root):
        return None
    det = _DET_DF
    fc = _FC_DF
    kind = _FC_KIND
    mcol = _model_col(det)
    sev = int(severity)
    drows = det[
        (det[mcol].astype(str) == str(model_id))
        & (det["corruption"].astype(str) == str(corruption))
        & (det["severity"].map(_severity_int) == sev)
        & (det["image_id"].astype(str) == str(image_id))
    ]
    if len(drows) == 0:
        return None

    fsub = fc[fc["corruption"].astype(str) == str(corruption)].copy()
    fsub["_sev"] = fsub["severity"].map(_severity_int)
    fsub = fsub[fsub["_sev"] == sev]
    if kind == "legacy" and "model" in fsub.columns:
        fsub = fsub[fsub["model"].astype(str) == str(model_id)]

    if kind == "tiny":
        lut = _fc_lookup_tiny(fsub)
        id_key = "object_uid"
    else:
        lut = _fc_lookup_legacy(fsub)
        id_key = "object_id"

    # L0 baselines for delta
    f0 = fc[fc["corruption"].astype(str) == str(corruption)].copy()
    f0["_sev"] = f0["severity"].map(_severity_int)
    f0 = f0[f0["_sev"] == 0]
    if kind == "legacy" and "model" in f0.columns:
        f0 = f0[f0["model"].astype(str) == str(model_id)]
    if kind == "tiny":
        lut0 = _fc_lookup_tiny(f0)
    else:
        lut0 = _fc_lookup_legacy(f0)

    first_path = None
    detections: List[Dict[str, Any]] = []
    concept_scores: List[Dict[str, float]] = []
    baselines: List[Optional[float]] = []

    sample_keys: List[str] = []
    for _, row in drows.iterrows():
        if first_path is None:
            first_path = _row_image_path(root, row)
        bbox = _bbox_from_detection(row)
        if bbox is None:
            continue
        oid = str(row.get(id_key, row.get("object_uid", row.get("object_id", ""))))
        concepts = lut.get((str(image_id), sev, oid), {})
        if not concepts:
            concepts = {}
        if concepts:
            sample_keys.extend(concepts.keys())

        label = str(row.get("pred_class_name", row.get("gt_class_name", "?")))
        try:
            conf = float(row.get("pred_score", row.get("score", float("nan"))))
        except (TypeError, ValueError):
            conf = float("nan")

        detections.append(
            {
                "bbox": bbox,
                "label": label,
                "confidence": conf,
            }
        )
        concept_scores.append(concepts)

        b0 = lut0.get((str(image_id), 0, oid), {})
        pk = primary_concept_key or _primary_concept_key(kind or "tiny", list(b0.keys()) or list(concepts.keys()))
        base_v = b0.get(pk) if pk in b0 else None
        baselines.append(base_v)

    pk_use = primary_concept_key or _primary_concept_key(kind or "tiny", list({k for c in concept_scores for k in c.keys()}))

    if first_path is None or not first_path.exists():
        return None

    im_bgr = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
    if im_bgr is None:
        return None
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

    if not detections:
        out = visualize_fastcav_bbox(
            im_rgb,
            [],
            [],
            primary_concept_key=pk_use,
            threshold=threshold,
        )
    else:
        out = visualize_fastcav_bbox(
            im_rgb,
            detections,
            concept_scores,
            primary_concept_key=pk_use,
            threshold=threshold,
            baseline_scores=baselines,
            show_delta=True,
            warn_below_threshold=True,
        )

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        return None
    return buf.tobytes()


def render_fastcav_response(root: Path, **kwargs) -> Optional[io.BytesIO]:
    data = render_fastcav_png_bytes(root, **kwargs)
    if data is None:
        return None
    return io.BytesIO(data)
