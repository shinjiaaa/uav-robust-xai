"""
Heatmap viewer UI: browse Grad-CAM results by model, corruption, and severity.

Run: python app.py
Open: http://127.0.0.1:5000
"""

import io
from pathlib import Path
import sys
from typing import Optional
from urllib.parse import quote

# Project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from flask import Flask, send_from_directory, send_file, jsonify, request, render_template_string
except ImportError:
    print("Install Flask: pip install flask")
    sys.exit(1)

from src.utils.io import load_yaml
from src.utils import fastcav_app_helpers as fc_app

app = Flask(__name__)

# Config
def get_heatmap_dir():
    config_path = ROOT / "configs" / "experiment.yaml"
    if config_path.exists():
        config = load_yaml(config_path)
        return ROOT / config.get("results", {}).get("heatmap_samples_dir", "results/heatmap_samples")
    return ROOT / "results" / "heatmap_samples"


HEATMAP_DIR = get_heatmap_dir()

RESULTS_DIR = ROOT / "results"


def _canonical_xai_method(name: str) -> str:
    """Legacy folder/config key ``fastcam`` → ``gradcampp`` (Grad-CAM++)."""
    s = str(name).strip()
    if s == "fastcam":
        return "gradcampp"
    return s


def _is_fastcav_viewer(xai_method: Optional[str]) -> bool:
    """Heatmap viewer: FastCAV pseudo-heatmap (no heatmap_samples folder)."""
    return bool(xai_method) and str(xai_method).strip().lower() == "fastcav"


def _print_results_path():
    """Print once at startup so you know which folder's results are being shown."""
    print(f"[app] Results dir (this folder's results): {RESULTS_DIR.resolve()}")
    print(f"[app] Heatmap dir: {HEATMAP_DIR.resolve()}")


def get_xai_methods_from_config():
    """Read gradcam.xai_methods from experiment.yaml (Grad-CAM / Grad-CAM++ / LayerCAM)."""
    config_path = ROOT / "configs" / "experiment.yaml"
    if not config_path.exists():
        return ["gradcam"]
    try:
        config = load_yaml(config_path)
        methods = config.get("gradcam", {}).get("xai_methods")
        if isinstance(methods, list) and methods:
            return [_canonical_xai_method(m) for m in methods]
    except Exception:
        pass
    return ["gradcam"]


def xai_method_display_label(method: str) -> str:
    """UI label; internal keys (paths, cam_records) stay gradcam / fastcam / layercam."""
    m = (method or "").strip().lower()
    if m in ("fastcam", "gradcampp"):
        return "Grad-CAM++"
    if m == "gradcam":
        return "Grad-CAM"
    if m == "layercam":
        return "LayerCAM"
    return str(method)


def get_xai_methods():
    """XAI method ids for UI/API: config + disk; legacy ``heatmap_samples/fastcam`` maps to ``gradcampp``."""
    from_config = get_xai_methods_from_config()
    existing = []
    if HEATMAP_DIR.exists():
        for p in sorted(HEATMAP_DIR.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                existing.append(_canonical_xai_method(p.name))
    if not existing:
        out = sorted(set(from_config))
    elif "gradcam" in existing or "gradcampp" in existing:
        out = sorted(set(existing) | set(from_config))
    else:
        out = sorted(set(from_config))
    fc_app.load_fastcav_tables(ROOT)
    if fc_app.fastcav_available(ROOT):
        out = sorted(set(out) | {"fastcav"})
    return out


def get_heatmap_base(xai_method=None):
    """Directory HEATMAP_DIR/<method>/ for CAM PNGs. ``gradcampp`` falls back to legacy ``fastcam`` folder."""
    if not xai_method:
        return HEATMAP_DIR
    xm = _canonical_xai_method(str(xai_method))
    sub = Path(xm)
    direct = HEATMAP_DIR / sub
    if direct.exists() and direct.is_dir():
        return direct
    if xm == "gradcampp":
        legacy = HEATMAP_DIR / "fastcam"
        if legacy.exists() and legacy.is_dir():
            return legacy
    return HEATMAP_DIR / sub


def _heatmap_url_prefix(xai_method: Optional[str]) -> str:
    """Path segment under heatmap_samples for /heatmaps/... URLs (handles gradcampp → fastcam legacy)."""
    if not xai_method:
        return ""
    base = get_heatmap_base(xai_method)
    try:
        rel = base.resolve().relative_to(HEATMAP_DIR.resolve())
        return rel.as_posix() + "/"
    except ValueError:
        return f"{xai_method}/"


def xai_method_ui_options():
    """Dropdown labels: gradcampp → Grad-CAM++."""
    labels = {
        "gradcam": "Grad-CAM",
        "gradcampp": "Grad-CAM++",
        "layercam": "LayerCAM",
        "fastcav": "FastCAV (pseudo heatmap)",
    }
    return [{"id": m, "label": labels.get(m, m)} for m in get_xai_methods()]


def get_metrics():
    """Load lead_stats and dasc_summary from results dir if present."""
    out = {"lead_stats": None, "dasc_summary": None}
    try:
        p = RESULTS_DIR / "lead_stats.json"
        if p.exists():
            from src.utils.io import load_json
            out["lead_stats"] = load_json(p)
    except Exception:
        pass
    try:
        p = RESULTS_DIR / "dasc_summary.json"
        if p.exists():
            from src.utils.io import load_json
            out["dasc_summary"] = load_json(p)
    except Exception:
        pass
    return out


@app.route("/")
def index():
    """Model list + viewer."""
    models = []
    existing = []
    if HEATMAP_DIR.exists():
        for p in sorted(HEATMAP_DIR.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                existing.append(p.name)
    if existing and "gradcam" not in existing and "gradcampp" not in existing:
        models = existing
    return render_template_string(
        INDEX_HTML,
        models=models,
        heatmap_dir=str(HEATMAP_DIR),
        xai_methods=xai_method_ui_options(),
    )


@app.route("/api/xai_methods")
def api_xai_methods():
    """List XAI method ids + optional display labels (gradcampp → Grad-CAM++)."""
    methods = get_xai_methods()
    return jsonify(
        xai_methods=methods,
        labels={m: xai_method_display_label(m) for m in methods},
    )


@app.route("/api/models")
def api_models():
    """List model names. Optional query: xai_method (e.g. gradcam, gradcampp, fastcav)."""
    xai_method = request.args.get("xai_method", "").strip() or None
    if _is_fastcav_viewer(xai_method):
        fc_app.load_fastcav_tables(ROOT)
        if not fc_app.fastcav_available(ROOT):
            return jsonify(models=[])
        return jsonify(models=fc_app.list_models(ROOT))
    base = get_heatmap_base(xai_method)
    models = []
    if base.exists():
        for p in sorted(base.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                models.append(p.name)
    return jsonify(models=models)


@app.route("/api/models/<model>/corruptions")
def api_corruptions(model):
    """List corruptions for a model. Optional query: xai_method. Tries HEATMAP_DIR/model if method subdir missing (legacy)."""
    xai_method = request.args.get("xai_method", "").strip() or None
    if _is_fastcav_viewer(xai_method):
        fc_app.load_fastcav_tables(ROOT)
        if not fc_app.fastcav_available(ROOT):
            return jsonify(corruptions=[]), 200
        return jsonify(corruptions=fc_app.list_corruptions(ROOT, model))
    path = get_heatmap_base(xai_method) / model
    if not path.exists() or not path.is_dir():
        path = HEATMAP_DIR / model
    if not path.exists() or not path.is_dir():
        return jsonify(corruptions=[]), 200
    corruptions = [p.name for p in sorted(path.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    return jsonify(corruptions=corruptions)


@app.route("/api/models/<model>/<corruption>/severities")
def api_severities(model, corruption):
    """List severity folders (L0..L4) for model/corruption. Optional query: xai_method."""
    xai_method = request.args.get("xai_method", "").strip() or None
    path = get_heatmap_base(xai_method) / model / corruption
    if not path.exists() or not path.is_dir():
        return jsonify(severities=[]), 404
    severities = [p.name for p in sorted(path.iterdir()) if p.is_dir() and p.name.startswith("L")]
    return jsonify(severities=severities)


@app.route("/api/models/<model>/<corruption>/<severity>/images")
def api_images(model, corruption, severity):
    """List image filenames for model/corruption/severity. Optional query: xai_method."""
    xai_method = request.args.get("xai_method", "").strip() or None
    path = get_heatmap_base(xai_method) / model / corruption / severity
    if not path.exists() or not path.is_dir():
        return jsonify(images=[]), 404
    images = [p.name for p in sorted(path.iterdir()) if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    return jsonify(images=images)


def _load_ideal_trend_samples():
    """Load ideal_trend_samples.json (이상적 추세만 샘플 목록)."""
    p = RESULTS_DIR / "ideal_trend_samples.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        from src.utils.io import load_json
        return load_json(p)
    except Exception:
        return None


@app.route("/api/models/<model>/<corruption>/samples")
def api_samples(model, corruption):
    """List sample IDs that have L0–L4 all present (intersection). ideal_only=1이면 이상적 추세만. Optional query: xai_method."""
    xai_method = request.args.get("xai_method", "").strip() or None
    if _is_fastcav_viewer(xai_method):
        fc_app.load_fastcav_tables(ROOT)
        if not fc_app.fastcav_available(ROOT):
            return jsonify(samples=[]), 404
        samples = fc_app.list_samples_intersection(ROOT, model, corruption)
        ideal_only = request.args.get("ideal_only", "").strip().lower() in ("1", "true", "yes")
        if ideal_only:
            ideal = _load_ideal_trend_samples()
            if ideal and isinstance(ideal, dict):
                ideal_list = ideal.get(str(model), {}).get(str(corruption))
                if ideal_list:
                    ideal_set = {str(x) for x in ideal_list}

                    def _ideal_match(s: str) -> bool:
                        s0 = str(s)
                        base = s0.rsplit(".", 1)[0] if "." in s0 else s0
                        return (
                            s0 in ideal_set
                            or base in ideal_set
                            or (base + ".png") in ideal_set
                            or (base + ".jpg") in ideal_set
                        )

                    samples = sorted(s for s in samples if _ideal_match(s))
        return jsonify(samples=samples)
    base = get_heatmap_base(xai_method) / model / corruption
    if not base.exists() or not base.is_dir():
        return jsonify(samples=[]), 404
    by_level = {}
    for p in sorted(base.iterdir()):
        if p.is_dir() and p.name.startswith("L"):
            by_level[p.name] = {
                f.name for f in p.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg")
            }
    required = ["L0", "L1", "L2", "L3", "L4"]
    if not all(lev in by_level for lev in required):
        return jsonify(samples=[])
    full = by_level["L0"]
    for lev in required[1:]:
        full = full & by_level[lev]
    samples = sorted(full)
    ideal_only = request.args.get("ideal_only", "").strip().lower() in ("1", "true", "yes")
    if ideal_only:
        ideal = _load_ideal_trend_samples()
        if ideal and isinstance(ideal, dict):
            ideal_list = ideal.get(str(model), {}).get(str(corruption))
            if ideal_list:
                ideal_set = set(ideal_list)
                samples = sorted(s for s in samples if s in ideal_set)
    return jsonify(samples=samples)


@app.route("/api/models/<model>/<corruption>/sample/<path:sample_id>")
def api_sample_severities(model, corruption, sample_id):
    """For one sample (filename), return severities that have it and image URLs. Optional query: xai_method."""
    xai_method = request.args.get("xai_method", "").strip() or None
    if _is_fastcav_viewer(xai_method):
        fc_app.load_fastcav_tables(ROOT)
        if not fc_app.fastcav_available(ROOT):
            return jsonify(severities=[]), 404
        image_id = str(sample_id)
        if image_id.lower().endswith((".png", ".jpg", ".jpeg")):
            image_id = image_id.rsplit(".", 1)[0]
        result = []
        for sev in range(5):
            q = (
                f"model={quote(str(model), safe='')}"
                f"&corruption={quote(str(corruption), safe='')}"
                f"&severity={sev}"
                f"&image_id={quote(str(image_id), safe='')}"
            )
            result.append(
                {
                    "severity": f"L{sev}",
                    "url": f"/fastcav/render_pseudo?{q}",
                }
            )
        return jsonify(severities=result)
    base = get_heatmap_base(xai_method) / model / corruption
    if not base.exists() or not base.is_dir():
        return jsonify(severities=[]), 404
    if not (sample_id.endswith(".png") or sample_id.endswith(".jpg") or sample_id.endswith(".jpeg")):
        sample_id = sample_id + ".png"
    prefix = _heatmap_url_prefix(xai_method) if xai_method else ""
    result = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and p.name.startswith("L"):
            f = p / sample_id
            if f.exists() and f.is_file():
                result.append({"severity": p.name, "url": f"/heatmaps/{prefix}{model}/{corruption}/{p.name}/{sample_id}"})
    return jsonify(severities=result)


def _load_cam_records_df():
    """Load cam_records.csv once and cache per process."""
    if not hasattr(_load_cam_records_df, "_df"):
        p = RESULTS_DIR / "cam_records.csv"
        if p.exists() and p.stat().st_size > 0:
            try:
                import pandas as pd
                _load_cam_records_df._df = pd.read_csv(p)
            except Exception:
                _load_cam_records_df._df = None
        else:
            _load_cam_records_df._df = None
    return _load_cam_records_df._df


def _sample_stem_from_sample_id(sample_id: str) -> str:
    s = str(sample_id).strip()
    if s.lower().endswith((".png", ".jpg", ".jpeg")):
        return s.rsplit(".", 1)[0]
    return s


def _heatmap_png_stem(image_id: str, object_uid: str) -> str:
    """Same stem as scripts/12_user_study_export.py Grad-CAM overlay filename (safe_uid[:80])."""
    safe = str(object_uid).replace("/", "_").replace("\\", "_")[:80]
    return f"{image_id}_{safe}"


def _load_user_study_manifest_df():
    """Export manifest: maps (model, corruption, severity, sample stem) → unit folder."""
    if not hasattr(_load_user_study_manifest_df, "_df"):
        p = RESULTS_DIR / "user_study_bundles" / "manifest.csv"
        if p.exists() and p.stat().st_size > 0:
            try:
                import pandas as pd
                _load_user_study_manifest_df._df = pd.read_csv(p)
            except Exception:
                _load_user_study_manifest_df._df = None
        else:
            _load_user_study_manifest_df._df = None
    return _load_user_study_manifest_df._df


def _filter_manifest_by_xai_method(sub, xai_method: Optional[str]):
    if sub is None or len(sub) == 0:
        return sub
    if not xai_method or _is_fastcav_viewer(xai_method):
        return sub
    if "gradcam_heatmap_subdir" not in sub.columns:
        return sub
    xm = _canonical_xai_method(str(xai_method))
    sub2 = sub[sub["gradcam_heatmap_subdir"].astype(str) == str(xm)]
    return sub2 if len(sub2) > 0 else sub


def _read_user_study_explanation(unit_dir: Path, name: str, max_chars: int = 12000) -> Optional[str]:
    p = unit_dir / name
    if not p.is_file():
        return None
    try:
        t = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if len(t) > max_chars:
        return t[:max_chars] + "\n… (truncated)"
    return t.strip() or None


@app.route("/api/user_study/llm_by_sample")
def api_user_study_llm_by_sample():
    """Per-severity LLM explanations for the selected heatmap sample (user_study_bundles/manifest.csv)."""
    model = request.args.get("model", "").strip()
    corruption = request.args.get("corruption", "").strip()
    sample_id = request.args.get("sample_id", "").strip()
    xai_method = request.args.get("xai_method", "").strip() or None
    if not model or not corruption or not sample_id:
        return jsonify(
            ok=False,
            message="쿼리에 model, corruption, sample_id가 필요합니다.",
            severities=[],
        ), 400
    stem = _sample_stem_from_sample_id(sample_id)
    df = _load_user_study_manifest_df()
    if df is None or len(df) == 0:
        return jsonify(
            ok=False,
            message="results/user_study_bundles/manifest.csv 가 없습니다. "
            "python scripts/12_user_study_export.py 로 생성하세요.",
            severities=[],
        )
    mcol = "model_id" if "model_id" in df.columns else "model"
    if mcol not in df.columns:
        return jsonify(ok=False, message="manifest에 model/model_id 열이 없습니다.", severities=[]), 500
    sub = df[
        (df[mcol].astype(str) == str(model))
        & (df["corruption"].astype(str) == str(corruption))
    ].copy()
    sub = _filter_manifest_by_xai_method(sub, xai_method)
    if len(sub) == 0:
        return jsonify(
            ok=False,
            message="manifest에서 해당 model/corruption(xai_method) 행이 없습니다.",
            severities=[],
        )
    sub["_stem"] = sub.apply(
        lambda r: _heatmap_png_stem(str(r.get("image_id", "")), str(r.get("object_uid", ""))),
        axis=1,
    )
    exact = sub[sub["_stem"] == stem]
    if len(exact) > 0:
        match_df = exact
        match_mode = "stem"
    else:
        match_df = sub[sub["image_id"].astype(str) == stem]
        match_mode = "image_id"
    if len(match_df) == 0:
        return jsonify(
            ok=False,
            message="manifest에 이 샘플과 일치하는 image_id·파일 stem 행이 없습니다. "
            "export에 포함됐는지, heatmap 파일명과 object_uid가 맞는지 확인하세요.",
            severities=[],
        )
    bundle_root = RESULTS_DIR / "user_study_bundles"
    out_sev = []
    for sev in range(5):
        rws = match_df[match_df["severity"].astype(int) == int(sev)]
        if len(rws) == 0:
            out_sev.append(
                {
                    "severity": f"L{sev}",
                    "unit_id": None,
                    "gradcam": None,
                    "fastcav": None,
                    "note": None,
                }
            )
            continue
        if match_mode == "stem":
            row = rws.iloc[0]
        else:
            sort_cols = [c for c in ("object_uid", "unit_id") if c in rws.columns]
            if not sort_cols:
                sort_cols = [mcol]
            row = rws.sort_values(by=sort_cols).iloc[0]
        uid = str(row.get("unit_id", "")).strip()
        if not uid:
            out_sev.append(
                {
                    "severity": f"L{sev}",
                    "unit_id": None,
                    "gradcam": None,
                    "fastcav": None,
                    "note": "unit_id 비어 있음",
                }
            )
            continue
        udir = bundle_root / uid
        note = None
        if match_mode == "image_id" and len(rws) > 1:
            note = f"image_id만 일치: object_uid={row.get('object_uid', '')} 기준 (동일 이미지 객체 {len(rws)}개)"
        g = _read_user_study_explanation(udir, "explanation_gradcam.txt")
        f = _read_user_study_explanation(udir, "explanation_fastcav.txt")
        out_sev.append(
            {
                "severity": f"L{sev}",
                "unit_id": uid,
                "gradcam": g,
                "fastcav": f,
                "note": note,
            }
        )
    any_text = any(
        (s.get("gradcam") or s.get("fastcav")) for s in out_sev
    )
    return jsonify(
        ok=True,
        match_mode=match_mode,
        message=None if any_text else "해당 샘플/심각도에 explanation_*.txt 가 없습니다. export 시 --skip-llm 을 빼세요.",
        severities=out_sev,
    )


# Grad-CAM metrics to show (and compute change rate from L0)
CAM_METRIC_KEYS = [
    "bbox_center_activation_distance", "peak_bbox_distance",
    "entropy", "activation_spread", "center_shift",
    "activation_fragmentation", "energy_in_bbox", "energy_in_bbox_1_1x", "energy_in_bbox_1_25x",
    "ring_energy_ratio",
    "full_cam_sum", "full_cam_entropy",
]


@app.route("/api/models/<model>/<corruption>/sample/<path:sample_id>/cam_metrics")
def api_sample_cam_metrics(model, corruption, sample_id):
    """Return Grad-CAM metrics per severity (L0..L4) and change rate from L0 for one sample. Optional query: xai_method."""
    xai_method = request.args.get("xai_method", "").strip() or None
    if _is_fastcav_viewer(xai_method):
        return jsonify(severities=[])
    sample_id_base = sample_id
    if sample_id_base.endswith(".png") or sample_id_base.endswith(".jpg") or sample_id_base.endswith(".jpeg"):
        sample_id_base = sample_id_base.rsplit(".", 1)[0]
    df = _load_cam_records_df()
    if df is None or len(df) == 0:
        return jsonify(severities=[])
    model_col = "model_id" if "model_id" in df.columns else "model"
    if model_col not in df.columns:
        return jsonify(severities=[])
    subset = df[
        (df[model_col].astype(str) == str(model)) &
        (df["corruption"].astype(str) == str(corruption))
    ].copy()
    if "xai_method" in subset.columns:
        if xai_method:
            xm = _canonical_xai_method(str(xai_method))
            if xm == "gradcampp":
                subset = subset[subset["xai_method"].astype(str).isin(["gradcampp", "fastcam"])]
            else:
                subset = subset[subset["xai_method"].astype(str) == str(xm)]
        else:
            # 기본값: gradcam (gradcampp가 먼저 있으면 0만 나오는 것 방지)
            subset = subset[subset["xai_method"].astype(str) == "gradcam"]
    if "object_id" not in subset.columns:
        subset["_key"] = subset["image_id"].astype(str)
    else:
        subset["_key"] = subset["image_id"].astype(str) + "_" + subset["object_id"].astype(str)
    subset = subset[subset["_key"] == sample_id_base]
    if "cam_status" in subset.columns:
        subset = subset[subset["cam_status"] == "ok"]
    if "layer_role" in subset.columns:
        subset = subset[subset["layer_role"] == "primary"]
    if len(subset) == 0:
        return jsonify(severities=[])
    # severity 컬럼이 문자열이어도 비교되도록 int로 통일
    if "severity" in subset.columns:
        subset = subset.copy()
        subset["_sev"] = subset["severity"].astype(int)
    else:
        subset["_sev"] = subset["severity"]
    out = []
    for sev in range(5):
        rows = subset[subset["_sev"] == sev]
        if len(rows) == 0:
            out.append({"severity": f"L{sev}", "metrics": None, "change_pct": None})
            continue
        # 여러 행(예: gradcam+gradcampp)이면 이미 xai_method로 필터됐으므로 첫 행 사용
        row = rows.iloc[0]
        metrics = {}
        for k in CAM_METRIC_KEYS:
            if k in row.index and row.get(k) is not None and str(row.get(k)) != "nan":
                try:
                    metrics[k] = round(float(row[k]), 6)
                except (TypeError, ValueError):
                    pass
        out.append({"severity": f"L{sev}", "metrics": metrics if metrics else None, "change_pct": None})
    # Compute change % from L0
    l0_metrics = out[0]["metrics"] or {}
    for i in range(1, 5):
        m = out[i]["metrics"]
        if not m or not l0_metrics:
            continue
        change = {}
        for k in CAM_METRIC_KEYS:
            if k not in l0_metrics or k not in m:
                continue
            base_val = l0_metrics[k]
            if base_val is None or base_val == 0:
                continue
            try:
                pct = 100.0 * (float(m[k]) - float(base_val)) / float(base_val)
                change[k] = round(pct, 2)
            except (TypeError, ValueError):
                pass
        out[i]["change_pct"] = change if change else None
    return jsonify(severities=out)


# 변조별 전체 집계용 메트릭 (그래프 4개와 동일)
# Primary for gradual curves: distance (continuous); E_bbox is auxiliary (bimodal for tiny objects)
AGGREGATE_METRIC_KEYS = [
    "bbox_center_activation_distance", "peak_bbox_distance",
    "entropy", "activation_spread",
    "energy_in_bbox_1_25x", "ring_energy_ratio",
]


@app.route("/api/aggregate/cam_metrics")
def api_aggregate_cam_metrics():
    """변조(corruption)별 전체 샘플 집계: L0~L4별 mean, std, n. Optional: model, xai_method."""
    xai_method = request.args.get("xai_method", "").strip() or None
    if _is_fastcav_viewer(xai_method):
        return jsonify(corruptions=[])
    df = _load_cam_records_df()
    if df is None or len(df) == 0:
        return jsonify(corruptions=[])
    model = request.args.get("model", "").strip() or None
    model_col = "model_id" if "model_id" in df.columns else "model"
    if model_col not in df.columns:
        return jsonify(corruptions=[])
    subset = df.copy()
    if model:
        subset = subset[subset[model_col].astype(str) == str(model)]
    if "xai_method" in subset.columns:
        if xai_method:
            xm = _canonical_xai_method(str(xai_method))
            if xm == "gradcampp":
                subset = subset[subset["xai_method"].astype(str).isin(["gradcampp", "fastcam"])]
            else:
                subset = subset[subset["xai_method"].astype(str) == str(xm)]
        else:
            # gradcam만 있으면 gradcam, 없으면 전체 사용 (집계가 비지 않도록)
            subset_g = subset[subset["xai_method"].astype(str) == "gradcam"]
            subset = subset_g if len(subset_g) > 0 else subset
    if "cam_status" in subset.columns:
        subset = subset[subset["cam_status"] == "ok"]
    if "layer_role" in subset.columns:
        subset = subset[subset["layer_role"] == "primary"]
    subset = subset.copy()
    subset["_sev"] = subset["severity"].astype(int)
    # 변조별 · severity별 집계
    out_corruptions = []
    for corruption in sorted(subset["corruption"].unique()):
        cdf = subset[subset["corruption"].astype(str) == str(corruption)]
        by_sev = []
        for sev in range(5):
            rows = cdf[cdf["_sev"] == sev]
            mean_dict = {}
            std_dict = {}
            n_dict = {}
            for k in AGGREGATE_METRIC_KEYS:
                if k not in rows.columns:
                    continue
                try:
                    vals = rows[k].astype(float)
                except (TypeError, ValueError):
                    continue
                vals = vals.replace([float("inf"), float("-inf")], float("nan")).dropna()
                if len(vals) == 0:
                    continue
                mean_dict[k] = round(float(vals.mean()), 6)
                std_dict[k] = round(float(vals.std()), 6) if len(vals) > 1 else 0.0
                n_dict[k] = int(len(vals))
            by_sev.append({
                "severity": f"L{sev}",
                "mean": mean_dict,
                "std": std_dict,
                "n": n_dict,
            })
        out_corruptions.append({"corruption": str(corruption), "by_severity": by_sev})
    return jsonify(corruptions=out_corruptions)


@app.route("/api/user_study/llm_preview")
def api_user_study_llm_preview():
    """First user_study_bundles unit with explanation_*.txt (Grad-CAM / FastCAV LLM outputs)."""
    bundle_root = RESULTS_DIR / "user_study_bundles"
    if not bundle_root.is_dir():
        return jsonify(
            ok=False,
            message="results/user_study_bundles 폴더가 없습니다. python scripts/12_user_study_export.py 를 실행하세요.",
            gradcam=None,
            fastcav=None,
            unit_id=None,
        )
    max_chars = 12000
    for unit_dir in sorted(bundle_root.iterdir()):
        if not unit_dir.is_dir() or not unit_dir.name.startswith("unit_"):
            continue
        eg = unit_dir / "explanation_gradcam.txt"
        ef = unit_dir / "explanation_fastcav.txt"
        if not eg.exists() and not ef.exists():
            continue
        g_text = f_text = None
        try:
            if eg.exists():
                g_text = eg.read_text(encoding="utf-8", errors="replace")
                if len(g_text) > max_chars:
                    g_text = g_text[:max_chars] + "\n… (truncated)"
            if ef.exists():
                f_text = ef.read_text(encoding="utf-8", errors="replace")
                if len(f_text) > max_chars:
                    f_text = f_text[:max_chars] + "\n… (truncated)"
        except OSError:
            continue
        return jsonify(
            ok=True,
            message=None,
            unit_id=unit_dir.name,
            gradcam=g_text,
            fastcav=f_text,
        )
    return jsonify(
        ok=False,
        message="explanation_gradcam.txt / explanation_fastcav.txt 가 없습니다. "
        "OPENAI_API_KEY 설정 후 --skip-llm 없이 export 하세요.",
        gradcam=None,
        fastcav=None,
        unit_id=None,
    )


@app.route("/api/metrics")
def api_metrics():
    """Overall experiment metrics: lead_stats, dasc_summary."""
    return jsonify(get_metrics())


@app.route("/fastcav")
def fastcav_page():
    """FastCAV: bbox visualization + pseudo-heatmap overlay (same concepts)."""
    fc_app.load_fastcav_tables(ROOT)
    if not fc_app.fastcav_available(ROOT):
        return render_template_string(FASTCAV_UNAVAILABLE_HTML), 503
    return render_template_string(FASTCAV_HTML)


@app.route("/api/fastcav/status")
def api_fastcav_status():
    fc_app.load_fastcav_tables(ROOT)
    return jsonify(ok=fc_app.fastcav_available(ROOT))


@app.route("/api/fastcav/models")
def api_fastcav_models():
    fc_app.load_fastcav_tables(ROOT)
    return jsonify(models=fc_app.list_models(ROOT))


@app.route("/api/fastcav/models/<path:model>/corruptions")
def api_fastcav_corruptions(model):
    fc_app.load_fastcav_tables(ROOT)
    return jsonify(corruptions=fc_app.list_corruptions(ROOT, model))


@app.route("/api/fastcav/models/<path:model>/<path:corruption>/samples")
def api_fastcav_samples(model, corruption):
    fc_app.load_fastcav_tables(ROOT)
    return jsonify(samples=fc_app.list_samples_intersection(ROOT, model, corruption))


@app.route("/api/fastcav/models/<path:model>/<path:corruption>/sample/<path:image_id>")
def api_fastcav_sample_severities(model, corruption, image_id):
    fc_app.load_fastcav_tables(ROOT)
    result = []
    for sev in range(5):
        q = (
            f"model={quote(str(model), safe='')}"
            f"&corruption={quote(str(corruption), safe='')}"
            f"&severity={sev}"
            f"&image_id={quote(str(image_id), safe='')}"
        )
        result.append(
            {
                "severity": f"L{sev}",
                "url": f"/fastcav/render?{q}",
                "pseudo_url": f"/fastcav/render_pseudo?{q}",
            }
        )
    return jsonify(severities=result)


@app.route("/fastcav/render")
def fastcav_render():
    """PNG: bbox + class / conf / concept (on-the-fly from CSV + corrupted image)."""
    model = request.args.get("model", "").strip()
    corruption = request.args.get("corruption", "").strip()
    image_id = request.args.get("image_id", "").strip()
    if not model or not corruption or not image_id:
        return "Bad request", 400
    try:
        severity = int(request.args.get("severity", "0"))
    except ValueError:
        return "Bad severity", 400
    if severity < 0 or severity > 4:
        return "Bad severity", 400
    try:
        thr = float(request.args.get("threshold", "0.3"))
    except ValueError:
        thr = 0.3
    fc_app.load_fastcav_tables(ROOT)
    data = fc_app.render_fastcav_png_bytes(
        ROOT, model, corruption, severity, image_id, threshold=thr
    )
    if data is None:
        return "Not found", 404
    return send_file(io.BytesIO(data), mimetype="image/png")


@app.route("/fastcav/render_pseudo")
def fastcav_render_pseudo():
    """PNG: FastCAV pseudo-heatmap overlay (Sobel prior × concept stress; not localization)."""
    model = request.args.get("model", "").strip()
    corruption = request.args.get("corruption", "").strip()
    image_id = request.args.get("image_id", "").strip()
    object_uid = request.args.get("object_uid", "").strip() or None
    if not model or not corruption or not image_id:
        return "Bad request", 400
    try:
        severity = int(request.args.get("severity", "0"))
    except ValueError:
        return "Bad severity", 400
    if severity < 0 or severity > 4:
        return "Bad severity", 400
    gaussian_sigma = None
    raw_sigma = request.args.get("sigma", "").strip()
    if raw_sigma:
        try:
            gaussian_sigma = float(raw_sigma)
        except ValueError:
            return "Bad sigma", 400
    fc_app.load_fastcav_tables(ROOT)
    data = fc_app.render_fastcav_pseudo_png_bytes(
        ROOT,
        model,
        corruption,
        severity,
        image_id,
        object_uid=object_uid,
        gaussian_sigma=gaussian_sigma,
    )
    if data is None:
        return "Not found", 404
    return send_file(io.BytesIO(data), mimetype="image/png")


@app.route("/heatmaps/<path:filepath>")
def serve_heatmap(filepath):
    """Serve a single image from heatmap_samples."""
    base = HEATMAP_DIR.resolve()
    path = (base / filepath).resolve()
    try:
        path.relative_to(base)
    except ValueError:
        return "Forbidden", 403
    if not path.exists() or not path.is_file():
        return "Not found", 404
    return send_from_directory(path.parent, path.name)


FASTCAV_UNAVAILABLE_HTML = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>FastCAV</title>
<style>body{font-family:system-ui;background:#0f1419;color:#e6edf3;padding:24px;}
a{color:#00d4aa;}</style></head><body>
<h1>FastCAV bbox 뷰어</h1>
<p>데이터가 없습니다. <code>results/detection_records.csv</code>와
<code>results/fastcav_tiny_concept_scores.csv</code> (또는 <code>fastcav_concept_scores.csv</code>)를 생성한 뒤 다시 시도하세요.</p>
<p><a href="/">Heatmap Viewer로</a></p>
</body></html>"""

FASTCAV_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FastCAV · bbox + pseudo heatmap</title>
  <style>
    :root {
      --bg: #0f1419;
      --card: #1a2332;
      --accent: #00d4aa;
      --text: #e6edf3;
      --muted: #8b949e;
      --border: rgba(139,148,158,0.25);
    }
    * { box-sizing: border-box; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 0; line-height: 1.5; min-height: 100vh; }
    .header { padding: 16px 24px; background: var(--card); border-bottom: 1px solid var(--border); }
    .header h1 { font-size: 1.35rem; margin: 0 0 4px 0; color: var(--accent); }
    .header .subtitle { color: var(--muted); font-size: 0.85rem; margin: 0; }
    .header a { color: var(--accent); }
    .layout { display: flex; padding: 0 24px 24px; gap: 20px; min-height: 60vh; }
    .sidebar { width: 300px; flex-shrink: 0; background: var(--card); border-radius: 12px; border: 1px solid var(--border); display: flex; flex-direction: column; max-height: 75vh; }
    .filters-inline { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; padding: 12px 14px; border-bottom: 1px solid var(--border); }
    .filters-inline label { color: var(--muted); font-size: 0.8rem; margin-right: 6px; }
    .filters-inline select, .filters-inline input[type="number"] {
      background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; font-size: 0.9rem; min-width: 100px;
    }
    .sidebar-head { padding: 12px 14px; border-bottom: 1px solid var(--border); font-size: 0.85rem; color: var(--muted); }
    .sample-list { flex: 1; overflow-y: auto; padding: 8px; }
    .sample-item { padding: 10px 12px; margin-bottom: 4px; border-radius: 8px; font-size: 0.75rem; cursor: pointer; word-break: break-all; color: var(--text); background: transparent; border: 1px solid transparent; width: 100%; text-align: left; }
    .sample-item:hover { background: rgba(0,212,170,0.1); border-color: var(--border); }
    .sample-item.active { background: rgba(0,212,170,0.2); border-color: var(--accent); color: var(--accent); }
    .main { flex: 1; min-width: 0; background: var(--card); border-radius: 12px; border: 1px solid var(--border); padding: 20px; }
    .sample-title { font-size: 0.85rem; color: var(--muted); margin-bottom: 16px; word-break: break-all; }
    .severity-row { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
    .severity-cell { flex: 1 1 140px; max-width: 240px; background: var(--bg); border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }
    .severity-cell .sev-label { padding: 8px 10px; font-size: 0.75rem; font-weight: 600; color: var(--accent); border-bottom: 1px solid var(--border); }
    .severity-cell img { width: 100%; height: auto; display: block; cursor: pointer; }
    .dual-stack { display: flex; flex-direction: column; gap: 6px; padding: 6px; }
    .mini-label { font-size: 0.65rem; color: var(--muted); font-weight: 600; }
    .empty-main { color: var(--muted); text-align: center; padding: 48px 24px; }
    .hint { font-size: 0.8rem; color: var(--muted); margin-top: 12px; }
    .modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9); z-index: 100; align-items: center; justify-content: center; padding: 24px; }
    .modal.show { display: flex; }
    .modal img { max-width: 100%; max-height: 90vh; border-radius: 8px; }
    .modal .close { position: absolute; top: 16px; right: 24px; color: #fff; font-size: 28px; cursor: pointer; }
  </style>
</head>
<body>
  <div class="header">
    <h1>FastCAV · bbox + pseudo heatmap</h1>
    <p class="subtitle">위: 박스·개념 점수. 아래: pseudo-heatmap (고정 공간 prior × 개념 스트레스, localization 아님). L0~L4 동일 <code>image_id</code>.
      <a href="/">Heatmap Viewer</a></p>
  </div>
  <div class="layout">
    <aside class="sidebar">
      <div class="filters-inline">
        <div><label>모델</label><select id="model"><option value="">선택</option></select></div>
        <div><label>Corruption</label><select id="corruption" disabled><option value="">선택</option></select></div>
        <div><label>임계값</label><input type="number" id="threshold" value="0.3" step="0.05" min="0" max="1" title="가시성 등: 이 값 미만이면 붉은 쪽"></div>
      </div>
      <div class="sidebar-head">샘플 (L0~L4 모두 있는 image_id만)</div>
      <div id="sampleList" class="sample-list"></div>
    </aside>
    <main class="main">
      <div id="sampleTitle" class="sample-title" style="display:none;"></div>
      <div id="severityRow" class="severity-row"></div>
      <div id="emptyMain" class="empty-main">모델·Corruption을 선택한 뒤 샘플을 고르세요.</div>
      <p class="hint">이미지는 <code>detection_records</code>의 경로에서 읽습니다. 임계값을 바꾼 뒤에는 샘플을 다시 클릭하면 반영됩니다.</p>
    </main>
  </div>
  <div id="modal" class="modal">
    <span class="close" onclick="document.getElementById('modal').classList.remove('show')">&times;</span>
    <img id="modalImg" src="" alt="">
  </div>
  <script>
    const modelEl = document.getElementById('model');
    const corruptionEl = document.getElementById('corruption');
    const thresholdEl = document.getElementById('threshold');
    const sampleListEl = document.getElementById('sampleList');
    const sampleTitleEl = document.getElementById('sampleTitle');
    const severityRowEl = document.getElementById('severityRow');
    const emptyMainEl = document.getElementById('emptyMain');
    const modal = document.getElementById('modal');
    const modalImg = document.getElementById('modalImg');

    function thrParam() {
      const t = parseFloat(thresholdEl.value);
      return Number.isFinite(t) ? t : 0.3;
    }

    function withThreshold(url) {
      const sep = url.indexOf('?') >= 0 ? '&' : '?';
      return url + sep + 'threshold=' + encodeURIComponent(String(thrParam()));
    }

    fetch('/api/fastcav/models').then(r => r.json()).then(d => {
      const list = d.models || [];
      list.forEach(m => {
        const o = document.createElement('option');
        o.value = m; o.textContent = m;
        modelEl.appendChild(o);
      });
      if (list.length === 1) { modelEl.value = list[0]; modelEl.dispatchEvent(new Event('change')); }
    });

    modelEl.addEventListener('change', () => {
      corruptionEl.innerHTML = '<option value="">선택</option>';
      corruptionEl.disabled = true;
      sampleListEl.innerHTML = '';
      severityRowEl.innerHTML = '';
      emptyMainEl.style.display = 'block';
      sampleTitleEl.style.display = 'none';
      const m = modelEl.value;
      if (!m) return;
      fetch('/api/fastcav/models/' + encodeURIComponent(m) + '/corruptions').then(r => r.json()).then(d => {
        (d.corruptions || []).forEach(c => {
          const o = document.createElement('option');
          o.value = c; o.textContent = c;
          corruptionEl.appendChild(o);
        });
        corruptionEl.disabled = false;
        if ((d.corruptions || []).length === 1) {
          corruptionEl.value = d.corruptions[0];
          corruptionEl.dispatchEvent(new Event('change'));
        }
      });
    });

    corruptionEl.addEventListener('change', () => {
      sampleListEl.innerHTML = '';
      severityRowEl.innerHTML = '';
      emptyMainEl.style.display = 'block';
      sampleTitleEl.style.display = 'none';
      const m = modelEl.value, c = corruptionEl.value;
      if (!m || !c) return;
      fetch('/api/fastcav/models/' + encodeURIComponent(m) + '/' + encodeURIComponent(c) + '/samples')
        .then(r => r.json()).then(d => {
          sampleListEl.innerHTML = '';
          const samples = d.samples || [];
          samples.forEach(id => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'sample-item';
            btn.textContent = id;
            btn.dataset.sample = id;
            btn.onclick = () => selectSample(m, c, id);
            sampleListEl.appendChild(btn);
          });
          if (samples.length === 0)
            sampleListEl.innerHTML = '<span style="color:var(--muted);font-size:0.85rem;">조건을 만족하는 샘플 없음</span>';
        });
    });

    function selectSample(model, corruption, imageId) {
      document.querySelectorAll('.sample-item').forEach(el => {
        el.classList.toggle('active', el.dataset.sample === imageId);
      });
      sampleTitleEl.textContent = imageId;
      sampleTitleEl.style.display = 'block';
      emptyMainEl.style.display = 'none';
      severityRowEl.innerHTML = '';
      fetch('/api/fastcav/models/' + encodeURIComponent(model) + '/' + encodeURIComponent(corruption) + '/sample/' + encodeURIComponent(imageId))
        .then(r => r.json()).then(d => {
          severityRowEl.innerHTML = '';
          (d.severities || []).forEach(({ severity, url, pseudo_url }) => {
            const cell = document.createElement('div');
            cell.className = 'severity-cell';
            cell.innerHTML = '<div class="sev-label">' + severity + '</div>';
            const stack = document.createElement('div');
            stack.className = 'dual-stack';
            const lb = document.createElement('div');
            lb.className = 'mini-label';
            lb.textContent = 'Bbox';
            const imgB = document.createElement('img');
            imgB.alt = severity + ' bbox';
            imgB.loading = 'lazy';
            imgB.src = withThreshold(url);
            imgB.onclick = () => { modalImg.src = withThreshold(url); modal.classList.add('show'); };
            const lp = document.createElement('div');
            lp.className = 'mini-label';
            lp.textContent = 'Pseudo heatmap';
            const imgP = document.createElement('img');
            imgP.alt = severity + ' pseudo';
            imgP.loading = 'lazy';
            imgP.src = pseudo_url || '';
            imgP.onclick = () => { modalImg.src = pseudo_url || ''; modal.classList.add('show'); };
            stack.appendChild(lb);
            stack.appendChild(imgB);
            stack.appendChild(lp);
            stack.appendChild(imgP);
            cell.appendChild(stack);
            severityRowEl.appendChild(cell);
          });
        });
    }
  </script>
</body>
</html>"""

INDEX_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Heatmap Viewer · 샘플별 변조 단계 비교</title>
  <style>
    :root {
      --bg: #0f1419;
      --card: #1a2332;
      --accent: #00d4aa;
      --text: #e6edf3;
      --muted: #8b949e;
      --border: rgba(139,148,158,0.25);
    }
    * { box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 0;
      line-height: 1.5;
      min-height: 100vh;
    }
    .header {
      padding: 16px 24px;
      background: var(--card);
      border-bottom: 1px solid var(--border);
    }
    .header h1 { font-size: 1.35rem; margin: 0 0 4px 0; color: var(--accent); }
    .header .subtitle { color: var(--muted); font-size: 0.85rem; margin: 0; }

    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 12px;
      padding: 16px 24px;
      background: var(--card);
      margin: 0 24px 16px 24px;
      border-radius: 12px;
      border: 1px solid var(--border);
    }
    .metric {
      padding: 10px 12px;
      background: var(--bg);
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    .metric .label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
    .metric .value { font-size: 1.1rem; font-weight: 600; color: var(--accent); }

    .layout {
      display: flex;
      padding: 0 24px 24px;
      gap: 20px;
      min-height: 60vh;
    }
    .sidebar {
      width: 280px;
      flex-shrink: 0;
      background: var(--card);
      border-radius: 12px;
      border: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      max-height: 70vh;
    }
    .sidebar .sidebar-head {
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      font-size: 0.85rem;
      color: var(--muted);
    }
    .sidebar .sample-list {
      flex: 1;
      overflow-y: auto;
      padding: 8px;
    }
    .sample-item {
      padding: 10px 12px;
      margin-bottom: 4px;
      border-radius: 8px;
      font-size: 0.8rem;
      cursor: pointer;
      word-break: break-all;
      color: var(--text);
      background: transparent;
      border: 1px solid transparent;
    }
    .sample-item:hover { background: rgba(0,212,170,0.1); border-color: var(--border); }
    .sample-item.active { background: rgba(0,212,170,0.2); border-color: var(--accent); color: var(--accent); }

    .filters-inline {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
    }
    .filters-inline label { color: var(--muted); font-size: 0.8rem; margin-right: 6px; }
    .filters-inline select {
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 0.9rem;
      min-width: 120px;
    }

    .main {
      flex: 1;
      min-width: 0;
      background: var(--card);
      border-radius: 12px;
      border: 1px solid var(--border);
      padding: 20px;
      display: flex;
      flex-direction: column;
    }
    .main .sample-title {
      font-size: 0.85rem;
      color: var(--muted);
      margin-bottom: 16px;
      word-break: break-all;
    }
    .severity-row {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: flex-start;
      justify-content: flex-start;
    }
    .severity-cell {
      flex: 1 1 140px;
      max-width: 220px;
      background: var(--bg);
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    .severity-cell .sev-label {
      padding: 8px 10px;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--accent);
      border-bottom: 1px solid var(--border);
    }
    .severity-cell img {
      width: 100%;
      height: auto;
      display: block;
      cursor: pointer;
    }
    .severity-cell .llm-under {
      margin-top: 8px;
      padding: 0 8px 10px 8px;
      border-top: 1px solid var(--border);
    }
    .severity-cell .llm-under .llm-arm-title {
      font-size: 0.65rem;
      font-weight: 600;
      color: var(--muted);
      margin: 8px 0 4px 0;
    }
    .severity-cell .llm-under pre.llm-pre-cell {
      margin: 0;
      padding: 8px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 0.65rem;
      line-height: 1.4;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 200px;
      overflow-y: auto;
      color: var(--text);
    }
    .severity-cell .llm-under .llm-miss {
      font-size: 0.65rem;
      color: var(--muted);
      margin: 8px 0 0 0;
    }
    .comparison-section {
      margin-top: 24px;
      padding-top: 20px;
      border-top: 1px solid var(--border);
    }
    .comparison-section h3 {
      font-size: 0.95rem;
      color: var(--accent);
      margin: 0 0 12px 0;
    }
    .metrics-table-wrap {
      overflow-x: auto;
      margin-bottom: 20px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    .metrics-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.8rem;
    }
    .metrics-table th, .metrics-table td {
      padding: 8px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    .metrics-table th {
      background: var(--bg);
      color: var(--muted);
      font-weight: 600;
    }
    .metrics-table tr:hover td { background: rgba(0,212,170,0.06); }
    .metrics-table .num { text-align: right; font-variant-numeric: tabular-nums; }
    .metrics-table .pct-pos { color: #7ee787; }
    .metrics-table .pct-neg { color: #f97583; }
    .charts-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 16px;
      margin-top: 12px;
    }
    .chart-cell {
      position: relative;
      height: 200px;
      background: var(--bg);
      border-radius: 8px;
      border: 1px solid var(--border);
      padding: 10px;
    }
    .chart-cell .chart-title {
      font-size: 0.75rem;
      color: var(--accent);
      margin-bottom: 6px;
    }
    .chart-cell .chart-hint {
      font-size: 0.65rem;
      color: var(--muted);
      font-weight: normal;
      margin-left: 6px;
    }
    .chart-cell canvas {
      width: 100% !important;
      height: 160px !important;
    }
    .empty-main {
      color: var(--muted);
      text-align: center;
      padding: 48px 24px;
    }
    .loading { color: var(--muted); padding: 24px; }

    .modal {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.9);
      z-index: 100;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .modal.show { display: flex; }
    .modal img { max-width: 100%; max-height: 90vh; border-radius: 8px; }
    .modal .close { position: absolute; top: 16px; right: 24px; color: #fff; font-size: 28px; cursor: pointer; }

    .aggregate-section { margin-top: 24px; padding-top: 20px; border-top: 1px solid var(--border); }
    .aggregate-section h2 { font-size: 1rem; color: var(--accent); margin: 0 0 12px 0; }
    .aggregate-corruption { margin-bottom: 28px; padding: 16px; background: var(--bg); border-radius: 12px; border: 1px solid var(--border); }
    .aggregate-corruption h3 { font-size: 0.95rem; color: var(--text); margin: 0 0 12px 0; }
    .aggregate-corruption .agg-table-wrap { overflow-x: auto; margin-bottom: 16px; border-radius: 8px; border: 1px solid var(--border); }
    .aggregate-corruption .agg-charts { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-top: 12px; }
    .aggregate-corruption .agg-chart-cell { height: 200px; background: var(--card); border-radius: 8px; border: 1px solid var(--border); padding: 10px; }
    .aggregate-corruption .agg-chart-cell .chart-title { font-size: 0.75rem; color: var(--accent); margin-bottom: 6px; }
    hr.agg-sep {
      border: none;
      border-top: 1px solid var(--border);
      margin: 24px 0 16px 0;
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>Heatmap Viewer</h1>
    <p class="subtitle">XAI 방법에서 <strong>FastCAV (pseudo heatmap)</strong>을 고르면 동일 UI로 L0~L4 pseudo-heatmap을 봅니다. Grad-CAM / Grad-CAM++ / LayerCAM은 heatmap_samples 기준 · <a href="/fastcav" style="color:var(--accent);">전용 FastCAV 페이지</a>(bbox+pseudo)</p>
  </div>

  <div id="metrics" class="metrics"></div>

  <div class="layout">
    <aside class="sidebar">
      <div class="filters-inline">
        <div>
          <label>XAI 방법</label>
          <select id="xaiMethod">
            <option value="">선택</option>
            {% for m in xai_methods %}
            <option value="{{ m.id }}">{{ m.label }}</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>모델</label>
          <select id="model" disabled>
            <option value="">선택</option>
          </select>
        </div>
        <div>
          <label>Corruption</label>
          <select id="corruption" disabled>
            <option value="">선택</option>
          </select>
        </div>
        <div style="display:flex;align-items:center;gap:6px;">
          <input type="checkbox" id="idealOnlyCheckbox" style="cursor:pointer;">
          <label for="idealOnlyCheckbox" style="margin:0;cursor:pointer;font-size:0.8rem;color:var(--muted);">이상적 추세만 보기</label>
        </div>
      </div>
      <div class="sidebar-head">샘플 목록 (선택 시 동일 이미지 L0~L4 표시)</div>
      <div id="sampleList" class="sample-list"></div>
    </aside>

    <main class="main">
      <div id="sampleTitle" class="sample-title" style="display:none;"></div>
      <div id="severityRow" class="severity-row"></div>
      <div id="emptyMain" class="empty-main">모델을 선택하면 변조별 전체 통계가 표시됩니다. 샘플 선택 시 L0~L4 시각화(Grad-CAM 계열 또는 FastCAV pseudo)가 표시됩니다.</div>
      <div id="loadingMain" class="loading" style="display:none;">로딩 중...</div>

      <div id="aggregateSection" class="aggregate-section">
        <h2>변조별 전체 통계 (전체 샘플 집계)</h2>
        <p style="color:var(--muted);font-size:0.85rem;margin:0 0 16px 0;">모델·XAI 방법 기준으로 변조(corruption)별 L0~L4 평균±표준편차 및 그래프</p>
        <div id="aggregateContent"></div>
      </div>
    </main>
  </div>

  <div id="modal" class="modal">
    <span class="close" onclick="document.getElementById('modal').classList.remove('show')">&times;</span>
    <img id="modalImg" src="" alt="">
  </div>

  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script>
    const xaiMethodEl = document.getElementById('xaiMethod');
    const modelEl = document.getElementById('model');
    const corruptionEl = document.getElementById('corruption');
    const sampleListEl = document.getElementById('sampleList');
    const sampleTitleEl = document.getElementById('sampleTitle');
    const severityRowEl = document.getElementById('severityRow');
    const emptyMainEl = document.getElementById('emptyMain');
    const loadingMainEl = document.getElementById('loadingMain');
    const metricsEl = document.getElementById('metrics');
    const modal = document.getElementById('modal');
    const modalImg = document.getElementById('modalImg');

    async function loadMetrics() {
      try {
        const r = await fetch('/api/metrics');
        const d = await r.json();
        let html = '';
        const ls = d.lead_stats;
        if (ls) {
          const nTotal = ls.n_total || 0;
          const nLead = ls.n_lead || 0;
          const nCoincident = ls.n_coincident || 0;
          const nLag = ls.n_lag || 0;
          const denom = nLead + nCoincident + nLag || 1;
          const leadRate = denom > 0 ? (100 * nLead / denom).toFixed(1) : '-';
          const leadCoincidentPct = nTotal > 0 ? (100 * (nLead + nCoincident) / nTotal).toFixed(1) : '-';
          html += '<div class="metric"><div class="label">선행률 (Lead %)</div><div class="value">' + leadRate + '%</div></div>';
          html += '<div class="metric"><div class="label">선행+동시 비율</div><div class="value">' + leadCoincidentPct + '%</div></div>';
          html += '<div class="metric"><div class="label">선행 건수</div><div class="value">' + nLead + '</div></div>';
          html += '<div class="metric"><div class="label">동시 건수</div><div class="value">' + nCoincident + '</div></div>';
          html += '<div class="metric"><div class="label">총 이벤트</div><div class="value">' + nTotal + '</div></div>';
          html += '<div class="metric"><div class="label">평균 Lead (프레임)</div><div class="value">' + (ls.mean_lead != null ? ls.mean_lead.toFixed(2) : '-') + '</div></div>';
          if (ls.sign_test && ls.sign_test.p_value != null)
            html += '<div class="metric"><div class="label">Sign test p-value</div><div class="value">' + ls.sign_test.p_value.toExponential(2) + '</div></div>';
          if (ls.permutation_test && ls.permutation_test.p_value != null)
            html += '<div class="metric"><div class="label">Permutation p-value</div><div class="value">' + ls.permutation_test.p_value.toExponential(2) + '</div></div>';
        }
        const ds = d.dasc_summary;
        if (ds && ds.miss_rate_curve && ds.miss_rate_curve.length) {
          const byCorr = {};
          ds.miss_rate_curve.forEach(function(e) {
            if (!byCorr[e.corruption]) byCorr[e.corruption] = [];
            byCorr[e.corruption].push(e);
          });
          Object.keys(byCorr).forEach(function(c) {
            const arr = byCorr[c];
            const l4 = arr.find(function(x) { return x.severity === 4; });
            if (l4) html += '<div class="metric"><div class="label">Miss rate ' + c + ' L4</div><div class="value">' + (l4.miss_rate * 100).toFixed(1) + '%</div></div>';
          });
        }
        metricsEl.innerHTML = html || '<div class="metric"><div class="label">지표</div><div class="value">데이터 없음</div></div>';
      } catch (e) {
        metricsEl.innerHTML = '<div class="metric"><div class="label">지표</div><div class="value">로드 실패</div></div>';
      }
    }

    function qs(extra) {
      const method = (xaiMethodEl && xaiMethodEl.value) ? xaiMethodEl.value : '';
      const s = method ? '?xai_method=' + encodeURIComponent(method) : '';
      return (extra ? s + (s ? '&' : '?') + extra : s);
    }
    if (xaiMethodEl) xaiMethodEl.addEventListener('change', async () => {
      modelEl.innerHTML = '<option value="">선택</option>';
      modelEl.disabled = true;
      corruptionEl.innerHTML = '<option value="">선택</option>';
      corruptionEl.disabled = true;
      sampleListEl.innerHTML = '';
      severityRowEl.innerHTML = '';
      emptyMainEl.style.display = 'block';
      sampleTitleEl.style.display = 'none';
      const method = xaiMethodEl.value;
      if (!method) return;
      const r = await fetch('/api/models' + qs());
      const d = await r.json();
      const modelList = d.models || [];
      modelList.forEach(m => {
        const o = document.createElement('option');
        o.value = m;
        o.textContent = m;
        modelEl.appendChild(o);
      });
      modelEl.disabled = false;
      if (modelList.length === 1) {
        modelEl.value = modelList[0];
        modelEl.dispatchEvent(new Event('change'));
      } else {
        loadAggregateSection();
      }
    }); else {
      // No XAI dropdown (legacy): enable model dropdown and load models from root
      modelEl.disabled = false;
      if (modelEl.options.length > 1) modelEl.dispatchEvent(new Event('change'));
    }
    modelEl.addEventListener('change', async () => {
      corruptionEl.innerHTML = '<option value="">선택</option>';
      corruptionEl.disabled = true;
      sampleListEl.innerHTML = '';
      severityRowEl.innerHTML = '';
      emptyMainEl.style.display = 'block';
      sampleTitleEl.style.display = 'none';
      const m = modelEl.value;
      if (!m) {
        corruptionEl.disabled = false;
        return;
      }
      try {
        const url = '/api/models/' + encodeURIComponent(m) + '/corruptions' + qs();
        const r = await fetch(url);
        const d = r.ok ? await r.json() : { corruptions: [] };
        const list = d.corruptions || [];
        list.forEach(c => {
          const o = document.createElement('option');
          o.value = c;
          o.textContent = c;
          corruptionEl.appendChild(o);
        });
        if (list.length === 1) {
          corruptionEl.value = list[0];
          corruptionEl.dispatchEvent(new Event('change'));
        }
      } catch (e) { /* leave list empty */ }
      corruptionEl.disabled = false;
      loadAggregateSection();
    });

    let aggregateChartInstances = [];
    const AGG_METRIC_KEYS = ['bbox_center_activation_distance', 'peak_bbox_distance', 'activation_spread', 'ring_energy_ratio'];

    async function loadAggregateSection() {
      const aggSection = document.getElementById('aggregateSection');
      const aggContent = document.getElementById('aggregateContent');
      if (!aggSection || !aggContent) return;
      const model = modelEl.value;
      aggregateChartInstances.forEach(function(ch) { if (ch) ch.destroy(); });
      aggregateChartInstances = [];
      aggContent.innerHTML = '<div class="loading">변조별 집계 로딩 중...</div>';
      try {
        const params = [];
        if (model) params.push('model=' + encodeURIComponent(model));
        const q = qs().replace(/^[?]/, '');
        if (q) params.push(q);
        const url = '/api/aggregate/cam_metrics' + (params.length ? '?' + params.join('&') : '');
        const r = await fetch(url);
        const data = r.ok ? await r.json() : { corruptions: [] };
        const corruptions = data.corruptions || [];
        if (corruptions.length === 0) {
          const fc = xaiMethodEl && xaiMethodEl.value === 'fastcav';
          aggContent.innerHTML = '<p style="color:var(--muted);">' + (fc ? 'FastCAV는 Grad-CAM 계열 지표(cam_records) 집계 대상이 아닙니다.' : '집계 데이터 없음 (cam_records.csv 확인)') + '</p>';
          return;
        }
        aggContent.innerHTML = '';
        corruptions.forEach(function(cobj) {
          const corr = cobj.corruption;
          const safeId = corr.replace(/[^a-zA-Z0-9_]/g, '_');
          const bySev = cobj.by_severity || [];
          const card = document.createElement('div');
          card.className = 'aggregate-corruption';
          let tableHtml = '<table class="metrics-table"><thead><tr><th>단계</th>';
          AGG_METRIC_KEYS.forEach(function(k) {
            tableHtml += '<th class="num">' + (METRIC_LABELS[k] || k) + '</th>';
          });
          tableHtml += '</tr></thead><tbody>';
          bySev.forEach(function(s) {
            tableHtml += '<tr><td>' + s.severity + '</td>';
            AGG_METRIC_KEYS.forEach(function(k) {
              const mean = (s.mean && s.mean[k] != null) ? s.mean[k] : null;
              const std = (s.std && s.std[k] != null) ? s.std[k] : null;
              let cell = '-';
              if (mean != null) {
                cell = Number(mean).toFixed(3);
                if (std != null && std > 0) cell += ' ± ' + Number(std).toFixed(3);
              }
              tableHtml += '<td class="num">' + cell + '</td>';
            });
            tableHtml += '</tr>';
          });
          tableHtml += '</tbody></table>';
          let chartsHtml = '';
          AGG_METRIC_KEYS.forEach(function(k) {
            const cid = 'agg_' + safeId + '_' + k;
            chartsHtml += '<div class="agg-chart-cell"><div class="chart-title">' + (METRIC_LABELS[k] || k) + '</div><canvas id="' + cid + '"></canvas></div>';
          });
          card.innerHTML = '<h3>' + corr + '</h3><div class="agg-table-wrap">' + tableHtml + '</div><div class="agg-charts">' + chartsHtml + '</div>';
          aggContent.appendChild(card);
          // 차트 그리기
          const labels = bySev.map(function(s) { return s.severity; });
          AGG_METRIC_KEYS.forEach(function(k) {
            const cid = 'agg_' + safeId + '_' + k;
            const canvas = document.getElementById(cid);
            if (!canvas) return;
            const values = bySev.map(function(s) {
              const v = (s.mean && s.mean[k] != null) ? s.mean[k] : null;
              return v;
            });
            const hasData = values.some(function(v) { return v != null; });
            if (!hasData) return;
            const chart = new Chart(canvas, {
              type: 'line',
              data: {
                labels: labels,
                datasets: [{ label: METRIC_LABELS[k] || k, data: values, borderColor: '#00d4aa', backgroundColor: '#00d4aa20', fill: true, tension: 0.2, spanGaps: true }]
              },
              options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                  x: { grid: { color: 'rgba(139,148,158,0.2)' }, ticks: { color: '#8b949e', maxTicksLimit: 5 } },
                  y: { grid: { color: 'rgba(139,148,158,0.2)' }, ticks: { color: '#8b949e' } }
                }
              }
            });
            aggregateChartInstances.push(chart);
          });
        });
      } catch (e) {
        aggContent.innerHTML = '<p style="color:var(--muted);">집계 로드 실패</p>';
      }
    }

    function loadSampleList() {
      const m = modelEl.value;
      const c = corruptionEl.value;
      if (!m || !c) return;
      const idealOnly = document.getElementById('idealOnlyCheckbox').checked;
      const url = '/api/models/' + encodeURIComponent(m) + '/' + encodeURIComponent(c) + '/samples' + qs(idealOnly ? 'ideal_only=1' : '');
      fetch(url).then(r => r.json()).then(d => {
        sampleListEl.innerHTML = '';
        const samples = d.samples || [];
        samples.forEach(name => {
          const item = document.createElement('button');
          item.type = 'button';
          item.className = 'sample-item';
          item.textContent = name.replace(/\\.(png|jpg|jpeg)$/i, '');
          item.dataset.sample = name;
          item.onclick = () => selectSample(m, c, name);
          sampleListEl.appendChild(item);
        });
        if (samples.length === 0) sampleListEl.innerHTML = '<span style="color:var(--muted);font-size:0.85rem;">' + (idealOnly ? '이상적 추세 샘플 없음' : '샘플 없음') + '</span>';
      });
    }
    corruptionEl.addEventListener('change', async () => {
      sampleListEl.innerHTML = '';
      severityRowEl.innerHTML = '';
      emptyMainEl.style.display = 'block';
      sampleTitleEl.style.display = 'none';
      loadSampleList();
    });
    document.getElementById('idealOnlyCheckbox').addEventListener('change', () => {
      if (modelEl.value && corruptionEl.value) loadSampleList();
    });
    // On load: if single xai_method (non-empty value), select it and load models → then Corruption loads if single model
    (function initXai() {
      if (!xaiMethodEl) return;
      const opts = Array.from(xaiMethodEl.options).filter(function(o) { return o.value; });
      if (opts.length === 1) { xaiMethodEl.value = opts[0].value; xaiMethodEl.dispatchEvent(new Event('change')); }
      else loadAggregateSection();
    })();

    const METRIC_LABELS = {
      energy_in_bbox: 'E_bbox',
      energy_in_bbox_1_1x: 'E_bbox_1.1x',
      energy_in_bbox_1_25x: 'E_bbox_1.25x',
      ring_energy_ratio: 'E_ring_ratio',
      activation_spread: 'spread',
      entropy: 'entropy',
      center_shift: 'center_shift',
      activation_fragmentation: 'frag',
      bbox_center_activation_distance: 'bbox_dist',
      peak_bbox_distance: 'peak_dist',
      full_cam_sum: 'full_sum',
      full_cam_entropy: 'full_ent'
    };

    function selectSample(model, corruption, sampleId) {
      document.querySelectorAll('.sample-item').forEach(el => { el.classList.remove('active'); if (el.dataset.sample === sampleId) el.classList.add('active'); });
      sampleTitleEl.textContent = sampleId.replace(/\\.(png|jpg|jpeg)$/i, '');
      sampleTitleEl.style.display = 'block';
      emptyMainEl.style.display = 'none';
      loadingMainEl.style.display = 'block';
      severityRowEl.innerHTML = '';
      const base = '/api/models/' + encodeURIComponent(model) + '/' + encodeURIComponent(corruption) + '/sample/' + encodeURIComponent(sampleId) + qs();
      const llmParams = 'model=' + encodeURIComponent(model)
        + '&corruption=' + encodeURIComponent(corruption)
        + '&sample_id=' + encodeURIComponent(sampleId);
      fetch(base).then(r => r.json()).then(async (d) => {
        loadingMainEl.style.display = 'none';
        const sevs = d.severities || [];
        severityRowEl.innerHTML = '';
        const cells = [];
        sevs.forEach(({ severity, url }) => {
          const cell = document.createElement('div');
          cell.className = 'severity-cell';
          cell.innerHTML = '<div class="sev-label">' + severity + '</div>';
          const img = document.createElement('img');
          img.src = url;
          img.alt = severity;
          img.loading = 'lazy';
          img.onclick = () => { modalImg.src = url; modal.classList.add('show'); };
          cell.appendChild(img);
          const llmMount = document.createElement('div');
          llmMount.className = 'llm-under';
          llmMount.innerHTML = '<p class="llm-miss">LLM 해석 로딩…</p>';
          cell.appendChild(llmMount);
          severityRowEl.appendChild(cell);
          cells.push({ cell, severity, llmMount });
        });
        if (sevs.length === 0) {
          emptyMainEl.textContent = '이 샘플에 대한 이미지가 없습니다.';
          emptyMainEl.style.display = 'block';
          return;
        }
        try {
          const llmUrl = '/api/user_study/llm_by_sample' + qs(llmParams);
          const lr = await fetch(llmUrl);
          const ldata = await lr.json();
          const bySev = {};
          (ldata.severities || []).forEach(function(s) { bySev[s.severity] = s; });
          const isFc = xaiMethodEl && xaiMethodEl.value === 'fastcav';
          cells.forEach(function({ severity, llmMount }) {
            llmMount.innerHTML = '';
            const block = bySev[severity];
            if (!ldata.ok || !block) {
              llmMount.innerHTML = '<p class="llm-miss">' + (ldata.message || 'LLM 데이터 없음') + '</p>';
              return;
            }
            if (block.note) {
              const n = document.createElement('p');
              n.className = 'llm-miss';
              n.textContent = block.note;
              llmMount.appendChild(n);
            }
            const g = block.gradcam;
            const f = block.fastcav;
            function addArm(label, text) {
              if (!text) return;
              const t = document.createElement('div');
              t.className = 'llm-arm-title';
              t.textContent = label;
              llmMount.appendChild(t);
              const pre = document.createElement('pre');
              pre.className = 'llm-pre-cell';
              pre.textContent = text;
              llmMount.appendChild(pre);
            }
            if (isFc) {
              addArm('FastCAV (LLM)', f);
              if (g) addArm('Grad-CAM (LLM)', g);
            } else {
              addArm('Grad-CAM (LLM)', g);
              if (f) addArm('FastCAV (LLM)', f);
            }
            if (!g && !f) {
              const miss = document.createElement('p');
              miss.className = 'llm-miss';
              miss.textContent = ldata.message || ('unit ' + (block.unit_id || '') + ': explanation_*.txt 없음');
              llmMount.appendChild(miss);
            }
          });
        } catch (e) {
          cells.forEach(function({ llmMount }) {
            llmMount.innerHTML = '<p class="llm-miss">LLM 해석 요청 실패</p>';
          });
        }
      }).catch(() => { loadingMainEl.style.display = 'none'; emptyMainEl.textContent = '로드 실패'; emptyMainEl.style.display = 'block'; });
    }

    loadMetrics();
    loadAggregateSection();
  </script>
</body>
</html>
"""


def main():
    if not HEATMAP_DIR.exists():
        print(f"Heatmap directory not found: {HEATMAP_DIR}")
        print("Run scripts/05_gradcam_failure_analysis.py first to generate heatmaps.")
    else:
        print(f"Heatmap root: {HEATMAP_DIR}")
    fc_app.load_fastcav_tables(ROOT)
    if fc_app.fastcav_available(ROOT):
        print("FastCAV (bbox + pseudo heatmap): http://127.0.0.1:5000/fastcav")
    print("Open http://127.0.0.1:5000")
    # host="127.0.0.1" = local only; use "0.0.0.0" to allow LAN access
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
