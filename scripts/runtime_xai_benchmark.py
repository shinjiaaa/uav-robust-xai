"""
Runtime benchmark: Detector vs Detector+XAI under identical inputs.

Measures (per method):
  - mean / std / max inference time (ms) for one image + one bbox (batch=1)
  - FPS = 1 / mean_time_s
  - overhead_percent vs detector-only (explanation cost)
  - within_deadline_pct: share of iterations finishing within --deadline-ms (default 33 ms, ~30 FPS budget)

Pipeline:
  - Detector only: letterbox preprocess (384) + YOLO forward, no backward.
  - XAI methods: full generate_cam (forward + backward + CAM map) on same layer as 05 script.
  - FastCAV: concept-score post-process only (no CAM generation).

Output:
  - results/runtime_benchmark.csv
  - results/runtime_summary.json (same rows + deadline_ms metadata for report.md)

Usage:
  python scripts/runtime_xai_benchmark.py
  python scripts/runtime_xai_benchmark.py --samples 150 --warmup 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from src.utils.io import load_yaml
from src.utils.seed import set_seed
from src.xai.gradcam_yolo import YOLOGradCAM, YOLOGradCAMPP, YOLOLayerCAM
from src.data.bbox_conversion import visdrone_to_yolo_bbox


# ---------------------------------------------------------------------------
# FastCAV concept helpers (aligned with scripts/11_fastcav_concept_detection.py)
# ---------------------------------------------------------------------------

def _normalize_metric_series(series: pd.Series, clip_zero: bool = False):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return lambda x: 0.5
    vmin, vmax = float(s.min()), float(s.max())
    if vmax == vmin:
        return lambda x: 0.5

    def norm(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0.5
        val = (float(x) - vmin) / (vmax - vmin)
        if clip_zero and val < 0:
            return 0.0
        return max(0.0, min(1.0, val))

    return norm


def _concept_scores_from_metrics(row: Dict, normalizers: Dict) -> None:
    """Mutates row with concept_* keys; timing includes this work."""
    bbox_dist = row.get("bbox_center_activation_distance", np.nan)
    spread = row.get("activation_spread", np.nan)
    ring_ratio = row.get("ring_energy_ratio", np.nan)

    nd = normalizers["bbox_distance"]
    ns = normalizers["spread"]
    nr = normalizers["ring_ratio"]

    if not (np.isnan(bbox_dist) or np.isnan(spread) or np.isnan(ring_ratio)):
        row["concept_Focused"] = max(
            0.0,
            min(
                1.0,
                (1 - nd(bbox_dist) - ns(spread)) / 2 + 0.3 * nr(ring_ratio),
            ),
        )
    else:
        row["concept_Focused"] = np.nan

    if not np.isnan(spread):
        row["concept_Diffused"] = ns(spread)
    else:
        row["concept_Diffused"] = np.nan

    if not (np.isnan(bbox_dist) or np.isnan(ring_ratio)):
        row["concept_Background"] = (nd(bbox_dist) + (1 - nr(ring_ratio))) / 2
    else:
        row["concept_Background"] = np.nan

    row["concept_Collapse"] = 1.0 if (not np.isnan(spread) and float(spread) < 1e-3) else 0.0


def _load_fastcav_normalizers(root: Path) -> Optional[Dict]:
    p = root / "results" / "cam_records.csv"
    if not p.exists() or p.stat().st_size == 0:
        return None
    cam_df = pd.read_csv(p)
    need = ["bbox_center_activation_distance", "activation_spread", "ring_energy_ratio"]
    if not all(c in cam_df.columns for c in need):
        return None
    metric_rows = cam_df[need].dropna().to_dict("records")
    return {
        "bbox_distance": _normalize_metric_series(cam_df["bbox_center_activation_distance"].dropna()),
        "spread": _normalize_metric_series(cam_df["activation_spread"].dropna()),
        "ring_ratio": _normalize_metric_series(cam_df["ring_energy_ratio"].dropna()),
        "metric_rows": metric_rows,
    }


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_explainer(
    cls, torch_model: torch.nn.Module, layer_name: str, device_str: str
) -> YOLOGradCAM:
    ex = cls(torch_model, target_layer_name=layer_name, device=device_str)
    torch_model.train()
    return ex


def measure_detector_forward(
    torch_model: torch.nn.Module,
    device: torch.device,
    img_rgb: np.ndarray,
    *,
    sync_fn: Callable[[], None],
) -> float:
    """Single timed pass: preprocess + forward only (no backward)."""
    x, _ = YOLOGradCAM._preprocess_np_image(img_rgb, target_size=384)
    x = x.to(device, dtype=torch.float32)
    sync_fn()
    t0 = time.perf_counter()
    with torch.set_grad_enabled(False):
        torch_model(x)
    sync_fn()
    return time.perf_counter() - t0


def measure_xai_generate_cam(
    explainer: YOLOGradCAM,
    img_rgb: np.ndarray,
    yolo_bbox: Tuple[float, float, float, float],
    class_id: int,
    *,
    sync_fn: Callable[[], None],
) -> Tuple[float, np.ndarray, Dict]:
    """Timed: full generate_cam (incl. forward, backward, CAM numpy)."""
    sync_fn()
    t0 = time.perf_counter()
    cam_np, letterbox_meta = explainer.generate_cam(img_rgb, yolo_bbox, int(class_id))
    sync_fn()
    return time.perf_counter() - t0, cam_np, letterbox_meta


def measure_fastcav_pipeline(
    metric_row: Dict,
    normalizers: Dict,
    *,
    sync_fn: Callable[[], None],
) -> float:
    """FastCAV concept-score post-process only (single-sample)."""
    sync_fn()
    t0 = time.perf_counter()
    row = dict(metric_row)
    _concept_scores_from_metrics(row, normalizers)
    sync_fn()
    return time.perf_counter() - t0


def collect_samples(
    root: Path, max_samples: int
) -> List[Tuple[Path, Tuple[float, float, float, float], int, Tuple[float, float, float, float]]]:
    """Paths and bboxes in deterministic CSV order (no shuffle)."""
    csv_path = root / "results" / "detection_records.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Need {csv_path}")
    det = pd.read_csv(csv_path)
    if "severity" in det.columns:
        det = det[det["severity"] == 0]
    if "corruption" in det.columns:
        det = det[det["corruption"] == "fog"]
    det = det.drop_duplicates(subset=["image_id"], keep="first")
    det = det.head(max_samples)
    out = []
    for _, row in det.iterrows():
        rel = row.get("corrupted_image_path") or row.get("image_path")
        if pd.isna(rel):
            continue
        path = root / str(rel)
        if not path.is_file():
            alt = root / "datasets" / str(rel)
            if alt.is_file():
                path = alt
            else:
                continue
        gx1, gy1, gx2, gy2 = (
            float(row["gt_x1"]),
            float(row["gt_y1"]),
            float(row["gt_x2"]),
            float(row["gt_y2"]),
        )
        cid = int(row.get("class_id", row.get("gt_class_id", 0)))
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
        yolo_bbox = visdrone_to_yolo_bbox((gx1, gy1, gx2 - gx1, gy2 - gy1), w, h)
        out.append((path, yolo_bbox, cid, (gx1, gy1, gx2, gy2)))
    return out


def bench_method(
    name: str,
    fn: Callable[[], float],
    n_warmup: int,
    n_measure: int,
    sync_fn: Callable[[], None],
    deadline_ms: float,
) -> Tuple[float, float, float, float, float]:
    """Returns mean_ms, std_ms, max_ms, fps, within_deadline_pct."""
    for _ in range(n_warmup):
        fn()
        sync_fn()
    times: List[float] = []
    for _ in tqdm(range(n_measure), desc=name, leave=False):
        times.append(fn())
        sync_fn()
    arr = np.array(times, dtype=np.float64) * 1000.0
    mean_ms = float(arr.mean())
    std_ms = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    max_ms = float(arr.max())
    fps = float(1000.0 / mean_ms) if mean_ms > 1e-9 else 0.0
    within_pct = float((arr <= float(deadline_ms)).mean() * 100.0) if len(arr) else 0.0
    return mean_ms, std_ms, max_ms, fps, within_pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200, help="Timed iterations (after warmup)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (discarded)")
    parser.add_argument(
        "--deadline-ms",
        type=float,
        default=33.0,
        help="Real-time budget (ms): pct of iterations with latency <= this value (e.g. 33 ms ~ 30 FPS).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    config = load_yaml(root / "configs" / "experiment.yaml")
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    samples = collect_samples(root, args.samples)
    if len(samples) == 0:
        print("[ERROR] No valid image samples. Check detection_records.csv paths.")
        sys.exit(1)
    n_measure = min(args.samples, len(samples))
    samples = samples[:n_measure]

    model_name = "yolo_generic"
    mc = config["models"][model_name]
    model_path = mc["checkpoint"] if mc.get("fine_tuned") and Path(mc["checkpoint"]).exists() else mc["pretrained"]
    layer_name = config["gradcam"]["layers"]["primary"]["name"]

    yolo = YOLO(model_path, task="detect")
    torch_model = yolo.model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_model = torch_model.to(device)
    torch_model.train()
    device_str = str(device)

    # Load images once (exclude I/O from timed region)
    loaded: List[Tuple[np.ndarray, Tuple[float, float, float, float], int, Tuple[float, float, float, float]]] = []
    for path, yolo_bbox, cid, bbox_xyxy in samples:
        with Image.open(path) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
        loaded.append((arr, yolo_bbox, cid, bbox_xyxy))

    idx_cycle = {"i": 0}

    def next_sample():
        j = idx_cycle["i"] % len(loaded)
        idx_cycle["i"] += 1
        return loaded[j]

    fastcav_norm = _load_fastcav_normalizers(root)
    if fastcav_norm is None:
        print(
            "[WARN] No cam_records.csv normalizers — FastCAV uses fallback 0.5 normalizers "
            "(run 05 first for faithful FastCAV post-process)."
        )
        fastcav_norm = {
            "bbox_distance": lambda x: 0.5,
            "spread": lambda x: 0.5,
            "ring_ratio": lambda x: 0.5,
            "metric_rows": [
                {
                    "bbox_center_activation_distance": 0.0,
                    "activation_spread": 0.0,
                    "ring_energy_ratio": 0.5,
                }
            ],
        }

    rows_out: List[Dict] = []

    # --- 1) Detector only ---
    def run_detector():
        img, _, _, _ = next_sample()
        return measure_detector_forward(torch_model, device, img, sync_fn=_cuda_sync)

    m_mean, m_std, m_max, m_fps, m_within = bench_method(
        "Detector only", run_detector, args.warmup, n_measure, _cuda_sync, args.deadline_ms
    )
    det_mean = m_mean
    rows_out.append(
        {
            "method": "Detector only",
            "mean_time_ms": m_mean,
            "std_time_ms": m_std,
            "max_time_ms": m_max,
            "fps": m_fps,
            "overhead_percent": 0.0,
            "within_deadline_pct": m_within,
        }
    )

    xai_specs: List[Tuple[str, type]] = [
        ("Detector + Grad-CAM", YOLOGradCAM),
        ("Detector + Grad-CAM++", YOLOGradCAMPP),
        ("Detector + LayerCAM", YOLOLayerCAM),
    ]

    for label, cls in xai_specs:
        idx_cycle["i"] = 0
        explainer = _make_explainer(cls, torch_model, layer_name, device_str)

        def make_fn(expl=explainer):
            def _run():
                img, yb, cid, _ = next_sample()
                dt, _, _ = measure_xai_generate_cam(expl, img, yb, cid, sync_fn=_cuda_sync)
                return dt

            return _run

        m_mean, m_std, m_max, m_fps, m_within = bench_method(
            label, make_fn(), args.warmup, n_measure, _cuda_sync, args.deadline_ms
        )
        oh = (m_mean - det_mean) / det_mean * 100.0 if det_mean > 1e-9 else float("nan")
        rows_out.append(
            {
                "method": label,
                "mean_time_ms": m_mean,
                "std_time_ms": m_std,
                "max_time_ms": m_max,
                "fps": m_fps,
                "overhead_percent": oh,
                "within_deadline_pct": m_within,
            }
        )
        explainer.close()
        del explainer

    # --- FastCAV (concept-score post-process only; CAM generation excluded) ---
    idx_cycle["i"] = 0
    metric_rows = fastcav_norm.get("metric_rows", [])
    if not metric_rows:
        metric_rows = [
            {
                "bbox_center_activation_distance": 0.0,
                "activation_spread": 0.0,
                "ring_energy_ratio": 0.5,
            }
        ]
    fastcav_idx = {"i": 0}

    def run_fastcav():
        j = fastcav_idx["i"] % len(metric_rows)
        fastcav_idx["i"] += 1
        return measure_fastcav_pipeline(
            metric_rows[j],
            fastcav_norm,
            sync_fn=_cuda_sync,
        )

    m_mean, m_std, m_max, m_fps, m_within = bench_method(
        "Detector + FastCAV", run_fastcav, args.warmup, n_measure, _cuda_sync, args.deadline_ms
    )
    oh = (m_mean - det_mean) / det_mean * 100.0 if det_mean > 1e-9 else float("nan")
    rows_out.append(
        {
            "method": "Detector + FastCAV",
            "mean_time_ms": m_mean,
            "std_time_ms": m_std,
            "max_time_ms": m_max,
            "fps": m_fps,
            "overhead_percent": oh,
            "within_deadline_pct": m_within,
        }
    )

    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "runtime_benchmark.csv"
    df = pd.DataFrame(rows_out)
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Wrote {out_csv}")

    summary = {
        "deadline_ms": float(args.deadline_ms),
        "n_timed_iterations": int(n_measure),
        "warmup_iterations": int(args.warmup),
        "latency_note": (
            "End-to-end wall time per iteration for the listed pipeline (batch=1, one bbox): "
            "detector = letterbox+forward only; Grad-CAM/Grad-CAM++/LayerCAM = full generate_cam; "
            "FastCAV = concept-score post-process only (CAM generation excluded). "
            "Interpret as time until explanation signal is available for that path."
        ),
        "methods": rows_out,
    }
    out_json = out_dir / "runtime_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {out_json}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
