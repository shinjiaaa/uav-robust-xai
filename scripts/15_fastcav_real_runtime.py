"""Runtime benchmark for *real* FastCAV (Schmalwasser-style closed-form projection).

Per-frame work measured:
    - forward pass through YOLOv8s up to model.8.cv2.conv (hook captures activation)
    - global average pool over (H, W) → (D=512,)
    - matmul against 3 stacked CAVs of shape (3, D) → 3 concept scores
    - Z-normalize against pre-stored (pos_mean, neg_mean) span (negligible)

This is fundamentally different from scripts/exp_B_runtime_comparison.py's `Detector + FastCAV`
which measures only the heuristic 4 weighted sums on pre-computed CAM metrics. Real FastCAV
needs the model forward, so it cannot be free-of-cost like the heuristic.

Outputs:
    results/fastcav_real_runtime.csv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.io import load_yaml
from src.xai.fastcav_real import FastCAVRealExtractor, CAV


LAYER_NAME = "model.8.cv2.conv"
TARGET_SIZE = 384


def _sync(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    cfg = load_yaml(ROOT / "configs" / "experiment.yaml")
    results_root = ROOT / cfg.get("results", {}).get("root", "results")

    # Load CAVs
    meta_path = results_root / "fastcav_real_cav_metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        cav_meta = json.load(f)
    cavs = [CAV.from_dict(cav_meta[k]) for k in sorted(cav_meta.keys())]
    direction_matrix = np.stack([c.direction for c in cavs], axis=0)  # (n_concepts, D)
    print(f"[load] {len(cavs)} CAVs, dim={direction_matrix.shape[1]}")

    # Load YOLO
    from ultralytics import YOLO
    weights = cfg.get("models", {}).get("yolo_generic", {}).get("pretrained", "yolov8s.pt")
    yolo = YOLO(weights, task="detect")
    extractor = FastCAVRealExtractor(yolo.model, LAYER_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] device={device}")

    # Pre-make a fixed dummy input on device (we measure per-frame inference + projection cost,
    # not per-frame I/O. Real production cost would also include cv2.imread + letterbox.)
    dummy = torch.randn(1, 3, TARGET_SIZE, TARGET_SIZE, device=extractor.device)

    # CAV matrix on device for matmul timing
    cav_torch = torch.from_numpy(direction_matrix.astype(np.float32)).to(extractor.device)
    pos_means = torch.tensor([c.pos_mean_proj for c in cavs], device=extractor.device, dtype=torch.float32)
    neg_means = torch.tensor([c.neg_mean_proj for c in cavs], device=extractor.device, dtype=torch.float32)
    spans = pos_means - neg_means

    NUM_RUNS = 100
    WARMUP = 20

    # ---- Detector-only ----
    print("\n[1] Detector only")
    for _ in range(WARMUP):
        with torch.no_grad():
            _ = yolo.model(dummy)
    _sync(device)
    times_det = []
    for _ in range(NUM_RUNS):
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = yolo.model(dummy)
        _sync(device)
        times_det.append((time.perf_counter() - t0) * 1000)
    times_det = np.array(times_det)
    print(f"  mean={times_det.mean():.3f} ms  p95={np.percentile(times_det, 95):.3f}  fps={1000/times_det.mean():.1f}")

    # ---- Detector + Real FastCAV (forward + pool + projection) ----
    print("\n[2] Detector + Real FastCAV")
    for _ in range(WARMUP):
        extractor._captured = None
        with torch.no_grad():
            _ = yolo.model(dummy)
        feat = extractor._captured
        pooled = feat.mean(dim=(2, 3))
        scores = (pooled @ cav_torch.T - neg_means.unsqueeze(0)) / spans.unsqueeze(0)
        _ = scores.cpu().numpy()
    _sync(device)
    times_real = []
    for _ in range(NUM_RUNS):
        _sync(device)
        t0 = time.perf_counter()
        extractor._captured = None
        with torch.no_grad():
            _ = yolo.model(dummy)
        feat = extractor._captured  # (1, 512, H, W)
        pooled = feat.mean(dim=(2, 3))  # (1, 512)
        scores = (pooled @ cav_torch.T - neg_means.unsqueeze(0)) / spans.unsqueeze(0)  # (1, 3)
        _ = scores.cpu().numpy()
        _sync(device)
        times_real.append((time.perf_counter() - t0) * 1000)
    times_real = np.array(times_real)
    print(f"  mean={times_real.mean():.3f} ms  p95={np.percentile(times_real, 95):.3f}  fps={1000/times_real.mean():.1f}")

    overhead = times_real.mean() - times_det.mean()
    print(f"  overhead = {overhead:.3f} ms ({overhead / times_det.mean() * 100:.2f}%)")

    # ---- Save ----
    out_rows = [
        {
            "method": "Detector only",
            "mean_ms": float(times_det.mean()),
            "std_ms": float(times_det.std()),
            "p95_ms": float(np.percentile(times_det, 95)),
            "fps": float(1000 / times_det.mean()),
            "overhead_ms": 0.0,
            "overhead_pct": 0.0,
        },
        {
            "method": "Detector + Real FastCAV",
            "mean_ms": float(times_real.mean()),
            "std_ms": float(times_real.std()),
            "p95_ms": float(np.percentile(times_real, 95)),
            "fps": float(1000 / times_real.mean()),
            "overhead_ms": float(overhead),
            "overhead_pct": float(overhead / times_det.mean() * 100),
        },
    ]
    out_df = pd.DataFrame(out_rows)
    out_path = results_root / "fastcav_real_runtime.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n[save] {out_path}")
    print(out_df.to_string(index=False))

    extractor.close()


if __name__ == "__main__":
    main()
