"""Real-model runtime benchmark: Detector / Detector+Grad-CAM / Detector+Real FastCAV.

Measures all three on actual YOLOv8s with letterboxed input, with proper CUDA sync.
Saves results/runtime_comparison_all.csv.
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
NUM_RUNS = 100
WARMUP = 20


def _sync(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    cfg = load_yaml(ROOT / "configs" / "experiment.yaml")
    results_root = ROOT / cfg.get("results", {}).get("root", "results")

    from ultralytics import YOLO
    weights = cfg.get("models", {}).get("yolo_generic", {}).get("pretrained", "yolov8s.pt")
    yolo = YOLO(weights, task="detect")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo.model.to(device).eval()
    print(f"[load] device={device}, weights={weights}")

    dummy = torch.randn(1, 3, TARGET_SIZE, TARGET_SIZE, device=device)

    # ---- 1. Detector only ----
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

    # ---- 2. Detector + Grad-CAM (forward + backward + activation-grad pool) ----
    print("\n[2] Detector + Grad-CAM (forward + backward + CAM compute)")
    # Attach a forward hook just to capture the target-layer activation, plus a backward hook to capture gradients.
    captured_act = {"a": None}
    captured_grad = {"g": None}

    def _fwd_h(module, inp, out):
        captured_act["a"] = out
        out.retain_grad()

    def _bwd_h(module, grad_in, grad_out):
        captured_grad["g"] = grad_out[0].detach()

    # locate target layer
    layer = yolo.model
    for tok in LAYER_NAME.split("."):
        layer = layer[int(tok)] if tok.isdigit() else getattr(layer, tok)
    h_fwd = layer.register_forward_hook(_fwd_h)
    h_bwd = layer.register_full_backward_hook(_bwd_h)
    yolo.model.train()  # required so retain_grad works through forward

    def _gradcam_step():
        captured_act["a"] = None
        captured_grad["g"] = None
        dummy_g = dummy.clone().requires_grad_(True)
        out = yolo.model(dummy_g)
        if isinstance(out, (list, tuple)):
            target = sum(o.sum() for o in out if isinstance(o, torch.Tensor))
        else:
            target = out.sum()
        target.backward()
        # Compute Grad-CAM: weights = mean(grad over H,W), cam = relu(sum_c (w_c * act_c))
        act = captured_act["a"]
        grad = captured_grad["g"]
        if act is not None and grad is not None:
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * act.detach()).sum(dim=1))
        yolo.model.zero_grad(set_to_none=True)

    for _ in range(WARMUP):
        _gradcam_step()
    _sync(device)
    times_gc = []
    for _ in range(NUM_RUNS):
        _sync(device)
        t0 = time.perf_counter()
        _gradcam_step()
        _sync(device)
        times_gc.append((time.perf_counter() - t0) * 1000)
    times_gc = np.array(times_gc)
    print(f"  mean={times_gc.mean():.3f} ms  p95={np.percentile(times_gc, 95):.3f}  fps={1000/times_gc.mean():.1f}")

    h_fwd.remove(); h_bwd.remove()
    yolo.model.eval()

    # ---- 3. Detector + Real FastCAV ----
    print("\n[3] Detector + Real FastCAV (forward + bbox pool + projection)")
    meta_path = results_root / "fastcav_real_visibility_cav_metadata.json"
    if not meta_path.exists():
        meta_path = results_root / "fastcav_real_cav_metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        cav_meta = json.load(f)
    cavs = [CAV.from_dict(cav_meta[k]) for k in sorted(cav_meta.keys())]
    direction_matrix = np.stack([c.direction for c in cavs], axis=0)
    cav_torch = torch.from_numpy(direction_matrix.astype(np.float32)).to(device)
    pos_means = torch.tensor([c.pos_mean_proj for c in cavs], device=device, dtype=torch.float32)
    neg_means = torch.tensor([c.neg_mean_proj for c in cavs], device=device, dtype=torch.float32)
    spans = pos_means - neg_means

    extractor = FastCAVRealExtractor(yolo.model, LAYER_NAME, device=device)

    for _ in range(WARMUP):
        extractor._captured = None
        with torch.no_grad():
            _ = yolo.model(dummy)
        feat = extractor._captured
        pooled = feat.mean(dim=(2, 3))
        scores = (pooled @ cav_torch.T - neg_means.unsqueeze(0)) / spans.unsqueeze(0)
        _ = scores.cpu().numpy()
    _sync(device)
    times_fc = []
    for _ in range(NUM_RUNS):
        _sync(device)
        t0 = time.perf_counter()
        extractor._captured = None
        with torch.no_grad():
            _ = yolo.model(dummy)
        feat = extractor._captured
        pooled = feat.mean(dim=(2, 3))
        scores = (pooled @ cav_torch.T - neg_means.unsqueeze(0)) / spans.unsqueeze(0)
        _ = scores.cpu().numpy()
        _sync(device)
        times_fc.append((time.perf_counter() - t0) * 1000)
    times_fc = np.array(times_fc)
    print(f"  mean={times_fc.mean():.3f} ms  p95={np.percentile(times_fc, 95):.3f}  fps={1000/times_fc.mean():.1f}")
    extractor.close()

    # ---- summary ----
    base_mean = float(times_det.mean())
    rows = [
        {
            "method": "Detector only",
            "mean_ms": base_mean,
            "p95_ms": float(np.percentile(times_det, 95)),
            "fps": 1000 / base_mean,
            "overhead_ms": 0.0,
            "overhead_pct": 0.0,
        },
        {
            "method": "Detector + Grad-CAM",
            "mean_ms": float(times_gc.mean()),
            "p95_ms": float(np.percentile(times_gc, 95)),
            "fps": 1000 / float(times_gc.mean()),
            "overhead_ms": float(times_gc.mean()) - base_mean,
            "overhead_pct": (float(times_gc.mean()) - base_mean) / base_mean * 100,
        },
        {
            "method": "Detector + Real FastCAV",
            "mean_ms": float(times_fc.mean()),
            "p95_ms": float(np.percentile(times_fc, 95)),
            "fps": 1000 / float(times_fc.mean()),
            "overhead_ms": float(times_fc.mean()) - base_mean,
            "overhead_pct": (float(times_fc.mean()) - base_mean) / base_mean * 100,
        },
    ]
    df = pd.DataFrame(rows)
    out_path = results_root / "runtime_comparison_all.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[save] {out_path}")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
