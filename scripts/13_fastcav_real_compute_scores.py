"""Real FastCAV: build probes, learn CAVs, compute per-image concept scores.

Pipeline:
    1. Reserve 50 random image_ids as probe set (seed=42), kept disjoint from evaluation pool.
    2. Per concept c in {fog_present, lowlight_present, motion_blur_present}:
       - Positive = probe images at corruption=c, severity=4
       - Negative = probe images at corruption=c, severity=0 (clean)
       - Extract pooled activations at model.8.cv2.conv
       - Learn CAV via difference-of-means (closed form)
    3. For every (image, corruption, severity) NOT in probe set:
       - Extract pooled activation
       - Project onto each CAV → 3 scores per image-condition
    4. Save: results/fastcav_real_concept_scores.csv
            results/fastcav_real_cav_metadata.json
            results/fastcav_real_probe_image_ids.txt  (held-out probe set)

Outputs use the prefix `fastcav_real_*` everywhere; no overlap with the heuristic
concept score artifacts (`fastcav_concept_scores.csv` etc.) produced by script 11.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.io import load_yaml
from src.xai.fastcav_real import FastCAVRealExtractor, learn_cav, project_score, CAV


CONCEPTS = ["fog_present", "lowlight_present", "motion_blur_present"]
CONCEPT_TO_CORRUPTION = {
    "fog_present": "fog",
    "lowlight_present": "lowlight",
    "motion_blur_present": "motion_blur",
}
LAYER_NAME = "model.8.cv2.conv"
TARGET_SIZE = 384


def preprocess_image(path: Path) -> torch.Tensor:
    """Load image and apply YOLOv8 letterbox preprocessing identical to gradcam_yolo._preprocess_np_image."""
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    h, w = img_bgr.shape[:2]
    scale = min(TARGET_SIZE / w, TARGET_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((TARGET_SIZE, TARGET_SIZE, 3), 114, dtype=np.uint8)
    top = (TARGET_SIZE - new_h) // 2
    left = (TARGET_SIZE - new_w) // 2
    padded[top:top + new_h, left:left + new_w] = resized
    x = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return x


def resolve_image_path(root: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute() and p.exists():
        return p
    cand = (root / rel).resolve()
    if cand.exists():
        return cand
    cand2 = (root / rel.replace("\\", "/")).resolve()
    return cand2


def select_probe_image_ids(det_df: pd.DataFrame, *, n_probe: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    all_ids = sorted(det_df["image_id"].astype(str).unique().tolist())
    if len(all_ids) <= n_probe:
        return all_ids
    idx = rng.choice(len(all_ids), size=n_probe, replace=False)
    return [all_ids[i] for i in sorted(idx)]


def extract_batch_activations(
    extractor: FastCAVRealExtractor,
    paths: List[Path],
    *,
    batch_size: int = 8,
) -> np.ndarray:
    feats = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        tensors = [preprocess_image(p) for p in batch_paths]
        batch = torch.cat(tensors, dim=0)
        feat = extractor.extract(batch)
        feats.append(feat)
    if not feats:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(feats, axis=0)


def build_image_path_lookup(det_df: pd.DataFrame, root: Path) -> Dict[Tuple[str, str, int], Path]:
    """For each (image_id, corruption, severity) → path to corrupted image (one row per condition)."""
    lookup: Dict[Tuple[str, str, int], Path] = {}
    sub = det_df.drop_duplicates(subset=["image_id", "corruption", "severity"]).copy()
    sub["severity"] = sub["severity"].astype(int)
    for _, r in sub.iterrows():
        key = (str(r["image_id"]), str(r["corruption"]), int(r["severity"]))
        rel = r.get("corrupted_image_path") or r.get("image_path")
        if not isinstance(rel, str) or not rel.strip():
            continue
        p = resolve_image_path(root, rel)
        if p.exists():
            lookup[key] = p
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-probe", type=int, default=50, help="Number of image_ids to hold out for probe building")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--limit-images", type=int, default=None, help="If set, only score this many image_ids (for quick smoke test)")
    args = ap.parse_args()

    cfg = load_yaml(ROOT / "configs" / "experiment.yaml")
    results_root = ROOT / cfg.get("results", {}).get("root", "results")

    det_path = results_root / "detection_records.csv"
    det_df = pd.read_csv(det_path)
    det_df["severity"] = det_df["severity"].astype(int)
    print(f"[step1] detection_records: {len(det_df)} rows, {det_df['image_id'].nunique()} unique images")

    # ---- Step 1: probe selection ----
    probe_ids = select_probe_image_ids(det_df, n_probe=args.n_probe, seed=args.seed)
    print(f"[step1] probe image_ids reserved: {len(probe_ids)}")
    (results_root / "fastcav_real_probe_image_ids.txt").write_text(
        "\n".join(probe_ids) + "\n", encoding="utf-8"
    )

    # ---- Step 2: load YOLO model ----
    from ultralytics import YOLO
    model_cfg = cfg.get("models", {}).get("yolo_generic", {})
    weights = model_cfg.get("pretrained", "yolov8s.pt")
    print(f"[step2] loading YOLO from {weights}")
    yolo = YOLO(weights, task="detect")
    torch_model = yolo.model
    extractor = FastCAVRealExtractor(torch_model, LAYER_NAME, device=args.device)
    print(f"[step2] hooked layer: {LAYER_NAME}, device: {extractor.device}")

    # ---- Step 3: build path lookup ----
    img_lookup = build_image_path_lookup(det_df, ROOT)
    print(f"[step3] resolved paths: {len(img_lookup)} (image, corruption, severity) keys")

    # ---- Step 4: per-concept probe activations + CAV ----
    cavs: Dict[str, CAV] = {}
    for concept in CONCEPTS:
        corr = CONCEPT_TO_CORRUPTION[concept]
        pos_paths, neg_paths = [], []
        for iid in probe_ids:
            kp = (iid, corr, 4)
            kn = (iid, corr, 0)
            if kp in img_lookup and kn in img_lookup:
                pos_paths.append(img_lookup[kp])
                neg_paths.append(img_lookup[kn])
        print(f"[cav] concept={concept}: pos={len(pos_paths)}, neg={len(neg_paths)}")
        if len(pos_paths) < 5 or len(neg_paths) < 5:
            print(f"[cav] WARNING: insufficient probes for {concept}; skipping")
            continue
        feats_pos = extract_batch_activations(extractor, pos_paths, batch_size=args.batch_size)
        feats_neg = extract_batch_activations(extractor, neg_paths, batch_size=args.batch_size)
        cav = learn_cav(feats_pos, feats_neg, concept_name=concept, layer_name=LAYER_NAME)
        cavs[concept] = cav
        print(
            f"[cav] {concept}: dim={cav.feature_dim}, pos_proj_mean={cav.pos_mean_proj:.3f}, "
            f"neg_proj_mean={cav.neg_mean_proj:.3f}, span={cav.pos_mean_proj - cav.neg_mean_proj:.3f}"
        )

    # Save CAV metadata
    meta_path = results_root / "fastcav_real_cav_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({c: cavs[c].to_dict() for c in cavs}, f, indent=2)
    print(f"[cav] saved metadata: {meta_path}")

    # ---- Step 5: compute concept scores for all NON-probe (image, corruption, severity) ----
    eval_keys = [
        k for k in img_lookup.keys() if k[0] not in set(probe_ids)
    ]
    if args.limit_images is not None:
        eval_image_ids = sorted({k[0] for k in eval_keys})[: int(args.limit_images)]
        eval_keys = [k for k in eval_keys if k[0] in set(eval_image_ids)]
        print(f"[score] --limit-images {args.limit_images} → {len(eval_keys)} keys")

    print(f"[score] scoring {len(eval_keys)} (image, corruption, severity) tuples")
    out_rows = []
    batch_paths: List[Path] = []
    batch_keys: List[Tuple[str, str, int]] = []
    BATCH = args.batch_size

    def flush(paths, keys):
        if not paths:
            return
        feats = extract_batch_activations(extractor, paths, batch_size=BATCH)
        for i, key in enumerate(keys):
            row = {
                "image_id": key[0],
                "corruption": key[1],
                "severity": key[2],
            }
            for cname, cav in cavs.items():
                s = float(project_score(feats[i:i + 1], cav, normalize=True)[0])
                row[f"score_{cname}"] = s
            out_rows.append(row)

    for k in eval_keys:
        batch_paths.append(img_lookup[k])
        batch_keys.append(k)
        if len(batch_paths) >= BATCH:
            flush(batch_paths, batch_keys)
            batch_paths, batch_keys = [], []
    flush(batch_paths, batch_keys)

    out_df = pd.DataFrame(out_rows)
    out_path = results_root / "fastcav_real_concept_scores.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[score] saved: {out_path}  ({len(out_df)} rows)")

    extractor.close()


if __name__ == "__main__":
    main()
