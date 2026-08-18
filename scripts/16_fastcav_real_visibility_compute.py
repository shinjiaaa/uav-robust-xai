"""Real FastCAV with `object_visible` concept (option E).

Per corruption c, learn a CAV that distinguishes:
    Positive: clean (L0) tiny-object regions   ↔  "visible / clearly perceivable"
    Negative: strong-corruption (L4) tiny-object regions of the SAME object  ↔  "obscured"

Score per object × (corruption, severity) = projection of the object's local activation
(at model.8.cv2.conv, restricted to GT bbox after letterbox mapping) onto the corresponding
visibility CAV. Score is normalized so 1 ≈ clean visibility, 0 ≈ corrupted obscurity.

Outputs (all use prefix `fastcav_real_visibility_*`):
    results/fastcav_real_visibility_concept_scores.csv
    results/fastcav_real_visibility_cav_metadata.json
    results/fastcav_real_visibility_probe_object_uids.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.io import load_yaml
from src.xai.fastcav_real import FastCAVRealExtractor, learn_cav, project_score, CAV


CONCEPTS = ["visible_under_fog", "visible_under_lowlight", "visible_under_motion_blur"]
CONCEPT_TO_CORRUPTION = {
    "visible_under_fog": "fog",
    "visible_under_lowlight": "lowlight",
    "visible_under_motion_blur": "motion_blur",
}
LAYER_NAME = "model.8.cv2.conv"
TARGET_SIZE = 384


def letterbox_preprocess(img_bgr: np.ndarray) -> Tuple[torch.Tensor, Dict]:
    h, w = img_bgr.shape[:2]
    scale = min(TARGET_SIZE / w, TARGET_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((TARGET_SIZE, TARGET_SIZE, 3), 114, dtype=np.uint8)
    pad_top = (TARGET_SIZE - new_h) // 2
    pad_left = (TARGET_SIZE - new_w) // 2
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    x = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    meta = {"scale": scale, "pad_left": pad_left, "pad_top": pad_top, "target_size": TARGET_SIZE}
    return x, meta


def map_bbox_to_feature(
    gt_xyxy_orig: Tuple[float, float, float, float],
    meta: Dict,
    feat_h: int,
    feat_w: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Map GT bbox in original image coords -> feature map indices (fy1, fx1, fy2, fx2)."""
    x1o, y1o, x2o, y2o = gt_xyxy_orig
    if x2o <= x1o or y2o <= y1o:
        return None
    scale = meta["scale"]
    pad_left = meta["pad_left"]
    pad_top = meta["pad_top"]
    # to letterbox coords
    x1 = x1o * scale + pad_left
    y1 = y1o * scale + pad_top
    x2 = x2o * scale + pad_left
    y2 = y2o * scale + pad_top
    # to feature coords (each pixel covers TARGET_SIZE/feat_w in input)
    sx = TARGET_SIZE / feat_w
    sy = TARGET_SIZE / feat_h
    fx1 = max(0, int(np.floor(x1 / sx)))
    fy1 = max(0, int(np.floor(y1 / sy)))
    fx2 = min(feat_w, int(np.ceil(x2 / sx)))
    fy2 = min(feat_h, int(np.ceil(y2 / sy)))
    if fx2 <= fx1:
        fx2 = min(feat_w, fx1 + 1)
    if fy2 <= fy1:
        fy2 = min(feat_h, fy1 + 1)
    return fy1, fx1, fy2, fx2


def pool_bbox_activation(activation_chw: torch.Tensor, fy1: int, fx1: int, fy2: int, fx2: int) -> np.ndarray:
    """Mean-pool an activation patch within feature-space bbox. Returns (D,) numpy array."""
    patch = activation_chw[:, fy1:fy2, fx1:fx2]
    if patch.numel() == 0:
        return np.zeros((activation_chw.shape[0],), dtype=np.float32)
    pooled = patch.mean(dim=(1, 2))
    return pooled.cpu().numpy().astype(np.float32)


def resolve_image_path(root: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel.strip():
        return Path("")
    p = Path(rel)
    if p.is_absolute() and p.exists():
        return p
    cand = (root / rel).resolve()
    if cand.exists():
        return cand
    cand2 = (root / rel.replace("\\", "/")).resolve()
    return cand2


def select_probe_object_uids(det_df: pd.DataFrame, *, n_probe: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    # Need objects that have ALL severities (0 and 4) for ALL corruptions
    grouped = det_df.groupby("object_uid")
    eligible = []
    for ouid, g in grouped:
        sev_per_corr = g.groupby("corruption")["severity"].apply(set).to_dict()
        ok = True
        for corr in ("fog", "lowlight", "motion_blur"):
            s = sev_per_corr.get(corr, set())
            if 0 not in s or 4 not in s:
                ok = False
                break
        if ok:
            eligible.append(str(ouid))
    eligible = sorted(eligible)
    if len(eligible) <= n_probe:
        return eligible
    idx = rng.choice(len(eligible), size=n_probe, replace=False)
    return [eligible[i] for i in sorted(idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-probe-objects", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--limit-objects", type=int, default=None, help="Smoke test: cap eval object count")
    args = ap.parse_args()

    cfg = load_yaml(ROOT / "configs" / "experiment.yaml")
    results_root = ROOT / cfg.get("results", {}).get("root", "results")

    det_df = pd.read_csv(results_root / "detection_records.csv")
    det_df["severity"] = det_df["severity"].astype(int)
    print(f"[load] detection_records: {len(det_df)} rows, {det_df['object_uid'].nunique()} unique objects")

    probe_object_uids = select_probe_object_uids(det_df, n_probe=args.n_probe_objects, seed=args.seed)
    probe_set = set(probe_object_uids)
    print(f"[probe] reserved {len(probe_object_uids)} object_uids for CAV training")
    (results_root / "fastcav_real_visibility_probe_object_uids.txt").write_text(
        "\n".join(probe_object_uids) + "\n", encoding="utf-8"
    )

    # ---- Load YOLO ----
    from ultralytics import YOLO
    weights = cfg.get("models", {}).get("yolo_generic", {}).get("pretrained", "yolov8s.pt")
    yolo = YOLO(weights, task="detect")
    extractor = FastCAVRealExtractor(yolo.model, LAYER_NAME, device=args.device)
    print(f"[load] hooked layer: {LAYER_NAME}, device: {extractor.device}")

    # Determine feature spatial size
    with torch.no_grad():
        dummy = torch.zeros(1, 3, TARGET_SIZE, TARGET_SIZE, device=extractor.device)
        extractor._captured = None
        _ = yolo.model(dummy)
        feat_shape = tuple(extractor._captured.shape)  # (1, C, H, W)
    feat_C, feat_H, feat_W = feat_shape[1], feat_shape[2], feat_shape[3]
    print(f"[load] feature shape: C={feat_C}, H={feat_H}, W={feat_W}")

    # ---- Helper: forward image, return activation tensor (C,H,W) ----
    @torch.no_grad()
    def forward_capture(img_bgr) -> Tuple[torch.Tensor, Dict]:
        x, meta = letterbox_preprocess(img_bgr)
        x = x.to(extractor.device)
        extractor._captured = None
        _ = yolo.model(x)
        if extractor._captured is None:
            raise RuntimeError("No activation captured")
        return extractor._captured[0], meta  # (C, H, W)

    def get_object_local_activation(
        image_path: Path, gt_xyxy: Tuple[float, float, float, float]
    ) -> Optional[np.ndarray]:
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        feat, meta = forward_capture(img_bgr)
        bbox_feat = map_bbox_to_feature(gt_xyxy, meta, feat.shape[1], feat.shape[2])
        if bbox_feat is None:
            return None
        return pool_bbox_activation(feat, *bbox_feat)

    # ---- Build cache: for each unique image_id, capture activation once and reuse for all objects ----
    # But objects span multiple images... no, each object_uid has 1 image_id. Each (image, corruption, severity) is unique.
    # So per (image_id, corruption, severity): one forward pass; all objects on that image use same feature map.
    image_keys = det_df.drop_duplicates(subset=["image_id", "corruption", "severity"]).copy()
    print(f"[plan] unique (image, corruption, severity) keys: {len(image_keys)}")

    # ---- Step A: collect probe activations per concept ----
    probe_pos_rows: Dict[str, List[np.ndarray]] = {c: [] for c in CONCEPTS}
    probe_neg_rows: Dict[str, List[np.ndarray]] = {c: [] for c in CONCEPTS}

    probe_det = det_df[det_df["object_uid"].isin(probe_set)].copy()
    print(f"[probe] probe rows: {len(probe_det)}")

    # We need bbox per row. detection_records has gt_x1,gt_y1,gt_x2,gt_y2.
    bbox_cols = ["gt_x1", "gt_y1", "gt_x2", "gt_y2"]
    if not all(c in probe_det.columns for c in bbox_cols):
        raise RuntimeError(f"Missing bbox columns in detection_records: need {bbox_cols}")

    # group by (image_id, corruption, severity) for batched forward
    probe_grouped = probe_det.groupby(["image_id", "corruption", "severity"])
    print(f"[probe] image-keys to forward: {probe_grouped.ngroups}")
    for (iid, corr, sev), grp in probe_grouped:
        rel = grp["corrupted_image_path"].iloc[0] if pd.notna(grp["corrupted_image_path"].iloc[0]) else grp["image_path"].iloc[0]
        p = resolve_image_path(ROOT, rel)
        if not p.exists():
            continue
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        feat, meta = forward_capture(img_bgr)
        for _, r in grp.iterrows():
            xyxy = (float(r["gt_x1"]), float(r["gt_y1"]), float(r["gt_x2"]), float(r["gt_y2"]))
            bbox_feat = map_bbox_to_feature(xyxy, meta, feat.shape[1], feat.shape[2])
            if bbox_feat is None:
                continue
            pooled = pool_bbox_activation(feat, *bbox_feat)
            for cname, ccorr in CONCEPT_TO_CORRUPTION.items():
                if str(corr) != ccorr:
                    continue
                if int(sev) == 0:
                    probe_pos_rows[cname].append(pooled)
                elif int(sev) == 4:
                    probe_neg_rows[cname].append(pooled)

    cavs: Dict[str, CAV] = {}
    for cname in CONCEPTS:
        pos = np.stack(probe_pos_rows[cname], axis=0) if probe_pos_rows[cname] else np.empty((0, feat_C))
        neg = np.stack(probe_neg_rows[cname], axis=0) if probe_neg_rows[cname] else np.empty((0, feat_C))
        print(f"[cav] {cname}: pos={pos.shape}, neg={neg.shape}")
        if pos.shape[0] < 5 or neg.shape[0] < 5:
            print(f"[cav] WARNING: insufficient probes for {cname}; skipping")
            continue
        cav = learn_cav(pos, neg, concept_name=cname, layer_name=LAYER_NAME)
        cavs[cname] = cav
        print(f"[cav] {cname}: pos_mean={cav.pos_mean_proj:.3f}, neg_mean={cav.neg_mean_proj:.3f}, span={cav.pos_mean_proj - cav.neg_mean_proj:.3f}")

    # Save CAV metadata
    meta_path = results_root / "fastcav_real_visibility_cav_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({c: cavs[c].to_dict() for c in cavs}, f, indent=2)
    print(f"[cav] saved metadata: {meta_path}")

    # ---- Step B: compute scores for non-probe objects ----
    eval_det = det_df[~det_df["object_uid"].isin(probe_set)].copy()
    if args.limit_objects:
        keep_uids = sorted(eval_det["object_uid"].unique())[: int(args.limit_objects)]
        eval_det = eval_det[eval_det["object_uid"].isin(set(keep_uids))]
        print(f"[score] --limit-objects {args.limit_objects} -> {eval_det['object_uid'].nunique()} objects, {len(eval_det)} rows")

    eval_grouped = eval_det.groupby(["image_id", "corruption", "severity"])
    print(f"[score] image-keys to forward: {eval_grouped.ngroups}")
    out_rows = []
    completed = 0
    for (iid, corr, sev), grp in eval_grouped:
        rel = grp["corrupted_image_path"].iloc[0] if pd.notna(grp["corrupted_image_path"].iloc[0]) else grp["image_path"].iloc[0]
        p = resolve_image_path(ROOT, rel)
        if not p.exists():
            continue
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        feat, meta = forward_capture(img_bgr)
        for _, r in grp.iterrows():
            xyxy = (float(r["gt_x1"]), float(r["gt_y1"]), float(r["gt_x2"]), float(r["gt_y2"]))
            bbox_feat = map_bbox_to_feature(xyxy, meta, feat.shape[1], feat.shape[2])
            if bbox_feat is None:
                continue
            pooled = pool_bbox_activation(feat, *bbox_feat)[None, :]
            row = {
                "object_uid": r["object_uid"],
                "image_id": r["image_id"],
                "corruption": corr,
                "severity": int(sev),
                "class_id": int(r["gt_class_id"]) if pd.notna(r.get("gt_class_id")) else None,
            }
            for cname, cav in cavs.items():
                row[f"score_{cname}"] = float(project_score(pooled, cav, normalize=True)[0])
            out_rows.append(row)
        completed += 1
        if completed % 200 == 0:
            print(f"[score]  ... {completed}/{eval_grouped.ngroups} keys done")

    out_df = pd.DataFrame(out_rows)
    out_path = results_root / "fastcav_real_visibility_concept_scores.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[score] saved: {out_path}  ({len(out_df)} rows)")

    extractor.close()


if __name__ == "__main__":
    main()
