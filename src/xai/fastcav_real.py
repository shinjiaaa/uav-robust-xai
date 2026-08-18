"""Real FastCAV (Schmalwasser et al. 2025): closed-form CAV learning + projection-based concept scoring.

This is the *real* FastCAV implementation, separate from the post-hoc heuristic concept-score
computed in scripts/11_fastcav_concept_detection.py. All artifacts produced by this module use
the prefix ``fastcav_real_`` to avoid any confusion with the heuristic pipeline.

Core idea:
    1. Per concept c, collect activations h_pos (from positive probe images) and h_neg (negative).
    2. Closed-form CAV: v_c = (mean(h_pos) - mean(h_neg)) / ||...||  (after centering)
       (Schmalwasser et al. derive a SVD-based closed form; for a binary linear classifier the
        difference-of-means direction is the optimal CAV up to scale, which is what we use here.)
    3. For a query image x, extract h(x) at the same layer, spatially pool, project: s_c(x) = h(x) . v_c.
    4. Score s_c is the concept activation strength for input x relative to the learned direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class CAV:
    """A single concept activation vector + statistics for normalization."""
    concept_name: str
    direction: np.ndarray          # shape (D,) unit vector
    pos_mean_proj: float           # mean projection of positive probes
    neg_mean_proj: float           # mean projection of negative probes
    pos_std_proj: float
    neg_std_proj: float
    layer_name: str
    feature_dim: int
    n_pos: int
    n_neg: int

    def to_dict(self) -> Dict:
        return {
            "concept_name": self.concept_name,
            "direction": self.direction.tolist(),
            "pos_mean_proj": float(self.pos_mean_proj),
            "neg_mean_proj": float(self.neg_mean_proj),
            "pos_std_proj": float(self.pos_std_proj),
            "neg_std_proj": float(self.neg_std_proj),
            "layer_name": self.layer_name,
            "feature_dim": int(self.feature_dim),
            "n_pos": int(self.n_pos),
            "n_neg": int(self.n_neg),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CAV":
        return cls(
            concept_name=d["concept_name"],
            direction=np.asarray(d["direction"], dtype=np.float32),
            pos_mean_proj=float(d["pos_mean_proj"]),
            neg_mean_proj=float(d["neg_mean_proj"]),
            pos_std_proj=float(d["pos_std_proj"]),
            neg_std_proj=float(d["neg_std_proj"]),
            layer_name=d["layer_name"],
            feature_dim=int(d["feature_dim"]),
            n_pos=int(d["n_pos"]),
            n_neg=int(d["n_neg"]),
        )


class FastCAVRealExtractor:
    """Spatial-pooled activation extractor with a forward hook on a chosen Conv layer.

    Activation pooling: global average pooling over (H, W) → vector of shape (B, D).
    """

    def __init__(self, torch_model: nn.Module, layer_name: str, device: Optional[str] = None):
        self.model = torch_model
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.layer_name = layer_name
        self.target_layer = self._get_layer(layer_name)
        self._captured: Optional[torch.Tensor] = None
        self._hook = self.target_layer.register_forward_hook(self._fwd_hook)

    def _get_layer(self, name: str) -> nn.Module:
        tokens = name.split(".")
        cur: nn.Module = self.model
        for t in tokens:
            if t.isdigit():
                cur = cur[int(t)]
            else:
                cur = getattr(cur, t)
        return cur

    def _fwd_hook(self, module, inp, out):
        self._captured = out.detach()

    def __del__(self):
        try:
            self._hook.remove()
        except Exception:
            pass

    def close(self):
        try:
            self._hook.remove()
        except Exception:
            pass

    @torch.no_grad()
    def extract(self, image_chw_normalized: torch.Tensor) -> np.ndarray:
        """image_chw_normalized: (B, 3, H, W) preprocessed tensor on self.device.
        Returns pooled activation array of shape (B, D).
        """
        self._captured = None
        _ = self.model(image_chw_normalized.to(self.device))
        if self._captured is None:
            raise RuntimeError(f"No activation captured at layer {self.layer_name}")
        feat = self._captured  # (B, C, H, W)
        if feat.ndim != 4:
            raise RuntimeError(f"Expected 4D activation, got shape {tuple(feat.shape)}")
        pooled = feat.mean(dim=(2, 3))  # (B, C)
        return pooled.cpu().numpy().astype(np.float32)


def learn_cav(
    pos_activations: np.ndarray,
    neg_activations: np.ndarray,
    *,
    concept_name: str,
    layer_name: str,
) -> CAV:
    """Closed-form CAV: difference of class means, normalized.

    pos_activations: (n_pos, D)
    neg_activations: (n_neg, D)
    """
    if pos_activations.ndim != 2 or neg_activations.ndim != 2:
        raise ValueError("pos/neg activations must be 2D arrays (n, D).")
    if pos_activations.shape[1] != neg_activations.shape[1]:
        raise ValueError("Pos/neg feature dimensions mismatch.")

    mu_pos = pos_activations.mean(axis=0)
    mu_neg = neg_activations.mean(axis=0)
    diff = mu_pos - mu_neg
    norm = float(np.linalg.norm(diff))
    if norm < 1e-12:
        raise RuntimeError(
            f"Concept '{concept_name}': pos/neg means are identical; cannot define a CAV."
        )
    direction = diff / norm  # unit vector
    pos_proj = pos_activations @ direction
    neg_proj = neg_activations @ direction
    return CAV(
        concept_name=concept_name,
        direction=direction.astype(np.float32),
        pos_mean_proj=float(pos_proj.mean()),
        neg_mean_proj=float(neg_proj.mean()),
        pos_std_proj=float(pos_proj.std()),
        neg_std_proj=float(neg_proj.std()),
        layer_name=layer_name,
        feature_dim=int(direction.shape[0]),
        n_pos=int(pos_activations.shape[0]),
        n_neg=int(neg_activations.shape[0]),
    )


def project_score(activations: np.ndarray, cav: CAV, *, normalize: bool = True) -> np.ndarray:
    """Compute concept score per row: optionally z-normalize against (pos_mean, neg_mean) span.

    activations: (N, D)
    Returns: (N,) array of concept scores. Higher = closer to positive concept.
    """
    if activations.ndim != 2:
        raise ValueError("activations must be (N, D).")
    if activations.shape[1] != cav.feature_dim:
        raise ValueError(
            f"feature dim mismatch: got {activations.shape[1]}, CAV expects {cav.feature_dim}"
        )
    raw = activations @ cav.direction  # (N,)
    if not normalize:
        return raw.astype(np.float32)
    span = (cav.pos_mean_proj - cav.neg_mean_proj)
    if abs(span) < 1e-12:
        return raw.astype(np.float32)
    # 0 ≈ neg mean, 1 ≈ pos mean (linear), unbounded outside
    return ((raw - cav.neg_mean_proj) / span).astype(np.float32)
