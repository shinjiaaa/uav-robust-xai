"""Plotting utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import cv2

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def plot_metric_curves(
    data: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: Optional[str] = None,
    ylabel: Optional[str] = None
):
    """Plot metric curves over severity for each corruption and model.
    
    Args:
        data: DataFrame with columns: model, corruption, severity, and metric
        metric: Column name for the metric to plot
        output_path: Path to save the plot
        title: Plot title (default: metric name)
        ylabel: Y-axis label (default: metric name)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    corruptions = data['corruption'].unique()
    
    for idx, corruption in enumerate(corruptions):
        ax = axes[idx]
        corruption_data = data[data['corruption'] == corruption]
        
        for model in corruption_data['model'].unique():
            model_data = corruption_data[corruption_data['model'] == model]
            model_data = model_data.sort_values('severity')
            ax.plot(
                model_data['severity'],
                model_data[metric],
                marker='o',
                label=model,
                linewidth=2,
                markersize=6
            )
        
        ax.set_xlabel('Severity', fontsize=12)
        ax.set_ylabel(ylabel or metric, fontsize=12)
        ax.set_title(f'{corruption.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1, 2, 3, 4])
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_tiny_object_curves(
    data: pd.DataFrame,
    metrics: List[str],
    output_path: Path,
    corruption: str
):
    """Plot tiny object analysis curves.
    
    Args:
        data: DataFrame with severity and metric columns
        metrics: List of metric column names to plot
        output_path: Path to save the plot
        corruption: Corruption type name
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for model in data['model'].unique():
            model_data = data[data['model'] == model].sort_values('severity')
            ax.plot(
                model_data['severity'],
                model_data[metric],
                marker='o',
                label=model,
                linewidth=2,
                markersize=6
            )
        
        ax.set_xlabel('Severity', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{corruption.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1, 2, 3, 4])
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def save_cam_overlay(
    cam: np.ndarray,
    image: np.ndarray,
    letterbox_meta: Optional[Dict],
    output_path: Path,
    alpha: float = 0.5
) -> bool:
    """Save Grad-CAM heatmap overlay on image (DASC prototype용).
    
    Args:
        cam: CAM heatmap (H, W) in letterbox/preprocessed space
        image: Original image (H, W, 3) RGB or BGR
        letterbox_meta: Dict with scale, pad_left, pad_top, target_size
        output_path: Path to save overlay image
        alpha: Blend factor for heatmap (0-1)
    Returns:
        True if saved successfully
    """
    try:
        orig_h, orig_w = image.shape[:2]
        cam_h, cam_w = cam.shape[:2]
        
        if letterbox_meta:
            scale = letterbox_meta.get('scale', 1.0)
            pad_left = int(letterbox_meta.get('pad_left', 0))
            pad_top = int(letterbox_meta.get('pad_top', 0))
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            y1, y2 = pad_top, min(pad_top + new_h, cam_h)
            x1, x2 = pad_left, min(pad_left + new_w, cam_w)
            cam_crop = cam[y1:y2, x1:x2]
            cam_resized = cv2.resize(cam_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            cam_resized = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        if len(image.shape) >= 3 and image.shape[2] == 3:
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            img_rgb = image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            heatmap_rgb = heatmap
            img_rgb = image
        overlay = np.clip(
            alpha * heatmap_rgb.astype(np.float32) + (1 - alpha) * img_rgb.astype(np.float32),
            0, 255
        ).astype(np.uint8)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), overlay_bgr)
        return True
    except Exception:
        return False
