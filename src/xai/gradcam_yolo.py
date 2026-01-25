"""Grad-CAM implementation for YOLO models."""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from PIL import Image
import cv2


class YOLOGradCAM:
    """Grad-CAM for Ultralytics YOLO models."""
    
    def __init__(self, model, target_layer_name: str = "model.model[-2]"):
        """Initialize Grad-CAM.
        
        Args:
            model: Ultralytics YOLO model
            target_layer_name: Name of target layer (e.g., "model.model[-2]")
        """
        self.model = model
        self.target_layer = self._get_target_layer(target_layer_name)
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _get_target_layer(self, layer_name: str):
        """Get target layer from model.
        
        Args:
            layer_name: Layer name or path
            
        Returns:
            Target layer module
        """
        # Parse layer name (e.g., "model.model[-2]")
        parts = layer_name.split('.')
        layer = self.model.model
        
        for part in parts[1:]:  # Skip first 'model'
            if part.startswith('[') and part.endswith(']'):
                # Index access like [-2]
                idx = int(part[1:-1])
                if idx < 0:
                    idx = len(layer) + idx
                layer = layer[idx]
            else:
                layer = getattr(layer, part)
        
        return layer
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        image: np.ndarray,
        target_box: Tuple[float, float, float, float],  # (x_center, y_center, width, height) normalized
        target_class: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap for a target detection.
        
        Args:
            image: Input image (H, W, C) in [0, 255])
            target_box: Target bounding box in normalized coordinates
            target_class: Target class ID
            device: Device to run on
            
        Returns:
            Heatmap (H, W) in [0, 1] range
        """
        # Reset gradients
        self.gradients = None
        self.activations = None
        
        # Preprocess image
        img_tensor = self._preprocess_image(image, device)
        
        # Forward pass
        self.model.model.eval()
        outputs = self.model.model(img_tensor)
        
        # Find the detection corresponding to target_box
        # For simplicity, we'll use the box with highest IoU
        # This is a simplified approach - in practice, you'd track the detection through NMS
        # For now, we'll compute CAM for the feature map and focus on the target region
        
        # Get activations and gradients
        if self.activations is None:
            raise ValueError("Activations not captured. Check target layer.")
        
        # Compute gradients w.r.t. the target box region
        # We'll use a proxy: maximize activation in the target region
        H, W = image.shape[:2]
        x_center, y_center, width, height = target_box
        x1 = int((x_center - width / 2) * W)
        y1 = int((y_center - height / 2) * H)
        x2 = int((x_center + width / 2) * W)
        y2 = int((y_center + height / 2) * H)
        
        # Backward pass - we'll use the detection score as target
        # For simplicity, use the maximum activation in target region
        activations = self.activations[0]  # Remove batch dimension
        target_region = activations[:, y1:y2, x1:x2] if len(activations.shape) == 3 else activations
        
        if target_region.numel() > 0:
            loss = target_region.max()
            loss.backward()
        else:
            # Fallback: use all activations
            loss = activations.max()
            loss.backward()
        
        # Generate CAM
        if self.gradients is None:
            raise ValueError("Gradients not captured.")
        
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input image size
        cam_np = cam.cpu().numpy()
        cam_resized = cv2.resize(cam_np, (W, H))
        
        return cam_resized
    
    def _preprocess_image(self, image: np.ndarray, device: str) -> torch.Tensor:
        """Preprocess image for model input.
        
        Args:
            image: Input image (H, W, C) in [0, 255]
            device: Device to run on
            
        Returns:
            Preprocessed tensor
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (640x640 for YOLO)
        img_resized = cv2.resize(image, (640, 640))
        
        # Normalize to [0, 1] and convert to tensor
        img_tensor = torch.from_numpy(img_resized).float() / 255.0
        
        # Convert to CHW format
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        return img_tensor


def compute_gradcam_metrics(
    cam_severity_0: np.ndarray,
    cam_severity_k: np.ndarray,
    gt_box: Tuple[float, float, float, float],  # normalized
    img_width: int,
    img_height: int,
    top_k_percent: int = 10
) -> Dict[str, float]:
    """Compute stability metrics between two CAM heatmaps.
    
    Args:
        cam_severity_0: CAM heatmap at severity 0
        cam_severity_k: CAM heatmap at severity k
        gt_box: GT bounding box (x_center, y_center, width, height) normalized
        img_width: Image width
        img_height: Image height
        top_k_percent: Top k percent for overlap computation
        
    Returns:
        Dictionary with metrics: energy_in_ratio_0, energy_in_ratio_k, energy_delta,
        topk_overlap, correlation
    """
    # Convert normalized box to pixel coordinates
    x_center, y_center, width, height = gt_box
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    
    # Clamp to image bounds
    x1 = max(0, min(img_width - 1, x1))
    y1 = max(0, min(img_height - 1, y1))
    x2 = max(0, min(img_width - 1, x2))
    y2 = max(0, min(img_height - 1, y2))
    
    # Extract region within GT box
    cam0_region = cam_severity_0[y1:y2, x1:x2]
    camk_region = cam_severity_k[y1:y2, x1:x2]
    
    # Energy metrics
    energy_total_0 = cam_severity_0.sum()
    energy_total_k = cam_severity_k.sum()
    energy_in_0 = cam0_region.sum()
    energy_in_k = camk_region.sum()
    
    energy_in_ratio_0 = energy_in_0 / energy_total_0 if energy_total_0 > 0 else 0.0
    energy_in_ratio_k = energy_in_k / energy_total_k if energy_total_k > 0 else 0.0
    energy_delta = energy_in_ratio_k - energy_in_ratio_0
    
    # Top-k overlap
    # Binarize at top k%
    threshold_0 = np.percentile(cam_severity_0, 100 - top_k_percent)
    threshold_k = np.percentile(cam_severity_k, 100 - top_k_percent)
    
    mask_0 = (cam_severity_0 >= threshold_0).astype(float)
    mask_k = (cam_severity_k >= threshold_k).astype(float)
    
    mask_0_region = mask_0[y1:y2, x1:x2]
    mask_k_region = mask_k[y1:y2, x1:x2]
    
    intersection = (mask_0_region * mask_k_region).sum()
    union = ((mask_0_region + mask_k_region) > 0).sum()
    topk_overlap = intersection / union if union > 0 else 0.0
    
    # Correlation within GT box
    cam0_flat = cam0_region.flatten()
    camk_flat = camk_region.flatten()
    
    if len(cam0_flat) > 1 and cam0_flat.std() > 0 and camk_flat.std() > 0:
        correlation = np.corrcoef(cam0_flat, camk_flat)[0, 1]
    else:
        correlation = 0.0
    
    return {
        'energy_in_ratio_0': float(energy_in_ratio_0),
        'energy_in_ratio_k': float(energy_in_ratio_k),
        'energy_delta': float(energy_delta),
        'topk_overlap': float(topk_overlap),
        'correlation': float(correlation)
    }
