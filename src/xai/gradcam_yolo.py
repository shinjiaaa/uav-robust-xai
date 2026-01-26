# src/xai/gradcam_yolo.py
from __future__ import annotations
import re
import numpy as np
import torch
import torch.nn.functional as F
import cv2

class YOLOGradCAM:
    def __init__(self, torch_model: torch.nn.Module, target_layer_name: str, device: str | None = None):
        self.model = torch_model
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Use train() mode to ensure gradient computation (even though we're doing inference)
        # This is necessary because eval() mode may use optimizations that break gradient graph
        self.model.train()

        self.target_layer = self._get_target_layer(target_layer_name)

        self.activations = None
        self.gradients = None

        def fwd_hook(m, inp, out):
            self.activations = out

        def bwd_hook(module, grad_input, grad_output):
            # register_full_backward_hook signature: (module, grad_input, grad_output)
            # grad_output is a tuple, get first element if available
            if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
                self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(fwd_hook)
        # Use register_full_backward_hook (removes FutureWarning and more stable)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def _get_target_layer(self, target_layer_name: str):
        layer = self.model # Already DetectionModel
        tokens = target_layer_name.split(".")
        for tok in tokens:
            m = re.match(r"^([A-Za-z_]\w*)\[(\d+)\]$", tok)
            if m:
                name, idx = m.group(1), int(m.group(2))
                layer = getattr(layer, name)[idx]
                continue
            if tok.isdigit():
                layer = layer[int(tok)]
                continue
            layer = getattr(layer, tok)
        return layer

    @staticmethod
    def _preprocess_np_image(image: np.ndarray, target_size: int = 640) -> torch.Tensor:
        """image: HxWxC uint8 RGB or BGR. Return: (1,3,H,W) float32 0~1
        Resize to target_size with stride alignment (YOLO uses stride=32)
        Uses letterbox padding to match Ultralytics YOLO preprocessing exactly."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected HxWx3 image")
        
        import cv2
        
        # Get original size
        h, w = image.shape[:2]
        
        # Calculate scale to fit target_size while maintaining aspect ratio
        # This matches Ultralytics YOLO preprocessing exactly
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image (maintain aspect ratio) - no stride alignment before resize
        # Stride alignment happens naturally after padding to 640x640 (640 is divisible by 32)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Letterbox padding to target_size (match Ultralytics YOLO preprocessing exactly)
        # Create padded image with target_size x target_size (always 640x640)
        # This ensures consistent input size regardless of original image dimensions
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)  # 114 = gray padding
        # Center the resized image
        top = (target_size - new_h) // 2
        left = (target_size - new_w) // 2
        padded[top:top+new_h, left:left+new_w] = resized
        
        # Final size is always target_size x target_size (640x640), which is divisible by stride=32
        # This ensures consistent feature map sizes across all images
        
        # Convert to tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
        x = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return x

    def generate_cam(self, image: np.ndarray, yolo_bbox, class_id: int):
        """
        image: np.ndarray HxWx3 (PIL->np라면 RGB일 가능성 높음)
        yolo_bbox: (xc, yc, w, h) normalized (현재 네 코드에서 그렇게 들어옴)
        class_id: int
        """
        # 1) grad 활성화 강제 (중요)
        torch.set_grad_enabled(True)

        # DEBUG: Log input image shape before preprocessing
        input_shape = image.shape[:2]
        
        x = self._preprocess_np_image(image, target_size=640).to(self.device)
        # DEBUG: Log preprocessed tensor shape (should ALWAYS be 640x640 after letterbox)
        preprocessed_shape = x.shape[-2:]  # (H, W)
        assert preprocessed_shape == (640, 640), f"Preprocessed shape must be (640, 640), got {preprocessed_shape}. Input shape was {input_shape}"
        
        # CRITICAL: Input must require grad for gradient flow
        x = x.requires_grad_(True)

        # Reset hooks
        self.activations = None
        self.gradients = None
        
        # 2) forward: no_grad / inference_mode 절대 금지
        #    Note: YOLOv8 forward output may be detached, so we use activations from hook instead
        _ = self.model(x)

        # 3) activations hook에서 캡처한 feature map을 사용 (그래프가 살아있음)
        if self.activations is None:
            raise RuntimeError("Activations not captured. Check target_layer_name.")
        
        # activations는 (1, C, H, W) 형태
        # activations는 그래프에 연결되어 있으므로 backward 가능
        activations = self.activations
        
        # Ensure activations have gradient
        if not activations.requires_grad:
            # If activations don't require grad, the computation graph is broken
            # This usually means the model forward uses torch.no_grad() or detaches
            raise RuntimeError(f"Activations do not require grad. "
                             f"This usually means model forward detaches outputs. "
                             f"activations.requires_grad={activations.requires_grad}, "
                             f"x.requires_grad={x.requires_grad}")
        
        # Use sum of activations (preserves gradient, more stable than max)
        target = activations.sum()

        # ✅ 여기서 target은 반드시 grad_fn이 있어야 함
        if (not target.requires_grad) or (target.grad_fn is None):
            raise RuntimeError("Target score has no grad_fn (graph is detached). "
                               "Check for torch.no_grad/inference_mode or detached outputs.")

        # 4) backward로 gradients 확보
        self.model.zero_grad(set_to_none=True)
        try:
            target.backward(retain_graph=False)
        except RuntimeError as e:
            if "element 0 of tensors does not require grad" in str(e):
                raise RuntimeError(f"Backward failed: {e}. "
                                 f"target.requires_grad={target.requires_grad}, "
                                 f"target.grad_fn={target.grad_fn}, "
                                 f"x.requires_grad={x.requires_grad}") from e
            raise

        # 5) CAM 계산
        if self.activations is None:
            raise RuntimeError("Activations not captured by forward hook. Check target_layer_name.")
        if self.gradients is None:
            raise RuntimeError(f"Gradients not captured by backward hook. "
                             f"activations captured: {self.activations is not None}, "
                             f"target.requires_grad={target.requires_grad}, "
                             f"target.grad_fn={target.grad_fn}. "
                             f"This usually means backward() was not called or hook was not triggered.")

        # activations/gradients: (1,C,h,w)
        w = self.gradients.mean(dim=(2, 3), keepdim=True)      # (1,C,1,1)
        cam = (w * self.activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy()
