"""Debug script to find stable target layers for Grad-CAM."""

from ultralytics import YOLO
import torch

def main():
    """Print YOLOv8s layer structure."""
    model = YOLO("yolov8s.pt")
    torch_model = model.model
    
    print("=" * 60)
    print("YOLOv8s Layer Structure")
    print("=" * 60)
    print()
    
    # Get all named modules
    print("All named modules (last 30 layers):")
    print("-" * 60)
    names = [n for n, _ in torch_model.named_modules()]
    for n in names[-30:]:
        module = dict(torch_model.named_modules())[n]
        # Check if it's a single feature map layer (not concat)
        if hasattr(module, 'out_channels'):
            print(f"  {n}: {type(module).__name__} (out_channels={module.out_channels})")
        else:
            print(f"  {n}: {type(module).__name__}")
    
    print()
    print("=" * 60)
    print("Recommended target layers (single feature map, stable):")
    print("=" * 60)
    
    # Check backbone end and neck (avoid concat layers)
    print("\nChecking layers for stability (avoiding Concat):")
    print("-" * 60)
    
    for idx in [15, 16, 17, 18, 19, 20, 21, 22]:
        try:
            layer = torch_model.model[idx]
            layer_type = type(layer).__name__
            
            # Skip Concat layers (unstable for CAM)
            if layer_type == 'Concat':
                print(f"  model.{idx}: {layer_type} ❌ (UNSTABLE - has concat)")
                continue
            
            # Check if it's a C2f or Conv block (stable single feature map)
            if hasattr(layer, 'cv2'):  # C2f block
                cv2_layer = layer.cv2
                if hasattr(cv2_layer, 'out_channels'):
                    print(f"  model.{idx}: {layer_type} ✅ (C2f block, out_channels={cv2_layer.out_channels})")
                    print(f"    -> Recommended: 'model.{idx}' or 'model.{idx}.cv2'")
                else:
                    print(f"  model.{idx}: {layer_type} ✅ (C2f block)")
            elif hasattr(layer, 'out_channels'):  # Conv layer
                print(f"  model.{idx}: {layer_type} ✅ (Conv, out_channels={layer.out_channels})")
                print(f"    -> Recommended: 'model.{idx}'")
            else:
                print(f"  model.{idx}: {layer_type} ⚠️  (check structure)")
        except Exception as e:
            print(f"  model.{idx}: Error - {e}")
    
    print()
    print("=" * 60)
    print("To test a layer, use: gradcam.target_layer = 'model.XX'")
    print("=" * 60)

if __name__ == "__main__":
    main()
