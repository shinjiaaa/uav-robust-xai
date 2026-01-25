"""Train YOLO model on VisDrone (for M2: fine-tuned model)."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.utils.io import load_yaml
from src.utils.seed import set_seed
import torch


def main():
    """Main training function."""
    config_path = Path("configs/experiment.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_yaml(config_path)
    set_seed(config['seed'])
    
    model_config = config['models']['yolo_ft']
    
    print("=" * 60)
    print("Training YOLO on VisDrone")
    print("=" * 60)
    print(f"Architecture: {model_config['architecture']}")
    print(f"Pretrained: {model_config['pretrained']}")
    print(f"Epochs: {model_config['train_epochs']}")
    print()
    
    # Dataset path
    dataset_yaml = Path(config['dataset']['visdrone_yolo_root']) / "visdrone.yaml"
    if not dataset_yaml.exists():
        print(f"Error: Dataset YAML not found at {dataset_yaml}")
        print("Please run scripts/02_prepare_dataset_yolo.py first")
        sys.exit(1)
    
    # Load model
    model = YOLO(model_config['pretrained'])
    
    # Training parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print()
    
    # Train
    try:
        results = model.train(
            data=str(dataset_yaml),
            epochs=model_config['train_epochs'],
            imgsz=model_config['train_imgsz'],
            batch=model_config['train_batch'],
            device=device,
            project="results",
            name="yolo_ft_visdrone",
            exist_ok=True,
            seed=config['seed']
        )
        
        # Save best model to expected location
        best_model_path = Path("results/yolo_ft_visdrone/weights/best.pt")
        output_path = Path(model_config['checkpoint'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if best_model_path.exists():
            import shutil
            shutil.copy2(best_model_path, output_path)
            print(f"\n✓ Model saved to {output_path}")
        else:
            print(f"\n⚠ Best model not found at {best_model_path}")
        
        print("\n✓ Training complete!")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
