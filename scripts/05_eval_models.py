"""Evaluate models on corrupted datasets."""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.metrics import evaluate_all_models
from src.utils.io import load_yaml
from src.utils.seed import set_seed


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate models on corrupted datasets")
    parser.add_argument(
        "--models",
        type=str,
        default="yolo_generic",
        help="Models to evaluate (comma-separated)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="val",
        help="Dataset splits to evaluate (comma-separated)"
    )
    
    args = parser.parse_args()
    
    config_path = Path("configs/experiment.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_yaml(config_path)
    set_seed(config['seed'])
    
    models = [m.strip() for m in args.models.split(',')]
    splits = [s.strip() for s in args.splits.split(',')]
    
    # Validate models
    for model in models:
        if model not in config['models']:
            print(f"Error: Unknown model '{model}'")
            print(f"Available models: {', '.join(config['models'].keys())}")
            sys.exit(1)
    
    print("=" * 60)
    print("Evaluating Models")
    print("=" * 60)
    print(f"Models: {', '.join(models)}")
    print(f"Splits: {', '.join(splits)}")
    print(f"Corruptions: {', '.join(config['corruptions']['types'])}")
    print(f"Severities: {config['corruptions']['severities']}")
    print()
    
    output_csv = Path(config['results']['metrics_csv'])
    
    try:
        evaluate_all_models(
            config,
            models=models,
            splits=splits,
            corruption_types=config['corruptions']['types'],
            severities=config['corruptions']['severities'],
            output_csv=output_csv
        )
        print("\n✓ Evaluation complete!")
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
