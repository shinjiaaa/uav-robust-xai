"""Pilot experiment: test with extreme severities and small sample."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import load_yaml
import subprocess


def main():
    """Run pilot experiment with extreme severities."""
    config_path = Path("configs/experiment.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_yaml(config_path)
    
    # Check if pilot mode is enabled
    experiment_config = config.get('experiment', {})
    if not experiment_config.get('pilot_mode', False):
        print("Warning: Pilot mode is not enabled in config.")
        print("Set experiment.pilot_mode: true in configs/experiment.yaml")
        return
    
    print("=" * 60)
    print("Pilot Experiment: Extreme Severity Test")
    print("=" * 60)
    print()
    print("This will run a small-scale experiment with:")
    print(f"  - Sample size: {experiment_config.get('sample_size', 100)} tiny objects")
    print(f"  - Extreme severity: {config['corruptions']['severities']}")
    print(f"  - One tiny bbox per image: {experiment_config.get('one_per_image', False)}")
    print()
    
    # Run pipeline steps
    steps = [
        {
            'name': '1. Sample tiny objects from single images',
            'script': 'scripts/01_sample_tiny_objects.py',
            'required': True
        },
        {
            'name': '2. Generate corruptions for single images (extreme severity)',
            'script': 'scripts/02_generate_corruption_sequences.py',
            'required': True
        },
        {
            'name': '3. Detect tiny objects',
            'script': 'scripts/03_detect_tiny_objects_timeseries.py',
            'required': True
        },
        {
            'name': '4. Detect failure events',
            'script': 'scripts/04_detect_failure_events.py',
            'required': True
        },
        {
            'name': '5. Grad-CAM failure analysis (with dynamic refinement)',
            'script': 'scripts/05_gradcam_failure_analysis.py',
            'required': True  # Required for complete analysis
        },
        {
            'name': '6. Generate LLM report',
            'script': 'scripts/06_llm_report.py',
            'required': False  # Optional: may fail if API key is not set
        }
    ]
    
    for step in steps:
        print(f"\n{'='*60}")
        print(f"{step['name']}")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(
                [sys.executable, step['script']],
                check=step['required']
            )
            print(f"[OK] {step['name']} completed")
        except subprocess.CalledProcessError as e:
            if step['required']:
                print(f"[ERROR] {step['name']} failed: {e}")
                print("Stopping pilot experiment.")
                return
            else:
                print(f"[SKIP] {step['name']} failed (optional): {e}")
    
    print("\n" + "=" * 60)
    print("[OK] Pilot experiment complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Review results in results/ directory")
    print("2. Analyze failure regions and CAM metrics")
    print("3. Adjust severity levels based on pilot results")
    print("4. Run full experiment with refined severities")


if __name__ == "__main__":
    main()
