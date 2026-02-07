"""Full experiment: standard severities [0, 1, 2, 3, 4] with full sample size."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import load_yaml
import subprocess


def main():
    """Run full experiment with standard severities."""
    config_path = Path("configs/experiment.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_yaml(config_path)
    
    # Check if pilot mode is disabled (full experiment mode)
    experiment_config = config.get('experiment', {})
    if experiment_config.get('pilot_mode', False):
        print("Warning: Pilot mode is still enabled in config.")
        print("Set experiment.pilot_mode: false in configs/experiment.yaml for full experiment")
        return
    
    print("=" * 60)
    print("Full Experiment: Standard Severities [0, 1, 2, 3, 4]")
    print("=" * 60)
    print()
    print("This will run the full experiment with:")
    print(f"  - Sample size: {experiment_config.get('sample_size', 100)} tiny objects")
    print(f"  - Severities: {config['corruptions']['severities']}")
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
            'name': '2. Generate corruptions for single images (severities 0-4)',
            'script': 'scripts/02_generate_corruption_sequences.py',
            'required': True
        },
        {
            'name': '3. Detect tiny objects',
            'script': 'scripts/03_detect_tiny_objects_timeseries.py',
            'required': True
        },
        {
            'name': '3.5. Evaluate dataset-wide mAP (all severities)',
            'script': 'scripts/05_eval_models.py',
            'required': False  # Optional: may take long time
        },
        {
            'name': '4. Detect risk events (Performance Axis → risk_events.csv)',
            'script': 'scripts/04_detect_risk_events.py',
            'required': True,
            'fallback': 'scripts/04_detect_failure_events.py'  # Fallback to legacy if new script fails
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
            # TOP 6 (1): UTF-8 for child to avoid cp949/Unicode errors on Windows
            env = os.environ.copy()
            env.setdefault('PYTHONUTF8', '1')
            env.setdefault('PYTHONIOENCODING', 'utf-8')
            result = subprocess.run(
                [sys.executable, step['script']],
                check=step['required'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env
            )
            print(result.stdout)
            print(f"[OK] {step['name']} completed")
        except subprocess.CalledProcessError as e:
            # Try fallback script if available
            if 'fallback' in step and step['fallback']:
                print(f"[WARN] {step['name']} failed, trying fallback: {step['fallback']}")
                try:
                    result = subprocess.run(
                        [sys.executable, step['fallback']],
                        check=step['required'],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        env=env
                    )
                    print(result.stdout)
                    print(f"[OK] Fallback script completed")
                except subprocess.CalledProcessError as e2:
                    if step['required']:
                        print(f"[ERROR] Both scripts failed:")
                        print(f"  Primary: {e}")
                        print(f"  Fallback: {e2}")
                        print("Stopping full experiment.")
                        return
                    else:
                        print(f"[SKIP] Both scripts failed (optional)")
            elif step['required']:
                print(f"[ERROR] {step['name']} failed: {e}")
                if hasattr(e, 'stderr') and e.stderr:
                    print(f"Error output: {e.stderr}")
                print("Stopping full experiment.")
                return
            else:
                print(f"[SKIP] {step['name']} failed (optional): {e}")
    
    print("\n" + "=" * 60)
    print("[OK] Full experiment complete!")
    print("=" * 60)
    print()
    print("Results saved in results/ directory:")
    print("  - metrics_dataset.csv: Dataset-wide metrics")
    print("  - tiny_records_timeseries.csv: Detection records")
    print("  - failure_events.csv: Failure event analysis")
    print("  - gradcam_metrics_timeseries.csv: CAM distribution metrics")
    print("  - gradcam_errors.csv: CAM error log (for bias analysis)")
    print("  - report.md: LLM-generated report")


if __name__ == "__main__":
    main()
