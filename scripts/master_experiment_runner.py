"""
Master Experiment Runner: Grad-CAM vs FastCAV 종합 비교 실험

이 스크립트는 4개의 개별 실험(A, B, C, D)을 순차 실행하고 종합 평가를 제공합니다.

실험 구성:
- 실험 A: 조기 경보성 비교 (early warning capability)
- 실험 B: 실시간성 비교 (runtime performance)
- 실험 C: 안정성 비교 (robustness & stability)
- 실험 D: 해석성 및 최종 평가 (interpretability & final scoring)

입력 데이터:
- results/failure_events.csv (성능 기준)
- results/cam_records.csv (Grad-CAM 지표)
- results/fastcav_concept_scores.csv (FastCAV 개념 점수, 11_fastcav_concept_detection.py 생성)

출력 결과:
- results/exp_A_*.csv (early warning results)
- results/exp_B_*.csv (runtime results)
- results/exp_C_*.csv (stability results)
- results/final_weighted_scores.csv (종합 점수)
- results/final_recommendation.txt (최종 권고안)

사용법:
  python scripts/master_experiment_runner.py

또는 개별 실험만 실행:
  python scripts/11_fastcav_concept_detection.py
  python scripts/exp_A_early_warning_comparison.py
  python scripts/exp_B_runtime_comparison.py
  python scripts/exp_C_stability_robustness.py
  python scripts/exp_D_interpretability.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent

def run_script(script_name, description):
    """Run a script and return success status."""
    script_path = ROOT / "scripts" / script_name
    
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {description}")
    print(f"{'='*80}")
    print(f"Executing: {script_path}")
    
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT),
            capture_output=False,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print(f"[OK] {description} completed successfully")
            return True
        else:
            print(f"[WARN] {description} exited with code {result.returncode}")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"[ERROR] {description} timed out (>1 hour)")
        return False
    except Exception as e:
        print(f"[ERROR] {description} failed: {e}")
        return False

def main():
    print("="*80)
    print("MASTER EXPERIMENT RUNNER: Grad-CAM vs FastCAV Comparison")
    print("="*80)
    print(f"Project root: {ROOT}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define experiment sequence
    experiments = [
        ("11_fastcav_concept_detection.py", "[Step 0] FastCAV Concept Detection"),
        ("exp_A_early_warning_comparison.py", "[Step 1/4] Experiment A: Early Warning Comparison"),
        ("exp_B_runtime_comparison.py", "[Step 2/4] Experiment B: Real-Time Performance"),
        ("exp_C_stability_robustness.py", "[Step 3/4] Experiment C: Stability & Robustness"),
        ("exp_D_interpretability.py", "[Step 4/4] Experiment D: Interpretability & Final Evaluation"),
    ]
    
    results = {}
    
    # Run experiments
    for script, description in experiments:
        success = run_script(script, description)
        results[script] = success
        
        if not success:
            print(f"\n[WARN] Continuing with next experiment despite error in {script}")
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    for script, description in experiments:
        status = "✓ OK" if results[script] else "✗ FAILED"
        print(f"{status} {description}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All experiments completed successfully!")
        print("\nKey output files:")
        print("  - results/exp_A_alignment_comparison.csv (early warning results)")
        print("  - results/exp_B_runtime_summary.csv (real-time performance)")
        print("  - results/exp_C_stability_metrics.csv (robustness)")
        print("  - results/final_weighted_scores.csv (evaluation results)")
        print("  - results/final_recommendation.txt (final decision)")
        
        # Try to read and print final recommendation
        rec_path = ROOT / "results" / "final_recommendation.txt"
        if rec_path.exists():
            print("\n" + "="*80)
            print("FINAL RECOMMENDATION")
            print("="*80)
            with open(rec_path, 'r', encoding='utf-8') as f:
                print(f.read())
    else:
        print("\n✗ Some experiments failed. Check output above for details.")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
