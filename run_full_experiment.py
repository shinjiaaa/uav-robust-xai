"""Run full experiment pipeline automatically."""

import subprocess
import sys
import time
from pathlib import Path
import json
import pandas as pd

def check_step_complete(step_name, check_func):
    """Check if a step is complete."""
    try:
        return check_func()
    except:
        return False

def wait_for_completion(check_func, max_wait=3600, check_interval=60):
    """Wait for a step to complete."""
    elapsed = 0
    while elapsed < max_wait:
        if check_func():
            return True
        time.sleep(check_interval)
        elapsed += check_interval
        print(f"[대기 중] {elapsed//60}분 경과...")
    return False

def main():
    """Main function."""
    print("=" * 60)
    print("전체 실험 파이프라인 자동 실행")
    print("=" * 60)
    print()
    
    steps = [
        {
            'name': '1. 프레임 클립 추출',
            'script': 'scripts/01_extract_frame_clips.py',
            'check': lambda: Path('results/frame_clips.json').exists()
        },
        {
            'name': '2. 변조 시퀀스 생성',
            'script': 'scripts/02_generate_corruption_sequences.py',
            'check': lambda: len(list(Path('datasets/visdrone_corrupt/sequences').rglob('*.jpg'))) > 10000
        },
        {
            'name': '3. 탐지 수행 (3개 모델)',
            'script': 'scripts/03_detect_tiny_objects_timeseries.py',
            'check': lambda: Path('results/tiny_records_timeseries.csv').exists() and len(pd.read_csv('results/tiny_records_timeseries.csv')) > 0
        },
        {
            'name': '4. 실패 이벤트 감지',
            'script': 'scripts/04_detect_failure_events.py',
            'check': lambda: Path('results/failure_events.csv').exists()
        },
        {
            'name': '5. Grad-CAM 분석',
            'script': 'scripts/05_gradcam_failure_analysis.py',
            'check': lambda: Path('results/gradcam_metrics_timeseries.csv').exists()
        },
        {
            'name': '6. LLM 리포트 생성',
            'script': 'scripts/06_llm_report.py',
            'check': lambda: Path('results/report.md').exists()
        }
    ]
    
    for step in steps:
        print(f"\n{'='*60}")
        print(f"{step['name']}")
        print(f"{'='*60}")
        
        # Check if already complete
        if check_step_complete(step['name'], step['check']):
            print(f"[SKIP] {step['name']} 이미 완료됨")
            continue
        
        # Run script
        print(f"실행 중: {step['script']}")
        try:
            result = subprocess.run(
                [sys.executable, step['script']],
                check=True,
                capture_output=False
            )
            print(f"[OK] {step['name']} 완료")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {step['name']} 실패: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("[OK] 전체 실험 파이프라인 완료!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    main()
