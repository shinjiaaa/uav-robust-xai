"""Check experiment progress."""

from pathlib import Path

print("=" * 60)
print("실험 진행 상태 확인")
print("=" * 60)
print()

# Check frame clips
clips_file = Path("results/frame_clips.json")
if clips_file.exists():
    import json
    with open(clips_file, 'r') as f:
        clips = json.load(f)
    print(f"[OK] 프레임 클립: {len(clips)}개")
else:
    print("[대기 중] 프레임 클립 없음")

# Check corruption sequences
corrupt_dir = Path("datasets/visdrone_corrupt/sequences")
if corrupt_dir.exists():
    jpg_files = list(corrupt_dir.rglob("*.jpg"))
    print(f"[진행 중] 변조 이미지: {len(jpg_files)}개 생성됨")
    
    # Check completion
    clips_file = Path("results/frame_clips.json")
    if clips_file.exists():
        import json
        with open(clips_file, 'r') as f:
            clips = json.load(f)
        
        total_expected = len(clips) * 3 * 5  # clips * corruptions * severities
        avg_frames_per_clip = sum(len(c['frames']) for c in clips) / len(clips) if clips else 0
        total_expected_images = int(total_expected * avg_frames_per_clip)
        
        if len(jpg_files) >= total_expected_images * 0.9:
            print(f"[OK] 변조 시퀀스 생성 완료!")
        else:
            print(f"[진행 중] 예상: {total_expected_images}개, 완료: {len(jpg_files)}개")
else:
    print("[대기 중] 변조 시퀀스 생성 중...")

# Check detection records
records_file = Path("results/tiny_records_timeseries.csv")
if records_file.exists():
    import pandas as pd
    df = pd.read_csv(records_file)
    print(f"[OK] 탐지 기록: {len(df)}개")
    print(f"  모델: {df['model'].unique().tolist()}")
else:
    print("[대기 중] 탐지 기록 없음")

print()
print("=" * 60)
