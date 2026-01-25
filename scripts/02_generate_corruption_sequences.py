"""Generate corruption sequences for frame clips."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corruption.corruptions import corrupt_image
from src.utils.io import load_yaml, load_json
from src.utils.seed import set_seed
from tqdm import tqdm
import shutil


def main():
    """Main function."""
    config_path = Path("configs/experiment.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_yaml(config_path)
    set_seed(config['seed'])
    
    print("=" * 60)
    print("Generating Corruption Sequences")
    print("=" * 60)
    print()
    
    # Load clips
    results_dir = Path(config['results']['root'])
    clips_file = results_dir / "frame_clips.json"
    if not clips_file.exists():
        print(f"Error: Frame clips not found. Run scripts/01_extract_frame_clips.py first")
        sys.exit(1)
    
    clips = load_json(clips_file)
    print(f"Loaded {len(clips)} clips")
    
    visdrone_root = Path(config['dataset']['visdrone_root'])
    corruptions_root = Path(config['dataset']['corruptions_root'])
    
    corruptions = config['corruptions']['types']
    severities = config['corruptions']['severities']
    
    print(f"\nGenerating corruptions for {len(corruptions)} types, {len(severities)} severities...")
    
    # Process each clip
    for clip in tqdm(clips, desc="Processing clips"):
        clip_id = clip['clip_id']
        
        for corruption_type in corruptions:
            for severity in severities:
                # Create output directory for this clip/corruption/severity
                clip_corrupt_dir = corruptions_root / "sequences" / clip_id / corruption_type / str(severity) / "images"
                clip_corrupt_dir.mkdir(parents=True, exist_ok=True)
                
                # Process each frame in clip
                for frame_rel_path in clip['frames']:
                    frame_path = visdrone_root / frame_rel_path
                    if not frame_path.exists():
                        continue
                    
                    # Generate corrupted image
                    output_path = clip_corrupt_dir / frame_path.name
                    corrupt_image(
                        frame_path,
                        corruption_type,
                        severity,
                        output_path,
                        seed=config['seed']
                    )
    
    print("\n[OK] Corruption sequence generation complete!")


if __name__ == "__main__":
    main()
