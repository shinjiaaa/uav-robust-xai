"""Extract continuous frame clips with tiny objects."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.frame_sequences import (
    group_images_by_sequence,
    extract_continuous_clips,
    find_tiny_object_clips
)
from src.utils.io import load_yaml, save_json
from src.utils.seed import set_seed


def main():
    """Main function."""
    config_path = Path("configs/experiment.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_yaml(config_path)
    set_seed(config['seed'])
    
    print("=" * 60)
    print("Extracting Frame Clips with Tiny Objects")
    print("=" * 60)
    print()
    
    # Load images
    visdrone_root = Path(config['dataset']['visdrone_root'])
    split = config['evaluation']['splits'][0]  # Use first split
    
    if (visdrone_root / f"VisDrone2019-DET-{split}").exists():
        split_dir = visdrone_root / f"VisDrone2019-DET-{split}"
    else:
        print(f"Error: Split directory not found")
        sys.exit(1)
    
    images_dir = split_dir / "images" if (split_dir / "images").exists() else split_dir
    annotations_dir = split_dir / "annotations" if (split_dir / "annotations").exists() else split_dir
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"Found {len(image_files)} images")
    
    # Group by sequence
    print("\n1. Grouping images by sequence...")
    sequences = group_images_by_sequence(image_files)
    print(f"   Found {len(sequences)} sequences")
    
    # Extract clips
    print("\n2. Extracting continuous clips...")
    frame_config = config['frame_sequences']
    clips = extract_continuous_clips(
        sequences,
        min_clip_length=frame_config['min_clip_length'],
        max_clip_length=frame_config['max_clip_length']
    )
    print(f"   Extracted {len(clips)} clips")
    
    # Find tiny object clips
    print("\n3. Finding clips with tiny objects...")
    tiny_config = config['tiny_objects']
    result = find_tiny_object_clips(
        clips,
        annotations_dir,
        tiny_config,
        sample_size=tiny_config['sample_size'],
        seed=config['seed']
    )
    if isinstance(result, tuple):
        tiny_clips, tiny_objects = result
    else:
        tiny_clips = result
        tiny_objects = []
    print(f"   Found {len(tiny_clips)} clips with tiny objects")
    print(f"   Total tiny objects: {len(tiny_objects)}")
    
    # Save results
    print("\n4. Saving results...")
    results_dir = Path(config['results']['root'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save clips info
    clips_info = []
    for clip in tiny_clips:
        # Convert frame paths to relative strings
        frame_paths = []
        for f in clip['frames']:
            if isinstance(f, Path):
                # Try relative to visdrone_root first
                try:
                    frame_paths.append(str(f.relative_to(visdrone_root)))
                except ValueError:
                    # If not relative, use as is
                    frame_paths.append(str(f))
            else:
                frame_paths.append(str(f))
        
        clips_info.append({
            'clip_id': clip['clip_id'],
            'sequence_id': clip['sequence_id'],
            'start_idx': clip['start_idx'],
            'end_idx': clip['end_idx'],
            'length': clip['length'],
            'frames': frame_paths,
            'tiny_object_count': len(clip.get('tiny_objects', []))
        })
    
    clips_file = results_dir / "frame_clips.json"
    save_json(clips_info, clips_file)
    print(f"   Saved clips to {clips_file}")
    
    # Save tiny objects
    tiny_objects_file = results_dir / "tiny_objects_samples.json"
    save_json(tiny_objects, tiny_objects_file)
    print(f"   Saved tiny objects to {tiny_objects_file}")
    
    print("\n[OK] Frame clip extraction complete!")


if __name__ == "__main__":
    main()
