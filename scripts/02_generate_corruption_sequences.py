"""Generate corruptions for single images."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corruption.corruptions import corrupt_image
from src.utils.io import load_yaml, load_json
from src.utils.seed import set_seed
from tqdm import tqdm


def main():
    """Main function."""
    config_path = Path("configs/experiment.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    config = load_yaml(config_path)
    set_seed(config['seed'])
    
    print("=" * 60)
    print("Generating Corruptions for Single Images")
    print("=" * 60)
    print()
    
    # Load tiny objects
    results_dir = Path(config['results']['root'])
    tiny_objects_file = results_dir / "tiny_objects_samples.json"
    if not tiny_objects_file.exists():
        print(f"Error: Tiny objects not found. Run scripts/01_sample_tiny_objects.py first")
        sys.exit(1)
    
    tiny_objects = load_json(tiny_objects_file)
    print(f"Loaded {len(tiny_objects)} tiny objects")
    
    visdrone_root = Path(config['dataset']['visdrone_root'])
    corruptions_root = Path(config['dataset']['corruptions_root'])
    
    corruptions = config['corruptions']['types']
    severities = config['corruptions']['severities']
    
    # Option A: 100 objects × 3 corruptions × 5 levels = 1500 frames (level0 = reference only, no save)
    # Stored as: corruptions_root / {corruption} / L{1..4} / images / {filename}.png
    print(f"\nGenerating corruptions: {len(tiny_objects)} objects × {len(corruptions)} types × {len(severities)} levels")
    print("Level 0: reference only (original image_path); levels 1-4: saved.")
    
    # Get unique (image_id, frame_path) from tiny objects (one_per_image → 100 images)
    unique_images = {}
    for tiny_obj in tiny_objects:
        image_id = tiny_obj['image_id']
        frame_path = tiny_obj['frame_path']
        if image_id not in unique_images:
            unique_images[image_id] = frame_path
    
    n_images = len(unique_images)
    n_saved = n_images * len(corruptions) * (len(severities) - 1)  # exclude level 0
    print(f"Processing {n_images} unique images → {n_saved} corrupted files (3×4 per image)")
    
    for image_id, frame_rel_path in tqdm(unique_images.items(), desc="Processing images"):
        frame_path = visdrone_root / frame_rel_path
        if not frame_path.exists():
            continue
        
        for corruption_type in corruptions:
            for severity in severities:
                if severity == 0:
                    # Level 0: reference only, do not save (avoid duplicate; use original image_path)
                    continue
                
                # Path: corruptions_root / {corruption} / L{1..4} / images / {filename}
                corrupt_dir = corruptions_root / corruption_type / f"L{severity}" / "images"
                corrupt_dir.mkdir(parents=True, exist_ok=True)
                output_path = corrupt_dir / frame_path.name
                
                if output_path.exists():
                    continue
                
                corrupt_image(
                    frame_path,
                    corruption_type,
                    severity,
                    output_path,
                    seed=config['seed']
                )
    
    print("\n[OK] Corruption generation complete! (Total 1500 frames: 500 per corruption type, level0 = reference only)")


if __name__ == "__main__":
    main()
