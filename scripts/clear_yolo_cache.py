"""Clear corrupted YOLO dataset cache files.

This script removes all *.cache files that may be corrupted and causing
EOFError: Ran out of input errors during dataset loading.
"""

import sys
from pathlib import Path

def main():
    """Clear all YOLO cache files."""
    project_root = Path(__file__).parent.parent
    
    # Common cache file patterns
    cache_patterns = [
        "**/*.cache",
        "**/labels.cache",
        "**/train.cache",
        "**/val.cache",
    ]
    
    # Dataset directories to search
    dataset_dirs = [
        project_root / "datasets" / "visdrone",
        project_root / "datasets" / "visdrone_yolo",
        project_root / "datasets" / "visdrone_corrupt",
    ]
    
    cache_files = []
    
    # Find all cache files
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            continue
        
        for pattern in cache_patterns:
            for cache_file in dataset_dir.glob(pattern):
                if cache_file.is_file():
                    cache_files.append(cache_file)
    
    if not cache_files:
        print("No cache files found.")
        return
    
    print(f"Found {len(cache_files)} cache file(s):")
    for cache_file in cache_files:
        size = cache_file.stat().st_size
        print(f"  {cache_file.relative_to(project_root)} ({size:,} bytes)")
    
    # Confirm deletion
    print(f"\nDeleting {len(cache_files)} cache file(s)...")
    deleted = 0
    for cache_file in cache_files:
        try:
            cache_file.unlink()
            deleted += 1
            print(f"  [OK] Deleted: {cache_file.relative_to(project_root)}")
        except Exception as e:
            print(f"  [ERROR] Failed to delete {cache_file.relative_to(project_root)}: {e}")
    
    print(f"\n[OK] Deleted {deleted}/{len(cache_files)} cache file(s).")
    print("[INFO] Cache files will be automatically regenerated on next dataset load.")


if __name__ == "__main__":
    main()
