"""Setup and environment check script."""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10+ is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"[OK] Python version: {sys.version.split()[0]}")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        ('numpy', 'numpy'),
        ('opencv-python', 'cv2'),
        ('pillow', 'PIL'),
        ('yaml', 'yaml'),
        ('torch', 'torch'),
        ('ultralytics', 'ultralytics'),
        ('grad-cam', 'pytorch_grad_cam'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('openai', 'openai'),
        ('tqdm', 'tqdm')
    ]
    
    missing = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"[OK] {package_name}")
        except ImportError:
            print(f"[MISSING] {package_name}")
            missing.append(package_name)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_env_file():
    """Check if .env file exists."""
    env_path = Path(".env")
    env_example = Path("env.example")
    
    if not env_path.exists():
        print(f"[WARNING] .env file not found.")
        if env_example.exists():
            print(f"  Copy env.example to .env and set OPENAI_API_KEY")
        else:
            print(f"  Create .env file with OPENAI_API_KEY=your_key")
        return False
    else:
        print("[OK] .env file exists")
        return True


def create_directories():
    """Create necessary directories."""
    dirs = [
        "results", "results/preds", "results/plots", "results/gradcam_samples",
        "datasets", "datasets/visdrone_yolo", "datasets/visdrone_corrupt"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("[OK] Directories created")


def main():
    """Main setup function."""
    print("=" * 60)
    print("Environment Setup Check")
    print("=" * 60)
    print()
    
    all_ok = True
    
    print("1. Checking Python version...")
    if not check_python_version():
        all_ok = False
    print()
    
    print("2. Checking dependencies...")
    if not check_dependencies():
        all_ok = False
    print()
    
    print("3. Checking .env file...")
    check_env_file()  # Warning only, not blocking
    print()
    
    print("4. Creating directories...")
    create_directories()
    print()
    
    if all_ok:
        print("=" * 60)
        print("[OK] Setup check passed!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("[WARNING] Some checks failed. Please fix the issues above.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
