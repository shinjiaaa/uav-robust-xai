"""Run 07 (lead analysis) and 10 (visualization) in sequence. Must run from project root or use this script."""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent


def main():
    print(f"Project root: {ROOT}")
    print(f"Working dir for 07/10: {ROOT}\n")
    scripts = [
        ROOT / "scripts" / "07_lead_analysis.py",
        ROOT / "scripts" / "10_lead_visualization.py",
    ]
    for script in scripts:
        print(f">>> Running {script.name}")
        ret = subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
        if ret.returncode != 0:
            sys.exit(ret.returncode)
    print(f"\nDone (07 + 10). Outputs under {ROOT / 'results'} — refresh app or re-open JSON/figures if you don't see changes.")


if __name__ == "__main__":
    main()
