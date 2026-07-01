from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import normalize_crossscore_dir
from crossscore_bridge import compute_crossscore_real, inspect_crossscore_repo


def main():
    parser = argparse.ArgumentParser(description="Inspect CrossScore repository structure.")
    parser.add_argument("--crossscore-dir", required=True)
    args = parser.parse_args()

    crossscore_dir = normalize_crossscore_dir(args.crossscore_dir)
    info = inspect_crossscore_repo(crossscore_dir)
    for key, value in info.items():
        print(f"{key}: {value}")
    print("compute_crossscore_real: real CLI adapter is implemented.")
    print(f"real_function: {compute_crossscore_real.__name__}")


if __name__ == "__main__":
    main()
