from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_path_utils import ensure_model_structure_from_ply, get_point_cloud_ply


def main():
    parser = argparse.ArgumentParser(description="Prepare official 3DGS model_path structure.")
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    args = parser.parse_args()

    model_dir = ensure_model_structure_from_ply(args.ply, args.out, args.iteration)
    point_cloud_ply = get_point_cloud_ply(model_dir, args.iteration)
    print(f"model_dir: {model_dir}")
    print(f"point_cloud_ply: {point_cloud_ply}")
    print(f"exists: {point_cloud_ply.exists()}")


if __name__ == "__main__":
    main()
