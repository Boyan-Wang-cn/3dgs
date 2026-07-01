from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import normalize_gaussian_splatting_dir
from render_bridge import find_gt_dir, find_render_dir, run_graphdeco_render


def main():
    parser = argparse.ArgumentParser(description="Render one 3DGS model_path.")
    parser.add_argument("--gaussian-splatting-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=4)
    args = parser.parse_args()

    gs_dir = normalize_gaussian_splatting_dir(args.gaussian_splatting_dir)
    run_graphdeco_render(
        gs_dir,
        args.model_path,
        args.source_path,
        iteration=args.iteration,
        resolution=args.resolution,
    )
    print(f"render_dir: {find_render_dir(args.model_path, args.iteration)}")
    print(f"gt_dir: {find_gt_dir(args.model_path, args.iteration)}")


if __name__ == "__main__":
    main()
