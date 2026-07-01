"""PLY helpers for reference_based.

This tool layer replaces the x265 bitstream/file interface used by the
original Dual-Critic code. The implementation is shared with the first
baseline because it already preserves 3DGS PLY field order and binary layout.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gs_baseline.ply_utils import *  # noqa: F401,F403


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a 3DGS PLY.")
    parser.add_argument("--ply", required=True)
    args = parser.parse_args()
    print_ply_summary(args.ply)
