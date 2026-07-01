"""Voxel grouping helpers for reference_based.

Voxel groups are the 3DGS replacement for frames in the original
Train/DRL_x265_TRAIN/Environment.py pipeline.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gs_baseline.voxel_grouping import *  # noqa: F401,F403
