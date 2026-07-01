"""Compression operations for reference_based.

This module replaces the x265 encoder action effect with 3DGS PLY pruning,
SH reduction, and float-domain quantization.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gs_baseline.compression_ops import *  # noqa: F401,F403
