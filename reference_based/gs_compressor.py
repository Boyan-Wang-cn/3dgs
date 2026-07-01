"""3DGS compressor bridge used by Environment_GS.

The original trainRDO.py sent QP decisions to x265. This bridge receives
voxel-level compression levels and writes a compressed PLY instead.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

try:
    from .compression_ops import apply_compression_to_vertices
    from .ply_utils import GaussianPLY, write_ply
except ImportError:
    from compression_ops import apply_compression_to_vertices
    from ply_utils import GaussianPLY, write_ply


class GSCompressor:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compress_scene(
        self,
        ply: GaussianPLY,
        group_indices: list[np.ndarray],
        compression_levels: list[int],
        scene_name: str,
        episode: int,
    ) -> tuple[Path, dict]:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / (
            f"{scene_name}_reference_based_ep{episode:04d}_{timestamp}.ply"
        )
        compressed_vertices, stats = apply_compression_to_vertices(
            ply.vertex_data,
            group_indices,
            compression_levels,
        )
        write_ply(ply, output_path, compressed_vertices)
        return output_path, stats
