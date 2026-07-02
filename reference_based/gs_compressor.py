"""3DGS compressor bridge used by Environment_GS.

The original trainRDO.py sent QP decisions to x265. This bridge receives
voxel-level factorized compression action IDs and writes:

1. a decoded float PLY for GraphDeCo rendering;
2. a LightGaussian-style compact zip package for real size measurement.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

try:
    from .compression_ops import apply_compression_to_vertices
    from .lightgaussian_compact import save_lightgaussian_compact_package
    from .ply_utils import GaussianPLY, write_ply
except ImportError:
    from compression_ops import apply_compression_to_vertices
    from lightgaussian_compact import save_lightgaussian_compact_package
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

        aux = stats.pop("_compact_aux", {}) or {}
        compact_root = self.output_dir / f"{output_path.stem}_lightgaussian_compact"
        compact_zip = self.output_dir / f"{output_path.stem}_compact.zip"
        compact_info = save_lightgaussian_compact_package(
            compressed_vertices,
            group_ids=aux.get("kept_group_ids", np.zeros(len(compressed_vertices), dtype=np.int32)),
            action_ids=aux.get("kept_action_ids", np.zeros(len(compressed_vertices), dtype=np.int16)),
            compact_root=compact_root,
            zip_path=compact_zip,
            original_vertex_count=int(len(ply.vertex_data)),
            original_size_bytes=0,
        )
        stats.update(compact_info)
        stats["render_ply_path"] = str(output_path)
        stats["render_ply_size_bytes"] = int(output_path.stat().st_size)
        return output_path, stats
