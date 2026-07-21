"""Formal full-scene factorized PLY and compact V2 compressor bridge."""

from __future__ import annotations

from pathlib import Path
import re
import tempfile
import time
from typing import Any

import numpy as np

try:
    from .compression_ops import (
        FactorizedActionInput,
        PRUNING_MODE_OPACITY_BASELINE,
        apply_factorized_compression_to_vertices,
    )
    from .lightgaussian_compact import (
        load_factorized_lightgaussian_compact_package,
        save_factorized_lightgaussian_compact_package,
    )
    from .ply_utils import GaussianPLY, PLYProperty, read_ply, write_ply
except ImportError:
    from compression_ops import (
        FactorizedActionInput,
        PRUNING_MODE_OPACITY_BASELINE,
        apply_factorized_compression_to_vertices,
    )
    from lightgaussian_compact import (
        load_factorized_lightgaussian_compact_package,
        save_factorized_lightgaussian_compact_package,
    )
    from ply_utils import GaussianPLY, PLYProperty, read_ply, write_ply


class GSCompressor:
    """Write a decoded render PLY and, only at terminal, compact V2."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._artifact_sequence = 0

    @staticmethod
    def _nonnegative_integer(value: Any, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be a nonnegative integer")
        result = int(value)
        if result < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return result

    @staticmethod
    def _safe_text(value: Any, name: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_.")
        if not normalized:
            raise ValueError(f"{name} must be nonempty")
        return normalized

    def _output_path(self, scene: str, episode: int, artifact: str) -> Path:
        self._artifact_sequence += 1
        nonce = time.time_ns()
        return self.output_dir / (
            f"{scene}_ep{episode:04d}_{artifact}_{nonce}_{self._artifact_sequence}.ply"
        )

    def compress_scene_factorized(
        self,
        ply: GaussianPLY,
        group_indices: list[np.ndarray],
        factorized_actions: list[FactorizedActionInput | None],
        scene_name: str,
        episode: int,
        original_size_bytes: int,
        *,
        artifact_tag: str = "terminal",
        write_compact: bool = True,
    ) -> tuple[Path, dict[str, Any]]:
        """Compress every group, filling undecided groups with exact identity."""
        if not isinstance(getattr(ply, "vertex_data", None), np.ndarray):
            raise ValueError("ply must contain a NumPy vertex_data array")
        if ply.vertex_data.dtype.names is None:
            raise ValueError("ply.vertex_data must be structured")
        if not isinstance(group_indices, list) or not isinstance(factorized_actions, list):
            raise ValueError("group_indices and factorized_actions must be lists")
        if len(factorized_actions) > len(group_indices):
            raise ValueError("factorized_actions cannot exceed the group count")
        scene = self._safe_text(scene_name, "scene_name")
        artifact = self._safe_text(artifact_tag, "artifact_tag")
        normalized_episode = self._nonnegative_integer(episode, "episode")
        original_size = self._nonnegative_integer(
            original_size_bytes, "original_size_bytes"
        )
        if original_size == 0:
            raise ValueError("original_size_bytes must be positive")
        if not isinstance(write_compact, (bool, np.bool_)):
            raise ValueError("write_compact must be bool")

        compressed_vertices, stats = apply_factorized_compression_to_vertices(
            ply.vertex_data, group_indices, factorized_actions
        )
        auxiliary = stats.pop("_compact_aux")
        output_path = self._output_path(scene, normalized_episode, artifact)
        write_ply(ply, output_path, compressed_vertices)
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise RuntimeError("decoded factorized PLY was not written")

        if bool(write_compact):
            compact_root = self.output_dir / f"{output_path.stem}_factorized_compact"
            compact_zip = self.output_dir / f"{output_path.stem}_factorized_compact.zip"
            compact_info = save_factorized_lightgaussian_compact_package(
                compressed_vertices,
                group_ids=auxiliary["kept_group_ids"],
                pruning_levels=auxiliary["kept_pruning_levels"],
                precision_levels=auxiliary["kept_precision_levels"],
                storage_ids=auxiliary["kept_storage_ids"],
                compact_root=compact_root,
                zip_path=compact_zip,
                original_vertex_count=len(ply.vertex_data),
                original_size_bytes=original_size,
            )
            package = Path(compact_info.get("compact_package_path", ""))
            if not package.is_file():
                raise RuntimeError("compact V2 writer did not create its package")
            actual_size = package.stat().st_size
            if compact_info.get("compact_format") != "rl_factorized_3dgs_compact_v2":
                raise RuntimeError("compact writer returned an incompatible format")
            if int(compact_info.get("compact_size_bytes", -1)) != actual_size:
                raise RuntimeError("compact writer size metadata is inconsistent")
            expected_ratio = actual_size / original_size
            if not np.isclose(compact_info.get("compact_size_ratio", np.nan), expected_ratio):
                raise RuntimeError("compact writer ratio metadata is inconsistent")
            stats.update(compact_info)
            stats["compact_written"] = True
        else:
            stats.update({
                "compact_written": False,
                "compact_package_path": "",
                "compact_size_bytes": 0,
                "compact_size_ratio": None,
                "compact_format": "",
            })

        stats.update({
            "artifact_tag": artifact,
            "output_kind": "terminal_compact" if write_compact else "checkpoint_render",
            "original_size_bytes": original_size,
            "original_vertex_count": len(ply.vertex_data),
            "render_ply_path": str(output_path),
            "render_ply_size_bytes": output_path.stat().st_size,
            "pruning_mode": PRUNING_MODE_OPACITY_BASELINE,
        })
        return output_path, stats


_FIRST_VERSION_COMPRESSOR_REPORT: dict[str, Any] = {}


def validate_first_version_compressor() -> bool:
    """Validate identity fill, checkpoint PLY, terminal compact, and decode."""
    if _FIRST_VERSION_COMPRESSOR_REPORT.get("validated") is True:
        return True

    def require(condition: Any, message: str) -> None:
        if not bool(condition):
            raise AssertionError(message)

    names = [
        "x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1",
        "f_dc_2", *[f"f_rest_{index}" for index in range(45)],
    ]
    vertices = np.zeros(20, dtype=np.dtype([(name, "<f4") for name in names]))
    for field_index, name in enumerate(names):
        vertices[name] = np.linspace(
            0.01 + field_index, 1.01 + field_index, 20, dtype=np.float32
        )
    vertices["opacity"] = np.arange(20, dtype=np.float32)
    properties = [
        PLYProperty(name=name, ply_type="float", numpy_type="f4") for name in names
    ]
    header = [
        "ply", "format binary_little_endian 1.0", "element vertex 20",
        *[f"property float {name}" for name in names], "end_header",
    ]
    groups = [np.arange(10, dtype=np.int64), np.arange(10, 20, dtype=np.int64)]
    with tempfile.TemporaryDirectory(prefix="first_version_compressor_") as directory:
        root = Path(directory)
        ply = GaussianPLY(
            path=root / "source.ply", fmt="binary_little_endian",
            header_lines=header, vertex_count=20,
            vertex_properties=properties, vertex_data=vertices, tail_bytes=b"",
        )
        compressor = GSCompressor(root / "outputs")
        checkpoint_path, checkpoint = compressor.compress_scene_factorized(
            ply, groups, [(0, 0)], "validation", 1, 2_000_000,
            artifact_tag="checkpoint_0008", write_compact=False,
        )
        checkpoint_ply = read_ply(checkpoint_path)
        require(checkpoint_path.is_file() and not checkpoint["compact_written"], "checkpoint PLY failed")
        require(checkpoint["identity_filled_group_count"] == 1, "identity fill stats failed")
        require(np.array_equal(checkpoint_ply.vertex_data[10:], vertices[10:]), "identity fill changed undecided group")

        terminal_path, terminal = compressor.compress_scene_factorized(
            ply, groups, [(4, 5), (0, 0)], "validation", 1, 2_000_000,
            artifact_tag="terminal", write_compact=True,
        )
        package = Path(terminal["compact_package_path"])
        decoded = load_factorized_lightgaussian_compact_package(package)
        require(terminal_path.is_file() and package.is_file(), "terminal artifacts failed")
        require(terminal["compact_format"] == "rl_factorized_3dgs_compact_v2", "compact format changed")
        require(terminal["pruned_vertices"] == 3 and terminal["kept_vertices"] == 17, "opacity pruning failed")
        require(len(decoded["vertices"]) == 17 and decoded["compact_format"] == terminal["compact_format"], "compact decoder roundtrip failed")
        require(np.all(np.isfinite(decoded["vertices"]["x"])), "decoded compact is non-finite")
        require(np.all(read_ply(terminal_path).vertex_data["f_rest_44"][:7] == 0.0), "precision profile was not applied")

        _FIRST_VERSION_COMPRESSOR_REPORT.update({
            "validated": True,
            "identity_fill": True,
            "opacity_pruning": True,
            "precision_application": True,
            "checkpoint_ply": True,
            "terminal_compact": True,
            "compact_decoder_roundtrip": True,
            "compact_format": terminal["compact_format"],
            "compact_size_bytes": terminal["compact_size_bytes"],
        })
    return True


def validate_factorized_gs_compressor() -> bool:
    """Compatibility validation name for the formal compressor suite."""
    return validate_first_version_compressor()
