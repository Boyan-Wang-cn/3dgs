from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ply_utils import numbered_fields


@dataclass(frozen=True)
class CompressionLevel:
    level: int
    pruning_rate: float
    sh_degree: int
    sh_bit: int
    geometry_bit: int


COMPRESSION_LEVELS: dict[int, CompressionLevel] = {
    0: CompressionLevel(0, pruning_rate=0.00, sh_degree=3, sh_bit=16, geometry_bit=16),
    1: CompressionLevel(1, pruning_rate=0.05, sh_degree=2, sh_bit=12, geometry_bit=16),
    2: CompressionLevel(2, pruning_rate=0.10, sh_degree=2, sh_bit=8, geometry_bit=12),
    3: CompressionLevel(3, pruning_rate=0.20, sh_degree=1, sh_bit=8, geometry_bit=10),
    4: CompressionLevel(4, pruning_rate=0.30, sh_degree=0, sh_bit=6, geometry_bit=8),
}

GEOMETRY_FIELDS = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
F_DC_FIELDS = ["f_dc_0", "f_dc_1", "f_dc_2"]


def quantize_array(values: np.ndarray, bits: int) -> np.ndarray:
    """Uniformly quantize values to a limited number of levels, stored as floats."""
    arr = np.asarray(values)
    if arr.size == 0:
        return arr.copy()
    if bits <= 0:
        return np.zeros_like(arr)

    finite = np.isfinite(arr)
    if not finite.any():
        return arr.copy()

    v_min = float(np.min(arr[finite]))
    v_max = float(np.max(arr[finite]))
    if np.isclose(v_min, v_max):
        return arr.copy()

    levels = float((1 << bits) - 1)
    normalized = (arr - v_min) / (v_max - v_min)
    quantized = np.round(np.clip(normalized, 0.0, 1.0) * levels) / levels
    return (quantized * (v_max - v_min) + v_min).astype(arr.dtype, copy=False)


def _validate_level(level: int) -> CompressionLevel:
    try:
        return COMPRESSION_LEVELS[int(level)]
    except KeyError as exc:
        raise ValueError(f"Compression level must be 0..4, got {level}.") from exc


def _prune_group_by_opacity(
    vertex_data: np.ndarray,
    indices: np.ndarray,
    pruning_rate: float,
    keep_mask: np.ndarray,
) -> int:
    if pruning_rate <= 0.0 or len(indices) == 0:
        return 0
    prune_count = int(np.floor(len(indices) * pruning_rate))
    if prune_count <= 0:
        return 0

    if "opacity" in vertex_data.dtype.names:
        values = vertex_data["opacity"][indices]
        local_order = np.argsort(values, kind="stable")
    else:
        local_order = np.arange(len(indices))
    prune_indices = indices[local_order[:prune_count]]
    keep_mask[prune_indices] = False
    return int(len(prune_indices))


def _apply_sh_degree(vertex_data: np.ndarray, indices: np.ndarray, sh_degree: int) -> None:
    # Baseline approximation: f_rest ordering is treated as low-to-high order.
    # For degree 3 keep all 45 rest coefficients, degree 2 keeps 24,
    # degree 1 keeps 9, and degree 0 keeps only f_dc.
    f_rest_fields = numbered_fields(vertex_data.dtype.names or [], "f_rest")
    if not f_rest_fields or sh_degree >= 3:
        return

    keep_count_by_degree = {2: 24, 1: 9, 0: 0}
    keep_count = keep_count_by_degree.get(sh_degree, 0)
    if len(f_rest_fields) != 45:
        keep_count = int(round(len(f_rest_fields) * keep_count / 45.0))
    for field in f_rest_fields[keep_count:]:
        vertex_data[field][indices] = 0.0


def _quantize_fields(vertex_data: np.ndarray, indices: np.ndarray, fields: list[str], bits: int) -> None:
    for field in fields:
        if field not in vertex_data.dtype.names:
            continue
        values = vertex_data[field]
        values[indices] = quantize_array(values[indices], bits)


def apply_compression_to_vertices(
    vertex_data: np.ndarray,
    group_indices: list[np.ndarray],
    actions: list[int],
) -> tuple[np.ndarray, dict]:
    """Apply pruning, SH reduction, and float-domain quantization to selected groups."""
    if len(group_indices) != len(actions):
        raise ValueError("group_indices and actions must have the same length.")

    compressed = vertex_data.copy()
    keep_mask = np.ones(len(compressed), dtype=bool)
    stats = {
        "original_vertices": int(len(vertex_data)),
        "processed_groups": int(len(group_indices)),
        "pruned_vertices": 0,
        "level_counts": {level: 0 for level in COMPRESSION_LEVELS},
    }

    for indices, action in zip(group_indices, actions):
        level = _validate_level(action)
        stats["level_counts"][level.level] += 1
        stats["pruned_vertices"] += _prune_group_by_opacity(
            compressed, indices, level.pruning_rate, keep_mask
        )
        _apply_sh_degree(compressed, indices, level.sh_degree)
        sh_fields = F_DC_FIELDS + numbered_fields(compressed.dtype.names or [], "f_rest")
        _quantize_fields(compressed, indices, sh_fields, level.sh_bit)
        _quantize_fields(compressed, indices, GEOMETRY_FIELDS, level.geometry_bit)

    compressed = compressed[keep_mask].copy()
    stats["kept_vertices"] = int(len(compressed))
    stats["kept_vertex_ratio"] = (
        float(len(compressed)) / float(len(vertex_data)) if len(vertex_data) else 0.0
    )
    return compressed, stats


def estimate_size_ratio_from_actions(
    group_indices: list[np.ndarray],
    actions: list[int | None],
    total_vertices: int,
) -> float:
    """Estimate current size ratio from pruning only; quantized floats keep file width."""
    if total_vertices <= 0:
        return 0.0
    kept_estimate = float(total_vertices)
    for indices, action in zip(group_indices, actions):
        if action is None:
            continue
        level = _validate_level(action)
        kept_estimate -= float(len(indices)) * level.pruning_rate
    return max(0.0, kept_estimate / float(total_vertices))
