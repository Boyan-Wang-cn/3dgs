"""3DGS compression operations for reference_based.

This version replaces the old coupled 0--4 compression level with a
factorized 25-action space:

    action_id = pruning_level * 5 + precision_level

The compact package follows the LightGaussian VecTree storage idea:
metadata + packed masks/indices + compact attribute arrays are written under
an ``extreme_saving`` directory and zipped.  A decoded float PLY is still
written for GraphDeCo rendering, while the real compression reward uses the
compact zip size.

LightGaussian attribution:
- VITA-Group/LightGaussian stores compact data under ``extreme_saving`` and
  uses ``np.savez_compressed`` plus ``np.packbits`` for compact masks/indices.
- This file adapts that storage pattern to the RL group-wise action setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ACTION_GRID_SIZE = 5
ACTION_MAX = ACTION_GRID_SIZE * ACTION_GRID_SIZE - 1

PRUNING_RATES = [0.0, 0.05, 0.10, 0.20, 0.30]
PRECISION_PROFILES = [
    {"sh_degree": 3, "sh_bit": 16, "geo_bit": 16},
    {"sh_degree": 2, "sh_bit": 12, "geo_bit": 16},
    {"sh_degree": 2, "sh_bit": 8, "geo_bit": 12},
    {"sh_degree": 1, "sh_bit": 8, "geo_bit": 10},
    {"sh_degree": 0, "sh_bit": 6, "geo_bit": 8},
]

GEO_FIELD_PREFIXES = ("x", "y", "z", "opacity")
GEO_SCALE_PREFIX = "scale_"
GEO_ROT_PREFIX = "rot_"
SH_DC_PREFIX = "f_dc_"
SH_REST_PREFIX = "f_rest_"


@dataclass(frozen=True)
class DecodedAction:
    action_id: int
    pruning_level: int
    precision_level: int
    pruning_rate: float
    sh_degree: int
    sh_bit: int
    geo_bit: int


def decode_action(action: int | float | np.ndarray) -> DecodedAction:
    value = int(round(float(np.asarray(action).reshape(-1)[0])))
    value = int(np.clip(value, 0, ACTION_MAX))
    pruning_level = value // ACTION_GRID_SIZE
    precision_level = value % ACTION_GRID_SIZE
    precision = PRECISION_PROFILES[precision_level]
    return DecodedAction(
        action_id=value,
        pruning_level=pruning_level,
        precision_level=precision_level,
        pruning_rate=float(PRUNING_RATES[pruning_level]),
        sh_degree=int(precision["sh_degree"]),
        sh_bit=int(precision["sh_bit"]),
        geo_bit=int(precision["geo_bit"]),
    )


def sh_rest_count_for_degree(sh_degree: int) -> int:
    sh_degree = int(np.clip(sh_degree, 0, 3))
    # 3 color channels * non-DC SH bases.
    return int(3 * ((sh_degree + 1) ** 2 - 1))


def _field_names(vertex_data: np.ndarray) -> list[str]:
    return list(vertex_data.dtype.names or [])


def _sorted_indexed_fields(names: list[str], prefix: str) -> list[str]:
    out = []
    for name in names:
        if name.startswith(prefix):
            try:
                idx = int(name[len(prefix):])
            except ValueError:
                continue
            out.append((idx, name))
    return [name for _, name in sorted(out)]


def geo_fields(vertex_data: np.ndarray) -> list[str]:
    names = _field_names(vertex_data)
    fields: list[str] = []
    for name in ("x", "y", "z", "opacity"):
        if name in names:
            fields.append(name)
    fields.extend(_sorted_indexed_fields(names, GEO_SCALE_PREFIX))
    fields.extend(_sorted_indexed_fields(names, GEO_ROT_PREFIX))
    return fields


def sh_dc_fields(vertex_data: np.ndarray) -> list[str]:
    return _sorted_indexed_fields(_field_names(vertex_data), SH_DC_PREFIX)


def sh_rest_fields(vertex_data: np.ndarray) -> list[str]:
    return _sorted_indexed_fields(_field_names(vertex_data), SH_REST_PREFIX)


def active_sh_fields(vertex_data: np.ndarray, sh_degree: int) -> list[str]:
    dc = sh_dc_fields(vertex_data)
    rest = sh_rest_fields(vertex_data)
    keep_rest = sh_rest_count_for_degree(sh_degree)
    return dc + rest[:keep_rest]


def _quantize_dequantize(values: np.ndarray, bits: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    if bits >= 32:
        return values.copy()
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return values.copy()
    levels = float((1 << int(bits)) - 1)
    q = np.rint((values - vmin) / (vmax - vmin) * levels)
    q = np.clip(q, 0, levels)
    deq = q / levels * (vmax - vmin) + vmin
    return deq.astype(np.float32)


def _apply_sh_degree(vertices: np.ndarray, indices: np.ndarray, sh_degree: int) -> None:
    rest = sh_rest_fields(vertices)
    keep_rest = sh_rest_count_for_degree(sh_degree)
    if keep_rest >= len(rest):
        return
    for field in rest[keep_rest:]:
        vertices[field][indices] = 0.0


def _apply_uniform_quantization(vertices: np.ndarray, indices: np.ndarray, fields: list[str], bits: int) -> None:
    if len(indices) == 0:
        return
    for field in fields:
        vertices[field][indices] = _quantize_dequantize(vertices[field][indices], bits)


def action_histograms(actions: list[int]) -> dict[str, Any]:
    decoded = [decode_action(a) for a in actions]
    action_ids = np.asarray([d.action_id for d in decoded], dtype=np.int32)
    pruning_levels = np.asarray([d.pruning_level for d in decoded], dtype=np.int32)
    precision_levels = np.asarray([d.precision_level for d in decoded], dtype=np.int32)

    def hist(arr: np.ndarray) -> dict[str, int]:
        if arr.size == 0:
            return {}
        vals, counts = np.unique(arr, return_counts=True)
        return {str(int(v)): int(c) for v, c in zip(vals, counts)}

    return {
        "action_histogram": hist(action_ids),
        "level_histogram": hist(action_ids),
        "pruning_level_histogram": hist(pruning_levels),
        "precision_level_histogram": hist(precision_levels),
        "mean_action": float(np.mean(action_ids)) if action_ids.size else 0.0,
        "mean_level": float(np.mean(action_ids)) if action_ids.size else 0.0,
        "mean_pruning_rate": float(np.mean([d.pruning_rate for d in decoded])) if decoded else 0.0,
        "mean_sh_degree": float(np.mean([d.sh_degree for d in decoded])) if decoded else 0.0,
        "mean_sh_bit": float(np.mean([d.sh_bit for d in decoded])) if decoded else 0.0,
        "mean_geo_bit": float(np.mean([d.geo_bit for d in decoded])) if decoded else 0.0,
    }


def estimate_size_ratio_from_actions(
    group_indices: list[np.ndarray],
    actions: list[int | None],
    total_vertices: int,
) -> float:
    """Estimate compact bit cost ratio relative to full 3DGS float attributes.

    The estimate intentionally mirrors the compact package rather than the
    decoded render PLY: pruning reduces vertex count, SH-degree reduces the
    number of SH coefficients saved, and bit-width controls packed bits.
    """
    if total_vertices <= 0:
        return 1.0
    original_bits_per_gaussian = 32.0 * (11 + 48)  # xyz+opacity+scale+rot + SH DC/rest.
    total_bits = 0.0
    for idx, group in enumerate(group_indices):
        action = 0 if idx >= len(actions) or actions[idx] is None else actions[idx]
        d = decode_action(action)
        n = int(len(group))
        kept = max(0, n - int(np.floor(n * d.pruning_rate)))
        sh_coeffs = 3 + sh_rest_count_for_degree(d.sh_degree)
        total_bits += kept * (11 * d.geo_bit + sh_coeffs * d.sh_bit)
    return float(total_bits / max(total_vertices * original_bits_per_gaussian, 1.0))


def apply_compression_to_vertices(
    vertex_data: np.ndarray,
    group_indices: list[np.ndarray],
    compression_levels: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply group-wise pruning, SH reduction, and quantize-dequantize.

    ``compression_levels`` are now factorized action IDs in [0, 24].  The
    returned PLY vertices are decoded float values for renderer compatibility.
    Compact-package metadata for real size measurement is returned in a hidden
    ``_compact_aux`` entry consumed by ``GSCompressor``.
    """
    vertices = vertex_data.copy()
    num_vertices = int(len(vertices))
    keep_mask = np.ones(num_vertices, dtype=bool)
    actions = [decode_action(a) for a in compression_levels]

    # First choose pruning mask on original indices, using low opacity as the
    # deterministic least-important criterion.  If opacity is unavailable, use
    # input order as fallback.
    for group_idx, indices in enumerate(group_indices):
        if len(indices) == 0:
            continue
        action = actions[group_idx] if group_idx < len(actions) else decode_action(0)
        prune_count = int(np.floor(len(indices) * action.pruning_rate))
        if prune_count <= 0:
            continue
        if "opacity" in vertices.dtype.names:
            order = np.argsort(np.asarray(vertices["opacity"][indices], dtype=np.float64))
            prune_indices = indices[order[:prune_count]]
        else:
            prune_indices = indices[:prune_count]
        keep_mask[prune_indices] = False

    vertex_group_ids = np.zeros(num_vertices, dtype=np.int32)
    vertex_action_ids = np.zeros(num_vertices, dtype=np.int16)
    g_fields = geo_fields(vertices)
    rest_fields = sh_rest_fields(vertices)

    # Apply SH-degree zeroing and quantize-dequantize only to surviving points.
    for group_idx, indices in enumerate(group_indices):
        action = actions[group_idx] if group_idx < len(actions) else decode_action(0)
        vertex_group_ids[indices] = int(group_idx)
        vertex_action_ids[indices] = int(action.action_id)
        kept = np.asarray(indices, dtype=np.int64)[keep_mask[indices]]
        if len(kept) == 0:
            continue
        _apply_sh_degree(vertices, kept, action.sh_degree)
        _apply_uniform_quantization(vertices, kept, g_fields, action.geo_bit)
        _apply_uniform_quantization(vertices, kept, active_sh_fields(vertices, action.sh_degree), action.sh_bit)
        # High-order fields were set to zero above.  They stay in the render PLY
        # but are not written into the compact package.
        if rest_fields:
            pass

    compressed_vertices = vertices[keep_mask].copy()
    kept_group_ids_arr = vertex_group_ids[keep_mask].astype(np.int32, copy=True)
    kept_action_ids_arr = vertex_action_ids[keep_mask].astype(np.int16, copy=True)
    kept_original_indices_arr = np.nonzero(keep_mask)[0].astype(np.int64, copy=False)

    hist_info = action_histograms([a.action_id for a in actions])
    estimated_ratio = estimate_size_ratio_from_actions(
        group_indices,
        [a.action_id for a in actions],
        total_vertices=num_vertices,
    )
    stats: dict[str, Any] = {
        "original_vertices": num_vertices,
        "processed_groups": int(len(group_indices)),
        "pruned_vertices": int(num_vertices - len(compressed_vertices)),
        "kept_vertices": int(len(compressed_vertices)),
        "kept_vertex_ratio": float(len(compressed_vertices) / max(num_vertices, 1)),
        "estimated_size_ratio": float(estimated_ratio),
        "action_mode": "factorized_25_lightgaussian_compact",
        **hist_info,
        "_compact_aux": {
            "kept_original_indices": kept_original_indices_arr,
            "kept_group_ids": kept_group_ids_arr,
            "kept_action_ids": kept_action_ids_arr,
        },
    }
    return compressed_vertices, stats
