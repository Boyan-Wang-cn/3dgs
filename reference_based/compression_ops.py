"""First-version two-axis compression operations for grouped 3DGS data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PRUNING_RATES = [0.0, 0.05, 0.10, 0.20, 0.30]
PRECISION_PROFILES_V2 = [
    {"sh_degree": 3, "sh_bit": 32, "geo_bit": 32},
    {"sh_degree": 3, "sh_bit": 16, "geo_bit": 16},
    {"sh_degree": 2, "sh_bit": 12, "geo_bit": 16},
    {"sh_degree": 2, "sh_bit": 8, "geo_bit": 12},
    {"sh_degree": 1, "sh_bit": 8, "geo_bit": 10},
    {"sh_degree": 0, "sh_bit": 6, "geo_bit": 8},
]

# Protected compact readers still decode their original 5-profile packages.
# This table is read-only compatibility for serialized artifacts and is never
# used by the formal Actor, Critic, Replay, or factorized executor.
PRECISION_PROFILES = [
    {"sh_degree": 3, "sh_bit": 16, "geo_bit": 16},
    {"sh_degree": 2, "sh_bit": 12, "geo_bit": 16},
    {"sh_degree": 2, "sh_bit": 8, "geo_bit": 12},
    {"sh_degree": 1, "sh_bit": 8, "geo_bit": 10},
    {"sh_degree": 0, "sh_bit": 6, "geo_bit": 8},
]

FACTORIZED_PRUNING_LEVELS = 5
FACTORIZED_PRECISION_LEVELS = 6
FACTORIZED_ACTION_COUNT = 30
PRUNING_MODE_OPACITY_BASELINE = "opacity_baseline"
FACTORIZED_PRUNING_MODES = (PRUNING_MODE_OPACITY_BASELINE,)
PRUNING_POLICY = "group_local_low_opacity_first"

GEO_SCALE_PREFIX = "scale_"
GEO_ROT_PREFIX = "rot_"
SH_DC_PREFIX = "f_dc_"
SH_REST_PREFIX = "f_rest_"


@dataclass(frozen=True)
class DecodedAction:
    """Read-only decoder result for protected original compact packages."""

    action_id: int
    pruning_level: int
    precision_level: int
    pruning_rate: float
    sh_degree: int
    sh_bit: int
    geo_bit: int


@dataclass(frozen=True)
class FactorizedAction:
    """Canonical independent pruning and precision levels."""

    pruning_level: int
    precision_level: int
    pruning_rate: float
    sh_degree: int
    sh_bit: int
    geo_bit: int


FactorizedActionInput = (
    FactorizedAction
    | tuple[int | float, int | float]
    | list[int | float]
    | np.ndarray
)


def decode_action(action: int | float | np.ndarray) -> DecodedAction:
    """Decode protected original compact metadata; not a training action API."""
    value = int(np.clip(round(float(np.asarray(action).reshape(-1)[0])), 0, 24))
    pruning_level, precision_level = divmod(value, 5)
    precision = PRECISION_PROFILES[precision_level]
    return DecodedAction(
        value, pruning_level, precision_level,
        float(PRUNING_RATES[pruning_level]),
        int(precision["sh_degree"]), int(precision["sh_bit"]),
        int(precision["geo_bit"]),
    )


def decode_factorized_action(action: FactorizedActionInput) -> FactorizedAction:
    """Round and clip the two action components independently."""
    values = (
        np.asarray([action.pruning_level, action.precision_level], dtype=np.float64)
        if isinstance(action, FactorizedAction)
        else np.asarray(action, dtype=np.float64)
    )
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("factorized action must be a finite shape-(2,) value")
    pruning_level = int(np.clip(round(float(values[0])), 0, 4))
    precision_level = int(np.clip(round(float(values[1])), 0, 5))
    precision = PRECISION_PROFILES_V2[precision_level]
    return FactorizedAction(
        pruning_level, precision_level, float(PRUNING_RATES[pruning_level]),
        int(precision["sh_degree"]), int(precision["sh_bit"]),
        int(precision["geo_bit"]),
    )


def encode_factorized_action(pruning_level: Any, precision_level: Any) -> int:
    """Return a categorical 0--29 storage ID, never an action-space metric."""
    action = decode_factorized_action((pruning_level, precision_level))
    return action.pruning_level * 6 + action.precision_level


def adjust_factorized_action(
    action: FactorizedActionInput, component: str, delta: int
) -> FactorizedAction:
    """Adjust exactly one discrete component and clip it to its own range."""
    if component not in {"pruning", "precision"}:
        raise ValueError("component must be 'pruning' or 'precision'")
    if isinstance(delta, (bool, np.bool_)) or not isinstance(delta, (int, np.integer)):
        raise ValueError("delta must be an integer")
    decoded = decode_factorized_action(action)
    values = [decoded.pruning_level, decoded.precision_level]
    values[0 if component == "pruning" else 1] += int(delta)
    return decode_factorized_action(values)


def factorized_action_to_array(action: FactorizedActionInput) -> np.ndarray:
    """Return ``[pruning_level, precision_level]`` as float32."""
    decoded = decode_factorized_action(action)
    return np.asarray(
        [decoded.pruning_level, decoded.precision_level], dtype=np.float32
    )


def sh_rest_count_for_degree(sh_degree: int) -> int:
    degree = int(np.clip(sh_degree, 0, 3))
    return 3 * ((degree + 1) ** 2 - 1)


def _field_names(vertex_data: np.ndarray) -> list[str]:
    return list(vertex_data.dtype.names or ())


def _sorted_indexed_fields(names: list[str], prefix: str) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for name in names:
        if not name.startswith(prefix):
            continue
        try:
            indexed.append((int(name[len(prefix):]), name))
        except ValueError:
            continue
    return [name for _, name in sorted(indexed)]


def geo_fields(vertex_data: np.ndarray) -> list[str]:
    names = _field_names(vertex_data)
    fields = [name for name in ("x", "y", "z", "opacity") if name in names]
    fields.extend(_sorted_indexed_fields(names, GEO_SCALE_PREFIX))
    fields.extend(_sorted_indexed_fields(names, GEO_ROT_PREFIX))
    return fields


def sh_dc_fields(vertex_data: np.ndarray) -> list[str]:
    return _sorted_indexed_fields(_field_names(vertex_data), SH_DC_PREFIX)


def sh_rest_fields(vertex_data: np.ndarray) -> list[str]:
    return _sorted_indexed_fields(_field_names(vertex_data), SH_REST_PREFIX)


def active_sh_fields(vertex_data: np.ndarray, sh_degree: int) -> list[str]:
    return sh_dc_fields(vertex_data) + sh_rest_fields(vertex_data)[
        :sh_rest_count_for_degree(sh_degree)
    ]


def _quantize_dequantize(values: np.ndarray, bits: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0 or bits >= 32:
        return array.copy()
    minimum, maximum = float(np.min(array)), float(np.max(array))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("quantized attributes must be finite")
    if maximum - minimum < 1e-12:
        return array.copy()
    levels = float((1 << int(bits)) - 1)
    quantized = np.rint((array - minimum) / (maximum - minimum) * levels)
    return (
        np.clip(quantized, 0.0, levels) / levels * (maximum - minimum) + minimum
    ).astype(np.float32)


def _apply_sh_degree(
    vertices: np.ndarray, indices: np.ndarray, sh_degree: int
) -> None:
    rest = sh_rest_fields(vertices)
    for field in rest[sh_rest_count_for_degree(sh_degree):]:
        vertices[field][indices] = 0.0


def _apply_uniform_quantization(
    vertices: np.ndarray, indices: np.ndarray, fields: list[str], bits: int
) -> None:
    for field in fields:
        vertices[field][indices] = _quantize_dequantize(vertices[field][indices], bits)


def _validate_factorized_group_indices(
    group_indices: list[np.ndarray], total_vertices: int
) -> list[np.ndarray]:
    if not isinstance(group_indices, list):
        raise ValueError("group_indices must be a list")
    seen = np.zeros(total_vertices, dtype=bool)
    validated: list[np.ndarray] = []
    for group_number, indices in enumerate(group_indices):
        array = np.asarray(indices)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"group_indices[{group_number}] must be a 1D integer array")
        normalized = array.astype(np.int64, copy=True)
        if (
            np.any(normalized < 0)
            or np.any(normalized >= total_vertices)
            or len(np.unique(normalized)) != len(normalized)
            or np.any(seen[normalized])
        ):
            raise ValueError("group indices are out of range or overlap")
        seen[normalized] = True
        validated.append(normalized)
    if total_vertices and not np.all(seen):
        raise ValueError("group_indices must cover every Gaussian exactly once")
    return validated


def _decode_actions(
    actions: list[FactorizedActionInput | None], group_count: int
) -> list[FactorizedAction]:
    if not isinstance(actions, list) or len(actions) > group_count:
        raise ValueError("factorized_actions must be a list no longer than the groups")
    return [
        decode_factorized_action(
            (0, 0) if index >= len(actions) or actions[index] is None else actions[index]
        )
        for index in range(group_count)
    ]


def factorized_action_histograms(
    actions: list[FactorizedActionInput]
) -> dict[str, Any]:
    decoded = [decode_factorized_action(action) for action in actions]

    def histogram(values: list[int], maximum: int) -> dict[str, int]:
        return {str(level): int(sum(value == level for value in values)) for level in range(maximum + 1)}

    pruning_levels = [action.pruning_level for action in decoded]
    precision_levels = [action.precision_level for action in decoded]
    storage_ids = [encode_factorized_action(*levels) for levels in zip(pruning_levels, precision_levels)]
    return {
        "action_mode": "factorized_v2_5x6",
        "storage_id_histogram": histogram(storage_ids, 29),
        "pruning_level_histogram": histogram(pruning_levels, 4),
        "precision_level_histogram": histogram(precision_levels, 5),
        "mean_pruning_level": float(np.mean(pruning_levels)) if decoded else 0.0,
        "mean_precision_level": float(np.mean(precision_levels)) if decoded else 0.0,
        "mean_pruning_rate": float(np.mean([action.pruning_rate for action in decoded])) if decoded else 0.0,
        "mean_sh_degree": float(np.mean([action.sh_degree for action in decoded])) if decoded else 0.0,
        "mean_sh_bit": float(np.mean([action.sh_bit for action in decoded])) if decoded else 0.0,
        "mean_geo_bit": float(np.mean([action.geo_bit for action in decoded])) if decoded else 0.0,
    }


def estimate_size_ratio_from_factorized_actions(
    group_indices: list[np.ndarray],
    actions: list[FactorizedActionInput | None],
    total_vertices: int,
) -> float:
    """Estimate compact bits, filling every undecided group with identity."""
    if isinstance(total_vertices, (bool, np.bool_)) or not isinstance(
        total_vertices, (int, np.integer)
    ) or int(total_vertices) < 0:
        raise ValueError("total_vertices must be a nonnegative integer")
    count = int(total_vertices)
    if count == 0:
        return 1.0
    groups = _validate_factorized_group_indices(group_indices, count)
    decoded = _decode_actions(actions, len(groups))
    total_bits = 0.0
    for indices, action in zip(groups, decoded):
        kept = len(indices) - int(np.floor(len(indices) * action.pruning_rate))
        sh_coefficients = 3 + sh_rest_count_for_degree(action.sh_degree)
        total_bits += kept * (
            11 * action.geo_bit + sh_coefficients * action.sh_bit
        )
    ratio = float(total_bits / (count * 32.0 * (11 + 48)))
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise RuntimeError("factorized size estimate must be finite and positive")
    return ratio


def apply_factorized_compression_to_vertices(
    vertex_data: np.ndarray,
    group_indices: list[np.ndarray],
    factorized_actions: list[FactorizedActionInput | None],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply identity fill, group-local opacity pruning, and V2 precision."""
    if not isinstance(vertex_data, np.ndarray) or vertex_data.dtype.names is None:
        raise ValueError("vertex_data must be a structured NumPy array")
    if "opacity" not in vertex_data.dtype.names:
        raise ValueError("formal opacity-baseline pruning requires an opacity field")
    if any(
        not np.all(np.isfinite(np.asarray(vertex_data[name], dtype=np.float64)))
        for name in vertex_data.dtype.names
    ):
        raise ValueError("vertex_data fields must be finite")
    vertices = vertex_data.copy()
    vertex_count = len(vertices)
    groups = _validate_factorized_group_indices(group_indices, vertex_count)
    actions = _decode_actions(factorized_actions, len(groups))
    keep_mask = np.ones(vertex_count, dtype=bool)
    group_ids = np.empty(vertex_count, dtype=np.int32)
    pruning_levels = np.zeros(vertex_count, dtype=np.uint8)
    precision_levels = np.zeros(vertex_count, dtype=np.uint8)
    storage_ids = np.zeros(vertex_count, dtype=np.uint8)

    raw_opacity = np.asarray(vertex_data["opacity"], dtype=np.float64)
    for group_id, (indices, action) in enumerate(zip(groups, actions)):
        group_ids[indices] = group_id
        pruning_levels[indices] = action.pruning_level
        precision_levels[indices] = action.precision_level
        storage_ids[indices] = encode_factorized_action(
            action.pruning_level, action.precision_level
        )
        prune_count = int(np.floor(len(indices) * action.pruning_rate))
        if prune_count:
            ordered = indices[np.lexsort((indices, raw_opacity[indices]))]
            keep_mask[ordered[:prune_count]] = False

    geometry_fields = geo_fields(vertices)
    for indices, action in zip(groups, actions):
        kept = indices[keep_mask[indices]]
        if not len(kept) or action.precision_level == 0:
            continue
        _apply_sh_degree(vertices, kept, action.sh_degree)
        if action.geo_bit < 32:
            _apply_uniform_quantization(vertices, kept, geometry_fields, action.geo_bit)
        if action.sh_bit < 32:
            _apply_uniform_quantization(
                vertices, kept, active_sh_fields(vertices, action.sh_degree), action.sh_bit
            )

    compressed = vertices[keep_mask].copy()
    kept_indices = np.nonzero(keep_mask)[0].astype(np.int64)
    stats: dict[str, Any] = {
        "action_mode": "factorized_v2_5x6",
        "original_vertices": vertex_count,
        "processed_groups": len(groups),
        "decided_group_count": len(factorized_actions),
        "identity_filled_group_count": len(groups) - len(factorized_actions),
        "total_group_count": len(groups),
        "pruned_vertices": vertex_count - len(compressed),
        "kept_vertices": len(compressed),
        "kept_vertex_ratio": float(len(compressed) / max(vertex_count, 1)),
        "estimated_size_ratio": estimate_size_ratio_from_factorized_actions(
            groups, actions, vertex_count
        ),
        "pruning_mode": PRUNING_MODE_OPACITY_BASELINE,
        "pruning_policy": PRUNING_POLICY if len(compressed) < vertex_count else "no_pruning_requested",
        "pruning_is_multiview": False,
        "pruning_uses_transmittance": False,
        "pruning_uses_background_replaceability": False,
        **factorized_action_histograms(actions),
        "_compact_aux": {
            "kept_original_indices": kept_indices,
            "kept_group_ids": group_ids[keep_mask].astype(np.int32, copy=True),
            "kept_pruning_levels": pruning_levels[keep_mask].astype(np.uint8, copy=True),
            "kept_precision_levels": precision_levels[keep_mask].astype(np.uint8, copy=True),
            "kept_storage_ids": storage_ids[keep_mask].astype(np.uint8, copy=True),
        },
    }
    return compressed, stats


_FIRST_VERSION_COMPRESSION_REPORT: dict[str, Any] = {}


def validate_first_version_compression_ops() -> bool:
    """Validate all 30 actions, identity, opacity pruning, and size estimates."""
    if _FIRST_VERSION_COMPRESSION_REPORT.get("validated") is True:
        return True

    def require(condition: Any, message: str) -> None:
        if not bool(condition):
            raise AssertionError(message)

    identity = decode_factorized_action((0, 0))
    require(
        (identity.pruning_rate, identity.sh_degree, identity.sh_bit, identity.geo_bit)
        == (0.0, 3, 32, 32),
        "identity profile changed",
    )
    ids = [encode_factorized_action(p, q) for p in range(5) for q in range(6)]
    require(ids == list(range(30)), "factorized storage IDs are invalid")
    require(adjust_factorized_action((0, 5), "pruning", 1).precision_level == 5, "pruning adjustment changed precision")
    require(adjust_factorized_action((3, 0), "precision", 1).pruning_level == 3, "precision adjustment changed pruning")

    names = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2", *[f"f_rest_{index}" for index in range(45)]]
    vertices = np.zeros(20, dtype=np.dtype([(name, "<f4") for name in names]))
    for index, name in enumerate(names):
        vertices[name] = np.linspace(index, index + 1.0, 20, dtype=np.float32)
    vertices["opacity"] = np.asarray([0.5, 0.1, 0.1, 0.2, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3] * 2, dtype=np.float32)
    groups = [np.arange(10, dtype=np.int64), np.arange(10, 20, dtype=np.int64)]
    identity_vertices, identity_stats = apply_factorized_compression_to_vertices(vertices, groups, [])
    require(np.array_equal(identity_vertices, vertices), "identity fill changed vertices")
    compressed, stats = apply_factorized_compression_to_vertices(vertices, groups, [(4, 5), (0, 0)])
    kept = stats["_compact_aux"]["kept_original_indices"]
    require(not np.any(np.isin(np.asarray([1, 2, 3]), kept)), "lowest-opacity vertices were not pruned deterministically")
    require(np.array_equal(compressed["opacity"][np.isin(kept, np.arange(10, 20))], vertices["opacity"][10:20]), "identity group changed")
    require(np.all(compressed["f_rest_44"][np.isin(kept, np.arange(0, 10))] == 0.0), "precision SH degree was not applied")
    ratio = estimate_size_ratio_from_factorized_actions(groups, [(4, 5)], len(vertices))
    require(ratio == estimate_size_ratio_from_factorized_actions(groups, [(4, 5)], len(vertices)), "size estimate is not deterministic")
    require(0.0 < ratio < 1.0, "size estimate is invalid")

    _FIRST_VERSION_COMPRESSION_REPORT.update({
        "validated": True,
        "action_combinations": 30,
        "identity_fill": identity_stats["identity_filled_group_count"] == 2,
        "opacity_pruning": True,
        "precision_application": True,
        "size_estimate": ratio,
        "multiview_removed": True,
    })
    return True


def validate_factorized_action_space() -> bool:
    """Compatibility validation name for the formal compression suite."""
    return validate_first_version_compression_ops()


def validate_factorized_compression_execution() -> bool:
    """Compatibility validation name for the formal compression suite."""
    return validate_first_version_compression_ops()
