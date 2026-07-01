from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ply_utils import numbered_fields


GROUP_FEATURE_NAMES = [
    "num_gaussians_ratio",
    "opacity_mean",
    "opacity_std",
    "opacity_low_ratio",
    "scale_mean",
    "scale_std",
    "sh_energy_mean",
    "xyz_extent_mean",
    "small_group_flag",
]

GLOBAL_FEATURE_NAMES = [
    "current_group_idx_ratio",
    "remaining_group_ratio",
    "current_estimated_size_ratio",
    "target_size_ratio",
    "previous_action_mean",
    "previous_action_last",
]

STATE_FEATURE_NAMES = GROUP_FEATURE_NAMES + GLOBAL_FEATURE_NAMES


@dataclass(frozen=True)
class VoxelGroups:
    group_indices: list[np.ndarray]
    cell_ids: list[int]
    small_group_flags: list[bool]
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    grid_size: int


def voxel_group_indices(
    xyz: np.ndarray,
    grid_size: int = 4,
    min_group_size: int = 10,
    max_groups: int | None = None,
) -> VoxelGroups:
    """Split Gaussian xyz coordinates into non-empty voxel groups."""
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N, 3], got {xyz.shape}.")
    if len(xyz) == 0:
        raise ValueError("Cannot group an empty xyz array.")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")

    finite_mask = np.isfinite(xyz).all(axis=1)
    if not finite_mask.all():
        xyz = xyz.copy()
        replacement = np.nanmean(xyz[finite_mask], axis=0) if finite_mask.any() else 0.0
        xyz[~finite_mask] = replacement

    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    extent = np.maximum(bbox_max - bbox_min, 1e-8)
    normalized = (xyz - bbox_min) / extent
    coords = np.floor(normalized * grid_size).astype(np.int32)
    coords = np.clip(coords, 0, grid_size - 1)
    cell_ids = (
        coords[:, 0] * grid_size * grid_size + coords[:, 1] * grid_size + coords[:, 2]
    )

    groups: list[np.ndarray] = []
    ids: list[int] = []
    small_flags: list[bool] = []
    for cell_id in sorted(np.unique(cell_ids).tolist()):
        indices = np.flatnonzero(cell_ids == cell_id).astype(np.int64)
        groups.append(indices)
        ids.append(int(cell_id))
        small_flags.append(len(indices) < min_group_size)

    if max_groups is not None:
        groups = groups[:max_groups]
        ids = ids[:max_groups]
        small_flags = small_flags[:max_groups]

    return VoxelGroups(
        group_indices=groups,
        cell_ids=ids,
        small_group_flags=small_flags,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        grid_size=grid_size,
    )


def _field_array(vertex_data: np.ndarray, field: str, indices: np.ndarray) -> np.ndarray:
    if field not in vertex_data.dtype.names:
        return np.zeros(len(indices), dtype=np.float32)
    return vertex_data[field][indices].astype(np.float32, copy=False)


def _stack_existing_fields(
    vertex_data: np.ndarray,
    fields: list[str],
    indices: np.ndarray,
) -> np.ndarray:
    existing = [field for field in fields if field in vertex_data.dtype.names]
    if not existing:
        return np.zeros((len(indices), 0), dtype=np.float32)
    return np.column_stack(
        [vertex_data[field][indices].astype(np.float32, copy=False) for field in existing]
    )


def extract_group_features(
    vertex_data: np.ndarray,
    indices: np.ndarray,
    total_gaussians: int,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    opacity_low_threshold: float = 0.01,
    small_group_flag: bool = False,
) -> np.ndarray:
    """Return the first-version group feature vector for one voxel group."""
    if len(indices) == 0:
        raise ValueError("Cannot extract features from an empty group.")

    xyz = _stack_existing_fields(vertex_data, ["x", "y", "z"], indices)
    opacity = _field_array(vertex_data, "opacity", indices)
    scale_values = _stack_existing_fields(
        vertex_data, ["scale_0", "scale_1", "scale_2"], indices
    )
    sh_fields = ["f_dc_0", "f_dc_1", "f_dc_2"] + numbered_fields(
        vertex_data.dtype.names or [], "f_rest"
    )
    sh_values = _stack_existing_fields(vertex_data, sh_fields, indices)

    scene_extent = np.maximum(np.asarray(bbox_max) - np.asarray(bbox_min), 1e-8)
    if xyz.shape[1] == 3:
        xyz_extent = (xyz.max(axis=0) - xyz.min(axis=0)) / scene_extent
        xyz_extent_mean = float(np.mean(xyz_extent))
    else:
        xyz_extent_mean = 0.0

    if scale_values.size:
        flat_scale = scale_values.reshape(-1)
        scale_mean = float(np.mean(flat_scale))
        scale_std = float(np.std(flat_scale))
    else:
        scale_mean = 0.0
        scale_std = 0.0

    if sh_values.size:
        sh_energy_mean = float(np.mean(np.sum(np.square(sh_values), axis=1)))
    else:
        sh_energy_mean = 0.0

    features = np.array(
        [
            len(indices) / max(float(total_gaussians), 1.0),
            float(np.mean(opacity)),
            float(np.std(opacity)),
            float(np.mean(opacity < opacity_low_threshold)),
            scale_mean,
            scale_std,
            sh_energy_mean,
            xyz_extent_mean,
            1.0 if small_group_flag else 0.0,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def build_state_vector(
    group_features: np.ndarray,
    current_group_idx: int,
    total_groups: int,
    current_estimated_size_ratio: float,
    target_size_ratio: float,
    previous_actions: list[int],
) -> np.ndarray:
    previous_action_mean = (
        float(np.mean(previous_actions)) / 4.0 if previous_actions else 0.0
    )
    previous_action_last = float(previous_actions[-1]) / 4.0 if previous_actions else 0.0
    total_groups = max(total_groups, 1)
    global_features = np.array(
        [
            current_group_idx / float(total_groups),
            (total_groups - current_group_idx) / float(total_groups),
            current_estimated_size_ratio,
            target_size_ratio,
            previous_action_mean,
            previous_action_last,
        ],
        dtype=np.float32,
    )
    state = np.concatenate([group_features.astype(np.float32), global_features])
    return np.nan_to_num(state.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
