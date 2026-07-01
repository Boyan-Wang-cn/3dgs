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
    requested_grid_size: int | None = None
    target_num_groups: int | None = None
    natural_group_count: int = 0
    actual_group_count: int = 0
    truncated_by_max_groups: bool = False
    max_groups: int | None = None


def _sanitize_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N, 3], got {xyz.shape}.")
    if len(xyz) == 0:
        raise ValueError("Cannot group an empty xyz array.")

    finite_mask = np.isfinite(xyz).all(axis=1)
    if not finite_mask.all():
        xyz = xyz.copy()
        replacement = np.nanmean(xyz[finite_mask], axis=0) if finite_mask.any() else 0.0
        xyz[~finite_mask] = replacement
    return xyz


def _cell_ids_for_grid(normalized_xyz: np.ndarray, grid_size: int) -> np.ndarray:
    coords = np.floor(normalized_xyz * grid_size).astype(np.int32)
    coords = np.clip(coords, 0, grid_size - 1)
    return coords[:, 0] * grid_size * grid_size + coords[:, 1] * grid_size + coords[:, 2]


def _cell_id_to_coord(cell_id: int, grid_size: int) -> tuple[int, int, int]:
    x = int(cell_id // (grid_size * grid_size))
    y = int((cell_id // grid_size) % grid_size)
    z = int(cell_id % grid_size)
    return x, y, z


def _part1by2(n: int) -> int:
    """Spread lower 21 bits of n so three coordinates can be interleaved."""
    n &= 0x1FFFFF
    n = (n | (n << 32)) & 0x1F00000000FFFF
    n = (n | (n << 16)) & 0x1F0000FF0000FF
    n = (n | (n << 8)) & 0x100F00F00F00F00F
    n = (n | (n << 4)) & 0x10C30C30C30C30C3
    n = (n | (n << 2)) & 0x1249249249249249
    return int(n)


def _morton3d(coord: tuple[int, int, int]) -> int:
    x, y, z = coord
    return _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(z) << 2)


def _count_nonempty_voxels(normalized_xyz: np.ndarray, grid_size: int) -> int:
    cell_ids = _cell_ids_for_grid(normalized_xyz, grid_size)
    return int(len(np.unique(cell_ids)))


def _build_voxel_groups(
    normalized_xyz: np.ndarray,
    grid_size: int,
    min_group_size: int,
) -> tuple[list[np.ndarray], list[int], list[bool], list[tuple[int, int, int]]]:
    cell_ids_per_point = _cell_ids_for_grid(normalized_xyz, grid_size)
    unique_ids = sorted(np.unique(cell_ids_per_point).tolist())

    groups: list[np.ndarray] = []
    ids: list[int] = []
    small_flags: list[bool] = []
    coords: list[tuple[int, int, int]] = []
    for cell_id in unique_ids:
        indices = np.flatnonzero(cell_ids_per_point == cell_id).astype(np.int64)
        groups.append(indices)
        ids.append(int(cell_id))
        small_flags.append(len(indices) < min_group_size)
        coords.append(_cell_id_to_coord(int(cell_id), grid_size))
    return groups, ids, small_flags, coords


def _select_grid_size_for_target(
    normalized_xyz: np.ndarray,
    target_num_groups: int,
    requested_grid_size: int,
    max_search_grid_size: int,
) -> int:
    if target_num_groups <= 0:
        raise ValueError("target_num_groups must be positive when provided.")
    if max_search_grid_size <= 0:
        raise ValueError("max_search_grid_size must be positive.")
    max_search_grid_size = max(int(max_search_grid_size), int(requested_grid_size), 1)

    counts: list[tuple[int, int]] = []
    for candidate_grid in range(1, max_search_grid_size + 1):
        count = _count_nonempty_voxels(normalized_xyz, candidate_grid)
        counts.append((candidate_grid, count))
        # Once the target has been reached, the smallest such grid is usually
        # the most spatially coherent choice. Stop early to avoid over-fragmenting.
        if count >= target_num_groups:
            return candidate_grid

    # If even the largest search grid does not reach the target, use the closest
    # available grid instead of truncating or duplicating groups.
    return min(counts, key=lambda item: abs(item[1] - target_num_groups))[0]


def _merge_groups_to_target(
    groups: list[np.ndarray],
    cell_ids: list[int],
    coords: list[tuple[int, int, int]],
    target_num_groups: int,
    min_group_size: int,
) -> tuple[list[np.ndarray], list[int], list[bool]]:
    """Merge neighboring voxel cells into exactly target_num_groups groups.

    The input cells are ordered by Morton code before merging. This keeps
    nearby cells close in the 1-D ordering and avoids the formal experiment
    relying on max_groups-style truncation, which would drop parts of a scene.
    """
    if target_num_groups <= 0:
        raise ValueError("target_num_groups must be positive.")
    if len(groups) <= target_num_groups:
        small_flags = [len(indices) < min_group_size for indices in groups]
        return groups, cell_ids, small_flags

    order = sorted(range(len(groups)), key=lambda idx: (_morton3d(coords[idx]), cell_ids[idx]))
    sorted_groups = [groups[idx] for idx in order]
    sorted_ids = [cell_ids[idx] for idx in order]
    counts = np.asarray([len(indices) for indices in sorted_groups], dtype=np.int64)
    cumulative = np.cumsum(counts)
    total_points = int(cumulative[-1])

    boundaries: list[int] = []
    prev = 0
    num_cells = len(sorted_groups)
    for k in range(1, target_num_groups):
        desired = total_points * k / float(target_num_groups)
        boundary = int(np.searchsorted(cumulative, desired, side="right")) + 1
        # Keep every merged group non-empty.
        boundary = max(boundary, prev + 1)
        # Leave enough cells for the remaining groups.
        boundary = min(boundary, num_cells - (target_num_groups - k))
        boundaries.append(boundary)
        prev = boundary

    starts = [0] + boundaries
    ends = boundaries + [num_cells]

    merged_groups: list[np.ndarray] = []
    merged_ids: list[int] = []
    small_flags: list[bool] = []
    for group_id, (start, end) in enumerate(zip(starts, ends)):
        merged = np.concatenate(sorted_groups[start:end]).astype(np.int64, copy=False)
        merged.sort()
        merged_groups.append(merged)
        # Use compact IDs for merged super-voxels. The original voxel cells are
        # still spatially ordered through the Morton ordering above.
        merged_ids.append(int(group_id))
        small_flags.append(len(merged) < min_group_size)

    return merged_groups, merged_ids, small_flags


def voxel_group_indices(
    xyz: np.ndarray,
    grid_size: int = 4,
    min_group_size: int = 10,
    max_groups: int | None = None,
    target_num_groups: int | None = None,
    max_search_grid_size: int = 32,
) -> VoxelGroups:
    """Split Gaussian xyz coordinates into voxel groups.

    If target_num_groups is provided, the function first chooses a voxel grid
    resolution that produces at least that many non-empty cells when possible,
    then merges Morton-ordered neighboring cells to obtain approximately or
    exactly target_num_groups groups. max_groups is kept only as a debug-time
    truncation switch and should be None for formal experiments.
    """
    xyz = _sanitize_xyz(xyz)
    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")
    if min_group_size <= 0:
        raise ValueError("min_group_size must be positive.")
    if max_groups is not None and max_groups <= 0:
        raise ValueError("max_groups must be positive when provided.")

    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    extent = np.maximum(bbox_max - bbox_min, 1e-8)
    normalized = (xyz - bbox_min) / extent

    requested_grid_size = int(grid_size)
    selected_grid_size = requested_grid_size
    if target_num_groups is not None:
        target_num_groups = int(target_num_groups)
        selected_grid_size = _select_grid_size_for_target(
            normalized,
            target_num_groups=target_num_groups,
            requested_grid_size=requested_grid_size,
            max_search_grid_size=int(max_search_grid_size),
        )

    groups, ids, small_flags, coords = _build_voxel_groups(
        normalized,
        grid_size=selected_grid_size,
        min_group_size=min_group_size,
    )
    natural_group_count = len(groups)

    if target_num_groups is not None and len(groups) > target_num_groups:
        groups, ids, small_flags = _merge_groups_to_target(
            groups,
            ids,
            coords,
            target_num_groups=target_num_groups,
            min_group_size=min_group_size,
        )

    truncated_by_max_groups = False
    if max_groups is not None and len(groups) > max_groups:
        groups = groups[:max_groups]
        ids = ids[:max_groups]
        small_flags = small_flags[:max_groups]
        truncated_by_max_groups = True

    return VoxelGroups(
        group_indices=groups,
        cell_ids=ids,
        small_group_flags=small_flags,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        grid_size=selected_grid_size,
        requested_grid_size=requested_grid_size,
        target_num_groups=target_num_groups,
        natural_group_count=natural_group_count,
        actual_group_count=len(groups),
        truncated_by_max_groups=truncated_by_max_groups,
        max_groups=max_groups,
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
