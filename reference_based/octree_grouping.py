"""Deterministic best-first spatial octree grouping for factorized 3DGS RL.

The octree supplies hierarchical, spatially local, identity-stable execution
units.  Best-first splitting refines Gaussian-dense regions first, while sparse
regions remain at shallow depths instead of creating the many empty cells of a
uniform voxel grid.  A group only defines the scope of one RL action; it does
not claim that all Gaussians in that group have identical contribution.

The first research version still removes low-opacity Gaussians first *inside*
each already-defined group.  Opacity, SH coefficients, scale, actions, rewards,
quality scores, and storage IDs never participate in grouping.  Whole-scene
quality reward constrains the effects of heterogeneous group actions.  The
octree itself does not solve every occlusion or replaceability question, and
this module intentionally implements no GNN, co-projection edges, renderer
contribution, or dynamic regrouping.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import numpy as np


OCTREE_GROUPING_METHOD = "deterministic_best_first_octree_v1"
OCTREE_CHILD_ORDER = "xyz_binary_morton_0_to_7"
OCTREE_CENTER_TIE_RULE = "coordinate_equal_center_goes_to_high_child"
OCTREE_LEAF_ORDER = "max_depth_morton_key_then_depth_then_node_id"
OCTREE_TARGET_POLICY = "largest_leaf_first_without_target_overshoot"
OCTREE_MIN_SIZE_POLICY = "all_nonempty_children_at_least_min_group_size"

_MAX_TARGET_NUM_GROUPS = 2**31
_MAX_OCTREE_DEPTH = 20


def _strict_integer(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Return a strict bounded integer, rejecting Python and NumPy booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a strict integer")
    normalized = int(value)
    if normalized < minimum or (maximum is not None and normalized > maximum):
        upper = "" if maximum is None else f" and at most {maximum}"
        raise ValueError(f"{name} must be at least {minimum}{upper}")
    return normalized


def _readonly_float_array(
    value: Any,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Copy one finite float64 array and make it immutable."""
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real array") from exc
    if source.dtype.kind in {"b", "O", "c", "S", "U", "V", "M", "m"}:
        raise ValueError(f"{name} must be a finite real array")
    try:
        array = np.array(source, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real array") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and be finite")
    array.setflags(write=False)
    return array


def _readonly_indices(value: Any, name: str = "indices") -> np.ndarray:
    """Copy one strictly increasing one-dimensional integer index array."""
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional integer array") from exc
    if source.ndim != 1 or source.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if source.size == 0:
        raise ValueError(f"{name} must not be empty")
    try:
        indices = np.array(source, dtype=np.int64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} cannot be represented as int64") from exc
    if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
        raise ValueError(f"{name} must be nonnegative and strictly increasing")
    indices.setflags(write=False)
    return indices


def _midpoint(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    """Compute a finite midpoint without overflowing a same-sign sum."""
    return bbox_min * 0.5 + bbox_max * 0.5


@dataclass(frozen=True)
class OctreeNode:
    """One immutable nonempty octree node using original global row indices."""

    indices: np.ndarray
    depth: int
    path_code: int
    node_id: int
    parent_id: int
    child_id: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    center: np.ndarray
    point_count: int
    morton_key: int

    def __post_init__(self) -> None:
        indices = _readonly_indices(self.indices)
        depth = _strict_integer(
            self.depth, "depth", minimum=0, maximum=_MAX_OCTREE_DEPTH
        )
        path_code = _strict_integer(self.path_code, "path_code", minimum=0)
        node_id = _strict_integer(self.node_id, "node_id", minimum=1)
        parent_id = _strict_integer(self.parent_id, "parent_id", minimum=0)
        child_id = _strict_integer(self.child_id, "child_id", minimum=-1, maximum=7)
        point_count = _strict_integer(self.point_count, "point_count", minimum=1)
        morton_key = _strict_integer(self.morton_key, "morton_key", minimum=0)
        bbox_min = _readonly_float_array(self.bbox_min, "bbox_min", (3,))
        bbox_max = _readonly_float_array(self.bbox_max, "bbox_max", (3,))
        center = _readonly_float_array(self.center, "center", (3,))

        if not np.all(bbox_max > bbox_min):
            raise ValueError("node bbox_max must be greater than bbox_min on every axis")
        node_extent = bbox_max - bbox_min
        if not np.array_equal(
            node_extent,
            np.full(3, node_extent[0], dtype=np.float64),
        ):
            raise ValueError("an octree node must be a bitwise-equal-sided cube")
        expected_center = _midpoint(bbox_min, bbox_max)
        if not np.array_equal(center, expected_center):
            raise ValueError("node center must equal the bounding-box midpoint")
        if point_count != len(indices):
            raise ValueError("point_count must equal len(indices)")
        if path_code >= (1 << (3 * depth)):
            raise ValueError("path_code does not fit the node depth")
        expected_node_id = (1 << (3 * depth)) | path_code
        if node_id != expected_node_id:
            raise ValueError("node_id is inconsistent with depth and path_code")
        if depth == 0:
            if path_code != 0 or node_id != 1 or parent_id != 0 or child_id != -1:
                raise ValueError("root identity fields are inconsistent")
        else:
            expected_parent = (1 << (3 * (depth - 1))) | (path_code // 8)
            if parent_id != expected_parent or child_id != path_code % 8:
                raise ValueError("child identity fields are inconsistent")
        if morton_key != path_code:
            raise ValueError("node morton_key must store its unpadded path_code")

        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "path_code", path_code)
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "parent_id", parent_id)
        object.__setattr__(self, "child_id", child_id)
        object.__setattr__(self, "bbox_min", bbox_min)
        object.__setattr__(self, "bbox_max", bbox_max)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "point_count", point_count)
        object.__setattr__(self, "morton_key", morton_key)


@dataclass(frozen=True)
class OctreeGroups:
    """Immutable compatibility and octree metadata for the final leaf partition."""

    group_indices: tuple[np.ndarray, ...]
    cell_ids: tuple[int, ...]
    small_group_flags: tuple[bool, ...]
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    grid_size: int
    requested_grid_size: int | None
    target_num_groups: int
    natural_group_count: int
    actual_group_count: int
    truncated_by_max_groups: bool
    max_groups: int | None
    grouping_method: str
    node_ids: tuple[int, ...]
    node_depths: tuple[int, ...]
    node_path_codes: tuple[int, ...]
    node_parent_ids: tuple[int, ...]
    node_child_ids: tuple[int, ...]
    node_bbox_mins: np.ndarray
    node_bbox_maxs: np.ndarray
    root_cube_min: np.ndarray
    root_cube_max: np.ndarray
    max_depth: int
    max_leaf_depth: int
    reached_target: bool
    target_gap: int
    split_count: int
    unsplittable_leaf_count: int
    min_group_size: int
    leaf_order: str
    center_tie_rule: str
    target_policy: str
    min_size_policy: str

    def __post_init__(self) -> None:
        groups = tuple(
            _readonly_indices(indices, f"group_indices[{index}]")
            for index, indices in enumerate(self.group_indices)
        )
        actual = _strict_integer(self.actual_group_count, "actual_group_count", minimum=1)
        target = _strict_integer(
            self.target_num_groups,
            "target_num_groups",
            minimum=1,
            maximum=_MAX_TARGET_NUM_GROUPS,
        )
        natural = _strict_integer(self.natural_group_count, "natural_group_count", minimum=1)
        grid_size = _strict_integer(self.grid_size, "grid_size", minimum=1)
        max_depth = _strict_integer(
            self.max_depth, "max_depth", minimum=0, maximum=_MAX_OCTREE_DEPTH
        )
        max_leaf_depth = _strict_integer(
            self.max_leaf_depth,
            "max_leaf_depth",
            minimum=0,
            maximum=max_depth,
        )
        min_group_size = _strict_integer(
            self.min_group_size, "min_group_size", minimum=1
        )
        split_count = _strict_integer(self.split_count, "split_count", minimum=0)
        unsplittable = _strict_integer(
            self.unsplittable_leaf_count,
            "unsplittable_leaf_count",
            minimum=0,
            maximum=actual,
        )
        target_gap = _strict_integer(self.target_gap, "target_gap", minimum=0)
        cell_ids = tuple(
            _strict_integer(value, f"cell_ids[{index}]", minimum=1)
            for index, value in enumerate(self.cell_ids)
        )
        node_ids = tuple(
            _strict_integer(value, f"node_ids[{index}]", minimum=1)
            for index, value in enumerate(self.node_ids)
        )
        node_depths = tuple(
            _strict_integer(
                value,
                f"node_depths[{index}]",
                minimum=0,
                maximum=max_depth,
            )
            for index, value in enumerate(self.node_depths)
        )
        node_path_codes = tuple(
            _strict_integer(value, f"node_path_codes[{index}]", minimum=0)
            for index, value in enumerate(self.node_path_codes)
        )
        node_parent_ids = tuple(
            _strict_integer(value, f"node_parent_ids[{index}]", minimum=0)
            for index, value in enumerate(self.node_parent_ids)
        )
        node_child_ids = tuple(
            _strict_integer(
                value, f"node_child_ids[{index}]", minimum=-1, maximum=7
            )
            for index, value in enumerate(self.node_child_ids)
        )
        flags = tuple(self.small_group_flags)
        if any(not isinstance(value, (bool, np.bool_)) for value in flags):
            raise ValueError("small_group_flags must contain only booleans")
        flags = tuple(bool(value) for value in flags)

        metadata_lengths = (
            len(groups), len(cell_ids), len(flags), len(node_ids), len(node_depths),
            len(node_path_codes), len(node_parent_ids), len(node_child_ids),
        )
        if any(length != actual for length in metadata_lengths):
            raise ValueError("leaf metadata lengths must equal actual_group_count")
        if cell_ids != node_ids:
            raise ValueError("cell_ids must exactly equal node_ids")
        if natural != actual:
            raise ValueError("natural_group_count must equal actual_group_count")
        if target_gap != target - actual or target_gap < 0:
            raise ValueError("target_gap must equal target_num_groups-actual_group_count")
        if self.reached_target is not (actual == target):
            raise ValueError("reached_target is inconsistent with group counts")
        if not isinstance(self.truncated_by_max_groups, bool) or self.truncated_by_max_groups:
            raise ValueError("octree grouping must never be truncated")
        if self.max_groups is not None or self.requested_grid_size is not None:
            raise ValueError("octree compatibility max/requested grid fields must be None")
        if grid_size != 2**max_leaf_depth:
            raise ValueError("grid_size must equal 2**max_leaf_depth")
        if actual > target:
            raise ValueError("actual_group_count cannot exceed target_num_groups")
        if self.grouping_method != OCTREE_GROUPING_METHOD:
            raise ValueError("grouping_method is invalid")
        if self.leaf_order != OCTREE_LEAF_ORDER:
            raise ValueError("leaf_order is invalid")
        if self.center_tie_rule != OCTREE_CENTER_TIE_RULE:
            raise ValueError("center_tie_rule is invalid")
        if self.target_policy != OCTREE_TARGET_POLICY:
            raise ValueError("target_policy is invalid")
        if self.min_size_policy != OCTREE_MIN_SIZE_POLICY:
            raise ValueError("min_size_policy is invalid")

        bbox_min = _readonly_float_array(self.bbox_min, "bbox_min", (3,))
        bbox_max = _readonly_float_array(self.bbox_max, "bbox_max", (3,))
        if np.any(bbox_max < bbox_min):
            raise ValueError("scene bbox_max cannot be below bbox_min")
        root_min = _readonly_float_array(self.root_cube_min, "root_cube_min", (3,))
        root_max = _readonly_float_array(self.root_cube_max, "root_cube_max", (3,))
        if not np.all(root_max > root_min):
            raise ValueError("root cube must have positive extent on every axis")
        root_extent = root_max - root_min
        if not np.all(np.isfinite(root_extent)) or not np.array_equal(
            root_extent,
            np.full(3, root_extent[0], dtype=np.float64),
        ):
            raise ValueError("root cube must have bitwise-identical float64 side lengths")
        if not np.array_equal(_midpoint(root_min, root_max), _midpoint(bbox_min, bbox_max)):
            raise ValueError("root cube center must equal the scene bounding-box center")
        if np.any(bbox_min < root_min) or np.any(bbox_max > root_max):
            raise ValueError("root cube must contain the scene bounding box")
        node_mins = _readonly_float_array(
            self.node_bbox_mins, "node_bbox_mins", (actual, 3)
        )
        node_maxs = _readonly_float_array(
            self.node_bbox_maxs, "node_bbox_maxs", (actual, 3)
        )
        if not np.all(node_maxs > node_mins):
            raise ValueError("all leaf bounding boxes must have positive extent")
        for leaf_index, leaf_extent in enumerate(node_maxs - node_mins):
            if not np.array_equal(
                leaf_extent,
                np.full(3, leaf_extent[0], dtype=np.float64),
            ):
                raise ValueError(
                    f"leaf {leaf_index} must have bitwise-identical side lengths"
                )

        object.__setattr__(self, "group_indices", groups)
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "small_group_flags", flags)
        object.__setattr__(self, "bbox_min", bbox_min)
        object.__setattr__(self, "bbox_max", bbox_max)
        object.__setattr__(self, "grid_size", grid_size)
        object.__setattr__(self, "target_num_groups", target)
        object.__setattr__(self, "natural_group_count", natural)
        object.__setattr__(self, "actual_group_count", actual)
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "node_depths", node_depths)
        object.__setattr__(self, "node_path_codes", node_path_codes)
        object.__setattr__(self, "node_parent_ids", node_parent_ids)
        object.__setattr__(self, "node_child_ids", node_child_ids)
        object.__setattr__(self, "node_bbox_mins", node_mins)
        object.__setattr__(self, "node_bbox_maxs", node_maxs)
        object.__setattr__(self, "root_cube_min", root_min)
        object.__setattr__(self, "root_cube_max", root_max)
        object.__setattr__(self, "max_depth", max_depth)
        object.__setattr__(self, "max_leaf_depth", max_leaf_depth)
        object.__setattr__(self, "reached_target", bool(self.reached_target))
        object.__setattr__(self, "target_gap", target_gap)
        object.__setattr__(self, "split_count", split_count)
        object.__setattr__(self, "unsplittable_leaf_count", unsplittable)
        object.__setattr__(self, "min_group_size", min_group_size)


def _validate_octree_xyz(xyz: Any) -> np.ndarray:
    """Return an owned finite float64 C-contiguous ``[N,3]`` coordinate copy."""
    try:
        source = np.asarray(xyz)
    except (TypeError, ValueError) as exc:
        raise ValueError("xyz must be a finite real array with shape [N, 3]") from exc
    if source.ndim != 2 or source.shape[1:] != (3,) or source.shape[0] == 0:
        raise ValueError(f"xyz must have nonempty shape [N, 3], got {source.shape}")
    if source.dtype.kind in {"b", "O", "c", "S", "U", "V", "M", "m"}:
        raise ValueError("xyz must contain real numeric values, not bool/object/complex")
    try:
        validated = np.array(source, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("xyz must be representable as float64") from exc
    if not np.all(np.isfinite(validated)):
        raise ValueError("xyz must contain only finite values")
    return validated


def _build_root_cube(
    xyz: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a finite scene-centred cube with bitwise-equal float64 sides.

    One common power-of-two half-side is used on every axis.  The starting
    scale covers both the scene extent and the largest ULP at the scene
    center.  If float64 rounding does not preserve exact side and center
    equality, the common scale is doubled; no axis is repaired independently.
    """
    coordinates = np.asarray(xyz, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,) or len(coordinates) == 0:
        raise ValueError("xyz must have nonempty shape [N, 3]")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("xyz must contain only finite values")
    scene_min = np.min(coordinates, axis=0).astype(np.float64, copy=True)
    scene_max = np.max(coordinates, axis=0).astype(np.float64, copy=True)
    center = _midpoint(scene_min, scene_max)
    with np.errstate(over="ignore", invalid="ignore"):
        extent = scene_max - scene_min
    desired_side = float(np.max(extent))
    if not np.isfinite(desired_side):
        raise ValueError("xyz range is too large for a finite float64 root cube")
    with np.errstate(over="ignore", invalid="ignore"):
        center_spacing = np.abs(np.spacing(center))
    if not np.all(np.isfinite(center_spacing)):
        raise ValueError("scene center has no finite float64 spacing")
    required_half_side = max(
        0.5 if desired_side == 0.0 else desired_side * 0.5,
        float(np.max(center_spacing)),
    )
    mantissa, exponent = np.frexp(np.float64(required_half_side))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        half_side = np.ldexp(
            np.float64(1.0),
            int(exponent - 1 if mantissa == 0.5 else exponent),
        )
    if not np.isfinite(half_side) or half_side <= 0.0:
        raise ValueError("required root cube side is not representable as float64")

    for _ in range(64):
        with np.errstate(over="ignore", invalid="ignore"):
            cube_min = center - half_side
            cube_max = center + half_side
            cube_extent = cube_max - cube_min
            cube_center = _midpoint(cube_min, cube_max)
        valid = (
            np.all(np.isfinite(cube_min))
            and np.all(np.isfinite(cube_max))
            and np.all(np.isfinite(cube_extent))
            and np.all(cube_max > cube_min)
            and np.all(coordinates >= cube_min)
            and np.all(coordinates <= cube_max)
            and np.array_equal(
                cube_extent,
                np.full(3, cube_extent[0], dtype=np.float64),
            )
            and np.array_equal(cube_center, center)
        )
        if valid:
            for array in (scene_min, scene_max, cube_min, cube_max):
                array.setflags(write=False)
            return scene_min, scene_max, cube_min, cube_max
        with np.errstate(over="ignore", invalid="ignore"):
            half_side = np.float64(half_side * 2.0)
        if not np.isfinite(half_side) or half_side <= 0.0:
            break
    raise ValueError(
        "unable to construct a finite bitwise-equal-sided float64 root cube "
        "that preserves the scene center"
    )


def _make_node(
    indices: np.ndarray,
    *,
    depth: int,
    path_code: int,
    parent_id: int,
    child_id: int,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> OctreeNode:
    """Construct one canonical immutable node."""
    node_id = (1 << (3 * depth)) | path_code
    center = _midpoint(np.asarray(bbox_min), np.asarray(bbox_max))
    return OctreeNode(
        indices=indices,
        depth=depth,
        path_code=path_code,
        node_id=node_id,
        parent_id=parent_id,
        child_id=child_id,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        center=center,
        point_count=len(indices),
        morton_key=path_code,
    )


def _split_octree_node_validated(
    node: OctreeNode,
    xyz: np.ndarray,
    *,
    min_group_size: int,
    max_depth: int,
) -> tuple[OctreeNode, ...]:
    """Split a node after xyz and scalar parameters have already been validated."""
    if node.depth >= max_depth:
        return ()
    if node.indices[-1] >= len(xyz):
        raise ValueError("node indices are outside xyz")
    points = xyz[node.indices]
    high = points >= node.center
    child_ids = (
        high[:, 0].astype(np.uint8)
        | (high[:, 1].astype(np.uint8) << 1)
        | (high[:, 2].astype(np.uint8) << 2)
    )
    counts = np.bincount(child_ids, minlength=8)
    nonempty = np.flatnonzero(counts)
    if len(nonempty) < 2 or np.any(counts[nonempty] < min_group_size):
        return ()

    children: list[OctreeNode] = []
    for raw_child_id in nonempty:
        child_id = int(raw_child_id)
        child_indices = node.indices[child_ids == child_id]
        child_min = np.array(node.bbox_min, dtype=np.float64, copy=True)
        child_max = np.array(node.bbox_max, dtype=np.float64, copy=True)
        for axis in range(3):
            if child_id & (1 << axis):
                child_min[axis] = node.center[axis]
            else:
                child_max[axis] = node.center[axis]
        if not np.all(child_max > child_min):
            return ()
        child_extent = child_max - child_min
        if not np.array_equal(
            child_extent,
            np.full(3, child_extent[0], dtype=np.float64),
        ):
            return ()
        path_code = node.path_code * 8 + child_id
        children.append(
            _make_node(
                child_indices,
                depth=node.depth + 1,
                path_code=path_code,
                parent_id=node.node_id,
                child_id=child_id,
                bbox_min=child_min,
                bbox_max=child_max,
            )
        )

    combined = np.concatenate([child.indices for child in children])
    if not np.array_equal(np.sort(combined), node.indices):
        raise RuntimeError("octree child partition does not equal parent indices")
    return tuple(children)


def split_octree_node(
    node: OctreeNode,
    xyz: Any,
    *,
    min_group_size: int,
    max_depth: int,
) -> tuple[OctreeNode, ...]:
    """Return a legal 2-to-8 child split, or ``()`` when the split is forbidden."""
    if not isinstance(node, OctreeNode):
        raise ValueError("node must be an OctreeNode")
    coordinates = _validate_octree_xyz(xyz)
    normalized_min_size = _strict_integer(
        min_group_size, "min_group_size", minimum=1
    )
    normalized_depth = _strict_integer(
        max_depth, "max_depth", minimum=0, maximum=_MAX_OCTREE_DEPTH
    )
    return _split_octree_node_validated(
        node,
        coordinates,
        min_group_size=normalized_min_size,
        max_depth=normalized_depth,
    )


def _padded_morton_key(node: OctreeNode, max_leaf_depth: int) -> int:
    """Pad a path with low-order zero octants to the final maximum depth."""
    return node.path_code << (3 * (max_leaf_depth - node.depth))


def octree_group_indices(
    xyz: Any,
    *,
    target_num_groups: int = 128,
    min_group_size: int = 10,
    max_depth: int = 10,
) -> OctreeGroups:
    """Build a deterministic best-first octree leaf partition without overshoot.

    Candidate priority is ``(-point_count, depth, node_id)``.  A split is
    accepted only when every nonempty child satisfies ``min_group_size`` and
    replacing its parent cannot exceed ``target_num_groups``.  No leaf is ever
    truncated, merged across parents, or duplicated to force an exact target.
    """
    coordinates = _validate_octree_xyz(xyz)
    target = _strict_integer(
        target_num_groups,
        "target_num_groups",
        minimum=1,
        maximum=_MAX_TARGET_NUM_GROUPS,
    )
    minimum_size = _strict_integer(min_group_size, "min_group_size", minimum=1)
    depth_limit = _strict_integer(
        max_depth, "max_depth", minimum=0, maximum=_MAX_OCTREE_DEPTH
    )
    scene_min, scene_max, root_min, root_max = _build_root_cube(coordinates)
    root_indices = np.arange(len(coordinates), dtype=np.int64)
    root = _make_node(
        root_indices,
        depth=0,
        path_code=0,
        parent_id=0,
        child_id=-1,
        bbox_min=root_min,
        bbox_max=root_max,
    )

    leaves: dict[int, OctreeNode] = {root.node_id: root}
    candidates: list[tuple[int, int, int, tuple[OctreeNode, ...]]] = []
    unsplittable_ids: set[int] = set()

    def register_candidate(node: OctreeNode) -> None:
        children = _split_octree_node_validated(
            node,
            coordinates,
            min_group_size=minimum_size,
            max_depth=depth_limit,
        )
        if children:
            heapq.heappush(
                candidates,
                (-node.point_count, node.depth, node.node_id, children),
            )
        else:
            unsplittable_ids.add(node.node_id)

    register_candidate(root)
    split_count = 0
    while len(leaves) < target and candidates:
        _, _, node_id, children = heapq.heappop(candidates)
        node = leaves.get(node_id)
        if node is None:
            continue
        proposed_count = len(leaves) - 1 + len(children)
        if proposed_count > target:
            continue
        del leaves[node_id]
        for child in children:
            leaves[child.node_id] = child
            register_candidate(child)
        split_count += 1

    max_leaf_depth = max(node.depth for node in leaves.values())
    ordered_leaves = tuple(
        sorted(
            leaves.values(),
            key=lambda node: (
                _padded_morton_key(node, max_leaf_depth),
                node.depth,
                node.node_id,
            ),
        )
    )
    actual = len(ordered_leaves)
    node_mins = np.stack([node.bbox_min for node in ordered_leaves], axis=0)
    node_maxs = np.stack([node.bbox_max for node in ordered_leaves], axis=0)
    result = OctreeGroups(
        group_indices=tuple(node.indices for node in ordered_leaves),
        cell_ids=tuple(node.node_id for node in ordered_leaves),
        small_group_flags=tuple(
            node.point_count < minimum_size for node in ordered_leaves
        ),
        bbox_min=scene_min,
        bbox_max=scene_max,
        grid_size=2**max_leaf_depth,
        requested_grid_size=None,
        target_num_groups=target,
        natural_group_count=actual,
        actual_group_count=actual,
        truncated_by_max_groups=False,
        max_groups=None,
        grouping_method=OCTREE_GROUPING_METHOD,
        node_ids=tuple(node.node_id for node in ordered_leaves),
        node_depths=tuple(node.depth for node in ordered_leaves),
        node_path_codes=tuple(node.path_code for node in ordered_leaves),
        node_parent_ids=tuple(node.parent_id for node in ordered_leaves),
        node_child_ids=tuple(node.child_id for node in ordered_leaves),
        node_bbox_mins=node_mins,
        node_bbox_maxs=node_maxs,
        root_cube_min=root_min,
        root_cube_max=root_max,
        max_depth=depth_limit,
        max_leaf_depth=max_leaf_depth,
        reached_target=actual == target,
        target_gap=target - actual,
        split_count=split_count,
        unsplittable_leaf_count=sum(
            node.node_id in unsplittable_ids for node in ordered_leaves
        ),
        min_group_size=minimum_size,
        leaf_order=OCTREE_LEAF_ORDER,
        center_tie_rule=OCTREE_CENTER_TIE_RULE,
        target_policy=OCTREE_TARGET_POLICY,
        min_size_policy=OCTREE_MIN_SIZE_POLICY,
    )
    _validate_octree_partition(result, coordinates)
    return result


def _path_child_ids(depth: int, path_code: int) -> tuple[int, ...]:
    """Decode a depth-qualified octal path, preserving leading zero children."""
    return tuple(
        (path_code >> (3 * (depth - level - 1))) & 7 for level in range(depth)
    )


def _validate_octree_partition(result: OctreeGroups, xyz: Any) -> None:
    """Check coverage, identity, geometry, order, and target invariants."""
    if not isinstance(result, OctreeGroups):
        raise ValueError("result must be OctreeGroups")
    coordinates = np.asarray(xyz, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,) or len(coordinates) == 0:
        raise ValueError("xyz must have nonempty shape [N, 3]")
    count = result.actual_group_count
    if count <= 0 or len(result.group_indices) != count:
        raise RuntimeError("octree result must contain at least one leaf")

    concatenated = np.concatenate(result.group_indices)
    if len(concatenated) != len(coordinates):
        raise RuntimeError("octree partition changed the number of Gaussian indices")
    if not np.array_equal(np.sort(concatenated), np.arange(len(coordinates))):
        raise RuntimeError("octree leaves must cover every global index exactly once")
    if len(set(result.node_ids)) != count:
        raise RuntimeError("leaf node_id values must be unique")
    if len(set(zip(result.node_depths, result.node_path_codes))) != count:
        raise RuntimeError("leaf (depth,path_code) pairs must be unique")
    if result.actual_group_count > result.target_num_groups:
        raise RuntimeError("octree leaf count exceeded target_num_groups")
    if result.target_gap != result.target_num_groups - count or result.target_gap < 0:
        raise RuntimeError("octree target gap is inconsistent")
    if result.truncated_by_max_groups or result.max_groups is not None:
        raise RuntimeError("octree partitions cannot be truncated")
    root_extent = result.root_cube_max - result.root_cube_min
    if not np.array_equal(
        root_extent,
        np.full(3, root_extent[0], dtype=np.float64),
    ):
        raise RuntimeError("root cube is not bitwise equal-sided in float64")
    scene_center = _midpoint(result.bbox_min, result.bbox_max)
    root_center = _midpoint(result.root_cube_min, result.root_cube_max)
    if not np.array_equal(root_center, scene_center):
        raise RuntimeError("root cube center differs from the scene bounding-box center")

    ordering: list[tuple[int, int, int]] = []
    for index, group in enumerate(result.group_indices):
        if len(group) == 0 or np.any(np.diff(group) <= 0):
            raise RuntimeError("each leaf index array must be nonempty and increasing")
        if group[0] < 0 or group[-1] >= len(coordinates):
            raise RuntimeError("leaf index is outside xyz")
        depth = result.node_depths[index]
        path_code = result.node_path_codes[index]
        node_id = result.node_ids[index]
        parent_id = result.node_parent_ids[index]
        child_id = result.node_child_ids[index]
        if node_id != (1 << (3 * depth)) | path_code:
            raise RuntimeError("leaf node_id is inconsistent")
        child_path = _path_child_ids(depth, path_code)
        if depth == 0:
            if (path_code, node_id, parent_id, child_id) != (0, 1, 0, -1):
                raise RuntimeError("root leaf identity is inconsistent")
        else:
            expected_parent = (1 << (3 * (depth - 1))) | (path_code // 8)
            if parent_id != expected_parent or child_id != child_path[-1]:
                raise RuntimeError("leaf parent/path prefix is inconsistent")

        expected_min = np.array(result.root_cube_min, copy=True)
        expected_max = np.array(result.root_cube_max, copy=True)
        points = coordinates[group]
        for expected_child in child_path:
            center = _midpoint(expected_min, expected_max)
            high = points >= center
            actual_child = (
                high[:, 0].astype(np.uint8)
                | (high[:, 1].astype(np.uint8) << 1)
                | (high[:, 2].astype(np.uint8) << 2)
            )
            if not np.all(actual_child == expected_child):
                raise RuntimeError("leaf points violate center-to-high child routing")
            for axis in range(3):
                if expected_child & (1 << axis):
                    expected_min[axis] = center[axis]
                else:
                    expected_max[axis] = center[axis]
            recursive_extent = expected_max - expected_min
            if not np.array_equal(
                recursive_extent,
                np.full(3, recursive_extent[0], dtype=np.float64),
            ):
                raise RuntimeError(
                    "recursive child bbox lost equal-sided float64 geometry"
                )
        if not np.array_equal(expected_min, result.node_bbox_mins[index]) or not np.array_equal(
            expected_max, result.node_bbox_maxs[index]
        ):
            raise RuntimeError("leaf bbox does not match its octree path")
        if np.any(points < expected_min) or np.any(points > expected_max):
            raise RuntimeError("leaf contains a point outside its bounding box")
        if depth > 0 and len(group) < result.min_group_size:
            raise RuntimeError("a generated child violates min_group_size")
        if result.small_group_flags[index] != (len(group) < result.min_group_size):
            raise RuntimeError("small_group_flags is inconsistent")
        ordering.append(
            (
                path_code << (3 * (result.max_leaf_depth - depth)),
                depth,
                node_id,
            )
        )
    if ordering != sorted(ordering):
        raise RuntimeError("leaf order is not padded-Morton deterministic order")
    if not np.array_equal(result.bbox_min, np.min(coordinates, axis=0)) or not np.array_equal(
        result.bbox_max, np.max(coordinates, axis=0)
    ):
        raise RuntimeError("scene bounding box is inconsistent with xyz")


def octree_grouping_summary(groups: OctreeGroups) -> dict[str, Any]:
    """Return scalar aggregate provenance without exposing full index arrays."""
    if not isinstance(groups, OctreeGroups):
        raise ValueError("groups must be OctreeGroups")
    sizes = np.asarray([len(indices) for indices in groups.group_indices], dtype=np.int64)
    unique_depths, depth_counts = np.unique(
        np.asarray(groups.node_depths, dtype=np.int64), return_counts=True
    )
    root_extent = groups.root_cube_max - groups.root_cube_min
    return {
        "grouping_method": groups.grouping_method,
        "target_num_groups": groups.target_num_groups,
        "actual_group_count": groups.actual_group_count,
        "target_gap": groups.target_gap,
        "reached_target": groups.reached_target,
        "min_group_size": groups.min_group_size,
        "max_depth": groups.max_depth,
        "max_leaf_depth": groups.max_leaf_depth,
        "min_leaf_size": int(np.min(sizes)),
        "median_leaf_size": float(np.median(sizes)),
        "max_leaf_size": int(np.max(sizes)),
        "split_count": groups.split_count,
        "unsplittable_leaf_count": groups.unsplittable_leaf_count,
        "root_cube_side": float(np.max(root_extent)),
        "depth_histogram": {
            int(depth): int(count) for depth, count in zip(unique_depths, depth_counts)
        },
        "small_group_count": int(sum(groups.small_group_flags)),
    }


def validate_octree_grouping() -> bool:
    """Exercise deterministic geometry, strict failures, and 100k-point scaling."""

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    def require_error(callback: Any, message: str) -> None:
        try:
            callback()
        except ValueError:
            return
        raise AssertionError(message)

    def same_result(left: OctreeGroups, right: OctreeGroups) -> bool:
        return (
            left.node_ids == right.node_ids
            and left.node_depths == right.node_depths
            and left.node_path_codes == right.node_path_codes
            and left.node_parent_ids == right.node_parent_ids
            and left.node_child_ids == right.node_child_ids
            and np.array_equal(left.node_bbox_mins, right.node_bbox_mins)
            and np.array_equal(left.node_bbox_maxs, right.node_bbox_maxs)
            and all(
                np.array_equal(a, b)
                for a, b in zip(left.group_indices, right.group_indices)
            )
        )

    def require_exact_root_cube(xyz: np.ndarray, message: str) -> OctreeGroups:
        groups = octree_group_indices(
            xyz, target_num_groups=1, min_group_size=1, max_depth=10
        )
        root_extent = groups.root_cube_max - groups.root_cube_min
        require(
            np.array_equal(
                root_extent,
                np.full(3, root_extent[0], dtype=np.float64),
            ),
            f"{message}: root extent is not bitwise equal-sided",
        )
        require(
            np.all(xyz >= groups.root_cube_min)
            and np.all(xyz <= groups.root_cube_max),
            f"{message}: root does not contain xyz",
        )
        require(
            np.array_equal(
                _midpoint(groups.root_cube_min, groups.root_cube_max),
                _midpoint(groups.bbox_min, groups.bbox_max),
            ),
            f"{message}: root center is not bitwise equal to scene center",
        )
        return groups

    octants = np.asarray(
        [
            [(-1.0 if child & 1 == 0 else 1.0),
             (-1.0 if child & 2 == 0 else 1.0),
             (-1.0 if child & 4 == 0 else 1.0)]
            for child in range(8)
        ],
        dtype=np.float64,
    )
    octant_groups = octree_group_indices(
        octants, target_num_groups=8, min_group_size=1, max_depth=1
    )
    require(octant_groups.actual_group_count == 8, "root must split into eight octants")
    require(octant_groups.node_child_ids == tuple(range(8)), "child order must be 0..7")
    require(
        all(np.array_equal(group, [child]) for child, group in enumerate(octant_groups.group_indices)),
        "octant points must enter their matching child",
    )

    tie_xyz = np.asarray([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    _, _, tie_root_min, tie_root_max = _build_root_cube(tie_xyz)
    tie_root = _make_node(
        np.arange(3, dtype=np.int64), depth=0, path_code=0, parent_id=0,
        child_id=-1, bbox_min=tie_root_min, bbox_max=tie_root_max,
    )
    tie_children = split_octree_node(tie_root, tie_xyz, min_group_size=1, max_depth=1)
    require(
        tie_children[-1].child_id == 7 and 1 in tie_children[-1].indices,
        "coordinate equality with center must route to high child",
    )

    repeated = octree_group_indices(
        np.zeros((32, 3)), target_num_groups=8, min_group_size=1, max_depth=20
    )
    require(
        repeated.actual_group_count == 1 and repeated.target_gap == 7,
        "duplicate coordinates must remain one finite root",
    )
    single = octree_group_indices(
        np.asarray([[3.0, -2.0, 7.0]]), target_num_groups=128,
        min_group_size=10, max_depth=10,
    )
    require(
        single.actual_group_count == 1
        and single.small_group_flags == (True,)
        and np.all(np.isfinite(single.root_cube_min))
        and np.all(np.isfinite(single.root_cube_max)),
        "single point must produce one finite group",
    )

    line = np.column_stack((np.linspace(-3.0, 3.0, 64), np.zeros(64), np.zeros(64)))
    plane_axis = np.linspace(-2.0, 2.0, 12)
    plane_x, plane_y = np.meshgrid(plane_axis, plane_axis, indexing="ij")
    plane = np.column_stack((plane_x.ravel(), plane_y.ravel(), np.zeros(plane_x.size)))
    line_groups = octree_group_indices(line, target_num_groups=16, min_group_size=2, max_depth=6)
    plane_groups = octree_group_indices(plane, target_num_groups=32, min_group_size=2, max_depth=6)
    require(line_groups.actual_group_count > 1, "collinear points must be supported")
    require(plane_groups.actual_group_count > 1, "coplanar points must be supported")

    noncube_xyz = np.asarray([[-100.0, -1.0, -0.1], [100.0, 1.0, 0.1], [0.0, 0.0, 0.0]])
    noncube = octree_group_indices(noncube_xyz, target_num_groups=8, min_group_size=1, max_depth=2)
    root_extent = noncube.root_cube_max - noncube.root_cube_min
    require(
        np.array_equal(
            root_extent,
            np.full(3, root_extent[0], dtype=np.float64),
        )
        and np.all(noncube_xyz >= noncube.root_cube_min)
        and np.all(noncube_xyz <= noncube.root_cube_max),
        "non-cubic scene must receive a containing root cube",
    )

    offset_cases = (
        np.asarray([[-1.0, 1e15, 0.0], [1.0, 1e15, 0.0]]),
        np.asarray([[-1.0, 1e20, 0.0], [1.0, 1e20, 0.0]]),
        np.asarray([[1e20 - 1e5, -3.0, 2.0], [1e20 + 1e5, 3.0, 4.0]]),
        np.asarray([[-1.0, 1e20 - 2e4, -1e15 - 2.0],
                    [1.0, 1e20 + 2e4, -1e15 + 2.0]]),
    )
    for case_index, offset_xyz in enumerate(offset_cases, start=1):
        require_exact_root_cube(offset_xyz, f"large-offset regression case {case_index}")
    maximum = np.finfo(np.float64).max
    require_error(
        lambda: _build_root_cube(
            np.asarray([[-maximum, 0.0, 0.0], [maximum, 0.0, 0.0]])
        ),
        "unrepresentable near-float64-limit cube must raise ValueError",
    )
    for extreme_xyz in (
        np.asarray([[-1e150, -2e149, -1e149], [1e150, 2e149, 1e149]]),
        np.asarray([[-1e-200, 0.0, 0.0], [1e-200, 0.0, 0.0]]),
    ):
        extreme = octree_group_indices(
            extreme_xyz, target_num_groups=4, min_group_size=1, max_depth=2
        )
        require(np.all(np.isfinite(extreme.root_cube_min)), "extreme finite xyz failed")

    stable_xyz = np.asarray(
        [[x, y, z] for x in (-1.0, -0.25, 0.25, 1.0)
         for y in (-1.0, -0.25, 0.25, 1.0)
         for z in (-1.0, -0.25, 0.25, 1.0)],
        dtype=np.float64,
    )
    stable_a = octree_group_indices(stable_xyz, target_num_groups=32, min_group_size=1, max_depth=5)
    stable_b = octree_group_indices(stable_xyz, target_num_groups=32, min_group_size=1, max_depth=5)
    stable_32 = octree_group_indices(stable_xyz.astype(np.float32), target_num_groups=32, min_group_size=1, max_depth=5)
    require(same_result(stable_a, stable_b), "identical runs must be bit-for-bit stable")
    require(same_result(stable_a, stable_32), "equal float32/float64 values must partition equally")

    root_only = octree_group_indices(stable_xyz, target_num_groups=1, min_group_size=1, max_depth=5)
    depth_zero = octree_group_indices(stable_xyz, target_num_groups=128, min_group_size=1, max_depth=0)
    undershoot = octree_group_indices(octants, target_num_groups=7, min_group_size=1, max_depth=2)
    require(root_only.actual_group_count == 1, "target=1 must return root")
    require(depth_zero.actual_group_count == 1, "max_depth=0 must return root")
    require(undershoot.actual_group_count == 1, "one octant split must not overshoot target")

    dense_axis = np.linspace(-1.0, 1.0, 16)
    dense_x, dense_y, dense_z = np.meshgrid(
        dense_axis, dense_axis, dense_axis, indexing="ij"
    )
    dense_xyz = np.column_stack((dense_x.ravel(), dense_y.ravel(), dense_z.ravel()))
    dense = octree_group_indices(
        dense_xyz, target_num_groups=128, min_group_size=1, max_depth=10
    )
    require(
        dense.actual_group_count <= 128
        and dense.target_gap >= 0
        and len(np.concatenate(dense.group_indices)) == len(dense_xyz),
        "dense target=128 partition is invalid",
    )
    expected_order = sorted(
        zip(dense.node_path_codes, dense.node_depths, dense.node_ids),
        key=lambda item: (
            item[0] << (3 * (dense.max_leaf_depth - item[1])), item[1], item[2]
        ),
    )
    require(
        list(zip(dense.node_path_codes, dense.node_depths, dense.node_ids)) == expected_order,
        "leaf Morton order is invalid",
    )
    require(
        all(
            node_id == (1 << (3 * depth)) | path
            for node_id, depth, path in zip(
                dense.node_ids, dense.node_depths, dense.node_path_codes
            )
        ),
        "path-code/node-id identity is invalid",
    )

    too_small = octree_group_indices(octants, target_num_groups=8, min_group_size=2, max_depth=2)
    imbalanced_xyz = np.vstack((np.full((5, 3), -1.0), np.asarray([[1.0, 1.0, 1.0]])))
    imbalanced = octree_group_indices(
        imbalanced_xyz, target_num_groups=8, min_group_size=2, max_depth=2
    )
    require(too_small.actual_group_count == 1, "min_group_size must block small children")
    require(
        imbalanced.actual_group_count == 1 and len(imbalanced.group_indices[0]) == 6,
        "one small child must reject the whole split without dropping it",
    )

    permutation = np.random.RandomState(3).permutation(len(stable_xyz))
    shuffled = octree_group_indices(
        stable_xyz[permutation], target_num_groups=32, min_group_size=1, max_depth=5
    )
    signature = lambda groups: sorted(
        (tuple(low.tolist()), tuple(high.tolist()), depth, path)
        for low, high, depth, path in zip(
            groups.node_bbox_mins, groups.node_bbox_maxs,
            groups.node_depths, groups.node_path_codes,
        )
    )
    require(
        signature(stable_a) == signature(shuffled)
        and np.array_equal(
            np.sort(np.concatenate(shuffled.group_indices)),
            np.arange(len(stable_xyz)),
        ),
        "row permutation must preserve spatial leaves and cover new global indices",
    )

    invalid_calls = (
        lambda: octree_group_indices(np.empty((0, 3))),
        lambda: octree_group_indices(np.zeros((3, 2))),
        lambda: octree_group_indices(np.zeros((3, 3), dtype=bool)),
        lambda: octree_group_indices(np.zeros((3, 3), dtype=object)),
        lambda: octree_group_indices(np.asarray([[np.nan, 0.0, 0.0]])),
        lambda: octree_group_indices(np.asarray([[np.inf, 0.0, 0.0]])),
        lambda: octree_group_indices(stable_xyz, target_num_groups=0),
        lambda: octree_group_indices(stable_xyz, target_num_groups=True),
        lambda: octree_group_indices(
            stable_xyz, target_num_groups=_MAX_TARGET_NUM_GROUPS + 1
        ),
        lambda: octree_group_indices(stable_xyz, min_group_size=0),
        lambda: octree_group_indices(stable_xyz, min_group_size=False),
        lambda: octree_group_indices(stable_xyz, max_depth=-1),
        lambda: octree_group_indices(stable_xyz, max_depth=True),
        lambda: octree_group_indices(stable_xyz, max_depth=21),
    )
    for invalid_call in invalid_calls:
        require_error(invalid_call, "invalid octree input was accepted")

    caller_xyz = stable_xyz.copy()
    caller_snapshot = caller_xyz.copy()
    validated_copy = _validate_octree_xyz(caller_xyz[::-1])
    require(
        validated_copy.dtype == np.float64
        and validated_copy.flags.c_contiguous
        and validated_copy.flags.owndata
        and not np.shares_memory(validated_copy, caller_xyz),
        "validated xyz must be an owned float64 C-contiguous copy",
    )
    immutable = octree_group_indices(
        caller_xyz, target_num_groups=16, min_group_size=1, max_depth=4
    )
    require(np.array_equal(caller_xyz, caller_snapshot), "caller xyz was modified")
    arrays = (
        *immutable.group_indices,
        immutable.bbox_min,
        immutable.bbox_max,
        immutable.node_bbox_mins,
        immutable.node_bbox_maxs,
        immutable.root_cube_min,
        immutable.root_cube_max,
    )
    require(all(not array.flags.writeable for array in arrays), "returned arrays must be read-only")
    summary = octree_grouping_summary(immutable)
    require(
        "group_indices" not in summary
        and not any(isinstance(value, np.ndarray) for value in summary.values()),
        "summary must not expose full index arrays",
    )

    random_xyz = np.random.RandomState(11).uniform(
        -10.0, 10.0, size=(100_000, 3)
    ).astype(np.float32)
    large = octree_group_indices(
        random_xyz, target_num_groups=128, min_group_size=10, max_depth=10
    )
    require(
        large.actual_group_count <= 128
        and np.array_equal(
            np.sort(np.concatenate(large.group_indices)), np.arange(100_000)
        ),
        "100k-point octree must complete with exact coverage and no overshoot",
    )
    return True


__all__ = [
    "OCTREE_GROUPING_METHOD",
    "OCTREE_CHILD_ORDER",
    "OCTREE_CENTER_TIE_RULE",
    "OCTREE_LEAF_ORDER",
    "OCTREE_TARGET_POLICY",
    "OCTREE_MIN_SIZE_POLICY",
    "OctreeNode",
    "OctreeGroups",
    "split_octree_node",
    "octree_group_indices",
    "octree_grouping_summary",
    "validate_octree_grouping",
]
