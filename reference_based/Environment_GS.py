"""Formal first-version factorized 3DGS compression environment.

The environment freezes one deterministic best-first geometry octree at the
start of each Episode.  The tree defines only the spatial scope of RL actions:
opacity, SH, scale, reward, quality, and actions do not affect grouping, and
post-pruning survivors never trigger regrouping.  Low-opacity-first pruning
still occurs inside each frozen leaf.  Visual effects remain constrained by
fixed-view whole-scene quality checkpoints.  Actions are
the independent continuous pruning and precision levels consumed by the V2
compact path; size and quality rewards remain separate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
from typing import Callable

import numpy as np

try:
    from .config_utils import (
        CODE_ROOT,
        normalize_crossscore_dir,
        normalize_gaussian_splatting_dir,
    )
    from .compression_ops import (
        PRUNING_MODE_OPACITY_BASELINE,
        decode_factorized_action,
        estimate_size_ratio_from_factorized_actions,
        factorized_action_to_array,
    )
    from .crossscore_bridge import (
        compute_crossscore_placeholder,
        compute_crossscore_real,
        load_score_cache,
        save_score_cache,
    )
    from .gs_compressor import GSCompressor
    from .model_path_utils import ensure_model_structure_from_ply
    from .ply_utils import get_xyz, numbered_fields, read_ply
    from .render_bridge import render_scene_pair_fixed_subset
    from .octree_grouping import (
        OCTREE_CHILD_ORDER,
        OCTREE_GROUPING_METHOD,
        OCTREE_LEAF_ORDER,
        OctreeGroups,
        octree_group_indices,
        octree_grouping_summary,
    )
except ImportError:
    from config_utils import (
        CODE_ROOT,
        normalize_crossscore_dir,
        normalize_gaussian_splatting_dir,
    )
    from compression_ops import (
        PRUNING_MODE_OPACITY_BASELINE,
        decode_factorized_action,
        estimate_size_ratio_from_factorized_actions,
        factorized_action_to_array,
    )
    from crossscore_bridge import (
        compute_crossscore_placeholder,
        compute_crossscore_real,
        load_score_cache,
        save_score_cache,
    )
    from gs_compressor import GSCompressor
    from model_path_utils import ensure_model_structure_from_ply
    from ply_utils import get_xyz, numbered_fields, read_ply
    from render_bridge import render_scene_pair_fixed_subset
    from octree_grouping import (
        OCTREE_CHILD_ORDER,
        OCTREE_GROUPING_METHOD,
        OCTREE_LEAF_ORDER,
        OctreeGroups,
        octree_group_indices,
        octree_grouping_summary,
    )


V1_GROUP_FEATURE_NAMES = (
    "num_gaussians_ratio",
    "opacity_mean",
    "opacity_std",
    "opacity_low_ratio",
    "scale_mean",
    "scale_std",
    "sh_energy_mean",
    "xyz_extent_mean",
    "small_group_flag",
)

FACTORIZED_STATE_FEATURE_NAMES = list(V1_GROUP_FEATURE_NAMES) + [
    "current_group_idx_ratio",
    "remaining_group_ratio",
    "current_estimated_size_ratio",
    "last_quality_drop",
    "quality_margin",
    "quality_observed_flag",
    "previous_pruning_mean",
    "previous_pruning_last",
    "previous_precision_mean",
    "previous_precision_last",
]

FACTORIZED_GROUPING_METHOD = OCTREE_GROUPING_METHOD
FACTORIZED_GROUPING_VERSION = "octree_grouping_v1"
FACTORIZED_GROUP_TRAVERSAL = OCTREE_LEAF_ORDER


def _field_array(
    vertex_data: np.ndarray,
    field: str,
    indices: np.ndarray,
) -> np.ndarray:
    """Return one existing PLY field for a leaf, or a zero fallback."""
    if field not in (vertex_data.dtype.names or ()):
        return np.zeros(len(indices), dtype=np.float32)
    return vertex_data[field][indices].astype(np.float32, copy=False)


def _stack_existing_fields(
    vertex_data: np.ndarray,
    fields: list[str],
    indices: np.ndarray,
) -> np.ndarray:
    """Stack existing PLY fields without modifying the source array."""
    names = vertex_data.dtype.names or ()
    existing = [field for field in fields if field in names]
    if not existing:
        return np.zeros((len(indices), 0), dtype=np.float32)
    return np.column_stack(
        [vertex_data[field][indices].astype(np.float32, copy=False) for field in existing]
    )


def _extract_octree_group_features(
    vertex_data: np.ndarray,
    indices: np.ndarray,
    *,
    total_gaussians: int,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    opacity_low_threshold: float,
    small_group_flag: bool,
) -> np.ndarray:
    """Return the unchanged nine first-version features for one octree leaf."""
    if len(indices) == 0:
        raise ValueError("Cannot extract features from an empty octree leaf")
    xyz = _stack_existing_fields(vertex_data, ["x", "y", "z"], indices)
    opacity = _field_array(vertex_data, "opacity", indices)
    scale_values = _stack_existing_fields(
        vertex_data, ["scale_0", "scale_1", "scale_2"], indices
    )
    sh_fields = ["f_dc_0", "f_dc_1", "f_dc_2"] + numbered_fields(
        vertex_data.dtype.names or (), "f_rest"
    )
    sh_values = _stack_existing_fields(vertex_data, sh_fields, indices)
    scene_extent = np.maximum(
        np.asarray(bbox_max) - np.asarray(bbox_min), 1e-8
    )
    xyz_extent_mean = (
        float(np.mean((xyz.max(axis=0) - xyz.min(axis=0)) / scene_extent))
        if xyz.shape[1] == 3
        else 0.0
    )
    if scale_values.size:
        flat_scale = scale_values.reshape(-1)
        scale_mean = float(np.mean(flat_scale))
        scale_std = float(np.std(flat_scale))
    else:
        scale_mean = scale_std = 0.0
    sh_energy_mean = (
        float(np.mean(np.sum(np.square(sh_values), axis=1)))
        if sh_values.size
        else 0.0
    )
    features = np.asarray(
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
    features = np.nan_to_num(
        features, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32, copy=False)
    if features.shape != (9,) or not np.all(np.isfinite(features)):
        raise RuntimeError("octree leaf features must be finite float32[9]")
    return features


class _FrozenGroupIndexList(list[np.ndarray]):
    """One reset-time list adapter for the existing compressor API.

    ``OctreeGroups`` owns the same read-only index arrays.  This adapter is
    created once, stored on that Episode's group object, and reused unchanged
    by every checkpoint because ``GSCompressor`` retains its historical strict
    ``list`` input contract.
    """

    @staticmethod
    def _reject_mutation(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("factorized octree group order is frozen for the Episode")

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    append = _reject_mutation
    clear = _reject_mutation
    extend = _reject_mutation
    insert = _reject_mutation
    pop = _reject_mutation
    remove = _reject_mutation
    reverse = _reject_mutation
    sort = _reject_mutation
    __iadd__ = _reject_mutation
    __imul__ = _reject_mutation


@dataclass(frozen=True)
class SizeRewardResult:
    """Result of one dense or terminal logarithmic size correction."""

    size_before: float
    size_after: float
    reward: float


@dataclass(frozen=True)
class QualityBlockRewardResult:
    """Detailed result of one externally observed quality-block reward."""

    original_score: float
    previous_score: float
    current_score: float
    previous_quality_drop: float
    current_quality_drop: float
    incremental_quality_drop: float
    effective_incremental_drop: float
    quality_violation: float
    reward: float
    quality_feasible: bool

def _canonical_factorized_group_partition_sha256(
    groups: OctreeGroups,
    vertex_count: int,
) -> str:
    """Hash the ordered octree leaf partition using canonical int64 fields."""
    if not isinstance(groups, OctreeGroups):
        raise ValueError("groups must be OctreeGroups")
    if isinstance(vertex_count, (bool, np.bool_)) or not isinstance(
        vertex_count, (int, np.integer)
    ):
        raise ValueError("vertex_count must be a strict positive integer")
    normalized_vertex_count = int(vertex_count)
    if normalized_vertex_count <= 0:
        raise ValueError("vertex_count must be a strict positive integer")
    if len(groups.group_indices) != groups.actual_group_count:
        raise ValueError("octree actual_group_count is inconsistent")
    metadata_columns = (
        groups.node_ids,
        groups.node_depths,
        groups.node_path_codes,
        groups.node_parent_ids,
        groups.node_child_ids,
    )
    if any(len(values) != groups.actual_group_count for values in metadata_columns):
        raise ValueError("octree leaf metadata lengths are inconsistent")

    digest = hashlib.sha256()
    digest.update(b"factorized_octree_partition_v1")

    def update_ints(values: Any) -> None:
        canonical = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
        digest.update(canonical.tobytes(order="C"))

    update_ints(
        (
            normalized_vertex_count,
            groups.actual_group_count,
            groups.target_num_groups,
            groups.min_group_size,
            groups.max_depth,
        )
    )
    for leaf_index, indices in enumerate(groups.group_indices):
        canonical_indices = np.asarray(indices)
        if (
            canonical_indices.ndim != 1
            or canonical_indices.dtype.kind not in {"i", "u"}
            or canonical_indices.size == 0
            or np.any(canonical_indices < 0)
            or np.any(np.diff(canonical_indices.astype(np.int64)) <= 0)
        ):
            raise ValueError("octree group indices must be nonempty increasing integers")
        update_ints(
            (
                groups.node_ids[leaf_index],
                groups.node_depths[leaf_index],
                groups.node_path_codes[leaf_index],
                groups.node_parent_ids[leaf_index],
                groups.node_child_ids[leaf_index],
                len(canonical_indices),
            )
        )
        update_ints(canonical_indices)
    return digest.hexdigest()




def _finite_reward_input(value: float, name: str) -> float:
    """Return a finite float for the new reward helpers."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _positive_size(value: float, name: str) -> float:
    """Return a finite positive size value."""
    normalized = _finite_reward_input(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return normalized


def _nonnegative_reward_parameter(value: float, name: str) -> float:
    """Return a finite nonnegative reward parameter."""
    normalized = _finite_reward_input(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return normalized


def _strict_checkpoint_int(value: int, name: str) -> int:
    """Return a strict integer for the new checkpoint helpers."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a strict integer")
    return int(value)


def compute_dense_size_reward(
    size_before: float,
    size_after: float,
) -> SizeRewardResult:
    """Compute the dense size reward ``log(size_before / size_after)``.

    The reward remains positive for compression, zero for unchanged size, and
    negative for expansion.  It is independent of any target size ratio and
    is not truncated after a target is reached.
    """
    normalized_before = _positive_size(size_before, "size_before")
    normalized_after = _positive_size(size_after, "size_after")
    return SizeRewardResult(
        size_before=normalized_before,
        size_after=normalized_after,
        reward=float(np.log(normalized_before / normalized_after)),
    )


def compute_terminal_size_correction(
    estimated_final_size: float,
    actual_compact_size: float,
) -> SizeRewardResult:
    """Correct an estimated final size to the real compact package size."""
    normalized_estimate = _positive_size(
        estimated_final_size, "estimated_final_size"
    )
    normalized_actual = _positive_size(
        actual_compact_size, "actual_compact_size"
    )
    return SizeRewardResult(
        size_before=normalized_estimate,
        size_after=normalized_actual,
        reward=float(np.log(normalized_estimate / normalized_actual)),
    )


def compute_quality_block_reward(
    original_score: float,
    previous_score: float,
    current_score: float,
    *,
    higher_is_better: bool,
    quality_epsilon: float,
    incremental_alpha: float,
    violation_alpha: float,
    noise_tolerance: float = 0.0,
) -> QualityBlockRewardResult:
    """Compute reward metadata for one externally evaluated quality block.

    The signed incremental term rewards meaningful quality recovery and
    penalizes new degradation.  The violation term continues penalizing a
    cumulative quality drop while it remains beyond ``quality_epsilon``.
    This function is pure: it neither evaluates quality nor mutates an
    environment.
    """
    normalized_original = _finite_reward_input(original_score, "original_score")
    normalized_previous = _finite_reward_input(previous_score, "previous_score")
    normalized_current = _finite_reward_input(current_score, "current_score")
    if not isinstance(higher_is_better, (bool, np.bool_)):
        raise ValueError("higher_is_better must be a boolean")
    normalized_epsilon = _nonnegative_reward_parameter(
        quality_epsilon, "quality_epsilon"
    )
    normalized_incremental_alpha = _nonnegative_reward_parameter(
        incremental_alpha, "incremental_alpha"
    )
    normalized_violation_alpha = _nonnegative_reward_parameter(
        violation_alpha, "violation_alpha"
    )
    normalized_noise = _nonnegative_reward_parameter(
        noise_tolerance, "noise_tolerance"
    )

    if bool(higher_is_better):
        previous_drop = normalized_original - normalized_previous
        current_drop = normalized_original - normalized_current
    else:
        previous_drop = normalized_previous - normalized_original
        current_drop = normalized_current - normalized_original

    incremental_drop = current_drop - previous_drop
    effective_incremental_drop = float(
        np.sign(incremental_drop)
        * max(abs(incremental_drop) - normalized_noise, 0.0)
    )
    at_quality_boundary = bool(
        np.isclose(current_drop, normalized_epsilon, rtol=1e-12, atol=1e-12)
    )
    quality_violation = (
        0.0
        if at_quality_boundary
        else max(0.0, current_drop - normalized_epsilon)
    )
    reward = (
        -normalized_incremental_alpha * effective_incremental_drop
        - normalized_violation_alpha * quality_violation
    )
    return QualityBlockRewardResult(
        original_score=normalized_original,
        previous_score=normalized_previous,
        current_score=normalized_current,
        previous_quality_drop=float(previous_drop),
        current_quality_drop=float(current_drop),
        incremental_quality_drop=float(incremental_drop),
        effective_incremental_drop=effective_incremental_drop,
        quality_violation=float(quality_violation),
        reward=float(reward),
        quality_feasible=bool(
            current_drop <= normalized_epsilon or at_quality_boundary
        ),
    )


def should_run_quality_checkpoint(
    groups_processed: int,
    total_groups: int,
    last_checkpoint_groups: int,
    interval: int = 8,
) -> bool:
    """Return whether a periodic or final partial-block check is due."""
    normalized_processed = _strict_checkpoint_int(
        groups_processed, "groups_processed"
    )
    normalized_total = _strict_checkpoint_int(total_groups, "total_groups")
    normalized_last = _strict_checkpoint_int(
        last_checkpoint_groups, "last_checkpoint_groups"
    )
    normalized_interval = _strict_checkpoint_int(interval, "interval")
    if normalized_total <= 0:
        raise ValueError("total_groups must be greater than zero")
    if normalized_interval <= 0:
        raise ValueError("interval must be greater than zero")
    if not 0 <= normalized_last <= normalized_processed <= normalized_total:
        raise ValueError(
            "checkpoint counts must satisfy 0 <= last_checkpoint_groups <= "
            "groups_processed <= total_groups"
        )
    if normalized_processed <= normalized_last:
        return False
    return bool(
        normalized_processed % normalized_interval == 0
        or normalized_processed == normalized_total
    )


def quality_block_step_bounds(
    last_checkpoint_groups: int,
    groups_processed: int,
    interval: int = 8,
) -> tuple[int, int]:
    """Convert completed group counts to a zero-based Replay step interval."""
    normalized_last = _strict_checkpoint_int(
        last_checkpoint_groups, "last_checkpoint_groups"
    )
    normalized_processed = _strict_checkpoint_int(
        groups_processed, "groups_processed"
    )
    normalized_interval = _strict_checkpoint_int(interval, "interval")
    if normalized_last < 0 or normalized_processed < 0:
        raise ValueError("checkpoint group counts must be nonnegative")
    if normalized_interval <= 0:
        raise ValueError("interval must be greater than zero")
    if normalized_last >= normalized_processed:
        raise ValueError(
            "last_checkpoint_groups must be less than groups_processed"
        )
    block_length = normalized_processed - normalized_last
    if not 1 <= block_length <= normalized_interval:
        raise ValueError(
            f"quality block length must be between 1 and {normalized_interval}"
        )
    return normalized_last, normalized_processed - 1


def validate_factorized_reward_design() -> bool:
    """Run lightweight invariants for the opt-in factorized reward helpers."""

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    def require_value_error(
        callable_object: Callable[[], object], message: str
    ) -> None:
        try:
            callable_object()
        except ValueError:
            return
        raise AssertionError(message)

    compressed = compute_dense_size_reward(100.0, 50.0)
    require(np.isclose(compressed.reward, np.log(2.0)), "100 -> 50 must be log(2)")
    require(compressed.reward > 0.0, "compression must have positive size reward")
    unchanged = compute_dense_size_reward(100.0, 100.0)
    require(np.isclose(unchanged.reward, 0.0), "unchanged size reward must be zero")
    expanded = compute_dense_size_reward(100.0, 200.0)
    require(expanded.reward < 0.0, "size expansion must have negative reward")
    for invalid in (0.0, -1.0, np.nan, np.inf, -np.inf):
        require_value_error(
            lambda value=invalid: compute_dense_size_reward(value, 1.0),
            "invalid size_before must raise ValueError",
        )
        require_value_error(
            lambda value=invalid: compute_dense_size_reward(1.0, value),
            "invalid size_after must raise ValueError",
        )

    telescoping_reward = sum(
        compute_dense_size_reward(before, after).reward
        for before, after in ((100.0, 80.0), (80.0, 60.0), (60.0, 50.0))
    )
    require(
        np.isclose(telescoping_reward, np.log(100.0 / 50.0)),
        "dense size rewards must telescope to the total log size ratio",
    )
    correction = compute_terminal_size_correction(60.0, 50.0)
    require(
        np.isclose(correction.reward, np.log(60.0 / 50.0)),
        "terminal size correction must use estimated/actual compact size",
    )

    degraded = compute_quality_block_reward(
        1.0,
        0.96,
        0.94,
        higher_is_better=True,
        quality_epsilon=0.05,
        incremental_alpha=1.0,
        violation_alpha=2.0,
    )
    require(
        np.isclose(degraded.previous_quality_drop, 0.04)
        and np.isclose(degraded.current_quality_drop, 0.06)
        and np.isclose(degraded.incremental_quality_drop, 0.02)
        and np.isclose(degraded.quality_violation, 0.01)
        and np.isclose(degraded.reward, -0.04),
        "higher-is-better degradation reward is incorrect",
    )
    recovered = compute_quality_block_reward(
        1.0,
        0.90,
        0.94,
        higher_is_better=True,
        quality_epsilon=0.05,
        incremental_alpha=1.0,
        violation_alpha=2.0,
    )
    require(
        np.isclose(recovered.incremental_quality_drop, -0.04)
        and recovered.effective_incremental_drop < 0.0
        and np.isclose(recovered.reward, 0.02),
        "quality recovery must produce the expected positive feedback",
    )
    noise_dead_zone = compute_quality_block_reward(
        1.0,
        0.95,
        0.9495,
        higher_is_better=True,
        quality_epsilon=1.0,
        incremental_alpha=1.0,
        violation_alpha=0.0,
        noise_tolerance=0.001,
    )
    require(
        np.isclose(noise_dead_zone.effective_incremental_drop, 0.0),
        "incremental changes within noise tolerance must be zeroed",
    )
    lower_is_better = compute_quality_block_reward(
        0.1,
        0.11,
        0.13,
        higher_is_better=False,
        quality_epsilon=1.0,
        incremental_alpha=1.0,
        violation_alpha=0.0,
    )
    require(
        lower_is_better.current_quality_drop > lower_is_better.previous_quality_drop
        and lower_is_better.incremental_quality_drop > 0.0,
        "lower-is-better quality degradation direction is incorrect",
    )
    boundary = compute_quality_block_reward(
        1.0,
        0.875,
        0.75,
        higher_is_better=True,
        quality_epsilon=0.25,
        incremental_alpha=1.0,
        violation_alpha=1.0,
    )
    require(
        boundary.quality_feasible and np.isclose(boundary.quality_violation, 0.0),
        "quality at epsilon must be feasible with zero violation",
    )
    for parameter_name in (
        "quality_epsilon",
        "incremental_alpha",
        "violation_alpha",
        "noise_tolerance",
    ):
        arguments = {
            "higher_is_better": True,
            "quality_epsilon": 0.0,
            "incremental_alpha": 1.0,
            "violation_alpha": 1.0,
            "noise_tolerance": 0.0,
        }
        arguments[parameter_name] = -1.0
        require_value_error(
            lambda kwargs=arguments: compute_quality_block_reward(
                1.0, 1.0, 1.0, **kwargs
            ),
            f"negative {parameter_name} must raise ValueError",
        )

    checkpoints: list[int] = []
    last_checkpoint = 0
    for groups_processed in range(1, 20):
        if should_run_quality_checkpoint(
            groups_processed,
            total_groups=19,
            last_checkpoint_groups=last_checkpoint,
            interval=8,
        ):
            checkpoints.append(groups_processed)
            last_checkpoint = groups_processed
    require(checkpoints == [8, 16, 19], "checkpoints must occur at 8, 16, and 19")
    require(
        not should_run_quality_checkpoint(8, 19, 8, interval=8),
        "an already completed checkpoint must not run again",
    )
    require(
        quality_block_step_bounds(0, 8) == (0, 7),
        "first full checkpoint bounds are incorrect",
    )
    require(
        quality_block_step_bounds(16, 19) == (16, 18),
        "final partial checkpoint bounds are incorrect",
    )
    require_value_error(
        lambda: quality_block_step_bounds(0, 9, interval=8),
        "a block longer than interval must raise ValueError",
    )
    require_value_error(
        lambda: should_run_quality_checkpoint(9, 8, 0),
        "groups_processed beyond total_groups must raise ValueError",
    )
    require_value_error(
        lambda: should_run_quality_checkpoint(7, 19, 8),
        "last_checkpoint_groups beyond groups_processed must raise ValueError",
    )
    require_value_error(
        lambda: should_run_quality_checkpoint(True, 19, 0),
        "boolean checkpoint counts must raise ValueError",
    )
    return True


class GS_Environment(object):
    def __init__(
        self,
        scenes: list[dict[str, Any]] | None = None,
        output_root: str | Path = "outputs",
        target_size_ratio: float = 0.3,
        target_num_groups: int | None = 128,
        min_group_size: int = 10,
        opacity_low_threshold: float = 0.01,
        use_dummy_reward: bool = True,
        use_render: bool = False,
        use_crossscore: bool = False,
        gaussian_splatting_dir: str | Path | None = None,
        crossscore_dir: str | Path | None = None,
        source_path: str | Path | None = None,
        original_model_path: str | Path | None = None,
        iteration: int = 30000,
        resolution: int = 4,
        quality_cache_dir: str | Path | None = None,
        cache_original_score: bool = True,
        score_higher_is_better: bool = True,
        crossscore_mode: str = "placeholder",
        crossscore_command_template: str = "",
        crossscore_score_output: str = "",
        crossscore_score_parse_mode: str = "auto",
        crossscore_preferred_score_key: str = "pred_ssim_0_1",
        render_python_executable: str = "python",
        crossscore_python_executable: str = "python",
        crossscore_ckpt: str | Path | None = None,
        crossscore_config: str | Path | None = None,
        crossscore_allow_image_fallback: bool = False,
        quality_epsilon: float = 0.0,
        quality_dense_alpha: float = 1.0,
        quality_violation_alpha: float = 2.0,
        allow_crossscore_placeholder: bool = False,
        force_recompute_original_score: bool = False,
        factorized_octree_max_depth: int = 10,
    ) -> None:
        """Create the first-version factorized octree environment."""
        self.scenes = scenes or [
            {
                "name": "train",
                "source_path": str(CODE_ROOT / "data" / "tandt" / "train"),
                "model_path": str(CODE_ROOT / "data" / "gs_models" / "train_original"),
                "ply_path": str(CODE_ROOT / "data" / "train.ply"),
            },
            {
                "name": "truck",
                "source_path": str(CODE_ROOT / "data" / "tandt" / "truck"),
                "model_path": str(CODE_ROOT / "data" / "gs_models" / "truck_original"),
                "ply_path": str(CODE_ROOT / "data" / "truck.ply"),
            },
        ]
        self.output_root = Path(output_root)
        self.target_size_ratio = target_size_ratio
        self._factorized_constructor_target_num_groups_is_strict_integer = (
            target_num_groups is None
            or (
                not isinstance(target_num_groups, (bool, np.bool_))
                and isinstance(target_num_groups, (int, np.integer))
            )
        )
        self.target_num_groups = int(target_num_groups) if target_num_groups is not None else None
        self.min_group_size = min_group_size
        self.opacity_low_threshold = opacity_low_threshold
        self.use_dummy_reward = use_dummy_reward
        self.use_render = use_render
        self.use_crossscore = use_crossscore
        self.gaussian_splatting_dir = (
            normalize_gaussian_splatting_dir(gaussian_splatting_dir)
            if gaussian_splatting_dir
            else None
        )
        self.crossscore_dir = (
            normalize_crossscore_dir(crossscore_dir) if crossscore_dir else None
        )
        self.default_source_path = Path(source_path) if source_path else None
        self.default_original_model_path = (
            Path(original_model_path) if original_model_path else None
        )
        self.iteration = int(iteration)
        self.resolution = int(resolution)
        self.cache_original_score = cache_original_score
        self.score_higher_is_better = score_higher_is_better
        self.crossscore_mode = crossscore_mode
        self.crossscore_command_template = crossscore_command_template
        self.crossscore_score_output = crossscore_score_output
        self.crossscore_score_parse_mode = crossscore_score_parse_mode
        self.crossscore_preferred_score_key = crossscore_preferred_score_key
        self.render_python_executable = render_python_executable
        self.crossscore_python_executable = crossscore_python_executable
        self.crossscore_ckpt = crossscore_ckpt
        self.crossscore_config = crossscore_config
        self.crossscore_allow_image_fallback = crossscore_allow_image_fallback
        self.quality_epsilon = float(quality_epsilon)
        self.quality_dense_alpha = float(quality_dense_alpha)
        self.quality_violation_alpha = float(quality_violation_alpha)
        self.allow_crossscore_placeholder = allow_crossscore_placeholder
        self.force_recompute_original_score = force_recompute_original_score
        if isinstance(factorized_octree_max_depth, (bool, np.bool_)) or not isinstance(
            factorized_octree_max_depth, (int, np.integer)
        ):
            raise ValueError("factorized_octree_max_depth must be a strict integer")
        self.factorized_octree_max_depth = int(factorized_octree_max_depth)
        if not 0 <= self.factorized_octree_max_depth <= 20:
            raise ValueError("factorized_octree_max_depth must be between 0 and 20")

        self.factorized_state_dim = len(FACTORIZED_STATE_FEATURE_NAMES)
        self.factorized_state_feature_names = tuple(FACTORIZED_STATE_FEATURE_NAMES)
        self.episode = 0
        self.total_t = 0
        self.seqNum = 0
        self.frameNum = 0
        self.current_group_idx = 0
        self.done = False
        self.scene_info: dict[str, Any] = {}
        self.scene_name = ""
        self.scene_path = None
        self.original_model_path = None
        self.ply_path = None
        self.ply = None
        self.groups = None
        self.original_size_bytes = 0
        self.target_size_bytes = 0.0
        self.compressor = GSCompressor(self.output_root)
        self.quality_cache_dir = Path(quality_cache_dir) if quality_cache_dir else (
            self.output_root / "crossscore_cache"
        )
        self.factorized_mode_active = False
        self.factorized_actions: list[tuple[int, int]] = []
        self.factorized_octree_groups: OctreeGroups | None = None
        self.factorized_grouping_method: str | None = None
        self.factorized_grouping_version: str | None = None
        self.factorized_group_traversal: str | None = None
        self.factorized_group_partition_sha256: str | None = None
        self._factorized_group_partition_sha256_frozen: str | None = None
        self.factorized_group_object_id: int | None = None
        self.factorized_group_sequence_object_id: int | None = None
        self.factorized_group_count = 0
        self.factorized_octree_target_num_groups: int | None = None
        self.factorized_octree_min_group_size: int | None = None
        self.factorized_octree_max_depth_active: int | None = None
        self.factorized_octree_target_gap: int | None = None
        self.factorized_octree_reached_target: bool | None = None
        self.factorized_octree_max_leaf_depth: int | None = None
        self.factorized_octree_split_count: int | None = None
        self.factorized_grouping_summary: dict[str, Any] | None = None
        self.pruning_mode = PRUNING_MODE_OPACITY_BASELINE
        self._reset_factorized_episode_logs()


    def _reset_factorized_episode_logs(self) -> None:
        """Initialize only records used by the formal factorized training path."""
        self.factorized_action_all: list[list[int]] = []
        self.factorized_reward_D_all: list[float] = []
        self.factorized_reward_P_all: list[float] = []
        self.factorized_quality_checkpoint_all: list[bool] = []
        self.factorized_size_ratio_all: list[float] = []
        self.factorized_action_continuous: list[list[float]] = []
        self.last_info: dict[str, Any] = {}

    def _resolve_factorized_octree_parameters(
        self,
        scene_info: dict[str, Any],
    ) -> tuple[int, int, int]:
        """Resolve strict scene-overridable octree controls for one Episode."""
        if not isinstance(scene_info, dict):
            raise ValueError("scene_info must be a dictionary")
        target_from_scene = "factorized_target_num_groups" in scene_info
        target = scene_info.get(
            "factorized_target_num_groups", self.target_num_groups
        )
        minimum_size = scene_info.get(
            "factorized_octree_min_group_size", self.min_group_size
        )
        maximum_depth = scene_info.get(
            "factorized_octree_max_depth", self.factorized_octree_max_depth
        )
        if target is None:
            raise ValueError(
                "factorized_target_num_groups or target_num_groups is required"
            )
        if (
            (
                not target_from_scene
                and not self._factorized_constructor_target_num_groups_is_strict_integer
            )
            or isinstance(target, (bool, np.bool_))
            or not isinstance(target, (int, np.integer))
            or int(target) <= 0
            or int(target) > 2**31
        ):
            raise ValueError("factorized target_num_groups must be a strict positive integer")
        if isinstance(minimum_size, (bool, np.bool_)) or not isinstance(
            minimum_size, (int, np.integer)
        ) or int(minimum_size) <= 0:
            raise ValueError("factorized min_group_size must be a strict positive integer")
        if isinstance(maximum_depth, (bool, np.bool_)) or not isinstance(
            maximum_depth, (int, np.integer)
        ) or not 0 <= int(maximum_depth) <= 20:
            raise ValueError("factorized max_depth must be a strict integer from 0 to 20")
        return int(target), int(minimum_size), int(maximum_depth)

    @staticmethod
    def _validate_prepared_factorized_octree(
        groups: OctreeGroups,
        vertex_count: int,
    ) -> None:
        """Reject any octree result that is not a complete immutable partition."""
        if not isinstance(groups, OctreeGroups):
            raise RuntimeError("factorized grouping did not return OctreeGroups")
        if not groups.group_indices:
            raise RuntimeError("factorized octree created no leaf groups")
        if groups.actual_group_count != len(groups.group_indices):
            raise RuntimeError("octree actual_group_count is inconsistent")
        if groups.grouping_method != OCTREE_GROUPING_METHOD:
            raise RuntimeError("factorized octree grouping_method is inconsistent")
        if groups.actual_group_count > groups.target_num_groups:
            raise RuntimeError("factorized octree exceeded target_num_groups")
        if groups.target_gap != groups.target_num_groups - groups.actual_group_count:
            raise RuntimeError("factorized octree target_gap is inconsistent")
        if len(groups.group_indices) != len(groups.node_ids):
            raise RuntimeError("factorized octree leaf metadata is inconsistent")
        arrays = tuple(groups.group_indices) + (
            groups.bbox_min,
            groups.bbox_max,
            groups.node_bbox_mins,
            groups.node_bbox_maxs,
            groups.root_cube_min,
            groups.root_cube_max,
        )
        if any(array.flags.writeable for array in arrays):
            raise RuntimeError("factorized octree arrays must be read-only")
        if any(len(indices) == 0 for indices in groups.group_indices):
            raise RuntimeError("factorized octree contains an empty leaf")
        flattened = np.concatenate(groups.group_indices).astype(np.int64, copy=False)
        if not np.array_equal(
            np.sort(flattened), np.arange(vertex_count, dtype=np.int64)
        ):
            raise RuntimeError(
                "factorized octree must cover every Gaussian exactly once"
            )

    def _prepare_factorized_octree_scene(
        self,
        scene_info: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Read one scene and build its geometry-only octree without committing."""
        automatic_scene = scene_info is None
        if automatic_scene:
            if not self.scenes:
                raise ValueError("scenes must contain at least one scene")
            selected_scene = self.scenes[self.seqNum % len(self.scenes)]
        else:
            selected_scene = scene_info
        if not isinstance(selected_scene, dict):
            raise ValueError("scene_info must be a dictionary")
        target, minimum_size, maximum_depth = (
            self._resolve_factorized_octree_parameters(selected_scene)
        )
        if "ply_path" not in selected_scene:
            raise ValueError("scene_info must contain ply_path")
        ply_path = Path(selected_scene["ply_path"])
        staged_ply = read_ply(ply_path)
        original_size_bytes = int(ply_path.stat().st_size)
        xyz = get_xyz(staged_ply.vertex_data)
        groups = octree_group_indices(
            xyz,
            target_num_groups=target,
            min_group_size=minimum_size,
            max_depth=maximum_depth,
        )
        self._validate_prepared_factorized_octree(groups, len(staged_ply.vertex_data))

        # Preserve the existing compressor's strict list contract without ever
        # rebuilding that list at a checkpoint.  Its arrays remain the exact
        # immutable arrays produced by OctreeGroups in the exact leaf order.
        frozen_group_list = _FrozenGroupIndexList(groups.group_indices)
        object.__setattr__(groups, "group_indices", frozen_group_list)
        self._validate_prepared_factorized_octree(groups, len(staged_ply.vertex_data))
        summary = octree_grouping_summary(groups)
        partition_sha256 = _canonical_factorized_group_partition_sha256(
            groups, len(staged_ply.vertex_data)
        )
        scene_name = selected_scene.get("name", ply_path.stem)
        scene_path = (
            selected_scene.get("source_path")
            or selected_scene.get("scene_path")
            or self.default_source_path
        )
        original_model_path = (
            selected_scene.get("model_path")
            or selected_scene.get("original_model_path")
            or self.default_original_model_path
        )
        return {
            "automatic_scene": automatic_scene,
            "scene_info": selected_scene,
            "scene_name": scene_name,
            "scene_path": scene_path,
            "original_model_path": original_model_path,
            "ply_path": ply_path,
            "ply": staged_ply,
            "original_size_bytes": original_size_bytes,
            "target_size_bytes": original_size_bytes * self.target_size_ratio,
            "groups": groups,
            "target": target,
            "minimum_size": minimum_size,
            "maximum_depth": maximum_depth,
            "summary": summary,
            "partition_sha256": partition_sha256,
        }

    def _validate_factorized_octree_frozen(
        self,
        *,
        verify_partition_hash: bool = False,
    ) -> None:
        """Verify the Episode still owns its reset-time immutable partition."""
        groups = self.groups
        if groups is not self.factorized_octree_groups:
            raise RuntimeError("factorized octree groups object was replaced")
        if id(groups) != self.factorized_group_object_id:
            raise RuntimeError("factorized octree groups identity changed")
        if not isinstance(groups, OctreeGroups):
            raise RuntimeError("factorized groups must remain OctreeGroups")
        if groups.grouping_method != FACTORIZED_GROUPING_METHOD:
            raise RuntimeError("factorized grouping method changed")
        if (
            self.frameNum != groups.actual_group_count
            or self.factorized_group_count != groups.actual_group_count
            or len(groups.group_indices) != groups.actual_group_count
        ):
            raise RuntimeError("factorized octree group count changed")
        if not isinstance(groups.group_indices, _FrozenGroupIndexList):
            raise RuntimeError("factorized octree group sequence changed")
        if id(groups.group_indices) != self.factorized_group_sequence_object_id:
            raise RuntimeError("factorized octree group sequence identity changed")
        if (
            self.factorized_group_partition_sha256
            != self._factorized_group_partition_sha256_frozen
        ):
            raise RuntimeError("factorized octree cached partition SHA-256 changed")
        if any(indices.flags.writeable for indices in groups.group_indices):
            raise RuntimeError("factorized octree group indices became writeable")
        if not 0 <= self.current_group_idx < self.frameNum:
            raise RuntimeError("factorized current_group_idx is outside the Episode")
        if len(self.factorized_actions) != self.current_group_idx:
            raise RuntimeError("factorized action history is inconsistent with group index")
        if self.pruning_mode != PRUNING_MODE_OPACITY_BASELINE:
            raise RuntimeError("first-version opacity pruning provenance changed")
        if verify_partition_hash:
            current_hash = _canonical_factorized_group_partition_sha256(
                groups, len(self.ply.vertex_data)
            )
            if current_hash != self.factorized_group_partition_sha256:
                raise RuntimeError("factorized octree partition SHA-256 changed")


    def reset_factorized(
        self,
        scene_info: dict[str, Any] | None = None,
        *,
        quality_interval: int = 8,
        requested_view_count: int = 8,
        noise_tolerance: float = 0.0,
    ) -> np.ndarray:
        """Reset the opacity-only V1 onto one frozen geometry octree.

        The tree is built once from XYZ and fixes the Morton leaf order for the
        Episode.  Opacity, SH, scale, actions, rewards, and quality never alter
        grouping.  Pruning remains group-local low-opacity-first, while fixed
        view whole-scene quality checkpoints constrain heterogeneous effects.
        """
        old_state = self.__dict__.copy()
        try:
            normalized_interval = _strict_checkpoint_int(
                quality_interval, "quality_interval"
            )
            normalized_view_count = _strict_checkpoint_int(
                requested_view_count, "requested_view_count"
            )
            if normalized_interval <= 0:
                raise ValueError("quality_interval must be a positive integer")
            if normalized_view_count <= 0:
                raise ValueError("requested_view_count must be a positive integer")
            normalized_noise = _nonnegative_reward_parameter(
                noise_tolerance, "noise_tolerance"
            )
            staged = self._prepare_factorized_octree_scene(scene_info)
            self.scene_info = staged["scene_info"]
            self.scene_name = staged["scene_name"]
            self.scene_path = staged["scene_path"]
            self.original_model_path = staged["original_model_path"]
            self.ply_path = staged["ply_path"]
            self.ply = staged["ply"]
            self.original_size_bytes = staged["original_size_bytes"]
            self.target_size_bytes = staged["target_size_bytes"]
            self.groups = staged["groups"]
            self.frameNum = staged["groups"].actual_group_count
            self.factorized_octree_groups = staged["groups"]
            self.factorized_grouping_method = FACTORIZED_GROUPING_METHOD
            self.factorized_grouping_version = FACTORIZED_GROUPING_VERSION
            self.factorized_group_traversal = FACTORIZED_GROUP_TRAVERSAL
            self.factorized_group_partition_sha256 = staged["partition_sha256"]
            self._factorized_group_partition_sha256_frozen = staged[
                "partition_sha256"
            ]
            self.factorized_group_object_id = id(staged["groups"])
            self.factorized_group_sequence_object_id = id(
                staged["groups"].group_indices
            )
            self.factorized_group_count = staged["groups"].actual_group_count
            self.factorized_octree_target_num_groups = staged["target"]
            self.factorized_octree_min_group_size = staged["minimum_size"]
            self.factorized_octree_max_depth_active = staged["maximum_depth"]
            self.factorized_octree_target_gap = staged["groups"].target_gap
            self.factorized_octree_reached_target = staged["groups"].reached_target
            self.factorized_octree_max_leaf_depth = staged["groups"].max_leaf_depth
            self.factorized_octree_split_count = staged["groups"].split_count
            self.factorized_grouping_summary = staged["summary"]
            self.current_group_idx = 0
            self.done = False
            self.episode = int(old_state["episode"]) + 1
            self.total_t = 0
            self.seqNum = (
                (int(old_state["seqNum"]) + 1) % len(self.scenes)
                if staged["automatic_scene"]
                else int(old_state["seqNum"])
            )

            self.factorized_actions = []
            self.factorized_quality_interval = normalized_interval
            self.factorized_requested_view_count = normalized_view_count
            self.factorized_noise_tolerance = normalized_noise
            self.pruning_mode = PRUNING_MODE_OPACITY_BASELINE
            self.factorized_mode_active = True
            self.factorized_last_checkpoint_groups = 0
            self.factorized_previous_quality_score: float | None = None
            self.factorized_last_quality_score: float | None = None
            self.factorized_last_quality_drop = 0.0
            self.factorized_quality_observed = False
            self.factorized_previous_estimated_size_bytes = float(
                self.original_size_bytes
            )
            self._reset_factorized_episode_logs()
            self._validate_factorized_octree_frozen(verify_partition_hash=True)
            state = self.getObservation_factorized(0)
            if (
                state.shape != (19,)
                or state.dtype != np.float32
                or not np.all(np.isfinite(state))
            ):
                raise RuntimeError(
                    "factorized octree initial state must be finite float32[19]"
                )
            return state
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(old_state)
            raise

    def _factorized_estimated_size_ratio(
        self,
        actions: list[tuple[int, int]] | None = None,
    ) -> float:
        """Estimate V2 compact size while identity-filling undecided groups."""
        if self.ply is None or self.groups is None:
            raise RuntimeError("No scene is loaded; call reset_factorized() first")
        selected_actions = self.factorized_actions if actions is None else actions
        ratio = float(
            estimate_size_ratio_from_factorized_actions(
                self.groups.group_indices,
                selected_actions,
                total_vertices=len(self.ply.vertex_data),
            )
        )
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise RuntimeError(f"Invalid factorized estimated size ratio: {ratio}")
        return ratio

    def _factorized_estimated_size_bytes(
        self,
        actions: list[tuple[int, int]] | None = None,
    ) -> float:
        """Return the finite positive estimated V2 compact size in bytes."""
        estimated = float(
            self.original_size_bytes * self._factorized_estimated_size_ratio(actions)
        )
        if not np.isfinite(estimated) or estimated <= 0.0:
            raise RuntimeError(f"Invalid factorized estimated size: {estimated}")
        return estimated

    def getObservation_factorized(
        self,
        group_index: int,
        *,
        actions: list[tuple[int, int]] | None = None,
        estimated_size_ratio: float | None = None,
        quality_drop: float | None = None,
        quality_observed: bool | None = None,
    ) -> np.ndarray:
        """Build the 19-dimensional V2 state without scalar action IDs."""
        if not self.factorized_mode_active:
            raise RuntimeError("Call reset_factorized() before requesting a V2 state")
        if group_index < 0 or group_index >= self.frameNum:
            return np.zeros(self.factorized_state_dim, dtype=np.float32)

        selected_actions = self.factorized_actions if actions is None else actions
        decoded_actions = [decode_factorized_action(action) for action in selected_actions]
        pruning_history = [action.pruning_level / 4.0 for action in decoded_actions]
        precision_history = [action.precision_level / 5.0 for action in decoded_actions]
        if estimated_size_ratio is None:
            normalized_size_ratio = self._factorized_estimated_size_ratio(
                selected_actions
            )
        else:
            normalized_size_ratio = _finite_reward_input(
                estimated_size_ratio, "estimated_size_ratio"
            )
            if normalized_size_ratio <= 0.0:
                raise ValueError("estimated_size_ratio must be greater than zero")

        observed = (
            self.factorized_quality_observed
            if quality_observed is None
            else bool(quality_observed)
        )
        if observed:
            normalized_quality_drop = _finite_reward_input(
                self.factorized_last_quality_drop
                if quality_drop is None
                else quality_drop,
                "quality_drop",
            )
        else:
            normalized_quality_drop = 0.0

        total_groups = max(int(self.frameNum), 1)
        global_features = np.asarray(
            [
                group_index / float(total_groups),
                (total_groups - group_index) / float(total_groups),
                normalized_size_ratio,
                normalized_quality_drop,
                float(self.quality_epsilon) - normalized_quality_drop,
                1.0 if observed else 0.0,
                float(np.mean(pruning_history)) if pruning_history else 0.0,
                pruning_history[-1] if pruning_history else 0.0,
                float(np.mean(precision_history)) if precision_history else 0.0,
                precision_history[-1] if precision_history else 0.0,
            ],
            dtype=np.float32,
        )
        observation = np.concatenate(
            [self._group_feature(group_index).astype(np.float32), global_features]
        ).astype(np.float32)
        if observation.shape != (self.factorized_state_dim,):
            raise RuntimeError(
                f"Factorized observation has shape {observation.shape}, expected "
                f"({self.factorized_state_dim},)"
            )
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("Factorized observation contains non-finite values")
        return observation

    @staticmethod
    def _safe_factorized_scene_name(scene_name: Any) -> str:
        """Convert a scene name to one safe path component."""
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(scene_name).strip())
        safe_name = safe_name.strip("._-")
        return safe_name or "scene"

    @staticmethod
    def _public_factorized_compression_stats(stats: dict[str, Any]) -> dict[str, Any]:
        """Remove per-vertex arrays and private compact payloads from public info."""
        public: dict[str, Any] = {}
        for key, value in stats.items():
            if key == "_compact_aux" or isinstance(value, np.ndarray):
                continue
            if isinstance(value, np.generic):
                public[key] = value.item()
            elif isinstance(value, dict):
                public[key] = GS_Environment._public_factorized_compression_stats(value)
            elif isinstance(value, (list, tuple)) and any(
                isinstance(item, np.ndarray) for item in value
            ):
                continue
            else:
                public[key] = value
        return public


    def _factorized_pruning_public_info(self) -> dict[str, Any]:
        """Return the fixed opacity-baseline provenance for formal V1."""
        return {
            "pruning_mode": "opacity_baseline",
            "pruning_policy": "group_local_low_opacity_first",
            "pruning_importance_source": "raw_ply_opacity",
            "pruning_importance_version": "opacity_baseline_v1",
            "pruning_is_multiview": False,
            "pruning_uses_transmittance": False,
            "pruning_uses_background_replaceability": False,
        }

    @staticmethod
    def _validate_opacity_baseline_checkpoint_stats(
        raw_compression_stats: dict[str, Any],
    ) -> None:
        """Reject any checkpoint report that is not strict opacity baseline."""
        if raw_compression_stats.get("pruning_mode") != "opacity_baseline":
            raise RuntimeError("compressor pruning_mode is not opacity_baseline")
        if raw_compression_stats.get("pruning_policy") not in {
            "group_local_low_opacity_first",
            "no_pruning_requested",
        }:
            raise RuntimeError("compressor pruning_policy is not first-version compatible")
        for name in (
            "pruning_is_multiview",
            "pruning_uses_transmittance",
            "pruning_uses_background_replaceability",
        ):
            if raw_compression_stats.get(name) is not False:
                raise RuntimeError(f"opacity baseline requires {name}=False")

    def _prepare_factorized_model_dir(
        self,
        compressed_ply_path: str | Path,
        groups_processed: int,
        is_terminal: bool,
    ) -> Path:
        """Create a clean model directory unique to one V2 checkpoint."""
        safe_scene = self._safe_factorized_scene_name(self.scene_name)
        artifact_name = "terminal" if is_terminal else f"checkpoint_{groups_processed:04d}"
        model_dir = (
            self.output_root
            / "factorized_compressed_models"
            / safe_scene
            / f"episode_{self.episode:04d}"
            / artifact_name
        )
        if model_dir.is_symlink() or model_dir.is_file():
            model_dir.unlink()
        elif model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir = ensure_model_structure_from_ply(
            compressed_ply_path,
            model_dir,
            iteration=self.iteration,
        )
        if self.original_model_path is not None:
            original_cfg = Path(self.original_model_path) / "cfg_args"
            if original_cfg.is_file():
                shutil.copy2(original_cfg, model_dir / "cfg_args")
        return model_dir

    def _factorized_manifest_path(self) -> Path:
        """Return the scene/resolution/view-specific persistent manifest path."""
        safe_scene = self._safe_factorized_scene_name(self.scene_name)
        return (
            self.quality_cache_dir
            / "fixed_view_manifests"
            / (
                f"{safe_scene}_res{self.resolution}_"
                f"views{self.factorized_requested_view_count}.json"
            )
        )

    def _factorized_original_score_cache_path(self) -> Path:
        """Return the first-version fixed-view original-score cache path."""
        safe_scene = self._safe_factorized_scene_name(self.scene_name)
        return (
            self.quality_cache_dir
            / "factorized_original_scores"
            / (
                f"{safe_scene}_res{self.resolution}_"
                f"views{self.factorized_requested_view_count}.json"
            )
        )

    def _load_factorized_original_score_cache(
        self,
        cache_path: Path,
        manifest_path: Path,
        selected_relative_paths: tuple[str, ...],
    ) -> dict[str, Any] | None:
        """Load an original score only when all fixed-view metadata matches."""
        if self.force_recompute_original_score or not self.cache_original_score:
            return None
        cached = load_score_cache(cache_path)
        if not isinstance(cached, dict):
            return None
        try:
            valid = (
                cached.get("scene") == self.scene_name
                and int(cached.get("resolution")) == self.resolution
                and int(cached.get("requested_view_count"))
                == self.factorized_requested_view_count
                and str(cached.get("manifest_path")) == str(manifest_path)
                and tuple(cached.get("selected_relative_paths", ()))
                == selected_relative_paths
                and np.isfinite(float(cached.get("score")))
            )
        except (TypeError, ValueError):
            valid = False
        return dict(cached) if valid else None

    def _evaluate_factorized_checkpoint(
        self,
        candidate_actions: list[tuple[int, int]],
        compressed_ply_path: Path,
        groups_processed: int,
        is_terminal: bool,
    ) -> dict[str, Any]:
        """Evaluate one complete-scene V2 checkpoint without mutating episode state."""
        previous_score = self.factorized_previous_quality_score
        checkpoint: dict[str, Any] = {
            "quality_observed": False,
            "quality_mode": "disabled",
            "reward_mode": "disabled",
            "original_score": None,
            "previous_score": previous_score,
            "current_score": None,
            "quality_result": None,
            "render_info": {},
            "compressed_model_dir": None,
            "fixed_view_manifest_path": "",
            "selected_view_count": 0,
        }

        if self.use_dummy_reward:
            decoded = [decode_factorized_action(action) for action in candidate_actions]
            mean_pruning = float(
                np.mean([action.pruning_level / 4.0 for action in decoded])
            )
            mean_precision = float(
                np.mean([action.precision_level / 5.0 for action in decoded])
            )
            dummy_drop = 0.05 * mean_pruning + 0.05 * mean_precision
            original_score = 1.0 if self.score_higher_is_better else 0.0
            current_score = (
                original_score - dummy_drop
                if self.score_higher_is_better
                else original_score + dummy_drop
            )
            previous_score = (
                original_score
                if self.factorized_previous_quality_score is None
                else self.factorized_previous_quality_score
            )
            quality_result = compute_quality_block_reward(
                original_score,
                previous_score,
                current_score,
                higher_is_better=self.score_higher_is_better,
                quality_epsilon=self.quality_epsilon,
                incremental_alpha=self.quality_dense_alpha,
                violation_alpha=self.quality_violation_alpha,
                noise_tolerance=self.factorized_noise_tolerance,
            )
            checkpoint.update(
                {
                    "quality_observed": True,
                    "quality_mode": "dummy_factorized",
                    "reward_mode": "dummy_factorized",
                    "original_score": original_score,
                    "previous_score": previous_score,
                    "current_score": current_score,
                    "quality_result": quality_result,
                }
            )
            return checkpoint

        if not self.use_render:
            return checkpoint
        if self.gaussian_splatting_dir is None:
            raise ValueError("gaussian_splatting_dir is required when use_render=True")
        if self.scene_path is None:
            raise ValueError(
                "source_path is required when use_render=True. "
                f"Scene '{self.scene_name}' has no source_path."
            )

        original_model_dir = self._prepare_original_model_dir()
        compressed_model_dir = self._prepare_factorized_model_dir(
            compressed_ply_path,
            groups_processed,
            is_terminal,
        )
        safe_scene = self._safe_factorized_scene_name(self.scene_name)
        artifact_name = "terminal" if is_terminal else f"checkpoint_{groups_processed:04d}"
        subset_output_root = (
            self.output_root
            / "factorized_quality_subsets"
            / safe_scene
            / f"episode_{self.episode:04d}"
            / artifact_name
        )
        manifest_path = self._factorized_manifest_path()
        render_info = render_scene_pair_fixed_subset(
            self.gaussian_splatting_dir,
            original_model_dir,
            compressed_model_dir,
            self.scene_path,
            subset_output_root=subset_output_root,
            manifest_path=manifest_path,
            requested_view_count=self.factorized_requested_view_count,
            iteration=self.iteration,
            resolution=self.resolution,
            python_executable=self.render_python_executable,
        )
        selected_paths = tuple(
            str(path) for path in render_info.get("selected_relative_paths", ())
        )
        if not selected_paths:
            raise RuntimeError("Fixed-view rendering returned an empty camera subset")
        returned_manifest = Path(render_info.get("manifest_path", manifest_path))
        selected_view_count = int(
            render_info.get("selected_view_count", len(selected_paths))
        )
        if selected_view_count != len(selected_paths):
            raise RuntimeError("Fixed-view selected_view_count does not match its paths")
        checkpoint.update(
            {
                "quality_mode": "render_only_fixed_subset",
                "reward_mode": "render_only_fixed_subset",
                "render_info": render_info,
                "compressed_model_dir": compressed_model_dir,
                "fixed_view_manifest_path": str(returned_manifest),
                "selected_view_count": selected_view_count,
            }
        )
        if not self.use_crossscore:
            return checkpoint

        cache_path = self._factorized_original_score_cache_path()
        original_score_info = self._load_factorized_original_score_cache(
            cache_path,
            returned_manifest,
            selected_paths,
        )
        if original_score_info is None:
            original_score_info = self._score_with_crossscore(
                render_info["original_render_subset_dir"],
                render_info["original_gt_subset_dir"],
                self.quality_cache_dir
                / "factorized_scores"
                / safe_scene
                / "original",
                tag="factorized_original",
            )
            original_score = _finite_reward_input(
                original_score_info["score"], "original_score"
            )
            if self.cache_original_score:
                save_score_cache(
                    cache_path,
                    original_score,
                    metadata={
                        "scene": self.scene_name,
                        "resolution": self.resolution,
                        "requested_view_count": self.factorized_requested_view_count,
                        "manifest_path": str(returned_manifest),
                        "selected_relative_paths": list(selected_paths),
                        "score_key": original_score_info.get("score_key", ""),
                        "parser_mode": original_score_info.get("parser_mode", ""),
                    },
                )
        else:
            original_score = _finite_reward_input(
                original_score_info["score"], "cached original_score"
            )

        compressed_score_info = self._score_with_crossscore(
            render_info["compressed_render_subset_dir"],
            render_info["compressed_gt_subset_dir"],
            self.quality_cache_dir
            / "factorized_scores"
            / safe_scene
            / f"episode_{self.episode:04d}"
            / artifact_name,
            tag=f"factorized_{artifact_name}",
        )
        current_score = _finite_reward_input(
            compressed_score_info["score"], "compressed_score"
        )
        previous_score = (
            original_score
            if self.factorized_previous_quality_score is None
            else self.factorized_previous_quality_score
        )
        quality_result = compute_quality_block_reward(
            original_score,
            previous_score,
            current_score,
            higher_is_better=self.score_higher_is_better,
            quality_epsilon=self.quality_epsilon,
            incremental_alpha=self.quality_dense_alpha,
            violation_alpha=self.quality_violation_alpha,
            noise_tolerance=self.factorized_noise_tolerance,
        )
        checkpoint.update(
            {
                "quality_observed": True,
                "quality_mode": "crossscore_fixed_subset",
                "reward_mode": "crossscore_fixed_subset",
                "original_score": original_score,
                "previous_score": previous_score,
                "current_score": current_score,
                "quality_result": quality_result,
                "original_score_info": original_score_info,
                "compressed_score_info": compressed_score_info,
            }
        )
        return checkpoint

    def step_factorized(
        self,
        action: Any,
    ) -> tuple[np.ndarray, float, float, bool, dict[str, Any]]:
        """Execute one transactional V2 action with independent action axes."""
        if not self.factorized_mode_active:
            raise RuntimeError("Call reset_factorized() before step_factorized()")
        if self.done:
            raise RuntimeError(
                "Factorized episode is done. Call reset_factorized() before stepping again."
            )
        self._validate_factorized_octree_frozen()
        if len(self.factorized_actions) != self.current_group_idx:
            raise RuntimeError("Factorized action history is inconsistent with group index")

        raw_action = np.asarray(action)
        if raw_action.shape != (2,):
            raise ValueError(
                "factorized action must have shape (2,) in "
                "[pruning_level, precision_level] order"
            )
        if raw_action.dtype == np.dtype(bool) or not np.issubdtype(
            raw_action.dtype, np.number
        ):
            raise ValueError("factorized action must contain two numeric values")
        continuous_action = raw_action.astype(np.float64, copy=True)
        if not np.all(np.isfinite(continuous_action)):
            raise ValueError("factorized action values must be finite")

        decoded_action = decode_factorized_action(continuous_action)
        executed_action = (
            int(decoded_action.pruning_level),
            int(decoded_action.precision_level),
        )
        candidate_actions = self.factorized_actions + [executed_action]
        group_idx = self.current_group_idx
        groups_processed = len(candidate_actions)
        is_terminal = groups_processed == self.frameNum
        size_before = float(self.factorized_previous_estimated_size_bytes)
        estimated_size_ratio_after = self._factorized_estimated_size_ratio(
            candidate_actions
        )
        estimated_size_after = float(
            self.original_size_bytes * estimated_size_ratio_after
        )
        if not np.isfinite(estimated_size_after) or estimated_size_after <= 0.0:
            raise RuntimeError("Candidate factorized estimated size is invalid")
        dense_size_result = compute_dense_size_reward(
            size_before,
            estimated_size_after,
        )
        quality_checkpoint_due = should_run_quality_checkpoint(
            groups_processed,
            total_groups=self.frameNum,
            last_checkpoint_groups=self.factorized_last_checkpoint_groups,
            interval=self.factorized_quality_interval,
        )
        if quality_checkpoint_due:
            self._validate_factorized_octree_frozen(verify_partition_hash=True)

        compressed_ply_path: Path | None = None
        raw_compression_stats: dict[str, Any] = {}
        public_compression_stats: dict[str, Any] = {}
        checkpoint_result: dict[str, Any] = {
            "quality_observed": False,
            "quality_mode": "disabled",
            "reward_mode": "disabled",
            "original_score": None,
            "previous_score": self.factorized_previous_quality_score,
            "current_score": None,
            "quality_result": None,
            "render_info": {},
            "compressed_model_dir": None,
            "fixed_view_manifest_path": "",
            "selected_view_count": 0,
        }
        block_start: int | None = None
        block_end: int | None = None
        if quality_checkpoint_due:
            block_start, block_end = quality_block_step_bounds(
                self.factorized_last_checkpoint_groups,
                groups_processed,
                interval=self.factorized_quality_interval,
            )
            artifact_tag = (
                "terminal" if is_terminal else f"checkpoint_{groups_processed:04d}"
            )
            compressed_ply_path, raw_compression_stats = (
                self.compressor.compress_scene_factorized(
                    self.ply,
                    self.groups.group_indices,
                    candidate_actions,
                    scene_name=self.scene_name,
                    episode=self.episode,
                    original_size_bytes=self.original_size_bytes,
                    artifact_tag=artifact_tag,
                    write_compact=is_terminal,
                )
            )
            compressed_ply_path = Path(compressed_ply_path)
            if not compressed_ply_path.is_file():
                raise RuntimeError("Factorized checkpoint did not create a render PLY")
            if "_compact_aux" in raw_compression_stats:
                raise RuntimeError("Factorized compressor leaked private compact arrays")
            if bool(raw_compression_stats.get("compact_written")) != is_terminal:
                raise RuntimeError("Factorized compressor compact_written is inconsistent")
            self._validate_opacity_baseline_checkpoint_stats(
                raw_compression_stats
            )
            checkpoint_result = self._evaluate_factorized_checkpoint(
                candidate_actions,
                compressed_ply_path,
                groups_processed,
                is_terminal,
            )
            public_compression_stats = self._public_factorized_compression_stats(
                raw_compression_stats
            )

        terminal_correction_reward = 0.0
        compact_size_bytes = 0
        compact_size_ratio: float | None = None
        render_ply_size_bytes = (
            int(compressed_ply_path.stat().st_size)
            if compressed_ply_path is not None
            else 0
        )
        if is_terminal:
            if not quality_checkpoint_due:
                raise RuntimeError("Terminal factorized step must run a checkpoint")
            compact_size_bytes = int(
                raw_compression_stats.get("compact_size_bytes", 0) or 0
            )
            if (
                not raw_compression_stats.get("compact_written")
                or compact_size_bytes <= 0
            ):
                raise RuntimeError(
                    "Terminal V2 compression must write a positive-size compact package"
                )
            if raw_compression_stats.get("compact_format") != (
                "rl_factorized_3dgs_compact_v2"
            ):
                raise RuntimeError("Terminal compact package is not V2 format")
            compact_package_path = Path(
                raw_compression_stats.get("compact_package_path", "")
            )
            if (
                not compact_package_path.is_file()
                or compact_package_path.stat().st_size != compact_size_bytes
            ):
                raise RuntimeError(
                    "Terminal compact_size_bytes must equal the real compact file size"
                )
            terminal_correction = compute_terminal_size_correction(
                estimated_final_size=estimated_size_after,
                actual_compact_size=compact_size_bytes,
            )
            terminal_correction_reward = terminal_correction.reward
            compact_size_ratio = compact_size_bytes / float(self.original_size_bytes)

        reward_P = float(dense_size_result.reward + terminal_correction_reward)
        quality_result = checkpoint_result.get("quality_result")
        quality_observed = bool(checkpoint_result.get("quality_observed", False))
        reward_D = float(quality_result.reward) if quality_result is not None else 0.0
        current_quality_drop = (
            float(quality_result.current_quality_drop)
            if quality_result is not None
            else None
        )
        current_quality_score = checkpoint_result.get("current_score")
        actual_or_estimated_size = (
            float(compact_size_bytes) if is_terminal else estimated_size_after
        )
        size_ratio = (
            float(compact_size_ratio)
            if compact_size_ratio is not None
            else estimated_size_ratio_after
        )
        left_bitbudget = float(self.target_size_bytes - actual_or_estimated_size)

        render_info = checkpoint_result.get("render_info", {}) or {}
        info: dict[str, Any] = {
            "action_mode": "factorized_v2_5x6",
            "action_continuous": continuous_action.astype(float).tolist(),
            "executed_pruning_level": decoded_action.pruning_level,
            "executed_precision_level": decoded_action.precision_level,
            "executed_pruning_rate": decoded_action.pruning_rate,
            "executed_sh_degree": decoded_action.sh_degree,
            "executed_sh_bit": decoded_action.sh_bit,
            "executed_geo_bit": decoded_action.geo_bit,
            "groups_processed": groups_processed,
            "total_group_count": self.frameNum,
            "current_group_index": group_idx,
            "size_ratio": float(size_ratio),
            "estimated_size_ratio": float(estimated_size_ratio_after),
            "left_bitbudget": left_bitbudget,
            "is_terminal": is_terminal,
            "quality_checkpoint_due": quality_checkpoint_due,
            "quality_observed": quality_observed,
            "quality_target_ready": quality_observed,
            "quality_reward_is_block_target": quality_observed,
            "quality_block_reward": reward_D if quality_observed else None,
            "quality_score": (
                float(current_quality_score)
                if current_quality_score is not None
                else None
            ),
            "quality_drop": current_quality_drop,
            "quality_feasible": (
                bool(quality_result.quality_feasible)
                if quality_result is not None
                else None
            ),
            "quality_violation": (
                float(quality_result.quality_violation)
                if quality_result is not None
                else None
            ),
            "quality_block_start_step_index": block_start,
            "quality_block_end_step_index": block_end,
            "quality_block_length": (
                block_end - block_start + 1
                if block_start is not None and block_end is not None
                else 0
            ),
            "quality_checkpoint_groups_processed": (
                groups_processed if quality_checkpoint_due else None
            ),
            "previous_quality_score": checkpoint_result.get("previous_score"),
            "original_quality_score": checkpoint_result.get("original_score"),
            "quality_mode": checkpoint_result.get("quality_mode", "disabled"),
            "reward_mode": checkpoint_result.get("reward_mode", "disabled"),
            "fixed_view_manifest_path": checkpoint_result.get(
                "fixed_view_manifest_path", ""
            ),
            "selected_view_count": int(
                checkpoint_result.get("selected_view_count", 0) or 0
            ),
            "dense_size_reward": float(dense_size_result.reward),
            "terminal_size_correction": float(terminal_correction_reward),
            "reward_D": reward_D,
            "reward_P": reward_P,
            "estimated_size_before": size_before,
            "estimated_size_after": estimated_size_after,
            "compact_size_bytes": compact_size_bytes,
            "compact_size_ratio": compact_size_ratio,
            "render_ply_size_bytes": render_ply_size_bytes,
            "compressed_ply_path": (
                str(compressed_ply_path) if compressed_ply_path is not None else ""
            ),
            "compressed_model_dir": (
                str(checkpoint_result["compressed_model_dir"])
                if checkpoint_result.get("compressed_model_dir") is not None
                else ""
            ),
            "compression_stats": public_compression_stats,
            **self._factorized_pruning_public_info(),
            **self._grouping_info(),
        }
        for key, value in render_info.items():
            if key.endswith("_dir") or key == "manifest_path":
                info[key] = str(value)
            elif key == "selected_relative_paths":
                info[key] = tuple(str(path) for path in value)

        # Transaction commit: no episode state above this point is changed.
        self.factorized_actions = candidate_actions
        self.factorized_action_continuous.append(
            continuous_action.astype(float).tolist()
        )
        self.factorized_previous_estimated_size_bytes = estimated_size_after
        if quality_observed:
            self.factorized_previous_quality_score = float(current_quality_score)
            self.factorized_last_quality_score = float(current_quality_score)
            self.factorized_last_quality_drop = float(current_quality_drop)
            self.factorized_quality_observed = True
        if quality_checkpoint_due:
            self.factorized_last_checkpoint_groups = groups_processed
        self.total_t += 1
        self.factorized_action_all.append(
            factorized_action_to_array(executed_action).astype(int).tolist()
        )
        self.factorized_reward_D_all.append(reward_D)
        self.factorized_reward_P_all.append(reward_P)
        self.factorized_quality_checkpoint_all.append(quality_checkpoint_due)
        self.factorized_size_ratio_all.append(float(size_ratio))

        if is_terminal:
            self.done = True
            self.current_group_idx = self.frameNum
            next_state = np.zeros(self.factorized_state_dim, dtype=np.float32)
        else:
            self.current_group_idx += 1
            next_state = self.getObservation_factorized(self.current_group_idx)
        self.last_info = info
        return next_state, reward_D, reward_P, self.done, info

    def _prepare_original_model_dir(self) -> Path:
        if self.original_model_path is None:
            raise ValueError(
                "source_path/model_path are required when render or CrossScore is enabled. "
                f"Scene '{self.scene_name}' has no model_path."
            )
        if self.ply_path is None:
            raise ValueError(f"Scene '{self.scene_name}' has no ply_path.")
        return ensure_model_structure_from_ply(
            self.ply_path,
            self.original_model_path,
            iteration=self.iteration,
        )

    def _score_with_crossscore(self, render_dir, gt_dir, score_output_dir, tag):
        if self.crossscore_mode == "placeholder" and not self.crossscore_command_template:
            if self.use_crossscore and not self.allow_crossscore_placeholder:
                raise RuntimeError(
                    "CrossScore mode is placeholder while use_crossscore=True. "
                    "Pass --allow-crossscore-placeholder only for debugging, or set "
                    "crossscore_mode=auto_from_predict_sh / command_template in config."
                )
            score = compute_crossscore_placeholder(render_dir, gt_dir)
            return {
                "score": float(score),
                "score_file": "",
                "score_key": "placeholder",
                "parser_mode": "placeholder",
            }
        if self.crossscore_dir is None:
            raise ValueError("crossscore_dir is required when use_crossscore=True.")
        score_output_dir = Path(score_output_dir)
        score = compute_crossscore_real(
            self.crossscore_dir,
            render_dir,
            gt_dir,
            score_output_dir,
            scene_name=self.scene_name,
            tag=tag,
            python_executable=self.crossscore_python_executable,
            command_template=self.crossscore_command_template,
            score_output=self.crossscore_score_output,
            score_parse_mode=self.crossscore_score_parse_mode,
            preferred_score_key=self.crossscore_preferred_score_key,
            ckpt=self.crossscore_ckpt,
            config=self.crossscore_config,
            allow_image_fallback=self.crossscore_allow_image_fallback,
        )
        score_json = score_output_dir / "score.json"
        if score_json.exists():
            import json

            score_info = json.loads(score_json.read_text(encoding="utf-8-sig"))
            score_info["score"] = float(score_info.get("score", score))
            return score_info
        return {
            "score": float(score),
            "score_file": "",
            "score_key": "",
            "parser_mode": self.crossscore_score_parse_mode,
        }

    def _grouping_info(self) -> dict[str, Any]:
        """Return frozen first-version octree provenance only."""
        if (
            not self.factorized_mode_active
            or self.groups is not self.factorized_octree_groups
            or not isinstance(self.groups, OctreeGroups)
            or self.factorized_grouping_summary is None
        ):
            raise RuntimeError("factorized octree grouping is not initialized")
        summary = self.factorized_grouping_summary
        depth_histogram = {
            int(depth): int(count)
            for depth, count in summary["depth_histogram"].items()
        }
        return {
            "grouping_method": self.factorized_grouping_method,
            "grouping_version": self.factorized_grouping_version,
            "grouping_traversal": self.factorized_group_traversal,
            "grouping_partition_sha256": self.factorized_group_partition_sha256,
            "grouping_is_octree": True,
            "num_groups": int(self.factorized_group_count),
            "actual_num_groups": int(summary["actual_group_count"]),
            "target_num_groups": int(summary["target_num_groups"]),
            "target_gap": int(summary["target_gap"]),
            "reached_target": bool(summary["reached_target"]),
            "min_group_size": int(summary["min_group_size"]),
            "octree_max_depth": int(summary["max_depth"]),
            "octree_max_leaf_depth": int(summary["max_leaf_depth"]),
            "octree_split_count": int(summary["split_count"]),
            "octree_unsplittable_leaf_count": int(
                summary["unsplittable_leaf_count"]
            ),
            "octree_root_cube_side": float(summary["root_cube_side"]),
            "octree_leaf_order": self.groups.leaf_order,
            "octree_center_tie_rule": self.groups.center_tie_rule,
            "octree_target_policy": self.groups.target_policy,
            "octree_min_size_policy": self.groups.min_size_policy,
            "octree_child_order": OCTREE_CHILD_ORDER,
            "octree_min_leaf_size": int(summary["min_leaf_size"]),
            "octree_median_leaf_size": float(summary["median_leaf_size"]),
            "octree_max_leaf_size": int(summary["max_leaf_size"]),
            "octree_small_group_count": int(summary["small_group_count"]),
            "octree_depth_histogram": depth_histogram,
        }

    def _group_feature(self, group_index: int) -> np.ndarray:
        """Extract the unchanged nine features for the current octree leaf."""
        indices = self.groups.group_indices[group_index]
        return _extract_octree_group_features(
            self.ply.vertex_data,
            indices,
            total_gaussians=len(self.ply.vertex_data),
            bbox_min=self.groups.bbox_min,
            bbox_max=self.groups.bbox_max,
            opacity_low_threshold=self.opacity_low_threshold,
            small_group_flag=self.groups.small_group_flags[group_index],
        )


def validate_factorized_environment_pipeline() -> bool:
    """Validate the complete opacity-only 15-leaf first-version pipeline."""

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    def require_error(
        exception_type: type[BaseException],
        callback: Callable[[], object],
        message: str,
    ) -> None:
        try:
            callback()
        except exception_type:
            return
        raise AssertionError(message)

    def validation_xyz() -> np.ndarray:
        points: list[list[float]] = []
        for child_id in range(8):
            coordinate = [
                -0.75 if child_id & 1 == 0 else -0.25,
                -0.75 if child_id & 2 == 0 else -0.25,
                -0.75 if child_id & 4 == 0 else -0.25,
            ]
            points.extend([coordinate, coordinate])
        for child_id in range(1, 8):
            coordinate = [
                0.5 if child_id & 1 else -0.5,
                0.5 if child_id & 2 else -0.5,
                0.5 if child_id & 4 else -0.5,
            ]
            points.extend([coordinate, coordinate])
        return np.asarray(points, dtype=np.float32)

    def write_validation_ply(
        path: Path,
        xyz: np.ndarray,
        *,
        attribute_offset: float = 0.0,
    ) -> None:
        names = [
            "x", "y", "z", "opacity",
            "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3",
            "f_dc_0", "f_dc_1", "f_dc_2",
            *[f"f_rest_{index}" for index in range(45)],
        ]
        vertices = np.zeros(
            len(xyz), dtype=np.dtype([(name, "<f4") for name in names])
        )
        vertices["x"], vertices["y"], vertices["z"] = xyz.T
        vertices["opacity"] = (
            np.linspace(-2.0, 2.0, len(xyz), dtype=np.float32)
            + np.float32(attribute_offset)
        )
        for field_index, name in enumerate(names[4:], start=4):
            vertices[name] = np.linspace(
                0.01 + field_index * 0.001 + attribute_offset,
                0.99 + field_index * 0.001 + attribute_offset,
                len(xyz),
                dtype=np.float32,
            )
        header = [
            "ply", "format binary_little_endian 1.0",
            f"element vertex {len(vertices)}",
            *[f"property float {name}" for name in names],
            "end_header",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.write(("\n".join(header) + "\n").encode("ascii"))
            vertices.tofile(handle)

    def make_environment(
        root: Path,
        ply_path: Path,
        **overrides: Any,
    ) -> GS_Environment:
        parameters: dict[str, Any] = {
            "scenes": [{"name": "octree_validation", "ply_path": str(ply_path)}],
            "output_root": root / "outputs",
            "target_num_groups": 15,
            "min_group_size": 2,
            "factorized_octree_max_depth": 2,
            "use_dummy_reward": True,
            "use_render": False,
            "use_crossscore": False,
        }
        parameters.update(overrides)
        return GS_Environment(**parameters)

    with tempfile.TemporaryDirectory(
        prefix="factorized_octree_environment_validation_"
    ) as temporary_directory:
        root = Path(temporary_directory)
        xyz = validation_xyz()
        ply_path = root / "scene.ply"
        write_validation_ply(ply_path, xyz)
        expected = octree_group_indices(
            xyz, target_num_groups=15, min_group_size=2, max_depth=2
        )
        require(expected.actual_group_count == 15, "validation geometry needs 15 leaves")

        environment = make_environment(root / "main", ply_path)
        initial_state = environment.reset_factorized(quality_interval=8)
        groups = environment.groups
        require(isinstance(groups, OctreeGroups), "V2 groups must be OctreeGroups")
        require(groups is environment.factorized_octree_groups, "group object was copied")
        require(groups.actual_group_count == environment.frameNum == 15, "wrong leaf count")
        require(groups.grouping_method == OCTREE_GROUPING_METHOD, "wrong grouping method")
        require(
            groups.node_ids == expected.node_ids
            and groups.node_depths == expected.node_depths
            and groups.node_path_codes == expected.node_path_codes
            and all(
                np.array_equal(left, right)
                for left, right in zip(groups.group_indices, expected.group_indices)
            ),
            "Environment changed the octree module leaf order",
        )
        flattened = np.concatenate(groups.group_indices)
        require(
            np.array_equal(np.sort(flattened), np.arange(len(xyz)))
            and len(np.unique(flattened)) == len(xyz),
            "octree leaves do not cover every Gaussian exactly once",
        )
        require(
            all(not indices.flags.writeable for indices in groups.group_indices),
            "octree group arrays are writeable",
        )
        require(
            initial_state.shape == (19,)
            and initial_state.dtype == np.float32
            and np.all(np.isfinite(initial_state)),
            "initial factorized state is not finite float32[19]",
        )

        first_signature = (
            groups.node_ids,
            groups.node_depths,
            groups.node_path_codes,
            tuple(np.array(indices, copy=True) for indices in groups.group_indices),
            environment.factorized_group_partition_sha256,
        )
        repeated_state = environment.reset_factorized(
            environment.scene_info, quality_interval=8
        )
        repeated = environment.groups
        require(
            repeated.node_ids == first_signature[0]
            and repeated.node_depths == first_signature[1]
            and repeated.node_path_codes == first_signature[2]
            and all(
                np.array_equal(left, right)
                for left, right in zip(repeated.group_indices, first_signature[3])
            )
            and environment.factorized_group_partition_sha256 == first_signature[4]
            and repeated_state.shape == (19,),
            "same PLY and parameters did not reset deterministically",
        )

        attribute_ply = root / "attribute_changed.ply"
        write_validation_ply(attribute_ply, xyz, attribute_offset=7.0)
        attribute_environment = make_environment(root / "attributes", attribute_ply)
        attribute_environment.reset_factorized()
        require(
            attribute_environment.groups.node_ids == first_signature[0]
            and attribute_environment.groups.node_depths == first_signature[1]
            and all(
                np.array_equal(left, right)
                for left, right in zip(
                    attribute_environment.groups.group_indices, first_signature[3]
                )
            )
            and attribute_environment.factorized_group_partition_sha256
            == first_signature[4],
            "opacity/SH/scale changed geometry-only grouping",
        )

        changed_xyz = xyz.copy()
        changed_xyz[0:2] = changed_xyz[14:16]
        changed_ply = root / "xyz_changed.ply"
        write_validation_ply(changed_ply, changed_xyz)
        changed_environment = make_environment(root / "xyz_changed", changed_ply)
        changed_environment.reset_factorized()
        require(
            changed_environment.factorized_group_partition_sha256
            != first_signature[4],
            "a deliberate leaf-boundary crossing did not change partition hash",
        )

        # Use the repeat-reset Episode for a complete 15-step dummy rollout.
        compressor_calls: list[dict[str, Any]] = []
        original_compress = environment.compressor.compress_scene_factorized

        def tracking_compress(*args: Any, **kwargs: Any) -> tuple[Path, dict[str, Any]]:
            require(
                len(args) >= 3 and args[1] is environment.groups.group_indices,
                "compressor did not receive the frozen octree group object",
            )
            result = original_compress(*args, **kwargs)
            compressor_calls.append(
                {
                    "group_object_id": id(args[1]),
                    "action_count": len(args[2]),
                    "stats": result[1],
                }
            )
            return result

        environment.compressor.compress_scene_factorized = tracking_compress
        infos: list[dict[str, Any]] = []
        reward_p_values: list[float] = []
        group_object_id = id(environment.groups)
        group_sequence_id = id(environment.groups.group_indices)
        for step_index in range(15):
            _, _, reward_p, done, info = environment.step_factorized((2.0, 3.0))
            infos.append(info)
            reward_p_values.append(reward_p)
            require(id(environment.groups) == group_object_id, "group object changed")
            require(
                id(environment.groups.group_indices) == group_sequence_id,
                "group sequence changed",
            )
            if step_index < 7:
                require(not info["quality_checkpoint_due"], "early checkpoint ran")
            if step_index < 14:
                require(not done, "15-leaf Episode terminated early")
        checkpoint_info, terminal_info = infos[7], infos[14]
        require(len(compressor_calls) == 2, "compressor must run once per checkpoint")
        require(
            compressor_calls[0]["action_count"] == 8
            and compressor_calls[0]["stats"]["identity_filled_group_count"] == 7
            and compressor_calls[1]["action_count"] == 15
            and compressor_calls[1]["stats"]["identity_filled_group_count"] == 0,
            "checkpoint whole-scene identity fill is incorrect",
        )
        require(
            checkpoint_info["quality_checkpoint_due"]
            and checkpoint_info["quality_block_start_step_index"] == 0
            and checkpoint_info["quality_block_end_step_index"] == 7
            and checkpoint_info["grouping_partition_sha256"] == first_signature[4],
            "leaf 8 checkpoint metadata is incorrect",
        )
        require(
            terminal_info["quality_checkpoint_due"]
            and terminal_info["quality_block_start_step_index"] == 8
            and terminal_info["quality_block_end_step_index"] == 14
            and terminal_info["quality_block_length"] == 7
            and terminal_info["groups_processed"] == 15
            and terminal_info["actual_num_groups"] == 15
            and environment.current_group_idx == environment.frameNum == 15,
            "terminal partial checkpoint metadata is incorrect",
        )
        require(
            len(checkpoint_info["action_continuous"]) == 2,
            "formal action must remain two-dimensional",
        )
        require(
            np.isclose(
                sum(reward_p_values),
                np.log(
                    environment.original_size_bytes
                    / terminal_info["compact_size_bytes"]
                ),
            ),
            "dense size reward and terminal correction did not telescope",
        )
        expected_pruning_info = {
            "pruning_mode": "opacity_baseline",
            "pruning_policy": "group_local_low_opacity_first",
            "pruning_importance_source": "raw_ply_opacity",
            "pruning_importance_version": "opacity_baseline_v1",
            "pruning_is_multiview": False,
            "pruning_uses_transmittance": False,
            "pruning_uses_background_replaceability": False,
        }
        require(
            environment.pruning_mode == PRUNING_MODE_OPACITY_BASELINE
            and all(
                all(info.get(key) == value for key, value in expected_pruning_info.items())
                for info in infos
            ),
            "opacity-baseline provenance changed",
        )
        require(
            not any(
                isinstance(value, np.ndarray)
                for info in infos
                for value in info.values()
            ),
            "public info leaked per-vertex arrays",
        )
        required_grouping_fields = {
            "grouping_method", "grouping_version", "grouping_traversal",
            "grouping_partition_sha256", "grouping_is_octree",
            "target_num_groups", "actual_num_groups",
            "num_groups", "target_gap", "reached_target", "min_group_size",
            "octree_max_depth", "octree_max_leaf_depth", "octree_split_count",
            "octree_unsplittable_leaf_count",
            "octree_root_cube_side", "octree_leaf_order",
            "octree_center_tie_rule", "octree_target_policy",
            "octree_min_size_policy", "octree_child_order",
            "octree_min_leaf_size", "octree_median_leaf_size",
            "octree_max_leaf_size", "octree_small_group_count",
            "octree_depth_histogram",
        }
        require(
            all(required_grouping_fields <= set(info) for info in infos)
            and not any(
                key in info
                for info in infos
                for key in ("group_indices", "node_bbox_mins", "node_bbox_maxs")
            ),
            "public grouping info is incomplete or exposes large arrays",
        )

        impossible = make_environment(
            root / "impossible", ply_path, target_num_groups=14
        )
        impossible.reset_factorized()
        require(
            impossible.frameNum <= 14
            and impossible.groups.target_gap == 14 - impossible.frameNum
            and impossible.groups.actual_group_count <= 14,
            "unreachable target was padded or truncated",
        )
        require_error(
            ValueError,
            make_environment(root / "missing_target", ply_path, target_num_groups=None).reset_factorized,
            "factorized reset accepted a missing target",
        )
        override_environment = make_environment(
            root / "override", ply_path, target_num_groups=None, min_group_size=99,
            factorized_octree_max_depth=0,
        )
        override_environment.reset_factorized(
            {
                "name": "override",
                "ply_path": str(ply_path),
                "factorized_target_num_groups": 15,
                "factorized_octree_min_group_size": 2,
                "factorized_octree_max_depth": 2,
            }
        )
        require(override_environment.frameNum == 15, "scene overrides were ignored")
        for bad_scene in (
            {"factorized_target_num_groups": True},
            {"factorized_target_num_groups": 0},
            {"factorized_octree_min_group_size": False},
            {"factorized_octree_min_group_size": 0},
            {"factorized_octree_max_depth": True},
            {"factorized_octree_max_depth": 21},
        ):
            invalid_scene = {"name": "invalid", "ply_path": str(ply_path), **bad_scene}
            require_error(
                ValueError,
                lambda value=invalid_scene: make_environment(
                    root / "invalid", ply_path
                ).reset_factorized(value),
                "invalid octree parameter was accepted",
            )
        for bad_depth in (True, -1, 21, 2.0):
            require_error(
                ValueError,
                lambda value=bad_depth: make_environment(
                    root / "invalid_constructor", ply_path,
                    factorized_octree_max_depth=value,
                ),
                "invalid constructor max depth was accepted",
            )
        for bad_target in (True, 15.0, "15"):
            require_error(
                ValueError,
                lambda value=bad_target: make_environment(
                    root / "invalid_target_constructor", ply_path,
                    target_num_groups=value,
                ).reset_factorized(),
                "nonstrict constructor target was accepted by factorized reset",
            )

        rollback = make_environment(root / "rollback", ply_path)
        rollback.reset_factorized()
        rollback_snapshot = {
            "scene_info": rollback.scene_info,
            "ply": rollback.ply,
            "groups": rollback.groups,
            "episode": rollback.episode,
            "total_t": rollback.total_t,
            "seqNum": rollback.seqNum,
            "current_group_idx": rollback.current_group_idx,
            "done": rollback.done,
            "active": rollback.factorized_mode_active,
            "actions": rollback.factorized_actions,
            "last_info": rollback.last_info,
        }
        require_error(
            ValueError,
            lambda: rollback.reset_factorized(
                {"name": "bad", "ply_path": str(ply_path),
                 "factorized_target_num_groups": True}
            ),
            "invalid octree reset did not fail",
        )
        require(
            rollback.scene_info is rollback_snapshot["scene_info"]
            and rollback.ply is rollback_snapshot["ply"]
            and rollback.groups is rollback_snapshot["groups"]
            and rollback.episode == rollback_snapshot["episode"]
            and rollback.total_t == rollback_snapshot["total_t"]
            and rollback.seqNum == rollback_snapshot["seqNum"]
            and rollback.current_group_idx == rollback_snapshot["current_group_idx"]
            and rollback.done == rollback_snapshot["done"]
            and rollback.factorized_mode_active == rollback_snapshot["active"]
            and rollback.factorized_actions is rollback_snapshot["actions"]
            and rollback.last_info is rollback_snapshot["last_info"],
            "failed octree reset did not fully roll back",
        )
        replacement = rollback.groups
        rollback.groups = expected
        step_snapshot = (
            len(rollback.factorized_actions), rollback.current_group_idx,
            rollback.total_t, rollback.done,
        )
        require_error(
            RuntimeError,
            lambda: rollback.step_factorized((0.0, 0.0)),
            "replaced group object was accepted",
        )
        require(
            step_snapshot == (
                len(rollback.factorized_actions), rollback.current_group_idx,
                rollback.total_t, rollback.done,
            ),
            "group replacement failure committed an action",
        )
        rollback.groups = replacement
        original_frame_num = rollback.frameNum
        rollback.frameNum += 1
        require_error(
            RuntimeError,
            lambda: rollback.step_factorized((0.0, 0.0)),
            "changed frameNum was accepted",
        )
        rollback.frameNum = original_frame_num

        checkpoint_failure = make_environment(
            root / "checkpoint_failure", ply_path
        )
        checkpoint_failure.reset_factorized(quality_interval=1)
        failure_snapshot = (
            len(checkpoint_failure.factorized_actions),
            checkpoint_failure.current_group_idx,
            checkpoint_failure.total_t,
            checkpoint_failure.done,
        )

        def fail_checkpoint(*_args: Any, **_kwargs: Any) -> object:
            raise RuntimeError("simulated checkpoint failure")

        checkpoint_failure.compressor.compress_scene_factorized = fail_checkpoint
        require_error(
            RuntimeError,
            lambda: checkpoint_failure.step_factorized((2.0, 3.0)),
            "checkpoint failure was not propagated",
        )
        require(
            failure_snapshot == (
                len(checkpoint_failure.factorized_actions),
                checkpoint_failure.current_group_idx,
                checkpoint_failure.total_t,
                checkpoint_failure.done,
            ),
            "checkpoint failure committed an action",
        )

        hash_environment = make_environment(root / "hash", ply_path)
        hash_environment.reset_factorized(quality_interval=1)
        hash_environment.factorized_group_partition_sha256 = "0" * 64
        hash_snapshot = (
            len(hash_environment.factorized_actions),
            hash_environment.current_group_idx,
            hash_environment.total_t,
            hash_environment.done,
        )
        require_error(
            RuntimeError,
            lambda: hash_environment.step_factorized((0.0, 0.0)),
            "checkpoint partition hash tampering was accepted",
        )
        require(
            hash_snapshot == (
                len(hash_environment.factorized_actions),
                hash_environment.current_group_idx,
                hash_environment.total_t,
                hash_environment.done,
            ),
            "hash failure committed an action",
        )

    return True


def validate_factorized_octree_environment() -> bool:
    """Compatibility wrapper for the single formal pipeline validation."""
    return validate_factorized_environment_pipeline()
