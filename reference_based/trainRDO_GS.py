"""First-version dual-critic octree 3DGS compression training.

The formal pipeline uses a frozen best-first octree, direct two-axis pruning
and precision actions, group-local opacity-baseline pruning, independent size
and quality critics, a fixed quality threshold, whole-scene fixed-view quality
checkpoints every eight groups, terminal partial checkpoints, and compact V2
output.  Quality violations select the quality critic for Actor learning;
feasible Episodes select the size critic.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from .config_utils import (
        CODE_ROOT,
        load_config,
        normalize_crossscore_dir,
        normalize_gaussian_splatting_dir,
        resolve_input_path,
        resolve_output_path,
        select_scenes,
    )
    from . import Network_GS as Network
    from . import Transition_GS as trans
    from .compression_ops import PRUNING_MODE_OPACITY_BASELINE
    from .Environment_GS import GS_Environment
except ImportError:
    from config_utils import (
        CODE_ROOT,
        load_config,
        normalize_crossscore_dir,
        normalize_gaussian_splatting_dir,
        resolve_input_path,
        resolve_output_path,
        select_scenes,
    )
    import Network_GS as Network
    import Transition_GS as trans
    from compression_ops import PRUNING_MODE_OPACITY_BASELINE
    from Environment_GS import GS_Environment


DEFAULT_BATCH_SIZE = 8
DEFAULT_BUFFER_SIZE = 5000
DEFAULT_ACTOR_LEARNING_RATE = 1e-4
DEFAULT_CRITIC_LEARNING_RATE = 1e-3
FACTORIZED_ACTION_DIM = 2
FACTORIZED_ACTION_BOUND: np.ndarray = np.asarray([4.0, 5.0], dtype=np.float64)
FACTORIZED_TRAINING_CHECKPOINT_SCHEMA = "factorized_dual_critic_v1"
FACTORIZED_GAMMA_SIZE = 0.99
FACTORIZED_GAMMA_QUALITY = 1.0
FACTORIZED_UPDATE_STEPS = 4
FACTORIZED_ACTOR_GRAD_CLIP_NORM = 10.0
FACTORIZED_V1_PRUNING_MODE = PRUNING_MODE_OPACITY_BASELINE
FACTORIZED_V1_PRUNING_POLICY = "group_local_low_opacity_first"
FACTORIZED_V1_IMPORTANCE_SOURCE = "raw_ply_opacity"
FACTORIZED_V1_IMPORTANCE_VERSION = "opacity_baseline_v1"


def _factorized_v1_pruning_provenance() -> dict[str, Any]:
    """Return immutable provenance for the formal first-version executor."""
    return {
        "pruning_mode": FACTORIZED_V1_PRUNING_MODE,
        "pruning_policy": FACTORIZED_V1_PRUNING_POLICY,
        "pruning_importance_source": FACTORIZED_V1_IMPORTANCE_SOURCE,
        "pruning_importance_version": FACTORIZED_V1_IMPORTANCE_VERSION,
        "pruning_is_multiview": False,
        "pruning_uses_transmittance": False,
        "pruning_uses_background_replaceability": False,
    }




def _validate_factorized_v1_pruning_environment(environment: Any) -> None:
    """Require a reset environment to expose the active opacity-only context."""
    if getattr(environment, "pruning_mode", None) != FACTORIZED_V1_PRUNING_MODE:
        raise RuntimeError("formal environment is not using opacity-baseline pruning")


def _validate_factorized_v1_pruning_info(info: dict[str, Any]) -> None:
    """Validate opacity-baseline provenance returned by every V2 step."""
    if not isinstance(info, dict):
        raise ValueError("step_factorized info must be a dict")
    expected = {
        "pruning_mode": FACTORIZED_V1_PRUNING_MODE,
        "pruning_policy": FACTORIZED_V1_PRUNING_POLICY,
        "pruning_importance_source": FACTORIZED_V1_IMPORTANCE_SOURCE,
        "pruning_importance_version": FACTORIZED_V1_IMPORTANCE_VERSION,
        "pruning_is_multiview": False,
        "pruning_uses_transmittance": False,
        "pruning_uses_background_replaceability": False,
    }
    for name, value in expected.items():
        if info.get(name) != value:
            raise RuntimeError(f"factorized V1 reported invalid {name}")
    compression_stats = info.get("compression_stats")
    if compression_stats in (None, {}):
        return
    if not isinstance(compression_stats, dict):
        raise RuntimeError("factorized compression_stats must be a dict")
    for name in (
        "pruning_is_multiview",
        "pruning_uses_transmittance",
        "pruning_uses_background_replaceability",
    ):
        if compression_stats.get(name) is True:
            raise RuntimeError(f"factorized V1 forbids {name}=True")
    pruned_vertices = compression_stats.get("pruned_vertices", 0)
    if isinstance(pruned_vertices, (bool, np.bool_)) or not isinstance(
        pruned_vertices, (int, np.integer)
    ):
        raise RuntimeError("compression_stats.pruned_vertices must be an integer")
    if int(pruned_vertices) > 0:
        expected = {
            "pruning_policy": FACTORIZED_V1_PRUNING_MODE,
            "pruning_score_source": "opacity_field",
            "pruning_is_multiview": False,
            "pruning_uses_transmittance": False,
            "pruning_uses_background_replaceability": False,
        }
        for name, value in expected.items():
            if compression_stats.get(name) != value:
                raise RuntimeError(
                    f"factorized V1 checkpoint reported invalid {name}: "
                    f"{compression_stats.get(name)!r}"
                )


def _validate_factorized_v1_checkpoint_metadata(
    checkpoint: dict[str, Any],
    checkpoint_name: str,
) -> None:
    """Reject V2 checkpoints without exact formal-V1 pruning provenance."""
    expected = {
        "pruning_mode": FACTORIZED_V1_PRUNING_MODE,
        "pruning_policy": FACTORIZED_V1_PRUNING_POLICY,
        "pruning_importance_source": FACTORIZED_V1_IMPORTANCE_SOURCE,
        "pruning_importance_version": FACTORIZED_V1_IMPORTANCE_VERSION,
        "state_dim": 19,
        "action_dim": FACTORIZED_ACTION_DIM,
        "action_bounds": FACTORIZED_ACTION_BOUND.tolist(),
    }
    missing = [name for name in expected if name not in checkpoint]
    if missing:
        raise ValueError(
            f"{checkpoint_name} predates formal opacity-baseline provenance "
            f"({', '.join(missing)} missing); retrain before formal resume"
        )
    for name, value in expected.items():
        if checkpoint[name] != value:
            raise ValueError(
                f"{checkpoint_name} has incompatible {name}: "
                f"expected {value!r}, got {checkpoint[name]!r}"
            )


@dataclass(frozen=True)
class FactorizedQualityCheckpointRecord:
    episode_id: int; scene: str; local_checkpoint_index: int; local_step_index: int
    groups_processed: int; total_group_count: int; block_start_step_index: int
    block_end_step_index: int; block_length: int; is_terminal: bool
    checkpoint_scope: str; quality_mode: str; quality_observed: bool
    quality_target_ready: bool; quality_reward_is_block_target: bool
    fixed_view_manifest_path: str; selected_view_count: int
    original_subset_quality_score: float | None; previous_subset_quality_score: float | None
    current_subset_quality_score: float | None; previous_subset_quality_drop: float | None
    current_subset_quality_drop: float | None; subset_quality_drop_delta: float | None
    quality_epsilon: float | None; subset_quality_margin: float | None
    subset_quality_feasible: bool | None; subset_quality_violation: float | None
    quality_block_reward: float | None
    estimated_size_bytes_before_block: float; estimated_size_bytes_after_block: float
    estimated_size_ratio_before_block: float; estimated_size_ratio_after_block: float
    block_size_ratio_reduction: float; block_log_size_reward_sum: float
    compact_size_bytes: int | None; compact_size_ratio: float | None
    terminal_size_correction: float | None; block_mean_actor_pruning: float
    block_mean_actor_precision: float; block_mean_environment_pruning: float
    block_mean_environment_precision: float; block_min_environment_pruning: float
    block_max_environment_pruning: float; block_min_environment_precision: float
    block_max_environment_precision: float; block_mean_executed_pruning: float
    block_mean_executed_precision: float; block_min_executed_pruning: int
    block_max_executed_pruning: int; block_min_executed_precision: int
    block_max_executed_precision: int; executed_pruning_histogram_json: str
    executed_precision_histogram_json: str; block_reward_P_sum: float
    checkpoint_reward_D: float


FACTORIZED_QUALITY_CHECKPOINT_LOG_FIELDS = (
    "global_step", "learner_mode", *[item.name for item in fields(FactorizedQualityCheckpointRecord)]
)


def _histogram_json(values: list[int], maximum: int) -> str:
    counts = {str(level): int(sum(value == level for value in values)) for level in range(maximum + 1)}
    return json.dumps(counts, separators=(",", ":"), sort_keys=True, allow_nan=False)


def append_factorized_quality_checkpoint_log(log_path, records, *, scene, global_step_before_episode, learner_mode):
    """Atomically append one strictly validated episode of checkpoint rows."""
    records = tuple(records)
    path = Path(log_path)
    if not records:
        return path
    if not isinstance(scene, str) or not scene or learner_mode != "factorized_dual_critic_ddpg":
        raise ValueError("invalid scene or learner_mode")
    _strict_nonnegative_int(global_step_before_episode, "global_step_before_episode")
    integer_fields = ("episode_id", "local_checkpoint_index", "local_step_index", "groups_processed", "total_group_count", "block_start_step_index", "block_end_step_index", "block_length", "selected_view_count")
    required_bools = ("is_terminal", "quality_observed", "quality_target_ready", "quality_reward_is_block_target")
    optional_bools = ("subset_quality_feasible",)
    episode_id = records[0].episode_id
    for index, record in enumerate(records):
        if not isinstance(record, FactorizedQualityCheckpointRecord):
            raise ValueError("records must contain FactorizedQualityCheckpointRecord")
        for name in integer_fields:
            value = getattr(record, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
                raise ValueError(f"{name} must be a nonnegative strict integer")
        if record.episode_id != episode_id or record.local_checkpoint_index != index:
            raise ValueError("records must have one episode and continuous checkpoint indices")
        if record.scene != scene or record.checkpoint_scope != "fixed_subset":
            raise ValueError("record scene or checkpoint_scope is invalid")
        if record.block_start_step_index > record.block_end_step_index or record.block_end_step_index != record.local_step_index or record.block_length != record.block_end_step_index - record.block_start_step_index + 1:
            raise ValueError("record block bounds are inconsistent")
        if index and record.local_step_index <= records[index - 1].local_step_index:
            raise ValueError("checkpoint records must be ordered")
        for name in required_bools:
            if not isinstance(getattr(record, name), bool):
                raise ValueError(f"{name} must be bool")
        for name in optional_bools:
            value = getattr(record, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be None or bool")
        for histogram, maximum in ((record.executed_pruning_histogram_json, 4), (record.executed_precision_histogram_json, 5)):
            try:
                parsed = json.loads(histogram, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid action histogram JSON") from exc
            if not isinstance(parsed, dict) or set(parsed) != {str(level) for level in range(maximum + 1)}:
                raise ValueError("histogram key set is invalid")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in parsed.values()) or sum(parsed.values()) != record.block_length:
                raise ValueError("histogram counts must be nonnegative strict integers summing to block_length")
        for value in asdict(record).values():
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                raise ValueError("checkpoint values must be finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing and tuple(next(csv.reader(existing.splitlines()))) != FACTORIZED_QUALITY_CHECKPOINT_LOG_FIELDS:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated = path.with_name(
            f"{path.stem}.schema_mismatch_{timestamp}{path.suffix}"
        )
        suffix = 1
        while rotated.exists():
            rotated = path.with_name(
                f"{path.stem}.schema_mismatch_{timestamp}_{suffix}{path.suffix}"
            )
            suffix += 1
        path.replace(rotated)
        existing = ""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as handle:
            temporary = Path(handle.name)
            handle.write(existing)
            if existing and not existing.endswith(("\n", "\r")):
                handle.write("\n")
            writer = csv.DictWriter(handle, fieldnames=FACTORIZED_QUALITY_CHECKPOINT_LOG_FIELDS)
            if not existing:
                writer.writeheader()
            for record in records:
                row = asdict(record)
                row["global_step"] = int(global_step_before_episode) + record.local_step_index + 1
                row["learner_mode"] = learner_mode
                writer.writerow({key: "" if value is None else str(value).lower() if isinstance(value, (bool, np.bool_)) else repr(float(value)) if isinstance(value, (float, np.floating)) else value for key, value in row.items()})
        import os
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path



def _fallback_flat_ply(scene: dict[str, Any]) -> dict[str, Any]:
    ply_path = Path(scene.get("ply_path", ""))
    if ply_path.exists():
        return scene
    fallback = CODE_ROOT / "data" / f"{scene.get('name', ply_path.stem)}.ply"
    if fallback.exists():
        scene = dict(scene)
        scene["ply_path"] = str(fallback)
    return scene


@contextmanager
def _open_csv_writer_with_schema_rotation(path: Path, fieldnames: list[str]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames)
    write_header = False
    mode = "a"

    if path.exists():
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fp:
            existing_header = fp.readline().strip().lstrip("\ufeff")
        if existing_header.split(",") != fieldnames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated = path.with_name(
                f"{path.stem}.schema_mismatch_{timestamp}{path.suffix}"
            )
            suffix = 1
            while rotated.exists():
                rotated = path.with_name(
                    f"{path.stem}.schema_mismatch_{timestamp}_{suffix}{path.suffix}"
                )
                suffix += 1
            try:
                path.replace(rotated)
            except PermissionError:
                shutil.copy2(path, rotated)
            mode = "w"
            write_header = True
    else:
        mode = "w"
        write_header = True

    with path.open(mode, newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        yield writer


def _strict_positive_int(value: Any, name: str) -> int:
    """Return a strict positive integer, rejecting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _strict_nonnegative_int(value: Any, name: str) -> int:
    """Return a strict nonnegative integer, rejecting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a nonnegative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return normalized


def _strict_finite_float(value: Any, name: str) -> float:
    """Return a finite float, rejecting booleans."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _strict_nonnegative_float(value: Any, name: str) -> float:
    """Return a finite nonnegative float, rejecting booleans."""
    normalized = _strict_finite_float(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return normalized


def _factorized_exploration_stds(
    pruning_exploration_std: Any,
    precision_exploration_std: Any,
) -> tuple[float, float]:
    """Validate and return independent exploration scales for both axes."""
    return (
        _strict_nonnegative_float(
            pruning_exploration_std, "pruning_exploration_std"
        ),
        _strict_nonnegative_float(
            precision_exploration_std, "precision_exploration_std"
        ),
    )


def apply_factorized_exploration(
    action: np.ndarray,
    *,
    pruning_std: float,
    precision_std: float,
    rng: Any,
    deterministic: bool = False,
) -> np.ndarray:
    """Apply independent continuous noise to pruning and precision components."""
    if not isinstance(deterministic, (bool, np.bool_)):
        raise ValueError("deterministic must be a boolean")
    try:
        continuous = np.asarray(action, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("factorized action must be a finite shape-(2,) array") from exc
    if continuous.shape != (FACTORIZED_ACTION_DIM,):
        raise ValueError(
            f"factorized action must have shape (2,), got {continuous.shape}"
        )
    if not np.all(np.isfinite(continuous)):
        raise ValueError("factorized action must contain only finite values")
    pruning_scale, precision_scale = _factorized_exploration_stds(
        pruning_std, precision_std
    )
    noisy = continuous.copy()
    if not bool(deterministic):
        if rng is None or not callable(getattr(rng, "normal", None)):
            raise ValueError("rng must provide a normal(mean, std) method")
        pruning_noise = float(rng.normal(0.0, pruning_scale))
        precision_noise = float(rng.normal(0.0, precision_scale))
        if not np.isfinite(pruning_noise) or not np.isfinite(precision_noise):
            raise ValueError("exploration RNG returned non-finite noise")
        noisy[0] += pruning_noise
        noisy[1] += precision_noise
    noisy[0] = np.clip(noisy[0], 0.0, FACTORIZED_ACTION_BOUND[0])
    noisy[1] = np.clip(noisy[1], 0.0, FACTORIZED_ACTION_BOUND[1])
    return noisy.astype(np.float64, copy=True)


def build_factorized_actors(
    state_dim: int,
    actor_learning_rate: float,
) -> tuple[Network.FactorizedActorNetwork, Network.FactorizedActorNetwork]:
    """Build identical behavior/target V2 Actors with direct two-axis output."""
    normalized_state_dim = _strict_positive_int(state_dim, "state_dim")
    learning_rate = _strict_nonnegative_float(
        actor_learning_rate, "actor_learning_rate"
    )
    behavior_actor = Network.FactorizedActorNetwork(
        None,
        normalized_state_dim,
        FACTORIZED_ACTION_DIM,
        FACTORIZED_ACTION_BOUND,
        learning_rate,
        scope="factorized_behavior_actor",
    )
    target_actor = Network.FactorizedActorNetwork(
        None,
        normalized_state_dim,
        FACTORIZED_ACTION_DIM,
        FACTORIZED_ACTION_BOUND,
        learning_rate,
        scope="factorized_target_actor",
    )
    target_actor.copy_from(behavior_actor)
    probe = np.zeros((1, normalized_state_dim), dtype=np.float64)
    behavior_output = behavior_actor.predict_action(probe)
    target_output = target_actor.predict_action(probe)
    if behavior_output.shape != (1, FACTORIZED_ACTION_DIM):
        raise RuntimeError("Factorized behavior Actor returned an invalid shape")
    if not np.array_equal(behavior_output, target_output):
        raise RuntimeError("Factorized behavior and target Actors were not copied")
    return behavior_actor, target_actor


def _validate_factorized_observation(
    observation: Any,
    expected_state_dim: int,
    name: str,
) -> np.ndarray:
    """Validate and copy one finite V2 state vector."""
    try:
        state = np.asarray(observation, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite state vector") from exc
    if state.shape != (expected_state_dim,):
        raise ValueError(
            f"{name} must have shape ({expected_state_dim},), got {state.shape}"
        )
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{name} must contain only finite values")
    return state.copy()


def _quality_checkpoint_record(env, scene, episode_id, checkpoint_index, step_index, info, actor_actions, environment_actions, executed_pruning, executed_precision, dense_rewards, size_after, ratio_after, previous_drop, checkpoint_reward_D):
    start = _strict_nonnegative_int(info.get("quality_block_start_step_index"), "quality block start")
    end = _strict_nonnegative_int(info.get("quality_block_end_step_index"), "quality block end")
    length = _strict_positive_int(info.get("quality_block_length"), "quality block length")
    if end != step_index or length != end - start + 1 or length > int(getattr(env, "factorized_quality_interval", 8)):
        raise ValueError("invalid quality checkpoint block bounds")
    actor = np.asarray(actor_actions[start:end + 1]); continuous = np.asarray(environment_actions[start:end + 1])
    pruning = executed_pruning[start:end + 1]; precision = executed_precision[start:end + 1]
    rewards = dense_rewards[start:end + 1]; terminal = bool(info.get("is_terminal"))
    observed = info.get("quality_observed") is True
    current_drop = _strict_finite_float(info["quality_drop"], "quality_drop") if observed else None
    prior_drop = float(previous_drop if previous_drop is not None else 0.0) if observed else None
    original_size = float(getattr(env, "original_size_bytes"))
    before = original_size if start == 0 else size_after[start - 1]
    before_ratio = 1.0 if start == 0 else ratio_after[start - 1]
    after = size_after[end]; after_ratio = ratio_after[end]
    correction = _strict_finite_float(info.get("terminal_size_correction", 0.0), "terminal correction") if terminal else None
    quality_value = lambda key: _strict_finite_float(info[key], key) if observed and info.get(key) is not None else None
    feasible = info.get("quality_feasible") if observed else None
    if feasible is not None and not isinstance(feasible, (bool, np.bool_)):
        raise ValueError("quality_feasible must be bool")
    epsilon = info.get("quality_epsilon", getattr(env, "quality_epsilon", None))
    epsilon = _strict_finite_float(epsilon, "quality_epsilon") if epsilon is not None else None
    margin = epsilon - current_drop if observed and epsilon is not None else None
    dense_sum = float(sum(rewards)); block_reward = dense_sum + (correction or 0.0)
    return FactorizedQualityCheckpointRecord(
        episode_id, str(scene.get("name", "")), checkpoint_index, step_index, int(info["groups_processed"]), int(info.get("total_group_count", info["groups_processed"])), start, end, length, terminal,
        "fixed_subset", str(info.get("quality_mode", "disabled")), observed, bool(info.get("quality_target_ready")), bool(info.get("quality_reward_is_block_target")), str(info.get("fixed_view_manifest_path", "")), int(info.get("selected_view_count", 0)),
        quality_value("original_quality_score"), quality_value("previous_quality_score"), quality_value("quality_score"), prior_drop, current_drop, current_drop - prior_drop if observed else None, epsilon, margin, bool(feasible) if feasible is not None else None, quality_value("quality_violation"), quality_value("quality_block_reward"),
        before, after, before_ratio, after_ratio, before_ratio - after_ratio, dense_sum,
        int(info["compact_size_bytes"]) if terminal and info.get("compact_size_bytes") is not None else None, _strict_finite_float(info["compact_size_ratio"], "compact ratio") if terminal and info.get("compact_size_ratio") is not None else None, correction,
        float(actor[:,0].mean()), float(actor[:,1].mean()), float(continuous[:,0].mean()), float(continuous[:,1].mean()), float(continuous[:,0].min()), float(continuous[:,0].max()), float(continuous[:,1].min()), float(continuous[:,1].max()),
        float(np.mean(pruning)), float(np.mean(precision)), int(min(pruning)), int(max(pruning)), int(min(precision)), int(max(precision)), _histogram_json(pruning, 4), _histogram_json(precision, 5), block_reward, _strict_finite_float(checkpoint_reward_D, "checkpoint reward D")
    )


def run_factorized_rollout(
    env: Any,
    scene: dict[str, Any],
    actor: Network.FactorizedActorNetwork,
    replay_memory: trans.FactorizedReplayMemory,
    *,
    episode_id: int,
    pruning_exploration_std: float,
    precision_exploration_std: float,
    quality_interval: int,
    requested_view_count: int,
    noise_tolerance: float,
    rng: Any,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Collect one opacity-baseline V2 episode without learner updates.

    Opacity sorting is only the Gaussian-selection executor.  Learning uses a
    joint group-level decision over pruning and SH precision, with independent
    critics and fixed-view whole-scene quality checkpoints.
    """
    if not isinstance(replay_memory, trans.FactorizedReplayMemory):
        raise ValueError("replay_memory must be a FactorizedReplayMemory")
    normalized_episode = _strict_nonnegative_int(episode_id, "episode_id")
    normalized_interval = _strict_positive_int(quality_interval, "quality_interval")
    if normalized_interval > trans.FactorizedReplayMemory.MAX_QUALITY_BLOCK_LENGTH:
        raise ValueError(
            "quality_interval cannot exceed FactorizedReplayMemory's maximum "
            f"block length of {trans.FactorizedReplayMemory.MAX_QUALITY_BLOCK_LENGTH}"
        )
    normalized_view_count = _strict_positive_int(
        requested_view_count, "requested_view_count"
    )
    normalized_noise = _strict_nonnegative_float(noise_tolerance, "noise_tolerance")
    pruning_std, precision_std = _factorized_exploration_stds(
        pruning_exploration_std, precision_exploration_std
    )
    if replay_memory.buffer_size < normalized_interval:
        raise ValueError(
            "FactorizedReplayMemory buffer_size must be at least quality_interval"
        )
    if not isinstance(deterministic, (bool, np.bool_)):
        raise ValueError("deterministic must be a boolean")
    state_dim = _strict_positive_int(
        getattr(env, "factorized_state_dim", None), "env.factorized_state_dim"
    )
    observation = _validate_factorized_observation(
        env.reset_factorized(
            scene,
            quality_interval=normalized_interval,
            requested_view_count=normalized_view_count,
            noise_tolerance=normalized_noise,
        ),
        state_dim,
        "initial factorized observation",
    )
    _validate_factorized_v1_pruning_environment(env)

    done = False
    step_index = 0
    final_info: dict[str, Any] = {}
    final_reward_D = 0.0
    final_reward_P = 0.0
    sum_reward_P = 0.0
    checkpoint_reward_D_sum = 0.0
    checkpoint_count = 0
    observed_block_count = 0
    actor_actions: list[np.ndarray] = []
    environment_actions: list[np.ndarray] = []
    executed_pruning: list[int] = []; executed_precision: list[int] = []
    dense_rewards: list[float] = []; size_after: list[float] = []; ratio_after: list[float] = []
    checkpoint_records: list[FactorizedQualityCheckpointRecord] = []
    previous_subset_drop: float | None = None
    fixed_subset_manifest: str | None = None

    while not done:
        actor_action_batch = np.asarray(
            actor.predict_action(observation.reshape(1, -1)), dtype=np.float64
        )
        if actor_action_batch.shape != (1, FACTORIZED_ACTION_DIM):
            raise ValueError(
                "Factorized Actor predict_action must return shape (1, 2), got "
                f"{actor_action_batch.shape}"
            )
        if not np.all(np.isfinite(actor_action_batch)):
            raise ValueError("Factorized Actor output must be finite")
        actor_action = actor_action_batch[0].copy()
        environment_action = apply_factorized_exploration(
            actor_action,
            pruning_std=pruning_std,
            precision_std=precision_std,
            rng=rng,
            deterministic=bool(deterministic),
        )

        next_observation_raw, reward_D, reward_P, done_raw, info = (
            env.step_factorized(environment_action)
        )
        next_observation = _validate_factorized_observation(
            next_observation_raw,
            state_dim,
            "next factorized observation",
        )
        if not isinstance(info, dict):
            raise ValueError("step_factorized info must be a dict")
        _validate_factorized_v1_pruning_info(info)
        done = bool(done_raw)
        reward_D_value = _strict_finite_float(reward_D, "reward_D")
        reward_P_value = _strict_finite_float(reward_P, "reward_P")
        left_bitbudget = _strict_finite_float(
            info.get("left_bitbudget"), "info.left_bitbudget"
        )
        groups_processed = _strict_positive_int(
            info.get("groups_processed"), "info.groups_processed"
        )
        if groups_processed != step_index + 1:
            raise ValueError(
                "info.groups_processed must equal the one-based rollout step count"
            )

        # Replay is written only after the environment step succeeds. Quality
        # metadata remains unset until the complete block is finalized below.
        replay_memory.append_transition(
            state=observation,
            action=environment_action,
            reward_D=reward_D_value,
            reward_P=reward_P_value,
            next_state=next_observation,
            done=done,
            left_bitbudget=left_bitbudget,
            episode_id=normalized_episode,
            step_index=step_index,
            groups_processed=groups_processed,
        )
        actor_actions.append(actor_action); environment_actions.append(environment_action.copy())
        executed_pruning.append(_strict_nonnegative_int(info.get("executed_pruning_level"), "executed pruning"))
        executed_precision.append(_strict_nonnegative_int(info.get("executed_precision_level"), "executed precision"))
        dense_rewards.append(_strict_finite_float(info.get("dense_size_reward"), "dense size reward"))
        size_after.append(_strict_finite_float(info.get("estimated_size_after"), "estimated size after"))
        ratio_after.append(size_after[-1] / float(getattr(env, "original_size_bytes")))

        checkpoint_due = info.get("quality_checkpoint_due") is True
        quality_observed = info.get("quality_observed") is True
        if quality_observed and not checkpoint_due:
            raise ValueError("quality_observed=True requires quality_checkpoint_due=True")
        if checkpoint_due:
            checkpoint_count += 1
            checkpoint_reward_D_sum += reward_D_value
        if checkpoint_due and quality_observed:
            if info.get("quality_target_ready") is not True:
                raise ValueError("observed quality checkpoint must have target_ready=True")
            if info.get("quality_reward_is_block_target") is not True:
                raise ValueError(
                    "observed quality checkpoint reward must be marked as a block target"
                )
            for required_key in (
                "quality_block_reward",
                "quality_score",
                "quality_drop",
            ):
                if info.get(required_key) is None:
                    raise ValueError(
                        f"observed quality checkpoint is missing {required_key}"
                    )
            start_index = _strict_nonnegative_int(
                info.get("quality_block_start_step_index"),
                "quality_block_start_step_index",
            )
            end_index = _strict_nonnegative_int(
                info.get("quality_block_end_step_index"),
                "quality_block_end_step_index",
            )
            block_length = _strict_positive_int(
                info.get("quality_block_length"), "quality_block_length"
            )
            if block_length != end_index - start_index + 1:
                raise ValueError("quality_block_length does not match start/end bounds")
            if block_length > normalized_interval:
                raise ValueError("quality block exceeds quality_interval")
            if end_index != step_index:
                raise ValueError("quality block end must equal the current step_index")
            replay_memory.finalize_quality_block(
                episode_id=normalized_episode,
                start_step_index=start_index,
                end_step_index=end_index,
                quality_block_reward=_strict_finite_float(
                    info["quality_block_reward"], "quality_block_reward"
                ),
                quality_score=_strict_finite_float(
                    info["quality_score"], "quality_score"
                ),
                quality_drop=_strict_finite_float(
                    info["quality_drop"], "quality_drop"
                ),
                bootstrap_state=next_observation,
                bootstrap_done=done,
            )
            observed_block_count += 1

        if checkpoint_due:
            record = _quality_checkpoint_record(env, scene, normalized_episode, len(checkpoint_records), step_index, info, actor_actions, environment_actions, executed_pruning, executed_precision, dense_rewards, size_after, ratio_after, previous_subset_drop, reward_D_value)
            manifest = record.fixed_view_manifest_path
            if manifest:
                if fixed_subset_manifest is None:
                    fixed_subset_manifest = manifest
                elif manifest != fixed_subset_manifest:
                    raise RuntimeError("fixed-view manifest changed within one episode")
            elif fixed_subset_manifest is not None:
                raise RuntimeError("fixed-view manifest became empty within one episode")
            checkpoint_records.append(record)
            if record.quality_observed:
                previous_subset_drop = record.current_subset_quality_drop

        sum_reward_P += reward_P_value
        step_index += 1
        observation = next_observation
        final_info = info
        final_reward_D = reward_D_value
        final_reward_P = reward_P_value

    final_quality_drop = final_info.get("quality_drop")
    final_quality_feasible = final_info.get("quality_feasible")
    if final_quality_drop is None or not isinstance(
        final_quality_feasible, (bool, np.bool_)
    ):
        raise RuntimeError(
            "formal Episode ended without an observed fixed-subset quality constraint"
        )
    final_quality_drop = _strict_finite_float(
        final_quality_drop, "final fixed-subset quality drop"
    )
    final_quality_feasible = bool(final_quality_feasible)
    actor_array = np.asarray(actor_actions, dtype=np.float64)
    environment_array = np.asarray(environment_actions, dtype=np.float64)
    return {
        "final_info": final_info,
        "steps": step_index,
        "episode_id": normalized_episode,
        "deterministic": bool(deterministic),
        "sum_reward_P": float(sum_reward_P),
        "checkpoint_reward_D_sum": float(checkpoint_reward_D_sum),
        "final_reward_P": float(final_reward_P),
        "final_reward_D": float(final_reward_D),
        "final_left_bitbudget": float(final_info.get("left_bitbudget", 0.0)),
        "mean_actor_pruning": float(np.mean(actor_array[:, 0])),
        "mean_actor_precision": float(np.mean(actor_array[:, 1])),
        "mean_environment_pruning": float(np.mean(environment_array[:, 0])),
        "mean_environment_precision": float(np.mean(environment_array[:, 1])),
        "quality_checkpoint_count": checkpoint_count,
        "quality_observed_block_count": observed_block_count,
        "replay_size_after": len(replay_memory.replay_memory),
        "final_quality_drop": final_quality_drop,
        "final_quality_feasible": final_quality_feasible,
        "final_compact_size_bytes": int(
            final_info.get("compact_size_bytes", 0) or 0
        ),
        "final_compact_size_ratio": final_info.get("compact_size_ratio"),
        "quality_checkpoint_records": tuple(checkpoint_records),
        "quality_checkpoint_record_count": len(checkpoint_records),
        "actor_actions": tuple(action.copy() for action in actor_actions),
        "environment_actions": tuple(
            action.copy() for action in environment_actions
        ),
        "executed_pruning_levels": tuple(executed_pruning),
        "executed_precision_levels": tuple(executed_precision),
        "mean_executed_pruning": float(np.mean(executed_pruning)),
        "mean_executed_precision": float(np.mean(executed_precision)),
        **_factorized_v1_pruning_provenance(),
    }


def normalize_factorized_action_batch(actions: Any) -> np.ndarray:
    """Normalize a finite continuous ``[pruning, precision]`` action batch."""
    try:
        action_array = np.asarray(actions, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("factorized actions must be a finite [B, 2] array") from exc
    if action_array.ndim != 2 or action_array.shape[1] != FACTORIZED_ACTION_DIM:
        raise ValueError(
            f"factorized actions must have shape [B, 2], got {action_array.shape}"
        )
    if not np.all(np.isfinite(action_array)):
        raise ValueError("factorized actions must contain only finite values")
    tolerance = 1e-9
    if np.any(action_array < -tolerance) or np.any(
        action_array > FACTORIZED_ACTION_BOUND + tolerance
    ):
        raise ValueError("factorized actions are outside [0, 4] x [0, 5]")
    clipped = np.clip(action_array, 0.0, FACTORIZED_ACTION_BOUND)
    return (clipped / FACTORIZED_ACTION_BOUND).astype(np.float64, copy=False)


def build_factorized_critics(
    state_dim: int,
    critic_learning_rate: float,
) -> tuple[Network.CriticNetwork, Network.CriticNetwork]:
    """Build identical behavior/target dual Critics for 19D observations."""
    normalized_state_dim = _strict_positive_int(state_dim, "state_dim")
    if normalized_state_dim != 19:
        raise ValueError("formal first-version Critics require state_dim=19")
    learning_rate = _strict_nonnegative_float(
        critic_learning_rate, "critic_learning_rate"
    )
    behavior_critic = Network.CriticNetwork(
        None,
        normalized_state_dim,
        FACTORIZED_ACTION_DIM,
        learning_rate,
        scope_D="factorized_behavior_quality_D",
        scope_P="factorized_behavior_size_P",
    )
    target_critic = Network.CriticNetwork(
        None,
        normalized_state_dim,
        FACTORIZED_ACTION_DIM,
        learning_rate,
        scope_D="factorized_target_quality_D",
        scope_P="factorized_target_size_P",
    )
    for name, critic in (
        ("behavior_critic", behavior_critic),
        ("target_critic", target_critic),
    ):
        if critic.s_dim != normalized_state_dim or critic.a_dim != 2:
            raise RuntimeError(
                f"{name} dimensions must be state={normalized_state_dim}, action=2"
            )
    target_critic.copy_from(behavior_critic)
    probe_states = np.zeros((2, normalized_state_dim), dtype=np.float64)
    probe_actions = np.zeros((2, FACTORIZED_ACTION_DIM), dtype=np.float64)
    behavior_D, behavior_P = behavior_critic.predict(probe_states, probe_actions)
    target_D, target_P = target_critic.predict(probe_states, probe_actions)
    if not np.array_equal(behavior_D, target_D) or not np.array_equal(
        behavior_P, target_P
    ):
        raise RuntimeError("factorized behavior and target Critics were not copied")
    return behavior_critic, target_critic


def _factorized_column(value: Any, name: str) -> np.ndarray:
    """Validate a finite B-by-1 numeric array."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape [B, 1]") from exc
    if array.ndim != 2 or array.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B, 1], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _factorized_gamma(value: Any, name: str) -> float:
    """Validate a finite discount factor in the closed interval [0, 1]."""
    gamma = _strict_finite_float(value, name)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return gamma


def compute_factorized_size_td_targets(
    rewards_P: Any,
    dones: Any,
    target_q_P: Any,
    gamma_size: float,
) -> np.ndarray:
    """Compute unclipped one-step Size Critic TD targets."""
    rewards = _factorized_column(rewards_P, "rewards_P")
    done_values = _factorized_column(dones, "dones")
    target_values = _factorized_column(target_q_P, "target_q_P")
    if rewards.shape != done_values.shape or rewards.shape != target_values.shape:
        raise ValueError("Size TD target inputs must have identical [B, 1] shapes")
    if np.any((done_values != 0.0) & (done_values != 1.0)):
        raise ValueError("dones must contain only 0/1 values")
    gamma = _factorized_gamma(gamma_size, "gamma_size")
    return (
        rewards + gamma * (1.0 - done_values) * target_values
    ).astype(np.float64)


def compute_factorized_quality_td_targets(
    quality_block_rewards: Any,
    quality_horizons: Any,
    bootstrap_dones: Any,
    target_q_D: Any,
    gamma_quality: float,
) -> np.ndarray:
    """Compute unclipped 1-to-8-step Quality Critic TD targets."""
    rewards = _factorized_column(
        quality_block_rewards, "quality_block_rewards"
    )
    raw_horizons = np.asarray(quality_horizons)
    if raw_horizons.ndim != 2 or raw_horizons.shape[1] != 1:
        raise ValueError(
            f"quality_horizons must have shape [B, 1], got {raw_horizons.shape}"
        )
    if raw_horizons.dtype.kind not in {"i", "u"}:
        raise ValueError("quality_horizons must contain strict integers")
    horizons = raw_horizons.astype(np.int64, copy=False)
    if np.any((horizons < 1) | (horizons > 8)):
        raise ValueError("quality_horizons must be between 1 and 8")
    done_values = _factorized_column(bootstrap_dones, "bootstrap_dones")
    target_values = _factorized_column(target_q_D, "target_q_D")
    if rewards.shape != horizons.shape or rewards.shape != done_values.shape or (
        rewards.shape != target_values.shape
    ):
        raise ValueError("Quality TD target inputs must have identical [B, 1] shapes")
    if np.any((done_values != 0.0) & (done_values != 1.0)):
        raise ValueError("bootstrap_dones must contain only 0/1 values")
    gamma = _factorized_gamma(gamma_quality, "gamma_quality")
    discounts = np.power(gamma, horizons.astype(np.float64))
    return (
        rewards + discounts * (1.0 - done_values) * target_values
    ).astype(np.float64)


def update_factorized_critics_from_replay(
    behavior_critic: Network.CriticNetwork,
    target_critic: Network.CriticNetwork,
    target_actor: Network.FactorizedActorNetwork,
    replay_memory: trans.FactorizedReplayMemory,
    *,
    batch_size: int,
    update_steps: int,
    gamma_size: float,
    gamma_quality: float,
) -> dict[str, Any]:
    """Independently update Size P and block-quality D Critics."""
    if not isinstance(replay_memory, trans.FactorizedReplayMemory):
        raise ValueError("replay_memory must be a FactorizedReplayMemory")
    normalized_batch_size = _strict_positive_int(batch_size, "batch_size")
    normalized_update_steps = _strict_positive_int(update_steps, "update_steps")
    normalized_gamma_size = _factorized_gamma(gamma_size, "gamma_size")
    normalized_gamma_quality = _factorized_gamma(
        gamma_quality, "gamma_quality"
    )
    general_count = len(replay_memory.replay_memory)
    quality_ready_count = sum(
        transition.quality_target_ready
        for transition in replay_memory.replay_memory
    )
    size_losses: list[float] = []
    quality_losses: list[float] = []
    size_targets: list[np.ndarray] = []
    quality_targets: list[np.ndarray] = []

    if general_count >= normalized_batch_size:
        for _ in range(normalized_update_steps):
            batch = replay_memory.sample_batch_arrays(normalized_batch_size)
            target_actions = normalize_factorized_action_batch(
                target_actor.predict_action(batch["next_states"])
            )
            _, target_q_P = target_critic.predict(batch["next_states"], target_actions)
            targets_P = compute_factorized_size_td_targets(
                batch["rewards_P"],
                batch["dones"],
                target_q_P,
                normalized_gamma_size,
            )
            normalized_actions = normalize_factorized_action_batch(batch["actions"])
            loss = behavior_critic.train_P(
                batch["states"], normalized_actions, targets_P
            )
            if not np.isfinite(loss):
                raise RuntimeError("Size Critic produced a non-finite loss")
            size_losses.append(float(loss))
            size_targets.append(targets_P)
            target_critic.soft_update_P_from(behavior_critic)

    if quality_ready_count >= normalized_batch_size:
        for _ in range(normalized_update_steps):
            batch = replay_memory.sample_quality_batch_arrays(
                normalized_batch_size
            )
            target_actions = normalize_factorized_action_batch(
                target_actor.predict_action(batch["quality_bootstrap_states"])
            )
            target_q_D, _ = target_critic.predict(
                batch["quality_bootstrap_states"], target_actions
            )
            targets_D = compute_factorized_quality_td_targets(
                batch["quality_block_rewards"],
                batch["quality_horizons"],
                batch["quality_bootstrap_dones"],
                target_q_D,
                normalized_gamma_quality,
            )
            normalized_actions = normalize_factorized_action_batch(batch["actions"])
            loss = behavior_critic.train_D(
                batch["states"], normalized_actions, targets_D
            )
            if not np.isfinite(loss):
                raise RuntimeError("Quality Critic produced a non-finite loss")
            quality_losses.append(float(loss))
            quality_targets.append(targets_D)
            target_critic.soft_update_D_from(behavior_critic)

    return {
        "size_critic_loss": float(np.mean(size_losses)) if size_losses else 0.0,
        "quality_critic_loss": (
            float(np.mean(quality_losses)) if quality_losses else 0.0
        ),
        "size_critic_update_count": len(size_losses),
        "quality_critic_update_count": len(quality_losses),
        "mean_size_td_target": (
            float(np.mean(np.concatenate(size_targets))) if size_targets else 0.0
        ),
        "mean_quality_td_target": (
            float(np.mean(np.concatenate(quality_targets)))
            if quality_targets
            else 0.0
        ),
        "quality_ready_transition_count": int(quality_ready_count),
        "general_transition_count": int(general_count),
    }


def select_factorized_actor_critic(final_quality_feasible: Any) -> str:
    """Hard-switch Actor learning solely from final quality feasibility."""
    if not isinstance(final_quality_feasible, (bool, np.bool_)):
        raise ValueError("final_quality_feasible must be a boolean")
    return "P" if bool(final_quality_feasible) else "D"


def _raw_factorized_action_gradients(
    normalized_action_gradients: Any,
) -> np.ndarray:
    """Convert dQ/d(normalized action) to dQ/d(raw two-axis action)."""
    gradients = np.asarray(normalized_action_gradients, dtype=np.float64)
    if gradients.ndim != 2 or gradients.shape[1] != FACTORIZED_ACTION_DIM:
        raise ValueError(
            "normalized action gradients must have shape [B, 2], got "
            f"{gradients.shape}"
        )
    if not np.all(np.isfinite(gradients)):
        raise ValueError("normalized action gradients must be finite")
    return gradients / FACTORIZED_ACTION_BOUND


def update_factorized_actor_from_episode(
    behavior_actor: Network.FactorizedActorNetwork,
    target_actor: Network.FactorizedActorNetwork,
    behavior_critic: Network.CriticNetwork,
    episode_transitions: list[trans.FactorizedTransition],
    *,
    selected_source: str,
    max_gradient_norm: float,
) -> dict[str, Any]:
    """Update the V2 Actor from exactly one selected Critic branch."""
    if selected_source not in {"D", "P"}:
        raise ValueError("selected_source must be 'D' or 'P'")
    gradient_limit = _strict_finite_float(max_gradient_norm, "max_gradient_norm")
    if gradient_limit <= 0.0:
        raise ValueError("max_gradient_norm must be greater than zero")
    if not episode_transitions:
        raise ValueError("episode_transitions must not be empty")
    if not all(
        isinstance(transition, trans.FactorizedTransition)
        for transition in episode_transitions
    ):
        raise ValueError("episode_transitions must contain FactorizedTransition objects")
    states = np.stack([transition.state for transition in episode_transitions]).astype(
        np.float64
    )
    raw_actions = np.asarray(behavior_actor.predict_action(states), dtype=np.float64)
    if raw_actions.shape != (len(states), FACTORIZED_ACTION_DIM):
        raise ValueError("Factorized Actor returned an invalid action batch")
    normalized_actions = normalize_factorized_action_batch(raw_actions)
    normalized_gradients = np.asarray(
        (
            behavior_critic.action_gradient_D(states, normalized_actions)
            if selected_source == "D"
            else behavior_critic.action_gradient_P(states, normalized_actions)
        ),
        dtype=np.float64,
    )
    raw_gradients = _raw_factorized_action_gradients(normalized_gradients)
    norms_before = np.linalg.norm(raw_gradients, axis=1)
    clipped_gradients = raw_gradients.copy()
    for index, norm in enumerate(norms_before):
        if not np.isfinite(norm):
            raise ValueError("Actor action gradient norm must be finite")
        if norm > gradient_limit:
            clipped_gradients[index] *= gradient_limit / norm
    norms_after = np.linalg.norm(clipped_gradients, axis=1)
    if not np.all(np.isfinite(clipped_gradients)):
        raise ValueError("clipped Actor gradients must be finite")
    behavior_actor.train(states, clipped_gradients)
    target_actor.soft_update_from(behavior_actor)

    gradient_rows: list[dict[str, Any]] = []
    for index in range(len(states)):
        gradient_rows.append(
            {
                "sample_idx": index,
                "selected_source": selected_source,
                "state_quality_drop": float(states[index, 12]),
                "state_quality_margin": float(states[index, 13]),
                "raw_pruning_action": float(raw_actions[index, 0]),
                "raw_precision_action": float(raw_actions[index, 1]),
                "normalized_pruning_action": float(normalized_actions[index, 0]),
                "normalized_precision_action": float(normalized_actions[index, 1]),
                "normalized_pruning_gradient": float(normalized_gradients[index, 0]),
                "normalized_precision_gradient": float(normalized_gradients[index, 1]),
                "raw_pruning_gradient": float(clipped_gradients[index, 0]),
                "raw_precision_gradient": float(clipped_gradients[index, 1]),
                "gradient_norm_before_clip": float(norms_before[index]),
                "gradient_norm_after_clip": float(norms_after[index]),
            }
        )
    return {
        "actor_update_source": selected_source,
        "actor_update_count": 1,
        "mean_actor_gradient_norm": float(np.mean(norms_after)),
        "max_actor_gradient_norm": float(np.max(norms_after)),
        "gradient_rows": gradient_rows,
    }




def save_factorized_training_checkpoint(
    checkpoints_dir: str | Path,
    behavior_actor: Network.FactorizedActorNetwork,
    target_actor: Network.FactorizedActorNetwork,
    behavior_critic: Network.CriticNetwork,
    target_critic: Network.CriticNetwork,
    replay_memory: trans.FactorizedReplayMemory,
    episode_id: int,
    global_step: int,
    config: Any,
    training_metadata: dict[str, Any],
) -> Path:
    """Save the complete V2 Actor, dual-Critic, target, and Replay state."""
    if not isinstance(behavior_actor, Network.FactorizedActorNetwork) or not isinstance(
        target_actor, Network.FactorizedActorNetwork
    ):
        raise ValueError("factorized training checkpoint requires two V2 Actors")
    if not isinstance(behavior_critic, Network.CriticNetwork) or not isinstance(
        target_critic, Network.CriticNetwork
    ):
        raise ValueError("factorized training checkpoint requires two dual Critics")
    if not isinstance(replay_memory, trans.FactorizedReplayMemory):
        raise ValueError("factorized training checkpoint requires FactorizedReplayMemory")
    if not isinstance(training_metadata, dict):
        raise ValueError("training_metadata must be a dict")
    if (
        behavior_actor.s_dim != 19
        or target_actor.s_dim != 19
        or behavior_critic.s_dim != 19
        or target_critic.s_dim != 19
        or behavior_actor.a_dim != 2
        or target_actor.a_dim != 2
        or behavior_critic.a_dim != 2
        or target_critic.a_dim != 2
    ):
        raise ValueError("factorized training checkpoint requires state_dim=19/action_dim=2")
    checkpoint = {
        "schema": FACTORIZED_TRAINING_CHECKPOINT_SCHEMA,
        "pruning_mode": FACTORIZED_V1_PRUNING_MODE,
        "pruning_policy": FACTORIZED_V1_PRUNING_POLICY,
        "pruning_importance_source": FACTORIZED_V1_IMPORTANCE_SOURCE,
        "pruning_importance_version": FACTORIZED_V1_IMPORTANCE_VERSION,
        "episode": _strict_nonnegative_int(episode_id, "episode_id"),
        "global_step": _strict_nonnegative_int(global_step, "global_step"),
        "behavior_actor": behavior_actor.state_dict(),
        "target_actor": target_actor.state_dict(),
        "behavior_critic": behavior_critic.state_dict(),
        "target_critic": target_critic.state_dict(),
        "replay_memory": replay_memory.state_dict(),
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "config": config,
        "training_metadata": {
            **dict(training_metadata),
            **_factorized_v1_pruning_provenance(),
        },
        "state_dim": 19,
        "action_dim": FACTORIZED_ACTION_DIM,
        "action_bounds": FACTORIZED_ACTION_BOUND.tolist(),
    }
    destination = Path(checkpoints_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / (
        f"factorized_dual_critic_ep{checkpoint['episode']:04d}.pkl"
    )
    with path.open("wb") as file_handle:
        pickle.dump(checkpoint, file_handle)
    return path


def load_factorized_training_checkpoint(
    checkpoint_path: str | Path,
    behavior_actor: Network.FactorizedActorNetwork,
    target_actor: Network.FactorizedActorNetwork,
    behavior_critic: Network.CriticNetwork,
    target_critic: Network.CriticNetwork,
    replay_memory: trans.FactorizedReplayMemory,
) -> dict[str, Any]:
    """Transactionally restore one complete factorized dual-Critic checkpoint.

    Every serialized field is validated with temporary networks and Replay
    storage before any destination object or process RNG is mutated.
    """
    try:
        with Path(checkpoint_path).open("rb") as file_handle:
            checkpoint = pickle.load(file_handle)
    except (OSError, EOFError, pickle.PickleError) as exc:
        raise ValueError(f"Unable to load factorized training checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("factorized training checkpoint must contain a dict")
    if checkpoint.get("schema") != FACTORIZED_TRAINING_CHECKPOINT_SCHEMA:
        raise ValueError(
            "incompatible training checkpoint schema; expected "
            f"{FACTORIZED_TRAINING_CHECKPOINT_SCHEMA!r}, got "
            f"{checkpoint.get('schema')!r}"
        )
    _validate_factorized_v1_checkpoint_metadata(
        checkpoint, "factorized training checkpoint"
    )
    if checkpoint.get("state_dim") != 19 or checkpoint.get("action_dim") != 2:
        raise ValueError("factorized training checkpoint dimensions must be 19 and 2")
    try:
        saved_bounds = np.asarray(checkpoint["action_bounds"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("training checkpoint is missing valid action_bounds") from exc
    if saved_bounds.shape != (2,) or not np.array_equal(
        saved_bounds, FACTORIZED_ACTION_BOUND
    ):
        raise ValueError("training checkpoint action_bounds must be [4.0, 5.0]")
    for actor_name in ("behavior_actor", "target_actor"):
        actor_state = checkpoint.get(actor_name)
        if not isinstance(actor_state, dict):
            raise ValueError(f"training checkpoint is missing {actor_name}")
        if actor_state.get("network_type") != "factorized_actor_v1":
            raise ValueError(f"{actor_name} is not factorized_actor_v1")
    for critic_name in ("behavior_critic", "target_critic"):
        critic_state = checkpoint.get(critic_name)
        if not isinstance(critic_state, dict) or not {
            "critic_D",
            "critic_P",
        }.issubset(critic_state):
            raise ValueError(f"training checkpoint is missing dual state {critic_name}")
    if (
        not isinstance(behavior_actor, Network.FactorizedActorNetwork)
        or not isinstance(target_actor, Network.FactorizedActorNetwork)
        or not isinstance(behavior_critic, Network.CriticNetwork)
        or not isinstance(target_critic, Network.CriticNetwork)
        or not isinstance(replay_memory, trans.FactorizedReplayMemory)
    ):
        raise ValueError("training checkpoint destinations have incompatible types")
    if (
        behavior_actor.s_dim != 19
        or target_actor.s_dim != 19
        or behavior_critic.s_dim != 19
        or target_critic.s_dim != 19
        or behavior_actor.a_dim != 2
        or target_actor.a_dim != 2
        or behavior_critic.a_dim != 2
        or target_critic.a_dim != 2
    ):
        raise ValueError("training checkpoint destinations must use dimensions 19 and 2")

    loaded_replay = checkpoint.get("replay_memory")
    if not isinstance(loaded_replay, dict):
        raise ValueError("training checkpoint replay_memory must be a formal state")
    validation_memory = trans.FactorizedReplayMemory(
        "factorized_training_checkpoint_validation",
        replay_memory.buffer_size,
    )
    try:
        validation_memory.load_state_dict(loaded_replay)
    except ValueError as exc:
        raise ValueError(f"invalid training checkpoint Replay: {exc}") from exc
    if "random_state" not in checkpoint or "numpy_random_state" not in checkpoint:
        raise ValueError("training checkpoint is missing random state")
    _strict_nonnegative_int(checkpoint.get("episode"), "checkpoint episode")
    _strict_nonnegative_int(checkpoint.get("global_step"), "checkpoint global_step")
    if "config" not in checkpoint:
        raise ValueError("training checkpoint is missing config")
    if not isinstance(checkpoint.get("training_metadata"), dict):
        raise ValueError("training checkpoint training_metadata must be a dict")

    actor_lr = float(getattr(behavior_actor, "learning_rate", DEFAULT_ACTOR_LEARNING_RATE))
    critic_lr = float(getattr(behavior_critic, "learning_rate", DEFAULT_CRITIC_LEARNING_RATE))
    try:
        validation_behavior_actor = Network.FactorizedActorNetwork(
            None, 19, 2, FACTORIZED_ACTION_BOUND, actor_lr, "validation_behavior_actor"
        )
        validation_target_actor = Network.FactorizedActorNetwork(
            None, 19, 2, FACTORIZED_ACTION_BOUND, actor_lr, "validation_target_actor"
        )
        validation_behavior_critic = Network.CriticNetwork(
            None, 19, 2, critic_lr, "validation_behavior_critic_D", "validation_behavior_critic_P"
        )
        validation_target_critic = Network.CriticNetwork(
            None, 19, 2, critic_lr, "validation_target_critic_D", "validation_target_critic_P"
        )
        validation_behavior_actor.load_state_dict(checkpoint["behavior_actor"])
        validation_target_actor.load_state_dict(checkpoint["target_actor"])
        validation_behavior_critic.load_state_dict(checkpoint["behavior_critic"])
        validation_target_critic.load_state_dict(checkpoint["target_critic"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid Actor or Critic state in checkpoint: {exc}") from exc
    try:
        random_probe = random.Random()
        random_probe.setstate(checkpoint["random_state"])
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(checkpoint["numpy_random_state"])
    except (TypeError, ValueError) as exc:
        raise ValueError("training checkpoint contains invalid random state") from exc

    actor_snapshots = (
        behavior_actor.state_dict(),
        target_actor.state_dict(),
    )
    critic_snapshots = (
        behavior_critic.state_dict(),
        target_critic.state_dict(),
    )
    replay_snapshot = replay_memory.state_dict()
    random_snapshot = random.getstate()
    numpy_snapshot = np.random.get_state()
    try:
        behavior_actor.load_state_dict(checkpoint["behavior_actor"])
        target_actor.load_state_dict(checkpoint["target_actor"])
        behavior_critic.load_state_dict(checkpoint["behavior_critic"])
        target_critic.load_state_dict(checkpoint["target_critic"])
        replay_memory.load_state_dict(loaded_replay)
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
    except Exception as exc:
        behavior_actor.load_state_dict(actor_snapshots[0])
        target_actor.load_state_dict(actor_snapshots[1])
        behavior_critic.load_state_dict(critic_snapshots[0])
        target_critic.load_state_dict(critic_snapshots[1])
        replay_memory.load_state_dict(replay_snapshot)
        random.setstate(random_snapshot)
        np.random.set_state(numpy_snapshot)
        raise ValueError(f"failed to commit validated training checkpoint: {exc}") from exc
    return checkpoint




def train(args: argparse.Namespace) -> None:
    """Run only the formal first-version factorized dual-Critic trainer."""
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)
    training_cfg = config.get("training", {})
    env_cfg = config.get("env", {})
    paths_cfg = config.get("paths", {})
    render_cfg = config.get("render", {})
    quality_cfg = config.get("quality", {})
    crossscore_cfg = config.get("crossscore", {})
    project_cfg = config.get("project", {})

    checkpoints_dir = resolve_output_path(
        project_cfg.get("checkpoint_dir", "./checkpoints"), config
    )
    logs_dir = resolve_output_path(project_cfg.get("log_dir", "./logs"), config)
    outputs_dir = resolve_output_path(project_cfg.get("output_dir", "./outputs"), config)
    for directory in (checkpoints_dir, logs_dir, outputs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    scenes = select_scenes(config, args.scene)
    if args.ply:
        scenes = [{
            "name": Path(args.ply).stem,
            "source_path": args.scene_path,
            "model_path": args.model_path,
            "ply_path": str(resolve_input_path(args.ply, config)),
        }]
    scenes = [_fallback_flat_ply(scene) for scene in scenes]
    scenes = [scene for scene in scenes if Path(scene["ply_path"]).exists()]
    if not scenes:
        raise FileNotFoundError(
            "No valid PLY path was found. Update the configuration or pass --ply."
        )

    use_dummy_reward = bool(env_cfg.get("use_dummy_reward", True))
    use_render = bool(env_cfg.get("use_render", False))
    use_crossscore = bool(env_cfg.get("use_crossscore", False))
    if args.use_dummy_reward:
        use_dummy_reward, use_render, use_crossscore = True, False, False
    elif args.use_crossscore:
        use_dummy_reward, use_render, use_crossscore = False, True, True
    elif args.use_render:
        use_dummy_reward, use_render, use_crossscore = False, True, False
    if not use_dummy_reward and not (use_render and use_crossscore):
        raise ValueError(
            "Formal training requires debug dummy quality or render plus CrossScore."
        )

    gaussian_splatting_dir = normalize_gaussian_splatting_dir(
        resolve_input_path(
            render_cfg.get(
                "gaussian_splatting_dir",
                paths_cfg.get("gaussian_splatting_dir", "./gaussian-splatting-main"),
            ),
            config,
        )
    )
    crossscore_dir = normalize_crossscore_dir(
        resolve_input_path(paths_cfg.get("crossscore_dir", "./CrossScore-main"), config)
    )
    target_num_groups = (
        args.target_groups
        if args.target_groups is not None
        else env_cfg.get("target_num_groups", 128)
    )
    if target_num_groups is not None:
        target_num_groups = _strict_positive_int(target_num_groups, "target_num_groups")
    min_group_size = _strict_positive_int(
        args.min_group_size
        if args.min_group_size is not None
        else env_cfg.get("min_group_size", 10),
        "min_group_size",
    )
    octree_depth = _strict_positive_int(
        args.octree_max_depth
        if args.octree_max_depth is not None
        else env_cfg.get("factorized_octree_max_depth", 10),
        "factorized_octree_max_depth",
    )
    env = GS_Environment(
        scenes=scenes,
        output_root=outputs_dir,
        target_size_ratio=(
            args.target_size_ratio
            if args.target_size_ratio is not None
            else float(env_cfg.get("target_size_ratio", 0.3))
        ),
        target_num_groups=target_num_groups,
        min_group_size=min_group_size,
        factorized_octree_max_depth=octree_depth,
        opacity_low_threshold=(
            args.opacity_low_threshold
            if args.opacity_low_threshold is not None
            else float(env_cfg.get("opacity_low_threshold", 0.01))
        ),
        use_dummy_reward=use_dummy_reward,
        use_render=use_render,
        use_crossscore=use_crossscore,
        gaussian_splatting_dir=gaussian_splatting_dir,
        crossscore_dir=crossscore_dir,
        source_path=args.scene_path or None,
        original_model_path=args.model_path or None,
        iteration=int(render_cfg.get("iteration", env_cfg.get("iteration", 30000))),
        resolution=int(render_cfg.get("resolution", env_cfg.get("resolution", 4))),
        quality_cache_dir=outputs_dir / "crossscore_cache",
        cache_original_score=bool(quality_cfg.get("cache_original_score", True)),
        score_higher_is_better=bool(quality_cfg.get("score_higher_is_better", True)),
        crossscore_mode=str(quality_cfg.get("crossscore_mode", "placeholder")),
        crossscore_command_template=str(crossscore_cfg.get("command_template", "") or ""),
        crossscore_score_output=str(crossscore_cfg.get("score_output", "") or ""),
        crossscore_score_parse_mode=str(crossscore_cfg.get("score_parse_mode", "auto") or "auto"),
        crossscore_preferred_score_key=str(crossscore_cfg.get("preferred_score_key", "pred_ssim_0_1") or "pred_ssim_0_1"),
        render_python_executable=str(render_cfg.get("python_executable", "python") or "python"),
        crossscore_python_executable=str(crossscore_cfg.get("python_executable", "python") or "python"),
        crossscore_ckpt=crossscore_cfg.get("ckpt") or None,
        crossscore_config=crossscore_cfg.get("config") or None,
        crossscore_allow_image_fallback=bool(crossscore_cfg.get("allow_image_fallback", False)),
        quality_epsilon=float(quality_cfg.get("epsilon", 0.0)),
        quality_dense_alpha=float(quality_cfg.get("dense_alpha", 1.0)),
        quality_violation_alpha=float(quality_cfg.get("violation_alpha", 2.0)),
        allow_crossscore_placeholder=bool(args.allow_crossscore_placeholder),
        force_recompute_original_score=bool(args.force_recompute_original_score),
    )

    quality_interval = _strict_positive_int(
        args.quality_interval
        if args.quality_interval is not None
        else quality_cfg.get("interval", 8),
        "quality_interval",
    )
    requested_view_count = _strict_positive_int(
        args.requested_view_count
        if args.requested_view_count is not None
        else quality_cfg.get("requested_view_count", 8),
        "requested_view_count",
    )
    noise_tolerance = _strict_nonnegative_float(
        args.quality_noise_tolerance
        if args.quality_noise_tolerance is not None
        else quality_cfg.get("noise_tolerance", 0.0),
        "quality_noise_tolerance",
    )
    pruning_std, precision_std = _factorized_exploration_stds(
        args.pruning_exploration_std
        if args.pruning_exploration_std is not None
        else training_cfg.get("factorized_pruning_exploration_std", 0.5),
        args.precision_exploration_std
        if args.precision_exploration_std is not None
        else training_cfg.get("factorized_precision_exploration_std", 0.5),
    )
    gamma_size = _factorized_gamma(
        args.factorized_gamma_size
        if args.factorized_gamma_size is not None
        else training_cfg.get("factorized_gamma_size", FACTORIZED_GAMMA_SIZE),
        "factorized_gamma_size",
    )
    gamma_quality = _factorized_gamma(
        args.factorized_gamma_quality
        if args.factorized_gamma_quality is not None
        else training_cfg.get("factorized_gamma_quality", FACTORIZED_GAMMA_QUALITY),
        "factorized_gamma_quality",
    )
    update_steps = _strict_positive_int(
        args.factorized_update_steps
        if args.factorized_update_steps is not None
        else training_cfg.get("factorized_update_steps", FACTORIZED_UPDATE_STEPS),
        "factorized_update_steps",
    )
    actor_grad_clip = _strict_finite_float(
        args.factorized_actor_grad_clip_norm
        if args.factorized_actor_grad_clip_norm is not None
        else training_cfg.get("factorized_actor_grad_clip_norm", FACTORIZED_ACTOR_GRAD_CLIP_NORM),
        "factorized_actor_grad_clip_norm",
    )
    if actor_grad_clip <= 0.0:
        raise ValueError("factorized_actor_grad_clip_norm must be positive")
    batch_size = _strict_positive_int(
        args.batch_size if args.batch_size is not None else training_cfg.get("batch_size", DEFAULT_BATCH_SIZE),
        "batch_size",
    )
    buffer_size = _strict_positive_int(
        args.buffer_size
        if args.buffer_size is not None
        else training_cfg.get("buffer_size", DEFAULT_BUFFER_SIZE),
        "buffer_size",
    )
    actor_lr = _strict_nonnegative_float(
        args.lr_actor if args.lr_actor is not None else training_cfg.get("lr_actor", DEFAULT_ACTOR_LEARNING_RATE),
        "lr_actor",
    )
    critic_lr = _strict_nonnegative_float(
        args.lr_critic if args.lr_critic is not None else training_cfg.get("lr_critic", DEFAULT_CRITIC_LEARNING_RATE),
        "lr_critic",
    )

    probe = env.reset_factorized(
        scenes[0],
        quality_interval=quality_interval,
        requested_view_count=requested_view_count,
        noise_tolerance=noise_tolerance,
    )
    _validate_factorized_v1_pruning_environment(env)
    _validate_factorized_observation(probe, 19, "formal training probe")
    behavior_actor, target_actor = build_factorized_actors(19, actor_lr)
    behavior_critic, target_critic = build_factorized_critics(19, critic_lr)
    replay_memory = trans.FactorizedReplayMemory("factorized_dual_critic_replay", buffer_size)
    resume_episode = 0
    global_step = 0
    if args.resume:
        restored = load_factorized_training_checkpoint(
            args.resume, behavior_actor, target_actor, behavior_critic,
            target_critic, replay_memory,
        )
        resume_episode = int(restored["episode"])
        global_step = int(restored["global_step"])
    env.episode = resume_episode

    training_fields = [
        "episode", "global_step", "scene", "learner_mode", "quality_mode",
        "state_dim", "action_dim", "steps", "replay_size",
        "quality_ready_transition_count", "quality_checkpoint_count",
        "quality_observed_block_count", "final_quality_drop",
        "final_quality_feasible", "selected_actor_critic", "actor_update_source",
        "actor_update_count", "mean_actor_gradient_norm", "max_actor_gradient_norm",
        "size_critic_loss", "quality_critic_loss", "size_critic_update_count",
        "quality_critic_update_count", "mean_size_td_target", "mean_quality_td_target",
        "gamma_size", "gamma_quality", "update_steps", "sum_reward_P",
        "checkpoint_reward_D_sum", "final_compact_size_bytes",
        "final_compact_size_ratio", "final_left_bitbudget", "mean_actor_pruning",
        "mean_actor_precision", "mean_executed_pruning", "mean_executed_precision",
        "pruning_mode", "pruning_policy", "pruning_importance_source",
        "pruning_importance_version", "pruning_is_multiview",
        "pruning_uses_transmittance", "pruning_uses_background_replaceability",
        "grouping_method", "grouping_partition_sha256", "actual_num_groups",
        "target_gap", "checkpoint_path",
    ]
    gradient_fields = [
        "episode", "sample_idx", "selected_source", "quality_feasible",
        "state_quality_drop", "state_quality_margin", "raw_pruning_action",
        "raw_precision_action", "normalized_pruning_action",
        "normalized_precision_action", "normalized_pruning_gradient",
        "normalized_precision_gradient", "raw_pruning_gradient",
        "raw_precision_gradient", "gradient_norm_before_clip",
        "gradient_norm_after_clip",
    ]
    training_log = logs_dir / "factorized_training_log.csv"
    gradient_log = logs_dir / "factorized_actor_gradient_log.csv"
    with _open_csv_writer_with_schema_rotation(training_log, training_fields):
        pass
    with _open_csv_writer_with_schema_rotation(gradient_log, gradient_fields):
        pass

    checkpoint_config = dict(config)
    checkpoint_config["factorized_training"] = {
        "gamma_size": gamma_size,
        "gamma_quality": gamma_quality,
        "update_steps": update_steps,
        "actor_grad_clip_norm": actor_grad_clip,
        "quality_interval": quality_interval,
        "requested_view_count": requested_view_count,
        "pruning_exploration_std": pruning_std,
        "precision_exploration_std": precision_std,
        **_factorized_v1_pruning_provenance(),
    }
    episodes = _strict_positive_int(
        args.episodes if args.episodes is not None else training_cfg.get("episodes", 10),
        "episodes",
    )
    if resume_episode >= episodes:
        return
    for episode_id in range(resume_episode + 1, episodes + 1):
        scene = random.choice(scenes)
        step_before = global_step
        rollout = run_factorized_rollout(
            env, scene, behavior_actor, replay_memory,
            episode_id=episode_id,
            pruning_exploration_std=pruning_std,
            precision_exploration_std=precision_std,
            quality_interval=quality_interval,
            requested_view_count=requested_view_count,
            noise_tolerance=noise_tolerance,
            rng=np.random,
            deterministic=bool(args.deterministic_factorized_rollout),
        )
        append_factorized_quality_checkpoint_log(
            logs_dir / "factorized_quality_checkpoint_log.csv",
            rollout["quality_checkpoint_records"],
            scene=scene.get("name", Path(scene["ply_path"]).stem),
            global_step_before_episode=step_before,
            learner_mode="factorized_dual_critic_ddpg",
        )
        global_step += int(rollout["steps"])
        selected_source = select_factorized_actor_critic(rollout["final_quality_feasible"])
        critic_update = update_factorized_critics_from_replay(
            behavior_critic, target_critic, target_actor, replay_memory,
            batch_size=batch_size,
            update_steps=update_steps,
            gamma_size=gamma_size,
            gamma_quality=gamma_quality,
        )
        ready_updates = (
            critic_update["quality_critic_update_count"]
            if selected_source == "D"
            else critic_update["size_critic_update_count"]
        )
        if ready_updates:
            episode_transitions = [
                transition for transition in replay_memory.replay_memory
                if transition.episode_id == episode_id
            ]
            actor_update = update_factorized_actor_from_episode(
                behavior_actor, target_actor, behavior_critic,
                episode_transitions, selected_source=selected_source,
                max_gradient_norm=actor_grad_clip,
            )
        else:
            actor_update = {
                "actor_update_source": "none_not_ready", "actor_update_count": 0,
                "mean_actor_gradient_norm": 0.0, "max_actor_gradient_norm": 0.0,
                "gradient_rows": [],
            }
        gradient_rows = [{
            "episode": episode_id,
            "quality_feasible": bool(rollout["final_quality_feasible"]),
            **row,
        } for row in actor_update["gradient_rows"]]
        if gradient_rows:
            with gradient_log.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=gradient_fields).writerows(gradient_rows)

        final_info = rollout["final_info"]
        checkpoint_path = save_factorized_training_checkpoint(
            checkpoints_dir, behavior_actor, target_actor, behavior_critic,
            target_critic, replay_memory, episode_id, global_step,
            checkpoint_config,
            {
                "learner_mode": "factorized_dual_critic_ddpg",
                "quality_mode": final_info.get("quality_mode", ""),
                "selected_actor_critic": selected_source,
                "actor_update_source": actor_update["actor_update_source"],
                "critic_update": critic_update,
                "grouping_method": final_info.get("grouping_method"),
                "grouping_partition_sha256": final_info.get(
                    "grouping_partition_sha256"
                ),
                "actual_num_groups": final_info.get("actual_num_groups"),
                "target_num_groups": final_info.get("target_num_groups"),
                "target_gap": final_info.get("target_gap"),
                **_factorized_v1_pruning_provenance(),
            },
        )
        row = {
            "episode": episode_id,
            "global_step": global_step,
            "scene": scene.get("name", Path(scene["ply_path"]).stem),
            "learner_mode": "factorized_dual_critic_ddpg",
            "quality_mode": final_info.get("quality_mode", ""),
            "state_dim": 19,
            "action_dim": 2,
            "steps": rollout["steps"],
            "replay_size": rollout["replay_size_after"],
            "quality_ready_transition_count": critic_update["quality_ready_transition_count"],
            "quality_checkpoint_count": rollout["quality_checkpoint_count"],
            "quality_observed_block_count": rollout["quality_observed_block_count"],
            "final_quality_drop": rollout["final_quality_drop"],
            "final_quality_feasible": rollout["final_quality_feasible"],
            "selected_actor_critic": selected_source,
            "actor_update_source": actor_update["actor_update_source"],
            "actor_update_count": actor_update["actor_update_count"],
            "mean_actor_gradient_norm": actor_update["mean_actor_gradient_norm"],
            "max_actor_gradient_norm": actor_update["max_actor_gradient_norm"],
            **{key: critic_update[key] for key in (
                "size_critic_loss", "quality_critic_loss",
                "size_critic_update_count", "quality_critic_update_count",
                "mean_size_td_target", "mean_quality_td_target",
            )},
            "gamma_size": gamma_size,
            "gamma_quality": gamma_quality,
            "update_steps": update_steps,
            "sum_reward_P": rollout["sum_reward_P"],
            "checkpoint_reward_D_sum": rollout["checkpoint_reward_D_sum"],
            "final_compact_size_bytes": rollout["final_compact_size_bytes"],
            "final_compact_size_ratio": rollout["final_compact_size_ratio"],
            "final_left_bitbudget": rollout["final_left_bitbudget"],
            "mean_actor_pruning": rollout["mean_actor_pruning"],
            "mean_actor_precision": rollout["mean_actor_precision"],
            "mean_executed_pruning": rollout["mean_executed_pruning"],
            "mean_executed_precision": rollout["mean_executed_precision"],
            **_factorized_v1_pruning_provenance(),
            "grouping_method": final_info.get("grouping_method", ""),
            "grouping_partition_sha256": final_info.get("grouping_partition_sha256", ""),
            "actual_num_groups": final_info.get("actual_num_groups", ""),
            "target_gap": final_info.get("target_gap", ""),
            "checkpoint_path": str(checkpoint_path),
        }
        with training_log.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=training_fields).writerow(row)
        print(
            f"episode={episode_id} scene={row['scene']} selected={selected_source} "
            f"steps={row['steps']} replay={row['replay_size']}"
        )


_FIRST_VERSION_VALIDATION_REPORT: dict[str, Any] = {}


def _validate_first_version_training_core() -> bool:
    """Run the consolidated 15-leaf first-version training smoke test."""
    if _FIRST_VERSION_VALIDATION_REPORT.get("validated") is True:
        return True

    def require(condition: Any, message: str) -> None:
        if not bool(condition):
            raise AssertionError(message)

    def expect_error(callback: Callable[[], Any], message: str) -> None:
        try:
            callback()
        except ValueError:
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

    def write_ply(path: Path, xyz: np.ndarray) -> None:
        names = [
            "x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1",
            "f_dc_2", *[f"f_rest_{index}" for index in range(45)],
        ]
        vertices = np.zeros(
            len(xyz), dtype=np.dtype([(name, "<f4") for name in names])
        )
        vertices["x"], vertices["y"], vertices["z"] = xyz.T
        vertices["opacity"] = np.linspace(-2.0, 2.0, len(xyz), dtype=np.float32)
        for field_index, name in enumerate(names[4:], start=4):
            vertices[name] = np.linspace(
                0.01 + field_index * 0.001,
                0.99 + field_index * 0.001,
                len(xyz),
                dtype=np.float32,
            )
        header = [
            "ply", "format binary_little_endian 1.0",
            f"element vertex {len(vertices)}",
            *[f"property float {name}" for name in names], "end_header",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.write(("\n".join(header) + "\n").encode("ascii"))
            vertices.tofile(handle)

    def params_changed(before: list[np.ndarray], after: list[np.ndarray]) -> bool:
        return any(not np.array_equal(left, right) for left, right in zip(before, after))

    def artifact_episode(path: str | Path) -> int:
        """Extract the strict four-digit Episode token from an artifact name."""
        name = Path(path).name
        marker_index = name.find("_ep")
        token_start = marker_index + 3
        token = name[token_start : token_start + 4]
        require(
            marker_index >= 0
            and len(token) == 4
            and token.isdigit()
            and name[token_start + 4 : token_start + 5] == "_",
            f"artifact path has no strict _epNNNN_ token: {path}",
        )
        return int(token)

    random.seed(1701)
    np.random.seed(1701)
    with tempfile.TemporaryDirectory(prefix="first_version_training_validation_") as directory:
        root = Path(directory)
        ply_path = root / "scene.ply"
        write_ply(ply_path, validation_xyz())
        scene = {"name": "first_version_validation", "ply_path": str(ply_path)}
        environment = GS_Environment(
            scenes=[scene], output_root=root / "outputs", target_num_groups=15,
            min_group_size=2, factorized_octree_max_depth=2,
            use_dummy_reward=True, use_render=False, use_crossscore=False,
            quality_epsilon=0.0,
        )
        initial_state = environment.reset_factorized(
            scene, quality_interval=8, requested_view_count=2,
            noise_tolerance=0.0,
        )
        probe_episode_after_reset = environment.episode
        require(
            probe_episode_after_reset == 1,
            "probe reset did not temporarily advance Environment episode to 1",
        )
        environment.episode = 0
        probe_episode_after_restore = environment.episode
        require(
            probe_episode_after_restore == 0,
            "formal-loop preparation did not restore Environment episode to 0",
        )
        _validate_factorized_v1_pruning_environment(environment)
        require(initial_state.shape == (19,), "formal state is not shape (19,)")
        require(np.all(np.isfinite(initial_state)), "formal state is not finite")
        require(
            environment.factorized_group_count == 15,
            "validation geometry did not produce 15 leaves",
        )

        behavior_actor, target_actor = build_factorized_actors(
            19, DEFAULT_ACTOR_LEARNING_RATE
        )
        behavior_critic, target_critic = build_factorized_critics(
            19, DEFAULT_CRITIC_LEARNING_RATE
        )
        action = behavior_actor.predict_action(initial_state.reshape(1, -1))
        require(action.shape == (1, 2), "Actor action is not shape (1, 2)")
        require(
            np.all(action >= 0.0) and np.all(action <= FACTORIZED_ACTION_BOUND),
            "Actor action is outside its independent bounds",
        )
        class FixedNoise:
            values = iter((0.25, -0.5))

            @classmethod
            def normal(cls, _mean: float, _std: float) -> float:
                return next(cls.values)

        explored = apply_factorized_exploration(
            action[0], pruning_std=1.0, precision_std=1.0,
            rng=FixedNoise, deterministic=False,
        )
        require(
            np.isclose(explored[0], action[0, 0] + 0.25)
            and np.isclose(explored[1], action[0, 1] - 0.5),
            "the two exploration axes are not independent",
        )

        replay = trans.FactorizedReplayMemory("first_version_validation", 64)
        rollout = run_factorized_rollout(
            environment, scene, behavior_actor, replay, episode_id=1,
            pruning_exploration_std=0.0, precision_exploration_std=0.0,
            quality_interval=8, requested_view_count=2, noise_tolerance=0.0,
            rng=np.random, deterministic=True,
        )
        require(rollout["steps"] == 15, "formal Episode is not 15 steps")
        require(len(replay.replay_memory) == 15, "Replay does not contain 15 transitions")
        require(
            all(transition.action.shape == (2,) for transition in replay.replay_memory),
            "Replay contains a non-factorized action",
        )
        horizons = [transition.quality_horizon for transition in replay.replay_memory]
        require(
            horizons == [8, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1],
            "quality block horizons are incorrect",
        )
        records = rollout["quality_checkpoint_records"]
        require(len(records) == 2, "quality log does not have the 8+7 checkpoints")
        require(
            records[0].block_start_step_index == 0
            and records[0].block_end_step_index == 7
            and records[0].block_length == 8
            and not records[0].is_terminal,
            "first quality checkpoint is incorrect",
        )
        require(
            records[1].block_start_step_index == 8
            and records[1].block_end_step_index == 14
            and records[1].block_length == 7
            and records[1].is_terminal,
            "terminal partial quality checkpoint is incorrect",
        )
        require(
            sum(record.checkpoint_reward_D for record in records)
            == rollout["checkpoint_reward_D_sum"],
            "checkpoint quality rewards were not recorded exactly once",
        )
        require(
            rollout["pruning_mode"] == FACTORIZED_V1_PRUNING_MODE
            and rollout["pruning_importance_source"] == FACTORIZED_V1_IMPORTANCE_SOURCE
            and rollout["pruning_is_multiview"] is False,
            "opacity-baseline provenance is incorrect",
        )
        final_info = rollout["final_info"]
        first_environment_episode = environment.episode
        require(
            first_environment_episode == rollout["episode_id"] == 1,
            "first formal rollout is not aligned to Environment episode 1",
        )
        require(
            all(transition.episode_id == 1 for transition in replay.replay_memory),
            "first Episode Replay IDs are not all 1",
        )
        require(
            all(record.episode_id == 1 for record in records),
            "first Episode quality records are not all 1",
        )
        first_ply_episode = artifact_episode(final_info["compressed_ply_path"])
        first_compact_path = final_info.get("compression_stats", {}).get(
            "compact_package_path", ""
        )
        first_compact_episode = artifact_episode(first_compact_path)
        require(
            first_ply_episode == first_compact_episode == 1,
            "first compression artifacts used ep0002 instead of ep0001",
        )
        require(final_info["actual_num_groups"] == 15, "wrong final group count")
        require(
            len(final_info["grouping_partition_sha256"]) == 64,
            "partition digest is missing",
        )
        require(rollout["final_compact_size_bytes"] > 0, "compact V2 output is missing")
        expected_size_return = float(
            np.log(
                environment.original_size_bytes
                / rollout["final_compact_size_bytes"]
            )
        )
        require(
            np.isclose(rollout["sum_reward_P"], expected_size_return),
            "dense size reward and terminal correction did not telescope",
        )
        compact_format = final_info.get("compression_stats", {}).get(
            "compact_format"
        )
        require(
            compact_format == "rl_factorized_3dgs_compact_v2",
            "terminal compact format is not the formal V2 schema",
        )

        second_replay = trans.FactorizedReplayMemory(
            "second_episode_alignment", 64
        )
        second_rollout = run_factorized_rollout(
            environment, scene, behavior_actor, second_replay, episode_id=2,
            pruning_exploration_std=0.0, precision_exploration_std=0.0,
            quality_interval=8, requested_view_count=2, noise_tolerance=0.0,
            rng=np.random, deterministic=True,
        )
        second_info = second_rollout["final_info"]
        second_environment_episode = environment.episode
        second_ply_episode = artifact_episode(second_info["compressed_ply_path"])
        second_compact_episode = artifact_episode(
            second_info.get("compression_stats", {}).get(
                "compact_package_path", ""
            )
        )
        require(
            (first_environment_episode, second_environment_episode) == (1, 2),
            "continuous Environment Episode sequence is not 1, 2",
        )
        require(
            second_rollout["episode_id"] == 2
            and all(
                transition.episode_id == 2
                for transition in second_replay.replay_memory
            )
            and all(
                record.episode_id == 2
                for record in second_rollout["quality_checkpoint_records"]
            )
            and second_ply_episode == second_compact_episode == 2,
            "second Episode Replay, Environment, or artifacts are misaligned",
        )
        episode_alignment_log = root / "episode_alignment_training_log.csv"
        with episode_alignment_log.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("episode",))
            writer.writeheader()
            writer.writerows(({"episode": 1}, {"episode": 2}))
        with episode_alignment_log.open("r", newline="", encoding="utf-8") as handle:
            logged_episodes = [
                int(row["episode"]) for row in csv.DictReader(handle)
            ]
        require(
            logged_episodes == [1, 2]
            and logged_episodes[0]
            == first_environment_episode
            == first_ply_episode
            == first_compact_episode
            == replay.replay_memory[0].episode_id
            and logged_episodes[1]
            == second_environment_episode
            == second_ply_episode
            == second_compact_episode
            == second_replay.replay_memory[0].episode_id,
            "Replay, training log, Environment, PLY, and compact Episodes differ",
        )

        critic_update = update_factorized_critics_from_replay(
            behavior_critic, target_critic, target_actor, replay,
            batch_size=4, update_steps=1,
            gamma_size=FACTORIZED_GAMMA_SIZE,
            gamma_quality=FACTORIZED_GAMMA_QUALITY,
        )
        require(critic_update["size_critic_update_count"] == 1, "Size Critic did not update")
        require(critic_update["quality_critic_update_count"] == 1, "Quality Critic did not update")
        require(
            critic_update["quality_ready_transition_count"] == 15,
            "Quality Critic readiness count is incorrect",
        )
        require(select_factorized_actor_critic(False) == "D", "quality violation did not select D")
        require(select_factorized_actor_critic(True) == "P", "quality feasibility did not select P")

        episode_transitions = list(replay.replay_memory)
        calls = {"D": 0, "P": 0}
        original_D = behavior_critic.action_gradient_D
        original_P = behavior_critic.action_gradient_P

        def tracked_D(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
            calls["D"] += 1
            return original_D(states, actions)

        def tracked_P(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
            calls["P"] += 1
            return original_P(states, actions)

        behavior_critic.action_gradient_D = tracked_D
        behavior_critic.action_gradient_P = tracked_P
        target_before = [value.copy() for value in target_actor.get_params()]
        update_D = update_factorized_actor_from_episode(
            behavior_actor, target_actor, behavior_critic, episode_transitions,
            selected_source="D", max_gradient_norm=FACTORIZED_ACTOR_GRAD_CLIP_NORM,
        )
        require(calls == {"D": 1, "P": 0}, "D Actor update used the wrong Critic")
        calls.update(D=0, P=0)
        update_P = update_factorized_actor_from_episode(
            behavior_actor, target_actor, behavior_critic, episode_transitions,
            selected_source="P", max_gradient_norm=FACTORIZED_ACTOR_GRAD_CLIP_NORM,
        )
        require(calls == {"D": 0, "P": 1}, "P Actor update used the wrong Critic")
        behavior_critic.action_gradient_D = original_D
        behavior_critic.action_gradient_P = original_P
        require(update_D["actor_update_source"] == "D", "D update report is wrong")
        require(update_P["actor_update_source"] == "P", "P update report is wrong")
        require(
            params_changed(target_before, target_actor.get_params()),
            "target Actor was not softly updated",
        )
        updated_action = behavior_actor.predict_action(initial_state.reshape(1, -1))
        require(
            np.all(updated_action >= 0.0)
            and np.all(updated_action <= FACTORIZED_ACTION_BOUND),
            "updated Actor left the two-axis bounds",
        )

        quality_log = root / "factorized_quality_checkpoint_log.csv"
        append_factorized_quality_checkpoint_log(
            quality_log, records, scene=scene["name"],
            global_step_before_episode=100,
            learner_mode="factorized_dual_critic_ddpg",
        )
        with quality_log.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == 2, "quality checkpoint CSV does not have two rows")
        require(
            tuple(rows[0]) == FACTORIZED_QUALITY_CHECKPOINT_LOG_FIELDS,
            "quality checkpoint CSV schema is incorrect",
        )
        rotated_log = root / "rotated_quality_log.csv"
        rotated_log.write_text("obsolete_header\n", encoding="utf-8")
        append_factorized_quality_checkpoint_log(
            rotated_log, records, scene=scene["name"],
            global_step_before_episode=100,
            learner_mode="factorized_dual_critic_ddpg",
        )
        require(
            tuple(next(csv.reader(rotated_log.read_text(encoding="utf-8").splitlines())))
            == FACTORIZED_QUALITY_CHECKPOINT_LOG_FIELDS
            and len(list(root.glob("rotated_quality_log.schema_mismatch_*.csv"))) == 1,
            "quality checkpoint CSV schema rotation failed",
        )

        probe_states = np.vstack((initial_state, initial_state)).astype(np.float64)
        probe_actions = normalize_factorized_action_batch(
            behavior_actor.predict_action(probe_states)
        )
        saved_actor = behavior_actor.predict_action(probe_states)
        saved_target_actor = target_actor.predict_action(probe_states)
        saved_q_D, saved_q_P = behavior_critic.predict(probe_states, probe_actions)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        python_probe = random.Random(); python_probe.setstate(python_state)
        numpy_probe = np.random.RandomState(); numpy_probe.set_state(numpy_state)
        expected_python = python_probe.random()
        expected_numpy = float(numpy_probe.random_sample())
        checkpoint_path = save_factorized_training_checkpoint(
            root / "checkpoints", behavior_actor, target_actor,
            behavior_critic, target_critic, replay, 1, 15,
            {"validation": True},
            {"learner_mode": "factorized_dual_critic_ddpg"},
        )
        random.random(); np.random.random_sample()
        restored_actor, restored_target_actor = build_factorized_actors(
            19, DEFAULT_ACTOR_LEARNING_RATE
        )
        restored_critic, restored_target_critic = build_factorized_critics(
            19, DEFAULT_CRITIC_LEARNING_RATE
        )
        restored_replay = trans.FactorizedReplayMemory("restored_validation", 64)
        restored_checkpoint = load_factorized_training_checkpoint(
            checkpoint_path, restored_actor, restored_target_actor,
            restored_critic, restored_target_critic, restored_replay,
        )
        restored_q_D, restored_q_P = restored_critic.predict(probe_states, probe_actions)
        require(np.array_equal(restored_actor.predict_action(probe_states), saved_actor), "Actor restore mismatch")
        require(np.array_equal(restored_target_actor.predict_action(probe_states), saved_target_actor), "target Actor restore mismatch")
        require(np.array_equal(restored_q_D, saved_q_D) and np.array_equal(restored_q_P, saved_q_P), "dual Critic restore mismatch")
        require(len(restored_replay.replay_memory) == 15, "Replay restore mismatch")
        require(np.array_equal(restored_replay.replay_memory[0].state, replay.replay_memory[0].state), "Replay state restore mismatch")
        require(random.random() == expected_python, "Python RNG restore mismatch")
        require(float(np.random.random_sample()) == expected_numpy, "NumPy RNG restore mismatch")
        require(restored_checkpoint["episode"] == 1 and restored_checkpoint["global_step"] == 15, "training counters restore mismatch")

        resume_checkpoint_episode = 7
        resume_checkpoint_path = save_factorized_training_checkpoint(
            root / "resume_alignment_checkpoints", behavior_actor, target_actor,
            behavior_critic, target_critic, replay, resume_checkpoint_episode,
            105, {"validation": True},
            {"learner_mode": "factorized_dual_critic_ddpg"},
        )
        resume_actor, resume_target_actor = build_factorized_actors(
            19, DEFAULT_ACTOR_LEARNING_RATE
        )
        resume_critic, resume_target_critic = build_factorized_critics(
            19, DEFAULT_CRITIC_LEARNING_RATE
        )
        resume_replay = trans.FactorizedReplayMemory(
            "resume_episode_alignment", 64
        )
        resume_checkpoint = load_factorized_training_checkpoint(
            resume_checkpoint_path, resume_actor, resume_target_actor,
            resume_critic, resume_target_critic, resume_replay,
        )
        resume_environment = GS_Environment(
            scenes=[scene], output_root=root / "resume_outputs",
            target_num_groups=15, min_group_size=2,
            factorized_octree_max_depth=2, use_dummy_reward=True,
            use_render=False, use_crossscore=False, quality_epsilon=0.0,
        )
        resume_environment.reset_factorized(
            scene, quality_interval=8, requested_view_count=2,
            noise_tolerance=0.0,
        )
        require(
            resume_environment.episode == 1,
            "resume-path probe did not temporarily set Environment episode to 1",
        )
        resumed_episode = int(resume_checkpoint["episode"])
        resume_environment.episode = resumed_episode
        resume_replay_size_before = len(resume_replay.replay_memory)
        resumed_rollout = run_factorized_rollout(
            resume_environment, scene, resume_actor, resume_replay,
            episode_id=resumed_episode + 1, pruning_exploration_std=0.0,
            precision_exploration_std=0.0, quality_interval=8,
            requested_view_count=2, noise_tolerance=0.0, rng=np.random,
            deterministic=True,
        )
        resumed_info = resumed_rollout["final_info"]
        resumed_ply_episode = artifact_episode(
            resumed_info["compressed_ply_path"]
        )
        resumed_compact_episode = artifact_episode(
            resumed_info.get("compression_stats", {}).get(
                "compact_package_path", ""
            )
        )
        resumed_transitions = resume_replay.replay_memory[
            resume_replay_size_before:
        ]
        require(
            resumed_episode == resume_checkpoint_episode
            and resume_environment.episode == resumed_episode + 1
            and resumed_rollout["episode_id"] == resumed_episode + 1
            and resumed_ply_episode == resumed_episode + 1
            and resumed_compact_episode == resumed_episode + 1
            and len(resumed_transitions) == 15
            and all(
                transition.episode_id == resumed_episode + 1
                for transition in resumed_transitions
            ),
            "checkpoint resume did not continue at Environment episode N+1",
        )

        with checkpoint_path.open("rb") as handle:
            invalid_checkpoint = pickle.load(handle)
        invalid_checkpoint["behavior_critic"]["critic_P"]["weights"][0] = (
            invalid_checkpoint["behavior_critic"]["critic_P"]["weights"][0][:-1]
        )
        invalid_path = root / "invalid.pkl"
        with invalid_path.open("wb") as handle:
            pickle.dump(invalid_checkpoint, handle)
        component_snapshots = {
            "behavior_actor": pickle.dumps(restored_actor.state_dict()),
            "target_actor": pickle.dumps(restored_target_actor.state_dict()),
            "behavior_critic": pickle.dumps(restored_critic.state_dict()),
            "target_critic": pickle.dumps(restored_target_critic.state_dict()),
            "replay": pickle.dumps(restored_replay.state_dict()),
            "python_rng": pickle.dumps(random.getstate()),
            "numpy_rng": pickle.dumps(np.random.get_state()),
        }
        expect_error(
            lambda: load_factorized_training_checkpoint(
                invalid_path, restored_actor, restored_target_actor,
                restored_critic, restored_target_critic, restored_replay,
            ),
            "invalid Critic branch in a training checkpoint was accepted",
        )
        require(
            component_snapshots["behavior_actor"]
            == pickle.dumps(restored_actor.state_dict())
            and component_snapshots["target_actor"]
            == pickle.dumps(restored_target_actor.state_dict())
            and component_snapshots["behavior_critic"]
            == pickle.dumps(restored_critic.state_dict())
            and component_snapshots["target_critic"]
            == pickle.dumps(restored_target_critic.state_dict())
            and component_snapshots["replay"]
            == pickle.dumps(restored_replay.state_dict())
            and component_snapshots["python_rng"]
            == pickle.dumps(random.getstate())
            and component_snapshots["numpy_rng"]
            == pickle.dumps(np.random.get_state()),
            "failed training checkpoint load partially committed state",
        )

        _FIRST_VERSION_VALIDATION_REPORT.update({
            "validated": True,
            "group_count": 15,
            "state_shape": tuple(initial_state.shape),
            "actor_action_shape": tuple(action.shape),
            "checkpoint_blocks": [(0, 7, 8), (8, 14, 7)],
            "replay_transition_count": len(replay.replay_memory),
            "quality_horizons": horizons,
            "quality_ready_transition_count": critic_update["quality_ready_transition_count"],
            "size_critic_update_count": critic_update["size_critic_update_count"],
            "quality_critic_update_count": critic_update["quality_critic_update_count"],
            "actor_sources": (update_D["actor_update_source"], update_P["actor_update_source"]),
            "compact_size_bytes": rollout["final_compact_size_bytes"],
            "compact_format": compact_format,
            "partition_sha256": final_info["grouping_partition_sha256"],
            "reward_P_sum": rollout["sum_reward_P"],
            "reward_P_telescoping_target": expected_size_return,
            "reward_telescoping": True,
            "checkpoint_roundtrip": True,
            "training_checkpoint_transactional_failure": True,
            "opacity_baseline": True,
            "probe_episode_after_reset": probe_episode_after_reset,
            "probe_episode_after_restore": probe_episode_after_restore,
            "environment_episode_sequence": (
                first_environment_episode,
                second_environment_episode,
            ),
            "training_log_episodes": tuple(logged_episodes),
            "first_ply_episode": first_ply_episode,
            "first_compact_episode": first_compact_episode,
            "resume_checkpoint_episode": resumed_episode,
            "resume_next_environment_episode": resume_environment.episode,
            "resume_next_ply_episode": resumed_ply_episode,
            "resume_next_compact_episode": resumed_compact_episode,
            "episode_alignment": True,
        })
    return True


def validate_first_version_training_pipeline() -> bool:
    """Validate the complete formal first-version training pipeline."""
    return _validate_first_version_training_core()


def validate_factorized_training_wiring() -> bool:
    """Compatibility name for the consolidated formal validation."""
    return _validate_first_version_training_core()


def validate_factorized_dual_critic_training() -> bool:
    """Compatibility name for the consolidated formal validation."""
    return _validate_first_version_training_core()


def validate_factorized_quality_checkpoint_logging() -> bool:
    """Compatibility name for the consolidated formal validation."""
    return _validate_first_version_training_core()


def validate_factorized_v1_opacity_training() -> bool:
    """Compatibility name for the consolidated formal validation."""
    return _validate_first_version_training_core()


def main() -> None:
    """Parse the formal first-version command line and start training."""
    parser = argparse.ArgumentParser(
        description="First-version dual-critic octree 3DGS compression training."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--ply", default="")
    parser.add_argument("--scene-path", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--lr-actor", type=float, default=None)
    parser.add_argument("--lr-critic", type=float, default=None)
    parser.add_argument("--pruning-exploration-std", type=float, default=None)
    parser.add_argument("--precision-exploration-std", type=float, default=None)
    parser.add_argument("--quality-interval", type=int, default=None)
    parser.add_argument("--requested-view-count", type=int, default=None)
    parser.add_argument("--quality-noise-tolerance", type=float, default=None)
    parser.add_argument("--factorized-gamma-size", type=float, default=None)
    parser.add_argument("--factorized-gamma-quality", type=float, default=None)
    parser.add_argument("--factorized-update-steps", type=int, default=None)
    parser.add_argument("--factorized-actor-grad-clip-norm", type=float, default=None)
    parser.add_argument("--deterministic-factorized-rollout", action="store_true")
    parser.add_argument("--target-groups", type=int, default=None)
    parser.add_argument("--min-group-size", type=int, default=None)
    parser.add_argument("--octree-max-depth", type=int, default=None)
    parser.add_argument("--target-size-ratio", type=float, default=None)
    parser.add_argument("--opacity-low-threshold", type=float, default=None)
    quality_mode = parser.add_mutually_exclusive_group()
    quality_mode.add_argument("--use-dummy-reward", action="store_true")
    quality_mode.add_argument("--use-render", action="store_true")
    quality_mode.add_argument("--use-crossscore", action="store_true")
    parser.add_argument("--allow-crossscore-placeholder", action="store_true")
    parser.add_argument("--force-recompute-original-score", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
