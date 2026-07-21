"""Deterministic, read-only diagnostics for factorized 3DGS training CSV logs.

Fixed-subset checkpoint quality is used only for within-episode trajectories
and cliff attribution. Terminal full-view quality is used for cross-episode RD
analysis when available; fixed-subset fallback points remain a separate scope.
This module never reads checkpoints, renders images, or changes training state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
import argparse
import csv
import json
import math
import os
import re
import tempfile

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


SCHEMA = "factorized_training_diagnostics_v1"


@dataclass(frozen=True)
class EpisodeDiagnosticRecord:
    episode: int
    global_step: int
    scene: str
    learner_mode: str
    steps: int
    final_quality_drop: float
    final_quality_feasible: bool
    selected_actor_critic: str
    actor_update_source: str
    actor_update_count: int
    mean_actor_gradient_norm: float
    max_actor_gradient_norm: float
    size_critic_loss: float
    quality_critic_loss: float
    size_critic_update_count: int
    quality_critic_update_count: int
    final_compact_size_bytes: int
    final_compact_size_ratio: float
    mean_actor_pruning: float
    mean_actor_precision: float
    mean_environment_pruning: float
    mean_environment_precision: float
    quality_checkpoint_count: int
    quality_observed_block_count: int


@dataclass(frozen=True)
class CheckpointDiagnosticRecord:
    global_step: int
    episode_id: int
    scene: str
    local_checkpoint_index: int
    local_step_index: int
    groups_processed: int
    total_group_count: int
    block_start_step_index: int
    block_end_step_index: int
    block_length: int
    is_terminal: bool
    checkpoint_scope: str
    quality_observed: bool
    fixed_view_manifest_path: str
    previous_subset_quality_drop: float | None
    current_subset_quality_drop: float | None
    subset_quality_drop_delta: float | None
    quality_epsilon: float | None
    subset_quality_margin: float | None
    subset_quality_feasible: bool | None
    subset_quality_violation: float | None
    terminal_full_view_quality_drop: float | None
    terminal_full_view_quality_feasible: bool | None
    terminal_full_view_quality_violation: float | None
    terminal_quality_scope: str | None
    estimated_size_ratio_before_block: float
    estimated_size_ratio_after_block: float
    block_size_ratio_reduction: float
    block_log_size_reward_sum: float
    compact_size_ratio: float | None
    terminal_size_correction: float | None
    block_mean_executed_pruning: float
    block_mean_executed_precision: float
    executed_pruning_histogram_json: str
    executed_precision_histogram_json: str


@dataclass(frozen=True)
class ActorGradientDiagnosticRecord:
    episode: int
    sample_idx: int
    selected_source: str
    quality_feasible: bool
    state_quality_drop: float
    state_quality_margin: float
    raw_pruning_action: float
    raw_precision_action: float
    normalized_pruning_action: float
    normalized_precision_action: float
    normalized_pruning_gradient: float
    normalized_precision_gradient: float
    raw_pruning_gradient: float
    raw_precision_gradient: float
    gradient_norm_before_clip: float
    gradient_norm_after_clip: float


@dataclass(frozen=True)
class QualityCliffEvent:
    scene: str
    episode_id: int
    global_step: int
    local_checkpoint_index: int
    block_start_step_index: int
    block_end_step_index: int
    block_length: int
    fixed_view_manifest_path: str
    is_terminal: bool
    previous_subset_quality_drop: float
    current_subset_quality_drop: float
    subset_quality_drop_delta: float
    quality_epsilon: float
    subset_quality_margin: float | None
    subset_quality_violation: float | None
    block_size_ratio_reduction: float
    quality_cost_per_size_gain: float | None
    block_mean_executed_pruning: float
    block_mean_executed_precision: float
    executed_pruning_histogram_json: str
    executed_precision_histogram_json: str
    detection_threshold: float
    robust_z_score: float | None
    detection_reasons: str


@dataclass(frozen=True)
class ParetoDiagnosticPoint:
    scene: str
    episode: int
    global_step: int
    quality_scope: str
    quality_drop: float
    quality_feasible: bool
    compact_size_ratio: float
    compression_factor: float
    is_pareto: bool
    dominated_by_count: int


@dataclass(frozen=True)
class TrainingDiagnosticsResult:
    schema: str
    source_paths: dict[str, str | None]
    episode_count: int
    checkpoint_count: int
    gradient_row_count: int
    scenes: tuple[str, ...]
    cliff_events: tuple[QualityCliffEvent, ...]
    pareto_points: tuple[ParetoDiagnosticPoint, ...]
    summaries: dict[str, Any]
    warnings: tuple[str, ...]
    generated_files: tuple[str, ...]


EPISODE_COLUMNS = {
    "episode", "global_step", "scene", "learner_mode", "steps",
    "final_quality_drop", "final_quality_feasible", "selected_actor_critic",
    "actor_update_source", "actor_update_count", "mean_actor_gradient_norm",
    "max_actor_gradient_norm", "size_critic_loss", "quality_critic_loss",
    "size_critic_update_count", "quality_critic_update_count",
    "final_compact_size_bytes", "final_compact_size_ratio",
    "mean_actor_pruning", "mean_actor_precision", "mean_environment_pruning",
    "mean_environment_precision", "quality_checkpoint_count",
    "quality_observed_block_count",
}
CHECKPOINT_COLUMNS = {
    "global_step", "episode_id", "scene", "local_checkpoint_index",
    "local_step_index", "groups_processed", "total_group_count",
    "block_start_step_index", "block_end_step_index", "block_length",
    "is_terminal", "checkpoint_scope", "quality_observed",
    "fixed_view_manifest_path", "previous_subset_quality_drop",
    "current_subset_quality_drop", "subset_quality_drop_delta", "quality_epsilon",
    "subset_quality_margin", "subset_quality_feasible", "subset_quality_violation",
    "terminal_full_view_quality_drop", "terminal_full_view_quality_feasible",
    "terminal_full_view_quality_violation", "terminal_quality_scope",
    "estimated_size_ratio_before_block", "estimated_size_ratio_after_block",
    "block_size_ratio_reduction", "block_log_size_reward_sum", "compact_size_ratio",
    "terminal_size_correction", "block_mean_executed_pruning",
    "block_mean_executed_precision", "executed_pruning_histogram_json",
    "executed_precision_histogram_json",
}
GRADIENT_COLUMNS = {
    "episode", "sample_idx", "selected_source", "quality_feasible",
    "state_quality_drop", "state_quality_margin", "raw_pruning_action",
    "raw_precision_action", "normalized_pruning_action",
    "normalized_precision_action", "normalized_pruning_gradient",
    "normalized_precision_gradient", "raw_pruning_gradient",
    "raw_precision_gradient", "gradient_norm_before_clip",
    "gradient_norm_after_clip",
}


def _rows(path: str | Path, required: set[str], *, allow_header_only: bool) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"CSV must be an existing regular file: {source}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        raw_header = next(csv.reader(handle), None)
        if not raw_header or any(not name for name in raw_header):
            raise ValueError(f"CSV header is empty: {source}")
        if len(raw_header) != len(set(raw_header)):
            raise ValueError(f"CSV has duplicate header fields: {source}")
        missing = required - set(raw_header)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        handle.seek(0)
        result = list(csv.DictReader(handle))
    if not result and not allow_header_only:
        raise ValueError(f"CSV contains no data rows: {source}")
    return result


def _integer(row: dict[str, str], name: str, *, positive: bool = False) -> int:
    text = row[name]
    if text == "" or text.lower() in {"true", "false"} or not re.fullmatch(r"[+-]?[0-9]+", text):
        raise ValueError(f"{name} must be a strict integer")
    value = int(text)
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"{name} is outside its valid integer range")
    return value


def _number(row: dict[str, str], name: str, *, optional: bool = False) -> float | None:
    text = row[name]
    if optional and text == "":
        return None
    if text == "":
        raise ValueError(f"{name} must not be empty")
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _boolean(row: dict[str, str], name: str, *, optional: bool = False) -> bool | None:
    text = row[name]
    if optional and text == "":
        return None
    if text not in {"true", "false"}:
        raise ValueError(f"{name} must be lowercase true or false")
    return text == "true"


def _histogram(text: str, maximum: int, block_length: int) -> str:
    try:
        value = json.loads(text, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("histogram must be strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {str(i) for i in range(maximum + 1)}:
        raise ValueError("histogram has an invalid key set")
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in value.values()):
        raise ValueError("histogram counts must be nonnegative strict integers")
    if sum(value.values()) != block_length:
        raise ValueError("histogram count sum does not equal block_length")
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def load_factorized_training_log(path: str | Path) -> tuple[EpisodeDiagnosticRecord, ...]:
    """Read and strictly validate the Episode-level factorized training CSV."""
    records: list[EpisodeDiagnosticRecord] = []
    for row in _rows(path, EPISODE_COLUMNS, allow_header_only=False):
        record = EpisodeDiagnosticRecord(
            episode=_integer(row, "episode", positive=True), global_step=_integer(row, "global_step", positive=True),
            scene=row["scene"], learner_mode=row["learner_mode"], steps=_integer(row, "steps", positive=True),
            final_quality_drop=float(_number(row, "final_quality_drop")), final_quality_feasible=bool(_boolean(row, "final_quality_feasible")),
            selected_actor_critic=row["selected_actor_critic"], actor_update_source=row["actor_update_source"],
            actor_update_count=_integer(row, "actor_update_count"), mean_actor_gradient_norm=float(_number(row, "mean_actor_gradient_norm")),
            max_actor_gradient_norm=float(_number(row, "max_actor_gradient_norm")), size_critic_loss=float(_number(row, "size_critic_loss")),
            quality_critic_loss=float(_number(row, "quality_critic_loss")), size_critic_update_count=_integer(row, "size_critic_update_count"),
            quality_critic_update_count=_integer(row, "quality_critic_update_count"), final_compact_size_bytes=_integer(row, "final_compact_size_bytes", positive=True),
            final_compact_size_ratio=float(_number(row, "final_compact_size_ratio")), mean_actor_pruning=float(_number(row, "mean_actor_pruning")),
            mean_actor_precision=float(_number(row, "mean_actor_precision")), mean_environment_pruning=float(_number(row, "mean_environment_pruning")),
            mean_environment_precision=float(_number(row, "mean_environment_precision")), quality_checkpoint_count=_integer(row, "quality_checkpoint_count"),
            quality_observed_block_count=_integer(row, "quality_observed_block_count"),
        )
        if not record.scene or record.learner_mode != "factorized_dual_critic_ddpg":
            raise ValueError("episode scene or learner_mode is invalid")
        if not 0.0 < record.final_compact_size_ratio <= 1.0 or record.size_critic_loss < 0 or record.quality_critic_loss < 0:
            raise ValueError("episode size ratio or Critic loss is invalid")
        expected = "P" if record.final_quality_feasible else "D"
        if record.selected_actor_critic != expected or record.actor_update_source not in {"D", "P", "none_not_ready"}:
            raise ValueError("episode Critic selection is inconsistent")
        if record.actor_update_source in {"D", "P"} and record.actor_update_source != expected:
            raise ValueError("Actor update source does not match selected Critic")
        if record.actor_update_count == 0 and record.actor_update_source != "none_not_ready":
            raise ValueError("zero Actor updates require none_not_ready")
        records.append(record)
    records.sort(key=lambda item: (item.scene, item.episode))
    if len({(item.scene, item.episode) for item in records}) != len(records):
        raise ValueError("duplicate (scene, episode) rows")
    for scene in sorted({item.scene for item in records}):
        group = [item for item in records if item.scene == scene]
        if any(right.global_step <= left.global_step for left, right in zip(group, group[1:])):
            raise ValueError("global_step must increase within each scene")
    return tuple(records)


def load_factorized_quality_checkpoint_log(path: str | Path) -> tuple[CheckpointDiagnosticRecord, ...]:
    """Read and strictly validate fixed-subset quality checkpoint records."""
    records: list[CheckpointDiagnosticRecord] = []
    for row in _rows(path, CHECKPOINT_COLUMNS, allow_header_only=False):
        length = _integer(row, "block_length", positive=True)
        record = CheckpointDiagnosticRecord(
            global_step=_integer(row, "global_step", positive=True), episode_id=_integer(row, "episode_id", positive=True), scene=row["scene"],
            local_checkpoint_index=_integer(row, "local_checkpoint_index"), local_step_index=_integer(row, "local_step_index"),
            groups_processed=_integer(row, "groups_processed", positive=True), total_group_count=_integer(row, "total_group_count", positive=True),
            block_start_step_index=_integer(row, "block_start_step_index"), block_end_step_index=_integer(row, "block_end_step_index"), block_length=length,
            is_terminal=bool(_boolean(row, "is_terminal")), checkpoint_scope=row["checkpoint_scope"], quality_observed=bool(_boolean(row, "quality_observed")),
            fixed_view_manifest_path=row["fixed_view_manifest_path"], previous_subset_quality_drop=_number(row, "previous_subset_quality_drop", optional=True),
            current_subset_quality_drop=_number(row, "current_subset_quality_drop", optional=True), subset_quality_drop_delta=_number(row, "subset_quality_drop_delta", optional=True),
            quality_epsilon=_number(row, "quality_epsilon", optional=True), subset_quality_margin=_number(row, "subset_quality_margin", optional=True),
            subset_quality_feasible=_boolean(row, "subset_quality_feasible", optional=True), subset_quality_violation=_number(row, "subset_quality_violation", optional=True),
            terminal_full_view_quality_drop=_number(row, "terminal_full_view_quality_drop", optional=True), terminal_full_view_quality_feasible=_boolean(row, "terminal_full_view_quality_feasible", optional=True),
            terminal_full_view_quality_violation=_number(row, "terminal_full_view_quality_violation", optional=True), terminal_quality_scope=row["terminal_quality_scope"] or None,
            estimated_size_ratio_before_block=float(_number(row, "estimated_size_ratio_before_block")), estimated_size_ratio_after_block=float(_number(row, "estimated_size_ratio_after_block")),
            block_size_ratio_reduction=float(_number(row, "block_size_ratio_reduction")), block_log_size_reward_sum=float(_number(row, "block_log_size_reward_sum")),
            compact_size_ratio=_number(row, "compact_size_ratio", optional=True), terminal_size_correction=_number(row, "terminal_size_correction", optional=True),
            block_mean_executed_pruning=float(_number(row, "block_mean_executed_pruning")), block_mean_executed_precision=float(_number(row, "block_mean_executed_precision")),
            executed_pruning_histogram_json=_histogram(row["executed_pruning_histogram_json"], 4, length), executed_precision_histogram_json=_histogram(row["executed_precision_histogram_json"], 5, length),
        )
        if not record.scene or record.checkpoint_scope != "fixed_subset":
            raise ValueError("checkpoint scene or scope is invalid")
        records.append(record)
    records.sort(key=lambda item: (item.scene, item.episode_id, item.local_checkpoint_index))
    for key in sorted({(item.scene, item.episode_id) for item in records}):
        group = [item for item in records if (item.scene, item.episode_id) == key]
        if [item.local_checkpoint_index for item in group] != list(range(len(group))) or group[0].block_start_step_index != 0:
            raise ValueError("checkpoint indices must be continuous and start at block zero")
        if not group[-1].is_terminal or any(item.is_terminal for item in group[:-1]):
            raise ValueError("only the final checkpoint may be terminal")
        manifests = {item.fixed_view_manifest_path for item in group if item.fixed_view_manifest_path}
        if len(manifests) > 1:
            raise ValueError("fixed-view manifest changes within an episode")
        previous_observed: CheckpointDiagnosticRecord | None = None
        for index, item in enumerate(group):
            if item.block_end_step_index != item.local_step_index or item.block_length != item.block_end_step_index - item.block_start_step_index + 1:
                raise ValueError("checkpoint block bounds are inconsistent")
            if index and (item.block_start_step_index != group[index - 1].block_end_step_index + 1 or item.global_step <= group[index - 1].global_step):
                raise ValueError("checkpoint blocks overlap or global_step does not increase")
            if item.quality_observed:
                needed = (item.current_subset_quality_drop, item.previous_subset_quality_drop, item.subset_quality_drop_delta, item.quality_epsilon, item.subset_quality_feasible)
                if any(value is None for value in needed):
                    raise ValueError("observed checkpoint is missing subset quality fields")
                if not math.isclose(item.subset_quality_drop_delta, item.current_subset_quality_drop - item.previous_subset_quality_drop, abs_tol=1e-10):
                    raise ValueError("subset delta formula is inconsistent")
                expected_previous = 0.0 if previous_observed is None else previous_observed.current_subset_quality_drop
                if not math.isclose(item.previous_subset_quality_drop, expected_previous, abs_tol=1e-10):
                    raise ValueError("previous subset drop chain is inconsistent")
                previous_observed = item
            pair = (item.terminal_full_view_quality_drop is None, item.terminal_full_view_quality_feasible is None)
            if pair[0] != pair[1]:
                raise ValueError("terminal full-view drop and feasible must be paired")
            if item.is_terminal:
                if item.terminal_quality_scope != "full_view_terminal":
                    raise ValueError("terminal checkpoint scope is invalid")
            elif item.terminal_quality_scope is not None or not pair[0] or item.terminal_full_view_quality_violation is not None:
                raise ValueError("nonterminal checkpoint contains terminal quality")
    return tuple(records)


def load_factorized_actor_gradient_log(path: str | Path) -> tuple[ActorGradientDiagnosticRecord, ...]:
    """Read and strictly validate per-sample two-axis Actor gradient records."""
    records: list[ActorGradientDiagnosticRecord] = []
    for row in _rows(path, GRADIENT_COLUMNS, allow_header_only=True):
        record = ActorGradientDiagnosticRecord(
            episode=_integer(row, "episode", positive=True), sample_idx=_integer(row, "sample_idx"), selected_source=row["selected_source"],
            quality_feasible=bool(_boolean(row, "quality_feasible")), state_quality_drop=float(_number(row, "state_quality_drop")), state_quality_margin=float(_number(row, "state_quality_margin")),
            raw_pruning_action=float(_number(row, "raw_pruning_action")), raw_precision_action=float(_number(row, "raw_precision_action")),
            normalized_pruning_action=float(_number(row, "normalized_pruning_action")), normalized_precision_action=float(_number(row, "normalized_precision_action")),
            normalized_pruning_gradient=float(_number(row, "normalized_pruning_gradient")), normalized_precision_gradient=float(_number(row, "normalized_precision_gradient")),
            raw_pruning_gradient=float(_number(row, "raw_pruning_gradient")), raw_precision_gradient=float(_number(row, "raw_precision_gradient")),
            gradient_norm_before_clip=float(_number(row, "gradient_norm_before_clip")), gradient_norm_after_clip=float(_number(row, "gradient_norm_after_clip")),
        )
        if record.selected_source not in {"D", "P"} or not 0 <= record.raw_pruning_action <= 4 or not 0 <= record.raw_precision_action <= 5:
            raise ValueError("gradient source or raw action is invalid")
        if not 0 <= record.normalized_pruning_action <= 1 or not 0 <= record.normalized_precision_action <= 1:
            raise ValueError("normalized action is outside [0, 1]")
        if not math.isclose(record.normalized_pruning_action, record.raw_pruning_action / 4, abs_tol=1e-10) or not math.isclose(record.normalized_precision_action, record.raw_precision_action / 5, abs_tol=1e-10):
            raise ValueError("normalized action does not match raw action")
        if record.gradient_norm_after_clip > record.gradient_norm_before_clip + 1e-10:
            raise ValueError("gradient norm increased after clipping")
        records.append(record)
    records.sort(key=lambda item: (item.episode, item.sample_idx))
    for episode in sorted({item.episode for item in records}):
        group = [item for item in records if item.episode == episode]
        if [item.sample_idx for item in group] != list(range(len(group))) or len({item.selected_source for item in group}) != 1:
            raise ValueError("gradient sample indices or source are inconsistent")
    return tuple(records)


def trailing_moving_average(values: Sequence[float | None], window: int) -> tuple[float | None, ...]:
    """Return a causal trailing mean, resetting history at each missing value."""
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive strict integer")
    output: list[float | None] = []
    history: list[float] = []
    for value in values:
        if value is None or not math.isfinite(float(value)):
            history = []
            output.append(None)
            continue
        history.append(float(value))
        output.append(float(math.fsum(history[-window:]) / len(history[-window:])))
    return tuple(output)


def build_terminal_quality_points(episodes: Sequence[EpisodeDiagnosticRecord], checkpoints: Sequence[CheckpointDiagnosticRecord]) -> tuple[ParetoDiagnosticPoint, ...]:
    """Build terminal RD points while preserving full-view/fallback scope."""
    terminal = {(item.scene, item.episode_id): item for item in checkpoints if item.is_terminal}
    points = []
    for episode in sorted(episodes, key=lambda item: (item.scene, item.episode)):
        checkpoint = terminal.get((episode.scene, episode.episode))
        if checkpoint is None:
            raise ValueError("episode has no terminal checkpoint")
        if checkpoint.terminal_full_view_quality_drop is not None:
            scope = "full_view_terminal"; quality_drop = checkpoint.terminal_full_view_quality_drop
            feasible = bool(checkpoint.terminal_full_view_quality_feasible)
        else:
            scope = "fixed_subset_fallback"; quality_drop = episode.final_quality_drop
            feasible = episode.final_quality_feasible
        points.append(ParetoDiagnosticPoint(episode.scene, episode.episode, episode.global_step, scope, quality_drop, feasible, episode.final_compact_size_ratio, 1.0 / episode.final_compact_size_ratio, False, 0))
    return tuple(points)


def compute_pareto_frontier(points: Sequence[ParetoDiagnosticPoint]) -> tuple[ParetoDiagnosticPoint, ...]:
    """Mark minimization Pareto fronts independently by scene and scope."""
    output = []
    ordered = sorted(points, key=lambda item: (item.scene, item.quality_scope, item.episode))
    for point in ordered:
        peers = [item for item in ordered if item.scene == point.scene and item.quality_scope == point.quality_scope]
        dominated = sum(
            other.compact_size_ratio <= point.compact_size_ratio
            and other.quality_drop <= point.quality_drop
            and (other.compact_size_ratio < point.compact_size_ratio or other.quality_drop < point.quality_drop)
            for other in peers
        )
        output.append(replace(point, is_pareto=dominated == 0, dominated_by_count=int(dominated)))
    return tuple(output)


def detect_quality_cliffs(checkpoints: Sequence[CheckpointDiagnosticRecord], *, minimum_delta: float = 0.01, mad_multiplier: float = 3.5) -> tuple[QualityCliffEvent, ...]:
    """Detect fixed-subset quality cliffs with robust per-manifest thresholds."""
    if not math.isfinite(minimum_delta) or minimum_delta < 0 or not math.isfinite(mad_multiplier) or mad_multiplier < 0:
        raise ValueError("cliff thresholds must be finite and nonnegative")
    observed = [item for item in checkpoints if item.quality_observed]
    events = []
    for key in sorted({(item.scene, item.fixed_view_manifest_path) for item in observed}):
        group = [item for item in observed if (item.scene, item.fixed_view_manifest_path) == key]
        deltas = np.asarray([item.subset_quality_drop_delta for item in group], dtype=np.float64)
        center = float(np.median(deltas)); mad = float(np.median(np.abs(deltas - center)))
        threshold = max(float(minimum_delta), center + mad_multiplier * 1.4826 * mad)
        for item in group:
            delta = float(item.subset_quality_drop_delta)
            reasons = []
            if delta > threshold: reasons.append("robust_positive_jump")
            if item.previous_subset_quality_drop <= item.quality_epsilon and item.current_subset_quality_drop > item.quality_epsilon: reasons.append("quality_constraint_crossing")
            if item.subset_quality_violation is not None and item.subset_quality_violation > 0: reasons.append("positive_quality_violation")
            if not reasons: continue
            z_score = (delta - center) / (1.4826 * mad) if mad > 0 else (0.0 if delta == center else None)
            gain = item.block_size_ratio_reduction
            positive_delta = max(delta, 0.0)
            if gain < 0 or (gain == 0 and positive_delta > 0):
                cost = None
            elif gain == 0:
                cost = 0.0
            else:
                cost = positive_delta / max(gain, 1e-12)
            events.append(QualityCliffEvent(
                item.scene, item.episode_id, item.global_step, item.local_checkpoint_index,
                item.block_start_step_index, item.block_end_step_index, item.block_length,
                item.fixed_view_manifest_path, item.is_terminal, float(item.previous_subset_quality_drop),
                float(item.current_subset_quality_drop), delta, float(item.quality_epsilon),
                item.subset_quality_margin, item.subset_quality_violation, gain, cost,
                item.block_mean_executed_pruning, item.block_mean_executed_precision,
                item.executed_pruning_histogram_json, item.executed_precision_histogram_json,
                threshold, z_score, json.dumps(reasons, separators=(",", ":"), allow_nan=False),
            ))
    return tuple(sorted(events, key=lambda item: (item.scene, item.episode_id, item.local_checkpoint_index)))


def _validate_cross_logs(
    episodes: Sequence[EpisodeDiagnosticRecord],
    checkpoints: Sequence[CheckpointDiagnosticRecord],
    gradients: Sequence[ActorGradientDiagnosticRecord],
    *,
    gradient_log_supplied: bool = True,
) -> list[str]:
    warnings: list[str] = []
    episode_map = {(item.scene, item.episode): item for item in episodes}
    checkpoints_by_episode: dict[tuple[str, int], list[CheckpointDiagnosticRecord]] = {}
    for item in checkpoints:
        checkpoints_by_episode.setdefault((item.scene, item.episode_id), []).append(item)
    if set(checkpoints_by_episode) != set(episode_map):
        raise ValueError("episode and checkpoint log keys do not match")
    for key, episode in episode_map.items():
        group = checkpoints_by_episode[key]
        if len(group) != episode.quality_checkpoint_count or sum(item.quality_observed for item in group) != episode.quality_observed_block_count:
            raise ValueError("episode checkpoint counts do not match checkpoint log")
        terminal = group[-1]
        if terminal.compact_size_ratio is not None and not math.isclose(terminal.compact_size_ratio, episode.final_compact_size_ratio, abs_tol=1e-10):
            raise ValueError("terminal compact_size_ratio differs from episode log")
        if any(item.block_size_ratio_reduction < 0 for item in group):
            warnings.append(f"negative block size reduction: scene={key[0]} episode={key[1]}")
    if not gradients:
        if gradient_log_supplied:
            warnings.append("Actor gradient log contains only a header")
        return warnings
    episodes_by_number: dict[int, EpisodeDiagnosticRecord] = {}
    for episode in episodes:
        if episode.episode in episodes_by_number:
            raise ValueError("gradient join requires globally unique episode numbers")
        episodes_by_number[episode.episode] = episode
    gradient_groups: dict[int, list[ActorGradientDiagnosticRecord]] = {}
    for item in gradients:
        gradient_groups.setdefault(item.episode, []).append(item)
    for number, group in gradient_groups.items():
        episode = episodes_by_number.get(number)
        if episode is None:
            raise ValueError("gradient row references unknown episode")
        if episode.actor_update_source == "none_not_ready":
            raise ValueError("none_not_ready episode must not contain gradient rows")
        if len(group) != episode.steps or any(item.selected_source != episode.actor_update_source or item.quality_feasible != episode.final_quality_feasible for item in group):
            raise ValueError("gradient rows do not match episode Actor update")
    for episode in episodes:
        if episode.actor_update_source in {"D", "P"} and episode.episode not in gradient_groups:
            raise ValueError("Actor-updated episode is missing gradient rows")
    return warnings


def _safe_value(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _safe_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_value(item) for item in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result): raise ValueError("diagnostic output cannot contain NaN or Infinity")
        return result
    return value


def _atomic_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(descriptor); temporary = Path(name)
    try:
        temporary.write_text(json.dumps(_safe_value(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()
    return path


def _csv_cell(value: Any) -> Any:
    if value is None: return ""
    if isinstance(value, bool): return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value): raise ValueError("CSV cannot contain non-finite values")
        return repr(float(value))
    return value


def _atomic_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(descriptor); temporary = Path(name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
            for row in rows: writer.writerow({name: _csv_cell(row.get(name)) for name in columns})
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()
    return path


def _scene_slugs(scenes: Sequence[str]) -> dict[str, str]:
    candidates = {scene: re.sub(r"[^A-Za-z0-9_-]", "_", scene) or "scene" for scene in scenes}
    counts: dict[str, int] = {}
    for slug in candidates.values(): counts[slug] = counts.get(slug, 0) + 1
    return {scene: slug if counts[slug] == 1 else f"{slug}_{sha256(scene.encode()).hexdigest()[:8]}" for scene, slug in candidates.items()}


def _figure(path: Path, draw: Callable[[Any], None]) -> Path:
    figure = Figure(figsize=(7.0, 4.5)); FigureCanvasAgg(figure); axes = figure.add_axes((0.12, 0.14, 0.83, 0.80))
    draw(axes); path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight"); figure.clear()
    return path


def _plot_scene(
    scene: str,
    episodes: Sequence[EpisodeDiagnosticRecord],
    checkpoints: Sequence[CheckpointDiagnosticRecord],
    gradients: Sequence[ActorGradientDiagnosticRecord],
    points: Sequence[ParetoDiagnosticPoint],
    cliffs: Sequence[QualityCliffEvent],
    directory: Path,
    latest_count: int,
    moving_window: int,
    warnings: list[str],
) -> list[Path]:
    generated: list[Path] = []
    x = [item.episode for item in episodes]
    def lines(name, series, ylabel, *, symlog=False):
        def draw(ax):
            for label, values in series:
                ax.plot(x, values, marker="o", label=label)
                averaged = trailing_moving_average([float(value) if value is not None else None for value in values], moving_window)
                ax.plot(x, [np.nan if value is None else value for value in averaged], linestyle="--", label=f"{label} MA")
            if symlog: ax.set_yscale("symlog", linthresh=1e-8)
            ax.set_xlabel("Episode"); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3); ax.legend()
        generated.append(_figure(directory / name, draw))
    scopes = ("full_view_terminal", "fixed_subset_fallback")
    terminal_map = {(item.episode, item.quality_scope): item for item in points}
    def quality_draw(ax):
        for scope in scopes:
            values = [terminal_map[(episode, scope)].quality_drop if (episode, scope) in terminal_map else np.nan for episode in x]
            if np.any(np.isfinite(values)): ax.plot(x, values, marker="o", label=scope)
        epsilons = [item.quality_epsilon for item in checkpoints if item.quality_epsilon is not None]
        if epsilons: ax.axhline(epsilons[-1], linestyle=":", label="quality epsilon")
        ax.set_xlabel("Episode"); ax.set_ylabel("Quality drop"); ax.grid(True, alpha=0.3); ax.legend()
    generated.append(_figure(directory / "terminal_quality_drop_vs_episode.png", quality_draw))
    lines("compact_size_ratio_vs_episode.png", [("size ratio", [item.final_compact_size_ratio for item in episodes])], "Compact size ratio")
    lines("compression_factor_vs_episode.png", [("compression factor", [1/item.final_compact_size_ratio for item in episodes])], "Compression factor")
    lines("critic_losses_vs_episode.png", [("Size Critic", [item.size_critic_loss for item in episodes]), ("Quality Critic", [item.quality_critic_loss for item in episodes])], "Critic loss", symlog=True)
    lines("actor_actions_vs_episode.png", [("pruning", [item.mean_actor_pruning for item in episodes]), ("precision", [item.mean_actor_precision for item in episodes])], "Actor action level")
    lines("environment_actions_vs_episode.png", [("pruning", [item.mean_environment_pruning for item in episodes]), ("precision", [item.mean_environment_precision for item in episodes])], "Environment action level")
    gradient_mean = [item.mean_actor_gradient_norm if item.actor_update_source != "none_not_ready" else None for item in episodes]
    gradient_max = [item.max_actor_gradient_norm if item.actor_update_source != "none_not_ready" else None for item in episodes]
    if gradients:
        lines("actor_gradient_norm_vs_episode.png", [("mean", gradient_mean), ("max", gradient_max)], "Actor gradient norm")
    latest = sorted({item.episode_id for item in checkpoints})[-latest_count:]
    def progress_draw(ax):
        for episode in latest:
            group = [item for item in checkpoints if item.episode_id == episode and item.quality_observed]
            ax.plot([item.groups_processed/item.total_group_count for item in group], [item.current_subset_quality_drop for item in group], marker="o", label=f"ep {episode}")
        ax.set_xlabel("Groups processed / total"); ax.set_ylabel("Fixed-subset drop"); ax.grid(True, alpha=0.3); ax.legend()
    generated.append(_figure(directory / "checkpoint_subset_drop_vs_progress.png", progress_draw))
    cliff_keys = {(item.episode_id, item.local_checkpoint_index) for item in cliffs}
    def delta_draw(ax):
        observed = [item for item in checkpoints if item.quality_observed]
        ax.plot([item.global_step for item in observed], [item.subset_quality_drop_delta for item in observed], marker="o")
        marked = [item for item in observed if (item.episode_id, item.local_checkpoint_index) in cliff_keys]
        if marked: ax.scatter([item.global_step for item in marked], [item.subset_quality_drop_delta for item in marked], marker="x", s=60, label="cliff")
        ax.set_xlabel("Global step"); ax.set_ylabel("Fixed-subset drop delta"); ax.grid(True, alpha=0.3)
        if marked: ax.legend()
    generated.append(_figure(directory / "checkpoint_subset_delta_vs_global_step.png", delta_draw))
    for scope, filename in (("full_view_terminal", "rd_pareto_full_view.png"), ("fixed_subset_fallback", "rd_pareto_fixed_subset_fallback.png")):
        scoped = [item for item in points if item.quality_scope == scope]
        if not scoped:
            warnings.append(f"no {scope} RD data for scene {scene}"); continue
        def rd_draw(ax, scoped=scoped):
            ax.scatter([item.compact_size_ratio for item in scoped], [item.quality_drop for item in scoped], marker="o", label="all")
            frontier = [item for item in scoped if item.is_pareto]
            ax.scatter([item.compact_size_ratio for item in frontier], [item.quality_drop for item in frontier], marker="x", s=70, label="Pareto")
            ax.set_xlabel("Compact size ratio"); ax.set_ylabel("Quality drop"); ax.grid(True, alpha=0.3); ax.legend()
        generated.append(_figure(directory / filename, rd_draw))
    return generated


EPISODE_OUTPUT_COLUMNS = (
    "scene", "episode", "global_step", "final_quality_scope",
    "final_quality_drop", "final_quality_feasible", "final_compact_size_ratio",
    "compression_factor", "selected_actor_critic", "actor_update_source",
    "size_critic_loss", "quality_critic_loss", "mean_actor_pruning",
    "mean_actor_precision", "mean_environment_pruning",
    "mean_environment_precision", "quality_checkpoint_count",
    "quality_observed_block_count", "cliff_count",
)
CLIFF_OUTPUT_COLUMNS = tuple(QualityCliffEvent.__dataclass_fields__)
PARETO_OUTPUT_COLUMNS = tuple(ParetoDiagnosticPoint.__dataclass_fields__)


def _mean(values: Iterable[float]) -> float:
    """Return a deterministic finite arithmetic mean for a nonempty sequence."""
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot summarize an empty numeric sequence")
    return float(math.fsum(materialized) / len(materialized))


def _build_summaries(
    episodes: Sequence[EpisodeDiagnosticRecord],
    checkpoints: Sequence[CheckpointDiagnosticRecord],
    gradients: Sequence[ActorGradientDiagnosticRecord],
    points: Sequence[ParetoDiagnosticPoint],
    cliffs: Sequence[QualityCliffEvent],
) -> dict[str, Any]:
    """Build deterministic global and per-scene diagnostic statistics."""
    point_map = {(item.scene, item.episode): item for item in points}
    scene_summaries: dict[str, Any] = {}
    for scene in sorted({item.scene for item in episodes}):
        scene_episodes = sorted(
            (item for item in episodes if item.scene == scene),
            key=lambda item: item.episode,
        )
        scene_points = [point_map[(scene, item.episode)] for item in scene_episodes]
        latest = scene_points[-1]
        best_ratio = min(item.compact_size_ratio for item in scene_points)
        feasible_count = sum(item.quality_feasible for item in scene_points)
        scene_summaries[scene] = {
            "episode_count": len(scene_episodes),
            "feasible_episode_count": feasible_count,
            "feasible_rate": feasible_count / len(scene_episodes),
            "best_compact_size_ratio": best_ratio,
            "best_compression_factor": 1.0 / best_ratio,
            "minimum_terminal_quality_drop": min(item.quality_drop for item in scene_points),
            "latest_terminal_quality_drop": latest.quality_drop,
            "latest_compact_size_ratio": latest.compact_size_ratio,
            "mean_size_critic_loss": _mean(item.size_critic_loss for item in scene_episodes),
            "mean_quality_critic_loss": _mean(item.quality_critic_loss for item in scene_episodes),
            "mean_actor_pruning": _mean(item.mean_actor_pruning for item in scene_episodes),
            "mean_actor_precision": _mean(item.mean_actor_precision for item in scene_episodes),
            "full_view_terminal_count": sum(item.quality_scope == "full_view_terminal" for item in scene_points),
            "fixed_subset_fallback_count": sum(item.quality_scope == "fixed_subset_fallback" for item in scene_points),
            "cliff_event_count": sum(item.scene == scene for item in cliffs),
        }
    actor_counts = {
        "D_update_episode_count": sum(item.actor_update_source == "D" for item in episodes),
        "P_update_episode_count": sum(item.actor_update_source == "P" for item in episodes),
        "none_not_ready_count": sum(item.actor_update_source == "none_not_ready" for item in episodes),
    }
    reason_counts: dict[str, int] = {}
    for event in cliffs:
        for reason in json.loads(event.detection_reasons):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "episode_count": len(episodes),
        "checkpoint_count": len(checkpoints),
        "observed_checkpoint_count": sum(item.quality_observed for item in checkpoints),
        "gradient_row_count": len(gradients),
        "cliff_event_count": len(cliffs),
        "pareto_point_count": len(points),
        "pareto_frontier_count": sum(item.is_pareto for item in points),
        "scenes": scene_summaries,
        "actor_updates": actor_counts,
        "cliff_reason_counts": {key: reason_counts[key] for key in sorted(reason_counts)},
    }


def generate_training_diagnostics(
    training_log_path: str | Path,
    checkpoint_log_path: str | Path,
    gradient_log_path: str | Path | None,
    output_dir: str | Path,
    *,
    moving_average_window: int = 10,
    latest_checkpoint_episodes: int = 10,
    minimum_cliff_delta: float = 0.01,
    cliff_mad_multiplier: float = 3.5,
) -> TrainingDiagnosticsResult:
    """Generate strict, deterministic, read-only diagnostics from three CSV logs.

    The function never modifies source logs or training state. RD/Pareto analysis
    is partitioned by both scene and terminal quality scope. Fixed-subset values
    are used only for checkpoint trajectories and cliff detection.
    """
    if isinstance(moving_average_window, bool) or not isinstance(moving_average_window, int) or moving_average_window <= 0:
        raise ValueError("moving_average_window must be a positive strict integer")
    if isinstance(latest_checkpoint_episodes, bool) or not isinstance(latest_checkpoint_episodes, int) or latest_checkpoint_episodes <= 0:
        raise ValueError("latest_checkpoint_episodes must be a positive strict integer")
    if not math.isfinite(float(minimum_cliff_delta)) or float(minimum_cliff_delta) < 0:
        raise ValueError("minimum_cliff_delta must be finite and nonnegative")
    if not math.isfinite(float(cliff_mad_multiplier)) or float(cliff_mad_multiplier) < 0:
        raise ValueError("cliff_mad_multiplier must be finite and nonnegative")

    training_path = Path(training_log_path).resolve()
    checkpoint_path = Path(checkpoint_log_path).resolve()
    gradient_path = Path(gradient_log_path).resolve() if gradient_log_path is not None else None
    destination = Path(output_dir).resolve()
    episodes = load_factorized_training_log(training_path)
    checkpoints = load_factorized_quality_checkpoint_log(checkpoint_path)
    warnings: list[str] = []
    if gradient_path is None:
        gradients: tuple[ActorGradientDiagnosticRecord, ...] = ()
        warnings.append("Actor gradient log was not supplied")
    else:
        gradients = load_factorized_actor_gradient_log(gradient_path)
    warnings.extend(_validate_cross_logs(episodes, checkpoints, gradients, gradient_log_supplied=gradient_path is not None))
    points = compute_pareto_frontier(build_terminal_quality_points(episodes, checkpoints))
    cliffs = detect_quality_cliffs(
        checkpoints,
        minimum_delta=float(minimum_cliff_delta),
        mad_multiplier=float(cliff_mad_multiplier),
    )
    summaries = _build_summaries(episodes, checkpoints, gradients, points, cliffs)

    cliff_counts: dict[tuple[str, int], int] = {}
    for event in cliffs:
        key = (event.scene, event.episode_id)
        cliff_counts[key] = cliff_counts.get(key, 0) + 1
    point_map = {(item.scene, item.episode): item for item in points}
    episode_rows: list[dict[str, Any]] = []
    for episode in episodes:
        point = point_map[(episode.scene, episode.episode)]
        episode_rows.append({
            "scene": episode.scene,
            "episode": episode.episode,
            "global_step": episode.global_step,
            "final_quality_scope": point.quality_scope,
            "final_quality_drop": point.quality_drop,
            "final_quality_feasible": point.quality_feasible,
            "final_compact_size_ratio": point.compact_size_ratio,
            "compression_factor": point.compression_factor,
            "selected_actor_critic": episode.selected_actor_critic,
            "actor_update_source": episode.actor_update_source,
            "size_critic_loss": episode.size_critic_loss,
            "quality_critic_loss": episode.quality_critic_loss,
            "mean_actor_pruning": episode.mean_actor_pruning,
            "mean_actor_precision": episode.mean_actor_precision,
            "mean_environment_pruning": episode.mean_environment_pruning,
            "mean_environment_precision": episode.mean_environment_precision,
            "quality_checkpoint_count": episode.quality_checkpoint_count,
            "quality_observed_block_count": episode.quality_observed_block_count,
            "cliff_count": cliff_counts.get((episode.scene, episode.episode), 0),
        })

    destination.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    generated.append(_atomic_csv(destination / "episode_diagnostics.csv", episode_rows, EPISODE_OUTPUT_COLUMNS))
    generated.append(_atomic_csv(destination / "quality_cliff_events.csv", [asdict(item) for item in cliffs], CLIFF_OUTPUT_COLUMNS))
    generated.append(_atomic_csv(destination / "rd_pareto_points.csv", [asdict(item) for item in points], PARETO_OUTPUT_COLUMNS))
    generated.append(_atomic_csv(destination / "rd_pareto_frontier.csv", [asdict(item) for item in points if item.is_pareto], PARETO_OUTPUT_COLUMNS))

    scenes = tuple(sorted({item.scene for item in episodes}))
    slugs = _scene_slugs(scenes)
    for scene in scenes:
        scene_episodes = [item for item in episodes if item.scene == scene]
        scene_checkpoints = [item for item in checkpoints if item.scene == scene]
        scene_episode_numbers = {item.episode for item in scene_episodes}
        scene_gradients = [item for item in gradients if item.episode in scene_episode_numbers]
        scene_points = [item for item in points if item.scene == scene]
        scene_cliffs = [item for item in cliffs if item.scene == scene]
        generated.extend(_plot_scene(
            scene,
            scene_episodes,
            scene_checkpoints,
            scene_gradients,
            scene_points,
            scene_cliffs,
            destination / "figures" / slugs[scene],
            latest_checkpoint_episodes,
            moving_average_window,
            warnings,
        ))

    source_paths = {
        "training_log": str(training_path),
        "checkpoint_log": str(checkpoint_path),
        "gradient_log": None if gradient_path is None else str(gradient_path),
    }
    summary_path = destination / "diagnostics_summary.json"
    generated_with_summary = tuple(str(path.resolve()) for path in [*generated, summary_path])
    summary_payload = {
        "schema": SCHEMA,
        "source_paths": source_paths,
        **summaries,
        "warnings": sorted(set(warnings)),
        "generated_files": list(generated_with_summary),
    }
    _atomic_json(summary_path, summary_payload)
    return TrainingDiagnosticsResult(
        schema=SCHEMA,
        source_paths=source_paths,
        episode_count=len(episodes),
        checkpoint_count=len(checkpoints),
        gradient_row_count=len(gradients),
        scenes=scenes,
        cliff_events=cliffs,
        pareto_points=points,
        summaries=summaries,
        warnings=tuple(sorted(set(warnings))),
        generated_files=generated_with_summary,
    )


def _write_validation_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    """Write a synthetic validation CSV using the training writer conventions."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_cell(row.get(name, "")) for name in columns})


def _validation_fixture() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build deterministic two-scene logs for the public self-check."""
    ratios = {1: 0.70, 2: 0.60, 3: 0.80, 4: 0.72, 5: 0.70, 6: 0.60, 7: 0.80, 8: 0.65}
    terminal_subset = {1: 0.04, 2: 0.22, 3: 0.01, 4: 0.12, 5: 0.04, 6: 0.05, 7: 0.12, 8: 0.06}
    full_view = {2: (0.08, True), 4: (0.12, False), 6: (0.07, True), 8: (0.09, True)}
    actor_sources = {1: "P", 2: "P", 3: "none_not_ready", 4: "D", 5: "P", 6: "none_not_ready", 7: "D", 8: "P"}
    episode_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    pruning_histograms = (
        json.dumps({str(i): (8 if i == 1 else 0) for i in range(5)}, separators=(",", ":")),
        json.dumps({str(i): (2 if i == 2 else 0) for i in range(5)}, separators=(",", ":")),
    )
    precision_histograms = (
        json.dumps({str(i): (8 if i == 2 else 0) for i in range(6)}, separators=(",", ":")),
        json.dumps({str(i): (2 if i == 3 else 0) for i in range(6)}, separators=(",", ":")),
    )
    for episode in range(1, 9):
        scene = "scene/A" if episode <= 4 else "scene:A"
        subset_drop = terminal_subset[episode]
        terminal_quality, feasible = full_view.get(episode, (subset_drop, subset_drop <= 0.1))
        source = actor_sources[episode]
        selected = "P" if feasible else "D"
        episode_rows.append({
            "episode": episode,
            "global_step": episode * 10,
            "scene": scene,
            "learner_mode": "factorized_dual_critic_ddpg",
            "steps": 10,
            "final_quality_drop": terminal_quality,
            "final_quality_feasible": feasible,
            "selected_actor_critic": selected,
            "actor_update_source": source,
            "actor_update_count": 0 if source == "none_not_ready" else 1,
            "mean_actor_gradient_norm": 0.0 if source == "none_not_ready" else 0.25,
            "max_actor_gradient_norm": 0.0 if source == "none_not_ready" else 0.50,
            "size_critic_loss": 0.01 * episode,
            "quality_critic_loss": 0.02 * episode,
            "size_critic_update_count": 1,
            "quality_critic_update_count": 1,
            "final_compact_size_bytes": int(1_000_000 * ratios[episode]),
            "final_compact_size_ratio": ratios[episode],
            "mean_actor_pruning": 1.0 + episode * 0.05,
            "mean_actor_precision": 2.0 + episode * 0.05,
            "mean_environment_pruning": 1.1 + episode * 0.05,
            "mean_environment_precision": 2.1 + episode * 0.05,
            "quality_checkpoint_count": 2,
            "quality_observed_block_count": 2,
        })
        for checkpoint_index in range(2):
            is_terminal = checkpoint_index == 1
            current = 0.02 if checkpoint_index == 0 else subset_drop
            previous = 0.0 if checkpoint_index == 0 else 0.02
            delta = current - previous
            violation = max(current - 0.1, 0.0)
            before = 1.0 if checkpoint_index == 0 else 0.85
            after = 0.85 if checkpoint_index == 0 else ratios[episode]
            checkpoint_rows.append({
                "global_step": episode * 10 - (2 if checkpoint_index == 0 else 0),
                "episode_id": episode,
                "scene": scene,
                "local_checkpoint_index": checkpoint_index,
                "local_step_index": 7 if checkpoint_index == 0 else 9,
                "groups_processed": 8 if checkpoint_index == 0 else 10,
                "total_group_count": 10,
                "block_start_step_index": 0 if checkpoint_index == 0 else 8,
                "block_end_step_index": 7 if checkpoint_index == 0 else 9,
                "block_length": 8 if checkpoint_index == 0 else 2,
                "is_terminal": is_terminal,
                "checkpoint_scope": "fixed_subset",
                "quality_observed": True,
                "fixed_view_manifest_path": f"manifests/{scene.replace('/', '_')}.json",
                "previous_subset_quality_drop": previous,
                "current_subset_quality_drop": current,
                "subset_quality_drop_delta": delta,
                "quality_epsilon": 0.1,
                "subset_quality_margin": 0.1 - current,
                "subset_quality_feasible": current <= 0.1,
                "subset_quality_violation": violation,
                "terminal_full_view_quality_drop": full_view[episode][0] if is_terminal and episode in full_view else "",
                "terminal_full_view_quality_feasible": full_view[episode][1] if is_terminal and episode in full_view else "",
                "terminal_full_view_quality_violation": max(full_view[episode][0] - 0.1, 0.0) if is_terminal and episode in full_view else "",
                "terminal_quality_scope": "full_view_terminal" if is_terminal else "",
                "estimated_size_ratio_before_block": before,
                "estimated_size_ratio_after_block": after,
                "block_size_ratio_reduction": before - after,
                "block_log_size_reward_sum": 0.1 + checkpoint_index,
                "compact_size_ratio": ratios[episode] if is_terminal else "",
                "terminal_size_correction": 0.0 if is_terminal else "",
                "block_mean_executed_pruning": 1.0 if checkpoint_index == 0 else 2.0,
                "block_mean_executed_precision": 2.0 if checkpoint_index == 0 else 3.0,
                "executed_pruning_histogram_json": pruning_histograms[checkpoint_index],
                "executed_precision_histogram_json": precision_histograms[checkpoint_index],
            })
        if source in {"D", "P"}:
            for sample_idx in range(10):
                gradient_rows.append({
                    "episode": episode,
                    "sample_idx": sample_idx,
                    "selected_source": source,
                    "quality_feasible": feasible,
                    "state_quality_drop": terminal_quality,
                    "state_quality_margin": 0.1 - terminal_quality,
                    "raw_pruning_action": 2.0,
                    "raw_precision_action": 2.5,
                    "normalized_pruning_action": 0.5,
                    "normalized_precision_action": 0.5,
                    "normalized_pruning_gradient": 0.1,
                    "normalized_precision_gradient": -0.2,
                    "raw_pruning_gradient": 0.025,
                    "raw_precision_gradient": -0.04,
                    "gradient_norm_before_clip": 0.3,
                    "gradient_norm_after_clip": 0.2,
                })
    return episode_rows, checkpoint_rows, gradient_rows


def _require(condition: bool, message: str) -> None:
    """Raise a validation-specific assertion with a useful message."""
    if not condition:
        raise AssertionError(message)


def _expect_value_error(call: Callable[[], Any], message: str) -> None:
    """Require a callable to reject invalid input with ValueError."""
    try:
        call()
    except ValueError:
        return
    raise AssertionError(message)


def validate_training_diagnostics() -> bool:
    """Exercise strict parsing, scope isolation, cliff/Pareto logic and outputs.

    The self-check creates two colliding scene slugs, eight episodes, 8+2
    checkpoint blocks, full-view and fallback endpoints, all Actor update states,
    a quality crossing, a robust jump, a negative delta, and dominated RD points.
    It performs no repository writes and removes all temporary artifacts.
    """
    episode_rows, checkpoint_rows, gradient_rows = _validation_fixture()
    episode_columns = tuple(sorted(EPISODE_COLUMNS))
    checkpoint_columns = tuple(sorted(CHECKPOINT_COLUMNS))
    gradient_columns = tuple(sorted(GRADIENT_COLUMNS))
    with tempfile.TemporaryDirectory(prefix="training_diagnostics_validation_") as temporary_name:
        root = Path(temporary_name)
        training_path = root / "factorized_training_log.csv"
        checkpoint_path = root / "factorized_quality_checkpoint_log.csv"
        gradient_path = root / "factorized_actor_gradient_log.csv"
        _write_validation_csv(training_path, episode_columns, episode_rows)
        _write_validation_csv(checkpoint_path, checkpoint_columns, checkpoint_rows)
        _write_validation_csv(gradient_path, gradient_columns, gradient_rows)

        episodes = load_factorized_training_log(training_path)
        checkpoints = load_factorized_quality_checkpoint_log(checkpoint_path)
        gradients = load_factorized_actor_gradient_log(gradient_path)
        _require(len(episodes) == 8, "synthetic episode count is wrong")
        _require(len(checkpoints) == 16, "synthetic checkpoint count is wrong")
        _require(len(gradients) == 60, "synthetic gradient count is wrong")
        _require({item.actor_update_source for item in episodes} == {"D", "P", "none_not_ready"}, "Actor states are incomplete")

        points = compute_pareto_frontier(build_terminal_quality_points(episodes, checkpoints))
        _require({item.quality_scope for item in points} == {"full_view_terminal", "fixed_subset_fallback"}, "terminal scopes are incomplete")
        _require(all(item.dominated_by_count >= 0 for item in points), "Pareto dominance count is invalid")
        _require(any(item.dominated_by_count > 0 for item in points), "synthetic Pareto dominance was not detected")
        duplicate = replace(points[0], episode=99)
        duplicate_result = compute_pareto_frontier((*points, duplicate))
        duplicate_pair = [item for item in duplicate_result if item.scene == points[0].scene and item.quality_scope == points[0].quality_scope and item.compact_size_ratio == points[0].compact_size_ratio and item.quality_drop == points[0].quality_drop]
        _require(len({item.is_pareto for item in duplicate_pair}) == 1, "duplicate RD points are nondeterministic")
        isolated = compute_pareto_frontier((
            ParetoDiagnosticPoint("isolated", 1, 1, "full_view_terminal", 0.01, True, 0.1, 10.0, False, 0),
            ParetoDiagnosticPoint("isolated", 2, 2, "fixed_subset_fallback", 0.10, False, 0.9, 1.0 / 0.9, False, 0),
        ))
        _require(all(item.is_pareto for item in isolated), "quality scopes were mixed during Pareto analysis")
        _require(trailing_moving_average((1.0, 3.0, None, 9.0), 2) == (1.0, 2.0, None, 9.0), "moving average crossed a missing value")

        cliffs = detect_quality_cliffs(checkpoints, minimum_delta=0.01, mad_multiplier=3.5)
        episode_two = [item for item in cliffs if item.episode_id == 2 and item.local_checkpoint_index == 1]
        _require(len(episode_two) == 1, "quality cliff was not detected")
        reasons = set(json.loads(episode_two[0].detection_reasons))
        _require({"robust_positive_jump", "quality_constraint_crossing", "positive_quality_violation"} <= reasons, "cliff reasons are incomplete")
        _require(math.isclose(episode_two[0].quality_cost_per_size_gain, 0.20 / (0.85 - 0.60), abs_tol=1e-12), "quality cost is wrong")
        _require(not any(item.episode_id == 3 and item.local_checkpoint_index == 1 for item in cliffs), "negative delta was incorrectly classified as a cliff")

        output = root / "output"
        first = generate_training_diagnostics(training_path, checkpoint_path, gradient_path, output, moving_average_window=2, latest_checkpoint_episodes=2)
        _require(first.episode_count == 8 and first.checkpoint_count == 16 and first.gradient_row_count == 60, "result counts are wrong")
        required_names = {"episode_diagnostics.csv", "quality_cliff_events.csv", "rd_pareto_points.csv", "rd_pareto_frontier.csv", "diagnostics_summary.json"}
        _require(required_names <= {Path(path).name for path in first.generated_files}, "derived reports are incomplete")
        pngs = [Path(path) for path in first.generated_files if path.endswith(".png")]
        _require(pngs and all(path.is_file() and path.stat().st_size > 0 for path in pngs), "a PNG is missing or empty")
        _require(len({path.parent.name for path in pngs}) == 2, "scene slug collision was not resolved")
        with (output / "diagnostics_summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        _require(summary["schema"] == SCHEMA and summary["pareto_point_count"] == 8, "summary JSON is invalid")

        deterministic_paths = [output / name for name in sorted(required_names)]
        before = {path.name: path.read_bytes() for path in deterministic_paths}
        second = generate_training_diagnostics(training_path, checkpoint_path, gradient_path, output, moving_average_window=2, latest_checkpoint_episodes=2)
        after = {path.name: path.read_bytes() for path in deterministic_paths}
        _require(before == after and first.generated_files == second.generated_files, "repeated output is not deterministic")

        _write_validation_csv(training_path, episode_columns, list(reversed(episode_rows)))
        _write_validation_csv(checkpoint_path, checkpoint_columns, list(reversed(checkpoint_rows)))
        _write_validation_csv(gradient_path, gradient_columns, list(reversed(gradient_rows)))
        third = generate_training_diagnostics(training_path, checkpoint_path, gradient_path, output, moving_average_window=2, latest_checkpoint_episodes=2)
        shuffled = {path.name: path.read_bytes() for path in deterministic_paths}
        _require(after == shuffled and second.generated_files == third.generated_files, "CSV row order changed diagnostics")

        axes_probe = root / "single_axes.png"
        _figure(axes_probe, lambda axes: _require(len(axes.figure.axes) == 1, "figure has more than one Axes"))
        _require(axes_probe.stat().st_size > 0, "single-Axes probe was not written")

        bad_training = root / "bad_training.csv"
        invalid_episode = [dict(item) for item in episode_rows]
        invalid_episode[0]["final_quality_feasible"] = "True"
        _write_validation_csv(bad_training, episode_columns, invalid_episode)
        _expect_value_error(lambda: load_factorized_training_log(bad_training), "invalid bool was accepted")
        invalid_episode = [dict(item) for item in episode_rows]
        invalid_episode[0]["final_compact_size_ratio"] = "NaN"
        _write_validation_csv(bad_training, episode_columns, invalid_episode)
        _expect_value_error(lambda: load_factorized_training_log(bad_training), "NaN was accepted")
        invalid_episode = [dict(item) for item in episode_rows]
        invalid_episode[0]["size_critic_loss"] = "Infinity"
        _write_validation_csv(bad_training, episode_columns, invalid_episode)
        _expect_value_error(lambda: load_factorized_training_log(bad_training), "Infinity was accepted")
        invalid_episode = [dict(item) for item in episode_rows]
        invalid_episode[0]["final_compact_size_ratio"] = 0
        _write_validation_csv(bad_training, episode_columns, invalid_episode)
        _expect_value_error(lambda: load_factorized_training_log(bad_training), "zero compact ratio was accepted")
        invalid_episode = [dict(item) for item in episode_rows]
        invalid_episode[1]["global_step"] = 5
        _write_validation_csv(bad_training, episode_columns, invalid_episode)
        _expect_value_error(lambda: load_factorized_training_log(bad_training), "regressing global_step was accepted")
        invalid_episode = [dict(item) for item in episode_rows]
        invalid_episode[0]["selected_actor_critic"] = "D"
        _write_validation_csv(bad_training, episode_columns, invalid_episode)
        _expect_value_error(lambda: load_factorized_training_log(bad_training), "inconsistent Critic selection was accepted")
        invalid_episode = [dict(item) for item in episode_rows]
        invalid_episode.append(dict(invalid_episode[0]))
        _write_validation_csv(bad_training, episode_columns, invalid_episode)
        _expect_value_error(lambda: load_factorized_training_log(bad_training), "duplicate episode was accepted")
        _write_validation_csv(bad_training, tuple(name for name in episode_columns if name != "scene"), episode_rows)
        _expect_value_error(lambda: load_factorized_training_log(bad_training), "missing column was accepted")
        bad_training.write_text("episode,episode\n1,1\n", encoding="utf-8")
        _expect_value_error(lambda: load_factorized_training_log(bad_training), "duplicate header was accepted")

        bad_checkpoint = root / "bad_checkpoint.csv"
        invalid_checkpoint = [dict(item) for item in checkpoint_rows]
        invalid_checkpoint[1]["subset_quality_drop_delta"] = 99
        _write_validation_csv(bad_checkpoint, checkpoint_columns, invalid_checkpoint)
        _expect_value_error(lambda: load_factorized_quality_checkpoint_log(bad_checkpoint), "bad subset delta was accepted")
        invalid_checkpoint = [dict(item) for item in checkpoint_rows]
        invalid_checkpoint[3]["terminal_full_view_quality_feasible"] = ""
        _write_validation_csv(bad_checkpoint, checkpoint_columns, invalid_checkpoint)
        _expect_value_error(lambda: load_factorized_quality_checkpoint_log(bad_checkpoint), "unpaired full-view fields were accepted")
        invalid_checkpoint = [dict(item) for item in checkpoint_rows]
        invalid_checkpoint[0]["executed_pruning_histogram_json"] = '{"0":8}'
        _write_validation_csv(bad_checkpoint, checkpoint_columns, invalid_checkpoint)
        _expect_value_error(lambda: load_factorized_quality_checkpoint_log(bad_checkpoint), "invalid histogram was accepted")
        invalid_checkpoint = [dict(item) for item in checkpoint_rows]
        invalid_checkpoint[0]["executed_pruning_histogram_json"] = json.dumps({str(i): 0 for i in range(5)})
        _write_validation_csv(bad_checkpoint, checkpoint_columns, invalid_checkpoint)
        _expect_value_error(lambda: load_factorized_quality_checkpoint_log(bad_checkpoint), "histogram count mismatch was accepted")
        invalid_checkpoint = [dict(item) for item in checkpoint_rows]
        invalid_checkpoint[1]["local_checkpoint_index"] = 2
        _write_validation_csv(bad_checkpoint, checkpoint_columns, invalid_checkpoint)
        _expect_value_error(lambda: load_factorized_quality_checkpoint_log(bad_checkpoint), "discontinuous checkpoint index was accepted")
        invalid_checkpoint = [dict(item) for item in checkpoint_rows]
        invalid_checkpoint[0]["is_terminal"] = True
        invalid_checkpoint[0]["terminal_quality_scope"] = "full_view_terminal"
        _write_validation_csv(bad_checkpoint, checkpoint_columns, invalid_checkpoint)
        _expect_value_error(lambda: load_factorized_quality_checkpoint_log(bad_checkpoint), "non-final terminal checkpoint was accepted")
        invalid_checkpoint = [dict(item) for item in checkpoint_rows]
        invalid_checkpoint[1]["block_start_step_index"] = 7
        _write_validation_csv(bad_checkpoint, checkpoint_columns, invalid_checkpoint)
        _expect_value_error(lambda: load_factorized_quality_checkpoint_log(bad_checkpoint), "overlapping blocks were accepted")
        invalid_checkpoint = [dict(item) for item in checkpoint_rows]
        invalid_checkpoint[1]["fixed_view_manifest_path"] = "changed.json"
        _write_validation_csv(bad_checkpoint, checkpoint_columns, invalid_checkpoint)
        _expect_value_error(lambda: load_factorized_quality_checkpoint_log(bad_checkpoint), "manifest change was accepted")

        bad_gradient = root / "bad_gradient.csv"
        invalid_gradient = [dict(item) for item in gradient_rows]
        invalid_gradient[0]["normalized_pruning_action"] = 0.25
        _write_validation_csv(bad_gradient, gradient_columns, invalid_gradient)
        _expect_value_error(lambda: load_factorized_actor_gradient_log(bad_gradient), "bad action normalization was accepted")
        invalid_gradient = [dict(item) for item in gradient_rows]
        invalid_gradient[0]["gradient_norm_after_clip"] = 0.4
        _write_validation_csv(bad_gradient, gradient_columns, invalid_gradient)
        _expect_value_error(lambda: load_factorized_actor_gradient_log(bad_gradient), "increased clipped norm was accepted")
        invalid_gradient = [dict(item) for item in gradient_rows]
        invalid_gradient[0]["selected_source"] = "D" if invalid_gradient[0]["selected_source"] == "P" else "P"
        _write_validation_csv(bad_gradient, gradient_columns, invalid_gradient)
        _expect_value_error(lambda: load_factorized_actor_gradient_log(bad_gradient), "mixed gradient source was accepted")

        cross_training = [dict(item) for item in episode_rows]
        cross_training[0]["quality_checkpoint_count"] = 3
        _write_validation_csv(bad_training, episode_columns, cross_training)
        _write_validation_csv(checkpoint_path, checkpoint_columns, checkpoint_rows)
        _write_validation_csv(gradient_path, gradient_columns, gradient_rows)
        _expect_value_error(
            lambda: generate_training_diagnostics(bad_training, checkpoint_path, gradient_path, root / "bad_cross_output"),
            "episode/checkpoint count mismatch was accepted",
        )

        header_only = root / "header_only_gradient.csv"
        _write_validation_csv(header_only, gradient_columns, [])
        _require(load_factorized_actor_gradient_log(header_only) == (), "header-only gradient log was rejected")
        header_only_result = generate_training_diagnostics(training_path, checkpoint_path, header_only, root / "header_only_output", moving_average_window=2, latest_checkpoint_episodes=2)
        _require("Actor gradient log contains only a header" in header_only_result.warnings, "header-only gradient warning was not recorded")
        no_gradient_output = root / "without_gradient"
        no_gradient = generate_training_diagnostics(training_path, checkpoint_path, None, no_gradient_output, moving_average_window=2, latest_checkpoint_episodes=2)
        _require("Actor gradient log was not supplied" in no_gradient.warnings, "missing gradient warning was not recorded")
        _require(not any(path.endswith("actor_gradient_norm_vs_episode.png") for path in no_gradient.generated_files), "gradient figure was generated without a log")
        source = Path(__file__).read_text(encoding="utf-8")
        forbidden_pattern = r"(?m)^\s*(?:import|from)\s+(?:pandas|seaborn|scipy|sklearn|plotly)(?:\s|\.)"
        _require(re.search(forbidden_pattern, source) is None, "module imports a forbidden dependency")
    return True


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser without searching for logs implicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", required=True, help="factorized_training_log.csv path")
    parser.add_argument("--checkpoint-log", required=True, help="factorized_quality_checkpoint_log.csv path")
    parser.add_argument("--gradient-log", help="optional factorized_actor_gradient_log.csv path")
    parser.add_argument("--output-dir", required=True, help="diagnostic output directory")
    parser.add_argument("--moving-average-window", type=int, default=10)
    parser.add_argument("--latest-checkpoint-episodes", type=int, default=10)
    parser.add_argument("--minimum-cliff-delta", type=float, default=0.01)
    parser.add_argument("--cliff-mad-multiplier", type=float, default=3.5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline diagnostics CLI and print a compact completion summary."""
    arguments = _build_argument_parser().parse_args(argv)
    result = generate_training_diagnostics(
        arguments.training_log,
        arguments.checkpoint_log,
        arguments.gradient_log,
        arguments.output_dir,
        moving_average_window=arguments.moving_average_window,
        latest_checkpoint_episodes=arguments.latest_checkpoint_episodes,
        minimum_cliff_delta=arguments.minimum_cliff_delta,
        cliff_mad_multiplier=arguments.cliff_mad_multiplier,
    )
    frontier_count = sum(item.is_pareto for item in result.pareto_points)
    summary_path = next(path for path in result.generated_files if path.endswith("diagnostics_summary.json"))
    print(f"episode_count={result.episode_count}")
    print(f"checkpoint_count={result.checkpoint_count}")
    print(f"cliff_event_count={len(result.cliff_events)}")
    print(f"pareto_frontier_count={frontier_count}")
    print(f"scenes={json.dumps(result.scenes, ensure_ascii=False)}")
    print(f"summary_json={summary_path}")
    print(f"figures_dir={Path(arguments.output_dir).resolve() / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
