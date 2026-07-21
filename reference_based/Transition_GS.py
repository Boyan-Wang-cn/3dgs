"""Formal replay storage for two-axis, dual-Critic 3DGS training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


STATE_DIM = 19
ACTION_DIM = 2
REPLAY_SCHEMA = "factorized_replay_v1"


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array.copy()


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite float")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be bool")
    return bool(value)


@dataclass
class FactorizedTransition:
    """One owned 19D/2D transition with independent D and P rewards."""

    state: np.ndarray
    action: np.ndarray
    reward_D: float
    reward_P: float
    next_state: np.ndarray
    done: bool
    left_bitbudget: float
    episode_id: int
    step_index: int
    groups_processed: int
    quality_observed: bool = False
    quality_score: float | None = None
    quality_drop: float | None = None
    quality_target_ready: bool = False
    quality_block_reward: float | None = None
    quality_horizon: int | None = None
    quality_block_end_step_index: int | None = None
    quality_checkpoint_id: int | None = None
    quality_bootstrap_state: np.ndarray | None = None
    quality_bootstrap_done: bool | None = None


class FactorizedReplayMemory:
    """Bounded replay with transactional quality blocks and checkpoints."""

    MAX_QUALITY_BLOCK_LENGTH = 8

    def __init__(self, title: str, buffer_size: int, seed: int = 0) -> None:
        if isinstance(buffer_size, (bool, np.bool_)) or not isinstance(
            buffer_size, (int, np.integer)
        ) or int(buffer_size) <= 0:
            raise ValueError("buffer_size must be a positive integer")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        self.title = str(title)
        self.buffer_size = int(buffer_size)
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.replay_memory: list[FactorizedTransition] = []
        self.position = 0
        self._rng = np.random.default_rng(int(seed))

    @staticmethod
    def _validate_transition_values(
        *,
        state: Any,
        action: Any,
        reward_D: Any,
        reward_P: Any,
        next_state: Any,
        done: Any,
        left_bitbudget: Any,
        episode_id: Any,
        step_index: Any,
        groups_processed: Any,
    ) -> FactorizedTransition:
        normalized_groups = _nonnegative_int(groups_processed, "groups_processed")
        if normalized_groups <= 0:
            raise ValueError("groups_processed must be positive")
        return FactorizedTransition(
            state=_finite_array(state, (STATE_DIM,), "state"),
            action=_finite_array(action, (ACTION_DIM,), "action"),
            reward_D=_finite_float(reward_D, "reward_D"),
            reward_P=_finite_float(reward_P, "reward_P"),
            next_state=_finite_array(next_state, (STATE_DIM,), "next_state"),
            done=_strict_bool(done, "done"),
            left_bitbudget=_finite_float(left_bitbudget, "left_bitbudget"),
            episode_id=_nonnegative_int(episode_id, "episode_id"),
            step_index=_nonnegative_int(step_index, "step_index"),
            groups_processed=normalized_groups,
        )

    def append_transition(
        self,
        *,
        state: Any,
        action: Any,
        reward_D: Any,
        reward_P: Any,
        next_state: Any,
        done: Any,
        left_bitbudget: Any,
        episode_id: Any,
        step_index: Any,
        groups_processed: Any,
    ) -> None:
        transition = self._validate_transition_values(
            state=state,
            action=action,
            reward_D=reward_D,
            reward_P=reward_P,
            next_state=next_state,
            done=done,
            left_bitbudget=left_bitbudget,
            episode_id=episode_id,
            step_index=step_index,
            groups_processed=groups_processed,
        )
        if len(self.replay_memory) < self.buffer_size:
            self.replay_memory.append(transition)
        else:
            self.replay_memory[self.position] = transition
        self.position = (self.position + 1) % self.buffer_size

    def finalize_quality_block(
        self,
        *,
        episode_id: Any,
        start_step_index: Any,
        end_step_index: Any,
        quality_block_reward: Any,
        quality_score: Any,
        quality_drop: Any,
        bootstrap_state: Any,
        bootstrap_done: Any,
    ) -> None:
        """Validate a complete block before backfilling any transition."""
        episode = _nonnegative_int(episode_id, "episode_id")
        start = _nonnegative_int(start_step_index, "start_step_index")
        end = _nonnegative_int(end_step_index, "end_step_index")
        if end < start or end - start + 1 > self.MAX_QUALITY_BLOCK_LENGTH:
            raise ValueError("quality block length must be between 1 and 8")
        reward = _finite_float(quality_block_reward, "quality_block_reward")
        score = _finite_float(quality_score, "quality_score")
        drop = _finite_float(quality_drop, "quality_drop")
        bootstrap = _finite_array(
            bootstrap_state, (STATE_DIM,), "bootstrap_state"
        )
        bootstrap_terminal = _strict_bool(bootstrap_done, "bootstrap_done")
        matching = sorted(
            (
                transition
                for transition in self.replay_memory
                if transition.episode_id == episode
                and start <= transition.step_index <= end
            ),
            key=lambda transition: transition.step_index,
        )
        if [transition.step_index for transition in matching] != list(range(start, end + 1)):
            raise ValueError("quality block transitions are missing or duplicated")
        if any(transition.quality_target_ready for transition in matching):
            raise ValueError("quality block overlaps a finalized checkpoint")
        checkpoint_id = end
        staged: list[dict[str, Any]] = []
        for transition in matching:
            observed = transition.step_index == end
            staged.append({
                "quality_observed": observed,
                "quality_score": score if observed else None,
                "quality_drop": drop if observed else None,
                "quality_target_ready": True,
                "quality_block_reward": reward,
                "quality_horizon": end - transition.step_index + 1,
                "quality_block_end_step_index": end,
                "quality_checkpoint_id": checkpoint_id,
                "quality_bootstrap_state": bootstrap.copy(),
                "quality_bootstrap_done": bootstrap_terminal,
            })
        for transition, values in zip(matching, staged):
            for name, value in values.items():
                setattr(transition, name, value)

    def __len__(self) -> int:
        return len(self.replay_memory)

    @property
    def quality_ready_count(self) -> int:
        return sum(transition.quality_target_ready for transition in self.replay_memory)

    def clear(self) -> None:
        self.replay_memory.clear()
        self.position = 0

    def _sample(self, candidates: list[FactorizedTransition], batch_size: Any) -> list[FactorizedTransition]:
        if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
            batch_size, (int, np.integer)
        ) or int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")
        count = int(batch_size)
        if count > len(candidates):
            raise ValueError("batch_size exceeds available transitions")
        indices = self._rng.choice(len(candidates), size=count, replace=False)
        return [candidates[int(index)] for index in indices]

    @staticmethod
    def _general_arrays(batch: list[FactorizedTransition]) -> dict[str, np.ndarray]:
        return {
            "states": np.stack([item.state for item in batch]).astype(np.float64),
            "actions": np.stack([item.action for item in batch]).astype(np.float64),
            "rewards_D": np.asarray([item.reward_D for item in batch], dtype=np.float64).reshape(-1, 1),
            "rewards_P": np.asarray([item.reward_P for item in batch], dtype=np.float64).reshape(-1, 1),
            "next_states": np.stack([item.next_state for item in batch]).astype(np.float64),
            "dones": np.asarray([item.done for item in batch], dtype=np.float64).reshape(-1, 1),
            "left_bitbudgets": np.asarray([item.left_bitbudget for item in batch], dtype=np.float64).reshape(-1, 1),
        }

    def sample_batch_arrays(self, batch_size: Any) -> dict[str, np.ndarray]:
        return self._general_arrays(self._sample(self.replay_memory, batch_size))

    def sample_quality_batch_arrays(self, batch_size: Any) -> dict[str, np.ndarray]:
        candidates = [item for item in self.replay_memory if item.quality_target_ready]
        batch = self._sample(candidates, batch_size)
        arrays = self._general_arrays(batch)
        arrays.update({
            "quality_block_rewards": np.asarray([item.quality_block_reward for item in batch], dtype=np.float64).reshape(-1, 1),
            "quality_horizons": np.asarray([item.quality_horizon for item in batch], dtype=np.int64).reshape(-1, 1),
            "quality_block_end_step_indices": np.asarray([item.quality_block_end_step_index for item in batch], dtype=np.int64).reshape(-1, 1),
            "quality_checkpoint_ids": np.asarray([item.quality_checkpoint_id for item in batch], dtype=np.int64).reshape(-1, 1),
            "quality_bootstrap_states": np.stack([item.quality_bootstrap_state for item in batch]).astype(np.float64),
            "quality_bootstrap_dones": np.asarray([item.quality_bootstrap_done for item in batch], dtype=np.float64).reshape(-1, 1),
        })
        return arrays

    @staticmethod
    def _transition_state(transition: FactorizedTransition) -> dict[str, Any]:
        result = asdict(transition)
        for name in ("state", "action", "next_state", "quality_bootstrap_state"):
            value = result[name]
            result[name] = None if value is None else np.asarray(value).copy()
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": REPLAY_SCHEMA,
            "capacity": self.buffer_size,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "transitions": [self._transition_state(item) for item in self.replay_memory],
            "position": self.position,
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    @classmethod
    def _transition_from_state(cls, state: dict[str, Any]) -> FactorizedTransition:
        if not isinstance(state, dict):
            raise ValueError("serialized transition must be a dict")
        required = {
            "state", "action", "reward_D", "reward_P", "next_state", "done",
            "left_bitbudget", "episode_id", "step_index", "groups_processed",
            "quality_observed", "quality_score", "quality_drop",
            "quality_target_ready", "quality_block_reward", "quality_horizon",
            "quality_block_end_step_index", "quality_checkpoint_id",
            "quality_bootstrap_state", "quality_bootstrap_done",
        }
        if set(state) != required:
            raise ValueError("serialized transition fields are incompatible")
        transition = cls._validate_transition_values(
            state=state["state"], action=state["action"],
            reward_D=state["reward_D"], reward_P=state["reward_P"],
            next_state=state["next_state"], done=state["done"],
            left_bitbudget=state["left_bitbudget"], episode_id=state["episode_id"],
            step_index=state["step_index"], groups_processed=state["groups_processed"],
        )
        observed = _strict_bool(state["quality_observed"], "quality_observed")
        ready = _strict_bool(state["quality_target_ready"], "quality_target_ready")
        if not ready:
            optional = (
                "quality_score", "quality_drop", "quality_block_reward",
                "quality_horizon", "quality_block_end_step_index",
                "quality_checkpoint_id", "quality_bootstrap_state",
                "quality_bootstrap_done",
            )
            if observed or any(state[name] is not None for name in optional):
                raise ValueError("unready transition contains quality metadata")
            return transition
        horizon = _nonnegative_int(state["quality_horizon"], "quality_horizon")
        if not 1 <= horizon <= cls.MAX_QUALITY_BLOCK_LENGTH:
            raise ValueError("quality_horizon must be between 1 and 8")
        end = _nonnegative_int(state["quality_block_end_step_index"], "quality block end")
        checkpoint_id = _nonnegative_int(state["quality_checkpoint_id"], "quality checkpoint id")
        if end - transition.step_index + 1 != horizon or checkpoint_id != end:
            raise ValueError("quality horizon/checkpoint metadata is inconsistent")
        transition.quality_observed = observed
        transition.quality_score = _finite_float(state["quality_score"], "quality_score") if observed else None
        transition.quality_drop = _finite_float(state["quality_drop"], "quality_drop") if observed else None
        if not observed and (state["quality_score"] is not None or state["quality_drop"] is not None):
            raise ValueError("non-checkpoint transition contains observed quality")
        transition.quality_target_ready = True
        transition.quality_block_reward = _finite_float(state["quality_block_reward"], "quality_block_reward")
        transition.quality_horizon = horizon
        transition.quality_block_end_step_index = end
        transition.quality_checkpoint_id = checkpoint_id
        transition.quality_bootstrap_state = _finite_array(state["quality_bootstrap_state"], (STATE_DIM,), "quality_bootstrap_state")
        transition.quality_bootstrap_done = _strict_bool(state["quality_bootstrap_done"], "quality_bootstrap_done")
        return transition

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Validate transitions, position, and RNG before committing anything."""
        if not isinstance(state, dict) or state.get("schema") != REPLAY_SCHEMA:
            raise ValueError("incompatible Replay checkpoint schema")
        if state.get("capacity") != self.buffer_size:
            raise ValueError("Replay checkpoint capacity is incompatible")
        if state.get("state_dim") != STATE_DIM or state.get("action_dim") != ACTION_DIM:
            raise ValueError("Replay checkpoint dimensions are incompatible")
        serialized = state.get("transitions")
        if not isinstance(serialized, list) or len(serialized) > self.buffer_size:
            raise ValueError("Replay checkpoint transitions are invalid")
        transitions = [self._transition_from_state(item) for item in serialized]
        position = _nonnegative_int(state.get("position"), "position")
        if position >= self.buffer_size or (
            len(transitions) < self.buffer_size and position != len(transitions)
        ):
            raise ValueError("Replay checkpoint position is inconsistent")
        candidate_rng = np.random.default_rng()
        try:
            candidate_rng.bit_generator.state = deepcopy(state["rng_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Replay checkpoint RNG state is invalid") from exc
        self.replay_memory = transitions
        self.position = position
        self._rng = candidate_rng


_FIRST_VERSION_REPLAY_REPORT: dict[str, Any] = {}


def validate_first_version_replay() -> bool:
    """Validate shapes, blocks, sampling, roundtrip, RNG, and rollback."""
    if _FIRST_VERSION_REPLAY_REPORT.get("validated") is True:
        return True

    def require(condition: Any, message: str) -> None:
        if not bool(condition):
            raise AssertionError(message)

    memory = FactorizedReplayMemory("validation", 32, seed=123)
    for step in range(15):
        memory.append_transition(
            state=np.full(19, step, dtype=np.float64),
            action=np.asarray([step % 5, step % 6], dtype=np.float64),
            reward_D=-0.5 if step in {7, 14} else 0.0,
            reward_P=0.1 + step,
            next_state=np.full(19, step + 1, dtype=np.float64),
            done=step == 14,
            left_bitbudget=1.0 - step / 15.0,
            episode_id=1, step_index=step, groups_processed=step + 1,
        )
    memory.finalize_quality_block(
        episode_id=1, start_step_index=0, end_step_index=7,
        quality_block_reward=-0.5, quality_score=0.9, quality_drop=0.1,
        bootstrap_state=np.full(19, 8.0), bootstrap_done=False,
    )
    memory.finalize_quality_block(
        episode_id=1, start_step_index=8, end_step_index=14,
        quality_block_reward=-0.2, quality_score=0.85, quality_drop=0.15,
        bootstrap_state=np.full(19, 15.0), bootstrap_done=True,
    )
    horizons = [item.quality_horizon for item in memory.replay_memory]
    require(horizons == [8, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1], "quality horizons failed")
    require(memory.quality_ready_count == 15, "quality-ready count failed")
    require(memory.replay_memory[0].state.dtype == np.float64 and memory.replay_memory[0].action.shape == (2,), "transition shapes failed")
    general = memory.sample_batch_arrays(4)
    quality = memory.sample_quality_batch_arrays(4)
    require(general["states"].shape == (4, 19) and general["actions"].shape == (4, 2), "general sampling failed")
    require(quality["quality_horizons"].shape == (4, 1), "quality sampling failed")

    checkpoint = memory.state_dict()
    expected_sample = memory.sample_batch_arrays(5)["states"]
    restored = FactorizedReplayMemory("restored", 32, seed=999)
    restored.load_state_dict(checkpoint)
    require(np.array_equal(restored.sample_batch_arrays(5)["states"], expected_sample), "Replay RNG restore failed")
    require(restored.quality_ready_count == 15 and restored.position == 15, "Replay roundtrip failed")
    before = restored.state_dict()
    invalid = deepcopy(checkpoint); invalid["transitions"][0]["action"] = np.zeros(1)
    try:
        restored.load_state_dict(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid Replay checkpoint was accepted")
    after = restored.state_dict()
    require(before["position"] == after["position"], "failed Replay load changed position")
    require(np.array_equal(before["transitions"][0]["state"], after["transitions"][0]["state"]), "failed Replay load changed buffer")
    require(before["rng_state"] == after["rng_state"], "failed Replay load changed RNG")

    _FIRST_VERSION_REPLAY_REPORT.update({
        "validated": True,
        "state_shape": (19,), "action_shape": (2,),
        "quality_horizons": horizons, "quality_ready_count": 15,
        "general_sampling": True, "quality_sampling": True,
        "checkpoint_roundtrip": True, "rng_restore": True,
        "transactional_failure": True,
    })
    return True


def validate_factorized_replay_memory() -> bool:
    """Compatibility validation name for the formal Replay suite."""
    return validate_first_version_replay()
