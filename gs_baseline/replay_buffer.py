from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward_quality: float
    reward_size: float
    next_state: np.ndarray
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class EpisodeBuffer:
    """Small on-policy episode buffer for the first Dual-Critic baseline."""

    def __init__(self) -> None:
        self.transitions: list[Transition] = []

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward_quality: float,
        reward_size: float,
        next_state: np.ndarray,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        self.transitions.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward_quality=float(reward_quality),
                reward_size=float(reward_size),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
                info={} if info is None else info,
            )
        )

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)

    def states(self) -> np.ndarray:
        return np.stack([transition.state for transition in self.transitions]).astype(np.float32)

    def actions(self) -> np.ndarray:
        return np.array([transition.action for transition in self.transitions], dtype=np.int64)

    def returns(self, reward_name: str, gamma: float) -> np.ndarray:
        running = 0.0
        returns: list[float] = []
        for transition in reversed(self.transitions):
            reward = getattr(transition, reward_name)
            running = reward + gamma * running * (0.0 if transition.done else 1.0)
            returns.append(running)
        returns.reverse()
        return np.array(returns, dtype=np.float32)
