"""This file is adapted from Train/DRL_x265_TRAIN/Transition.py.

Field mapping:
- baseQP = base compression level.
- nextBaseQP = next base compression level.
- reward_D = quality reward, matching the original distortion critic slot.
- reward_P = size reward / size penalty, matching the original rate critic slot.
- left_bitbudget = remaining size budget. Negative means the compressed model
  is still larger than the target size budget.
"""

from __future__ import annotations

import pickle
import random
from collections import namedtuple


Transition = namedtuple(
    "Transition",
    [
        "state",
        "action",
        "baseQP",
        "nextBaseQP",
        "reward_D",
        "reward_P",
        "next_state",
        "done",
        "left_bitbudget",
    ],
)


class reMemory:
    def __init__(self, title, buffer_size):
        self.title = title
        self.buffer_size = int(buffer_size)
        self.filename = self.title.replace(" ", "_") + ".pkl"
        self.replayMemory = []

    def check_filename(self, filename):
        if filename is not None:
            self.filename = filename

    def save(self, filename=None):
        self.check_filename(filename)
        with open(self.filename, "wb") as fh:
            pickle.dump(self.replayMemory, fh)

    def clear(self):
        self.replayMemory = []

    def appendTransition(
        self,
        observation,
        action,
        baseQP,
        nextBaseQP,
        reward_D,
        reward_P,
        next_observation,
        done,
        left_bitbudget,
    ):
        while len(self.replayMemory) >= self.buffer_size:
            self.replayMemory.pop(0)
        self.replayMemory.append(
            Transition(
                observation,
                action,
                baseQP,
                nextBaseQP,
                reward_D,
                reward_P,
                next_observation,
                done,
                left_bitbudget,
            )
        )

    def extend_replay_memory(self, m):
        self.replayMemory.extend(m.replayMemory)
        while len(self.replayMemory) > self.buffer_size:
            self.replayMemory.pop(0)

    def sample_batch(self, batch_size):
        if len(self.replayMemory) < batch_size:
            raise ValueError(
                f"Not enough transitions to sample {batch_size}; "
                f"buffer has {len(self.replayMemory)}."
            )
        return random.sample(self.replayMemory, batch_size)

    def load(self, filename=None):
        self.check_filename(filename)
        try:
            with open(self.filename, "rb") as fh:
                self.replayMemory = pickle.load(fh)
        except (OSError, pickle.PickleError, EOFError):
            self.replayMemory = []
