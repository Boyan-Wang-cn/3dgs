from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
except ModuleNotFoundError:
    torch = None
    nn = None
    Categorical = None


def require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for gs_baseline.networks and training. "
            "Install a CPU or CUDA build of torch, then rerun training."
        )


def _mlp(state_dim: int, output_dim: int):
    require_torch()
    return nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, output_dim),
    )


class Actor(nn.Module if nn is not None else object):
    """Categorical policy over five compression levels."""

    def __init__(self, state_dim: int, action_dim: int = 5) -> None:
        require_torch()
        super().__init__()
        self.net = _mlp(state_dim, action_dim)

    def forward(self, state):
        return self.net(state)

    def distribution(self, state):
        logits = self.forward(state)
        return Categorical(logits=logits)

    def sample_action(self, state):
        dist = self.distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy


class QualityCritic(nn.Module if nn is not None else object):
    """State-value critic for quality reward."""

    def __init__(self, state_dim: int) -> None:
        require_torch()
        super().__init__()
        self.net = _mlp(state_dim, 1)

    def forward(self, state):
        return self.net(state).squeeze(-1)


class SizeCritic(nn.Module if nn is not None else object):
    """State-value critic for size-constraint reward."""

    def __init__(self, state_dim: int) -> None:
        require_torch()
        super().__init__()
        self.net = _mlp(state_dim, 1)

    def forward(self, state):
        return self.net(state).squeeze(-1)
