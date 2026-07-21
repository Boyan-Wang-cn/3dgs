"""Deterministic NumPy networks for first-version factorized 3DGS training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


TAU = 0.001
STATE_DIM = 19
ACTION_DIM = 2
ACTION_BOUNDS = np.asarray([4.0, 5.0], dtype=np.float64)


def _finite_2d(value: Any, width: int, name: str) -> np.ndarray:
    """Return a finite float64 batch with an exact second dimension."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape [B, {width}], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _strict_rate(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite nonnegative float")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative float") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative float")
    return result


@dataclass
class _AdamState:
    m_w: list[np.ndarray]
    v_w: list[np.ndarray]
    m_b: list[np.ndarray]
    v_b: list[np.ndarray]
    t: int = 0


class _NumpyMLP:
    """Minimal float64 MLP with Adam and transactional checkpoints."""

    SCHEMA = "numpy_mlp_v1"

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        learning_rate: float,
        output_activation: str = "linear",
        seed: int = 0,
    ) -> None:
        dims = [int(input_dim), *[int(value) for value in hidden_dims], int(output_dim)]
        if len(dims) < 2 or any(value <= 0 for value in dims):
            raise ValueError("all MLP dimensions must be positive")
        if output_activation not in {"linear", "sigmoid", "tanh"}:
            raise ValueError("unsupported output activation")
        self.learning_rate = _strict_rate(learning_rate, "learning_rate")
        self.output_activation = output_activation
        generator = np.random.default_rng(int(seed))
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for input_width, output_width in zip(dims[:-1], dims[1:]):
            limit = 1.0 / np.sqrt(max(input_width, 1))
            self.weights.append(
                generator.uniform(
                    -limit, limit, size=(input_width, output_width)
                ).astype(np.float64)
            )
            self.biases.append(np.zeros((1, output_width), dtype=np.float64))
        self.adam = self._zero_adam()

    def _zero_adam(self) -> _AdamState:
        return _AdamState(
            [np.zeros_like(value) for value in self.weights],
            [np.zeros_like(value) for value in self.weights],
            [np.zeros_like(value) for value in self.biases],
            [np.zeros_like(value) for value in self.biases],
        )

    def _activate_output(self, value: np.ndarray) -> np.ndarray:
        if self.output_activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))
        if self.output_activation == "tanh":
            return np.tanh(value)
        return value

    def _output_derivative(self, activated: np.ndarray) -> np.ndarray:
        if self.output_activation == "sigmoid":
            return activated * (1.0 - activated)
        if self.output_activation == "tanh":
            return 1.0 - np.square(activated)
        return np.ones_like(activated)

    def forward(
        self, value: Any, *, keep_cache: bool = False
    ) -> np.ndarray | tuple[np.ndarray, tuple[list[np.ndarray], list[np.ndarray]]]:
        current = _finite_2d(value, self.weights[0].shape[0], "MLP input")
        activations = [current]
        pre_activations: list[np.ndarray] = []
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            linear = current @ weight + bias
            pre_activations.append(linear)
            current = (
                self._activate_output(linear)
                if index == len(self.weights) - 1
                else np.maximum(linear, 0.0)
            )
            activations.append(current)
        if not np.all(np.isfinite(current)):
            raise RuntimeError("MLP produced non-finite output")
        return (current, (activations, pre_activations)) if keep_cache else current

    def _backward(
        self,
        cache: tuple[list[np.ndarray], list[np.ndarray]],
        output_gradient: Any,
        *,
        update: bool,
    ) -> np.ndarray:
        activations, pre_activations = cache
        gradient = _finite_2d(
            output_gradient, self.weights[-1].shape[1], "output_gradient"
        )
        if gradient.shape[0] != activations[-1].shape[0]:
            raise ValueError("output_gradient batch size does not match the input")
        gradient = gradient * self._output_derivative(activations[-1])
        gradients_w = [np.empty_like(value) for value in self.weights]
        gradients_b = [np.empty_like(value) for value in self.biases]
        for layer in reversed(range(len(self.weights))):
            gradients_w[layer] = activations[layer].T @ gradient
            gradients_b[layer] = np.sum(gradient, axis=0, keepdims=True)
            gradient = gradient @ self.weights[layer].T
            if layer:
                gradient *= pre_activations[layer - 1] > 0.0
        if update:
            self._adam_step(gradients_w, gradients_b)
        return gradient

    def _adam_step(
        self, gradients_w: list[np.ndarray], gradients_b: list[np.ndarray]
    ) -> None:
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        self.adam.t += 1
        for index in range(len(self.weights)):
            self.adam.m_w[index] = (
                beta1 * self.adam.m_w[index] + (1.0 - beta1) * gradients_w[index]
            )
            self.adam.v_w[index] = (
                beta2 * self.adam.v_w[index]
                + (1.0 - beta2) * np.square(gradients_w[index])
            )
            self.adam.m_b[index] = (
                beta1 * self.adam.m_b[index] + (1.0 - beta1) * gradients_b[index]
            )
            self.adam.v_b[index] = (
                beta2 * self.adam.v_b[index]
                + (1.0 - beta2) * np.square(gradients_b[index])
            )
            correction1 = 1.0 - beta1**self.adam.t
            correction2 = 1.0 - beta2**self.adam.t
            self.weights[index] -= self.learning_rate * (
                self.adam.m_w[index] / correction1
            ) / (np.sqrt(self.adam.v_w[index] / correction2) + epsilon)
            self.biases[index] -= self.learning_rate * (
                self.adam.m_b[index] / correction1
            ) / (np.sqrt(self.adam.v_b[index] / correction2) + epsilon)

    def train_mse(self, value: Any, target: Any) -> float:
        prediction, cache = self.forward(value, keep_cache=True)
        normalized_target = _finite_2d(target, prediction.shape[1], "target")
        if normalized_target.shape != prediction.shape:
            raise ValueError("target shape does not match prediction shape")
        difference = prediction - normalized_target
        loss = float(np.mean(np.square(difference)))
        self._backward(
            cache,
            2.0 * difference / max(len(difference), 1),
            update=True,
        )
        return loss

    def apply_output_gradient(self, value: Any, output_gradient: Any) -> None:
        prediction, cache = self.forward(value, keep_cache=True)
        gradient = _finite_2d(
            output_gradient, prediction.shape[1], "output_gradient"
        )
        if gradient.shape != prediction.shape:
            raise ValueError("output_gradient shape does not match prediction shape")
        self._backward(cache, -gradient / max(len(prediction), 1), update=True)

    def input_gradient(self, value: Any, output_gradient: Any) -> np.ndarray:
        prediction, cache = self.forward(value, keep_cache=True)
        gradient = _finite_2d(
            output_gradient, prediction.shape[1], "output_gradient"
        )
        if gradient.shape != prediction.shape:
            raise ValueError("output_gradient shape does not match prediction shape")
        return self._backward(cache, gradient, update=False)

    def update_lr(self, learning_rate: float) -> None:
        self.learning_rate = _strict_rate(learning_rate, "learning_rate")

    def copy_from(self, other: "_NumpyMLP") -> None:
        if not isinstance(other, _NumpyMLP):
            raise ValueError("MLP source has an incompatible type")
        self.load_state_dict(other.state_dict())

    def soft_update_from(self, other: "_NumpyMLP", tau: float = TAU) -> None:
        if not isinstance(other, _NumpyMLP):
            raise ValueError("MLP source has an incompatible type")
        normalized_tau = float(tau)
        if not np.isfinite(normalized_tau) or not 0.0 <= normalized_tau <= 1.0:
            raise ValueError("tau must be finite and in [0, 1]")
        if [value.shape for value in self.weights] != [
            value.shape for value in other.weights
        ]:
            raise ValueError("MLP parameter shapes do not match")
        self.weights = [
            normalized_tau * source + (1.0 - normalized_tau) * destination
            for destination, source in zip(self.weights, other.weights)
        ]
        self.biases = [
            normalized_tau * source + (1.0 - normalized_tau) * destination
            for destination, source in zip(self.biases, other.biases)
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "weights": [value.copy() for value in self.weights],
            "biases": [value.copy() for value in self.biases],
            "output_activation": self.output_activation,
            "learning_rate": float(self.learning_rate),
            "adam": {
                "m_w": [value.copy() for value in self.adam.m_w],
                "v_w": [value.copy() for value in self.adam.v_w],
                "m_b": [value.copy() for value in self.adam.m_b],
                "v_b": [value.copy() for value in self.adam.v_b],
                "t": int(self.adam.t),
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Validate an entire MLP checkpoint before committing any field."""
        if not isinstance(state, dict) or state.get("schema") != self.SCHEMA:
            raise ValueError("incompatible MLP checkpoint schema")
        if state.get("output_activation") != self.output_activation:
            raise ValueError("MLP output activation is incompatible")
        try:
            weights = [np.asarray(value, dtype=np.float64).copy() for value in state["weights"]]
            biases = [np.asarray(value, dtype=np.float64).copy() for value in state["biases"]]
            adam = state["adam"]
            m_w = [np.asarray(value, dtype=np.float64).copy() for value in adam["m_w"]]
            v_w = [np.asarray(value, dtype=np.float64).copy() for value in adam["v_w"]]
            m_b = [np.asarray(value, dtype=np.float64).copy() for value in adam["m_b"]]
            v_b = [np.asarray(value, dtype=np.float64).copy() for value in adam["v_b"]]
            step = int(adam["t"])
            learning_rate = _strict_rate(state["learning_rate"], "learning_rate")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("MLP checkpoint structure is invalid") from exc
        expected_w = [value.shape for value in self.weights]
        expected_b = [value.shape for value in self.biases]
        if (
            [value.shape for value in weights] != expected_w
            or [value.shape for value in biases] != expected_b
            or [value.shape for value in m_w] != expected_w
            or [value.shape for value in v_w] != expected_w
            or [value.shape for value in m_b] != expected_b
            or [value.shape for value in v_b] != expected_b
            or step < 0
        ):
            raise ValueError("MLP checkpoint parameter shapes are incompatible")
        arrays = weights + biases + m_w + v_w + m_b + v_b
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("MLP checkpoint contains non-finite values")
        self.weights = weights
        self.biases = biases
        self.learning_rate = learning_rate
        self.adam = _AdamState(m_w, v_w, m_b, v_b, step)


class FactorizedActorNetwork:
    """Direct 19D-to-2D continuous compression Actor."""

    NETWORK_TYPE = "factorized_actor_v1"

    def __init__(
        self,
        sess: object | None,
        state_dim: int,
        action_dim: int,
        action_bound: Iterable[float],
        learning_rate: float,
        scope: object,
    ) -> None:
        if sess is not None:
            raise ValueError("the NumPy Actor requires sess=None")
        if int(state_dim) != STATE_DIM:
            raise ValueError("Factorized Actor state_dim must be 19")
        if int(action_dim) != ACTION_DIM:
            raise ValueError("Factorized Actor action_dim must be 2")
        bounds = np.asarray(action_bound, dtype=np.float64)
        if bounds.shape != (2,) or not np.array_equal(bounds, ACTION_BOUNDS):
            raise ValueError("Factorized Actor action_bound must be [4.0, 5.0]")
        self.sess = None
        self.s_dim = STATE_DIM
        self.a_dim = ACTION_DIM
        self.action_bound = bounds.copy()
        self.learning_rate = _strict_rate(learning_rate, "learning_rate")
        self.scope = scope
        self.pruning_bound = 4.0
        self.precision_bound = 5.0
        self.action_net = _NumpyMLP(
            STATE_DIM, (128, 128), ACTION_DIM, self.learning_rate,
            output_activation="sigmoid", seed=37,
        )

    def predict_action(self, states: Any) -> np.ndarray:
        normalized_states = _finite_2d(states, STATE_DIM, "states")
        return np.asarray(self.action_net.forward(normalized_states)) * ACTION_BOUNDS

    def predict(self, states: Any) -> np.ndarray:
        return self.predict_action(states)

    def train(self, states: Any, action_gradient: Any) -> None:
        normalized_states = _finite_2d(states, STATE_DIM, "states")
        gradients = _finite_2d(action_gradient, ACTION_DIM, "action_gradient")
        if gradients.shape[0] != normalized_states.shape[0]:
            raise ValueError("action_gradient batch size does not match states")
        self.action_net.apply_output_gradient(
            normalized_states, gradients * ACTION_BOUNDS
        )

    def update_lr(self, learning_rate: float) -> None:
        self.learning_rate = _strict_rate(learning_rate, "learning_rate")
        self.action_net.update_lr(self.learning_rate)

    def copy_from(self, other: "FactorizedActorNetwork") -> None:
        if not isinstance(other, FactorizedActorNetwork):
            raise ValueError("Actor source has an incompatible type")
        self.action_net.copy_from(other.action_net)
        self.learning_rate = other.learning_rate

    def soft_update_from(
        self, other: "FactorizedActorNetwork", tau: float = TAU
    ) -> None:
        if not isinstance(other, FactorizedActorNetwork):
            raise ValueError("Actor source has an incompatible type")
        self.action_net.soft_update_from(other.action_net, tau)

    def state_dict(self) -> dict[str, Any]:
        return {
            "network_type": self.NETWORK_TYPE,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "action_bounds": ACTION_BOUNDS.tolist(),
            "action_net": self.action_net.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Transactionally load only the formal Actor checkpoint schema."""
        if not isinstance(state, dict) or state.get("network_type") != self.NETWORK_TYPE:
            raise ValueError("incompatible Factorized Actor checkpoint")
        if state.get("state_dim") != STATE_DIM or state.get("action_dim") != ACTION_DIM:
            raise ValueError("Factorized Actor checkpoint dimensions are incompatible")
        bounds = np.asarray(state.get("action_bounds"), dtype=np.float64)
        if bounds.shape != (2,) or not np.array_equal(bounds, ACTION_BOUNDS):
            raise ValueError("Factorized Actor checkpoint bounds are incompatible")
        candidate = _NumpyMLP(
            STATE_DIM, (128, 128), ACTION_DIM, self.learning_rate,
            output_activation="sigmoid", seed=37,
        )
        candidate.load_state_dict(state.get("action_net"))
        self.action_net = candidate
        self.learning_rate = candidate.learning_rate

    def get_params(self) -> list[np.ndarray]:
        return self.action_net.weights + self.action_net.biases

    def get_scope(self) -> object:
        return self.scope


class _CriticQNetwork:
    """One independent state/action value branch."""

    def __init__(self, learning_rate: float, seed: int) -> None:
        self.net = _NumpyMLP(
            STATE_DIM + ACTION_DIM, (128, 128), 1, learning_rate,
            output_activation="linear", seed=seed,
        )

    @staticmethod
    def _input(states: Any, actions: Any) -> np.ndarray:
        normalized_states = _finite_2d(states, STATE_DIM, "states")
        normalized_actions = _finite_2d(actions, ACTION_DIM, "actions")
        if normalized_states.shape[0] != normalized_actions.shape[0]:
            raise ValueError("state and action batch sizes do not match")
        if np.any(normalized_actions < 0.0) or np.any(normalized_actions > 1.0):
            raise ValueError("Critic actions must be normalized to [0, 1]")
        return np.concatenate((normalized_states, normalized_actions), axis=1)

    def predict(self, states: Any, actions: Any) -> np.ndarray:
        return np.asarray(self.net.forward(self._input(states, actions)))

    def train(self, states: Any, actions: Any, targets: Any) -> float:
        return self.net.train_mse(self._input(states, actions), targets)

    def action_gradient(self, states: Any, actions: Any) -> np.ndarray:
        model_input = self._input(states, actions)
        full_gradient = self.net.input_gradient(
            model_input, np.ones((len(model_input), 1), dtype=np.float64)
        )
        return full_gradient[:, -ACTION_DIM:]

    def copy_from(self, other: "_CriticQNetwork") -> None:
        self.net.copy_from(other.net)

    def soft_update_from(self, other: "_CriticQNetwork", tau: float) -> None:
        self.net.soft_update_from(other.net, tau)

    def state_dict(self) -> dict[str, Any]:
        return self.net.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.net.load_state_dict(state)


class CriticNetwork:
    """Two fully independent Critics for quality D and size P."""

    NETWORK_TYPE = "dual_critic_v1"
    ACTION_INPUT_SPACE = "normalized_0_1"

    def __init__(
        self,
        sess: object | None,
        state_dim: int,
        action_dim: int,
        learning_rate: float,
        scope_D: object,
        scope_P: object,
    ) -> None:
        if sess is not None:
            raise ValueError("the NumPy Critic requires sess=None")
        if int(state_dim) != STATE_DIM or int(action_dim) != ACTION_DIM:
            raise ValueError("Critic dimensions must be state_dim=19/action_dim=2")
        self.sess = None
        self.s_dim = STATE_DIM
        self.a_dim = ACTION_DIM
        self.learning_rate = _strict_rate(learning_rate, "learning_rate")
        self.scope_D = scope_D
        self.scope_P = scope_P
        self.critic_D = _CriticQNetwork(self.learning_rate, seed=23)
        self.critic_P = _CriticQNetwork(self.learning_rate, seed=29)

    def predict(self, states: Any, actions: Any) -> tuple[np.ndarray, np.ndarray]:
        return self.critic_D.predict(states, actions), self.critic_P.predict(states, actions)

    def train_D(self, states: Any, actions: Any, targets: Any) -> float:
        return self.critic_D.train(states, actions, targets)

    def train_P(self, states: Any, actions: Any, targets: Any) -> float:
        return self.critic_P.train(states, actions, targets)

    def action_gradient_D(self, states: Any, actions: Any) -> np.ndarray:
        return self.critic_D.action_gradient(states, actions)

    def action_gradient_P(self, states: Any, actions: Any) -> np.ndarray:
        return self.critic_P.action_gradient(states, actions)

    def update_lr(self, learning_rate: float) -> None:
        self.learning_rate = _strict_rate(learning_rate, "learning_rate")
        self.critic_D.net.update_lr(self.learning_rate)
        self.critic_P.net.update_lr(self.learning_rate)

    def copy_from(self, other: "CriticNetwork") -> None:
        if not isinstance(other, CriticNetwork):
            raise ValueError("Critic source has an incompatible type")
        self.critic_D.copy_from(other.critic_D)
        self.critic_P.copy_from(other.critic_P)
        self.learning_rate = other.learning_rate

    def soft_update_D_from(self, other: "CriticNetwork", tau: float = TAU) -> None:
        if not isinstance(other, CriticNetwork):
            raise ValueError("Critic source has an incompatible type")
        self.critic_D.soft_update_from(other.critic_D, tau)

    def soft_update_P_from(self, other: "CriticNetwork", tau: float = TAU) -> None:
        if not isinstance(other, CriticNetwork):
            raise ValueError("Critic source has an incompatible type")
        self.critic_P.soft_update_from(other.critic_P, tau)

    def state_dict(self) -> dict[str, Any]:
        return {
            "network_type": self.NETWORK_TYPE,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "action_input_space": self.ACTION_INPUT_SPACE,
            "critic_D": self.critic_D.state_dict(),
            "critic_P": self.critic_P.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Validate both branches before atomically replacing either branch."""
        if not isinstance(state, dict) or state.get("network_type") != self.NETWORK_TYPE:
            raise ValueError("incompatible dual-Critic checkpoint")
        if state.get("state_dim") != STATE_DIM or state.get("action_dim") != ACTION_DIM:
            raise ValueError("dual-Critic checkpoint dimensions are incompatible")
        if state.get("action_input_space") != self.ACTION_INPUT_SPACE:
            raise ValueError("dual-Critic action input space is incompatible")
        candidate_D = _CriticQNetwork(self.learning_rate, seed=23)
        candidate_P = _CriticQNetwork(self.learning_rate, seed=29)
        try:
            candidate_D.load_state_dict(state["critic_D"])
            candidate_P.load_state_dict(state["critic_P"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("dual-Critic checkpoint branch is invalid") from exc
        self.critic_D = candidate_D
        self.critic_P = candidate_P
        self.learning_rate = candidate_D.net.learning_rate

    def get_params_D(self) -> list[np.ndarray]:
        return self.critic_D.net.weights + self.critic_D.net.biases

    def get_params_P(self) -> list[np.ndarray]:
        return self.critic_P.net.weights + self.critic_P.net.biases

    def get_scope_D(self) -> object:
        return self.scope_D

    def get_scope_P(self) -> object:
        return self.scope_P


_FIRST_VERSION_NETWORK_REPORT: dict[str, Any] = {}


def validate_first_version_networks() -> bool:
    """Validate direct Actor gradients, independent Critics, and transactions."""
    if _FIRST_VERSION_NETWORK_REPORT.get("validated") is True:
        return True

    def require(condition: Any, message: str) -> None:
        if not bool(condition):
            raise AssertionError(message)

    states = np.linspace(-1.0, 1.0, 57, dtype=np.float64).reshape(3, STATE_DIM)
    actor = FactorizedActorNetwork(None, 19, 2, ACTION_BOUNDS, 1e-3, "actor")
    actions = actor.predict_action(states)
    require(actions.shape == (3, 2), "Actor shape is incorrect")
    require(np.all(actions >= 0.0) and np.all(actions <= ACTION_BOUNDS), "Actor bounds failed")

    captured: dict[str, np.ndarray] = {}
    original_apply = actor.action_net.apply_output_gradient

    def capture(value: Any, gradient: Any) -> None:
        captured["gradient"] = np.asarray(gradient).copy()
        original_apply(value, gradient)

    actor.action_net.apply_output_gradient = capture
    actor_gradient = np.asarray([[1.0, -2.0], [0.5, 0.25], [-1.0, 3.0]])
    actor.train(states, actor_gradient)
    actor.action_net.apply_output_gradient = original_apply
    require(
        np.array_equal(captured["gradient"], actor_gradient * ACTION_BOUNDS),
        "Actor axes were merged or scaled incorrectly",
    )

    actor_state = actor.state_dict()
    actor_copy = FactorizedActorNetwork(None, 19, 2, ACTION_BOUNDS, 1e-3, "copy")
    actor_copy.load_state_dict(actor_state)
    require(np.array_equal(actor.predict_action(states), actor_copy.predict_action(states)), "Actor roundtrip failed")
    before_bad_actor = actor_copy.state_dict()
    invalid_actor = dict(actor_state); invalid_actor["action_net"] = dict(actor_state["action_net"])
    invalid_actor["action_net"]["weights"] = [np.zeros((1, 1))]
    try:
        actor_copy.load_state_dict(invalid_actor)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid Actor checkpoint was accepted")
    require(
        all(np.array_equal(left, right) for left, right in zip(before_bad_actor["action_net"]["weights"], actor_copy.state_dict()["action_net"]["weights"])),
        "failed Actor load changed parameters",
    )

    critic = CriticNetwork(None, 19, 2, 1e-3, "D", "P")
    normalized_actions = actions / ACTION_BOUNDS
    require(
        not any(
            np.shares_memory(left, right)
            for left in critic.get_params_D()
            for right in critic.get_params_P()
        ),
        "Critic branches share parameter memory",
    )
    targets = np.arange(3, dtype=np.float64).reshape(-1, 1)
    p_before = [value.copy() for value in critic.get_params_P()]
    critic.train_D(states, normalized_actions, targets)
    require(all(np.array_equal(left, right) for left, right in zip(p_before, critic.get_params_P())), "D-only training changed P")
    d_before = [value.copy() for value in critic.get_params_D()]
    critic.train_P(states, normalized_actions, targets)
    require(all(np.array_equal(left, right) for left, right in zip(d_before, critic.get_params_D())), "P-only training changed D")
    require(critic.action_gradient_D(states, normalized_actions).shape == (3, 2), "D gradient shape failed")
    require(critic.action_gradient_P(states, normalized_actions).shape == (3, 2), "P gradient shape failed")

    critic_state = critic.state_dict()
    critic_copy = CriticNetwork(None, 19, 2, 1e-3, "D2", "P2")
    critic_copy.load_state_dict(critic_state)
    expected_D, expected_P = critic.predict(states, normalized_actions)
    actual_D, actual_P = critic_copy.predict(states, normalized_actions)
    require(np.array_equal(expected_D, actual_D) and np.array_equal(expected_P, actual_P), "Critic roundtrip failed")
    before_bad_D = [value.copy() for value in critic_copy.get_params_D()]
    before_bad_P = [value.copy() for value in critic_copy.get_params_P()]
    invalid_critic = dict(critic_state); invalid_critic["critic_P"] = {"schema": "invalid"}
    try:
        critic_copy.load_state_dict(invalid_critic)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid Critic checkpoint was accepted")
    require(all(np.array_equal(left, right) for left, right in zip(before_bad_D, critic_copy.get_params_D())), "failed Critic load changed D")
    require(all(np.array_equal(left, right) for left, right in zip(before_bad_P, critic_copy.get_params_P())), "failed Critic load changed P")

    _FIRST_VERSION_NETWORK_REPORT.update({
        "validated": True,
        "actor_range": True,
        "actor_axis_gradient_scaling": True,
        "actor_checkpoint_roundtrip": True,
        "critic_checkpoint_roundtrip": True,
        "critic_parameters_independent": True,
        "D_only_preserves_P": True,
        "P_only_preserves_D": True,
        "transactional_failure": True,
    })
    return True


def validate_factorized_actor_network() -> bool:
    """Compatibility validation name for the formal network suite."""
    return validate_first_version_networks()
