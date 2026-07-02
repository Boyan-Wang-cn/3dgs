"""This file is adapted from Train/DRL_x265_TRAIN/Network.py.

QP-to-3DGS mapping:
- baseQP is kept as a variable name but now means base compression level.
- deltaQP is migrated to delta compression level.
- final action id = round(clip(baseAction + deltaAction, 0, 24)).
- q_value_D is the quality critic Q_D(s, a).
- q_value_P is the size critic Q_P(s, a).

The original code used TensorFlow 1.x placeholders and optimizer ops. This
reference-based version keeps the same class/API structure but implements the
small networks in NumPy so the migration can run without a TF1 environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


GAMMA = 0.99
TAU = 0.001
LEVEL_MAX = 24.0


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


@dataclass
class _AdamState:
    m_w: list[np.ndarray]
    v_w: list[np.ndarray]
    m_b: list[np.ndarray]
    v_b: list[np.ndarray]
    t: int = 0


class _NumpyMLP:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        learning_rate: float,
        output_activation: str = "linear",
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        dims = [input_dim, *hidden_dims, output_dim]
        self.weights = []
        self.biases = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            limit = 1.0 / np.sqrt(max(in_dim, 1))
            self.weights.append(rng.uniform(-limit, limit, size=(in_dim, out_dim)))
            self.biases.append(np.zeros((1, out_dim), dtype=np.float64))
        self.learning_rate = learning_rate
        self.output_activation = output_activation
        self.adam = _AdamState(
            m_w=[np.zeros_like(w) for w in self.weights],
            v_w=[np.zeros_like(w) for w in self.weights],
            m_b=[np.zeros_like(b) for b in self.biases],
            v_b=[np.zeros_like(b) for b in self.biases],
        )

    def _activate_output(self, z):
        if self.output_activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
        if self.output_activation == "tanh":
            return np.tanh(z)
        return z

    def _output_derivative(self, activated):
        if self.output_activation == "sigmoid":
            return activated * (1.0 - activated)
        if self.output_activation == "tanh":
            return 1.0 - np.square(activated)
        return np.ones_like(activated)

    def forward(self, x, keep_cache=False):
        a = _as_2d(x)
        activations = [a]
        pre_activations = []
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = a @ weight + bias
            pre_activations.append(z)
            if idx == len(self.weights) - 1:
                a = self._activate_output(z)
            else:
                a = np.maximum(z, 0.0)
            activations.append(a)
        if keep_cache:
            return a, (activations, pre_activations)
        return a

    def _backward(self, cache, grad_output, update=True):
        activations, pre_activations = cache
        grad = np.asarray(grad_output, dtype=np.float64)
        grad = grad * self._output_derivative(activations[-1])
        grad_w = [None for _ in self.weights]
        grad_b = [None for _ in self.biases]
        for layer in reversed(range(len(self.weights))):
            grad_w[layer] = activations[layer].T @ grad
            grad_b[layer] = np.sum(grad, axis=0, keepdims=True)
            grad = grad @ self.weights[layer].T
            if layer > 0:
                grad = grad * (pre_activations[layer - 1] > 0.0)
        grad_input = grad
        if update:
            self._adam_step(grad_w, grad_b)
        return grad_input

    def _adam_step(self, grad_w, grad_b):
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        self.adam.t += 1
        for i in range(len(self.weights)):
            self.adam.m_w[i] = beta1 * self.adam.m_w[i] + (1.0 - beta1) * grad_w[i]
            self.adam.v_w[i] = beta2 * self.adam.v_w[i] + (1.0 - beta2) * np.square(grad_w[i])
            self.adam.m_b[i] = beta1 * self.adam.m_b[i] + (1.0 - beta1) * grad_b[i]
            self.adam.v_b[i] = beta2 * self.adam.v_b[i] + (1.0 - beta2) * np.square(grad_b[i])
            m_w_hat = self.adam.m_w[i] / (1.0 - beta1 ** self.adam.t)
            v_w_hat = self.adam.v_w[i] / (1.0 - beta2 ** self.adam.t)
            m_b_hat = self.adam.m_b[i] / (1.0 - beta1 ** self.adam.t)
            v_b_hat = self.adam.v_b[i] / (1.0 - beta2 ** self.adam.t)
            self.weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
            self.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

    def train_mse(self, x, target):
        y, cache = self.forward(x, keep_cache=True)
        target = _as_2d(target)
        diff = y - target
        loss = float(np.mean(np.square(diff)))
        grad_output = 2.0 * diff / max(len(diff), 1)
        self._backward(cache, grad_output, update=True)
        return loss

    def apply_output_gradient(self, x, output_gradient):
        y, cache = self.forward(x, keep_cache=True)
        grad = -_as_2d(output_gradient) / max(len(y), 1)
        self._backward(cache, grad, update=True)

    def input_gradient(self, x, output_gradient):
        y, cache = self.forward(x, keep_cache=True)
        _ = y
        return self._backward(cache, _as_2d(output_gradient), update=False)

    def copy_from(self, other: "_NumpyMLP") -> None:
        self.weights = [w.copy() for w in other.weights]
        self.biases = [b.copy() for b in other.biases]
        self.learning_rate = other.learning_rate
        self.output_activation = other.output_activation
        self.adam = _AdamState(
            m_w=[m.copy() for m in other.adam.m_w],
            v_w=[v.copy() for v in other.adam.v_w],
            m_b=[m.copy() for m in other.adam.m_b],
            v_b=[v.copy() for v in other.adam.v_b],
            t=int(other.adam.t),
        )

    def soft_update_from(self, other: "_NumpyMLP", tau: float = TAU) -> None:
        for i in range(len(self.weights)):
            self.weights[i] = tau * other.weights[i] + (1.0 - tau) * self.weights[i]
            self.biases[i] = tau * other.biases[i] + (1.0 - tau) * self.biases[i]

    def state_dict(self):
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
            "output_activation": self.output_activation,
            "learning_rate": float(self.learning_rate),
            "adam": {
                "m_w": [m.copy() for m in self.adam.m_w],
                "v_w": [v.copy() for v in self.adam.v_w],
                "m_b": [m.copy() for m in self.adam.m_b],
                "v_b": [v.copy() for v in self.adam.v_b],
                "t": int(self.adam.t),
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self.weights = [np.asarray(w, dtype=np.float64).copy() for w in state["weights"]]
        self.biases = [np.asarray(b, dtype=np.float64).copy() for b in state["biases"]]
        self.output_activation = str(state.get("output_activation", self.output_activation))
        self.learning_rate = float(state.get("learning_rate", self.learning_rate))
        adam_state = state.get("adam") or {}
        self.adam = _AdamState(
            m_w=[
                np.asarray(m, dtype=np.float64).copy()
                for m in adam_state.get("m_w", [np.zeros_like(w) for w in self.weights])
            ],
            v_w=[
                np.asarray(v, dtype=np.float64).copy()
                for v in adam_state.get("v_w", [np.zeros_like(w) for w in self.weights])
            ],
            m_b=[
                np.asarray(m, dtype=np.float64).copy()
                for m in adam_state.get("m_b", [np.zeros_like(b) for b in self.biases])
            ],
            v_b=[
                np.asarray(v, dtype=np.float64).copy()
                for v in adam_state.get("v_b", [np.zeros_like(b) for b in self.biases])
            ],
            t=int(adam_state.get("t", 0)),
        )


class ActorNetwork(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, scope):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.scope = scope

        base_bound = float(action_bound[0]) if action_bound else LEVEL_MAX
        delta_bound = float(action_bound[1]) if len(action_bound) > 1 else 2.0
        self.base_bound = base_bound
        self.delta_bound = delta_bound

        # behavior network: baseLevel network + deltaLevel network, matching
        # the original baseQP + deltaQP decomposition.
        self.base_net = _NumpyMLP(
            state_dim,
            hidden_dims=(128, 128),
            output_dim=action_dim,
            learning_rate=learning_rate,
            output_activation="sigmoid",
            seed=11,
        )
        self.delta_net = _NumpyMLP(
            state_dim + 1,
            hidden_dims=(128, 128),
            output_dim=action_dim,
            learning_rate=learning_rate,
            output_activation="tanh",
            seed=17,
        )
        self.network_params_delta_qp = self.delta_net.weights + self.delta_net.biases
        self.network_params_qp = self.base_net.weights + self.base_net.biases

    def _delta_state(self, X, baseQP):
        return np.concatenate((_as_2d(X), _as_2d(baseQP) / LEVEL_MAX), axis=-1)

    def create_actor_network(self, scope, action_bound, s_dims):
        # Kept for code-level continuity with the TensorFlow original.
        return f"{scope}_input", f"{scope}_q_value", f"{scope}_scaled_out"

    def train(self, X, a_gradient):
        """Update both base-level and delta-level sub-networks.

        The original dual_critic baseline optimizes a baseQP branch and a
        deltaQP branch.  In this 3DGS migration the same decomposition is
        kept as base factorized action id + delta action id.  The critic
        provides dQ / d(final_level).  The final level depends on the base
        branch directly and also indirectly through the delta branch input
        (the delta network receives base_level / LEVEL_MAX).  We therefore
        update both branches instead of only training delta_net.
        """
        X = _as_2d(X)
        action_gradient = _as_2d(a_gradient)

        # Forward with the current base branch.  Gradients are computed before
        # either branch is updated so both branches see the same actor output.
        base_level = self.base_net.forward(X) * self.base_bound
        delta_state = self._delta_state(X, base_level)

        # dQ/d(delta_raw_output). delta_level = delta_raw * delta_bound.
        delta_output_gradient = action_gradient * self.delta_bound

        # Indirect contribution to dQ/d(base_level) through the delta network
        # input column that stores base_level / LEVEL_MAX.
        delta_input_gradient = self.delta_net.input_gradient(
            delta_state,
            delta_output_gradient,
        )
        base_norm_gradient = delta_input_gradient[:, -1:].copy()

        # Update delta branch first.
        self.delta_net.apply_output_gradient(delta_state, delta_output_gradient)

        # dQ/d(base_level) has a direct path through final_level = base + delta
        # and an indirect path through delta_state[-1] = base_level / LEVEL_MAX.
        base_level_gradient = action_gradient + base_norm_gradient / LEVEL_MAX

        # base_level = base_raw_output * base_bound, so convert gradient to the
        # raw output of base_net before applying the actor-gradient update.
        base_output_gradient = base_level_gradient * self.base_bound
        self.base_net.apply_output_gradient(X, base_output_gradient)
        return None

    def predict(self, X):
        baseQP = self.base_net.forward(X) * self.base_bound
        deleta_state = self._delta_state(X, baseQP)
        delta_level = self.delta_net.forward(deleta_state) * self.delta_bound
        return delta_level, baseQP

    def predict_action(self, X):
        delta_level, baseQP = self.predict(X)
        final_level = np.clip(baseQP + delta_level, 0.0, LEVEL_MAX)
        return final_level, delta_level, baseQP

    def get_scope(self):
        return self.scope[0], self.scope[1]

    def update_lr(self, lr):
        self.learning_rate = lr
        self.base_net.learning_rate = lr
        self.delta_net.learning_rate = lr

    def get_params(self):
        return self.network_params_delta_qp, self.network_params_qp

    def copy_from(self, other: "ActorNetwork") -> None:
        self.base_net.copy_from(other.base_net)
        self.delta_net.copy_from(other.delta_net)

    def soft_update_from(self, other: "ActorNetwork", tau: float = TAU) -> None:
        self.base_net.soft_update_from(other.base_net, tau)
        self.delta_net.soft_update_from(other.delta_net, tau)

    def state_dict(self):
        return {"base_net": self.base_net.state_dict(), "delta_net": self.delta_net.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.base_net.load_state_dict(state["base_net"])
        self.delta_net.load_state_dict(state["delta_net"])
        self.learning_rate = float(self.base_net.learning_rate)
        self.network_params_delta_qp = self.delta_net.weights + self.delta_net.biases
        self.network_params_qp = self.base_net.weights + self.base_net.biases


class _CriticQNetwork:
    def __init__(self, state_aug_dim, action_dim, learning_rate, seed):
        self.state_aug_dim = state_aug_dim
        self.action_dim = action_dim
        self.net = _NumpyMLP(
            state_aug_dim + action_dim,
            hidden_dims=(128, 128),
            output_dim=1,
            learning_rate=learning_rate,
            output_activation="linear",
            seed=seed,
        )

    def _input(self, X, action):
        return np.concatenate((_as_2d(X), _as_2d(action)), axis=-1)

    def predict(self, X, action):
        return self.net.forward(self._input(X, action))

    def train(self, X, action, target):
        return self.net.train_mse(self._input(X, action), target)

    def loss(self, X, action, target):
        pred = self.predict(X, action)
        return float(np.mean(np.square(pred - _as_2d(target))))

    def action_gradient(self, X, action):
        model_input = self._input(X, action)
        grad_input = self.net.input_gradient(model_input, np.ones((len(model_input), 1)))
        return grad_input[:, -self.action_dim :]

    def copy_from(self, other: "_CriticQNetwork") -> None:
        self.net.copy_from(other.net)

    def soft_update_from(self, other: "_CriticQNetwork", tau: float = TAU) -> None:
        self.net.soft_update_from(other.net, tau)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.net.load_state_dict(state)


class CriticNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, scope_D, scope_P):
        self.sess = sess
        self.s_dim = state_dim + 1
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.scope_D = scope_D
        self.scope_P = scope_P

        # Create the dual critic networks. Names are kept from the original:
        # q_value_D = quality/distortion critic, q_value_P = size/rate critic.
        self.critic_D = _CriticQNetwork(self.s_dim, action_dim, learning_rate, seed=23)
        self.critic_P = _CriticQNetwork(self.s_dim, action_dim, learning_rate, seed=29)
        self.q_value_D = None
        self.q_value_P = None
        self.network_params_D = self.critic_D.net.weights + self.critic_D.net.biases
        self.network_params_P = self.critic_P.net.weights + self.critic_P.net.biases

    def create_critic_network(self, scope):
        # Kept for naming continuity with the TensorFlow original.
        return f"{scope}_q_value"

    def get_loss(self, X, action, target_q_value_D, target_q_value_P):
        loss_D = self.critic_D.loss(X, action, target_q_value_D)
        loss_P = self.critic_P.loss(X, action, target_q_value_P)
        return loss_D, loss_P, loss_D + loss_P

    def train(self, X, action, target_q_value_D, target_q_value_P):
        loss_D = self.critic_D.train(X, action, target_q_value_D)
        loss_P = self.critic_P.train(X, action, target_q_value_P)
        return loss_D + loss_P

    def predict(self, X, action):
        self.q_value_D = self.critic_D.predict(X, action)
        self.q_value_P = self.critic_P.predict(X, action)
        return self.q_value_D, self.q_value_P

    def action_gradients(self, X, action):
        action_grads_P = self.critic_P.action_gradient(X, action)
        action_grads_D = self.critic_D.action_gradient(X, action)
        return [action_grads_P], [action_grads_D]

    def get_scope_D(self):
        return self.scope_D

    def get_scope_P(self):
        return self.scope_P

    def get_params_D(self):
        return self.network_params_D

    def get_params_P(self):
        return self.network_params_P

    def update_lr(self, lr):
        self.learning_rate = lr
        self.critic_D.net.learning_rate = lr
        self.critic_P.net.learning_rate = lr

    def copy_from(self, other: "CriticNetwork") -> None:
        self.critic_D.copy_from(other.critic_D)
        self.critic_P.copy_from(other.critic_P)

    def soft_update_D_from(self, other: "CriticNetwork", tau: float = TAU) -> None:
        self.critic_D.soft_update_from(other.critic_D, tau)

    def soft_update_P_from(self, other: "CriticNetwork", tau: float = TAU) -> None:
        self.critic_P.soft_update_from(other.critic_P, tau)

    def state_dict(self):
        return {"critic_D": self.critic_D.state_dict(), "critic_P": self.critic_P.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.critic_D.load_state_dict(state["critic_D"])
        self.critic_P.load_state_dict(state["critic_P"])
        self.learning_rate = float(self.critic_D.net.learning_rate)
        self.network_params_D = self.critic_D.net.weights + self.critic_D.net.biases
        self.network_params_P = self.critic_P.net.weights + self.critic_P.net.biases


def copy_weights_ops_actor(sess, to_net, from_net):
    to_net.copy_from(from_net)
    return ["copy_actor_weights"]


def copy_weights_ops_critic_D(sess, to_net, from_net):
    to_net.critic_D.copy_from(from_net.critic_D)
    return ["copy_critic_D_weights"]


def copy_weights_ops_critic_P(sess, to_net, from_net):
    to_net.critic_P.copy_from(from_net.critic_P)
    return ["copy_critic_P_weights"]


def soft_update_ops_actor(sess, target_net, behavior_net):
    target_net.soft_update_from(behavior_net, TAU)
    return ["soft_update_actor"]


def soft_update_ops_critic_D(sess, target_net, behavior_net):
    target_net.soft_update_D_from(behavior_net, TAU)
    return ["soft_update_critic_D"]


def soft_update_ops_critic_P(sess, target_net, behavior_net):
    target_net.soft_update_P_from(behavior_net, TAU)
    return ["soft_update_critic_P"]
