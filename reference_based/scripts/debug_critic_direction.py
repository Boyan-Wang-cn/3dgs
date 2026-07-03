from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = SCRIPT_DIR.parent
if str(REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(REFERENCE_DIR))

import Network_GS as Network
from compression_ops import decode_action
from trainRDO_GS import ACTION_BOUND, ACTION_DIM, CRITIC_LEARNING_RATE, SIZE_THRESHOLD, _state_with_base


def _profile(action_value: float) -> str:
    decoded = decode_action(action_value)
    return (
        f"p{decoded.pruning_level}_prec{decoded.precision_level}"
        f"_r{decoded.pruning_rate}_sh{decoded.sh_degree}"
        f"_sb{decoded.sh_bit}_gb{decoded.geo_bit}"
    )


def _load_samples(checkpoint: dict, max_samples: int):
    samples = checkpoint.get("replay_memory_actor") or checkpoint.get("replay_memory") or []
    if max_samples > 0:
        samples = samples[-max_samples:]
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect local Q_D/Q_P action direction around actor outputs."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--delta", type=float, default=0.5)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    with checkpoint_path.open("rb") as fp:
        checkpoint = pickle.load(fp)

    samples = _load_samples(checkpoint, args.max_samples)
    if not samples:
        raise SystemExit("No replay samples found in checkpoint.")

    state_dim = int(np.asarray(samples[0].state).reshape(-1).shape[0])
    actor = Network.ActorNetwork(
        None,
        state_dim,
        ACTION_DIM,
        ACTION_BOUND,
        learning_rate=1e-4,
        scope=["debug_baseLevel", "debug_deltaLevel"],
    )
    critic = Network.CriticNetwork(
        None,
        state_dim,
        ACTION_DIM,
        CRITIC_LEARNING_RATE,
        scope_D="debug_quality_D",
        scope_P="debug_size_P",
    )
    actor.load_state_dict(checkpoint["b_actor"])
    critic.load_state_dict(checkpoint["b_critic"])

    fieldnames = [
        "sample_idx",
        "left_bitbudget",
        "selected_source",
        "a_minus",
        "a_mid",
        "a_plus",
        "level_minus",
        "level_mid",
        "level_plus",
        "profile_minus",
        "profile_mid",
        "profile_plus",
        "Q_D_minus",
        "Q_D_mid",
        "Q_D_plus",
        "Q_P_minus",
        "Q_P_mid",
        "Q_P_plus",
        "grad_D",
        "grad_P",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()

    for sample_idx, sample in enumerate(samples):
        state = np.asarray(sample.state, dtype=np.float64).reshape(1, -1)
        action_mid, _, base_qp = actor.predict_action(state)
        a_mid = float(action_mid[0][0])
        a_minus = float(np.clip(a_mid - float(args.delta), 0.0, ACTION_BOUND[0]))
        a_plus = float(np.clip(a_mid + float(args.delta), 0.0, ACTION_BOUND[0]))
        x = _state_with_base(state, base_qp.reshape(-1))

        q_d_minus, q_p_minus = critic.predict(x, np.asarray([[a_minus]], dtype=np.float64))
        q_d_mid, q_p_mid = critic.predict(x, np.asarray([[a_mid]], dtype=np.float64))
        q_d_plus, q_p_plus = critic.predict(x, np.asarray([[a_plus]], dtype=np.float64))
        action_grads_p, action_grads_d = critic.action_gradients(
            x, np.asarray([[a_mid]], dtype=np.float64)
        )

        level_minus = decode_action(a_minus).action_id
        level_mid = decode_action(a_mid).action_id
        level_plus = decode_action(a_plus).action_id
        left_bitbudget = float(sample.left_bitbudget)
        writer.writerow(
            {
                "sample_idx": sample_idx,
                "left_bitbudget": left_bitbudget,
                "selected_source": "P" if left_bitbudget < SIZE_THRESHOLD else "D",
                "a_minus": f"{a_minus:.6f}",
                "a_mid": f"{a_mid:.6f}",
                "a_plus": f"{a_plus:.6f}",
                "level_minus": int(level_minus),
                "level_mid": int(level_mid),
                "level_plus": int(level_plus),
                "profile_minus": _profile(a_minus),
                "profile_mid": _profile(a_mid),
                "profile_plus": _profile(a_plus),
                "Q_D_minus": f"{float(q_d_minus[0][0]):.8f}",
                "Q_D_mid": f"{float(q_d_mid[0][0]):.8f}",
                "Q_D_plus": f"{float(q_d_plus[0][0]):.8f}",
                "Q_P_minus": f"{float(q_p_minus[0][0]):.8f}",
                "Q_P_mid": f"{float(q_p_mid[0][0]):.8f}",
                "Q_P_plus": f"{float(q_p_plus[0][0]):.8f}",
                "grad_D": f"{float(action_grads_d[0][0][0]):.8f}",
                "grad_P": f"{float(action_grads_p[0][0][0]):.8f}",
            }
        )


if __name__ == "__main__":
    main()
