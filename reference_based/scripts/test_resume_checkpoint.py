"""Smoke test that a reference_based checkpoint restores all networks."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Network_GS as Network
from trainRDO_GS import ACTION_BOUND, ACTION_DIM, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to reference_based_epXXXX.pkl")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    with checkpoint_path.open("rb") as fp:
        raw = pickle.load(fp)
    state_dim = int(raw["b_actor"]["base_net"]["weights"][0].shape[0])

    b_actor = Network.ActorNetwork(None, state_dim, ACTION_DIM, ACTION_BOUND, 1e-4, ["b_base", "b_delta"])
    t_actor = Network.ActorNetwork(None, state_dim, ACTION_DIM, ACTION_BOUND, 1e-4, ["t_base", "t_delta"])
    b_critic = Network.CriticNetwork(None, state_dim, ACTION_DIM, 1e-3, "b_D", "b_P")
    t_critic = Network.CriticNetwork(None, state_dim, ACTION_DIM, 1e-3, "t_D", "t_P")

    loaded = load_checkpoint(checkpoint_path, b_actor, b_critic, t_actor, t_critic)
    assert int(loaded.get("episode", 0)) >= 1, loaded.keys()
    assert "global_step" in loaded, loaded.keys()
    assert b_actor.base_net.adam.t == raw["b_actor"]["base_net"]["adam"]["t"]
    assert b_critic.critic_D.net.adam.t == raw["b_critic"]["critic_D"]["adam"]["t"]

    print(
        "Checkpoint resume smoke test passed: "
        f"episode={loaded.get('episode')} global_step={loaded.get('global_step')}"
    )


if __name__ == "__main__":
    main()
