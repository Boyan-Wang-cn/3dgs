from __future__ import annotations

import argparse
import random
from pathlib import Path

from .env import GSCompressionEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the environment with random actions.")
    parser.add_argument("--ply", required=True, help="Path to a 3DGS .ply file.")
    parser.add_argument("--out", required=True, help="Output directory for compressed PLY.")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--target-size-ratio", type=float, default=0.3)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    env = GSCompressionEnv(
        ply_path=args.ply,
        output_dir=Path(args.out),
        grid_size=args.grid_size,
        target_size_ratio=args.target_size_ratio,
        max_groups=args.max_groups,
    )
    state = env.reset()
    done = False
    info = {}
    while not done:
        action = random.randint(0, 4)
        state, reward_quality, reward_size, done, info = env.step(action)

    print(f"state_dim={state.shape[0]}")
    print(f"reward_quality={reward_quality:.6f}")
    print(f"reward_size={reward_size:.6f}")
    print(f"size_ratio={info['size_ratio']:.6f}")
    print(f"compressed_ply_path={info['compressed_ply_path']}")


if __name__ == "__main__":
    main()
