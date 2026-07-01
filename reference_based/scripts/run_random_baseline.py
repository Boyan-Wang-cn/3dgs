from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import resolve_output_path
from pipeline_utils import (
    get_scene,
    load_pipeline_config,
    make_env_from_config,
    random_actions,
    run_action_sequence,
)


FIELDS = [
    "scene",
    "method",
    "trial",
    "mean_action",
    "action_histogram",
    "size_ratio",
    "original_size",
    "compressed_size",
    "num_gaussians_original",
    "num_gaussians_compressed",
    "quality_mode",
    "reward_mode",
    "reward_D",
    "reward_P",
    "original_score",
    "compressed_score",
    "quality_drop",
    "crossscore_is_placeholder",
    "original_render_dir",
    "compressed_render_dir",
    "reference_dir",
    "compressed_model_dir",
]


def main():
    parser = argparse.ArgumentParser(description="Run random-action 3DGS compression baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-render", action="store_true")
    parser.add_argument("--use-crossscore", action="store_true")
    parser.add_argument("--allow-crossscore-placeholder", action="store_true")
    parser.add_argument("--force-recompute-original-score", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    config = load_pipeline_config(args.config)
    scene = get_scene(config, args.scene)
    output_dir = resolve_output_path("outputs/baselines", config)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"random_{args.scene}.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        for trial in range(1, args.trials + 1):
            env = make_env_from_config(
                config,
                scene,
                use_dummy_reward=not (args.use_render or args.use_crossscore),
                use_render=args.use_render or args.use_crossscore,
                use_crossscore=args.use_crossscore,
                allow_crossscore_placeholder=args.allow_crossscore_placeholder,
                force_recompute_original_score=args.force_recompute_original_score,
            )
            env.reset(scene)
            actions = random_actions(env.frameNum, rng)
            histogram = {str(level): actions.count(level) for level in range(5)}
            info = run_action_sequence(env, scene, actions, reset=False)
            row = {
                "scene": args.scene,
                "method": "random",
                "trial": trial,
                "mean_action": float(np.mean(actions)),
                "action_histogram": json.dumps(histogram),
                "size_ratio": info.get("size_ratio"),
                "original_size": info.get("original_size"),
                "compressed_size": info.get("compressed_size"),
                "num_gaussians_original": info.get("num_gaussians_original"),
                "num_gaussians_compressed": info.get("num_gaussians_compressed"),
                "quality_mode": info.get("quality_mode", ""),
                "reward_mode": info.get("reward_mode", ""),
                "reward_D": info.get("reward_D"),
                "reward_P": info.get("reward_P"),
                "original_score": info.get("original_score", ""),
                "compressed_score": info.get("compressed_score", ""),
                "quality_drop": info.get("quality_drop", ""),
                "crossscore_is_placeholder": info.get("crossscore_is_placeholder", ""),
                "original_render_dir": info.get("original_render_dir", ""),
                "compressed_render_dir": info.get("compressed_render_dir", ""),
                "reference_dir": info.get("reference_dir", ""),
                "compressed_model_dir": info.get("compressed_model_dir"),
            }
            writer.writerow(row)
            print(row)
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
