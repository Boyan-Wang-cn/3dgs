from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_utils import (
    get_scene,
    load_pipeline_config,
    make_env_from_config,
    run_action_sequence,
)


def main():
    parser = argparse.ArgumentParser(description="Test terminal reward pipeline without RL training.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--use-render", action="store_true")
    parser.add_argument("--use-crossscore", action="store_true")
    parser.add_argument("--allow-crossscore-placeholder", action="store_true")
    parser.add_argument("--force-recompute-original-score", action="store_true")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    scene = get_scene(config, args.scene)
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
    actions = [args.level for _ in range(env.frameNum)]
    info = run_action_sequence(env, scene, actions, reset=False)
    print(json.dumps(info, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
