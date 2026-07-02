"""Shared helpers for non-training pipeline and baseline scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from .config_utils import (
        fallback_flat_scene_ply,
        load_config,
        normalize_crossscore_dir,
        normalize_gaussian_splatting_dir,
        resolve_input_path,
        resolve_output_path,
        select_scenes,
    )
    from .Environment_GS import GS_Environment
except ImportError:
    from config_utils import (
        fallback_flat_scene_ply,
        load_config,
        normalize_crossscore_dir,
        normalize_gaussian_splatting_dir,
        resolve_input_path,
        resolve_output_path,
        select_scenes,
    )
    from Environment_GS import GS_Environment


def load_pipeline_config(config_path):
    return load_config(config_path)


def get_scene(config: dict[str, Any], scene_name: str) -> dict[str, Any]:
    scenes = [fallback_flat_scene_ply(scene) for scene in select_scenes(config, scene_name)]
    scenes = [scene for scene in scenes if Path(scene["ply_path"]).exists()]
    if not scenes:
        raise FileNotFoundError(
            f"No valid PLY found for scene '{scene_name}'. Update config or create model_path layout."
        )
    return scenes[0]


def make_env_from_config(
    config: dict[str, Any],
    scene: dict[str, Any],
    use_dummy_reward: bool | None = None,
    use_render: bool | None = None,
    use_crossscore: bool | None = None,
    allow_crossscore_placeholder: bool = False,
    force_recompute_original_score: bool = False,
    output_dir=None,
    target_num_groups=None,
    max_groups=None,
    grid_size=None,
    max_search_grid_size=None,
):
    env_cfg = config.get("env", {})
    paths_cfg = config.get("paths", {})
    render_cfg = config.get("render", {})
    quality_cfg = config.get("quality", {})
    crossscore_cfg = config.get("crossscore", {})
    project_cfg = config.get("project", {})
    output_root = (
        Path(output_dir)
        if output_dir is not None
        else resolve_output_path(project_cfg.get("output_dir", "./outputs"), config)
    )
    gaussian_splatting_dir = normalize_gaussian_splatting_dir(
        resolve_input_path(
            render_cfg.get(
                "gaussian_splatting_dir",
                paths_cfg.get("gaussian_splatting_dir", "./gaussian-splatting-main"),
            ),
            config,
        )
    )
    crossscore_dir = normalize_crossscore_dir(
        resolve_input_path(paths_cfg.get("crossscore_dir", "./CrossScore-main"), config)
    )

    if use_dummy_reward is None:
        use_dummy_reward = bool(env_cfg.get("use_dummy_reward", True))
    if use_render is None:
        use_render = bool(env_cfg.get("use_render", False))
    if use_crossscore is None:
        use_crossscore = bool(env_cfg.get("use_crossscore", False))

    return GS_Environment(
        scenes=[scene],
        output_root=output_root,
        grid_size=int(grid_size if grid_size is not None else env_cfg.get("grid_size", 4)),
        target_size_ratio=float(env_cfg.get("target_size_ratio", 0.3)),
        max_groups=max_groups if max_groups is not None else env_cfg.get("max_groups"),
        target_num_groups=target_num_groups if target_num_groups is not None else env_cfg.get("target_num_groups", 128),
        max_search_grid_size=int(max_search_grid_size if max_search_grid_size is not None else env_cfg.get("max_search_grid_size", 32)),
        min_group_size=int(env_cfg.get("min_group_size", 10)),
        use_dummy_reward=use_dummy_reward,
        use_render=use_render,
        use_crossscore=use_crossscore,
        gaussian_splatting_dir=gaussian_splatting_dir,
        crossscore_dir=crossscore_dir,
        iteration=int(render_cfg.get("iteration", env_cfg.get("iteration", 30000))),
        resolution=int(render_cfg.get("resolution", env_cfg.get("resolution", 4))),
        quality_cache_dir=output_root / "crossscore_cache",
        cache_original_score=bool(quality_cfg.get("cache_original_score", True)),
        score_higher_is_better=bool(quality_cfg.get("score_higher_is_better", True)),
        crossscore_mode=str(quality_cfg.get("crossscore_mode", "placeholder")),
        quality_epsilon=float(quality_cfg.get("epsilon", 0.0)),
        crossscore_command_template=str(crossscore_cfg.get("command_template", "") or ""),
        crossscore_score_output=str(crossscore_cfg.get("score_output", "") or ""),
        crossscore_score_parse_mode=str(crossscore_cfg.get("score_parse_mode", "auto") or "auto"),
        crossscore_preferred_score_key=str(
            crossscore_cfg.get("preferred_score_key", "pred_ssim_0_1") or "pred_ssim_0_1"
        ),
        render_python_executable=str(render_cfg.get("python_executable", "python") or "python"),
        crossscore_python_executable=str(crossscore_cfg.get("python_executable", "python") or "python"),
        crossscore_ckpt=crossscore_cfg.get("ckpt") or None,
        crossscore_config=crossscore_cfg.get("config") or None,
        crossscore_allow_image_fallback=bool(crossscore_cfg.get("allow_image_fallback", False)),
        allow_crossscore_placeholder=bool(allow_crossscore_placeholder),
        force_recompute_original_score=bool(force_recompute_original_score),
        terminal_reward_only=bool(env_cfg.get("terminal_reward_only", True)),
    )


def run_action_sequence(
    env: GS_Environment,
    scene: dict[str, Any],
    actions,
    reset: bool = True,
) -> dict[str, Any]:
    state = env.reset(scene) if reset else env.getObservation(env.current_group_idx)
    _ = state
    done = False
    info = {}
    reward_D = 0.0
    reward_P = 0.0
    action_list = list(actions)
    idx = 0
    while not done:
        action = action_list[idx] if idx < len(action_list) else action_list[-1]
        _, reward_D, reward_P, done, info = env.step(action, baseQP=action)
        idx += 1
    info["reward_D"] = float(reward_D)
    info["reward_P"] = float(reward_P)
    info["num_gaussians_original"] = int(env.ply.vertex_count)
    info["num_gaussians_compressed"] = int(info.get("compression_stats", {}).get("kept_vertices", 0))
    return info


def run_fixed_level(
    config: dict[str, Any],
    scene_name: str,
    level: int,
    use_render: bool = False,
    use_crossscore: bool = False,
    allow_crossscore_placeholder: bool = False,
    force_recompute_original_score: bool = False,
    output_dir=None,
    target_num_groups=None,
    max_groups=None,
    grid_size=None,
    max_search_grid_size=None,
):
    scene = get_scene(config, scene_name)
    env = make_env_from_config(
        config,
        scene,
        use_dummy_reward=not (use_render or use_crossscore),
        use_render=use_render or use_crossscore,
        use_crossscore=use_crossscore,
        allow_crossscore_placeholder=allow_crossscore_placeholder,
        force_recompute_original_score=force_recompute_original_score,
        output_dir=output_dir,
    )
    env.reset(scene)
    actions = [int(level) for _ in range(env.frameNum)]
    return run_action_sequence(env, scene, actions, reset=False)


def random_actions(num_groups: int, rng: np.random.Generator):
    return rng.integers(0, 25, size=int(num_groups)).astype(int).tolist()
