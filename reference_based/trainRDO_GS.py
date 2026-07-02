"""This file is adapted from Train/DRL_x265_TRAIN/trainRDO.py.

The x265 socket and encoder calls are replaced by GS_Environment.step(), but
the training skeleton remains Dual-Critic/DDPG-like:
- behavior actor / critic and target actor / critic.
- replay memory random sampling.
- target Q updates for q_value_D and q_value_P.
- actor update through critic action gradients.
- left_bitbudget decides whether the actor follows size critic P or quality
  critic D.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from pathlib import Path

import numpy as np

try:
    from .config_utils import (
        CODE_ROOT,
        load_config,
        normalize_crossscore_dir,
        normalize_gaussian_splatting_dir,
        resolve_input_path,
        resolve_output_path,
        select_scenes,
    )
    from . import Network_GS as Network
    from . import Transition_GS as trans
    from .Environment_GS import GS_Environment
except ImportError:
    from config_utils import (
        CODE_ROOT,
        load_config,
        normalize_crossscore_dir,
        normalize_gaussian_splatting_dir,
        resolve_input_path,
        resolve_output_path,
        select_scenes,
    )
    import Network_GS as Network
    import Transition_GS as trans
    from Environment_GS import GS_Environment


GAMMA = 0.99
BATCH_SIZE = 8
BUFFER_SIZE = 5000
UPDATE_STEP = 4
SIZE_THRESHOLD = 0.0
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3
ACTION_BOUND = [24.0, 4.0]
ACTION_DIM = 1


def default_scenes():
    return [
        {
            "name": "train",
            "source_path": str(CODE_ROOT / "data" / "tandt" / "train"),
            "model_path": str(CODE_ROOT / "data" / "gs_models" / "train_original"),
            "ply_path": str(CODE_ROOT / "data" / "train.ply"),
        },
        {
            "name": "truck",
            "source_path": str(CODE_ROOT / "data" / "tandt" / "truck"),
            "model_path": str(CODE_ROOT / "data" / "gs_models" / "truck_original"),
            "ply_path": str(CODE_ROOT / "data" / "truck.ply"),
        },
    ]


def _fallback_flat_ply(scene):
    ply_path = Path(scene.get("ply_path", ""))
    if ply_path.exists():
        return scene
    fallback = CODE_ROOT / "data" / f"{scene.get('name', ply_path.stem)}.ply"
    if fallback.exists():
        scene = dict(scene)
        scene["ply_path"] = str(fallback)
    return scene


def _state_with_base(state_batch, baseQP_batch):
    return np.concatenate((state_batch, (baseQP_batch / 24.0).reshape(-1, 1)), axis=-1)


def _safe_normalize_pair(grad_d, grad_p):
    eps = 1e-8
    if abs(float(grad_d)) > abs(float(grad_p)):
        return grad_d, grad_p * (abs(float(grad_d)) / (abs(float(grad_p)) + eps))
    return grad_d * (abs(float(grad_p)) / (abs(float(grad_d)) + eps)), grad_p


def update_from_replay(
    b_actor,
    b_critic,
    t_actor,
    t_critic,
    memory_train,
    memory_train_actor,
    logs_dir,
):
    critic_loss_D_tmp_print = []
    critic_loss_P_tmp_print = []
    actor_update_count_P = 0
    actor_update_count_D = 0

    if len(memory_train.replayMemory) < BATCH_SIZE:
        return None

    for update_idx in range(UPDATE_STEP):
        samples = memory_train.sample_batch(BATCH_SIZE)
        (
            s_batch,
            a_batch,
            baseQP_batch,
            nextQP_batch,
            r_D_batch,
            r_P_batch,
            s2_batch,
            t_batch,
            left_bitbudget_batch,
        ) = map(np.array, zip(*samples))

        target_action_batch, _, target_baseQP_batch = t_actor.predict_action(s2_batch)
        target_q_value_D, target_q_value_P = t_critic.predict(
            _state_with_base(s2_batch, target_baseQP_batch.reshape(-1)),
            target_action_batch,
        )

        y_i_D = []
        y_i_P = []
        for k in range(BATCH_SIZE):
            if t_batch[k]:
                y_i_D.append([r_D_batch[k]])
                y_i_P.append([r_P_batch[k]])
            else:
                if target_q_value_D[k][0] > 0:
                    target_q_value_D[k][0] = 0
                if target_q_value_P[k][0] > 0:
                    target_q_value_P[k][0] = 0
                y_i_D.append(r_D_batch[k] + GAMMA * target_q_value_D[k])
                y_i_P.append(r_P_batch[k] + GAMMA * target_q_value_P[k])

        b_critic.train(
            _state_with_base(s_batch, baseQP_batch),
            np.asarray(a_batch, dtype=np.float64).reshape(-1, 1),
            np.asarray(y_i_D, dtype=np.float64).reshape(-1, 1),
            np.asarray(y_i_P, dtype=np.float64).reshape(-1, 1),
        )
        predicted_critic_loss_D, predicted_critic_loss_P, predicted_critic_loss = (
            b_critic.get_loss(
                _state_with_base(s_batch, baseQP_batch),
                np.asarray(a_batch, dtype=np.float64).reshape(-1, 1),
                np.asarray(y_i_D, dtype=np.float64).reshape(-1, 1),
                np.asarray(y_i_P, dtype=np.float64).reshape(-1, 1),
            )
        )
        _ = predicted_critic_loss
        critic_loss_D_tmp_print.append(predicted_critic_loss_D)
        critic_loss_P_tmp_print.append(predicted_critic_loss_P)

        Network.soft_update_ops_critic_D(None, t_critic, b_critic)
        Network.soft_update_ops_critic_P(None, t_critic, b_critic)

        if update_idx == UPDATE_STEP - 1 and len(memory_train_actor.replayMemory) >= BATCH_SIZE:
            samples_actor = memory_train_actor.sample_batch(BATCH_SIZE)
            (
                s_batch_actor,
                _,
                baseQP_batch_actor,
                _nextQP_batch_actor,
                _r_D_batch_actor,
                _r_P_batch_actor,
                _s2_batch_actor,
                _t_batch_actor,
                left_bitbudget_batch_actor,
            ) = map(np.array, zip(*samples_actor))

            predict_action_batch, _, baseQP_predict_batch = b_actor.predict_action(s_batch_actor)
            action_grads_P, action_grads_D = b_critic.action_gradients(
                _state_with_base(s_batch_actor, baseQP_predict_batch.reshape(-1)),
                predict_action_batch,
            )

            mixed_action_grads = []
            grad_rows = []
            for k in range(BATCH_SIZE):
                grad_P = float(action_grads_P[0][k][0])
                grad_D = float(action_grads_D[0][k][0])
                grad_D, grad_P = _safe_normalize_pair(grad_D, grad_P)
                if left_bitbudget_batch_actor[k] < SIZE_THRESHOLD:
                    action_grads_beta = 1.0
                    actor_update_count_P += 1
                else:
                    action_grads_beta = 0.0
                    actor_update_count_D += 1
                action_grads_D_P = (1.0 - action_grads_beta) * grad_D + action_grads_beta * grad_P
                mixed_action_grads.append(action_grads_D_P)
                grad_rows.append(
                    {
                        "left_bitbudget": float(left_bitbudget_batch_actor[k]),
                        "use_size_P": action_grads_beta,
                        "grad_P": grad_P,
                        "grad_D": grad_D,
                        "mixed_grad": action_grads_D_P,
                    }
                )

            b_actor.train(s_batch_actor, np.asarray(mixed_action_grads).reshape(-1, 1))
            Network.soft_update_ops_actor(None, t_actor, b_actor)

            grad_log = logs_dir / "actor_gradient_log.csv"
            write_header = not grad_log.exists()
            with grad_log.open("a", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(
                    fp,
                    fieldnames=["left_bitbudget", "use_size_P", "grad_P", "grad_D", "mixed_grad"],
                )
                if write_header:
                    writer.writeheader()
                writer.writerows(grad_rows)

    return {
        "critic_loss_D": float(np.mean(critic_loss_D_tmp_print)),
        "critic_loss_P": float(np.mean(critic_loss_P_tmp_print)),
        "actor_update_count_P": actor_update_count_P,
        "actor_update_count_D": actor_update_count_D,
    }


def save_checkpoint(
    checkpoints_dir,
    b_actor,
    b_critic,
    t_actor,
    t_critic,
    episode_id,
    global_step,
    config,
    memory_train=None,
    memory_train_actor=None,
):
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 2,
        "episode": episode_id,
        "global_step": int(global_step),
        "b_actor": b_actor.state_dict(),
        "b_critic": b_critic.state_dict(),
        "t_actor": t_actor.state_dict(),
        "t_critic": t_critic.state_dict(),
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "config": config,
        "replay_memory": list(memory_train.replayMemory) if memory_train is not None else [],
        "replay_memory_actor": (
            list(memory_train_actor.replayMemory) if memory_train_actor is not None else []
        ),
    }
    path = checkpoints_dir / f"reference_based_ep{episode_id:04d}.pkl"
    with path.open("wb") as fp:
        pickle.dump(checkpoint, fp)
    return path


def load_checkpoint(checkpoint_path, b_actor, b_critic, t_actor, t_critic):
    checkpoint_path = Path(checkpoint_path)
    with checkpoint_path.open("rb") as fp:
        checkpoint = pickle.load(fp)
    b_actor.load_state_dict(checkpoint["b_actor"])
    b_critic.load_state_dict(checkpoint["b_critic"])
    t_actor.load_state_dict(checkpoint.get("t_actor", checkpoint["b_actor"]))
    t_critic.load_state_dict(checkpoint.get("t_critic", checkpoint["b_critic"]))
    if "random_state" in checkpoint:
        random.setstate(checkpoint["random_state"])
    if "numpy_random_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_random_state"])
    return checkpoint


def train(args):
    global GAMMA, BATCH_SIZE
    random.seed(args.seed)
    np.random.seed(args.seed)

    project_dir = Path(__file__).resolve().parent
    config = load_config(args.config)
    training_cfg = config.get("training", {})
    env_cfg = config.get("env", {})
    paths_cfg = config.get("paths", {})
    render_cfg = config.get("render", {})
    quality_cfg = config.get("quality", {})
    crossscore_cfg = config.get("crossscore", {})
    project_cfg = config.get("project", {})

    GAMMA = float(args.gamma if args.gamma is not None else training_cfg.get("gamma", GAMMA))
    BATCH_SIZE = int(args.batch_size if args.batch_size is not None else training_cfg.get("batch_size", BATCH_SIZE))

    checkpoints_dir = resolve_output_path(
        project_cfg.get("checkpoint_dir", "./checkpoints"), config
    )
    logs_dir = resolve_output_path(project_cfg.get("log_dir", "./logs"), config)
    outputs_dir = resolve_output_path(project_cfg.get("output_dir", "./outputs"), config)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    scenes = select_scenes(config, args.scene)
    if args.ply:
        scenes = [
            {
                "name": Path(args.ply).stem,
                "source_path": args.scene_path,
                "model_path": args.model_path,
                "ply_path": str(resolve_input_path(args.ply, config)),
            }
        ]
    scenes = [_fallback_flat_ply(scene) for scene in scenes]
    scenes = [scene for scene in scenes if Path(scene["ply_path"]).exists()]
    if not scenes:
        raise FileNotFoundError(
            "No valid PLY path was found for training. Update configs/default.yaml "
            "or pass --ply explicitly."
        )

    use_dummy_reward = bool(env_cfg.get("use_dummy_reward", True))
    use_render = bool(env_cfg.get("use_render", False))
    use_crossscore = bool(env_cfg.get("use_crossscore", False))
    if args.use_dummy_reward:
        use_dummy_reward = True
        use_render = False
        use_crossscore = False
    if args.use_render:
        use_dummy_reward = False
        use_render = True
        use_crossscore = False
    if args.use_crossscore:
        use_dummy_reward = False
        use_render = True
        use_crossscore = True

    gaussian_splatting_dir = resolve_input_path(
        render_cfg.get(
            "gaussian_splatting_dir",
            paths_cfg.get("gaussian_splatting_dir", "./gaussian-splatting-main"),
        ),
        config,
    )
    crossscore_dir = resolve_input_path(
        paths_cfg.get("crossscore_dir", "./CrossScore-main"), config
    )
    gaussian_splatting_dir = normalize_gaussian_splatting_dir(gaussian_splatting_dir)
    crossscore_dir = normalize_crossscore_dir(crossscore_dir)
    render_python_executable = str(render_cfg.get("python_executable", "python") or "python")
    render_iteration = int(render_cfg.get("iteration", env_cfg.get("iteration", 30000)))
    render_resolution = int(render_cfg.get("resolution", env_cfg.get("resolution", 4)))
    max_groups = args.max_groups if args.max_groups is not None else env_cfg.get("max_groups")
    target_num_groups = (
        args.target_groups
        if args.target_groups is not None
        else env_cfg.get("target_num_groups", 128)
    )
    if target_num_groups is not None:
        target_num_groups = int(target_num_groups)
    max_search_grid_size = int(
        args.max_search_grid_size
        if args.max_search_grid_size is not None
        else env_cfg.get("max_search_grid_size", 32)
    )
    min_group_size = int(env_cfg.get("min_group_size", 10))
    if max_groups is not None:
        print(
            "WARNING: max_groups is enabled. This is useful for debug/smoke tests, "
            "but it means the episode does not traverse every natural voxel group."
        )
    if target_num_groups is not None:
        print(
            f"Grouping target enabled: target_num_groups={target_num_groups}, "
            f"max_search_grid_size={max_search_grid_size}."
        )

    env = GS_Environment(
        scenes=scenes,
        output_root=outputs_dir,
        grid_size=args.grid_size if args.grid_size is not None else int(env_cfg.get("grid_size", 4)),
        target_size_ratio=args.target_size_ratio
        if args.target_size_ratio is not None
        else float(env_cfg.get("target_size_ratio", 0.3)),
        max_groups=max_groups,
        target_num_groups=target_num_groups,
        max_search_grid_size=max_search_grid_size,
        min_group_size=min_group_size,
        use_dummy_reward=use_dummy_reward,
        use_render=use_render,
        use_crossscore=use_crossscore,
        gaussian_splatting_dir=gaussian_splatting_dir,
        crossscore_dir=crossscore_dir,
        iteration=render_iteration,
        resolution=render_resolution,
        quality_cache_dir=outputs_dir / "crossscore_cache",
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
        render_python_executable=render_python_executable,
        crossscore_python_executable=str(crossscore_cfg.get("python_executable", "python") or "python"),
        crossscore_ckpt=crossscore_cfg.get("ckpt") or None,
        crossscore_config=crossscore_cfg.get("config") or None,
        crossscore_allow_image_fallback=bool(crossscore_cfg.get("allow_image_fallback", False)),
        allow_crossscore_placeholder=bool(args.allow_crossscore_placeholder),
        force_recompute_original_score=bool(args.force_recompute_original_score),
        terminal_reward_only=bool(env_cfg.get("terminal_reward_only", True)),
    )
    probe_state = env.reset(scenes[0])
    state_dim = int(probe_state.shape[0])
    env.episode = 0

    b_actor = Network.ActorNetwork(
        None,
        state_dim,
        ACTION_DIM,
        ACTION_BOUND,
        args.lr_actor,
        scope=["behavior_baseLevel", "behavior_deltaLevel"],
    )
    t_actor = Network.ActorNetwork(
        None,
        state_dim,
        ACTION_DIM,
        ACTION_BOUND,
        args.lr_actor,
        scope=["target_baseLevel", "target_deltaLevel"],
    )
    b_critic = Network.CriticNetwork(
        None,
        state_dim,
        ACTION_DIM,
        args.lr_critic,
        scope_D="behavior_quality_D",
        scope_P="behavior_size_P",
    )
    t_critic = Network.CriticNetwork(
        None,
        state_dim,
        ACTION_DIM,
        args.lr_critic,
        scope_D="target_quality_D",
        scope_P="target_size_P",
    )

    Network.copy_weights_ops_actor(None, t_actor, b_actor)
    Network.copy_weights_ops_critic_D(None, t_critic, b_critic)
    Network.copy_weights_ops_critic_P(None, t_critic, b_critic)

    memory_train = trans.reMemory("replay_memory_train_all_GS", BUFFER_SIZE)
    memory_train_actor = trans.reMemory("replay_memory_train_all_actor_GS", BUFFER_SIZE)
    resume_from = str(Path(args.resume).resolve()) if args.resume else ""
    resume_episode = 0
    global_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, b_actor, b_critic, t_actor, t_critic)
        resume_episode = int(checkpoint.get("episode", 0))
        global_step = int(checkpoint.get("global_step", 0))
        env.episode = resume_episode
        memory_train.replayMemory = list(checkpoint.get("replay_memory", []))
        memory_train_actor.replayMemory = list(checkpoint.get("replay_memory_actor", []))
        print(
            f"Resumed checkpoint: path={resume_from} "
            f"episode={resume_episode} global_step={global_step} "
            f"replay={len(memory_train.replayMemory)} "
            f"actor_replay={len(memory_train_actor.replayMemory)}"
        )

    episode_log = logs_dir / "train_episode_log.csv"
    log_fields = [
        "episode",
        "global_step",
        "resume_from",
        "scene",
        "num_groups",
        "target_num_groups",
        "actual_num_groups",
        "requested_grid_size",
        "grid_size",
        "natural_group_count",
        "truncated_by_max_groups",
        "max_groups",
        "mean_action",
        "mean_level",
        "level_histogram",
        "action_mode",
        "action_histogram",
        "pruning_level_histogram",
        "precision_level_histogram",
        "mean_pruning_rate",
        "mean_sh_degree",
        "mean_sh_bit",
        "mean_geo_bit",
        "compact_size_ratio",
        "render_ply_size_ratio",
        "estimated_size_ratio",
        "compact_size",
        "render_ply_size",
        "compact_package_path",
        "render_ply_path",
        "quality_mode",
        "reward_mode",
        "reward_D",
        "reward_P",
        "left_bitbudget",
        "size_ratio",
        "disk_size_ratio",
        "original_size",
        "compressed_size",
        "original_gaussians",
        "compressed_gaussians",
        "pruned_gaussians",
        "compressed_ply_path",
        "compressed_model_dir",
        "original_render_dir",
        "original_gt_dir",
        "compressed_render_dir",
        "compressed_gt_dir",
        "reference_dir",
        "original_score",
        "compressed_score",
        "quality_drop",
        "quality_epsilon",
        "penalized_quality_drop",
        "original_score_file",
        "original_score_key",
        "original_parser_mode",
        "original_preferred_score_key",
        "compressed_score_file",
        "compressed_score_key",
        "compressed_parser_mode",
        "compressed_preferred_score_key",
        "crossscore_is_placeholder",
        "critic_loss_D",
        "critic_loss_P",
        "actor_update_count_P",
        "actor_update_count_D",
        "actor_update_source",
        "checkpoint_path",
    ]
    log_mode = "a" if args.resume else "w"
    write_log_header = (not episode_log.exists()) or (not args.resume)
    with episode_log.open(log_mode, newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=log_fields,
        )
        if write_log_header:
            writer.writeheader()

    episodes = int(args.episodes or training_cfg.get("episodes", 10))
    start_episode = resume_episode + 1
    if start_episode > episodes:
        print(
            f"No training episodes to run: resume episode={resume_episode}, requested episodes={episodes}."
        )
        return
    for episode_id in range(start_episode, episodes + 1):
        scene = random.choice(scenes)
        observation = env.reset(scene)
        done = False
        final_info = {}
        final_reward_D = 0.0
        final_reward_P = 0.0
        episode_actor_update_count_P = 0
        episode_actor_update_count_D = 0
        last_update_info = None

        while not done:
            action_batch, delta_action_batch, baseQP_batch = b_actor.predict_action(
                observation.reshape(1, -1)
            )
            _ = delta_action_batch
            exploration = np.random.normal(0.0, args.exploration_std)
            action = float(np.clip(action_batch[0][0] + exploration, 0.0, 24.0))
            baseQP = float(baseQP_batch[0][0])

            next_observation, reward_D, reward_P, done, info = env.step(action, baseQP)
            global_step += 1
            next_action_batch, _next_delta_action_batch, nextBaseQP_batch = t_actor.predict_action(
                next_observation.reshape(1, -1)
            )
            _ = next_action_batch
            nextBaseQP = float(nextBaseQP_batch[0][0])
            left_bitbudget = float(info["left_bitbudget"])

            memory_train.appendTransition(
                observation,
                action,
                baseQP,
                nextBaseQP,
                reward_D,
                reward_P,
                next_observation,
                float(done),
                left_bitbudget,
            )
            memory_train_actor.appendTransition(
                observation,
                action,
                baseQP,
                nextBaseQP,
                reward_D,
                reward_P,
                next_observation,
                float(done),
                left_bitbudget,
            )

            update_info = update_from_replay(
                b_actor,
                b_critic,
                t_actor,
                t_critic,
                memory_train,
                memory_train_actor,
                logs_dir,
            )
            if update_info is not None:
                last_update_info = update_info
                episode_actor_update_count_P += int(update_info.get("actor_update_count_P", 0))
                episode_actor_update_count_D += int(update_info.get("actor_update_count_D", 0))

            observation = next_observation
            if done:
                final_info = info
                final_reward_D = reward_D
                final_reward_P = reward_P

        checkpoint_path = save_checkpoint(
            checkpoints_dir,
            b_actor,
            b_critic,
            t_actor,
            t_critic,
            episode_id,
            global_step,
            config,
            memory_train=memory_train,
            memory_train_actor=memory_train_actor,
        )
        critic_loss_D = last_update_info["critic_loss_D"] if last_update_info else 0.0
        critic_loss_P = last_update_info["critic_loss_P"] if last_update_info else 0.0

        row = {
            "episode": episode_id,
            "global_step": global_step,
            "resume_from": resume_from,
            "scene": scene.get("name", Path(scene["ply_path"]).stem),
            "num_groups": final_info.get("num_groups", 0),
            "target_num_groups": final_info.get("target_num_groups", ""),
            "actual_num_groups": final_info.get("actual_num_groups", final_info.get("num_groups", 0)),
            "requested_grid_size": final_info.get("requested_grid_size", ""),
            "grid_size": final_info.get("grid_size", ""),
            "natural_group_count": final_info.get("natural_group_count", ""),
            "truncated_by_max_groups": final_info.get("truncated_by_max_groups", ""),
            "max_groups": final_info.get("max_groups", ""),
            "mean_action": final_info.get("mean_action", 0.0),
            "mean_level": final_info.get("mean_level", final_info.get("mean_action", 0.0)),
            "level_histogram": json.dumps(final_info.get("level_histogram", {}), sort_keys=True),
            "action_mode": final_info.get("action_mode", ""),
            "action_histogram": json.dumps(final_info.get("action_histogram", {}), sort_keys=True),
            "pruning_level_histogram": json.dumps(final_info.get("pruning_level_histogram", {}), sort_keys=True),
            "precision_level_histogram": json.dumps(final_info.get("precision_level_histogram", {}), sort_keys=True),
            "mean_pruning_rate": final_info.get("mean_pruning_rate", ""),
            "mean_sh_degree": final_info.get("mean_sh_degree", ""),
            "mean_sh_bit": final_info.get("mean_sh_bit", ""),
            "mean_geo_bit": final_info.get("mean_geo_bit", ""),
            "compact_size_ratio": final_info.get("compact_size_ratio", ""),
            "render_ply_size_ratio": final_info.get("render_ply_size_ratio", ""),
            "estimated_size_ratio": final_info.get("estimated_size_ratio", ""),
            "compact_size": final_info.get("compact_size", ""),
            "render_ply_size": final_info.get("render_ply_size", ""),
            "compact_package_path": final_info.get("compact_package_path", ""),
            "render_ply_path": final_info.get("render_ply_path", ""),
            "quality_mode": final_info.get("quality_mode", ""),
            "reward_mode": final_info.get("reward_mode", ""),
            "reward_D": final_reward_D,
            "reward_P": final_reward_P,
            "left_bitbudget": final_info.get("left_bitbudget", 0.0),
            "size_ratio": final_info.get("size_ratio", 0.0),
            "disk_size_ratio": final_info.get("disk_size_ratio", final_info.get("size_ratio", 0.0)),
            "original_size": final_info.get("original_size", 0),
            "compressed_size": final_info.get("compressed_size", 0),
            "original_gaussians": final_info.get("original_gaussians", ""),
            "compressed_gaussians": final_info.get("compressed_gaussians", ""),
            "pruned_gaussians": final_info.get("pruned_gaussians", ""),
            "compressed_ply_path": final_info.get("compressed_ply_path", ""),
            "compressed_model_dir": final_info.get("compressed_model_dir", ""),
            "original_render_dir": final_info.get("original_render_dir", ""),
            "original_gt_dir": final_info.get("original_gt_dir", ""),
            "compressed_render_dir": final_info.get("compressed_render_dir", ""),
            "compressed_gt_dir": final_info.get("compressed_gt_dir", ""),
            "reference_dir": final_info.get("reference_dir", ""),
            "original_score": final_info.get("original_score", ""),
            "compressed_score": final_info.get("compressed_score", ""),
            "quality_drop": final_info.get("quality_drop", ""),
            "quality_epsilon": final_info.get("quality_epsilon", quality_cfg.get("epsilon", 0.0)),
            "penalized_quality_drop": final_info.get("penalized_quality_drop", ""),
            "original_score_file": final_info.get("original_score_file", ""),
            "original_score_key": final_info.get("original_score_key", ""),
            "original_parser_mode": final_info.get("original_parser_mode", ""),
            "original_preferred_score_key": final_info.get("original_preferred_score_key", ""),
            "compressed_score_file": final_info.get("compressed_score_file", ""),
            "compressed_score_key": final_info.get("compressed_score_key", ""),
            "compressed_parser_mode": final_info.get("compressed_parser_mode", ""),
            "compressed_preferred_score_key": final_info.get("compressed_preferred_score_key", ""),
            "crossscore_is_placeholder": final_info.get("crossscore_is_placeholder", ""),
            "critic_loss_D": critic_loss_D,
            "critic_loss_P": critic_loss_P,
            "actor_update_count_P": episode_actor_update_count_P,
            "actor_update_count_D": episode_actor_update_count_D,
            "actor_update_source": (
                "mixed"
                if episode_actor_update_count_P and episode_actor_update_count_D
                else "P"
                if episode_actor_update_count_P
                else "D"
                if episode_actor_update_count_D
                else "none"
            ),
            "checkpoint_path": str(checkpoint_path),
        }
        with episode_log.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=log_fields)
            writer.writerow(row)

        print(
            "episode={episode} scene={scene} num_groups={num_groups} "
            "target_groups={target_groups} grid_size={grid_size} "
            "mean_action={mean_action:.3f} compact_ratio={size_ratio:.6f} "
            "render_ply_ratio={render_ply_ratio:.6f} estimated_ratio={estimated_ratio:.6f} "
            "left_bitbudget={left_bitbudget:.1f} reward_D={reward_D:.6f} "
            "reward_P={reward_P:.6f} compact_package={path}".format(
                episode=row["episode"],
                scene=row["scene"],
                num_groups=row["num_groups"],
                target_groups=row["target_num_groups"],
                grid_size=row["grid_size"],
                mean_action=row["mean_action"],
                size_ratio=row["size_ratio"],
                render_ply_ratio=float(row.get("render_ply_size_ratio") or 0.0),
                estimated_ratio=float(row.get("estimated_size_ratio") or 0.0),
                left_bitbudget=row["left_bitbudget"],
                reward_D=row["reward_D"],
                reward_P=row["reward_P"],
                path=row.get("compact_package_path") or row["compressed_ply_path"],
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Reference-based Dual-Critic 3DGS training.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scene", default="all", help="Scene name from config, or 'all'.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--ply", default=None, help="Optional single-scene PLY path.")
    parser.add_argument("--scene-path", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--target-size-ratio", type=float, default=None)
    parser.add_argument("--max-groups", type=int, default=None, help="Debug-only truncation after grouping. Leave unset for formal runs.")
    parser.add_argument("--target-groups", type=int, default=None, help="Target number of voxel groups per scene, e.g. 128.")
    parser.add_argument("--max-search-grid-size", type=int, default=None, help="Largest grid_size searched when target_groups is enabled.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr-actor", type=float, default=ACTOR_LEARNING_RATE)
    parser.add_argument("--lr-critic", type=float, default=CRITIC_LEARNING_RATE)
    parser.add_argument("--exploration-std", type=float, default=0.2)
    parser.add_argument("--use-dummy-reward", action="store_true")
    parser.add_argument("--use-render", action="store_true")
    parser.add_argument("--use-crossscore", action="store_true")
    parser.add_argument("--allow-crossscore-placeholder", action="store_true")
    parser.add_argument("--force-recompute-original-score", action="store_true")
    parser.add_argument("--resume", default=None, help="Resume from a reference_based_epXXXX.pkl checkpoint.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
