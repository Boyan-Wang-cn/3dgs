from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .env import GSCompressionEnv
from .replay_buffer import EpisodeBuffer
from . import networks


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Tiny parser for configs/default.yaml when PyYAML is unavailable."""
    result: dict[str, Any] = {}
    current_section: str | None = None
    current_key: str | None = None
    current_item: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0 and stripped.endswith(":"):
            current_section = stripped[:-1]
            result[current_section] = {}
            current_key = None
            current_item = None
        elif indent == 2 and current_section is not None:
            key, _, value = stripped.partition(":")
            if value.strip():
                result[current_section][key] = _parse_scalar(value)
                current_key = None
            else:
                result[current_section][key] = []
                current_key = key
            current_item = None
        elif indent == 4 and stripped.startswith("- ") and current_section and current_key:
            current_item = {}
            result[current_section][current_key].append(current_item)
            item_text = stripped[2:]
            if item_text:
                key, _, value = item_text.partition(":")
                current_item[key] = _parse_scalar(value)
        elif indent >= 4 and current_item is not None:
            key, _, value = stripped.partition(":")
            current_item[key] = _parse_scalar(value)
        else:
            raise ValueError(f"Unsupported config line: {raw_line}")
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        return _simple_yaml_load(text)


def make_env(scene: dict[str, Any], config: dict[str, Any], episode_id: int) -> GSCompressionEnv:
    output_root = Path(config["project"]["output_root"])
    scene_name = scene.get("name", Path(scene["ply_path"]).stem)
    output_dir = output_root / "train" / f"episode_{episode_id:04d}_{scene_name}"
    env_cfg = config["env"]
    return GSCompressionEnv(
        ply_path=scene["ply_path"],
        scene_path=scene.get("scene_path"),
        output_dir=output_dir,
        grid_size=int(env_cfg.get("grid_size", 4)),
        target_size_ratio=float(env_cfg.get("target_size_ratio", 0.3)),
        max_groups=env_cfg.get("max_groups"),
        use_dummy_reward=bool(env_cfg.get("use_dummy_reward", True)),
        use_crossscore=bool(env_cfg.get("use_crossscore", False)),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    networks.require_torch()
    networks.torch.manual_seed(seed)


def run_training(config: dict[str, Any], episodes_override: int | None = None) -> None:
    networks.require_torch()
    torch = networks.torch
    nn = networks.nn

    train_cfg = config["train"]
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    scenes = [scene for scene in config["data"]["scenes"] if Path(scene["ply_path"]).exists()]
    if not scenes:
        raise FileNotFoundError("No configured scene has an existing ply_path.")

    probe_env = make_env(scenes[0], config, episode_id=0)
    probe_state = probe_env.reset()
    state_dim = int(probe_state.shape[0])

    actor = networks.Actor(state_dim=state_dim, action_dim=5)
    quality_critic = networks.QualityCritic(state_dim=state_dim)
    size_critic = networks.SizeCritic(state_dim=state_dim)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(train_cfg.get("lr_actor", 1e-4)))
    quality_optimizer = torch.optim.Adam(
        quality_critic.parameters(), lr=float(train_cfg.get("lr_critic", 1e-3))
    )
    size_optimizer = torch.optim.Adam(
        size_critic.parameters(), lr=float(train_cfg.get("lr_critic", 1e-3))
    )

    gamma = float(train_cfg.get("gamma", 0.99))
    episodes = int(episodes_override or train_cfg.get("episodes", 10))
    buffer = EpisodeBuffer()

    for episode_id in range(1, episodes + 1):
        scene = random.choice(scenes)
        scene_name = scene.get("name", Path(scene["ply_path"]).stem)
        env = make_env(scene, config, episode_id)
        state = env.reset()
        buffer.clear()
        log_probs = []
        entropies = []
        done = False
        final_info: dict[str, Any] = {}
        final_reward_quality = 0.0
        final_reward_size = 0.0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_tensor, log_prob, entropy = actor.sample_action(state_tensor)
            action = int(action_tensor.item())
            next_state, reward_quality, reward_size, done, info = env.step(action)
            buffer.append(state, action, reward_quality, reward_size, next_state, done, info)
            log_probs.append(log_prob.squeeze(0))
            entropies.append(entropy.squeeze(0))
            state = next_state
            if done:
                final_info = info
                final_reward_quality = reward_quality
                final_reward_size = reward_size

        states = torch.tensor(buffer.states(), dtype=torch.float32)
        returns_quality = torch.tensor(
            buffer.returns("reward_quality", gamma), dtype=torch.float32
        )
        returns_size = torch.tensor(buffer.returns("reward_size", gamma), dtype=torch.float32)

        values_quality = quality_critic(states)
        quality_loss = nn.functional.mse_loss(values_quality, returns_quality)
        quality_optimizer.zero_grad()
        quality_loss.backward()
        quality_optimizer.step()

        values_size = size_critic(states)
        size_loss = nn.functional.mse_loss(values_size, returns_size)
        size_optimizer.zero_grad()
        size_loss.backward()
        size_optimizer.step()

        with torch.no_grad():
            advantage_quality = returns_quality - quality_critic(states)
            advantage_size = returns_size - size_critic(states)
        size_violation = final_info["size_ratio"] > final_info["target_size_ratio"]
        chosen_advantage = advantage_size if size_violation else advantage_quality
        log_probs_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies)
        actor_loss = -(log_probs_tensor * chosen_advantage).mean() - 0.001 * entropy_tensor.mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        print(
            "episode={episode} scene={scene} num_groups={num_groups} "
            "mean_action={mean_action:.3f} original_size={original_size} "
            "compressed_size={compressed_size} size_ratio={size_ratio:.6f} "
            "reward_quality={reward_quality:.6f} reward_size={reward_size:.6f} "
            "compressed_ply_path={path}".format(
                episode=episode_id,
                scene=scene_name,
                num_groups=final_info["num_groups"],
                mean_action=final_info["mean_action"],
                original_size=final_info["original_size"],
                compressed_size=final_info["compressed_size"],
                size_ratio=final_info["size_ratio"],
                reward_quality=final_reward_quality,
                reward_size=final_reward_size,
                path=final_info["compressed_ply_path"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Dual-Critic 3DGS baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        run_training(config, episodes_override=args.episodes)
    except ModuleNotFoundError as exc:
        if "PyTorch is required" in str(exc):
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from None
        raise


if __name__ == "__main__":
    main()
