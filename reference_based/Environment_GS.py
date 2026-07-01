"""This file is adapted from Train/DRL_x265_TRAIN/Environment.py.

Mapping from the original video RDO environment:
- sequence / GOP -> one 3DGS scene episode.
- frame -> voxel group.
- QP -> compression level.
- bitbudget / targetbit -> target model size ratio and byte budget.
- left_bitbudget -> remaining size budget. Negative means size violation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from .config_utils import (
        CODE_ROOT,
        normalize_crossscore_dir,
        normalize_gaussian_splatting_dir,
    )
    from .compression_ops import estimate_size_ratio_from_actions
    from .crossscore_bridge import (
        compute_crossscore_placeholder,
        compute_crossscore_real,
        compute_quality_reward,
        load_score_cache,
        save_score_cache,
    )
    from .gs_compressor import GSCompressor
    from .model_path_utils import ensure_model_structure_from_ply, prepare_compressed_model_dir
    from .ply_utils import get_xyz, read_ply
    from .render_bridge import render_scene_pair
    from .voxel_grouping import (
        STATE_FEATURE_NAMES,
        build_state_vector,
        extract_group_features,
        voxel_group_indices,
    )
except ImportError:
    from config_utils import (
        CODE_ROOT,
        normalize_crossscore_dir,
        normalize_gaussian_splatting_dir,
    )
    from compression_ops import estimate_size_ratio_from_actions
    from crossscore_bridge import (
        compute_crossscore_placeholder,
        compute_crossscore_real,
        compute_quality_reward,
        load_score_cache,
        save_score_cache,
    )
    from gs_compressor import GSCompressor
    from model_path_utils import ensure_model_structure_from_ply, prepare_compressed_model_dir
    from ply_utils import get_xyz, read_ply
    from render_bridge import render_scene_pair
    from voxel_grouping import (
        STATE_FEATURE_NAMES,
        build_state_vector,
        extract_group_features,
        voxel_group_indices,
    )


class GS_Environment(object):
    def __init__(
        self,
        scenes: list[dict[str, Any]] | None = None,
        output_root: str | Path = "outputs",
        grid_size: int = 4,
        target_size_ratio: float = 0.3,
        max_groups: int | None = 32,
        min_group_size: int = 10,
        opacity_low_threshold: float = 0.01,
        use_dummy_reward: bool = True,
        use_render: bool = False,
        use_crossscore: bool = False,
        gaussian_splatting_dir: str | Path | None = None,
        crossscore_dir: str | Path | None = None,
        source_path: str | Path | None = None,
        original_model_path: str | Path | None = None,
        iteration: int = 30000,
        resolution: int = 4,
        quality_cache_dir: str | Path | None = None,
        cache_original_score: bool = True,
        score_higher_is_better: bool = True,
        crossscore_mode: str = "placeholder",
        crossscore_command_template: str = "",
        crossscore_score_output: str = "",
        crossscore_score_parse_mode: str = "auto",
        crossscore_preferred_score_key: str = "pred_ssim_0_1",
        render_python_executable: str = "python",
        crossscore_python_executable: str = "python",
        crossscore_ckpt: str | Path | None = None,
        crossscore_config: str | Path | None = None,
        crossscore_allow_image_fallback: bool = False,
        quality_epsilon: float = 0.0,
        allow_crossscore_placeholder: bool = False,
        force_recompute_original_score: bool = False,
        terminal_reward_only: bool = True,
    ) -> None:
        self.scenes = scenes or [
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
        self.output_root = Path(output_root)
        self.grid_size = grid_size
        self.target_size_ratio = target_size_ratio
        self.max_groups = max_groups
        self.min_group_size = min_group_size
        self.opacity_low_threshold = opacity_low_threshold
        self.use_dummy_reward = use_dummy_reward
        self.use_render = use_render
        self.use_crossscore = use_crossscore
        self.gaussian_splatting_dir = (
            normalize_gaussian_splatting_dir(gaussian_splatting_dir)
            if gaussian_splatting_dir
            else None
        )
        self.crossscore_dir = (
            normalize_crossscore_dir(crossscore_dir) if crossscore_dir else None
        )
        self.default_source_path = Path(source_path) if source_path else None
        self.default_original_model_path = (
            Path(original_model_path) if original_model_path else None
        )
        self.iteration = int(iteration)
        self.resolution = int(resolution)
        self.cache_original_score = cache_original_score
        self.score_higher_is_better = score_higher_is_better
        self.crossscore_mode = crossscore_mode
        self.crossscore_command_template = crossscore_command_template
        self.crossscore_score_output = crossscore_score_output
        self.crossscore_score_parse_mode = crossscore_score_parse_mode
        self.crossscore_preferred_score_key = crossscore_preferred_score_key
        self.render_python_executable = render_python_executable
        self.crossscore_python_executable = crossscore_python_executable
        self.crossscore_ckpt = crossscore_ckpt
        self.crossscore_config = crossscore_config
        self.crossscore_allow_image_fallback = crossscore_allow_image_fallback
        self.quality_epsilon = float(quality_epsilon)
        self.allow_crossscore_placeholder = allow_crossscore_placeholder
        self.force_recompute_original_score = force_recompute_original_score
        self.terminal_reward_only = terminal_reward_only

        self.state_dim = len(STATE_FEATURE_NAMES)
        self.episode = 0
        self.total_t = 0
        self.seqNum = 0
        self.frameNum = 0
        self.targetbit = target_size_ratio
        self.current_group_idx = 0
        self.done = False
        self.scene_info: dict[str, Any] = {}
        self.scene_name = ""
        self.scene_path = None
        self.ply_path = None
        self.ply = None
        self.groups = None
        self.original_size_bytes = 0
        self.target_size_bytes = 0.0
        self.compression_levels: list[int | None] = []
        self.compressor = GSCompressor(self.output_root)
        self.quality_cache_dir = Path(quality_cache_dir) if quality_cache_dir else (
            self.output_root / "crossscore_cache"
        )
        self.last_info: dict[str, Any] = {}
        self.readFile()

    def readFile(self):
        self.bitbudget_all = []
        self.action_all = []
        self.base_action_all = []
        self.reward_all = []
        self.noise_all = []
        self.reward_all_train_D = []
        self.reward_all_train_P = []
        self.q_value_D = []
        self.q_value_P = []

        self.reward_all_noNoise = []
        self.bitbudget_all_noNoise = []
        self.action_all_noNoise = []
        self.base_action_all_noNoise = []

    def readTargetbit(self, fname=None, QPs=None):
        _ = (fname, QPs)
        self.targetbit = self.target_size_ratio
        self.target_size_bytes = self.original_size_bytes * self.target_size_ratio

    def readTrainingSetting(self):
        return self.episode, self.total_t

    def updateTrainingSetting(self):
        return None

    def writeRecord(self, text):
        self.output_root.mkdir(parents=True, exist_ok=True)
        with (self.output_root / "episodeRewardRecord_GS.txt").open("a", encoding="utf-8") as fp:
            fp.write(text)

    def reset(self, scene_info: dict[str, Any] | None = None):
        self.readFile()
        if scene_info is None:
            scene_info = self.scenes[self.seqNum % len(self.scenes)]
            self.seqNum = (self.seqNum + 1) % len(self.scenes)

        self.scene_info = scene_info
        self.scene_name = scene_info.get("name", Path(scene_info["ply_path"]).stem)
        self.scene_path = (
            scene_info.get("source_path")
            or scene_info.get("scene_path")
            or self.default_source_path
        )
        self.original_model_path = (
            scene_info.get("model_path")
            or scene_info.get("original_model_path")
            or self.default_original_model_path
        )
        self.ply_path = Path(scene_info["ply_path"])
        self.ply = read_ply(self.ply_path)
        self.original_size_bytes = self.ply_path.stat().st_size
        self.target_size_bytes = self.original_size_bytes * self.target_size_ratio
        self.readTargetbit(self.scene_name, None)

        xyz = get_xyz(self.ply.vertex_data)
        self.groups = voxel_group_indices(
            xyz,
            grid_size=self.grid_size,
            min_group_size=self.min_group_size,
            max_groups=self.max_groups,
        )
        self.frameNum = len(self.groups.group_indices)
        if self.frameNum == 0:
            raise RuntimeError("No voxel groups were created for this scene.")

        self.compression_levels = [None for _ in range(self.frameNum)]
        self.current_group_idx = 0
        self.done = False
        self.episode += 1
        self.total_t = 0
        self.bitbudget_all.append(self.target_size_bytes)
        self.bitbudget_all_noNoise.append(self.target_size_bytes)
        return self.getObservation(0)

    def _prepare_original_model_dir(self) -> Path:
        if self.original_model_path is None:
            raise ValueError(
                "source_path/model_path are required when render or CrossScore is enabled. "
                f"Scene '{self.scene_name}' has no model_path."
            )
        if self.ply_path is None:
            raise ValueError(f"Scene '{self.scene_name}' has no ply_path.")
        return ensure_model_structure_from_ply(
            self.ply_path,
            self.original_model_path,
            iteration=self.iteration,
        )

    def _score_with_crossscore(self, render_dir, gt_dir, score_output_dir, tag):
        if self.crossscore_mode == "placeholder" and not self.crossscore_command_template:
            if self.use_crossscore and not self.allow_crossscore_placeholder:
                raise RuntimeError(
                    "CrossScore mode is placeholder while use_crossscore=True. "
                    "Pass --allow-crossscore-placeholder only for debugging, or set "
                    "crossscore_mode=auto_from_predict_sh / command_template in config."
                )
            score = compute_crossscore_placeholder(render_dir, gt_dir)
            return {
                "score": float(score),
                "score_file": "",
                "score_key": "placeholder",
                "parser_mode": "placeholder",
            }
        if self.crossscore_dir is None:
            raise ValueError("crossscore_dir is required when use_crossscore=True.")
        score_output_dir = Path(score_output_dir)
        score = compute_crossscore_real(
            self.crossscore_dir,
            render_dir,
            gt_dir,
            score_output_dir,
            scene_name=self.scene_name,
            tag=tag,
            python_executable=self.crossscore_python_executable,
            command_template=self.crossscore_command_template,
            score_output=self.crossscore_score_output,
            score_parse_mode=self.crossscore_score_parse_mode,
            preferred_score_key=self.crossscore_preferred_score_key,
            ckpt=self.crossscore_ckpt,
            config=self.crossscore_config,
            allow_image_fallback=self.crossscore_allow_image_fallback,
        )
        score_json = score_output_dir / "score.json"
        if score_json.exists():
            import json

            score_info = json.loads(score_json.read_text(encoding="utf-8-sig"))
            score_info["score"] = float(score_info.get("score", score))
            return score_info
        return {
            "score": float(score),
            "score_file": "",
            "score_key": "",
            "parser_mode": self.crossscore_score_parse_mode,
        }

    def _compute_terminal_quality_reward(self, compressed_model_dir: Path, info: dict[str, Any]):
        if self.use_dummy_reward:
            mean_action = info["mean_action"]
            info["reward_mode"] = "dummy"
            info["quality_mode"] = "dummy"
            info["crossscore_is_placeholder"] = False
            return -float(mean_action / 4.0 * 0.1), info

        if not self.use_render:
            info["reward_mode"] = "none"
            info["quality_mode"] = "no_dummy_no_render"
            info["crossscore_is_placeholder"] = False
            return 0.0, info

        if self.gaussian_splatting_dir is None:
            raise ValueError("gaussian_splatting_dir is required when use_render=True.")
        if self.scene_path is None:
            raise ValueError(
                "source_path is required when use_render=True. "
                f"Scene '{self.scene_name}' has no source_path."
            )
        original_model_dir = self._prepare_original_model_dir()
        render_info = render_scene_pair(
            self.gaussian_splatting_dir,
            original_model_dir,
            compressed_model_dir,
            self.scene_path,
            iteration=self.iteration,
            resolution=self.resolution,
            python_executable=self.render_python_executable,
        )
        info.update({key: str(value) for key, value in render_info.items()})
        info["reference_dir"] = str(render_info["compressed_gt_dir"])
        info["gt_dir"] = str(render_info["compressed_gt_dir"])

        if not self.use_crossscore:
            info["reward_mode"] = "render_only"
            info["quality_mode"] = "render_only"
            info["crossscore_is_placeholder"] = False
            return 0.0, info

        cache_path = self.quality_cache_dir / f"{self.scene_name}_original_score.json"
        cache_data = (
            None
            if self.force_recompute_original_score or not self.cache_original_score
            else load_score_cache(cache_path)
        )
        original_score_info = dict(cache_data) if cache_data is not None else None
        original_score = (
            float(original_score_info["score"]) if original_score_info is not None else None
        )
        if original_score is None:
            original_score_info = self._score_with_crossscore(
                render_info["original_render_dir"],
                render_info["original_gt_dir"],
                self.quality_cache_dir / self.scene_name / "original",
                tag="original",
            )
            original_score = float(original_score_info["score"])
            if self.cache_original_score:
                save_score_cache(
                    cache_path,
                    original_score,
                    metadata={
                        "scene": self.scene_name,
                        "original_model_path": str(original_model_dir),
                        "source_path": str(self.scene_path),
                        "render_dir": str(render_info["original_render_dir"]),
                        "reference_dir": str(render_info["original_gt_dir"]),
                        "crossscore_dir": str(self.crossscore_dir),
                        "crossscore_mode": self.crossscore_mode,
                        "score_file": original_score_info.get("score_file", ""),
                        "score_key": original_score_info.get("score_key", ""),
                        "parser_mode": original_score_info.get("parser_mode", ""),
                        "preferred_score_key": original_score_info.get(
                            "preferred_score_key", self.crossscore_preferred_score_key
                        ),
                    },
                )

        compressed_score_info = self._score_with_crossscore(
            render_info["compressed_render_dir"],
            render_info["compressed_gt_dir"],
            self.quality_cache_dir / self.scene_name / f"episode_{self.episode:04d}",
            tag=f"episode_{self.episode:04d}",
        )
        compressed_score = float(compressed_score_info["score"])
        quality_info = compute_quality_reward(
            original_score,
            compressed_score,
            higher_is_better=self.score_higher_is_better,
            epsilon=self.quality_epsilon,
        )
        info.update(
            {
                "reward_mode": "crossscore",
                "quality_mode": "crossscore",
                "original_score": float(original_score),
                "compressed_score": float(compressed_score),
                "quality_drop": float(quality_info["quality_drop"]),
                "quality_epsilon": float(quality_info["quality_epsilon"]),
                "penalized_quality_drop": float(quality_info["penalized_quality_drop"]),
                "original_score_file": original_score_info.get("score_file", ""),
                "original_score_key": original_score_info.get("score_key", ""),
                "original_parser_mode": original_score_info.get("parser_mode", ""),
                "original_preferred_score_key": original_score_info.get(
                    "preferred_score_key", self.crossscore_preferred_score_key
                ),
                "compressed_score_file": compressed_score_info.get("score_file", ""),
                "compressed_score_key": compressed_score_info.get("score_key", ""),
                "compressed_parser_mode": compressed_score_info.get("parser_mode", ""),
                "compressed_preferred_score_key": compressed_score_info.get(
                    "preferred_score_key", self.crossscore_preferred_score_key
                ),
                "crossscore_is_placeholder": self.crossscore_mode == "placeholder",
            }
        )
        return float(quality_info["reward_D"]), info

    def _previous_levels(self):
        return [int(level) for level in self.compression_levels if level is not None]

    def _estimated_size_ratio(self):
        return estimate_size_ratio_from_actions(
            self.groups.group_indices,
            self.compression_levels,
            total_vertices=len(self.ply.vertex_data),
        )

    def _estimated_left_bitbudget(self):
        estimated_size = self.original_size_bytes * self._estimated_size_ratio()
        return float(self.target_size_bytes - estimated_size)

    def _group_feature(self, fNum):
        indices = self.groups.group_indices[fNum]
        return extract_group_features(
            self.ply.vertex_data,
            indices,
            total_gaussians=len(self.ply.vertex_data),
            bbox_min=self.groups.bbox_min,
            bbox_max=self.groups.bbox_max,
            opacity_low_threshold=self.opacity_low_threshold,
            small_group_flag=self.groups.small_group_flags[fNum],
        )

    def getObservation(self, fNum):
        if self.ply is None or self.groups is None or fNum >= self.frameNum:
            return np.zeros([self.state_dim], dtype=np.float32)
        feature = self._group_feature(fNum)
        observation = build_state_vector(
            group_features=feature,
            current_group_idx=fNum,
            total_groups=self.frameNum,
            current_estimated_size_ratio=self._estimated_size_ratio(),
            target_size_ratio=self.target_size_ratio,
            previous_actions=self._previous_levels(),
        )
        return observation.astype(np.float32)

    def getObservation_noNoise(self, fNum):
        return self.getObservation(fNum)

    def getTrainReward(self, frameNum):
        start = max(int(frameNum), 0)
        distortTotal_D = float(np.sum(self.reward_all_train_D[start:]))
        distortTotal_P = float(np.sum(self.reward_all_train_P[start:]))
        return distortTotal_D, distortTotal_P

    def getDistortion(self, frameNum):
        start = max(int(frameNum), 0)
        return float(np.sum(self.reward_all[start:]))

    def step(self, action, baseQP=0.0):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before step().")
        action_continuous = float(np.asarray(action).reshape(-1)[0])
        base_level = float(np.asarray(baseQP).reshape(-1)[0])
        compression_level = int(round(np.clip(action_continuous, 0.0, 4.0)))

        group_idx = self.current_group_idx
        self.compression_levels[group_idx] = compression_level
        self.action_all.append(action_continuous)
        self.base_action_all.append(base_level)
        self.action_all_noNoise.append(action_continuous)
        self.base_action_all_noNoise.append(base_level)
        self.noise_all.append(0.0)
        self.total_t += 1

        is_last_group = group_idx >= self.frameNum - 1
        if not is_last_group:
            left_bitbudget = self._estimated_left_bitbudget()
            reward_D = 0.0
            reward_P = 0.0
            self.reward_all_train_D.append(reward_D)
            self.reward_all_train_P.append(reward_P)
            self.reward_all.append(reward_D + reward_P)
            self.bitbudget_all.append(left_bitbudget)
            self.bitbudget_all_noNoise.append(left_bitbudget)
            self.current_group_idx += 1
            next_state = self.getObservation(self.current_group_idx)
            info = {
                "left_bitbudget": left_bitbudget,
                "compression_level": compression_level,
                "baseQP": base_level,
                "size_ratio": self._estimated_size_ratio(),
                "num_groups": self.frameNum,
            }
            self.last_info = info
            return next_state, reward_D, reward_P, False, info

        levels = [int(level) if level is not None else 0 for level in self.compression_levels]
        compressed_ply_path, compression_stats = self.compressor.compress_scene(
            self.ply,
            self.groups.group_indices,
            levels,
            scene_name=self.scene_name,
            episode=self.episode,
        )
        compressed_model_dir = prepare_compressed_model_dir(
            compressed_ply_path,
            self.output_root,
            self.scene_name,
            self.episode,
            iteration=self.iteration,
        )
        compressed_size_bytes = compressed_ply_path.stat().st_size
        size_ratio = compressed_size_bytes / max(float(self.original_size_bytes), 1.0)
        left_bitbudget = float(self.target_size_bytes - compressed_size_bytes)
        mean_action = float(np.mean(levels)) if levels else 0.0
        level_histogram = {
            str(level): int(count)
            for level, count in zip(*np.unique(np.asarray(levels, dtype=int), return_counts=True))
        }
        original_gaussians = int(compression_stats.get("original_vertices", len(self.ply.vertex_data)))
        compressed_gaussians = int(compression_stats.get("kept_vertices", 0))
        pruned_gaussians = int(compression_stats.get("pruned_vertices", original_gaussians - compressed_gaussians))

        reward_P = -float(max(0.0, -left_bitbudget) / max(float(self.original_size_bytes), 1.0))
        info = {
            "left_bitbudget": left_bitbudget,
            "target_size_bytes": float(self.target_size_bytes),
            "original_size": int(self.original_size_bytes),
            "compressed_size": int(compressed_size_bytes),
            "size_ratio": float(size_ratio),
            "disk_size_ratio": float(size_ratio),
            "target_size_ratio": float(self.target_size_ratio),
            "compressed_ply_path": str(compressed_ply_path),
            "compressed_model_dir": str(compressed_model_dir),
            "mean_action": mean_action,
            "mean_level": mean_action,
            "level_histogram": level_histogram,
            "original_gaussians": original_gaussians,
            "compressed_gaussians": compressed_gaussians,
            "pruned_gaussians": pruned_gaussians,
            "num_groups": self.frameNum,
            "compression_level": compression_level,
            "baseQP": base_level,
            "compression_stats": compression_stats,
        }
        reward_D, info = self._compute_terminal_quality_reward(compressed_model_dir, info)
        info["reward_D"] = float(reward_D)
        info["reward_P"] = float(reward_P)
        self.reward_all_train_D.append(reward_D)
        self.reward_all_train_P.append(reward_P)
        self.reward_all.append(reward_D + reward_P)
        self.bitbudget_all.append(left_bitbudget)
        self.bitbudget_all_noNoise.append(left_bitbudget)
        self.done = True
        self.current_group_idx = self.frameNum
        next_state = self.getObservation(self.current_group_idx)
        self.last_info = info
        return next_state, reward_D, reward_P, True, info
