from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from .compression_ops import (
    COMPRESSION_LEVELS,
    apply_compression_to_vertices,
    estimate_size_ratio_from_actions,
)
from .ply_utils import get_xyz, read_ply, write_ply
from .voxel_grouping import (
    STATE_FEATURE_NAMES,
    build_state_vector,
    extract_group_features,
    voxel_group_indices,
)


class GSCompressionEnv:
    """Voxel-group 3DGS compression environment with dual rewards."""

    state_dim = len(STATE_FEATURE_NAMES)

    def __init__(
        self,
        ply_path: str | Path,
        scene_path: str | Path | None = None,
        output_dir: str | Path = "outputs",
        grid_size: int = 4,
        target_size_ratio: float = 0.3,
        max_groups: int | None = None,
        use_dummy_reward: bool = True,
        use_crossscore: bool = False,
        min_group_size: int = 10,
        opacity_low_threshold: float = 0.01,
    ) -> None:
        self.ply_path = Path(ply_path)
        self.scene_path = Path(scene_path) if scene_path else None
        self.output_dir = Path(output_dir)
        self.grid_size = grid_size
        self.target_size_ratio = target_size_ratio
        self.max_groups = max_groups
        self.use_dummy_reward = use_dummy_reward
        self.use_crossscore = use_crossscore
        self.min_group_size = min_group_size
        self.opacity_low_threshold = opacity_low_threshold

        self.ply = None
        self.groups = None
        self.current_group_idx = 0
        self.actions: list[int | None] = []
        self.original_size_bytes = 0
        self.episode_counter = 0
        self.done = False

    @property
    def num_groups(self) -> int:
        return len(self.groups.group_indices) if self.groups is not None else 0

    def reset(self) -> np.ndarray:
        self.ply = read_ply(self.ply_path)
        self.original_size_bytes = self.ply_path.stat().st_size
        xyz = get_xyz(self.ply.vertex_data)
        self.groups = voxel_group_indices(
            xyz,
            grid_size=self.grid_size,
            min_group_size=self.min_group_size,
            max_groups=self.max_groups,
        )
        if not self.groups.group_indices:
            raise RuntimeError("Voxel grouping produced no groups.")

        self.current_group_idx = 0
        self.actions = [None for _ in self.groups.group_indices]
        self.done = False
        self.episode_counter += 1
        return self._state_for_current_group()

    def _previous_actions(self) -> list[int]:
        return [int(action) for action in self.actions if action is not None]

    def _current_estimated_size_ratio(self) -> float:
        return estimate_size_ratio_from_actions(
            self.groups.group_indices,
            self.actions,
            total_vertices=len(self.ply.vertex_data),
        )

    def _state_for_current_group(self) -> np.ndarray:
        if self.ply is None or self.groups is None:
            raise RuntimeError("Call reset() before requesting state.")
        idx = min(self.current_group_idx, self.num_groups - 1)
        indices = self.groups.group_indices[idx]
        group_features = extract_group_features(
            self.ply.vertex_data,
            indices,
            total_gaussians=len(self.ply.vertex_data),
            bbox_min=self.groups.bbox_min,
            bbox_max=self.groups.bbox_max,
            opacity_low_threshold=self.opacity_low_threshold,
            small_group_flag=self.groups.small_group_flags[idx],
        )
        return build_state_vector(
            group_features=group_features,
            current_group_idx=idx,
            total_groups=self.num_groups,
            current_estimated_size_ratio=self._current_estimated_size_ratio(),
            target_size_ratio=self.target_size_ratio,
            previous_actions=self._previous_actions(),
        )

    def step(self, action: int) -> tuple[np.ndarray, float, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self.ply is None or self.groups is None:
            raise RuntimeError("Call reset() before step().")
        if int(action) not in COMPRESSION_LEVELS:
            raise ValueError(f"Action must be a compression level 0..4, got {action}.")

        self.actions[self.current_group_idx] = int(action)
        is_last_group = self.current_group_idx >= self.num_groups - 1
        if not is_last_group:
            self.current_group_idx += 1
            return self._state_for_current_group(), 0.0, 0.0, False, {}

        compressed_vertices, compression_stats = apply_compression_to_vertices(
            self.ply.vertex_data,
            self.groups.group_indices,
            [int(action) for action in self.actions],
        )
        compressed_ply_path = self._write_compressed_ply(compressed_vertices)
        compressed_size_bytes = compressed_ply_path.stat().st_size
        size_ratio = compressed_size_bytes / max(float(self.original_size_bytes), 1.0)
        mean_action = float(np.mean(self._previous_actions())) if self.actions else 0.0

        if self.use_dummy_reward:
            estimated_quality_drop = mean_action / 4.0 * 0.1
            quality_reward = -float(estimated_quality_drop)
        else:
            quality_reward = 0.0

        size_reward = -float(max(0.0, size_ratio - self.target_size_ratio))
        self.done = True

        info = {
            "original_size": int(self.original_size_bytes),
            "compressed_size": int(compressed_size_bytes),
            "size_ratio": float(size_ratio),
            "target_size_ratio": float(self.target_size_ratio),
            "compressed_ply_path": str(compressed_ply_path),
            "mean_action": float(mean_action),
            "num_groups": int(self.num_groups),
            "compression_stats": compression_stats,
        }
        return self._state_for_current_group(), quality_reward, size_reward, True, info

    def _write_compressed_ply(self, compressed_vertices: np.ndarray) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"{self.ply_path.stem}_compressed_ep{self.episode_counter:04d}_{timestamp}.ply"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return write_ply(self.ply, self.output_dir / name, compressed_vertices)
