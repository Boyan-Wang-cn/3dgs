"""Utilities for official GraphDeco gaussian-splatting model_path layout."""

from __future__ import annotations

from pathlib import Path
import shutil


def get_point_cloud_ply(model_dir: str | Path, iteration: int = 30000) -> Path:
    model_dir = Path(model_dir)
    return model_dir / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"


def ensure_model_structure_from_ply(
    ply_path: str | Path,
    model_dir: str | Path,
    iteration: int = 30000,
) -> Path:
    ply_path = Path(ply_path).resolve()
    model_dir = Path(model_dir).resolve()
    target_ply = get_point_cloud_ply(model_dir, iteration).resolve()

    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file does not exist: {ply_path}")
    if ply_path == target_ply and target_ply.exists():
        return model_dir

    target_ply.parent.mkdir(parents=True, exist_ok=True)
    if not target_ply.exists():
        shutil.copy2(ply_path, target_ply)
    return model_dir


def prepare_compressed_model_dir(
    compressed_ply: str | Path,
    output_root: str | Path,
    scene_name: str,
    episode_id: int,
    iteration: int = 30000,
    original_model_path: str | Path | None = None,
) -> Path:
    output_root = Path(output_root)
    model_dir = output_root / "compressed_models" / scene_name / f"episode_{int(episode_id):04d}"
    model_dir = ensure_model_structure_from_ply(compressed_ply, model_dir, iteration)

    if original_model_path is not None:
        original_cfg = Path(original_model_path) / "cfg_args"
        target_cfg = model_dir / "cfg_args"
        if original_cfg.exists():
            shutil.copy2(original_cfg, target_cfg)
        else:
            print(f"WARNING: original cfg_args not found: {original_cfg}")

    return model_dir
