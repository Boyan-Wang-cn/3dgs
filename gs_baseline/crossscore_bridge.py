from __future__ import annotations

from pathlib import Path


def compute_crossscore(render_dir: str | Path, reference_dir: str | Path) -> float:
    """Placeholder for CrossScore inference.

    TODO:
    - Call D:/download/Code/CrossScore-main/CrossScore-main predict.sh or its
      Python inference entrypoint.
    - Input rendered 3DGS images and reference images.
    - Return one scalar CrossScore score.
    """
    _ = (Path(render_dir), Path(reference_dir))
    return 0.0


def compute_quality_reward(
    original_render_dir: str | Path,
    compressed_render_dir: str | Path,
    reference_dir: str | Path,
) -> float:
    """Return compressed_score - original_score as the future quality reward."""
    original_score = compute_crossscore(original_render_dir, reference_dir)
    compressed_score = compute_crossscore(compressed_render_dir, reference_dir)
    return float(compressed_score - original_score)
