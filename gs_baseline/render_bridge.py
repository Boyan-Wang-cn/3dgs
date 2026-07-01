from __future__ import annotations

from pathlib import Path
import subprocess


def build_render_command(
    gaussian_splatting_dir: str | Path,
    scene_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path | None = None,
) -> list[str]:
    gs_dir = Path(gaussian_splatting_dir)
    command = [
        "python",
        str(gs_dir / "render.py"),
        "-m",
        str(model_path),
        "-s",
        str(scene_path),
    ]
    if output_dir is not None:
        command.extend(["--out", str(output_dir)])
    return command


def render_with_3dgs(
    gaussian_splatting_dir: str | Path,
    scene_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    run: bool = False,
) -> list[str]:
    """Bridge to the official 3DGS renderer.

    First version only builds and prints the command. Some 3DGS render.py
    variants require model_path to be a model directory containing
    point_cloud/iteration_xxx/point_cloud.ply; README documents this TODO.
    """
    command = build_render_command(gaussian_splatting_dir, scene_path, model_path, output_dir)
    print("3DGS render command:")
    print(" ".join(command))
    if run:
        subprocess.run(command, check=True, cwd=str(gaussian_splatting_dir))
    return command
