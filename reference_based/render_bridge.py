"""Optional bridge to GraphDeco gaussian-splatting render.py."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import time

try:
    from .model_path_utils import get_point_cloud_ply
except ImportError:
    from model_path_utils import get_point_cloud_ply


def _ours_dir(iteration: int) -> str:
    return f"ours_{int(iteration)}"


def _resolve_python_executable(python_executable: str | Path) -> str:
    executable = str(python_executable or "python")
    executable_path = Path(executable)
    has_path_separator = "/" in executable or "\\" in executable
    if executable_path.is_absolute() or has_path_separator:
        resolved = executable_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Configured render.python_executable does not exist: {resolved}")
        return str(resolved)
    found = shutil.which(executable)
    if found is None:
        raise FileNotFoundError(
            f"Configured render.python_executable is not on PATH: {executable}"
        )
    return found


def run_graphdeco_render(
    gaussian_splatting_dir,
    model_path,
    source_path,
    iteration: int = 30000,
    resolution: int = 4,
    split: str = "train",
    timeout=None,
    python_executable: str | Path = "python",
):
    gaussian_splatting_dir = Path(gaussian_splatting_dir).resolve()
    model_path = Path(model_path).resolve()
    source_path = Path(source_path).resolve()
    render_py = gaussian_splatting_dir / "render.py"
    point_cloud_ply = get_point_cloud_ply(model_path, iteration)
    python_executable = _resolve_python_executable(python_executable)
    expected_render_dirs = [
        model_path / "train" / _ours_dir(iteration) / "renders",
        model_path / "test" / _ours_dir(iteration) / "renders",
        model_path / "renders",
    ]
    expected_gt_dirs = [
        model_path / "train" / _ours_dir(iteration) / "gt",
        model_path / "test" / _ours_dir(iteration) / "gt",
        model_path / "gt",
    ]

    if not render_py.exists():
        raise FileNotFoundError(f"render.py was not found: {render_py}")
    if not source_path.exists():
        raise FileNotFoundError(f"3DGS source path does not exist: {source_path}")
    if not point_cloud_ply.exists():
        raise FileNotFoundError(
            f"model_path is missing point cloud PLY: {point_cloud_ply}"
        )

    command = [
        python_executable,
        "render.py",
        "-m",
        str(model_path),
        "-s",
        str(source_path),
        "--iteration",
        str(int(iteration)),
        "-r",
        str(int(resolution)),
    ]
    print("Preparing 3DGS render:")
    print(f"  python_executable: {python_executable}")
    print(f"  gaussian_splatting_dir: {gaussian_splatting_dir}")
    print(f"  render_py: {render_py}")
    print(f"  model_path: {model_path}")
    print(f"  source_path: {source_path}")
    print(f"  point_cloud_ply: {point_cloud_ply}")
    print(f"  iteration: {int(iteration)}")
    print(f"  expected_render_dirs: {[str(path) for path in expected_render_dirs]}")
    print(f"  expected_gt_dirs: {[str(path) for path in expected_gt_dirs]}")
    print("Running 3DGS render command:")
    print(" ".join(command))

    completed = subprocess.run(
        command,
        cwd=str(gaussian_splatting_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"render_{split}_{int(time.time())}.log"
        log_path.write_text(
            "COMMAND:\n"
            + " ".join(command)
            + "\n\nSTDOUT:\n"
            + completed.stdout
            + "\n\nSTDERR:\n"
            + completed.stderr,
            encoding="utf-8",
        )
        raise RuntimeError(
            f"3DGS render failed with code {completed.returncode}. Log: {log_path}"
        )
    return completed


def find_render_dir(model_path, iteration: int = 30000) -> Path:
    model_path = Path(model_path)
    candidates = [
        model_path / "train" / _ours_dir(iteration) / "renders",
        model_path / "test" / _ours_dir(iteration) / "renders",
        model_path / "renders",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(model_path.rglob("renders")) if model_path.exists() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No renders directory found under {model_path}")


def find_gt_dir(model_path, iteration: int = 30000) -> Path:
    model_path = Path(model_path)
    candidates = [
        model_path / "train" / _ours_dir(iteration) / "gt",
        model_path / "test" / _ours_dir(iteration) / "gt",
        model_path / "gt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(model_path.rglob("gt")) if model_path.exists() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No gt directory found under {model_path}")


def render_scene_pair(
    gaussian_splatting_dir,
    original_model_path,
    compressed_model_path,
    source_path,
    iteration: int = 30000,
    resolution: int = 4,
    python_executable: str | Path = "python",
):
    run_graphdeco_render(
        gaussian_splatting_dir,
        original_model_path,
        source_path,
        iteration=iteration,
        resolution=resolution,
        split="original",
        python_executable=python_executable,
    )
    run_graphdeco_render(
        gaussian_splatting_dir,
        compressed_model_path,
        source_path,
        iteration=iteration,
        resolution=resolution,
        split="compressed",
        python_executable=python_executable,
    )
    return {
        "original_render_dir": find_render_dir(original_model_path, iteration),
        "original_gt_dir": find_gt_dir(original_model_path, iteration),
        "compressed_render_dir": find_render_dir(compressed_model_path, iteration),
        "compressed_gt_dir": find_gt_dir(compressed_model_path, iteration),
    }
