"""Optional bridge to GraphDeco gaussian-splatting render.py."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
import shutil
import subprocess
import tempfile
import time
from typing import Any

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


_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_FIXED_VIEW_STRATEGY = "deterministic_evenly_spaced"
_LEGACY_RENDER_SCENE_PAIR_SOURCE_SHA256 = (
    "b9ed812253ff83e7abc70c5a510abf2b39f8cf187aeb42bbf846fe326b88b854"
)


@dataclass(frozen=True)
class FixedViewSubset:
    """Metadata for a persistent, deterministically selected camera subset."""

    selected_relative_paths: tuple[str, ...]
    available_view_count: int
    selected_view_count: int
    requested_view_count: int
    selection_strategy: str
    manifest_path: str

    def __post_init__(self) -> None:
        if self.selection_strategy != _FIXED_VIEW_STRATEGY:
            raise ValueError(
                "selection_strategy must be "
                f"{_FIXED_VIEW_STRATEGY!r}, got {self.selection_strategy!r}"
            )


def _require_positive_int(value: Any, name: str) -> int:
    """Return *value* as an int after strict positive-integer validation."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer (bool is not allowed)")
    return value


def _validate_safe_relative_path(relative_path: Any) -> str:
    """Validate and return a safe POSIX relative path without normalizing it."""

    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("selected_relative_paths must contain non-empty strings")
    if "\\" in relative_path:
        raise ValueError(
            f"Manifest path must use POSIX separators: {relative_path!r}"
        )
    if (
        PurePosixPath(relative_path).is_absolute()
        or PureWindowsPath(relative_path).is_absolute()
    ):
        raise ValueError(f"Absolute manifest path is not allowed: {relative_path!r}")
    components = relative_path.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError(f"Unsafe manifest relative path: {relative_path!r}")
    return relative_path


def _collect_image_paths(directory: str | Path) -> set[str]:
    """Collect supported image files as exact POSIX relative paths."""

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Render image directory does not exist: {root}")
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    }


def collect_common_render_views(
    original_render_dir: str | Path,
    original_gt_dir: str | Path,
    compressed_render_dir: str | Path,
    compressed_gt_dir: str | Path,
) -> list[str]:
    """Return the sorted exact relative image paths shared by all four inputs.

    Matching is case-sensitive and strictly based on each file's POSIX relative
    path. No positional pairing, fuzzy matching, or fallback is performed.
    """

    named_directories = (
        ("original_render_dir", original_render_dir),
        ("original_gt_dir", original_gt_dir),
        ("compressed_render_dir", compressed_render_dir),
        ("compressed_gt_dir", compressed_gt_dir),
    )
    image_sets = {
        name: _collect_image_paths(directory)
        for name, directory in named_directories
    }
    common = set.intersection(*image_sets.values())
    if not common:
        counts = ", ".join(
            f"{name}={len(paths)}" for name, paths in image_sets.items()
        )
        raise ValueError(f"No common render views exist across the four directories; {counts}")
    return sorted(common)


def select_deterministic_view_subset(
    common_relative_paths: list[str],
    requested_view_count: int,
) -> tuple[str, ...]:
    """Select an ordered, deterministic, evenly spaced subset of view paths."""

    requested = _require_positive_int(requested_view_count, "requested_view_count")
    ordered_paths = sorted(set(common_relative_paths))
    for relative_path in ordered_paths:
        if not isinstance(relative_path, str):
            raise ValueError("common_relative_paths must contain only strings")
    num_views = len(ordered_paths)
    if num_views <= requested:
        return tuple(ordered_paths)

    if requested == 1:
        indices = [0]
    else:
        # This is the integer-floor equivalent of linspace(0, n - 1, k, dtype=int).
        indices = [
            index * (num_views - 1) // (requested - 1)
            for index in range(requested)
        ]
    if len(indices) != requested or len(set(indices)) != requested:
        # The branch is defensive: num_views > requested makes the formula unique.
        indices = list(range(requested))
    if len(indices) != requested or len(set(indices)) != requested:
        raise RuntimeError("Could not construct the requested unique view indices")
    return tuple(ordered_paths[index] for index in indices)


def _validated_manifest_payload(payload: Any, manifest_path: Path) -> tuple[int, list[str]]:
    """Validate a version-1 fixed-view manifest and return its core fields."""

    if not isinstance(payload, dict):
        raise ValueError(f"Fixed-view manifest must contain a JSON object: {manifest_path}")
    if isinstance(payload.get("version"), bool) or payload.get("version") != 1:
        raise ValueError(f"Fixed-view manifest version must be 1: {manifest_path}")
    if payload.get("selection_strategy") != _FIXED_VIEW_STRATEGY:
        raise ValueError(
            "Fixed-view manifest has an incompatible selection_strategy: "
            f"{payload.get('selection_strategy')!r}"
        )
    requested = _require_positive_int(
        payload.get("requested_view_count"), "manifest requested_view_count"
    )
    available_at_creation = payload.get("available_view_count_at_creation")
    if (
        isinstance(available_at_creation, bool)
        or not isinstance(available_at_creation, int)
        or available_at_creation < 0
    ):
        raise ValueError(
            "manifest available_view_count_at_creation must be a non-negative integer"
        )
    selected = payload.get("selected_relative_paths")
    if not isinstance(selected, list) or not selected:
        raise ValueError(
            "manifest selected_relative_paths must be a non-empty string list"
        )
    validated = [_validate_safe_relative_path(path) for path in selected]
    if len(set(validated)) != len(validated):
        raise ValueError("manifest selected_relative_paths must not contain duplicates")
    if available_at_creation < len(validated):
        raise ValueError(
            "manifest available_view_count_at_creation is smaller than the selection"
        )
    return requested, validated


def load_or_create_fixed_view_manifest(
    common_relative_paths: list[str],
    requested_view_count: int,
    manifest_path: str | Path,
) -> FixedViewSubset:
    """Load a fixed view selection, or create it once without future resampling."""

    requested = _require_positive_int(requested_view_count, "requested_view_count")
    ordered_common = sorted(set(common_relative_paths))
    for relative_path in ordered_common:
        _validate_safe_relative_path(relative_path)
    if not ordered_common:
        raise ValueError("Cannot create or validate a fixed-view manifest with no common views")

    manifest = Path(manifest_path)
    if manifest.exists():
        if not manifest.is_file():
            raise ValueError(f"Fixed-view manifest path is not a file: {manifest}")
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read fixed-view manifest {manifest}: {exc}") from exc
        manifest_requested, selected_list = _validated_manifest_payload(payload, manifest)
        if manifest_requested != requested:
            raise ValueError(
                "requested_view_count does not match the existing fixed-view manifest: "
                f"requested={requested}, manifest={manifest_requested}"
            )
        common_set = set(ordered_common)
        missing = [path for path in selected_list if path not in common_set]
        if missing:
            raise ValueError(
                "Fixed-view manifest contains views missing from the current common set: "
                + ", ".join(missing)
            )
        selected = tuple(selected_list)
    else:
        selected = select_deterministic_view_subset(ordered_common, requested)
        if not selected:
            raise ValueError("A fixed-view manifest cannot contain an empty selection")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "selection_strategy": _FIXED_VIEW_STRATEGY,
            "requested_view_count": requested,
            "available_view_count_at_creation": len(ordered_common),
            "selected_relative_paths": list(selected),
        }
        manifest.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return FixedViewSubset(
        selected_relative_paths=selected,
        available_view_count=len(ordered_common),
        selected_view_count=len(selected),
        requested_view_count=requested,
        selection_strategy=_FIXED_VIEW_STRATEGY,
        manifest_path=str(manifest),
    )


def _clean_materialization_directory(directory: Path) -> None:
    """Remove one previous subset target and recreate it as an ordinary directory."""

    if directory.is_symlink() or directory.is_file():
        directory.unlink()
    elif directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def materialize_fixed_view_subset(
    original_render_dir: str | Path,
    original_gt_dir: str | Path,
    compressed_render_dir: str | Path,
    compressed_gt_dir: str | Path,
    output_root: str | Path,
    subset: FixedViewSubset,
) -> dict[str, Path]:
    """Hard-link, or copy, all four matched sources into clean subset directories."""

    if not isinstance(subset, FixedViewSubset):
        raise TypeError("subset must be a FixedViewSubset")
    selected = subset.selected_relative_paths
    if subset.selected_view_count != len(selected) or not selected:
        raise ValueError("subset selected_view_count must match a non-empty path tuple")
    if len(set(selected)) != len(selected):
        raise ValueError("subset selected_relative_paths must not contain duplicates")
    for relative_path in selected:
        _validate_safe_relative_path(relative_path)

    sources = {
        "original_render_subset_dir": Path(original_render_dir),
        "original_gt_subset_dir": Path(original_gt_dir),
        "compressed_render_subset_dir": Path(compressed_render_dir),
        "compressed_gt_subset_dir": Path(compressed_gt_dir),
    }
    for source in sources.values():
        if not source.is_dir():
            raise FileNotFoundError(f"Render image directory does not exist: {source}")

    root = Path(output_root)
    targets = {
        key: root / key.removesuffix("_subset_dir")
        for key in sources
    }
    resolved_sources = [source.resolve() for source in sources.values()]
    for target in targets.values():
        resolved_target = target.resolve()
        if any(
            resolved_target == source
            or resolved_target in source.parents
            or source in resolved_target.parents
            for source in resolved_sources
        ):
            raise ValueError(
                f"Subset output directory must not overlap a source directory: {target}"
            )

    for target in targets.values():
        _clean_materialization_directory(target)

    for key, source_root in sources.items():
        target_root = targets[key]
        for relative_path in selected:
            source = source_root / Path(*PurePosixPath(relative_path).parts)
            if not source.is_file():
                raise FileNotFoundError(
                    f"Fixed view is missing from {source_root}: {relative_path}"
                )
            destination = target_root / Path(*PurePosixPath(relative_path).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)

    for target in targets.values():
        output_files = [path for path in target.rglob("*") if path.is_file()]
        if len(output_files) != subset.selected_view_count:
            raise RuntimeError(
                f"Materialized file count mismatch for {target}: "
                f"expected {subset.selected_view_count}, got {len(output_files)}"
            )
        for relative_path in selected:
            destination = target / Path(*PurePosixPath(relative_path).parts)
            if not destination.is_file() or destination.is_symlink():
                raise RuntimeError(
                    f"Materialized fixed view is missing or is a symlink: {destination}"
                )
    return targets


def prepare_fixed_render_subset(
    render_info: dict[str, Any],
    requested_view_count: int,
    manifest_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Collect, freeze, and materialize a strictly paired four-way view subset."""

    required_keys = (
        "original_render_dir",
        "original_gt_dir",
        "compressed_render_dir",
        "compressed_gt_dir",
    )
    missing_keys = [key for key in required_keys if key not in render_info]
    if missing_keys:
        raise KeyError("render_info is missing required keys: " + ", ".join(missing_keys))
    common = collect_common_render_views(
        render_info["original_render_dir"],
        render_info["original_gt_dir"],
        render_info["compressed_render_dir"],
        render_info["compressed_gt_dir"],
    )
    subset = load_or_create_fixed_view_manifest(
        common,
        requested_view_count,
        manifest_path,
    )
    result: dict[str, Any] = materialize_fixed_view_subset(
        render_info["original_render_dir"],
        render_info["original_gt_dir"],
        render_info["compressed_render_dir"],
        render_info["compressed_gt_dir"],
        output_root,
        subset,
    )
    result.update(
        {
            "selected_relative_paths": subset.selected_relative_paths,
            "selected_view_count": subset.selected_view_count,
            "requested_view_count": subset.requested_view_count,
            "available_view_count": subset.available_view_count,
            "selection_strategy": subset.selection_strategy,
            "manifest_path": Path(subset.manifest_path),
        }
    )
    return result


def render_scene_pair_fixed_subset(
    gaussian_splatting_dir: str | Path,
    original_model_path: str | Path,
    compressed_model_path: str | Path,
    source_path: str | Path,
    *,
    subset_output_root: str | Path,
    manifest_path: str | Path,
    requested_view_count: int,
    iteration: int = 30000,
    resolution: int = 4,
    python_executable: str | Path = "python",
) -> dict[str, Any]:
    """Render the full scene pair and expose one persistent matched camera subset.

    Here, "partial/stage quality rendering" means rendering the complete 3DGS
    scene after some groups have been processed, while undecided groups remain at
    identity, and evaluating quality on a fixed camera subset. It does *not* mean
    rendering only the latest eight groups, hiding other Gaussians, or changing
    occlusion, blending, or the surrounding scene context. GraphDeco may still
    render every camera; downstream quality evaluation should read only the four
    returned subset directories.
    """

    render_info = render_scene_pair(
        gaussian_splatting_dir,
        original_model_path,
        compressed_model_path,
        source_path,
        iteration=iteration,
        resolution=resolution,
        python_executable=python_executable,
    )
    subset_info = prepare_fixed_render_subset(
        render_info,
        requested_view_count=requested_view_count,
        manifest_path=manifest_path,
        output_root=subset_output_root,
    )
    return {**render_info, **subset_info}


def validate_fixed_render_subset() -> bool:
    """Run lightweight filesystem-only checks for the fixed-view subset API."""

    def expect_error(exception_type: type[BaseException], callback: Any) -> None:
        try:
            callback()
        except exception_type:
            return
        raise AssertionError(f"Expected {exception_type.__name__}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        source_roots = [temporary_root / f"source_{index}" for index in range(4)]
        common_paths = [f"camera_{index:03d}.PNG" for index in range(19)]
        common_paths.append("nested/camera_019.webp")
        for source_root in source_roots:
            for relative_path in common_paths:
                image_path = source_root / Path(*PurePosixPath(relative_path).parts)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(b"fixed-view-test")

        common = collect_common_render_views(*source_roots)
        assert common == sorted(common_paths)
        selected = select_deterministic_view_subset(common, 5)
        expected_indices = [0, 4, 9, 14, 19]
        assert selected == tuple(common[index] for index in expected_indices)
        assert len(selected) == len(set(selected)) == 5
        assert select_deterministic_view_subset(common, 5) == selected

        manifest = temporary_root / "manifest" / "fixed_views.json"
        subset = load_or_create_fixed_view_manifest(common, 5, manifest)
        assert manifest.is_file()
        assert subset.selected_relative_paths == selected
        expanded_common = common + ["zz_new_camera.png"]
        reloaded = load_or_create_fixed_view_manifest(expanded_common, 5, manifest)
        assert reloaded.selected_relative_paths == selected
        expect_error(
            ValueError,
            lambda: load_or_create_fixed_view_manifest(common[1:], 5, manifest),
        )

        missing_one = "missing_in_compressed_gt.jpg"
        for source_root in source_roots[:3]:
            (source_root / missing_one).write_bytes(b"not-common")
        assert missing_one not in collect_common_render_views(*source_roots)

        (source_roots[0] / "position_a.png").write_bytes(b"a")
        (source_roots[1] / "position_a.png").write_bytes(b"a")
        (source_roots[2] / "position_b.png").write_bytes(b"b")
        (source_roots[3] / "position_b.png").write_bytes(b"b")
        strict_common = collect_common_render_views(*source_roots)
        assert "position_a.png" not in strict_common
        assert "position_b.png" not in strict_common

        empty_roots = [temporary_root / f"empty_{index}" for index in range(4)]
        for index, empty_root in enumerate(empty_roots):
            empty_root.mkdir()
            (empty_root / f"different_{index}.png").write_bytes(b"different")
        expect_error(ValueError, lambda: collect_common_render_views(*empty_roots))
        expect_error(
            FileNotFoundError,
            lambda: collect_common_render_views(
                empty_roots[0],
                empty_roots[1],
                empty_roots[2],
                temporary_root / "does_not_exist",
            ),
        )

        assert select_deterministic_view_subset(common[:3], 10) == tuple(common[:3])
        for invalid_count in (0, -1, True, False):
            expect_error(
                ValueError,
                lambda invalid_count=invalid_count: select_deterministic_view_subset(
                    common, invalid_count
                ),
            )

        malicious_cases = (
            ["/absolute.png"],
            ["../escape.png"],
            [common[0], common[0]],
            [""],
        )
        for case_index, malicious_selection in enumerate(malicious_cases):
            malicious_manifest = temporary_root / f"malicious_{case_index}.json"
            malicious_manifest.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "selection_strategy": _FIXED_VIEW_STRATEGY,
                        "requested_view_count": 5,
                        "available_view_count_at_creation": 20,
                        "selected_relative_paths": malicious_selection,
                    }
                ),
                encoding="utf-8",
            )
            expect_error(
                ValueError,
                lambda path=malicious_manifest: load_or_create_fixed_view_manifest(
                    common, 5, path
                ),
            )

        output_root = temporary_root / "materialized"
        materialized = materialize_fixed_view_subset(
            *source_roots,
            output_root,
            subset,
        )
        for directory in materialized.values():
            assert len([path for path in directory.rglob("*") if path.is_file()]) == 5
            assert (directory / "nested" / "camera_019.webp").is_file()
            (directory / "stale.png").write_bytes(b"stale")
        rematerialized = materialize_fixed_view_subset(
            *source_roots,
            output_root,
            subset,
        )
        for directory in rematerialized.values():
            assert not (directory / "stale.png").exists()
            assert len([path for path in directory.rglob("*") if path.is_file()]) == 5

    current_source = inspect.getsource(render_scene_pair)
    assert hashlib.sha256(current_source.encode("utf-8")).hexdigest() == (
        _LEGACY_RENDER_SCENE_PAIR_SOURCE_SHA256
    )
    legacy_globals = globals()
    original_helpers = (
        legacy_globals["run_graphdeco_render"],
        legacy_globals["find_render_dir"],
        legacy_globals["find_gt_dir"],
    )
    render_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_render(*args: Any, **kwargs: Any) -> None:
        render_calls.append((args, kwargs))

    def fake_find_render(model_path: Any, iteration: int = 30000) -> Path:
        return Path(f"render_{model_path}_{iteration}")

    def fake_find_gt(model_path: Any, iteration: int = 30000) -> Path:
        return Path(f"gt_{model_path}_{iteration}")

    try:
        legacy_globals["run_graphdeco_render"] = fake_render
        legacy_globals["find_render_dir"] = fake_find_render
        legacy_globals["find_gt_dir"] = fake_find_gt
        legacy_result = render_scene_pair(
            "graphdeco",
            "original",
            "compressed",
            "source",
            iteration=7,
            resolution=2,
            python_executable="python-test",
        )
    finally:
        (
            legacy_globals["run_graphdeco_render"],
            legacy_globals["find_render_dir"],
            legacy_globals["find_gt_dir"],
        ) = original_helpers
    assert len(render_calls) == 2
    assert render_calls[0] == (
        ("graphdeco", "original", "source"),
        {
            "iteration": 7,
            "resolution": 2,
            "split": "original",
            "python_executable": "python-test",
        },
    )
    assert render_calls[1] == (
        ("graphdeco", "compressed", "source"),
        {
            "iteration": 7,
            "resolution": 2,
            "split": "compressed",
            "python_executable": "python-test",
        },
    )
    assert legacy_result == {
        "original_render_dir": Path("render_original_7"),
        "original_gt_dir": Path("gt_original_7"),
        "compressed_render_dir": Path("render_compressed_7"),
        "compressed_gt_dir": Path("gt_compressed_7"),
    }
    return True
