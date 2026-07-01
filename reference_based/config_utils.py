"""Config and path helpers for the reference_based pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any


REFERENCE_DIR = Path(__file__).resolve().parent
BASELINE_DIR = REFERENCE_DIR.parent
CODE_ROOT = BASELINE_DIR.parent


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _simple_yaml_load(text: str) -> dict[str, Any]:
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


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        candidates = [Path.cwd() / path, REFERENCE_DIR / path, CODE_ROOT / path]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        config = yaml.safe_load(text)
    except ModuleNotFoundError:
        config = _simple_yaml_load(text)
    config["_config_path"] = str(path.resolve())
    return config


def _candidate_bases(config: dict[str, Any] | None = None) -> list[Path]:
    bases = [Path.cwd(), REFERENCE_DIR, BASELINE_DIR, CODE_ROOT]
    if config:
        project_root = config.get("project", {}).get("root")
        if project_root is not None:
            root_path = Path(project_root)
            if root_path.is_absolute():
                bases.insert(0, root_path)
            else:
                bases.insert(0, (BASELINE_DIR / root_path).resolve())
                bases.insert(1, (CODE_ROOT / root_path).resolve())
                bases.insert(2, (REFERENCE_DIR / root_path).resolve())
    deduped: list[Path] = []
    for base in bases:
        resolved = base.resolve()
        if resolved not in deduped:
            deduped.append(resolved)
    return deduped


def resolve_input_path(value: str | Path | None, config: dict[str, Any] | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    for base in _candidate_bases(config):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (_candidate_bases(config)[0] / path).resolve()


def resolve_output_path(value: str | Path, config: dict[str, Any] | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    bases = _candidate_bases(config)
    preferred_base = REFERENCE_DIR if path.parts and path.parts[0] in {"outputs", "logs", "checkpoints"} else bases[0]
    return (preferred_base / path).resolve()


def find_nested_file_dir(path: str | Path, filename: str) -> Path:
    root = Path(path)
    direct = root / filename
    if direct.exists():
        return root
    if root.exists():
        matches = list(root.glob(f"*/{filename}"))
        if len(matches) == 1:
            return matches[0].parent
    return root


def normalize_gaussian_splatting_dir(path: str | Path) -> Path:
    return find_nested_file_dir(path, "render.py")


def normalize_crossscore_dir(path: str | Path) -> Path:
    return find_nested_file_dir(path, "predict.sh")


def resolve_scene(scene: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(scene)
    for key in ["source_path", "model_path", "ply_path"]:
        if key in resolved:
            resolved[key] = str(resolve_input_path(resolved[key], config))
    return resolved


def fallback_flat_scene_ply(scene: dict[str, Any]) -> dict[str, Any]:
    ply_path = Path(scene.get("ply_path", ""))
    if ply_path.exists():
        return scene
    for fallback in [
        BASELINE_DIR / "data" / f"{scene.get('name', ply_path.stem)}.ply",
        CODE_ROOT / "data" / f"{scene.get('name', ply_path.stem)}.ply",
    ]:
        if not fallback.exists():
            continue
        scene = dict(scene)
        scene["ply_path"] = str(fallback)
        return scene
    return scene


def select_scenes(config: dict[str, Any], scene_name: str = "all") -> list[dict[str, Any]]:
    scenes = config.get("data", {}).get("scenes", [])
    if scene_name != "all":
        scenes = [scene for scene in scenes if scene.get("name") == scene_name]
        if not scenes:
            raise ValueError(f"Scene '{scene_name}' was not found in config.")
    return [resolve_scene(scene, config) for scene in scenes]
