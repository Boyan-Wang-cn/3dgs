"""CrossScore bridge for render -> CrossScore -> reward_D."""

from __future__ import annotations

from pathlib import Path
import json


# ---- Round 4 real CrossScore integration ----

import csv
from datetime import datetime
import re
import shlex
import subprocess
from typing import Any

import numpy as np

try:
    from .crossscore_input_utils import IMAGE_SUFFIXES, image_files, prepare_crossscore_input
except ImportError:
    from crossscore_input_utils import IMAGE_SUFFIXES, image_files, prepare_crossscore_input


DEFAULT_PREFERRED_SCORE_KEY = "pred_ssim_0_1"
FALLBACK_SCORE_KEYS = [
    "mean_score",
    "score",
    "crossscore",
    "cross_score",
    "pred_score",
    "mean_pred_score",
]
# Keys with different directions/scales such as pred_mae and pred_mse are not
# mixed automatically.  Formal training should set preferred_score_key to the
# exact CrossScore scalar used as reward_D.
SCORE_KEYS = {DEFAULT_PREFERRED_SCORE_KEY, *FALLBACK_SCORE_KEYS}


def inspect_crossscore_repo(crossscore_dir) -> dict:
    crossscore_dir = Path(crossscore_dir).resolve()
    predict_sh = crossscore_dir / "predict.sh"
    predict_preview = []
    if predict_sh.exists():
        predict_preview = predict_sh.read_text(encoding="utf-8", errors="replace").splitlines()[:40]
    ckpt_path = crossscore_dir / "ckpt" / "CrossScore-v1.0.0.ckpt"
    ckpt_is_lfs_pointer = False
    if ckpt_path.exists() and ckpt_path.stat().st_size < 1024:
        text = ckpt_path.read_text(encoding="utf-8", errors="ignore")
        ckpt_is_lfs_pointer = "git-lfs" in text
    return {
        "crossscore_dir": str(crossscore_dir),
        "exists": crossscore_dir.exists(),
        "environment_yaml": (crossscore_dir / "environment.yaml").exists(),
        "predict_sh": predict_sh.exists(),
        "predict_entry": "task/predict.py",
        "default_config": "config/default_predict.yaml",
        "query_arg": "data.dataset.query_dir",
        "reference_arg": "data.dataset.reference_dir",
        "output_arg": "logger.predict.out_dir",
        "ckpt": (crossscore_dir / "ckpt").exists(),
        "ckpt_path": str(ckpt_path),
        "ckpt_size_bytes": ckpt_path.stat().st_size if ckpt_path.exists() else None,
        "ckpt_is_lfs_pointer": ckpt_is_lfs_pointer,
        "config": (crossscore_dir / "config").exists(),
        "task": (crossscore_dir / "task").exists(),
        "model": (crossscore_dir / "model").exists(),
        "dataloading": (crossscore_dir / "dataloading").exists(),
        "expected_input": "query_dir with rendered images and reference_dir with captured/gt images",
        "needs_reference": True,
        "needs_list_file": False,
        "expected_output": "logger.predict.out_dir/score_summary/**.csv plus optional score-map images",
        "score_direction": "higher_is_better for default pred_ssim_0_1",
        "predict_sh_preview": predict_preview,
    }


def compute_crossscore_placeholder(render_dir, reference_dir) -> float:
    render_images = image_files(render_dir)
    reference_images = image_files(reference_dir)
    if not render_images:
        raise FileNotFoundError(f"No render images found in {render_dir}")
    if not reference_images:
        raise FileNotFoundError(f"No reference/gt images found in {reference_dir}")
    return 1.0


def _format_path(path: str | Path) -> str:
    return str(Path(path).resolve())


def _resolve_crossscore_file(crossscore_dir: Path, value: str | Path | None, default: Path) -> Path:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = crossscore_dir / path
    return path


def _check_ckpt(ckpt_path: Path) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"CrossScore checkpoint was not found: {ckpt_path}")
    if ckpt_path.stat().st_size < 1024:
        text = ckpt_path.read_text(encoding="utf-8", errors="ignore")
        if "git-lfs" in text:
            raise FileNotFoundError(
                f"CrossScore checkpoint is a Git LFS pointer, not real weights: {ckpt_path}. "
                "Run git lfs pull in CrossScore-main or provide a real ckpt path."
            )


def _default_command(
    crossscore_dir: Path,
    prepared: dict,
    predict_output_dir: Path,
    scene_name: str | None,
    tag: str | None,
    python_executable: str,
    ckpt_path: Path,
) -> list[str]:
    _check_ckpt(ckpt_path)
    alias = tag or scene_name or "crossscore"
    return [
        python_executable,
        "task/predict.py",
        "trainer.devices=[0]",
        f"trainer.ckpt_path_to_load={_format_path(ckpt_path)}",
        f"data.dataset.query_dir={_format_path(prepared['prepared_render_dir'])}",
        f"data.dataset.reference_dir={_format_path(prepared['prepared_reference_dir'])}",
        f"logger.predict.out_dir={_format_path(predict_output_dir)}",
        f"alias={alias}",
        "this_main.force_batch_size=True",
    ]


def _template_command(command_template: str, variables: dict[str, Any]) -> str:
    return command_template.format(**{k: str(v) for k, v in variables.items()})


def _score_output_path(base_output_dir: Path, score_output: str) -> Path:
    path = Path(score_output)
    if path.is_absolute():
        return path
    return base_output_dir / path


def compute_crossscore_real(
    crossscore_dir,
    render_dir,
    reference_dir,
    output_dir,
    scene_name=None,
    tag=None,
    python_executable="python",
    command_template: str = "",
    score_output: str = "",
    score_parse_mode: str = "auto",
    preferred_score_key: str = DEFAULT_PREFERRED_SCORE_KEY,
    ckpt: str | Path | None = None,
    config: str | Path | None = None,
    allow_image_fallback: bool = False,
) -> float:
    crossscore_dir = Path(crossscore_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (crossscore_dir / "task" / "predict.py").exists():
        raise FileNotFoundError(f"CrossScore task/predict.py was not found under {crossscore_dir}")
    stale_score = output_dir / "score.json"
    if stale_score.exists():
        stale_score.unlink()
    prepared = prepare_crossscore_input(render_dir, reference_dir, output_dir, scene_name, tag)
    predict_output_dir = Path(prepared["output_dir"])

    ckpt_path = _resolve_crossscore_file(
        crossscore_dir,
        ckpt,
        Path("ckpt") / "CrossScore-v1.0.0.ckpt",
    )
    config_path = _resolve_crossscore_file(
        crossscore_dir,
        config,
        Path("config") / "default_predict.yaml",
    )
    variables = {
        "crossscore_dir": crossscore_dir,
        "render_dir": prepared["prepared_render_dir"],
        "reference_dir": prepared["prepared_reference_dir"],
        "list_file": prepared["list_file"],
        "output_dir": output_dir,
        "predict_output_dir": predict_output_dir,
        "ckpt": ckpt_path,
        "config": config_path,
    }
    if command_template:
        command = _template_command(command_template, variables)
        run_kwargs = {"args": command, "shell": True}
        command_for_log = command
        default_score_target = output_dir
    else:
        if not config_path.exists():
            raise FileNotFoundError(f"CrossScore config was not found: {config_path}")
        command_list = _default_command(
            crossscore_dir,
            prepared,
            predict_output_dir,
            scene_name,
            tag,
            python_executable,
            ckpt_path,
        )
        run_kwargs = {"args": command_list, "shell": False}
        command_for_log = " ".join(shlex.quote(str(part)) for part in command_list)
        default_score_target = predict_output_dir

    command_path = output_dir / "crossscore_command.txt"
    stdout_path = output_dir / "crossscore_stdout.txt"
    stderr_path = output_dir / "crossscore_stderr.txt"
    command_path.write_text(command_for_log, encoding="utf-8")
    print("Running CrossScore command:")
    print(command_for_log)

    completed = subprocess.run(
        cwd=str(crossscore_dir),
        capture_output=True,
        text=True,
        **run_kwargs,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(completed.stderr, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            f"CrossScore command failed with code {completed.returncode}. "
            f"command={command_path}, stdout={stdout_path}, stderr={stderr_path}, output_dir={output_dir}"
        )

    score_target = _score_output_path(output_dir, score_output) if score_output else default_score_target
    score_info = parse_crossscore_score_info(
        score_target,
        parse_mode=score_parse_mode or "auto",
        preferred_score_key=preferred_score_key,
        allow_image_fallback=allow_image_fallback,
    )
    score = float(score_info["score"])
    stale_score.write_text(
        json.dumps(
            {
                **score_info,
                "source": str(score_target),
                "num_pairs": int(prepared["num_pairs"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return float(score)


def _first_float(text: str) -> float | None:
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else None


def _normal_score_key(key: str) -> str:
    return str(key).strip().lower()


def _score_key_priority(preferred_score_key: str | None = DEFAULT_PREFERRED_SCORE_KEY) -> list[str]:
    keys: list[str] = []
    if preferred_score_key:
        keys.append(_normal_score_key(preferred_score_key))
    for key in FALLBACK_SCORE_KEYS:
        norm = _normal_score_key(key)
        if norm not in keys:
            keys.append(norm)
    return keys


def _is_score_key(key: str, preferred_score_key: str | None = DEFAULT_PREFERRED_SCORE_KEY) -> bool:
    return _normal_score_key(key) in set(_score_key_priority(preferred_score_key))


def _collect_json_scores(
    data: Any,
    allowed_keys: set[str],
    parent_key: str = "",
) -> list[dict[str, Any]]:
    is_score_key = _normal_score_key(parent_key) in allowed_keys
    if isinstance(data, (int, float)) and is_score_key:
        return [{"score": float(data), "score_key": parent_key}]
    if isinstance(data, list):
        values: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, (int, float)) and is_score_key:
                values.append({"score": float(item), "score_key": parent_key})
            else:
                values.extend(_collect_json_scores(item, allowed_keys, parent_key))
        return values
    if isinstance(data, dict):
        values = []
        for key, value in data.items():
            values.extend(_collect_json_scores(value, allowed_keys, str(key)))
        return values
    return []


def _parse_json_file(path: Path, preferred_score_key: str | None = DEFAULT_PREFERRED_SCORE_KEY) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in _score_key_priority(preferred_score_key):
        records = _collect_json_scores(data, {key})
        if records:
            return records
    return []


def _parse_csv_file(path: Path, preferred_score_key: str | None = DEFAULT_PREFERRED_SCORE_KEY) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8", errors="replace") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            return []
        rows = list(reader)
        field_by_norm = {_normal_score_key(col): col for col in reader.fieldnames}
        for key in _score_key_priority(preferred_score_key):
            if key not in field_by_norm:
                continue
            col = field_by_norm[key]
            values = []
            for row in rows:
                try:
                    values.append({"score": float(row[col]), "score_key": col})
                except (TypeError, ValueError):
                    pass
            if values:
                return values
    return []


def _score_summary_fields(path: Path) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    for file in _official_score_files(path, "csv"):
        try:
            with file.open("r", newline="", encoding="utf-8", errors="replace") as fp:
                reader = csv.DictReader(fp)
                fields[str(file)] = list(reader.fieldnames or [])
        except OSError:
            fields[str(file)] = []
    for file in _official_score_files(path, "json"):
        fields.setdefault(str(file), ["<json: recursive key search>"])
    for file in _official_score_files(path, "txt"):
        fields.setdefault(str(file), ["<txt: first float>"])
    return fields


def _parse_image_file(path: Path) -> list[dict[str, Any]]:
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return []
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return [{"score": float(np.nanmean(arr) / 255.0), "score_key": "image_mean"}]


def _describe_output_tree(path: Path, max_entries: int = 200) -> list[dict[str, Any]]:
    if path.is_file():
        return [{"path": str(path), "type": "file", "size_bytes": int(path.stat().st_size)}]
    entries = []
    for item in sorted(path.rglob("*"))[:max_entries]:
        rel = item.relative_to(path)
        entry = {"path": str(rel), "type": "dir" if item.is_dir() else "file"}
        if item.is_file():
            entry["size_bytes"] = int(item.stat().st_size)
        entries.append(entry)
    return entries


def _official_score_files(path: Path, parse_mode: str) -> list[Path]:
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    suffixes_by_mode = {
        "json": {".json"},
        "csv": {".csv"},
        "txt": {".txt", ".log"},
    }
    allowed_suffixes = (
        suffixes_by_mode.get(parse_mode, {".json", ".csv", ".txt", ".log"})
        if parse_mode != "auto"
        else {".json", ".csv", ".txt", ".log"}
    )
    if path.is_file():
        return [path] if path.suffix.lower() in allowed_suffixes else []
    official = []
    path_is_score_summary = path.name.lower() == "score_summary"
    for file in files:
        if file.suffix.lower() not in allowed_suffixes:
            continue
        rel = file.relative_to(path)
        rel_parts = {part.lower() for part in rel.parts}
        is_root_score_json = len(rel.parts) == 1 and rel.name.lower() == "score.json"
        if path_is_score_summary or "score_summary" in rel_parts or is_root_score_json:
            official.append(file)
    return official


def parse_crossscore_score_info(
    output_dir,
    parse_mode: str = "auto",
    preferred_score_key: str = DEFAULT_PREFERRED_SCORE_KEY,
    allow_image_fallback: bool = False,
) -> dict[str, Any]:
    path = Path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"CrossScore output path does not exist: {path}")
    modes = [parse_mode] if parse_mode != "auto" else ["json", "csv", "txt"]
    records: list[dict[str, Any]] = []
    for mode in modes:
        if mode == "json":
            for file in _official_score_files(path, mode):
                for record in _parse_json_file(file, preferred_score_key=preferred_score_key):
                    records.append({**record, "score_file": str(file), "parser_mode": "json"})
        elif mode == "csv":
            for file in _official_score_files(path, mode):
                for record in _parse_csv_file(file, preferred_score_key=preferred_score_key):
                    records.append({**record, "score_file": str(file), "parser_mode": "csv"})
        elif mode == "txt":
            for file in _official_score_files(path, mode):
                parsed = _first_float(file.read_text(encoding="utf-8", errors="replace"))
                if parsed is not None:
                    records.append(
                        {
                            "score": float(parsed),
                            "score_key": "first_float",
                            "score_file": str(file),
                            "parser_mode": "txt",
                        }
                    )
        elif mode == "image":
            if not allow_image_fallback:
                raise ValueError(
                    "CrossScore image fallback is disabled. Set allow_image_fallback=True "
                    "only for debugging score-map outputs."
                )
            print(
                "WARNING: parsing CrossScore score from image mean fallback. "
                "This is for debugging only and is not an official CrossScore scalar output."
            )
            files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
            for file in files:
                if file.suffix.lower() in IMAGE_SUFFIXES:
                    for record in _parse_image_file(file):
                        records.append({**record, "score_file": str(file), "parser_mode": "image"})
        else:
            raise ValueError(f"Unsupported CrossScore parse_mode: {parse_mode}")
        if records:
            scores = [float(record["score"]) for record in records]
            first = records[0]
            return {
                "score": float(np.mean(scores)),
                "score_file": first.get("score_file", ""),
                "score_key": first.get("score_key", ""),
                "parser_mode": first.get("parser_mode", mode),
                "num_score_values": int(len(scores)),
                "preferred_score_key": str(preferred_score_key or ""),
                "score_files": sorted({str(record.get("score_file", "")) for record in records}),
            }
    tree = _describe_output_tree(path)
    score_fields = _score_summary_fields(path)
    raise ValueError(
        "Could not parse official CrossScore scalar score. Expected csv/json/txt files under "
        f"'score_summary' in {path} with parse_mode={parse_mode}, "
        f"preferred_score_key={preferred_score_key!r}, fallback_keys={FALLBACK_SCORE_KEYS}. "
        f"Score summary fields: {json.dumps(score_fields, ensure_ascii=False)[:6000]}. "
        f"Output tree: {json.dumps(tree, ensure_ascii=False)[:12000]}"
    )


def parse_crossscore_score(
    output_dir,
    parse_mode: str = "auto",
    preferred_score_key: str = DEFAULT_PREFERRED_SCORE_KEY,
    allow_image_fallback: bool = False,
) -> float:
    return float(
        parse_crossscore_score_info(
            output_dir,
            parse_mode=parse_mode,
            preferred_score_key=preferred_score_key,
            allow_image_fallback=allow_image_fallback,
        )["score"]
    )


def compute_quality_reward(
    original_score,
    compressed_score,
    higher_is_better: bool = True,
    epsilon: float = 0.0,
    dense_alpha: float = 1.0,
    violation_alpha: float = 2.0,
) -> dict:
    if higher_is_better:
        quality_drop = float(original_score) - float(compressed_score)
    else:
        quality_drop = float(compressed_score) - float(original_score)
    penalized_quality_drop = max(0.0, quality_drop - float(epsilon))
    dense_quality_drop = max(0.0, quality_drop)
    reward_D = (
        -float(dense_alpha) * dense_quality_drop
        - float(violation_alpha) * penalized_quality_drop
    )
    return {
        "quality_drop": quality_drop,
        "quality_epsilon": float(epsilon),
        "penalized_quality_drop": penalized_quality_drop,
        "dense_quality_drop": dense_quality_drop,
        "dense_alpha": float(dense_alpha),
        "violation_alpha": float(violation_alpha),
        "reward_D": reward_D,
    }


def load_score_cache(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_score_cache(path: str | Path, score: float, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"score": float(score), "created_at": datetime.now().isoformat(timespec="seconds")}
    if metadata:
        data.update(metadata)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
