from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import normalize_crossscore_dir
from crossscore_bridge import compute_crossscore_real


def _print_debug_paths(output_dir: Path) -> None:
    for name in ["crossscore_command.txt", "crossscore_stdout.txt", "crossscore_stderr.txt", "score.json"]:
        path = output_dir / name
        status = "exists" if path.exists() else "missing"
        print(f"{name}: {path} ({status})")
    if output_dir.exists():
        files = sorted(path for path in output_dir.rglob("*") if path.is_file())
        print("output_files:")
        for path in files[:100]:
            print(f"  {path.relative_to(output_dir)}")
        if len(files) > 100:
            print(f"  ... {len(files) - 100} more files")


def main():
    parser = argparse.ArgumentParser(description="Run CrossScore on one render/reference directory pair.")
    parser.add_argument("--crossscore-dir", required=True)
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-name", default="debug_scene")
    parser.add_argument("--tag", default="debug_pair")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--command-template", default="")
    parser.add_argument("--score-output", default="")
    parser.add_argument("--score-parse-mode", default="auto")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    crossscore_dir = normalize_crossscore_dir(args.crossscore_dir)

    try:
        score = compute_crossscore_real(
            crossscore_dir,
            args.render_dir,
            args.reference_dir,
            output_dir,
            scene_name=args.scene_name,
            tag=args.tag,
            python_executable=args.python_executable,
            command_template=args.command_template,
            score_output=args.score_output,
            score_parse_mode=args.score_parse_mode,
            ckpt=args.ckpt,
            config=args.config,
        )
    except Exception as exc:
        print(f"CrossScore failed: {exc}", file=sys.stderr)
        print("Debug artifacts:")
        _print_debug_paths(output_dir)
        print("Check predict.sh, checkpoint path, Hydra overrides, and output score format.")
        raise SystemExit(1) from exc

    print(f"score: {score:.8f}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
