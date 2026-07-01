from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT.parent
DEPLOY_ROOT = BASELINE_DIR
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from config_utils import load_config, normalize_crossscore_dir, normalize_gaussian_splatting_dir


def _ok(label: str, path: Path) -> bool:
    exists = path.exists()
    status = "OK" if exists else "MISSING"
    print(f"[{status}] {label}: {path}")
    return exists


def _check_ckpt(crossscore_dir: Path) -> bool:
    ckpt = crossscore_dir / "ckpt" / "CrossScore-v1.0.0.ckpt"
    if not _ok("CrossScore ckpt", ckpt):
        return False
    size = ckpt.stat().st_size
    is_pointer = False
    if size < 1024:
        text = ckpt.read_text(encoding="utf-8", errors="ignore")
        is_pointer = "git-lfs" in text
    status = "OK" if size > 1_000_000 and not is_pointer else "BAD"
    print(f"[{status}] CrossScore ckpt size: {size} bytes")
    return status == "OK"


def _check_imports() -> bool:
    try:
        import Environment_GS  # noqa: F401
        import Network_GS  # noqa: F401
        import Transition_GS  # noqa: F401
        import crossscore_bridge  # noqa: F401
        import pipeline_utils  # noqa: F401
        import gs_baseline.ply_utils  # noqa: F401
    except Exception as exc:
        print(f"[BAD] Python import check failed: {exc}")
        return False
    print("[OK] Python import check")
    return True


def main():
    parser = argparse.ArgumentParser(description="Check server deployment layout for reference_based.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config.get("paths", {})

    print(f"DEPLOY_ROOT: {DEPLOY_ROOT}")
    all_ok = True
    all_ok &= _ok("GS_DualCritic_3DGS_Baseline", BASELINE_DIR)
    all_ok &= _ok("reference_based", ROOT)
    all_ok &= _ok("gs_baseline", BASELINE_DIR / "gs_baseline")

    crossscore_dir = normalize_crossscore_dir(DEPLOY_ROOT / paths_cfg.get("crossscore_dir", "./CrossScore-main"))
    gaussian_dir = normalize_gaussian_splatting_dir(
        DEPLOY_ROOT / paths_cfg.get("gaussian_splatting_dir", "./gaussian-splatting-main")
    )
    all_ok &= _ok("CrossScore predict.sh", crossscore_dir / "predict.sh")
    all_ok &= _ok("CrossScore task/predict.py", crossscore_dir / "task" / "predict.py")
    all_ok &= _check_ckpt(crossscore_dir)
    all_ok &= _ok("gaussian-splatting render.py", gaussian_dir / "render.py")

    for scene in config.get("data", {}).get("scenes", []):
        name = scene.get("name", "scene")
        source_path = DEPLOY_ROOT / scene["source_path"]
        ply_path = DEPLOY_ROOT / scene["ply_path"]
        model_path = DEPLOY_ROOT / scene["model_path"]
        all_ok &= _ok(f"{name} source_path", source_path)
        all_ok &= _ok(f"{name} images", source_path / "images")
        all_ok &= _ok(f"{name} sparse/0", source_path / "sparse" / "0")
        all_ok &= _ok(f"{name} ply", ply_path)
        print(f"[INFO] {name} model_path will be prepared at: {model_path}")

    all_ok &= _check_imports()
    if all_ok:
        print("DEPLOY_LAYOUT_OK")
    else:
        print("DEPLOY_LAYOUT_HAS_MISSING_ITEMS")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
