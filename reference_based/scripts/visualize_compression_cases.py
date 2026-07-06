#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize and evaluate 3DGS compression cases.

Put this file at:
  reference_based/scripts/visualize_compression_cases.py

It restores compact zip packages into Graphdeco-renderable model dirs,
renders original/good/bad models, computes PSNR/SSIM/LPIPS, and exports:
  - metrics_summary.csv
  - per_view_metrics.csv
  - side_by_side/*.png
  - representative_views/*.png
  - plots/*.png
  - summary.md
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def try_import_ssim():
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim, "skimage"
    except Exception:
        return None, "fallback_global_ssim"


def try_import_lpips():
    try:
        import torch
        import lpips
        return torch, lpips
    except Exception:
        return None, None


SSIM_FUNC, SSIM_MODE = try_import_ssim()


def run_cmd(cmd: Sequence[str | Path], cwd: Optional[str | Path] = None) -> None:
    cmd = [str(x) for x in cmd]
    print("\n[Run]")
    print(" ".join(cmd))
    ret = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(ret.stdout)
    if ret.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {ret.returncode}: {' '.join(cmd)}\n{ret.stdout}"
        )


def ensure_exists(path: str | Path, name: str = "path") -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} does not exist: {p}")
    return p


def image_files(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    files: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        files.extend(folder.glob(ext))
    return sorted(files)


def image_dict(folder: str | Path) -> Dict[str, Path]:
    return {p.stem: p for p in image_files(folder)}


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def save_rgb(arr: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def resize_to_match(img: np.ndarray, ref_shape: Tuple[int, int, int]) -> np.ndarray:
    h, w = ref_shape[:2]
    if img.shape[:2] == (h, w):
        return img
    pil = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
    pil = pil.resize((w, h), Image.BICUBIC)
    return np.asarray(pil).astype(np.float32) / 255.0


def psnr(img: np.ndarray, ref: np.ndarray) -> float:
    img = resize_to_match(img, ref.shape)
    mse = float(np.mean((img - ref) ** 2))
    if mse <= 1e-12:
        return 100.0
    return -10.0 * math.log10(mse)


def ssim_metric(img: np.ndarray, ref: np.ndarray) -> float:
    img = resize_to_match(img, ref.shape)
    if SSIM_FUNC is not None:
        return float(SSIM_FUNC(ref, img, data_range=1.0, channel_axis=2))

    # Fallback global SSIM; less standard than skimage but avoids hard failure.
    c1, c2 = 0.01**2, 0.03**2
    x, y = ref.reshape(-1, 3), img.reshape(-1, 3)
    mux, muy = x.mean(axis=0), y.mean(axis=0)
    vx, vy = x.var(axis=0), y.var(axis=0)
    cov = ((x - mux) * (y - muy)).mean(axis=0)
    val = ((2 * mux * muy + c1) * (2 * cov + c2)) / (
        (mux**2 + muy**2 + c1) * (vx + vy + c2)
    )
    return float(np.mean(val))


class LPIPSEvaluator:
    def __init__(self, enable: bool = False, device: str = "cuda") -> None:
        self.enabled = False
        self.model = None
        self.torch = None
        self.device = device
        if not enable:
            return
        torch, lpips = try_import_lpips()
        if torch is None or lpips is None:
            print("[Warning] LPIPS not installed. LPIPS values will be NaN. Install: pip install lpips")
            return
        if device == "cuda" and not torch.cuda.is_available():
            print("[Warning] CUDA unavailable for LPIPS. Use CPU.")
            device = "cpu"
        self.device = device
        self.torch = torch
        self.model = lpips.LPIPS(net="alex").to(device)
        self.model.eval()
        self.enabled = True

    def __call__(self, img: np.ndarray, ref: np.ndarray) -> float:
        if not self.enabled:
            return float("nan")
        img = resize_to_match(img, ref.shape)
        torch = self.torch
        with torch.no_grad():
            a = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
            b = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0).float()
            a, b = a * 2.0 - 1.0, b * 2.0 - 1.0
            val = self.model(a.to(self.device), b.to(self.device))
            return float(val.item())


def find_largest_ply(root: str | Path) -> Path:
    plys = list(Path(root).rglob("*.ply"))
    if not plys:
        raise FileNotFoundError(f"No .ply found under {root}")
    return sorted(plys, key=lambda p: p.stat().st_size, reverse=True)[0]


def extract_zip(zip_path: str | Path, extract_dir: str | Path, force: bool = False) -> Path:
    zip_path = ensure_exists(zip_path, "compact zip")
    extract_dir = Path(extract_dir)
    marker = extract_dir / ".extracted_ok"
    if marker.exists() and not force:
        print(f"[Reuse] extracted zip: {extract_dir}")
        return extract_dir
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Extract] {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    marker.write_text(str(zip_path), encoding="utf-8")
    return extract_dir


def prepare_renderable_model_from_zip(
    case_name: str,
    zip_path: str | Path,
    original_model: str | Path,
    out_dir: str | Path,
    iteration: int,
    force: bool = False,
) -> Path:
    original_model = ensure_exists(original_model, "original model")
    out_dir = Path(out_dir)
    extracted_dir = out_dir / "extracted_zips" / case_name
    model_dir = out_dir / "restored_models" / case_name
    target_ply = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"

    if target_ply.exists() and not force:
        print(f"[Reuse] restored model: {model_dir}")
        return model_dir

    extract_zip(zip_path, extracted_dir, force=force)
    src_ply = find_largest_ply(extracted_dir)

    if model_dir.exists():
        shutil.rmtree(model_dir)
    target_ply.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ply, target_ply)

    src_cfg = original_model / "cfg_args"
    if src_cfg.exists():
        shutil.copy2(src_cfg, model_dir / "cfg_args")
    else:
        print(f"[Warning] original cfg_args not found: {src_cfg}")

    print(f"[Restore] {case_name}")
    print(f"  source ply: {src_ply}")
    print(f"  target ply: {target_ply}")
    return model_dir


def expected_render_dirs(model_dir: str | Path, iteration: int, split: str) -> List[Path]:
    model_dir = Path(model_dir)
    return [
        model_dir / split / f"ours_{iteration}" / "renders",
        model_dir / split / "ours_30000" / "renders",
        model_dir / split / "ours_00000" / "renders",
        model_dir / "renders",
    ]


def expected_gt_dirs(model_dir: str | Path, iteration: int, split: str) -> List[Path]:
    model_dir = Path(model_dir)
    return [
        model_dir / split / f"ours_{iteration}" / "gt",
        model_dir / split / "ours_30000" / "gt",
        model_dir / split / "ours_00000" / "gt",
        model_dir / "gt",
    ]


def first_existing_image_dir(paths: Iterable[str | Path]) -> Optional[Path]:
    for p in paths:
        p = Path(p)
        if p.exists() and image_files(p):
            return p
    return None


def render_model(
    model_dir: str | Path,
    scene_path: str | Path,
    gaussian_splatting_dir: str | Path,
    iteration: int,
    resolution: int,
    python_exe: str | Path,
    split: str,
    force: bool = False,
) -> Tuple[Path, Path]:
    model_dir = ensure_exists(model_dir, "model_dir")
    scene_path = ensure_exists(scene_path, "scene_path")
    gs_dir = ensure_exists(gaussian_splatting_dir, "gaussian_splatting_dir")
    if not (gs_dir / "render.py").exists():
        raise FileNotFoundError(f"render.py not found: {gs_dir / 'render.py'}")

    render_dir = first_existing_image_dir(expected_render_dirs(model_dir, iteration, split))
    gt_dir = first_existing_image_dir(expected_gt_dirs(model_dir, iteration, split))
    if render_dir is not None and gt_dir is not None and not force:
        print(f"[Reuse] rendered output: {model_dir}")
        print(f"  render_dir: {render_dir}")
        print(f"  gt_dir    : {gt_dir}")
        return render_dir, gt_dir

    cmd = [
        str(python_exe),
        "render.py",
        "-m", str(model_dir),
        "-s", str(scene_path),
        "--iteration", str(iteration),
        "-r", str(resolution),
    ]
    run_cmd(cmd, cwd=gs_dir)

    render_dir = first_existing_image_dir(expected_render_dirs(model_dir, iteration, split))
    gt_dir = first_existing_image_dir(expected_gt_dirs(model_dir, iteration, split))
    if render_dir is None:
        raise FileNotFoundError(f"Could not find rendered images under {model_dir}")
    if gt_dir is None:
        raise FileNotFoundError(f"Could not find GT images under {model_dir}")
    return render_dir, gt_dir


def align_views(gt_dir: str | Path, orig_dir: str | Path, good_dir: str | Path, bad_dir: str | Path):
    gtd, od, gd, bd = image_dict(gt_dir), image_dict(orig_dir), image_dict(good_dir), image_dict(bad_dir)
    common = sorted(set(gtd) & set(od) & set(gd) & set(bd))
    if common:
        return [(k, gtd[k], od[k], gd[k], bd[k]) for k in common]
    print("[Warning] image names do not match. Falling back to sorted order.")
    gt_files, o_files, g_files, b_files = image_files(gt_dir), image_files(orig_dir), image_files(good_dir), image_files(bad_dir)
    n = min(len(gt_files), len(o_files), len(g_files), len(b_files))
    return [(f"{i:05d}", gt_files[i], o_files[i], g_files[i], b_files[i]) for i in range(n)]


def get_font(size: int = 18):
    for c in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            pass
    return ImageFont.load_default()


def add_label(img: np.ndarray, text: str, height: int = 36) -> np.ndarray:
    pil = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
    w, h = pil.size
    canvas = Image.new("RGB", (w, h + height), "white")
    canvas.paste(pil, (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), text, fill=(0, 0, 0), font=get_font(18))
    return np.asarray(canvas).astype(np.float32) / 255.0


def diff_map(img: np.ndarray, ref: np.ndarray, gain: float = 4.0) -> np.ndarray:
    img = resize_to_match(img, ref.shape)
    d = np.mean(np.abs(img - ref), axis=2)
    d = np.clip(d * gain, 0, 1)
    return np.stack([d, d, d], axis=2)


def concat_h(images: Sequence[np.ndarray]) -> np.ndarray:
    max_h = max(im.shape[0] for im in images)
    out = []
    for im in images:
        if im.shape[0] < max_h:
            pad = np.ones((max_h - im.shape[0], im.shape[1], 3), dtype=np.float32)
            im = np.concatenate([im, pad], axis=0)
        out.append(im)
    return np.concatenate(out, axis=1)


def make_comparison(gt, original, good, bad, out_path: str | Path) -> None:
    original = resize_to_match(original, gt.shape)
    good = resize_to_match(good, gt.shape)
    bad = resize_to_match(bad, gt.shape)

    row1 = concat_h([
        add_label(gt, "GT"),
        add_label(original, "Original render"),
        add_label(good, "Good compression"),
        add_label(bad, "Bad compression"),
    ])
    blank = np.ones_like(gt)
    row2 = concat_h([
        add_label(blank, "Diff maps"),
        add_label(diff_map(original, gt), "|Original - GT|"),
        add_label(diff_map(good, gt), "|Good - GT|"),
        add_label(diff_map(bad, gt), "|Bad - GT|"),
    ])
    gap = np.ones((12, row1.shape[1], 3), dtype=np.float32)
    save_rgb(np.concatenate([row1, gap, row2], axis=0), out_path)


def compute_case_metrics(case_name, render_img, gt_img, original_img, lpips_eval):
    render_img = resize_to_match(render_img, gt_img.shape)
    original_img = resize_to_match(original_img, gt_img.shape)
    return {
        "case_name": case_name,
        "psnr_vs_gt": psnr(render_img, gt_img),
        "ssim_vs_gt": ssim_metric(render_img, gt_img),
        "lpips_vs_gt": lpips_eval(render_img, gt_img),
        "psnr_vs_original": psnr(render_img, original_img),
        "ssim_vs_original": ssim_metric(render_img, original_img),
        "lpips_vs_original": lpips_eval(render_img, original_img),
    }


def make_plots(per_view_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: str | Path) -> None:
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Warning] matplotlib not installed. Skip plots.")
        return

    for metric in ["psnr_vs_gt", "ssim_vs_gt", "lpips_vs_gt"]:
        if metric not in per_view_df or per_view_df[metric].isna().all():
            continue
        plt.figure(figsize=(10, 4))
        for case_name in ["Original", "Good", "Bad"]:
            sub = per_view_df[per_view_df["case_name"] == case_name].sort_values("view_index")
            if len(sub):
                plt.plot(sub["view_index"], sub[metric], marker="o", linewidth=1, markersize=2, label=case_name)
        plt.xlabel("View index")
        plt.ylabel(metric)
        plt.title(f"Per-view {metric}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"per_view_{metric}.png", dpi=200)
        plt.close()

    for metric in ["compact_size_ratio", "compression_ratio", "crossscore", "quality_drop", "mean_psnr_vs_gt", "mean_ssim_vs_gt", "mean_lpips_vs_gt"]:
        if metric not in summary_df or summary_df[metric].isna().all():
            continue
        plt.figure(figsize=(6, 4))
        plt.bar(summary_df["case_name"], summary_df[metric])
        plt.ylabel(metric)
        plt.title(metric)
        plt.tight_layout()
        plt.savefig(plot_dir / f"summary_{metric}.png", dpi=200)
        plt.close()


def safe_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def write_summary(summary_df: pd.DataFrame, per_view_df: pd.DataFrame, out_dir: str | Path, quality_epsilon: float) -> None:
    out_dir = Path(out_dir)
    good = summary_df[summary_df["case_name"] == "Good"].iloc[0]
    bad = summary_df[summary_df["case_name"] == "Bad"].iloc[0]
    good_views = per_view_df[per_view_df["case_name"] == "Good"].sort_values("ssim_vs_gt", ascending=False).head(5)
    bad_views = per_view_df[per_view_df["case_name"] == "Bad"].sort_values("ssim_vs_gt", ascending=True).head(5)

    text = [
        "# 3DGS Compression Visualization Summary\n",
        f"- SSIM implementation: `{SSIM_MODE}`",
        f"- Quality epsilon: `{quality_epsilon}`",
        "\n## Cases\n",
        safe_markdown(summary_df),
        "\n## Main conclusion\n",
        f"- Good case quality valid: `{bool(good['quality_drop'] <= quality_epsilon)}`. compact_size_ratio={good['compact_size_ratio']:.6f}, compression={good['compression_ratio']:.2f}x, quality_drop={good['quality_drop']:.6f}.",
        f"- Bad case quality valid: `{bool(bad['quality_drop'] <= quality_epsilon)}`. compact_size_ratio={bad['compact_size_ratio']:.6f}, compression={bad['compression_ratio']:.2f}x, quality_drop={bad['quality_drop']:.6f}.",
        "\n## Good case highest SSIM views\n",
        safe_markdown(good_views[["view_id", "ssim_vs_gt", "psnr_vs_gt", "lpips_vs_gt"]]),
        "\n## Bad case lowest SSIM views\n",
        safe_markdown(bad_views[["view_id", "ssim_vs_gt", "psnr_vs_gt", "lpips_vs_gt"]]),
        "\n## Interpretation\n",
        "The good compressed model preserves visual quality under the quality threshold while reducing model size. The bad compressed model achieves stronger compression but violates the quality constraint, so it should show more visible degradation in side-by-side comparisons and error maps.",
    ]
    (out_dir / "summary.md").write_text("\n".join(text), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize original/good/bad 3DGS compression cases.")
    parser.add_argument("--original-model", default="/home/wby/GS_DualCritic_3DGS_Baseline/output/garden_trained")
    parser.add_argument("--scene-path", default="/home/wby/GS_DualCritic_3DGS_Baseline/data/mipnerf360/garden")
    parser.add_argument("--good-zip", default="/home/wby/3dgs/reference_based/outputs/point_cloud_reference_based_ep0016_20260705_222430_compact.zip")
    parser.add_argument("--bad-zip", default="/home/wby/3dgs/reference_based/outputs/point_cloud_reference_based_ep0020_20260705_223742_compact.zip")
    parser.add_argument("--out-dir", default="/home/wby/3dgs/reference_based/viz_results/garden_case_compare")
    parser.add_argument("--gaussian-splatting-dir", default="/home/wby/3dgs/gaussian-splatting-main/gaussian-splatting-main")
    parser.add_argument("--python-exe", default="/home/wby/miniconda3/envs/3dgs/bin/python")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--original-crossscore", type=float, default=0.887459)
    parser.add_argument("--good-compact-size-ratio", type=float, default=0.308368)
    parser.add_argument("--good-crossscore", type=float, default=0.884786)
    parser.add_argument("--good-quality-drop", type=float, default=0.002673)
    parser.add_argument("--bad-compact-size-ratio", type=float, default=0.083120)
    parser.add_argument("--bad-crossscore", type=float, default=0.779982)
    parser.add_argument("--bad-quality-drop", type=float, default=0.107477)
    parser.add_argument("--quality-epsilon", type=float, default=0.05)
    parser.add_argument("--max-side-by-side", type=int, default=12)
    parser.add_argument("--enable-lpips", action="store_true")
    parser.add_argument("--lpips-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--force-render", action="store_true")
    parser.add_argument("--force-restore", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_model = ensure_exists(args.original_model, "original model")
    scene_path = ensure_exists(args.scene_path, "scene path")
    good_zip = ensure_exists(args.good_zip, "good compact zip")
    bad_zip = ensure_exists(args.bad_zip, "bad compact zip")

    print("[Config]")
    print(f"  original_model: {original_model}")
    print(f"  scene_path    : {scene_path}")
    print(f"  good_zip      : {good_zip}")
    print(f"  bad_zip       : {bad_zip}")
    print(f"  out_dir       : {out_dir}")

    print("\n[Step 1] Restore compressed models")
    good_model = prepare_renderable_model_from_zip("good", good_zip, original_model, out_dir, args.iteration, force=args.force_restore)
    bad_model = prepare_renderable_model_from_zip("bad", bad_zip, original_model, out_dir, args.iteration, force=args.force_restore)

    print("\n[Step 2] Render original / good / bad")
    original_render_dir, original_gt_dir = render_model(original_model, scene_path, args.gaussian_splatting_dir, args.iteration, args.resolution, args.python_exe, args.split, force=args.force_render)
    good_render_dir, _ = render_model(good_model, scene_path, args.gaussian_splatting_dir, args.iteration, args.resolution, args.python_exe, args.split, force=args.force_render)
    bad_render_dir, _ = render_model(bad_model, scene_path, args.gaussian_splatting_dir, args.iteration, args.resolution, args.python_exe, args.split, force=args.force_render)

    gt_dir = original_gt_dir
    print("\n[Render dirs]")
    print(f"  GT      : {gt_dir}")
    print(f"  Original: {original_render_dir}")
    print(f"  Good    : {good_render_dir}")
    print(f"  Bad     : {bad_render_dir}")

    aligned = align_views(gt_dir, original_render_dir, good_render_dir, bad_render_dir)
    if not aligned:
        raise RuntimeError("No aligned views found.")
    print(f"\n[Step 3] Aligned views: {len(aligned)}")

    lpips_eval = LPIPSEvaluator(enable=args.enable_lpips, device=args.lpips_device)
    side_dir = out_dir / "side_by_side"
    rep_dir = out_dir / "representative_views"
    side_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    per_view_rows: List[Dict[str, object]] = []
    for idx, (view_id, gt_p, o_p, g_p, b_p) in enumerate(aligned):
        gt, original, good, bad = load_rgb(gt_p), load_rgb(o_p), load_rgb(g_p), load_rgb(b_p)
        for case_name, img, path in [("Original", original, o_p), ("Good", good, g_p), ("Bad", bad, b_p)]:
            row = compute_case_metrics(case_name, img, gt, original, lpips_eval)
            row["view_index"] = idx
            row["view_id"] = view_id
            row["image_path"] = str(path)
            per_view_rows.append(row)
        if idx < args.max_side_by_side:
            make_comparison(gt, original, good, bad, side_dir / f"view_{idx:03d}_{view_id}_comparison.png")

    per_view_df = pd.DataFrame(per_view_rows)
    per_view_df.to_csv(out_dir / "per_view_metrics.csv", index=False)

    def summarize(case_name: str, compact_size_ratio: float, crossscore: float, quality_drop: float) -> Dict[str, object]:
        sub = per_view_df[per_view_df["case_name"] == case_name]
        return {
            "case_name": case_name,
            "compact_size_ratio": compact_size_ratio,
            "compression_ratio": 1.0 / compact_size_ratio,
            "crossscore": crossscore,
            "quality_drop": quality_drop,
            "quality_valid": quality_drop <= args.quality_epsilon,
            "mean_psnr_vs_gt": sub["psnr_vs_gt"].mean(),
            "mean_ssim_vs_gt": sub["ssim_vs_gt"].mean(),
            "mean_lpips_vs_gt": sub["lpips_vs_gt"].mean(),
            "mean_psnr_vs_original": sub["psnr_vs_original"].mean(),
            "mean_ssim_vs_original": sub["ssim_vs_original"].mean(),
            "mean_lpips_vs_original": sub["lpips_vs_original"].mean(),
        }

    summary_df = pd.DataFrame([
        summarize("Original", 1.0, args.original_crossscore, 0.0),
        summarize("Good", args.good_compact_size_ratio, args.good_crossscore, args.good_quality_drop),
        summarize("Bad", args.bad_compact_size_ratio, args.bad_crossscore, args.bad_quality_drop),
    ])
    summary_df.to_csv(out_dir / "metrics_summary.csv", index=False)

    good_best = per_view_df[per_view_df["case_name"] == "Good"].sort_values("ssim_vs_gt", ascending=False).head(3)["view_index"].astype(int).tolist()
    bad_worst = per_view_df[per_view_df["case_name"] == "Bad"].sort_values("ssim_vs_gt", ascending=True).head(3)["view_index"].astype(int).tolist()
    n = len(aligned)
    evenly = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    selected: List[int] = []
    for x in good_best + bad_worst + evenly:
        if x not in selected:
            selected.append(x)

    for idx in selected:
        view_id, gt_p, o_p, g_p, b_p = aligned[idx]
        make_comparison(load_rgb(gt_p), load_rgb(o_p), load_rgb(g_p), load_rgb(b_p), rep_dir / f"view_{idx:03d}_{view_id}_representative.png")

    make_plots(per_view_df, summary_df, out_dir)
    write_summary(summary_df, per_view_df, out_dir, args.quality_epsilon)

    print("\n[Done]")
    print(f"  output dir           : {out_dir}")
    print(f"  metrics_summary.csv  : {out_dir / 'metrics_summary.csv'}")
    print(f"  per_view_metrics.csv : {out_dir / 'per_view_metrics.csv'}")
    print(f"  side_by_side         : {side_dir}")
    print(f"  representative_views : {rep_dir}")
    print(f"  plots                : {out_dir / 'plots'}")
    print(f"  summary.md           : {out_dir / 'summary.md'}")
    print("\n[Metrics summary]")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
