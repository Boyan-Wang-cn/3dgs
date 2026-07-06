#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize rendered 3DGS compression cases from existing render directories.

Use this when compressed models are LightGaussian VecTree/VQ compact representations
(.npz files such as extreme_saving/*.npz). In that case, render the compressed models
with LightGaussian render.py --load_vq first, then feed the produced render directories
to this script.

Outputs:
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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def image_files(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    files: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        files.extend(folder.glob(ext))
    return sorted(files)


def image_dict(folder: str | Path) -> Dict[str, Path]:
    return {p.stem: p for p in image_files(folder)}


def ensure_image_dir(path: str | Path, name: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} does not exist: {p}")
    if len(image_files(p)) == 0:
        raise FileNotFoundError(f"{name} has no image files: {p}")
    return p


def load_rgb(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


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

    c1 = 0.01**2
    c2 = 0.03**2
    x = ref.reshape(-1, 3)
    y = img.reshape(-1, 3)
    mux, muy = x.mean(axis=0), y.mean(axis=0)
    vx, vy = x.var(axis=0), y.var(axis=0)
    cov = ((x - mux) * (y - muy)).mean(axis=0)
    s = ((2 * mux * muy + c1) * (2 * cov + c2)) / (
        (mux**2 + muy**2 + c1) * (vx + vy + c2)
    )
    return float(np.mean(s))


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
            print("[Warning] LPIPS is not installed. LPIPS will be NaN. Install with: pip install lpips")
            return

        if device == "cuda" and not torch.cuda.is_available():
            print("[Warning] CUDA unavailable. LPIPS will run on CPU.")
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
            a = (a * 2.0 - 1.0).to(self.device)
            b = (b * 2.0 - 1.0).to(self.device)
            return float(self.model(a, b).item())


def align_views(
    gt_dir: str | Path,
    original_dir: str | Path,
    good_dir: str | Path,
    bad_dir: str | Path,
) -> List[Tuple[str, Path, Path, Path, Path]]:
    gtd = image_dict(gt_dir)
    od = image_dict(original_dir)
    gd = image_dict(good_dir)
    bd = image_dict(bad_dir)

    common = sorted(set(gtd) & set(od) & set(gd) & set(bd))
    if common:
        return [(k, gtd[k], od[k], gd[k], bd[k]) for k in common]

    print("[Warning] Image names do not match. Falling back to sorted order.")
    gt_files = image_files(gt_dir)
    o_files = image_files(original_dir)
    g_files = image_files(good_dir)
    b_files = image_files(bad_dir)
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


def concat_h(images: Sequence[np.ndarray]) -> np.ndarray:
    max_h = max(im.shape[0] for im in images)
    padded = []
    for im in images:
        if im.shape[0] < max_h:
            pad = np.ones((max_h - im.shape[0], im.shape[1], 3), dtype=np.float32)
            im = np.concatenate([im, pad], axis=0)
        padded.append(im)
    return np.concatenate(padded, axis=1)


def diff_map(img: np.ndarray, ref: np.ndarray, gain: float = 4.0) -> np.ndarray:
    img = resize_to_match(img, ref.shape)
    d = np.mean(np.abs(img - ref), axis=2)
    d = np.clip(d * gain, 0, 1)
    return np.stack([d, d, d], axis=2)


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
    final = np.concatenate([row1, gap, row2], axis=0)
    save_rgb(final, out_path)


def compute_metrics(case_name, render_img, gt_img, original_img, lpips_eval):
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
    out_dir = Path(out_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Warning] matplotlib is not installed. Skip plots.")
        return

    for metric in ["psnr_vs_gt", "ssim_vs_gt", "lpips_vs_gt"]:
        if metric not in per_view_df.columns or per_view_df[metric].isna().all():
            continue
        plt.figure(figsize=(10, 4))
        for case_name in ["Original", "Good", "Bad"]:
            sub = per_view_df[per_view_df["case_name"] == case_name].sort_values("view_index")
            if len(sub) == 0:
                continue
            plt.plot(sub["view_index"], sub[metric], marker="o", linewidth=1, markersize=2, label=case_name)
        plt.xlabel("View index")
        plt.ylabel(metric)
        plt.title(f"Per-view {metric}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"per_view_{metric}.png", dpi=200)
        plt.close()

    for metric in [
        "compact_size_ratio",
        "compression_ratio",
        "crossscore",
        "quality_drop",
        "mean_psnr_vs_gt",
        "mean_ssim_vs_gt",
        "mean_lpips_vs_gt",
    ]:
        if metric not in summary_df.columns or summary_df[metric].isna().all():
            continue
        plt.figure(figsize=(6, 4))
        plt.bar(summary_df["case_name"], summary_df[metric])
        plt.ylabel(metric)
        plt.title(metric)
        plt.tight_layout()
        plt.savefig(plot_dir / f"summary_{metric}.png", dpi=200)
        plt.close()


def safe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def write_summary(summary_df: pd.DataFrame, per_view_df: pd.DataFrame, out_dir: str | Path, eps: float) -> None:
    out_dir = Path(out_dir)
    good = summary_df[summary_df["case_name"] == "Good"].iloc[0]
    bad = summary_df[summary_df["case_name"] == "Bad"].iloc[0]
    good_views = per_view_df[per_view_df["case_name"] == "Good"].sort_values("ssim_vs_gt", ascending=False).head(5)
    bad_views = per_view_df[per_view_df["case_name"] == "Bad"].sort_values("ssim_vs_gt", ascending=True).head(5)

    text = []
    text.append("# 3DGS Compression Visualization Summary\n")
    text.append(f"- SSIM implementation: `{SSIM_MODE}`")
    text.append(f"- Quality epsilon: `{eps}`")
    text.append("\n## Cases\n")
    text.append(safe_to_markdown(summary_df))
    text.append("\n## Main conclusion\n")
    text.append(
        f"- Good case quality valid: `{bool(good['quality_drop'] <= eps)}`. "
        f"compact_size_ratio={good['compact_size_ratio']:.6f}, "
        f"compression={good['compression_ratio']:.2f}x, "
        f"quality_drop={good['quality_drop']:.6f}."
    )
    text.append(
        f"- Bad case quality valid: `{bool(bad['quality_drop'] <= eps)}`. "
        f"compact_size_ratio={bad['compact_size_ratio']:.6f}, "
        f"compression={bad['compression_ratio']:.2f}x, "
        f"quality_drop={bad['quality_drop']:.6f}."
    )
    text.append("\n## Good case highest SSIM views\n")
    text.append(safe_to_markdown(good_views[["view_id", "ssim_vs_gt", "psnr_vs_gt", "lpips_vs_gt"]]))
    text.append("\n## Bad case lowest SSIM views\n")
    text.append(safe_to_markdown(bad_views[["view_id", "ssim_vs_gt", "psnr_vs_gt", "lpips_vs_gt"]]))
    text.append("\n## Interpretation\n")
    text.append(
        "The good compressed model preserves visual quality under the quality threshold while reducing model size. "
        "The bad compressed model achieves stronger compression but violates the quality constraint, so its rendered "
        "images should show more visible degradation in side-by-side comparisons and error maps."
    )

    (out_dir / "summary.md").write_text("\n".join(text), encoding="utf-8")


def build_argparser():
    parser = argparse.ArgumentParser(description="Visualize 3DGS compression cases from rendered image directories.")

    parser.add_argument("--gt-dir", required=True, help="Ground-truth image directory.")
    parser.add_argument("--original-render-dir", required=True, help="Original 3DGS render image directory.")
    parser.add_argument("--good-render-dir", required=True, help="Good compressed render image directory.")
    parser.add_argument("--bad-render-dir", required=True, help="Bad compressed render image directory.")
    parser.add_argument("--out-dir", default="/home/wby/3dgs/reference_based/viz_results/garden_case_compare_from_dirs")

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

    return parser


def main() -> None:
    args = build_argparser().parse_args()

    gt_dir = ensure_image_dir(args.gt_dir, "gt-dir")
    original_dir = ensure_image_dir(args.original_render_dir, "original-render-dir")
    good_dir = ensure_image_dir(args.good_render_dir, "good-render-dir")
    bad_dir = ensure_image_dir(args.bad_render_dir, "bad-render-dir")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Input dirs]")
    print(f"  GT      : {gt_dir}")
    print(f"  Original: {original_dir}")
    print(f"  Good    : {good_dir}")
    print(f"  Bad     : {bad_dir}")
    print(f"  Out     : {out_dir}")

    aligned = align_views(gt_dir, original_dir, good_dir, bad_dir)
    if not aligned:
        raise RuntimeError("No aligned images found.")

    print(f"[Aligned views] {len(aligned)}")

    lpips_eval = LPIPSEvaluator(enable=args.enable_lpips, device=args.lpips_device)
    side_dir = out_dir / "side_by_side"
    rep_dir = out_dir / "representative_views"
    side_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    per_view_rows = []

    for idx, (view_id, gt_p, o_p, g_p, b_p) in enumerate(aligned):
        gt = load_rgb(gt_p)
        original = load_rgb(o_p)
        good = load_rgb(g_p)
        bad = load_rgb(b_p)

        for case_name, img, path in [
            ("Original", original, o_p),
            ("Good", good, g_p),
            ("Bad", bad, b_p),
        ]:
            row = compute_metrics(case_name, img, gt, original, lpips_eval)
            row["view_index"] = idx
            row["view_id"] = view_id
            row["image_path"] = str(path)
            per_view_rows.append(row)

        if idx < args.max_side_by_side:
            make_comparison(gt, original, good, bad, side_dir / f"view_{idx:03d}_{view_id}_comparison.png")

    per_view_df = pd.DataFrame(per_view_rows)
    per_view_df.to_csv(out_dir / "per_view_metrics.csv", index=False)

    def summarize(case_name, compact_size_ratio, crossscore, quality_drop):
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

    good_best = per_view_df[per_view_df["case_name"] == "Good"].sort_values("ssim_vs_gt", ascending=False).head(3)["view_index"].tolist()
    bad_worst = per_view_df[per_view_df["case_name"] == "Bad"].sort_values("ssim_vs_gt", ascending=True).head(3)["view_index"].tolist()
    n = len(aligned)
    evenly = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))

    selected = []
    for x in good_best + bad_worst + evenly:
        x = int(x)
        if x not in selected:
            selected.append(x)

    for idx in selected:
        view_id, gt_p, o_p, g_p, b_p = aligned[idx]
        make_comparison(
            load_rgb(gt_p),
            load_rgb(o_p),
            load_rgb(g_p),
            load_rgb(b_p),
            rep_dir / f"view_{idx:03d}_{view_id}_representative.png",
        )

    make_plots(per_view_df, summary_df, out_dir)
    write_summary(summary_df, per_view_df, out_dir, args.quality_epsilon)

    print("\n[Done]")
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
