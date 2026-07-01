"""Prepare 3DGS render outputs for CrossScore SimpleReference inference."""

from __future__ import annotations

from pathlib import Path
import shutil


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def image_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if not path.exists():
        return []
    return sorted(
        file
        for file in path.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES
    )


def _match_images(render_images: list[Path], reference_images: list[Path]) -> list[tuple[Path, Path]]:
    reference_by_name = {p.name: p for p in reference_images}
    matches = [(render, reference_by_name[render.name]) for render in render_images if render.name in reference_by_name]
    if len(matches) == len(render_images) == len(reference_images):
        return matches

    reference_by_stem = {p.stem: p for p in reference_images}
    stem_matches = [
        (render, reference_by_stem[render.stem])
        for render in render_images
        if render.stem in reference_by_stem
    ]
    if len(stem_matches) == len(render_images) == len(reference_images):
        return stem_matches

    render_sample = [p.name for p in render_images[:5]]
    reference_sample = [p.name for p in reference_images[:5]]
    raise ValueError(
        "Render/reference image sets do not match. "
        f"render_count={len(render_images)}, reference_count={len(reference_images)}, "
        f"matched_by_name={len(matches)}, matched_by_stem={len(stem_matches)}, "
        f"render_sample={render_sample}, reference_sample={reference_sample}"
    )


def prepare_crossscore_input(
    render_dir,
    reference_dir,
    work_dir,
    scene_name,
    tag,
    use_symlink: bool = False,
) -> dict:
    render_dir = Path(render_dir).resolve()
    reference_dir = Path(reference_dir).resolve()
    work_dir = Path(work_dir).resolve()
    scene_name = str(scene_name or "scene")
    tag = str(tag or "crossscore")

    render_images = image_files(render_dir)
    reference_images = image_files(reference_dir)
    if not render_images:
        raise FileNotFoundError(f"No png/jpg images found in render_dir: {render_dir}")
    if not reference_images:
        raise FileNotFoundError(f"No png/jpg images found in reference_dir: {reference_dir}")

    matches = _match_images(render_images, reference_images)
    prepared_root = work_dir / "prepared_inputs" / scene_name / tag
    prepared_render_dir = prepared_root / "query" / "test" / "ours_00000" / "renders"
    prepared_reference_dir = prepared_root / "reference" / "train" / "ours_00000" / "gt"
    output_dir = work_dir / "crossscore_outputs" / scene_name / tag
    list_file = work_dir / "input_list.txt"
    prepared_render_dir.mkdir(parents=True, exist_ok=True)
    prepared_reference_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for render_src, reference_src in matches:
        render_dst = prepared_render_dir / render_src.name
        reference_dst = prepared_reference_dir / reference_src.name
        for src, dst in [(render_src, render_dst), (reference_src, reference_dst)]:
            if dst.exists():
                continue
            if use_symlink:
                try:
                    dst.symlink_to(src)
                except OSError:
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
        rows.append(f"{render_dst}\t{reference_dst}")

    list_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {
        "prepared_render_dir": prepared_render_dir,
        "prepared_reference_dir": prepared_reference_dir,
        "list_file": list_file,
        "output_dir": output_dir,
        "num_pairs": len(matches),
        "original_render_dir": render_dir,
        "original_reference_dir": reference_dir,
    }
