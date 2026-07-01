"""Smoke test for official CrossScore score_summary parsing."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossscore_bridge import parse_crossscore_score_info


def main() -> None:
    tmp_root = ROOT / "outputs" / "tmp_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_root / "parser_score_summary" / "predict"
    score_dir = output_dir / "score_summary"
    score_dir.mkdir(parents=True, exist_ok=True)
    csv_path = score_dir / "scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_name", "pred_ssim_0_1"])
        writer.writeheader()
        writer.writerow({"image_name": "00001.png", "pred_ssim_0_1": "0.8"})
        writer.writerow({"image_name": "00002.png", "pred_ssim_0_1": "0.6"})

    info = parse_crossscore_score_info(output_dir)
    assert abs(float(info["score"]) - 0.7) < 1e-9, info
    assert info["parser_mode"] == "csv", info
    assert info["score_key"] == "pred_ssim_0_1", info
    assert info["score_file"].endswith("scores.csv"), info

    image_dir = tmp_root / "parser_image_only"
    image_dir.mkdir(parents=True, exist_ok=True)
    fake_png = image_dir / "score_map.png"
    try:
        from PIL import Image
    except ModuleNotFoundError:
        fake_png.write_bytes(b"not a real image")
        try:
            parse_crossscore_score_info(image_dir)
        except ValueError:
            print("PIL unavailable; image fallback enable-path skipped.")
        else:
            raise AssertionError("image-only output should not parse by default")
    else:
        Image.new("L", (2, 2), color=128).save(fake_png)
        try:
            parse_crossscore_score_info(image_dir)
        except ValueError:
            pass
        else:
            raise AssertionError("image-only output should not parse by default")
        fallback = parse_crossscore_score_info(
            image_dir,
            parse_mode="image",
            allow_image_fallback=True,
        )
        assert fallback["parser_mode"] == "image", fallback
        assert fallback["score_key"] == "image_mean", fallback

    print("CrossScore parser smoke test passed.")


if __name__ == "__main__":
    main()
