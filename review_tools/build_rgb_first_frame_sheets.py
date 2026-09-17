#!/usr/bin/env python3
"""Build labeled contact sheets from cached UMI first-frame thumbnails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont, ImageStat


EPISODE_PATTERN = re.compile(r"^\d{8}_\d{6}_\d+_\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--thumbnail-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--view", default="right")
    args = parser.parse_args()
    if args.columns <= 0 or args.rows <= 0:
        parser.error("--columns and --rows must be positive")
    return args


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def main() -> int:
    args = parse_args()
    episodes = sorted(
        path.name
        for path in args.root.iterdir()
        if path.is_dir() and EPISODE_PATTERN.fullmatch(path.name)
    )
    missing = [
        episode for episode in episodes
        if not (args.thumbnail_dir / f"{episode}.first.{args.view}.jpg").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} thumbnails; first: {missing[0]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    thumb_width, thumb_height, label_height = 320, 240, 52
    card_width, card_height = thumb_width, thumb_height + label_height
    title_height = 44
    per_page = args.columns * args.rows
    title_font, label_font, metric_font = font(18), font(11), font(10)
    metrics: list[dict[str, object]] = []

    for offset in range(0, len(episodes), per_page):
        page_episodes = episodes[offset : offset + per_page]
        page_index = offset // per_page + 1
        page_count = (len(episodes) + per_page - 1) // per_page
        sheet = Image.new(
            "RGB",
            (args.columns * card_width, title_height + args.rows * card_height),
            (18, 20, 25),
        )
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (12, 10),
            f"task_v1 first RGB frame (right eye)  page {page_index}/{page_count}",
            fill=(240, 243, 247),
            font=title_font,
        )
        for slot, episode in enumerate(page_episodes):
            path = args.thumbnail_dir / f"{episode}.first.{args.view}.jpg"
            image = Image.open(path).convert("RGB").resize((thumb_width, thumb_height))
            gray = image.convert("L")
            stat = ImageStat.Stat(gray)
            mean = float(stat.mean[0])
            std = float(stat.stddev[0])
            histogram = gray.histogram()
            total = max(1, sum(histogram))
            dark_fraction = sum(histogram[:16]) / total
            bright_fraction = sum(histogram[240:]) / total
            flags: list[str] = []
            if mean < 35 or dark_fraction > 0.65:
                flags.append("DARK")
            if mean > 220 or bright_fraction > 0.65:
                flags.append("BRIGHT")
            if std < 20:
                flags.append("LOW_CONTRAST")
            metrics.append({
                "episode_id": episode,
                "mean_luma": round(mean, 3),
                "std_luma": round(std, 3),
                "dark_fraction": round(dark_fraction, 5),
                "bright_fraction": round(bright_fraction, 5),
                "flags": flags,
                "page": page_index,
                "slot": slot,
            })
            column, row = slot % args.columns, slot // args.columns
            x, y = column * card_width, title_height + row * card_height
            sheet.paste(image, (x, y))
            border = (225, 77, 85) if flags else (64, 72, 84)
            draw.rectangle((x, y, x + card_width - 1, y + card_height - 1), outline=border, width=2)
            draw.text((x + 5, y + thumb_height + 5), episode, fill=(242, 244, 247), font=label_font)
            annotation = f"mean={mean:.1f} std={std:.1f} {'/'.join(flags)}"
            draw.text((x + 5, y + thumb_height + 26), annotation, fill=border if flags else (158, 168, 181), font=metric_font)
        target = args.output_dir / f"task_v1_first_frames_{page_index:02d}.jpg"
        sheet.save(target, quality=90, optimize=True)

    (args.output_dir / "first_frame_metrics.json").write_text(
        json.dumps({"episodes": metrics}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    flagged = [entry["episode_id"] for entry in metrics if entry["flags"]]
    print(json.dumps({
        "episodes": len(episodes),
        "pages": (len(episodes) + per_page - 1) // per_page,
        "flagged_by_luma": flagged,
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
