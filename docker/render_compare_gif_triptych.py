import argparse
import shlex
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageSequence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, help="Left GIF path")
    parser.add_argument("--center", required=True, help="Center GIF path")
    parser.add_argument("--right", required=True, help="Right GIF path")
    parser.add_argument("--output", required=True, help="Output combined GIF path")
    parser.add_argument("--left-label", default="MAGAT")
    parser.add_argument("--center-label", default="HMAGAT")
    parser.add_argument("--right-label", default="HMAGAT-CS")
    parser.add_argument(
        "--max-width", type=int, default=None, help="Optional max output width"
    )
    return parser.parse_args()


def load_gif(path: Path):
    image = Image.open(path)
    frames = [frame.convert("RGBA") for frame in ImageSequence.Iterator(image)]
    durations = [
        frame.info.get("duration", image.info.get("duration", 100))
        for frame in ImageSequence.Iterator(image)
    ]
    total = sum(durations)
    return frames, durations, total


def frame_index_at(t, boundaries):
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= t < boundaries[i + 1]:
            return i
    return max(0, len(boundaries) - 2)


def resize_if_needed(img: Image.Image, target_width: int) -> Image.Image:
    if img.width <= target_width:
        return img
    ratio = target_width / img.width
    return img.resize((target_width, int(img.height * ratio)), Image.LANCZOS)


def get_header_height(font):
    bbox = font.getbbox("Ag")
    text_h = bbox[3] - bbox[1]
    return text_h + 28


def build_gif_with_ffmpeg(frames_dir: Path, output_path: Path, durations_ms):
    palette_path = output_path.with_suffix(".palette.png")
    manifest_path = frames_dir / "frames.txt"
    frame_paths = [frames_dir / f"frame_{idx:04d}.png" for idx in range(len(durations_ms))]
    with manifest_path.open("w", encoding="utf-8") as fh:
        for frame_path, duration_ms in zip(frame_paths, durations_ms):
            fh.write(f"file {shlex.quote(str(frame_path))}\n")
            fh.write(f"duration {duration_ms / 1000.0:.6f}\n")
        fh.write(f"file {shlex.quote(str(frame_paths[-1]))}\n")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest_path),
            "-vf",
            "palettegen",
            str(palette_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest_path),
            "-i",
            str(palette_path),
            "-lavfi",
            "paletteuse",
            str(output_path),
        ],
        check=True,
    )
    palette_path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)


def align_duration(frames, durations, total, target_total):
    if total < target_total:
        frames = frames + [frames[-1].copy()]
        durations = durations + [target_total - total]
    return frames, durations


def boundaries_from_durations(durations):
    boundaries = [0]
    for d in durations:
        boundaries.append(boundaries[-1] + d)
    return boundaries


def main():
    args = parse_args()
    left_path = Path(args.left)
    center_path = Path(args.center)
    right_path = Path(args.right)
    output_path = Path(args.output)

    left_frames, left_durations, left_total = load_gif(left_path)
    center_frames, center_durations, center_total = load_gif(center_path)
    right_frames, right_durations, right_total = load_gif(right_path)

    target_total = max(left_total, center_total, right_total)
    left_frames, left_durations = align_duration(
        left_frames, left_durations, left_total, target_total
    )
    center_frames, center_durations = align_duration(
        center_frames, center_durations, center_total, target_total
    )
    right_frames, right_durations = align_duration(
        right_frames, right_durations, right_total, target_total
    )

    left_boundaries = boundaries_from_durations(left_durations)
    center_boundaries = boundaries_from_durations(center_durations)
    right_boundaries = boundaries_from_durations(right_durations)

    boundaries = sorted(
        set(left_boundaries + center_boundaries + right_boundaries)
    )
    segments = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except OSError:
        font = ImageFont.load_default()
    header_height = get_header_height(font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="compare_gif_triptych_") as tmpdir:
        frames_dir = Path(tmpdir)
        out_idx = 0
        segment_durations_ms = []
        for start, duration in zip(boundaries[:-1], segments):
            li = frame_index_at(start, left_boundaries)
            ci = frame_index_at(start, center_boundaries)
            ri = frame_index_at(start, right_boundaries)

            lf = left_frames[li]
            cf = center_frames[ci]
            rf = right_frames[ri]

            if args.max_width is not None:
                third_width = max(64, args.max_width // 3)
                lf = resize_if_needed(lf, third_width)
                cf = resize_if_needed(cf, third_width)
                rf = resize_if_needed(rf, third_width)

            total_width = lf.width + cf.width + rf.width
            max_height = max(lf.height, cf.height, rf.height)
            canvas = Image.new(
                "RGBA",
                (total_width, max_height + header_height),
                (255, 255, 255, 255),
            )
            canvas.paste(lf, (0, header_height), lf)
            canvas.paste(cf, (lf.width, header_height), cf)
            canvas.paste(rf, (lf.width + cf.width, header_height), rf)

            draw = ImageDraw.Draw(canvas)
            div1 = lf.width
            div2 = lf.width + cf.width
            draw.line((0, header_height, total_width, header_height), fill=(80, 80, 80, 255), width=3)
            draw.line((div1, 0, div1, canvas.height), fill=(80, 80, 80, 255), width=3)
            draw.line((div2, 0, div2, canvas.height), fill=(80, 80, 80, 255), width=3)

            labels = (
                (args.left_label, 0, lf.width),
                (args.center_label, lf.width, lf.width + cf.width),
                (args.right_label, lf.width + cf.width, total_width),
            )
            for label, x0, x1 in labels:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                tx = x0 + (x1 - x0 - text_w) // 2
                ty = 10
                draw.rounded_rectangle(
                    (tx - 10, ty - 6, tx + text_w + 10, ty + text_h + 6),
                    radius=10,
                    fill=(255, 255, 255, 220),
                )
                draw.text((tx, ty), label, font=font, fill=(20, 20, 20, 255))

            canvas.save(frames_dir / f"frame_{out_idx:04d}.png")
            segment_durations_ms.append(duration)
            out_idx += 1

        build_gif_with_ffmpeg(frames_dir, output_path, segment_durations_ms)

    print(output_path)
    print(f"left_total_ms={target_total}")
    print(f"center_total_ms={target_total}")
    print(f"right_total_ms={target_total}")
    print(f"combined_frames={len(segments)}")
    print(f"combined_total_ms={sum(segments)}")


if __name__ == "__main__":
    main()
