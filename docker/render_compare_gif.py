import argparse
import shlex
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageSequence, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', required=True, help='Left GIF path')
    parser.add_argument('--right', required=True, help='Right GIF path')
    parser.add_argument('--output', required=True, help='Output combined GIF path')
    parser.add_argument('--left-label', default='MAGAT')
    parser.add_argument('--right-label', default='HMAGAT')
    parser.add_argument('--max-width', type=int, default=None, help='Optional max output width')
    return parser.parse_args()


def load_gif(path: Path):
    image = Image.open(path)
    frames = [frame.convert('RGBA') for frame in ImageSequence.Iterator(image)]
    durations = [frame.info.get('duration', image.info.get('duration', 100)) for frame in ImageSequence.Iterator(image)]
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


def main():
    args = parse_args()
    left_path = Path(args.left)
    right_path = Path(args.right)
    output_path = Path(args.output)

    left_frames, left_durations, left_total = load_gif(left_path)
    right_frames, right_durations, right_total = load_gif(right_path)

    if left_total < right_total:
        left_frames.append(left_frames[-1].copy())
        left_durations.append(right_total - left_total)
    elif right_total < left_total:
        right_frames.append(right_frames[-1].copy())
        right_durations.append(left_total - right_total)

    left_boundaries = [0]
    for d in left_durations:
        left_boundaries.append(left_boundaries[-1] + d)
    right_boundaries = [0]
    for d in right_durations:
        right_boundaries.append(right_boundaries[-1] + d)

    boundaries = sorted(set(left_boundaries + right_boundaries))
    segments = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
    except OSError:
        font = ImageFont.load_default()
    header_height = get_header_height(font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="compare_gif_") as tmpdir:
        frames_dir = Path(tmpdir)
        out_idx = 0
        segment_durations_ms = []
        for idx, (start, duration) in enumerate(zip(boundaries[:-1], segments)):
            li = frame_index_at(start, left_boundaries)
            ri = frame_index_at(start, right_boundaries)
            lf = left_frames[li]
            rf = right_frames[ri]

            if args.max_width is not None:
                half_width = max(64, args.max_width // 2)
                lf = resize_if_needed(lf, half_width)
                rf = resize_if_needed(rf, half_width)

            canvas = Image.new(
                'RGBA',
                (lf.width + rf.width, max(lf.height, rf.height) + header_height),
                (255, 255, 255, 255),
            )
            canvas.paste(lf, (0, header_height), lf)
            canvas.paste(rf, (lf.width, header_height), rf)

            draw = ImageDraw.Draw(canvas)
            divider_x = lf.width
            draw.line((0, header_height, canvas.width, header_height), fill=(80, 80, 80, 255), width=3)
            draw.line((divider_x, 0, divider_x, canvas.height), fill=(80, 80, 80, 255), width=3)

            for label, x0, x1 in ((args.left_label, 0, lf.width), (args.right_label, lf.width, lf.width + rf.width)):
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                tx = x0 + (x1 - x0 - text_w) // 2
                ty = 10
                draw.rounded_rectangle((tx - 10, ty - 6, tx + text_w + 10, ty + text_h + 6), radius=10, fill=(255, 255, 255, 220))
                draw.text((tx, ty), label, font=font, fill=(20, 20, 20, 255))

            canvas.save(frames_dir / f"frame_{out_idx:04d}.png")
            segment_durations_ms.append(duration)
            out_idx += 1

        build_gif_with_ffmpeg(frames_dir, output_path, segment_durations_ms)

    print(output_path)
    print(f'left_total_ms={left_total}')
    print(f'right_total_ms={right_total}')
    print(f'combined_frames={len(segments)}')
    print(f'combined_total_ms={sum(segments)}')


if __name__ == '__main__':
    main()
