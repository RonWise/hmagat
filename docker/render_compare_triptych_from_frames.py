import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-frames", required=True, help="Left PNG frames dir")
    parser.add_argument("--center-frames", required=True, help="Center PNG frames dir")
    parser.add_argument("--right-frames", required=True, help="Right PNG frames dir")
    parser.add_argument("--output-gif", required=True, help="Output GIF path")
    parser.add_argument("--output-mp4", required=True, help="Output MP4 path")
    parser.add_argument("--left-label", default="MAGAT")
    parser.add_argument("--center-label", default="HMAGAT")
    parser.add_argument("--right-label", default="HMAGAT-CS")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--max-width", type=int, default=1152)
    return parser.parse_args()


def list_frames(path: Path):
    frames = sorted(path.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames found in {path}")
    return frames


def resize_if_needed(img: Image.Image, target_width: int) -> Image.Image:
    if img.width <= target_width:
        return img
    ratio = target_width / img.width
    return img.resize((target_width, int(img.height * ratio)), Image.LANCZOS)


def load_font():
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except OSError:
        return ImageFont.load_default()


def draw_panel_label(draw, label, x0, x1, font):
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


def get_header_height(font):
    bbox = font.getbbox("Ag")
    text_h = bbox[3] - bbox[1]
    return text_h + 28


def build_gif(frames_dir: Path, output_gif: Path, fps: int):
    palette = output_gif.with_suffix(".palette.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-vf",
            "palettegen",
            str(palette),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-i",
            str(palette),
            "-lavfi",
            "paletteuse",
            str(output_gif),
        ],
        check=True,
    )
    palette.unlink(missing_ok=True)


def build_mp4(frames_dir: Path, output_mp4: Path, fps: int):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-vf",
            "format=yuv420p",
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            str(output_mp4),
        ],
        check=True,
    )


def main():
    args = parse_args()
    left_frames = list_frames(Path(args.left_frames))
    center_frames = list_frames(Path(args.center_frames))
    right_frames = list_frames(Path(args.right_frames))

    output_gif = Path(args.output_gif)
    output_mp4 = Path(args.output_mp4)
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    target_count = max(len(left_frames), len(center_frames), len(right_frames))
    font = load_font()
    header_height = get_header_height(font)

    with tempfile.TemporaryDirectory(prefix="triptych_frames_") as tmpdir:
        frames_dir = Path(tmpdir)
        for out_idx in range(target_count):
            left_path = left_frames[min(out_idx, len(left_frames) - 1)]
            center_path = center_frames[min(out_idx, len(center_frames) - 1)]
            right_path = right_frames[min(out_idx, len(right_frames) - 1)]

            with Image.open(left_path).convert("RGBA") as lf, Image.open(
                center_path
            ).convert("RGBA") as cf, Image.open(right_path).convert("RGBA") as rf:
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

                draw_panel_label(draw, args.left_label, 0, lf.width, font)
                draw_panel_label(draw, args.center_label, lf.width, lf.width + cf.width, font)
                draw_panel_label(draw, args.right_label, lf.width + cf.width, total_width, font)

                canvas.save(frames_dir / f"frame_{out_idx:04d}.png")

        build_gif(frames_dir, output_gif, args.fps)
        build_mp4(frames_dir, output_mp4, args.fps)

    print(f"output_gif={output_gif}")
    print(f"output_mp4={output_mp4}")
    print(f"frames={target_count}")
    print(f"fps={args.fps}")


if __name__ == "__main__":
    main()
