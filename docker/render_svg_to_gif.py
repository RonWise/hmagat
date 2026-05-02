import argparse
import asyncio
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pyppeteer import launch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input animated SVG path")
    parser.add_argument("--output", required=True, help="Output GIF path")
    parser.add_argument(
        "--frames-dir",
        default="/workspace/outputs/gif_frames",
        help="Temporary directory for PNG frames",
    )
    parser.add_argument("--fps", type=int, default=6, help="GIF frame rate")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--label-mode",
        choices=["none", "step", "time", "both"],
        default="both",
        help="Overlay label on each rendered frame",
    )
    parser.add_argument(
        "--label-prefix",
        default="Step",
        help="Prefix for step counter label",
    )
    return parser.parse_args()


def extract_duration(svg_text: str) -> float:
    marker = 'dur="'
    start = svg_text.find(marker)
    if start == -1:
        return 10.0
    start += len(marker)
    end = svg_text.find('s"', start)
    if end == -1:
        return 10.0
    return float(svg_text[start:end])


async def render_frames(svg_path: Path, frames_dir: Path, fps: int, width: int, height: int, scale: float):
    text = svg_path.read_text()
    duration = extract_duration(text)
    num_frames = max(1, int(duration * fps))
    js = '(t) => { const svgEl = document.querySelector("svg"); svgEl.pauseAnimations(); svgEl.setCurrentTime(t); }'

    browser = await launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
    page = await browser.newPage()
    await page.setViewport(
        {"width": width, "height": height, "deviceScaleFactor": scale}
    )
    await page.goto(svg_path.resolve().as_uri(), {"waitUntil": "load"})
    await page.waitForSelector("svg")

    for i in range(num_frames):
        t = min(duration, i / fps)
        await page.evaluate(js, t)
        await page.screenshot({"path": str(frames_dir / f"frame_{i:04d}.png")})

    await browser.close()
    return duration, num_frames


def add_frame_labels(
    frames_dir: Path,
    fps: int,
    duration: float,
    num_frames: int,
    label_mode: str,
    label_prefix: str,
):
    if label_mode == "none":
        return

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
        )
    except OSError:
        font = ImageFont.load_default()

    for i in range(num_frames):
        frame_path = frames_dir / f"frame_{i:04d}.png"
        with Image.open(frame_path).convert("RGBA") as image:
            draw = ImageDraw.Draw(image)
            step_idx = i
            time_s = min(duration, i / fps)

            if label_mode == "step":
                label = f"{label_prefix} {step_idx}"
            elif label_mode == "time":
                label = f"t = {time_s:.1f}s"
            else:
                label = f"{label_prefix} {step_idx} | t = {time_s:.1f}s"

            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = max(12, (image.width - text_w) // 2)
            y = max(10, image.height - text_h - 22)

            # Soft white banner behind the label to keep it readable on any map.
            draw.rounded_rectangle(
                (x - 12, y - 8, x + text_w + 12, y + text_h + 8),
                radius=10,
                fill=(255, 255, 255, 220),
            )
            draw.text((x, y), label, font=font, fill=(20, 20, 20, 255))
            image.save(frame_path)


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


def main():
    args = parse_args()
    svg_path = Path(args.input)
    output_gif = Path(args.output)
    frames_dir = Path(args.frames_dir)

    if not svg_path.exists():
        raise FileNotFoundError(f"Input SVG not found: {svg_path}")

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_gif.parent.mkdir(parents=True, exist_ok=True)

    duration, num_frames = asyncio.run(
        render_frames(
            svg_path=svg_path,
            frames_dir=frames_dir,
            fps=args.fps,
            width=args.width,
            height=args.height,
            scale=args.scale,
        )
    )
    add_frame_labels(
        frames_dir=frames_dir,
        fps=args.fps,
        duration=duration,
        num_frames=num_frames,
        label_mode=args.label_mode,
        label_prefix=args.label_prefix,
    )
    build_gif(frames_dir=frames_dir, output_gif=output_gif, fps=args.fps)

    print(f"input={svg_path}")
    print(f"output={output_gif}")
    print(f"duration={duration}")
    print(f"fps={args.fps}")
    print(f"frames={num_frames}")


if __name__ == "__main__":
    main()
