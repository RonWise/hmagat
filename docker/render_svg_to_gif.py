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
    parser.add_argument(
        "--sample-mode",
        choices=["env-steps", "fps"],
        default="env-steps",
        help=(
            "How to sample the animated SVG: one frame per environment step "
            "or dense fps-based sampling."
        ),
    )
    parser.add_argument(
        "--step-time-scale",
        type=float,
        default=0.25,
        help=(
            "Environment-step to SVG-time scale used by POGEMA animation. "
            "Needed to reconstruct true step count from SVG duration."
        ),
    )
    parser.add_argument(
        "--include-success-fraction",
        action="store_true",
        help=(
            "Append current fraction of agents already standing on their own "
            "targets to the frame label."
        ),
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


async def render_frames(
    svg_path: Path,
    frames_dir: Path,
    sample_times: list[float],
    width: int,
    height: int,
    scale: float,
):
    prepare_js = """
    (width, height) => {
        const html = document.documentElement;
        const body = document.body;
        const svgEl = document.querySelector("svg");

        html.style.margin = "0";
        html.style.padding = "0";
        html.style.width = `${width}px`;
        html.style.height = `${height}px`;
        html.style.overflow = "hidden";
        html.style.background = "white";

        if (body) {
            body.style.margin = "0";
            body.style.padding = "0";
            body.style.width = `${width}px`;
            body.style.height = `${height}px`;
            body.style.overflow = "hidden";
            body.style.background = "white";
        }

        svgEl.style.display = "block";
        svgEl.style.width = `${width}px`;
        svgEl.style.height = `${height}px`;
        svgEl.style.maxWidth = `${width}px`;
        svgEl.style.maxHeight = `${height}px`;
        svgEl.style.margin = "0";
        svgEl.style.padding = "0";
        svgEl.style.overflow = "visible";
        return {
            svgWidth: svgEl.width.baseVal.value,
            svgHeight: svgEl.height.baseVal.value,
            clientWidth: svgEl.clientWidth,
            clientHeight: svgEl.clientHeight,
        };
    }
    """
    js = '(t) => { const svgEl = document.querySelector("svg"); svgEl.pauseAnimations(); svgEl.setCurrentTime(t); }'
    success_js = """
    () => {
        const agents = Array.from(document.querySelectorAll("circle.agent")).map((node) => ({
            cx: node.cx.animVal.value,
            cy: node.cy.animVal.value,
            visibility: getComputedStyle(node).visibility,
        }));
        const targets = Array.from(document.querySelectorAll("circle.target")).map((node) => ({
            cx: node.cx.animVal.value,
            cy: node.cy.animVal.value,
            visibility: getComputedStyle(node).visibility,
        }));

        const total = Math.min(agents.length, targets.length);
        const positionTolerance = 0.5;
        let eligible = 0;
        let matched = 0;
        for (let i = 0; i < total; i += 1) {
            if (agents[i].visibility === "hidden" || targets[i].visibility === "hidden") {
                continue;
            }
            eligible += 1;
            const dx = Math.abs(agents[i].cx - targets[i].cx);
            const dy = Math.abs(agents[i].cy - targets[i].cy);
            if (dx <= positionTolerance && dy <= positionTolerance) {
                matched += 1;
            }
        }
        return { matched, total: eligible };
    }
    """

    browser = await launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
    page = await browser.newPage()
    await page.setViewport(
        {"width": width, "height": height, "deviceScaleFactor": scale}
    )
    await page.goto(svg_path.resolve().as_uri(), {"waitUntil": "load"})
    await page.waitForSelector("svg")
    await page.evaluate(prepare_js, width, height)

    success_fractions = []
    for i, t in enumerate(sample_times):
        await page.evaluate(js, t)
        success_stats = await page.evaluate(success_js)
        total = success_stats["total"]
        success_fraction = 0.0 if total == 0 else success_stats["matched"] / total
        success_fractions.append(success_fraction)
        await page.screenshot({"path": str(frames_dir / f"frame_{i:04d}.png")})

    await browser.close()
    return sample_times, success_fractions


def infer_env_steps(duration: float, step_time_scale: float) -> int:
    if step_time_scale <= 0:
        raise ValueError("step_time_scale must be positive")
    # POGEMA animations append one duplicate terminal state and use:
    # dur = time_scale * (num_states - 1)
    # where num_states = env_steps + 2.
    return max(0, int(round(duration / step_time_scale)) - 1)


def build_sample_times(duration: float, fps: int, sample_mode: str, step_time_scale: float):
    if sample_mode == "fps":
        return [min(duration, i / fps) for i in range(max(1, int(duration * fps)))]

    env_steps = infer_env_steps(duration, step_time_scale)
    max_time = max(0.0, duration - 1e-6)
    step_time_epsilon = 1e-2
    return [
        0.0
        if step_idx == 0
        else min(max_time, step_idx * step_time_scale + step_time_epsilon)
        for step_idx in range(env_steps + 1)
    ]


def add_frame_labels(
    frames_dir: Path,
    sample_times: list[float],
    label_mode: str,
    label_prefix: str,
    success_fractions: list[float] | None = None,
):
    if label_mode == "none":
        return

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
        )
    except OSError:
        font = ImageFont.load_default()

    for i, time_s in enumerate(sample_times):
        frame_path = frames_dir / f"frame_{i:04d}.png"
        with Image.open(frame_path).convert("RGBA") as image:
            draw = ImageDraw.Draw(image)
            step_idx = i

            if label_mode == "step":
                label = f"{label_prefix} {step_idx}"
            elif label_mode == "time":
                label = f"t = {time_s:.1f}s"
            else:
                label = f"{label_prefix} {step_idx} | t = {time_s:.1f}s"

            if success_fractions is not None:
                label = f"{label} | {success_fractions[i]:.2f}"

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

    duration = extract_duration(svg_path.read_text())
    sample_times = build_sample_times(
        duration=duration,
        fps=args.fps,
        sample_mode=args.sample_mode,
        step_time_scale=args.step_time_scale,
    )
    rendered_sample_times, success_fractions = asyncio.run(
        render_frames(
            svg_path=svg_path,
            frames_dir=frames_dir,
            sample_times=sample_times,
            width=args.width,
            height=args.height,
            scale=args.scale,
        )
    )
    add_frame_labels(
        frames_dir=frames_dir,
        sample_times=rendered_sample_times,
        label_mode=args.label_mode,
        label_prefix=args.label_prefix,
        success_fractions=success_fractions if args.include_success_fraction else None,
    )
    build_gif(frames_dir=frames_dir, output_gif=output_gif, fps=args.fps)

    print(f"input={svg_path}")
    print(f"output={output_gif}")
    print(f"duration={duration}")
    print(f"fps={args.fps}")
    print(f"frames={len(rendered_sample_times)}")
    print(f"sample_mode={args.sample_mode}")
    print(f"env_steps={infer_env_steps(duration, args.step_time_scale)}")


if __name__ == "__main__":
    main()
