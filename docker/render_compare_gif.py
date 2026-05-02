import argparse
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

    combined = []
    for start, duration in zip(boundaries[:-1], segments):
        li = frame_index_at(start, left_boundaries)
        ri = frame_index_at(start, right_boundaries)
        lf = left_frames[li]
        rf = right_frames[ri]

        if args.max_width is not None:
            half_width = max(64, args.max_width // 2)
            lf = resize_if_needed(lf, half_width)
            rf = resize_if_needed(rf, half_width)

        canvas = Image.new('RGBA', (lf.width + rf.width, max(lf.height, rf.height)), (255, 255, 255, 255))
        canvas.paste(lf, (0, 0), lf)
        canvas.paste(rf, (lf.width, 0), rf)

        draw = ImageDraw.Draw(canvas)
        divider_x = lf.width
        draw.line((divider_x, 0, divider_x, canvas.height), fill=(80, 80, 80, 255), width=3)

        for label, x0, x1 in ((args.left_label, 0, lf.width), (args.right_label, lf.width, lf.width + rf.width)):
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            tx = x0 + (x1 - x0 - text_w) // 2
            ty = 10
            draw.rounded_rectangle((tx - 10, ty - 6, tx + text_w + 10, ty + text_h + 6), radius=10, fill=(255, 255, 255, 220))
            draw.text((tx, ty), label, font=font, fill=(20, 20, 20, 255))

        combined.append((canvas.convert('P', palette=Image.ADAPTIVE), duration))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first, first_duration = combined[0]
    rest = [frame for frame, _ in combined[1:]]
    durations = [first_duration] + [duration for _, duration in combined[1:]]
    first.save(output_path, save_all=True, append_images=rest, duration=durations, loop=0, disposal=2)

    print(output_path)
    print(f'left_total_ms={left_total}')
    print(f'right_total_ms={right_total}')
    print(f'combined_frames={len(combined)}')
    print(f'combined_total_ms={sum(durations)}')


if __name__ == '__main__':
    main()
