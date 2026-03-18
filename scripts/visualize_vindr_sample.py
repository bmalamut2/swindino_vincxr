#!/usr/bin/env python
from __future__ import annotations

import argparse
import colorsys
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - dependency error path
    raise SystemExit(
        'Pillow is required for visualization. Install it with `pip install pillow`.'
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render VinDr-CXR COCO annotations onto JPEG images.')
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test'],
        default='val',
        help='Prepared annotation split to visualize.',
    )
    parser.add_argument(
        '--ann-file',
        type=Path,
        help='Override the COCO annotation file path.',
    )
    parser.add_argument(
        '--image-root',
        type=Path,
        help='Override the image directory. Defaults to vincxr/train or vincxr/test.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('artifacts/visualizations'),
        help='Directory to write rendered images into.',
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=9,
        help='Number of images to render when --image-id is not set.',
    )
    parser.add_argument(
        '--image-id',
        action='append',
        default=[],
        help='Specific image stem(s) to render, for example 000434271f63a053c4128a0ba6352c7f.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed used for sampling images.',
    )
    parser.add_argument(
        '--include-empty',
        action='store_true',
        help='Allow empty images into the random sample pool.',
    )
    return parser.parse_args()


def default_ann_file(split: str) -> Path:
    return PROJECT_ROOT / 'artifacts' / 'vindr_cxr' / 'annotations' / f'{split}.json'


def default_image_root(split: str) -> Path:
    folder = 'test' if split == 'test' else 'train'
    return PROJECT_ROOT / 'vincxr' / folder


def class_color(category_id: int) -> tuple[int, int, int]:
    hue = ((category_id * 0.173) % 1.0)
    saturation = 0.85
    value = 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


def load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype('DejaVuSans.ttf', 18)
    except OSError:
        return ImageFont.load_default()


def text_bbox(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
              font: ImageFont.ImageFont) -> tuple[int, int, int, int]:
    if hasattr(draw, 'textbbox'):
        return draw.textbbox(xy, text, font=font)
    width, height = draw.textsize(text, font=font)
    x, y = xy
    return int(x), int(y), int(x + width), int(y + height)


def choose_images(
    images: list[dict],
    annotations_by_image: dict[int, list[dict]],
    requested_image_ids: list[str],
    sample_count: int,
    include_empty: bool,
    seed: int,
) -> list[dict]:
    if requested_image_ids:
        key_to_image = {
            Path(image['file_name']).stem: image for image in images
        }
        selected = []
        missing = []
        for image_id in requested_image_ids:
            image = key_to_image.get(image_id)
            if image is None:
                missing.append(image_id)
            else:
                selected.append(image)
        if missing:
            raise SystemExit(f'Unknown image id(s): {", ".join(missing)}')
        return selected

    candidates = []
    for image in images:
        has_boxes = bool(annotations_by_image.get(image['id']))
        if has_boxes or include_empty:
            candidates.append(image)
    if not candidates:
        raise SystemExit('No images matched the sampling criteria.')

    rng = random.Random(seed)
    if sample_count >= len(candidates):
        return sorted(candidates, key=lambda item: item['file_name'])
    return sorted(rng.sample(candidates, sample_count), key=lambda item: item['file_name'])


def draw_annotations(image: Image.Image, image_info: dict, annotations: list[dict],
                     categories_by_id: dict[int, dict]) -> Image.Image:
    rendered = image.convert('RGB')
    draw = ImageDraw.Draw(rendered)
    font = load_font()

    line_width = max(2, round(max(rendered.size) / 512))
    text_padding = 3

    for ann in annotations:
        x, y, w, h = ann['bbox']
        x2 = x + w
        y2 = y + h
        color = class_color(ann['category_id'])
        draw.rectangle((x, y, x2, y2), outline=color, width=line_width)

        category = categories_by_id[ann['category_id']]['name']
        support = ann.get('support')
        label = f'{category} (n={support})' if support and support > 1 else category

        label_bbox = text_bbox(draw, (x, y), label, font=font)
        tx1 = x
        ty1 = max(0, y - (label_bbox[3] - label_bbox[1]) - 2 * text_padding)
        tx2 = tx1 + (label_bbox[2] - label_bbox[0]) + 2 * text_padding
        ty2 = ty1 + (label_bbox[3] - label_bbox[1]) + 2 * text_padding
        draw.rectangle((tx1, ty1, tx2, ty2), fill=color)
        draw.text((tx1 + text_padding, ty1 + text_padding), label, fill='black', font=font)

    header = f"{Path(image_info['file_name']).stem} | boxes={len(annotations)}"
    header_bbox = text_bbox(draw, (8, 8), header, font=font)
    draw.rectangle(
        (4, 4, header_bbox[2] + 12, header_bbox[3] + 12),
        fill=(0, 0, 0),
    )
    draw.text((8, 8), header, fill='white', font=font)
    return rendered


def main() -> None:
    args = parse_args()

    ann_file = args.ann_file or default_ann_file(args.split)
    image_root = args.image_root or default_image_root(args.split)
    output_dir = args.output_dir / args.split

    if not ann_file.exists():
        raise SystemExit(
            f'Annotation file not found: {ann_file}. '
            'Run `python scripts/prepare_vindr_cxr.py` first.')
    if not image_root.exists():
        raise SystemExit(f'Image root not found: {image_root}')

    payload = json.loads(ann_file.read_text(encoding='utf-8'))
    images = payload['images']
    categories_by_id = {category['id']: category for category in payload['categories']}

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in payload['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    selected_images = choose_images(
        images=images,
        annotations_by_image=annotations_by_image,
        requested_image_ids=args.image_id,
        sample_count=args.sample,
        include_empty=args.include_empty,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for image_info in selected_images:
        image_path = image_root / image_info['file_name']
        if not image_path.exists():
            raise SystemExit(f'Image file referenced in annotations is missing: {image_path}')
        image = Image.open(image_path)
        rendered = draw_annotations(
            image=image,
            image_info=image_info,
            annotations=annotations_by_image.get(image_info['id'], []),
            categories_by_id=categories_by_id,
        )
        output_path = output_dir / f"{Path(image_info['file_name']).stem}_annotated.png"
        rendered.save(output_path)
        print(output_path)


if __name__ == '__main__':
    main()
