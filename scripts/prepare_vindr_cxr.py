#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vindr_detector.constants import LOCAL_CLASSES


SOF_MARKERS = set(list(range(0xC0, 0xC4)) + list(range(0xC5, 0xC8)) + list(range(0xC9, 0xCC)) +
                  list(range(0xCD, 0xD0)))
CLASS_TO_ID = {name: idx + 1 for idx, name in enumerate(LOCAL_CLASSES)}


@dataclass(frozen=True)
class BoxRecord:
    image_id: str
    class_name: str
    bbox: tuple[float, float, float, float]
    support: int = 1


def jpeg_size(path: Path) -> tuple[int, int]:
    with path.open('rb') as handle:
        if handle.read(2) != b'\xff\xd8':
            raise ValueError(f'Not a JPEG file: {path}')
        while True:
            marker_start = handle.read(1)
            if not marker_start:
                raise ValueError(f'No SOF marker found in {path}')
            if marker_start != b'\xff':
                continue
            while marker_start == b'\xff':
                marker_start = handle.read(1)
            marker = marker_start[0]
            if marker in (0xD8, 0xD9):
                continue
            seglen = int.from_bytes(handle.read(2), 'big')
            if marker in SOF_MARKERS:
                data = handle.read(5)
                height = int.from_bytes(data[1:3], 'big')
                width = int.from_bytes(data[3:5], 'big')
                return width, height
            handle.seek(seglen - 2, 1)


def pairwise_iou(box_a: tuple[float, float, float, float],
                 box_b: tuple[float, float, float, float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    denom = max(area_a + area_b - inter, 1e-12)
    return inter / denom


def pairwise_ioa(box_a: tuple[float, float, float, float],
                 box_b: tuple[float, float, float, float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    return inter / max(area_a, 1e-12)


def connected_components(
    boxes: list[tuple[float, float, float, float]],
    link_fn,
) -> list[list[int]]:
    neighbors: list[list[int]] = [[] for _ in boxes]
    for left in range(len(boxes)):
        for right in range(left + 1, len(boxes)):
            if link_fn(boxes[left], boxes[right]):
                neighbors[left].append(right)
                neighbors[right].append(left)

    visited = [False] * len(boxes)
    components: list[list[int]] = []
    for start in range(len(boxes)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for nxt in neighbors[node]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        components.append(component)
    return components


def clip_bbox(bbox: tuple[float, float, float, float], width: int,
              height: int) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = bbox
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def load_split_ids(path: Path) -> set[str]:
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        return {row['image_id'] for row in reader}


def ensure_image_sizes(image_ids: Iterable[str], image_dir: Path,
                       image_sizes: dict[str, tuple[int, int]]) -> None:
    for image_id in image_ids:
        if image_id not in image_sizes:
            image_sizes[image_id] = jpeg_size(image_dir / f'{image_id}.jpeg')


def load_train_boxes(annotations_path: Path, image_dir: Path) -> tuple[dict[str, list[BoxRecord]], dict]:
    image_sizes: dict[str, tuple[int, int]] = {}
    grouped: dict[str, list[BoxRecord]] = defaultdict(list)
    raw_box_count = 0
    dropped_boxes = 0

    with annotations_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            class_name = row['class_name']
            if class_name not in CLASS_TO_ID:
                continue

            image_id = row['image_id']
            if image_id not in image_sizes:
                image_sizes[image_id] = jpeg_size(image_dir / f'{image_id}.jpeg')
            width, height = image_sizes[image_id]
            bbox = tuple(map(float, (row['x_min'], row['y_min'], row['x_max'], row['y_max'])))
            clipped = clip_bbox(bbox, width, height)
            raw_box_count += 1
            if clipped is None:
                dropped_boxes += 1
                continue
            grouped[image_id].append(BoxRecord(image_id=image_id, class_name=class_name, bbox=clipped))

    return grouped, {
        'raw_box_count': raw_box_count,
        'dropped_invalid_boxes': dropped_boxes,
        'image_sizes': image_sizes,
    }


def merge_reader_boxes(grouped_boxes: dict[str, list[BoxRecord]], iou_thr: float) -> tuple[dict[str, list[BoxRecord]], dict]:
    merged_by_image: dict[str, list[BoxRecord]] = {}
    merged_box_count = 0
    support_hist = Counter()
    class_image_count: dict[str, set[str]] = defaultdict(set)

    for image_id, boxes in grouped_boxes.items():
        per_class: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
        for record in boxes:
            per_class[record.class_name].append(record.bbox)

        merged_records: list[BoxRecord] = []
        for class_name, class_boxes in per_class.items():
            components = connected_components(
                class_boxes,
                lambda left, right: pairwise_iou(left, right) >= iou_thr,
            )
            for component in components:
                support_hist[len(component)] += 1
                x1 = median(class_boxes[idx][0] for idx in component)
                y1 = median(class_boxes[idx][1] for idx in component)
                x2 = median(class_boxes[idx][2] for idx in component)
                y2 = median(class_boxes[idx][3] for idx in component)
                merged_records.append(
                    BoxRecord(
                        image_id=image_id,
                        class_name=class_name,
                        bbox=(x1, y1, x2, y2),
                        support=len(component),
                    ))
                class_image_count[class_name].add(image_id)
                merged_box_count += 1
        merged_by_image[image_id] = merged_records

    return merged_by_image, {
        'merged_box_count': merged_box_count,
        'merge_ratio': merged_box_count / max(
            sum(len(v) for v in grouped_boxes.values()), 1),
        'cluster_support_hist': dict(sorted(support_hist.items())),
        'class_image_count': {
            class_name: len(class_image_count[class_name])
            for class_name in LOCAL_CLASSES
        },
    }


def dedup_same_class_boxes(
    grouped_boxes: dict[str, list[BoxRecord]],
    dedup_iou_thr: float,
    dedup_ioa_thr: float,
) -> tuple[dict[str, list[BoxRecord]], dict]:
    deduped_by_image: dict[str, list[BoxRecord]] = {}
    before_count = 0
    after_count = 0
    component_hist = Counter()
    merged_components = 0

    def is_duplicate(left: tuple[float, float, float, float],
                     right: tuple[float, float, float, float]) -> bool:
        iou = pairwise_iou(left, right)
        ioa = max(pairwise_ioa(left, right), pairwise_ioa(right, left))
        return iou >= dedup_iou_thr or ioa >= dedup_ioa_thr

    for image_id, records in grouped_boxes.items():
        per_class: dict[str, list[BoxRecord]] = defaultdict(list)
        for record in records:
            per_class[record.class_name].append(record)

        deduped_records: list[BoxRecord] = []
        for class_name, class_records in per_class.items():
            class_boxes = [record.bbox for record in class_records]
            before_count += len(class_records)
            components = connected_components(class_boxes, is_duplicate)
            after_count += len(components)

            for component in components:
                component_hist[len(component)] += 1
                if len(component) > 1:
                    merged_components += 1
                x1 = median(class_boxes[idx][0] for idx in component)
                y1 = median(class_boxes[idx][1] for idx in component)
                x2 = median(class_boxes[idx][2] for idx in component)
                y2 = median(class_boxes[idx][3] for idx in component)
                deduped_records.append(
                    BoxRecord(
                        image_id=image_id,
                        class_name=class_name,
                        bbox=(x1, y1, x2, y2),
                        support=sum(class_records[idx].support for idx in component),
                    ))
        deduped_by_image[image_id] = deduped_records

    return deduped_by_image, {
        'dedup_input_box_count': before_count,
        'deduped_box_count': after_count,
        'dedup_ratio': after_count / max(before_count, 1),
        'dedup_component_hist': dict(sorted(component_hist.items())),
        'dedup_merged_components': merged_components,
    }


def load_test_boxes(annotations_path: Path, image_dir: Path) -> tuple[dict[str, list[BoxRecord]], dict[str, tuple[int, int]]]:
    image_sizes: dict[str, tuple[int, int]] = {}
    grouped: dict[str, list[BoxRecord]] = defaultdict(list)

    with annotations_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            class_name = row['class_name']
            image_id = row['image_id']
            if image_id not in image_sizes:
                image_sizes[image_id] = jpeg_size(image_dir / f'{image_id}.jpeg')
            if class_name not in CLASS_TO_ID:
                continue
            width, height = image_sizes[image_id]
            bbox = tuple(map(float, (row['x_min'], row['y_min'], row['x_max'], row['y_max'])))
            clipped = clip_bbox(bbox, width, height)
            if clipped is None:
                continue
            grouped[image_id].append(BoxRecord(image_id=image_id, class_name=class_name, bbox=clipped))

    return grouped, image_sizes


def build_coco_dataset(
    image_ids: list[str],
    image_dir: Path,
    image_sizes: dict[str, tuple[int, int]],
    annotations_by_image: dict[str, list[BoxRecord]],
) -> tuple[dict, dict]:
    images = []
    annotations = []
    ann_id = 1
    empty_images = 0

    for image_index, image_id in enumerate(image_ids, start=1):
        width, height = image_sizes[image_id]
        images.append(
            dict(
                id=image_index,
                file_name=f'{image_id}.jpeg',
                width=width,
                height=height,
            ))
        records = annotations_by_image.get(image_id, [])
        if not records:
            empty_images += 1
            continue
        for record in records:
            x1, y1, x2, y2 = record.bbox
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            annotations.append(
                dict(
                    id=ann_id,
                    image_id=image_index,
                    category_id=CLASS_TO_ID[record.class_name],
                    bbox=[x1, y1, bbox_w, bbox_h],
                    area=bbox_w * bbox_h,
                    iscrowd=0,
                    support=record.support,
                ))
            ann_id += 1

    coco = dict(
        images=images,
        annotations=annotations,
        categories=[
            dict(id=CLASS_TO_ID[name], name=name, supercategory='lesion')
            for name in LOCAL_CLASSES
        ],
    )
    return coco, {
        'n_images': len(images),
        'n_annotations': len(annotations),
        'n_empty_images': empty_images,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare VinDr-CXR COCO annotations.')
    parser.add_argument('--repo-root', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/vindr_cxr'))
    parser.add_argument('--merge-iou', type=float, default=0.5)
    parser.add_argument('--dedup-iou', type=float, default=0.6)
    parser.add_argument('--dedup-ioa', type=float, default=0.85)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    data_root = repo_root / 'vincxr'
    annotations_root = data_root / 'annotations'
    output_root = (repo_root / args.output_dir).resolve()
    output_ann_root = output_root / 'annotations'

    train_image_dir = data_root / 'train'
    test_image_dir = data_root / 'test'
    train_split_ids = load_split_ids(annotations_root / 'vindr_cxr_train_90pct.csv')
    val_split_ids = load_split_ids(annotations_root / 'vindr_cxr_val_10pct.csv')
    test_ids = sorted({path.stem for path in test_image_dir.glob('*.jpeg')})

    raw_grouped, raw_stats = load_train_boxes(annotations_root / 'annotations_train.csv', train_image_dir)
    merged_grouped, merge_stats = merge_reader_boxes(raw_grouped, args.merge_iou)
    deduped_grouped, dedup_stats = dedup_same_class_boxes(
        merged_grouped,
        dedup_iou_thr=args.dedup_iou,
        dedup_ioa_thr=args.dedup_ioa,
    )

    image_sizes = raw_stats['image_sizes']
    train_ids = sorted(train_split_ids)
    val_ids = sorted(val_split_ids)
    ensure_image_sizes(train_ids, train_image_dir, image_sizes)
    ensure_image_sizes(val_ids, train_image_dir, image_sizes)
    if train_split_ids & val_split_ids:
        raise ValueError('Train and validation image splits overlap.')

    test_grouped, test_sizes = load_test_boxes(annotations_root / 'annotations_test.csv', test_image_dir)

    train_coco, train_stats = build_coco_dataset(train_ids, train_image_dir, image_sizes, deduped_grouped)
    val_coco, val_stats = build_coco_dataset(val_ids, train_image_dir, image_sizes, deduped_grouped)
    test_coco, test_stats = build_coco_dataset(test_ids, test_image_dir, test_sizes, test_grouped)

    write_json(output_ann_root / 'train.json', train_coco)
    write_json(output_ann_root / 'val.json', val_coco)
    write_json(output_ann_root / 'test.json', test_coco)

    stats = {
        'data': {
            'n_local_classes': len(LOCAL_CLASSES),
            'n_images_train': len(train_ids),
            'n_images_val': len(val_ids),
            'n_images_test': len(test_ids),
            'n_empty_images_train': train_stats['n_empty_images'],
            'n_empty_images_val': val_stats['n_empty_images'],
            'n_empty_images_test': test_stats['n_empty_images'],
        },
        'labels': {
            'raw_box_count': raw_stats['raw_box_count'],
            'merged_box_count': merge_stats['merged_box_count'],
            'merge_ratio': merge_stats['merge_ratio'],
            'cluster_support_hist': merge_stats['cluster_support_hist'],
            'dedup_input_box_count': dedup_stats['dedup_input_box_count'],
            'deduped_box_count': dedup_stats['deduped_box_count'],
            'dedup_ratio': dedup_stats['dedup_ratio'],
            'dedup_component_hist': dedup_stats['dedup_component_hist'],
            'dedup_merged_components': dedup_stats['dedup_merged_components'],
            'class_image_count': merge_stats['class_image_count'],
            'dropped_invalid_boxes': raw_stats['dropped_invalid_boxes'],
        },
        'exports': {
            'train': train_stats,
            'val': val_stats,
            'test': test_stats,
        },
        'paths': {
            'train_json': str(output_ann_root / 'train.json'),
            'val_json': str(output_ann_root / 'val.json'),
            'test_json': str(output_ann_root / 'test.json'),
        },
    }
    write_json(output_root / 'stats.json', stats)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
