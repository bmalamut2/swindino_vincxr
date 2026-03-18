from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from mmengine.evaluator import BaseMetric

from mmdet.registry import METRICS


def _to_numpy(value) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pairwise_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1])
    union = np.maximum(box_area + boxes_area - inter, 1e-12)
    return inter / union


def _infer_num_classes(results: Iterable[dict]) -> int:
    max_label = -1
    for result in results:
        if result['gt_labels'].size:
            max_label = max(max_label, int(result['gt_labels'].max()))
        if result['pred_labels'].size:
            max_label = max(max_label, int(result['pred_labels'].max()))
    return max_label + 1


@METRICS.register_module()
class VinDRMetric(BaseMetric):
    """Custom VinDr-CXR metrics for mAP@0.4 and FROC AUC 0-8 FP/img."""

    default_prefix = 'vindr'

    def __init__(
        self,
        iou_thr: float = 0.4,
        froc_max_fp_per_img: float = 8.0,
        collect_device: str = 'cpu',
        prefix: str | None = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not 0.0 < iou_thr <= 1.0:
            raise ValueError(f'iou_thr must be in (0, 1], got {iou_thr}')
        if froc_max_fp_per_img <= 0:
            raise ValueError('froc_max_fp_per_img must be > 0')
        self.iou_thr = iou_thr
        self.froc_max_fp_per_img = froc_max_fp_per_img

    def process(self, data_batch: dict, data_samples: list) -> None:
        gt_samples = None
        if isinstance(data_batch, dict):
            gt_samples = data_batch.get('data_samples')

        for idx, pred_sample in enumerate(data_samples):
            gt_sample = pred_sample
            if not hasattr(gt_sample, 'gt_instances') and gt_samples is not None:
                gt_sample = gt_samples[idx]

            metainfo = {}
            if hasattr(pred_sample, 'metainfo'):
                metainfo = pred_sample.metainfo
            elif hasattr(pred_sample, '_metainfo'):
                metainfo = pred_sample._metainfo

            image_id = metainfo.get('img_id', metainfo.get('img_path', idx))
            gt_instances = getattr(gt_sample, 'gt_instances', None)
            pred_instances = getattr(pred_sample, 'pred_instances', None)

            gt_boxes = _to_numpy(getattr(gt_instances, 'bboxes', None)).reshape(-1, 4)
            gt_labels = _to_numpy(getattr(gt_instances, 'labels', None)).astype(np.int64)
            pred_boxes = _to_numpy(getattr(pred_instances, 'bboxes', None)).reshape(-1, 4)
            pred_labels = _to_numpy(getattr(pred_instances, 'labels', None)).astype(np.int64)
            pred_scores = _to_numpy(getattr(pred_instances, 'scores', None)).astype(np.float32)

            self.results.append(
                dict(
                    image_id=image_id,
                    gt_boxes=gt_boxes.astype(np.float32),
                    gt_labels=gt_labels,
                    pred_boxes=pred_boxes.astype(np.float32),
                    pred_labels=pred_labels,
                    pred_scores=pred_scores,
                ))

    def compute_metrics(self, results: list[dict]) -> dict[str, float]:
        if not results:
            return {'mAP_0.4': 0.0, 'froc_auc_0_8': 0.0}

        if self.dataset_meta and 'classes' in self.dataset_meta:
            num_classes = len(self.dataset_meta['classes'])
        else:
            num_classes = _infer_num_classes(results)

        ap_values: list[float] = []
        for class_id in range(num_classes):
            ap = self._compute_class_ap(results, class_id)
            if ap is not None:
                ap_values.append(ap)

        return {
            'mAP_0.4': float(np.mean(ap_values)) if ap_values else 0.0,
            'froc_auc_0_8': self._compute_froc_auc(results),
        }

    def _compute_class_ap(self, results: list[dict], class_id: int) -> float | None:
        gt_by_image: dict[object, np.ndarray] = {}
        predictions: list[tuple[float, object, np.ndarray]] = []
        total_gt = 0

        for result in results:
            gt_mask = result['gt_labels'] == class_id
            gt_boxes = result['gt_boxes'][gt_mask]
            if gt_boxes.size:
                gt_by_image[result['image_id']] = gt_boxes
                total_gt += gt_boxes.shape[0]

            pred_mask = result['pred_labels'] == class_id
            pred_boxes = result['pred_boxes'][pred_mask]
            pred_scores = result['pred_scores'][pred_mask]
            for score, box in zip(pred_scores.tolist(), pred_boxes):
                predictions.append((float(score), result['image_id'], box))

        if total_gt == 0:
            return None

        predictions.sort(key=lambda item: item[0], reverse=True)
        matched = {
            image_id: np.zeros(len(boxes), dtype=bool)
            for image_id, boxes in gt_by_image.items()
        }

        tp = np.zeros(len(predictions), dtype=np.float32)
        fp = np.zeros(len(predictions), dtype=np.float32)
        for idx, (_, image_id, box) in enumerate(predictions):
            gt_boxes = gt_by_image.get(image_id)
            if gt_boxes is None or not len(gt_boxes):
                fp[idx] = 1.0
                continue

            ious = _pairwise_iou(box, gt_boxes)
            if matched[image_id].any():
                ious[matched[image_id]] = -1.0

            best_gt = int(ious.argmax()) if ious.size else -1
            if best_gt >= 0 and ious[best_gt] >= self.iou_thr:
                matched[image_id][best_gt] = True
                tp[idx] = 1.0
            else:
                fp[idx] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / max(float(total_gt), 1e-12)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

        recall_points = np.linspace(0.0, 1.0, 101)
        ap = 0.0
        for recall_target in recall_points:
            mask = recalls >= recall_target
            precision = float(precisions[mask].max()) if mask.any() else 0.0
            ap += precision
        return ap / 101.0

    def _compute_froc_auc(self, results: list[dict]) -> float:
        num_images = len(results)
        total_gt = int(sum(result['gt_boxes'].shape[0] for result in results))
        if num_images == 0 or total_gt == 0:
            return 0.0

        gt_by_key: dict[tuple[object, int], np.ndarray] = defaultdict(
            lambda: np.zeros((0, 4), dtype=np.float32))
        matched: dict[tuple[object, int], np.ndarray] = {}
        predictions: list[tuple[float, object, int, np.ndarray]] = []

        for result in results:
            image_id = result['image_id']
            for class_id in np.unique(result['gt_labels']):
                class_mask = result['gt_labels'] == class_id
                boxes = result['gt_boxes'][class_mask]
                key = (image_id, int(class_id))
                gt_by_key[key] = boxes
                matched[key] = np.zeros(boxes.shape[0], dtype=bool)

            for score, label, box in zip(
                    result['pred_scores'].tolist(),
                    result['pred_labels'].tolist(),
                    result['pred_boxes']):
                predictions.append((float(score), image_id, int(label), box))

        predictions.sort(key=lambda item: item[0], reverse=True)

        tp_count = 0
        fp_count = 0
        sensitivity_by_fp: dict[float, float] = {0.0: 0.0}

        for _, image_id, class_id, box in predictions:
            key = (image_id, class_id)
            gt_boxes = gt_by_key.get(key)
            if gt_boxes is None or not len(gt_boxes):
                fp_count += 1
            else:
                ious = _pairwise_iou(box, gt_boxes)
                used = matched[key]
                if used.any():
                    ious[used] = -1.0
                best_gt = int(ious.argmax()) if ious.size else -1
                if best_gt >= 0 and ious[best_gt] >= self.iou_thr:
                    matched[key][best_gt] = True
                    tp_count += 1
                else:
                    fp_count += 1

            fp_per_img = min(fp_count / float(num_images), self.froc_max_fp_per_img)
            sensitivity = tp_count / float(total_gt)
            current = sensitivity_by_fp.get(fp_per_img, 0.0)
            sensitivity_by_fp[fp_per_img] = max(current, sensitivity)

        xs = sorted(sensitivity_by_fp)
        ys = []
        best = 0.0
        for x in xs:
            best = max(best, sensitivity_by_fp[x])
            ys.append(best)

        auc = 0.0
        for idx, x in enumerate(xs):
            if x >= self.froc_max_fp_per_img:
                break
            next_x = xs[idx + 1] if idx + 1 < len(xs) else self.froc_max_fp_per_img
            next_x = min(next_x, self.froc_max_fp_per_img)
            auc += max(next_x - x, 0.0) * ys[idx]

        return float(auc / self.froc_max_fp_per_img)

