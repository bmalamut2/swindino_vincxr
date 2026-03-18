from __future__ import annotations

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class EnsureThreeChannelGray(BaseTransform):
    """Force grayscale JPEGs into 3 identical channels."""

    def __init__(self, force_grayscale: bool = True) -> None:
        self.force_grayscale = force_grayscale

    def transform(self, results: dict) -> dict:
        img = results['img']

        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
            diff_max = 0
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
            diff_max = 0
        elif img.ndim == 3 and img.shape[2] == 3:
            c0 = img[..., 0].astype(np.int32)
            c1 = img[..., 1].astype(np.int32)
            c2 = img[..., 2].astype(np.int32)
            diff_max = int(
                max(
                    np.abs(c0 - c1).max(initial=0),
                    np.abs(c0 - c2).max(initial=0),
                    np.abs(c1 - c2).max(initial=0),
                ))
            if self.force_grayscale and diff_max:
                img = np.repeat(img[..., :1], 3, axis=2)
                diff_max = 0
        else:
            raise ValueError(f'Unsupported image shape for grayscale repeat: {img.shape}')

        results['img'] = img
        height, width = img.shape[:2]
        results['img_shape'] = (height, width)
        if 'ori_shape' in results:
            results['ori_shape'] = (height, width)
        results['vindr_channel_abs_diff_max'] = diff_max
        return results

