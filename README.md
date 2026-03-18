# VinDr-CXR DINO Training

This repository now contains a local MMDetection project for VinDr-CXR lesion detection using the DINO 5-scale Swin-L recipe from `training_recipe.md`.

## What It Does

- Merges duplicate reader boxes in the train split with IoU `>= 0.5`
- Keeps empty images in train/val/test
- Exports COCO-format annotations under `artifacts/vindr_cxr/annotations/`
- Trains DINO Swin-L with resize-only augmentation and class-balanced oversampling
- Logs standard COCO bbox metrics plus custom `mAP_0.4` and `froc_auc_0_8`
- Supports checkpoint resume from the latest work dir checkpoint or an explicit path

## Install

Install a CUDA-matched PyTorch first, then install the MMDetection stack:

```bash
pip install -U openmim
mim install "mmengine>=0.10.0,<1.0.0" "mmcv>=2.0.0,<2.2.0"
pip install "mmdet>=3.2.0,<3.4.0" pycocotools
```

## Prepare Data

Run from the repository root:

```bash
python scripts/prepare_vindr_cxr.py
```

This writes:

- `artifacts/vindr_cxr/annotations/train.json`
- `artifacts/vindr_cxr/annotations/val.json`
- `artifacts/vindr_cxr/annotations/test.json`
- `artifacts/vindr_cxr/stats.json`

## Train

Single GPU:

```bash
python scripts/train_vindr.py configs/vindr_dino_swinl_36e.py --work-dir work_dirs/vindr_dino_swinl_36e
```

Multi-GPU:

```bash
torchrun --nproc_per_node=4 scripts/train_vindr.py configs/vindr_dino_swinl_36e.py --launcher pytorch --work-dir work_dirs/vindr_dino_swinl_36e
```

Resume:

```bash
python scripts/train_vindr.py configs/vindr_dino_swinl_36e.py --work-dir work_dirs/vindr_dino_swinl_36e --resume
python scripts/train_vindr.py configs/vindr_dino_swinl_36e.py --work-dir work_dirs/vindr_dino_swinl_36e --resume work_dirs/vindr_dino_swinl_36e/epoch_12.pth
```

## Evaluate

```bash
python scripts/eval_vindr.py configs/vindr_dino_swinl_36e.py work_dirs/vindr_dino_swinl_36e/best_coco_bbox_mAP.pth --split val
python scripts/eval_vindr.py configs/vindr_dino_swinl_36e.py work_dirs/vindr_dino_swinl_36e/best_coco_bbox_mAP.pth --split test
```

Custom metrics are reported as:

- `vindr/mAP_0.4`
- `vindr/froc_auc_0_8`

The default initialization checkpoint is `checkpoints/checkpoint0027_5scale_swin.pth`.

## Visualize Annotations

Render a few random prepared annotations:

```bash
python scripts/visualize_vindr_sample.py --split val --sample 9
```

Render a specific image:

```bash
python scripts/visualize_vindr_sample.py --split train --image-id 000434271f63a053c4128a0ba6352c7f
```

Outputs are written under `artifacts/visualizations/<split>/`.
