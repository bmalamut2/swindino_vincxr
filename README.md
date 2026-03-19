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

Use Python 3.10 or 3.11 with a wheel-supported PyTorch/CUDA pair. Current Colab Python 3.12 + Torch 2.10 / CUDA 12.8 runtimes are too new for the prebuilt MMCV wheels used by MMDetection 3.x. A known-good combo for this project is `torch 2.1.0 + cu121 + mmcv 2.1.0`.

Preferred local setup, using the pinned Conda environment in [environment.yml](/home/bmalamut/Desktop/swindino_vincxr/environment.yml):

```bash
conda env create -f environment.yml
conda activate vindr-dino
```

This environment uses `conda-forge` for the base packages and installs the CUDA 12.1 PyTorch/MMCV wheel set through `pip`, so it assumes an NVIDIA driver new enough for CUDA 12.1.

If an existing environment drifts to `numpy 2.x`, repair it before training:

```bash
pip install --force-reinstall "numpy==1.26.4" "opencv-python==4.11.0.86"
```

```bash
pip install -U pip setuptools wheel packaging
pip install --force-reinstall "numpy==1.26.4" "opencv-python==4.11.0.86"
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
pip install "mmengine>=0.10.0,<1.0.0" pycocotools
pip install --only-binary=mmcv "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
pip install "mmdet>=3.2.0,<3.4.0"
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
python scripts/eval_vindr.py configs/vindr_dino_swinl_36e.py work_dirs/vindr_dino_swinl_36e/best_coco_bbox_mAP_epoch_XX.pth --split val
python scripts/eval_vindr.py configs/vindr_dino_swinl_36e.py work_dirs/vindr_dino_swinl_36e/best_coco_bbox_mAP_epoch_XX.pth --split test
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

## HPC Runtime Note

On some clusters, the job environment injects an older system `libstdc++.so.6` ahead of the active Conda environment. When that happens, imports can fail with errors like `GLIBCXX_3.4.29 not found` from Pillow or MMCV.

With the `vindr-dino` environment activated, run training or evaluation through the wrapper in [scripts/with_conda_libs.sh](/home/bmalamut/Desktop/swindino_vincxr/scripts/with_conda_libs.sh):

```bash
bash scripts/with_conda_libs.sh python scripts/train_vindr.py configs/vindr_dino_swinl_36e.py --work-dir work_dirs/vindr_dino_swinl_36e
```

If you prefer to run commands directly, prepend the Conda lib directory yourself:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## Google Colab

A Colab notebook is available at `colab_vindr_dino_swinl.ipynb`. It mounts Google Drive, creates an isolated `micromamba` Python 3.10 environment, installs a wheel-supported Torch/MMCV/MMDetection stack, then prepares annotations, visualizes samples, trains, resumes, and evaluates from notebook cells.
