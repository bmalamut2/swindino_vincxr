#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train VinDr-CXR detector.')
    parser.add_argument('config', help='Path to the config file.')
    parser.add_argument('--work-dir', help='Directory to save logs and checkpoints.')
    parser.add_argument(
        '--resume',
        nargs='?',
        const='auto',
        default=None,
        help='Resume from the latest checkpoint in work_dir or from a specific checkpoint path.',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override settings in the config. Example: train_dataloader.batch_size=2',
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher for distributed training.',
    )
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    return parser.parse_args()


def ensure_prepared_annotations(cfg: Config) -> None:
    required = [
        Path(cfg.train_dataloader.dataset.dataset.ann_file),
        Path(cfg.val_dataloader.ann_file if hasattr(cfg.val_dataloader, 'ann_file') else cfg.val_dataloader.dataset.ann_file),
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f'Missing prepared annotation file: {path}. '
                'Run `python scripts/prepare_vindr_cxr.py` first.')


def main() -> None:
    args = parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))

    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    ensure_prepared_annotations(cfg)

    if args.resume:
        cfg.resume = True
        if args.resume != 'auto':
            cfg.load_from = args.resume
    else:
        cfg.resume = False

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()

