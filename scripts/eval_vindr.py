#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vindr_detector.compat import ensure_pkg_resources

ensure_pkg_resources()

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate VinDr-CXR detector.')
    parser.add_argument('config', help='Path to the config file.')
    parser.add_argument('checkpoint', help='Checkpoint to evaluate.')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--work-dir', help='Optional work directory override.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override settings in the config.',
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher for distributed evaluation.',
    )
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))

    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    runner = Runner.from_cfg(cfg)
    if args.split == 'val':
        runner.val()
    else:
        runner.test()


if __name__ == '__main__':
    main()
