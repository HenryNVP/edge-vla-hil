#!/usr/bin/env python3
"""Stamp a trained ACT checkpoint with the action convention it was trained on (invariant 1).

An ACT checkpoint is 7-dim whether it learned delta actions or absolute EE targets — the mode is
not recoverable from the weights, and guessing it wrong is the silent garbage-motion failure the
plant's `/policy/absolute` cross-check exists to catch. So the mode is carried:

    robomimic_to_lerobot.py  ->  <dataset>/meta/evh_action_mode.json
    this script              ->  <checkpoint>/evh_policy.json
    ACTBackend               ->  reads it, announces it on /policy/absolute
    plant                    ->  aborts if it disagrees with its own `absolute` arg

The dataset is found through LeRobot's own `train_config.json`, which every checkpoint carries,
so the stamp is derived from what was actually trained on rather than typed in by hand. That
path is recorded as the *training* process saw it, which for a run inside the host container is
`/ws/...` and does not exist on the host — hence `--dataset-root`. `--absolute` overrides
outright, for a checkpoint whose dataset is gone (copied to the Jetson, say).

    python scripts/stamp_act_checkpoint.py \
        outputs/train/act_lift_ph_abs/checkpoints/last/pretrained_model \
        --dataset-root data/lerobot/lift_ph_abs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / 'ros2_ws' / 'src' / 'evh_controller'))

from evh_controller.policy import POLICY_SIDECAR  # noqa: E402

ACTION_MODE_FILE = 'evh_action_mode.json'


def dataset_root_from_checkpoint(ckpt: Path) -> Path | None:
    """LeRobot writes the whole training config next to the weights; the dataset root is in it."""
    config = ckpt / 'train_config.json'
    if not config.is_file():
        return None
    dataset = json.loads(config.read_text()).get('dataset') or {}
    root = dataset.get('root')
    return Path(root) if root else None


def mode_from_dataset(root: Path) -> bool | None:
    path = root / 'meta' / ACTION_MODE_FILE
    if not path.is_file():
        return None
    return bool(json.loads(path.read_text())['absolute_actions'])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('checkpoint', help='.../checkpoints/<step>/pretrained_model')
    p.add_argument('--dataset-root', default='',
                   help='where the training dataset lives now (train_config.json records the '
                        'path training saw, e.g. /ws/... from inside the container)')
    p.add_argument('--absolute', choices=('true', 'false'), default='',
                   help='override instead of reading the training dataset')
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        raise SystemExit(f'{ckpt} is not a directory')

    if args.absolute:
        absolute, source = args.absolute == 'true', '--absolute'
    else:
        root = Path(args.dataset_root) if args.dataset_root else dataset_root_from_checkpoint(ckpt)
        if root is None:
            raise SystemExit(f'{ckpt}/train_config.json has no dataset root — pass --absolute')
        if not root.is_dir():
            raise SystemExit(
                f'{root} does not exist here — train_config.json records the path the training '
                'process saw (/ws/... inside the container). Pass --dataset-root, or run this '
                'in the same environment that trained.')
        absolute = mode_from_dataset(root)
        if absolute is None:
            raise SystemExit(
                f'{root}/meta/{ACTION_MODE_FILE} is missing — the dataset predates the stamp '
                '(re-run scripts/robomimic_to_lerobot.py) or pass --absolute')
        source = str(root)

    out = ckpt / POLICY_SIDECAR
    out.write_text(json.dumps({'absolute_actions': absolute}, indent=2) + '\n')
    print(f'{out}: absolute_actions={absolute}  (from {source})')


if __name__ == '__main__':
    main()
