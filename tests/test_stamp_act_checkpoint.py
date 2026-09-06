"""Tests for the ACT checkpoint action-convention stamp (invariant 1's carrier).

The stamp is the only thing standing between an absolute-action ACT checkpoint and a run that
looks fine while commanding deltas as world coordinates, so the failure modes that matter here
are the quiet ones: a missing dataset, a dataset that predates the mode file, a checkpoint with
no training config. Each must refuse rather than pick a side.
"""
import json

import pytest

from stamp_act_checkpoint import ACTION_MODE_FILE, dataset_root_from_checkpoint, mode_from_dataset


def _dataset(tmp_path, absolute, actions='abs-derived'):
    meta = tmp_path / 'meta'
    meta.mkdir(parents=True)
    (meta / ACTION_MODE_FILE).write_text(json.dumps(
        {'absolute_actions': absolute, 'actions': actions}))
    return tmp_path


def _checkpoint(tmp_path, dataset_root):
    ckpt = tmp_path / 'pretrained_model'
    ckpt.mkdir(parents=True)
    if dataset_root is not None:
        (ckpt / 'train_config.json').write_text(json.dumps(
            {'dataset': {'repo_id': 'local/x', 'root': str(dataset_root)}}))
    return ckpt


def test_dataset_root_comes_from_the_training_config(tmp_path):
    root = tmp_path / 'ds'
    ckpt = _checkpoint(tmp_path, root)
    assert dataset_root_from_checkpoint(ckpt) == root


def test_dataset_root_is_none_without_a_training_config(tmp_path):
    assert dataset_root_from_checkpoint(_checkpoint(tmp_path, None)) is None


def test_dataset_root_is_none_when_the_config_has_no_root(tmp_path):
    ckpt = tmp_path / 'pretrained_model'
    ckpt.mkdir()
    (ckpt / 'train_config.json').write_text(json.dumps({'dataset': {'repo_id': 'local/x'}}))
    assert dataset_root_from_checkpoint(ckpt) is None


@pytest.mark.parametrize('absolute', [True, False])
def test_mode_is_read_back_from_the_dataset(tmp_path, absolute):
    assert mode_from_dataset(_dataset(tmp_path, absolute)) is absolute


def test_mode_is_none_for_a_dataset_predating_the_stamp(tmp_path):
    (tmp_path / 'meta').mkdir()
    assert mode_from_dataset(tmp_path) is None
