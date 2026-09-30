"""Tests for run provenance — pure Python, no git binary required."""
import json

import run_manifest


def test_the_commit_is_read_without_a_git_binary(tmp_path, monkeypatch):
    """Inside the ROS container there may be no git; the commit must still be recorded."""
    git = tmp_path / '.git'
    (git / 'refs' / 'heads').mkdir(parents=True)
    (git / 'HEAD').write_text('ref: refs/heads/paper1\n')
    (git / 'refs' / 'heads' / 'paper1').write_text('abc123\n')
    monkeypatch.setattr(run_manifest.shutil, 'which', lambda _name: None)
    assert run_manifest.head_commit(str(tmp_path)) == 'abc123'
    assert run_manifest.repo_state(str(tmp_path))['dirty'] is None   # unknown, not "clean"


def test_a_packed_ref_is_found(tmp_path, monkeypatch):
    git = tmp_path / '.git'
    git.mkdir()
    (git / 'HEAD').write_text('ref: refs/heads/main\n')
    (git / 'packed-refs').write_text('# pack-refs\ndef456 refs/heads/main\n')
    monkeypatch.setattr(run_manifest.shutil, 'which', lambda _name: None)
    assert run_manifest.head_commit(str(tmp_path)) == 'def456'


def test_each_invocation_appends_one_entry_next_to_its_results(tmp_path):
    out = tmp_path / 'sweep.csv'
    run_manifest.record(str(out), {'config': {'delays': '1,2'}})
    run_manifest.record(str(out), {'config': {'delays': '3'}})
    runs = json.loads((tmp_path / 'sweep.manifest.json').read_text())
    assert [r['config']['delays'] for r in runs] == ['1,2', '3']
    assert {'time', 'argv', 'repo', 'packages', 'python'} <= set(runs[0])
