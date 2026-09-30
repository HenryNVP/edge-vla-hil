#!/usr/bin/env python3
"""Provenance for every results file: which code, config and environment produced it.

Each runner (the ROS benchmark, the step harness, the Kinetix and LIBERO runners) calls
`record(out_path)` once at start-up. It appends one entry to `<out>.manifest.json`, a JSON list,
because results files are appended to across invocations and each invocation can run different
code. An entry holds:

  * the repo's git commit, branch and whether the tree was dirty (a dirty tree means the commit
    alone does not reproduce the run; the diff summary says what differed);
  * the pinned commits of the external repos the run used (external/rtc-kinetix, ...);
  * the command line, working directory, host, time;
  * versions of the packages that decide the numbers (numpy, torch, jax, robosuite, mujoco,
    lerobot, ...), read from package metadata, never by importing them;
  * the GPU, when nvidia-smi exists.

Standard library only, and git is optional: inside the ROS container there may be no git binary,
so the commit is read from .git directly and "dirty" becomes unknown rather than a guess.
`scripts/check_conditions.py` reads these back.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from importlib import metadata

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = ('numpy', 'torch', 'jax', 'jaxlib', 'flax', 'robosuite', 'mujoco', 'lerobot',
            'hf-libero', 'kinetix', 'diffusers', 'onnxruntime')
EXTERNAL = ('external/rtc-kinetix', 'external/rtc-kinetix/third_party/kinetix')


def _git(repo: str, *args: str, strip: bool = True) -> str | None:
    if shutil.which('git') is None:
        return None
    try:
        out = subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True,
                             timeout=10, check=True).stdout
        return out.strip() if strip else out
    except (subprocess.SubprocessError, OSError):
        return None


def head_commit(repo: str) -> str | None:
    """The checked-out commit, without needing a git binary."""
    via_git = _git(repo, 'rev-parse', 'HEAD')
    if via_git:
        return via_git
    gitdir = os.path.join(repo, '.git')
    if os.path.isfile(gitdir):                      # a submodule: "gitdir: <path>"
        with open(gitdir) as f:
            gitdir = os.path.normpath(os.path.join(repo, f.read().split(':', 1)[1].strip()))
    try:
        with open(os.path.join(gitdir, 'HEAD')) as f:
            head = f.read().strip()
    except OSError:
        return None
    if not head.startswith('ref:'):
        return head
    ref = head.split(' ', 1)[1]
    try:
        with open(os.path.join(gitdir, ref)) as f:
            return f.read().strip()
    except OSError:
        pass
    try:
        with open(os.path.join(gitdir, 'packed-refs')) as f:
            for line in f:
                if line.rstrip().endswith(' ' + ref):
                    return line.split(' ', 1)[0]
    except OSError:
        pass
    return None


def repo_state(repo: str) -> dict:
    # unstripped: porcelain lines start with a status column that may be a space
    porcelain = _git(repo, 'status', '--porcelain', '--untracked-files=no', strip=False)
    return {
        'commit': head_commit(repo),
        'branch': _git(repo, 'rev-parse', '--abbrev-ref', 'HEAD'),
        # None = could not tell (no git binary); a list = the tracked files that differed
        'dirty': None if porcelain is None else [ln[3:] for ln in porcelain.splitlines() if ln],
    }


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def gpu() -> str | None:
    if shutil.which('nvidia-smi') is None:
        return None
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version',
                               '--format=csv,noheader'], capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def manifest_path(out_path: str) -> str:
    root, _ = os.path.splitext(out_path)
    return f'{root}.manifest.json'


def entry(extra: dict | None = None) -> dict:
    return {
        'time': _dt.datetime.now().astimezone().isoformat(timespec='seconds'),
        'argv': list(sys.argv),
        'cwd': os.getcwd(),
        'host': socket.gethostname(),
        'python': platform.python_version(),
        'repo': repo_state(REPO),
        'external': {p: head_commit(os.path.join(REPO, p)) for p in EXTERNAL
                     if os.path.exists(os.path.join(REPO, p))},
        'packages': package_versions(),
        'gpu': gpu(),
        **(extra or {}),
    }


def record(out_path: str, extra: dict | None = None) -> dict:
    """Append this invocation's entry to `<out>.manifest.json`; returns the entry."""
    e = entry(extra)
    path = manifest_path(out_path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    runs = []
    if os.path.exists(path):
        with open(path) as f:
            runs = json.load(f)
    runs.append(e)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(runs, f, indent=1)
    os.replace(tmp, path)
    return e
