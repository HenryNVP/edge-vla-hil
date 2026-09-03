"""The dependency declarations must not drift apart.

The project declares its runtime dependencies twice: `[project].dependencies` in pyproject.toml,
and requirements-host.txt for the x86 Docker image. That is not an oversight — see the note atop
requirements-host.txt — but it means two install paths that must agree:

    pip install -r requirements-dev.txt   ->  -e .[dev]  ->  pyproject
    docker build -f docker/Dockerfile.host ->  requirements-host.txt

A silent divergence gives a developer a different environment than CI and than the image the
benchmarks run in, and the symptom shows up as a version-dependent behaviour difference rather
than as an install error. Same failure shape as config/default.yaml drifting from the node
parameter defaults, and guarded the same way.

requirements-jetson.txt is deliberately NOT checked: arm64 legitimately differs (torch comes from
the L4T base image, onnxruntime's pip wheel is CPU-only, robosuite/mujoco are host-only).
"""
from __future__ import annotations   # the Jetson image is py3.8: `set[str]` in a signature

import pathlib
import re

_REPO = pathlib.Path(__file__).resolve().parent.parent


def _pyproject_dependencies() -> set[str]:
    text = (_REPO / 'pyproject.toml').read_text()
    try:                                        # py3.11+; the container is 3.10
        import tomllib
        return set(tomllib.loads(text)['project']['dependencies'])
    except ModuleNotFoundError:
        pass
    block = re.search(r'^dependencies = \[(.*?)^\]', text, re.S | re.M)
    assert block, 'could not find [project].dependencies in pyproject.toml'
    return set(re.findall(r'"([^"]+)"', block.group(1)))


def _requirements(name: str) -> set[str]:
    out = set()
    for line in (_REPO / name).read_text().splitlines():
        line = line.split('#', 1)[0].strip()     # drop inline comments
        if line and not line.startswith('-'):    # '-e .' and friends are not pins
            out.add(line)
    return out


def test_host_requirements_mirror_pyproject_exactly():
    declared = _pyproject_dependencies()
    mirrored = _requirements('requirements-host.txt')

    assert mirrored == declared, (
        'requirements-host.txt and pyproject.toml [project].dependencies have drifted.\n'
        f'  only in requirements-host.txt: {sorted(mirrored - declared)}\n'
        f'  only in pyproject.toml:        {sorted(declared - mirrored)}')


def test_dev_requirements_defer_to_pyproject_rather_than_relisting():
    """requirements-dev.txt is the pattern the others cannot follow: it just installs the extra."""
    raw = (_REPO / 'requirements-dev.txt').read_text()

    assert '-e .[dev]' in raw or '-e .' in raw
    assert not _requirements('requirements-dev.txt'), (
        'requirements-dev.txt has started pinning packages directly; it should defer to '
        "pyproject's [project.optional-dependencies]")


def test_the_test_extra_is_where_test_deps_live():
    """pytest must not leak into the runtime dependency set — it would ship in the host image."""
    declared = _pyproject_dependencies()
    assert not [d for d in declared if d.split('>')[0].split('=')[0].strip() in
                ('pytest', 'ruff', 'mypy')], 'a dev tool is declared as a runtime dependency'


def test_jetson_requirements_stay_free_of_the_host_only_packages():
    """The Orin Nano image must not pull torch (L4T wheels), robosuite or mujoco (the plant runs
    on the x86 side), or onnxruntime (the pip arm64 wheel is CPU-only)."""
    jetson = {r.split('>')[0].split('=')[0].split('<')[0].strip()
              for r in _requirements('requirements-jetson.txt')}

    for forbidden in ('torch', 'robosuite', 'mujoco', 'onnxruntime', 'lerobot'):
        assert forbidden not in jetson, (
            f'{forbidden} must not be pip-installed on the Jetson — see the note in '
            'requirements-jetson.txt')
