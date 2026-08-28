"""Wiring tests for the evh_bringup launch graphs — the only package with no logic of its own.

A launch file fails at *run* time, on a real robot run, and a mis-wired one often fails silently:
a typo'd launch argument resolves to an empty string, a renamed console_script is a startup error
you only see in a 3-minute sweep cell's log. These tests build each LaunchDescription and assert
the graph properties the rest of the system's invariants rest on:

  * `absolute` reaches the plant AND the reactive layer from ONE argument (invariant #1: this is
    precisely why the plant's cross-check does not also police the reactive layer),
  * exactly the three observation topics are relayed, and `/obs/ee_pose` never is (invariant #6),
  * every relay output is what the controller actually subscribes to,
  * every LaunchConfiguration referenced is declared, and
  * every (package, executable) launched exists as a console_script.

Needs the `launch` / `launch_ros` packages, hence the ros2 mark. Nothing is executed.
"""
import importlib.util
import os
import re

import pytest
from conftest import requires_ros2

_LAUNCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'ros2_ws', 'src', 'evh_bringup', 'launch')
_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ros2_ws', 'src')

LAUNCH_FILES = ['hil.launch.py', 'host.launch.py', 'controller.launch.py']
GRAPH_LAUNCHES = ['hil.launch.py', 'host.launch.py']     # the two that run plant + reactive
PACKAGES = ['evh_plant', 'evh_controller', 'evh_reactive', 'evh_latency', 'evh_bringup']

OBS_RELAYED = {'/obs/image', '/obs/image_wrist', '/obs/proprio'}


# ------------------------------------------------------------------------ loading
def _load(filename):
    """Import a `*.launch.py` file. The dot in the name makes it un-importable normally."""
    path = os.path.join(_LAUNCH_DIR, filename)
    spec = importlib.util.spec_from_file_location(filename.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


def _context(ld):
    """A LaunchContext with every declared argument set to its default."""
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument

    ctx = LaunchContext()
    for entity in ld.entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(ctx)
    return ctx


def _perform(ctx, value):
    """Resolve a substitution list to a plain string.

    Two quirks of launch_ros: some fields (package, executable) hold a bare str rather than a
    substitution list, and literal parameter values are round-tripped through yaml.dump, which
    tacks a `\n...\n` document-end marker onto scalars.
    """
    from launch.utilities import perform_substitutions

    text = value if isinstance(value, str) else perform_substitutions(ctx, list(value))
    return text.removesuffix('\n...\n').strip()


def _nodes(ld):
    from launch_ros.actions import Node
    return [e for e in ld.entities if isinstance(e, Node)]


def _declared(ld):
    from launch.actions import DeclareLaunchArgument
    return [e for e in ld.entities if isinstance(e, DeclareLaunchArgument)]


def _params(ctx, node):
    """{param name: resolved string value} for a node's single inline parameter dict."""
    out = {}
    for block in node._Node__parameters or ():
        for key, value in block.items():
            out[_perform(ctx, key)] = _perform(ctx, value)
    return out


def _param_refs(ctx, node):
    """{param name: launch-argument name it reads}, skipping literal parameters."""
    from launch.substitutions import LaunchConfiguration

    refs = {}
    for block in node._Node__parameters or ():
        for key, value in block.items():
            subs = [value] if not isinstance(value, (list, tuple)) else list(value)
            for sub in subs:
                if isinstance(sub, LaunchConfiguration):
                    refs[_perform(ctx, key)] = _perform(ctx, sub.variable_name)
    return refs


def _remaps(ctx, node):
    return {_perform(ctx, src): _perform(ctx, dst) for src, dst in (node._Node__remappings or ())}


def _relays(ctx, ld):
    return [n for n in _nodes(ld) if _perform(ctx, n._Node__package) == 'evh_latency']


def _only(ctx, ld, package):
    matches = [n for n in _nodes(ld) if _perform(ctx, n._Node__package) == package]
    assert len(matches) == 1, f'expected exactly one {package} node, got {len(matches)}'
    return matches[0]


# ------------------------------------------------------------------------- basics
@requires_ros2
@pytest.mark.parametrize('filename', LAUNCH_FILES)
def test_launch_description_builds_and_declares_each_argument_once(filename):
    ld = _load(filename)
    names = [a._DeclareLaunchArgument__name for a in _declared(ld)]

    assert _nodes(ld), 'launch file starts no nodes'
    assert len(names) == len(set(names)), f'duplicate launch arguments: {names}'


@requires_ros2
@pytest.mark.parametrize('filename', LAUNCH_FILES)
def test_every_launch_configuration_used_is_declared(filename):
    """A LaunchConfiguration for an undeclared argument does not raise — it resolves to nothing,
    and the node silently runs on its own parameter default."""
    ld = _load(filename)
    ctx = _context(ld)
    declared = {a._DeclareLaunchArgument__name for a in _declared(ld)}

    for node in _nodes(ld):
        for param, arg in _param_refs(ctx, node).items():
            assert arg in declared, (
                f'{filename}: parameter {param!r} reads undeclared launch arg {arg!r}')


# ------------------------------------------------------- invariant 1: one absolute arg
@requires_ros2
@pytest.mark.parametrize('filename', GRAPH_LAUNCHES)
def test_absolute_reaches_plant_and_reactive_from_a_single_argument(filename):
    """The plant cross-checks its own mode against the policy's and aborts on a mismatch, but it
    does NOT check the reactive layer. That is only safe because both read the SAME launch arg —
    if this ever splits, the reactive layer needs its own check (see CLAUDE.md invariant 1)."""
    ld = _load(filename)
    ctx = _context(ld)

    plant_arg = _param_refs(ctx, _only(ctx, ld, 'evh_plant'))['absolute_actions']
    reactive_arg = _param_refs(ctx, _only(ctx, ld, 'evh_reactive'))['absolute_waypoints']

    assert plant_arg == reactive_arg == 'absolute', (
        f'{filename}: plant reads {plant_arg!r}, reactive reads {reactive_arg!r} — '
        'the reactive layer can now diverge from the plant with nothing to catch it')


@requires_ros2
@pytest.mark.parametrize('filename', GRAPH_LAUNCHES)
def test_absolute_and_the_mode_guard_default_on(filename):
    """Defaults match the abs-action DP Lift checkpoint, with the mismatch abort armed."""
    ld = _load(filename)
    defaults = {a._DeclareLaunchArgument__name: a.default_value for a in _declared(ld)}
    ctx = _context(ld)

    assert _perform(ctx, defaults['absolute']) == 'true'
    assert _perform(ctx, defaults['strict_mode_check']) == 'true'
    assert _params(ctx, _only(ctx, ld, 'evh_plant'))['strict_mode_check'] == 'true'


# --------------------------------------------------- invariant 6: what gets delayed
@requires_ros2
@pytest.mark.parametrize('filename', GRAPH_LAUNCHES)
def test_only_the_three_observation_topics_are_relayed(filename):
    """/obs/ee_pose is the reactive layer's ZERO-DELAY local anchor. Routing it through a relay
    would delay the anchor too and quietly destroy the thing the experiment measures."""
    ld = _load(filename)
    ctx = _context(ld)
    relayed = {_params(ctx, r)['input_topic'] for r in _relays(ctx, ld)}

    assert relayed == OBS_RELAYED, f'{filename}: relayed topics changed: {sorted(relayed)}'
    assert '/obs/ee_pose' not in relayed
    assert '/cmd/action' not in relayed, 'the reactive->plant link is local, not networked'


@requires_ros2
@pytest.mark.parametrize('filename', GRAPH_LAUNCHES)
def test_all_relays_share_the_network_condition_arguments(filename):
    """One degradation knob per condition: a relay wired to its own arg (or to none) would leave
    part of the observation path undegraded and flatten the sweep curve."""
    ld = _load(filename)
    ctx = _context(ld)

    for relay in _relays(ctx, ld):
        refs = _param_refs(ctx, relay)
        assert refs.get('latency_ms') == 'latency_ms'
        assert refs.get('jitter_ms') == 'jitter_ms'
        assert refs.get('drop_prob') == 'drop_prob'


@requires_ros2
def test_the_controller_subscribes_to_exactly_the_relay_outputs():
    """The controller is deliberately unaware of the relay; the remap is the whole mechanism.
    A relay publishing where nobody listens is a fully silent no-op degradation."""
    ld = _load('hil.launch.py')
    ctx = _context(ld)

    outputs = {_params(ctx, r)['input_topic']: _params(ctx, r)['output_topic']
               for r in _relays(ctx, ld)}
    remaps = _remaps(ctx, _only(ctx, ld, 'evh_controller'))

    assert remaps == outputs, (
        f'controller remaps {remaps} but the relays publish {outputs}')


@requires_ros2
@pytest.mark.parametrize('filename', ['hil.launch.py', 'controller.launch.py'])
def test_the_controller_reads_delayed_topics_not_raw_ones(filename):
    ld = _load(filename)
    ctx = _context(ld)
    remaps = _remaps(ctx, _only(ctx, ld, 'evh_controller'))

    assert set(remaps) == OBS_RELAYED
    assert all(dst == f'{src}/delayed' for src, dst in remaps.items())


# ------------------------------------------------------------------- entry points
def _console_scripts(package):
    """{executable: 'module:function'} parsed out of a package's setup.py."""
    with open(os.path.join(_SRC, package, 'setup.py')) as fh:
        setup_py = fh.read()
    block = re.search(r"'console_scripts'\s*:\s*\[(.*?)\]", setup_py, re.S)
    assert block, f'{package}/setup.py declares no console_scripts'
    return dict(re.findall(r"'\s*([\w-]+)\s*=\s*([\w.]+:\w+)\s*'", block.group(1)))


@requires_ros2
@pytest.mark.parametrize('filename', LAUNCH_FILES)
def test_every_launched_executable_is_a_registered_console_script(filename):
    """colcon happily builds a package whose setup.py no longer exports the executable a launch
    file names; the failure surfaces as a node that just never comes up."""
    ld = _load(filename)
    ctx = _context(ld)

    for node in _nodes(ld):
        package = _perform(ctx, node._Node__package)
        executable = _perform(ctx, node._Node__node_executable)
        scripts = _console_scripts(package)
        assert executable in scripts, (
            f'{filename} launches {package}/{executable}, not in {sorted(scripts)}')


@pytest.mark.parametrize('package', PACKAGES)
def test_each_packages_entry_points_resolve_to_a_real_main(package):
    """Every console_script points at a module that exists and defines the named function."""
    for executable, target in _console_scripts(package).items():
        module_path, func = target.split(':')
        source = os.path.join(_SRC, package, *module_path.split('.')) + '.py'
        assert os.path.exists(source), f'{package}/{executable} -> missing {source}'
        with open(source) as fh:
            assert re.search(rf'^def {func}\(', fh.read(), re.M), (
                f'{source} defines no {func}()')
