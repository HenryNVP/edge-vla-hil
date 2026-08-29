"""Wiring tests for the evh_bringup launch graphs — the only package with no logic of its own.

A launch file fails at *run* time, on a real robot run, and a mis-wired one often fails silently:
a typo'd launch argument resolves to an empty string, a renamed console_script is a startup error
you only see in a 3-minute sweep cell's log. These tests build each LaunchDescription and assert
the graph properties the rest of the system's invariants rest on:

  * `absolute` reaches the plant AND the reactive layer from ONE argument (invariant #1: this is
    precisely why the plant's cross-check does not also police the reactive layer),
  * exactly the three observation topics are relayed, and `/obs/ee_pose` never is (invariant #6),
  * every relay output is what the controller actually subscribes to,
  * every LaunchConfiguration referenced is declared,
  * every (package, executable) launched exists as a console_script, and
  * config/default.yaml still mirrors the node parameter defaults it claims to document.

Needs the `launch` / `launch_ros` packages, hence the ros2 mark. Nothing is executed.
"""
import ast
import importlib.util
import os
import pathlib
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


def _unwrap(value):
    """Strip a ParameterValue wrapper down to the substitution list inside it."""
    from launch_ros.parameter_descriptions import ParameterValue

    return value.value if isinstance(value, ParameterValue) else value


def _launch_configs(value):
    """The LaunchConfiguration objects a parameter value reads, seeing through typed()."""
    from launch.substitutions import LaunchConfiguration

    from evh_bringup.launch_utils import _TypedArgument

    inner = _unwrap(value)
    subs = [inner] if not isinstance(inner, (list, tuple)) else list(inner)
    for sub in subs:
        if isinstance(sub, _TypedArgument):
            yield sub.source
        elif isinstance(sub, LaunchConfiguration):
            yield sub


def _perform(ctx, value):
    """Resolve a substitution list to a plain string.

    Three quirks of launch_ros: some fields (package, executable) hold a bare str rather than a
    substitution list, typed parameters are wrapped in a ParameterValue, and literal parameter
    values are round-tripped through yaml.dump, which tacks a `\n...\n` marker onto scalars.
    """
    from launch.utilities import perform_substitutions

    value = _unwrap(value)
    text = value if isinstance(value, str) else perform_substitutions(ctx, list(value))
    return text.removesuffix('\n...\n').strip()


def _value_type(value):
    """The type a parameter declares at the launch boundary, or None if it declares none."""
    from launch_ros.parameter_descriptions import ParameterValue

    return value.value_type if isinstance(value, ParameterValue) else None


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

    refs = {}
    for block in node._Node__parameters or ():
        for key, value in block.items():
            for config in _launch_configs(value):
                refs[_perform(ctx, key)] = _perform(ctx, config.variable_name)
    return refs


def _typed_params(node):
    """{param name: declared value_type or None} for every parameter reading a launch arg."""

    out = {}
    for block in node._Node__parameters or ():
        for key, value in block.items():
            if any(True for _ in _launch_configs(value)):
                name = ''.join(s.text for s in key if hasattr(s, 'text')) or str(key)
                out[name] = _value_type(value)
    return out


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


# --------------------------------------------------------- launch-argument types
def _declared_param_types(package: str, executable: str) -> dict:
    """{parameter name: Python type} parsed from the node's declare_parameter() defaults."""
    module_path, _func = _console_scripts(package)[executable].split(':')
    source = os.path.join(_SRC, package, *module_path.split('.')) + '.py'
    tree = ast.parse(pathlib.Path(source).read_text())

    types = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'declare_parameter' and len(node.args) >= 2):
            try:
                name = ast.literal_eval(node.args[0])
                types[name] = type(ast.literal_eval(node.args[1]))
            except ValueError:
                continue     # a non-literal default; nothing to check against
    return types


@requires_ros2
@pytest.mark.parametrize('filename', LAUNCH_FILES)
def test_non_string_parameters_declare_their_type_at_the_launch_boundary(filename):
    """A launch argument is a string, and launch_ros infers its parameter type by YAML-parsing it.
    So `latency_ms:=40` becomes an INTEGER against a node that declared a DOUBLE, rclpy raises,
    and that node alone dies at startup — the rest of the graph runs and reports a full, undegraded
    result under a label saying otherwise. Wrapping in ParameterValue(..., value_type=) is what
    makes `40`, `40.0` and `0` all arrive as a float. See evh_bringup/launch_utils.typed."""
    ld = _load(filename)
    ctx = _context(ld)

    for node in _nodes(ld):
        package = _perform(ctx, node._Node__package)
        executable = _perform(ctx, node._Node__node_executable)
        declared = _declared_param_types(package, executable)

        for param, launch_type in _typed_params(node).items():
            expected = declared.get(param)
            if expected in (None, str):
                continue        # strings need no coercion; YAML leaves them alone
            assert launch_type is expected, (
                f'{filename}: {package}/{param} is declared {expected.__name__} by the node but '
                f'the launch file passes it as {launch_type}; wrap it with '
                f"typed('<arg>', {expected.__name__})")


@requires_ros2
@pytest.mark.parametrize('raw', ['40', '40.0', '0'])
def test_an_integer_latency_argument_still_reaches_the_relay_as_a_float(raw):
    """The exact input that used to kill all three relays."""
    from launch import LaunchContext
    from launch.substitutions import TextSubstitution
    from launch_ros.parameter_descriptions import ParameterValue

    value = ParameterValue(TextSubstitution(text=raw), value_type=float).evaluate(LaunchContext())
    assert isinstance(value, float)
    assert value == float(raw)


# ----------------------------------------------------------------- default.yaml
def _declared_param_defaults(package: str, module: str) -> dict:
    """{parameter name: default value} parsed from a node's declare_parameter() calls."""
    source = os.path.join(_SRC, package, package, f'{module}.py')
    tree = ast.parse(pathlib.Path(source).read_text())

    defaults = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'declare_parameter' and len(node.args) >= 2):
            try:
                defaults[ast.literal_eval(node.args[0])] = ast.literal_eval(node.args[1])
            except ValueError:
                continue
    return defaults


CONFIGURED_NODES = [
    ('/evh_plant', 'evh_plant', 'plant_node'),
    ('/evh_controller', 'evh_controller', 'controller_node'),
    ('/evh_reactive', 'evh_reactive', 'reactive_node'),
]


@pytest.mark.parametrize('key,package,module', CONFIGURED_NODES)
def test_default_config_still_mirrors_the_node_defaults(key, package, module):
    """config/default.yaml says "values mirror the node parameter defaults" and is what someone
    copies to build an experiment config. Nothing loads it at launch, so drift is invisible: a
    parameter added to a node just never appears, and a config copied from it silently omits the
    knob. `strict_mode_check` had already gone missing this way."""
    import yaml

    config = yaml.safe_load(
        pathlib.Path(os.path.join(_SRC, 'evh_bringup/config/default.yaml')).read_text())
    listed = config.get(key, {}).get('ros__parameters', {})
    declared = _declared_param_defaults(package, module)

    assert not set(declared) - set(listed), (
        f'default.yaml is missing {sorted(set(declared) - set(listed))} under {key}')
    assert not set(listed) - set(declared), (
        f'default.yaml documents {sorted(set(listed) - set(declared))} which {module} '
        'no longer declares')
    for name, value in listed.items():
        assert value == declared[name] and type(value) is type(declared[name]), (
            f'{key}/{name}: default.yaml says {value!r}, {module}.py declares {declared[name]!r}')
