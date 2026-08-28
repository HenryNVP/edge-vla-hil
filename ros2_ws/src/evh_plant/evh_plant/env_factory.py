"""robosuite environment construction — everything version-fragile about the simulator, in one place.

Split out of `plant_node.py` so the node holds only the HiL logic (episodes, actions, the mode
cross-check) and this holds the parts that break when robosuite moves: the 1.4-vs-1.5 controller
config API, where `control_delta` lives in each config shape, and the `seed` kwarg that only
1.5 accepts. No ROS imports, so it is unit-testable in the fast suite.

`control_delta` is invariant 1's mechanism: absolute mode means OSC takes `action[:3]` as a
world-frame position and `action[3:6]` as a world-frame axis-angle, rather than as deltas.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvSpec:
    """Everything `build_env` needs, so the node's parameter block is the only place they're read."""
    env_name: str = 'Lift'
    robot: str = 'Panda'
    cameras: tuple[str, ...] = ('agentview', 'robot0_eye_in_hand')
    image_size: int = 84            # DP checkpoints are trained at 84x84
    action_hz: float = 200.0
    max_episode_s: float = 20.0     # horizon; hitting it = timeout = recorded failure
    seed: int = 0
    absolute_actions: bool = True

    @property
    def horizon(self) -> int:
        """Episode horizon in control steps — what robosuite counts down to `done`."""
        return int(self.max_episode_s * self.action_hz)


@dataclass
class BuiltEnv:
    """A constructed env plus what the node needs to know about it."""
    env: object
    obs: dict
    action_dim: int
    notes: list[str] = field(default_factory=list)   # things the node should log


def make_controller_config():
    """OSC_POSE-style controller config across robosuite 1.4 / 1.5 APIs."""
    try:
        from robosuite.controllers import load_controller_config
        return load_controller_config(default_controller='OSC_POSE')
    except Exception:
        pass
    try:
        from robosuite.controllers import load_composite_controller_config
        return load_composite_controller_config(controller='BASIC')
    except Exception:
        return None


def set_control_delta(config: dict, value: bool) -> None:
    """Set OSC control_delta across robosuite config shapes (1.4 flat / 1.5 composite)."""
    if 'control_delta' in config:
        config['control_delta'] = value
        return
    for part_cfg in config.get('body_parts', {}).values():
        if isinstance(part_cfg, dict) and part_cfg.get('type', '').startswith('OSC'):
            part_cfg['control_delta'] = value


def build_env(spec: EnvSpec) -> BuiltEnv:
    """Construct the robosuite env, already reset. Raises if robosuite is unavailable."""
    import robosuite as suite

    np.random.seed(spec.seed)
    notes: list[str] = []

    kwargs = {
        'env_name': spec.env_name,
        'robots': spec.robot,          # robosuite >=1.5 uses `robots`
        'has_renderer': False,
        'has_offscreen_renderer': True,
        'use_camera_obs': True,
        'camera_names': list(spec.cameras),
        'camera_heights': [spec.image_size] * len(spec.cameras),
        'camera_widths': [spec.image_size] * len(spec.cameras),
        'control_freq': spec.action_hz,
        'horizon': spec.horizon,
        'reward_shaping': False,
        'seed': spec.seed,
    }

    controller = make_controller_config()
    if controller is not None:
        if spec.absolute_actions:
            set_control_delta(controller, False)
        kwargs['controller_configs'] = controller
    elif spec.absolute_actions:
        raise RuntimeError('absolute_actions needs an OSC controller config')
    else:
        notes.append('using robosuite default controller config')

    try:
        env = suite.make(**kwargs)
    except TypeError:            # robosuite 1.4 has no seed kwarg (np.random covers it)
        kwargs.pop('seed', None)
        env = suite.make(**kwargs)

    obs = env.reset()
    low, _high = env.action_spec
    return BuiltEnv(env=env, obs=obs, action_dim=len(low), notes=notes)
