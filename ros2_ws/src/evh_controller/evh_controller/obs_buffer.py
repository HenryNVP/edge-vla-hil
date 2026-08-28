"""The controller's observation history — the other half of the policy's input contract.

Split out of `controller_node.py` so the node holds subscriptions and a tick, and the shape of
what reaches the policy is defined (and tested) in one ROS-free place.

The contract, stacked over the last `n_obs_steps` control ticks, oldest first:

    agentview  uint8 [To, H, W, 3]
    wrist      uint8 [To, H, W, 3]     only when the policy needs it
    proprio    float [To, D]           [eef_pos(3), eef_quat(4, xyzw), gripper_qpos(2)]

Two details are load-bearing. Sampling happens on the CONTROL tick, not on message arrival: the
history must be a regular time series at `control_hz` regardless of how erratically the relay
delivers observations, because that is the spacing the policy was trained on — under heavy delay
the same frame legitimately repeats in consecutive slots, and that is the signal, not a bug. And
`ready()` gates on every required stream having arrived at least once, so the first chunk is never
computed against a half-populated observation.
"""
from __future__ import annotations

import collections

import numpy as np


class ObsBuffer:
    """Latest-message slots plus the per-tick history the policy consumes."""

    def __init__(self, n_obs_steps: int, needs_wrist: bool) -> None:
        self.needs_wrist = needs_wrist
        self.image: np.ndarray | None = None
        self.wrist: np.ndarray | None = None
        self.proprio: np.ndarray | None = None
        self._history: collections.deque = collections.deque(maxlen=max(1, n_obs_steps))

    def clear(self) -> None:
        """Drop the history at an episode boundary; the latest-message slots stay valid."""
        self._history.clear()

    def ready(self) -> bool:
        return (self.image is not None and self.proprio is not None
                and (self.wrist is not None or not self.needs_wrist))

    def sample(self) -> dict | None:
        """Append the current observation to the history and return the stacked dict.

        None until every required stream has arrived. Call exactly once per control tick.
        """
        if not self.ready():
            return None
        self._history.append((self.image, self.wrist, self.proprio))
        obs = {
            'agentview': np.stack([h[0] for h in self._history]),
            'proprio': np.stack([h[2] for h in self._history]),
        }
        if self.wrist is not None:
            obs['wrist'] = np.stack([h[1] for h in self._history])
        return obs
