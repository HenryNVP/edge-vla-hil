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

`put()` is the arrival path, and it keys on CAPTURE time (the plant's header stamp), not arrival
order. A slot keeps the newest capture: a message stamped earlier than the one already held is a
reordered straggler and is dropped (`stale_drops` counts them). Keeping the latest ARRIVAL
instead would let a jittered relay overwrite a fresh frame with an old one, so the policy would
see time run backwards; that is a bug in naive latest-value code, not a network effect worth
measuring, and dropping by stamp removes it.

`age()` is how stale the observation the policy is about to use is: now minus the capture time of
the oldest of the latest required streams. It is measured at the control tick, so it includes the
wait for the tick as well as the relay; at zero injected delay it sits between 0 and one
observation period. That is d_obs, the observation component of the delay a chunk has to bridge.
"""
from __future__ import annotations

import collections

import numpy as np

_STREAMS = ('image', 'wrist', 'proprio')


class ObsBuffer:
    """Latest-message slots plus the per-tick history the policy consumes."""

    def __init__(self, n_obs_steps: int, needs_wrist: bool) -> None:
        self.needs_wrist = needs_wrist
        self.image: np.ndarray | None = None
        self.wrist: np.ndarray | None = None
        self.proprio: np.ndarray | None = None
        self._history: collections.deque = collections.deque(maxlen=max(1, n_obs_steps))
        self._stamps: dict[str, float] = {}    # capture time (s) of each slot's current message
        self._since = float('-inf')            # captures before this belong to an old episode
        self.stale_drops = 0

    def put(self, stream: str, value: np.ndarray, stamp_s: float | None = None) -> bool:
        """Store a message in its slot unless it was captured before the one already there.

        Returns False (and counts it) for a reordered straggler. A message without a stamp is
        always accepted, but then contributes nothing to `age()`.
        """
        if stream not in _STREAMS:
            raise ValueError(f'unknown observation stream {stream!r}')
        if stamp_s is not None:
            if stamp_s < self._since:
                self.stale_drops += 1
                return False
            held = self._stamps.get(stream)
            if held is not None and stamp_s < held:
                self.stale_drops += 1
                return False
            self._stamps[stream] = stamp_s
        setattr(self, stream, value)
        return True

    def age(self, now_s: float) -> float | None:
        """Seconds since the oldest of the latest required streams was captured, or None."""
        required = ['image', 'proprio'] + (['wrist'] if self.needs_wrist else [])
        stamps = [self._stamps.get(k) for k in required]
        if any(t is None for t in stamps):
            return None
        return max(0.0, now_s - min(stamps))

    def clear(self, since_s: float | None = None) -> None:
        """Drop the history at an episode boundary.

        With `since_s` (the reset time), the latest-message slots are emptied too and anything
        captured before it is refused: those frames show the previous episode's scene, and under
        injected observation delay they keep arriving after the reset. Sampling them made the
        first chunk of every episode a plan for the scene that had just ended. Without it the
        slots stay valid (the old behaviour).
        """
        self._history.clear()
        if since_s is not None:
            self._since = since_s
            self.image = self.wrist = self.proprio = None
            self._stamps.clear()

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
