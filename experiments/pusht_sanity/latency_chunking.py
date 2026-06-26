"""Delayed-chunk execution model + strategies (pure numpy, no ROS / no heavy deps).

This is the experimental core of the PushT sanity check. It simulates the async observation-action
delay loop *deterministically* (no threads): a replan issued at step t observes o_t but the chunk
only becomes available at step t+d, where d is the inference+network delay in control steps. While
waiting, the robot keeps executing the previous chunk (open-loop). When the new chunk arrives, the
chosen STRATEGY decides how to splice it onto the in-flight actions.

This mirrors the real ROS testbed (evh_latency injects the delay; evh_controller's chunk_executor
splices) closely enough to validate the *phenomenon and the experiment design* before investing in
robosuite. The strategy names match evh_controller/chunk_executor.py.

Strategies:
  synchronous       -- pause (hold last action) until the chunk arrives, then execute it fully.
  naive_async       -- keep executing the old chunk while waiting; on arrival jump to new[0]
                       (the discontinuity RTC is designed to avoid).
  temporal_ensemble -- exponentially-weighted average over time-aligned overlapping chunks.
  rtc_freeze        -- the FREEZE-ONLY approximation of RTC: on arrival, skip the d entries that
                       overlapped the in-flight execution and continue from new[d], so the
                       committed prefix is preserved. NOTE: this is freeze-only; true RTC also
                       *guided-inpaints* the unfrozen tail during denoising, which needs access to
                       the policy's denoiser and is out of scope for this black-box sanity rig.

Latency knobs: latency_steps (mean d), jitter_steps (std, gaussian), drop_prob (chunk lost).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Callable, Protocol

import numpy as np

# A policy maps an observation to an action chunk of shape [H, action_dim].
ChunkPolicyFn = Callable[[np.ndarray], np.ndarray]


class Env(Protocol):
    def reset(self) -> np.ndarray: ...
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]: ...


# --------------------------------------------------------------------- delay
@dataclass
class DelayModel:
    latency_steps: float = 0.0
    jitter_steps: float = 0.0
    drop_prob: float = 0.0
    seed: int = 0
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def sample_delay(self) -> int:
        d = self.latency_steps
        if self.jitter_steps > 0.0:
            d += self._rng.gauss(0.0, self.jitter_steps)
        return max(0, int(round(d)))

    def dropped(self) -> bool:
        return self.drop_prob > 0.0 and self._rng.random() < self.drop_prob


# ----------------------------------------------------------------- strategies
class Strategy:
    """Splice policy for newly-arrived chunks. Subclasses mirror chunk_executor.py."""
    name = 'base'
    pause_when_waiting = False     # synchronous holds the arm while waiting

    def reset(self) -> None:
        self.chunk: np.ndarray | None = None
        self.idx = 0

    def on_arrival(self, executed_since_issue: int, new_chunk: np.ndarray) -> None:
        self.chunk = new_chunk
        self.idx = 0

    def act(self, last_action: np.ndarray | None) -> np.ndarray:
        if self.chunk is None:
            return last_action if last_action is not None else np.zeros(1)
        a = self.chunk[min(self.idx, self.chunk.shape[0] - 1)]
        self.idx += 1
        return a


class Synchronous(Strategy):
    name = 'synchronous'
    pause_when_waiting = True
    # on_arrival: default (start the fresh chunk at 0 — robot was paused, nothing lost).


class NaiveAsync(Strategy):
    name = 'naive_async'
    # keep executing old chunk while waiting (handled by runner); jump to new[0] on arrival.


class RTCFreeze(Strategy):
    name = 'rtc_freeze'

    def on_arrival(self, executed_since_issue: int, new_chunk: np.ndarray) -> None:
        # Freeze the d=executed_since_issue entries that overlapped the in-flight execution:
        # continue from the time-aligned index instead of replaying the past.
        self.chunk = new_chunk
        self.idx = min(executed_since_issue, new_chunk.shape[0] - 1)


class TemporalEnsemble(Strategy):
    name = 'temporal_ensemble'

    def __init__(self, m: float = 0.1) -> None:
        self.m = m
        self.reset()

    def reset(self) -> None:
        self.chunk = None        # read by the runner's `exhausted`/replan gate
        self.idx = 0
        self.buffer: list[tuple[int, np.ndarray]] = []   # (start_step, chunk)
        self.t = 0

    def on_arrival(self, executed_since_issue: int, new_chunk: np.ndarray) -> None:
        # the new chunk's index 0 corresponds to the issue time = now - executed_since_issue
        self.buffer.append((self.t - executed_since_issue, new_chunk))
        self.chunk = new_chunk   # mark non-None so the replan cadence follows replan_period

    def act(self, last_action: np.ndarray | None) -> np.ndarray:
        votes, weights = [], []
        for start, chunk in self.buffer:
            k = self.t - start
            if 0 <= k < chunk.shape[0]:
                votes.append(chunk[k])
                weights.append(np.exp(-self.m * k))
        self.t += 1
        if not votes:
            return last_action if last_action is not None else np.zeros(
                self.buffer[-1][1].shape[1] if self.buffer else 1)
        w = np.asarray(weights)
        return np.average(np.stack(votes), axis=0, weights=w / w.sum())


_STRATEGIES = {s.name: s for s in (Synchronous, NaiveAsync, RTCFreeze, TemporalEnsemble)}


def make_strategy(name: str) -> Strategy:
    if name not in _STRATEGIES:
        raise ValueError(f'unknown strategy {name!r}; options: {sorted(_STRATEGIES)}')
    return _STRATEGIES[name]()


# -------------------------------------------------------------------- episode
@dataclass
class EpisodeResult:
    steps: int
    total_reward: float
    max_reward: float
    success: bool
    paused_steps: int            # control steps spent waiting (synchronous throughput cost)
    replans: int


def run_episode(
    env: Env,
    policy: ChunkPolicyFn,
    strategy: Strategy,
    delay: DelayModel,
    *,
    replan_period: int = 8,        # issue a replan every N control steps (async strategies)
    max_steps: int = 300,
    success_reward: float = 0.95,  # PushT coverage threshold
) -> EpisodeResult:
    """Run one episode of the delayed-chunk loop with the given strategy and delay model."""
    obs = env.reset()
    strategy.reset()

    pending: tuple[int, int, np.ndarray] | None = None   # (arrival_t, issue_t, chunk)
    last_issue = -(10 ** 9)
    last_action: np.ndarray | None = None
    total_r, max_r, paused, replans = 0.0, -np.inf, 0, 0
    done = False

    for t in range(max_steps):
        exhausted = strategy.chunk is None or strategy.idx >= strategy.chunk.shape[0]

        # 1) issue a replan?
        if pending is None:
            want = exhausted if strategy.pause_when_waiting else (t - last_issue) >= replan_period
            if want or strategy.chunk is None:
                if not delay.dropped():
                    chunk = np.asarray(policy(obs), dtype=np.float64)
                    pending = (t + delay.sample_delay(), t, chunk)
                    replans += 1
                last_issue = t

        # 2) deliver an arrived chunk
        if pending is not None and t >= pending[0]:
            arrival_t, issue_t, chunk = pending
            strategy.on_arrival(t - issue_t, chunk)
            pending = None

        # 3) pick the action for this step
        waiting = pending is not None and (
            strategy.chunk is None or strategy.idx >= strategy.chunk.shape[0])
        if strategy.pause_when_waiting and waiting:
            paused += 1                       # holding position, waiting for the chunk to arrive
            action = last_action if last_action is not None else np.zeros(pending[2].shape[1])
        else:
            action = strategy.act(last_action)
        last_action = action

        obs, reward, done = env.step(np.asarray(action, dtype=np.float64))
        total_r += reward
        max_r = max(max_r, reward)
        if done:
            break

    return EpisodeResult(
        steps=t + 1, total_reward=total_r, max_reward=float(max_r),
        success=bool(max_r >= success_reward), paused_steps=paused, replans=replans)


# ------------------------------------------------------- mock env/policy (tests)
class PointMassEnv:
    """Toy 2-D double integrator reaching a goal; reward = coverage in [0,1].

    Jerky action switching overshoots (momentum), so it qualitatively rewards continuity — enough
    to exercise the strategies in tests without gym/torch.
    """
    def __init__(self, goal=(1.0, 1.0), dt=0.1, tol=0.03, max_speed=2.0):
        self.goal = np.asarray(goal, float)
        self.dt, self.tol, self.max_speed = dt, tol, max_speed

    def reset(self):
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        return self._obs()

    def _obs(self):
        return np.concatenate([self.pos, self.vel])

    def step(self, action):
        action = np.clip(action, -self.max_speed, self.max_speed)
        self.vel = 0.6 * self.vel + 0.4 * action         # inertia -> overshoot on jumps
        self.pos = self.pos + self.vel * self.dt
        dist = np.linalg.norm(self.goal - self.pos)
        reward = float(np.exp(-dist))                    # in (0,1], 1 at the goal
        done = dist < self.tol
        return self._obs(), reward, done


def make_mock_policy(goal=(1.0, 1.0), horizon=16, gain=3.0) -> ChunkPolicyFn:
    """A closed-form 'policy': from the observed state, plan a chunk of velocities to the goal."""
    goal = np.asarray(goal, float)

    def policy(obs: np.ndarray) -> np.ndarray:
        pos = obs[:2]
        chunk = []
        p = pos.copy()
        for _ in range(horizon):
            v = gain * (goal - p)
            chunk.append(v)
            p = p + v * 0.1 * 0.4
        return np.asarray(chunk)
    return policy
