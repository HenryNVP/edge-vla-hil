#!/usr/bin/env python3
"""Step-based software-in-the-loop: the chunk executors on robosuite, delay in ticks, no ROS.

The HiL testbed measures what a real network does to chunked execution; this measures the same
thing with the network replaced by its effect on the one quantity the executors see, the number
of control ticks between a request and its chunk. Everything else is the testbed's own code:

  * the strategies are `evh_controller.chunk_executor`, unchanged (make_executor);
  * the policy is any `ChunkPolicy` from `make_policy` (DP, ACT, ONNX), with the same chunk
    truncation (`max_chunk_actions`);
  * the env is `evh_plant.env_factory.build_env` with the plant's seeding, so scene k is the same
    scene here as in the testbed; the idle-tick hold is the plant's `hold_action`.

What differs from the testbed, deliberately:

  * the OSC runs at the policy's 20 Hz (the co-located eval's protocol, and the rate the
    checkpoint was trained at) with no reactive layer between them — E3 found the layer changes
    nothing;
  * delay is an integer number of ticks, fixed or drawn per request, instead of emerging from a
    channel. E4 found autocorrelated and independent delay at equal mean indistinguishable for
    buffered execution, which is what licenses this; the validation gate is that this reproduces
    the testbed's cells;
  * a WORLD-SPEED knob for ticks on which the executor has no action: 1 = the world keeps
    stepping (the arm holds its last target), 0 = the world is frozen (not stepped). Every tick
    counts toward the deadline either way. On a quasi-static task the two should agree; that
    agreement is a measurement, not an assumption.

Timing: a request issued at tick t is computed immediately and delivered at tick t + d (d >= 1:
the executor polls before it issues, so a chunk requested at t is first usable at t + 1, like
the testbed, where inference alone is ~2 ticks).

Run inside the host image (robosuite, torch and the DP deps are there):
    MUJOCO_GL=egl python3 scripts/sil_step.py --strategies synchronous,rtc --delays 10,18 \
        --episodes 30 --out outputs/sil/check.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import os
import sys
import time
from typing import Callable

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _pkg in ('evh_controller', 'evh_plant'):
    _path = os.path.join(_REPO, 'ros2_ws', 'src', _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evh_controller.inference_worker import Arrival  # noqa: E402  (numpy only)
from evh_plant.env_factory import hold_action  # noqa: E402  (numpy only)

CONTROL_HZ = 20.0


# ------------------------------------------------------------------ delay
def fixed_delay(d: int) -> Callable[[], int]:
    if d < 1:
        raise ValueError(f'delay must be >= 1 tick (a chunk is first usable the tick after it is '
                         f'requested), got {d}')
    return lambda: d


def sampled_delay(values_ticks, seed: int = 0) -> Callable[[], int]:
    """Draw each request's delay from an empirical set (e.g. a WiFi trace's round trips, in
    ticks). Uses its own generator: the scene sequence depends on numpy's global RNG, which the
    plant seeds, and must not shift because a delay was drawn."""
    vals = np.maximum(1, np.asarray(values_ticks, dtype=int))
    rng = np.random.default_rng(seed)
    return lambda: int(rng.choice(vals))


def wifi_rtt_ms(robot_rows: list[dict], server_rows: list[dict], inference_ms: float = 80.0,
                trim_s: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Round trip, in ms, of a chunk request made with each camera frame of a recorded WiFi trace.

    `robot_rows`/`server_rows` are the two sides of `scripts/wifi_trace.py record`. A request sent
    with the image at robot time s travels up with that image (its measured one-way delay), is
    computed for `inference_ms`, and comes back with the one-way delay the downlink had at that
    moment: that of the first chunk message the recorder sent after it. The recorder sends chunks
    on a 5 Hz schedule, but a real server replies as soon as inference ends, so the wait for the
    schedule is NOT charged (it added ~100 ms to every round trip, 2026-09-30). A lost message is
    not a lost request: it costs the wait for the next message that did arrive, which is what a
    retry on a best-effort link costs, and that wait IS charged. Clocks
    are aligned with the recorder's own probe offset. Returns (send times in s, from 0; rtt in
    ms), in time order, so the trace's autocorrelation and stalls are kept, not resampled.
    """
    from wifi_trace import estimate_offset

    probes = [(float(r['t1']), float(r['t2']), float(r['t3']), float(r['t4']))
              for r in robot_rows if r['stream'] == 'probe']
    off = estimate_offset(probes).offset_s            # server clock = robot clock + off
    sent = {int(r['seq']): float(r['sent']) for r in robot_rows if r['stream'] == 'sent:image'}
    got = {int(r['seq']): float(r['recv']) - off for r in server_rows if r['stream'] == 'image'}
    down_sent = sorted((float(r['sent']) - off, int(r['seq'])) for r in server_rows
                       if r['stream'] == 'sent:chunk')
    down_got = {int(r['seq']): float(r['recv']) for r in robot_rows if r['stream'] == 'chunk'}
    if not sent or not got or not down_sent or not down_got:
        raise ValueError('trace lacks image or chunk streams in one direction')
    t_first, t_last = min(sent.values()) + trim_s, max(sent.values()) - trim_s
    d_times = np.array([t for t, _ in down_sent])
    d_arrive = np.array([down_got.get(q, np.inf) for _, q in down_sent])
    # earliest delivery among messages sent at or after index k (a later message stands in for a
    # lost one): a suffix minimum, so each lookup is one index
    d_first_arrival = np.minimum.accumulate(d_arrive[::-1])[::-1]
    seqs = sorted(q for q, t in sent.items() if t_first <= t <= t_last)
    up_arrive = np.array([got.get(q, np.inf) for q in seqs])
    up_first_arrival = np.minimum.accumulate(up_arrive[::-1])[::-1]
    times, rtts = [], []
    for k, q in enumerate(seqs):
        at_server = up_first_arrival[k]
        if not np.isfinite(at_server):
            break
        ready = at_server + inference_ms / 1e3
        j = int(np.searchsorted(d_times, ready))
        if j >= len(d_times) or not np.isfinite(d_first_arrival[j]):
            break
        # downlink: this message's one-way delay, plus, if it was lost, the wait until a later one
        # was delivered (measured from this message's send, so a loss costs its real gap)
        downlink = d_first_arrival[j] - d_times[j]
        times.append(sent[q] - t_first)
        rtts.append(1e3 * (ready - sent[q] + downlink))
    return np.asarray(times), np.asarray(rtts)


class TraceDelay:
    """Per-request delay, in ticks, read from a recorded round-trip series at the request's time.

    A request issued at episode tick t of episode k reads the series at `offset_k + t / 20 s`, so
    delay keeps the trace's time structure (a stall lasts as long as it did on the air) and every
    strategy sees the same channel in episode k (`for_episode(k)` sets a seeded offset). `scale`
    multiplies the network part only: rtt = inference + scale * (recorded rtt - inference).
    """

    def __init__(self, times_s, rtt_ms, *, scale: float = 1.0, inference_ms: float = 80.0,
                 seed: int = 0) -> None:
        self.times = np.asarray(times_s, dtype=float)
        self.inference_ms = inference_ms
        rtt = np.asarray(rtt_ms, dtype=float)
        self.rtt_ms = inference_ms + scale * np.maximum(rtt - inference_ms, 0.0)
        self.seed = seed
        self.offset_s = 0.0

    def for_episode(self, k: int) -> TraceDelay:
        span = self.times[-1] - 60.0          # leave a minute of trace after the start
        self.offset_s = float(np.random.default_rng((self.seed, k)).uniform(0.0, max(span, 0.0)))
        return self

    needs_tick = True                          # StepWorker passes the request's tick

    def rtt_ms_at(self, t_issue: int = 0) -> float:
        at = self.offset_s + t_issue / CONTROL_HZ
        return float(self.rtt_ms[min(int(np.searchsorted(self.times, at)), len(self.times) - 1)])

    def __call__(self, t_issue: int = 0) -> int:
        return max(1, int(np.ceil(self.rtt_ms_at(t_issue) / (1e3 / CONTROL_HZ))))


WIFI_DIR = os.path.join(_REPO, 'outputs', 'wifi')
_TRACE_CACHE: dict = {}


def make_delay(token: str, wifi_dir: str = WIFI_DIR):
    """A `--delays` token -> the worker's delay: `10` is a fixed 10 ticks; `trace:<session>:<scale>`
    replays the recorded WiFi session `<session>_rob.csv`/`<session>_srv.csv` with its network
    delay multiplied by <scale> (a TraceDelay; call `.for_episode(k)` before episode k)."""
    if not token.startswith('trace:'):
        return fixed_delay(int(token))
    _, session, *rest = token.split(':')
    scale = float(rest[0]) if rest else 1.0
    if session not in _TRACE_CACHE:
        rows = []
        for side in ('rob', 'srv'):
            with open(os.path.join(wifi_dir, f'{session}_{side}.csv'), newline='') as f:
                rows.append(list(csv.DictReader(f)))
        _TRACE_CACHE[session] = wifi_rtt_ms(*rows)
    return TraceDelay(*_TRACE_CACHE[session], scale=scale)


def make_strategy(strategy: str, worker, policy, s_wait: int = 0, s_min: int = 0):
    """make_executor, plus the coverage-aware family's knobs when given: the waiting-mode chunk
    length `s_wait`, and `s_min`, the fewest ticks between time-aligned requests (on a large model
    replanning every tick costs a guided inference per tick; LIBERO uses 10, like its RTC)."""
    from evh_controller.chunk_executor import _REGISTRY, CoverageAwareExecutor, make_executor
    if strategy == 'continuation_inpaint':
        # the queue as a hard inpainting prefix, no state model: oracle_lift's `_inpaint` arm
        from evh_controller.chunk_executor.continuation import ContinuationExecutor
        return ContinuationExecutor(worker, policy, roll_fn=None, inpaint=True)
    cls = _REGISTRY.get(strategy)
    # the whole family, subclasses included (they take the knobs through **kw)
    if cls is not None and issubclass(cls, CoverageAwareExecutor) and (s_wait or s_min):
        kw = {}
        if s_wait:
            kw['s_wait'] = s_wait
        if s_min:
            kw['exec_horizon_min'] = s_min
        return cls(worker, policy, **kw)
    return make_executor(strategy, worker, policy)


class StepWorker:
    """The `InferenceWorker` interface, in ticks: compute on request, deliver `delay()` polls later.

    One request in flight, as in the testbed (invariant 5). `compute_s` is the real inference
    time, reported but not charged: in this harness only `delay` spends ticks.
    """

    def __init__(self, policy, delay: Callable[[], int]) -> None:
        self.policy = policy
        self.delay = delay
        self._pending: list | None = None     # [polls remaining, Arrival]
        self.requests = 0

    @property
    def busy(self) -> bool:
        return self._pending is not None

    def try_request(self, obs, t_issue, epoch, prefix=None, weights=None) -> bool:
        if self._pending is not None:
            return False
        t0 = time.perf_counter()
        if prefix is not None and len(prefix) and weights is not None:
            chunk = self.policy.predict_inpaint(obs, prefix, weights)
        else:
            chunk = self.policy.predict(obs)
        self.requests += 1
        delay = self.delay(t_issue) if getattr(self.delay, 'needs_tick', False) else self.delay()
        self._pending = [delay, Arrival(np.asarray(chunk), t_issue, epoch,
                                        time.perf_counter() - t0)]
        return True

    def poll(self):
        if self._pending is None:
            return None
        self._pending[0] -= 1
        if self._pending[0] > 0:
            return None
        arrival, self._pending = self._pending[1], None
        return arrival

    def shutdown(self) -> None:
        self._pending = None


# ------------------------------------------------------------------ episode
def frames(obs: dict, needs_wrist: bool = True):
    """robosuite obs -> (agentview, wrist, proprio), the plant's observation contract."""
    agent = np.flipud(np.asarray(obs['agentview_image'])).astype(np.uint8)
    wrist = (np.flipud(np.asarray(obs['robot0_eye_in_hand_image'])).astype(np.uint8)
             if needs_wrist else None)
    proprio = np.concatenate([np.asarray(obs['robot0_eef_pos'], np.float32),
                              np.asarray(obs['robot0_eef_quat'], np.float32),
                              np.asarray(obs['robot0_gripper_qpos'], np.float32)])
    return agent, wrist, proprio


def stack(hist, needs_wrist: bool) -> dict:
    out = {'agentview': np.stack([h[0] for h in hist]),
           'proprio': np.stack([h[2] for h in hist])}
    if needs_wrist:
        out['wrist'] = np.stack([h[1] for h in hist])
    return out


@dataclasses.dataclass
class Episode:
    success: bool
    ticks: int            # control ticks elapsed when it ended (success tick + 1, or the deadline)
    env_steps: int        # ticks on which the world advanced
    idle: int             # ticks with no action from the executor
    path_m: float         # commanded end-effector path (sum of |dpos| between executed actions)
    requests: int
    wall_s: float

    @property
    def sim_s(self) -> float:
        return self.ticks / CONTROL_HZ


def run_episode(env, obs, policy, executor, worker, **kw) -> Episode:
    """One episode from an already-reset env whose first observation is `obs` (see `episode`)."""
    gen = episode(env, obs, policy, executor, worker, **kw)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value


def episode(env, obs, policy, executor, worker, *, deadline_ticks: int,
            world_speed: float = 1.0, absolute: bool = True,
            is_success=None, frame_fn=frames, stack_fn=stack, hold_fn=None):
    """One episode as a generator: yields once per control tick, returns the Episode. Driving
    several of these in lockstep is how sil_fleet.py shares one server between robots.

    `frame_fn`/`stack_fn` turn env observations into the policy's input, and `hold_fn(obs)` is
    the command for an idle tick before any action (default: the plant's `hold_action`), so an
    env with a different observation or action convention (LIBERO through LeRobot) plugs in
    without changing the loop.

    An idle tick after the first action repeats the last command in ABSOLUTE mode (it is a
    target: repeating it holds the pose) but sends the hold in DELTA mode, where repeating the
    last delta would keep the arm moving (invariant 2).
    """
    is_success = is_success or (lambda e: bool(e._check_success()))
    hold_fn = hold_fn or (lambda o: hold_action(o, absolute, policy.action_dim))
    needs_wrist = bool(getattr(policy, 'needs_wrist', True))
    executor.reset()
    requests0 = worker.requests
    hist = collections.deque(maxlen=policy.n_obs_steps)
    first = frame_fn(obs, needs_wrist)
    for _ in range(policy.n_obs_steps):     # pad the history with the first frame, as at reset
        hist.append(first)
    last_action = None
    env_steps = idle = 0
    path = 0.0
    acc = 0.0
    t0 = time.perf_counter()
    for t in range(deadline_ticks):
        action = executor.step(stack_fn(hist, needs_wrist), t)
        if action is None:
            idle += 1
            acc += world_speed
            if acc < 1.0 - 1e-9:
                yield                      # frozen world: the tick passes, nothing moves
                continue
            acc -= 1.0
            # absolute: the arm keeps its last target; delta, or before any action: the hold
            command = last_action if (absolute and last_action is not None) else hold_fn(obs)
        else:
            command = np.asarray(action, dtype=np.float32)
            if absolute and last_action is not None:   # a path only exists between targets
                path += float(np.linalg.norm(command[:3] - last_action[:3]))
            last_action = command
        obs, _r, _done, _info = env.step(command)
        env_steps += 1
        hist.append(frame_fn(obs, needs_wrist))
        if is_success(env):
            return Episode(True, t + 1, env_steps, idle, path, worker.requests - requests0,
                           time.perf_counter() - t0)
        yield
    return Episode(False, deadline_ticks, env_steps, idle, path, worker.requests - requests0,
                   time.perf_counter() - t0)


# ------------------------------------------------------------------ CLI
EPISODE_COLUMNS = ['condition', 'scene', 'success', 'wall_s', 'sim_s', 'env_steps', 'idle_frac',
                   'path_mm', 'requests']


def label(strategy: str, d, k: int, world_speed: float, deadline_s: float) -> str:
    return f'strat={strategy}_d={d}_k={k}_alpha={world_speed:g}_deadline={deadline_s:g}'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', required=True, help='episodes CSV (appended)')
    p.add_argument('--env', default='NutAssemblySquare')
    p.add_argument('--backend', default='dp')
    p.add_argument('--weights', default=os.path.join(_REPO, 'checkpoints',
                                                     'dp_square_ph_image_cnn.ckpt'))
    p.add_argument('--denoise-steps', type=int, default=4, help='paper-1 cells used 4')
    p.add_argument('--strategies', default='synchronous,naive_async,temporal_ensemble,rtc')
    p.add_argument('--delays', default='2,10,18',
                   help='request->arrival: ticks (10) or a replayed WiFi session '
                        '(trace:far_nlos:2 = network delay x2), comma-separated')
    p.add_argument('--s-wait', type=int, default=0,
                   help='coverage-aware family: waiting-mode chunk length (0 = full chunk)')
    p.add_argument('--max-chunk-actions', type=int, default=0, help='0 = the full chunk')
    p.add_argument('--world-speed', type=float, default=1.0)
    p.add_argument('--deadline-s', type=float, default=20.0)
    p.add_argument('--episodes', type=int, default=30)
    p.add_argument('--seed', type=int, default=0, help='the plant default; scene k follows it')
    args = p.parse_args()
    import run_manifest
    run_manifest.record(args.out, {'kind': 'sil_step', 'config': vars(args)})

    from evh_controller.chunk_executor import make_executor
    from evh_controller.policy import make_policy
    from evh_plant.env_factory import EnvSpec, build_env

    policy = make_policy(args.backend, args.weights, denoise_steps=args.denoise_steps,
                         max_chunk_actions=args.max_chunk_actions)
    k = policy.chunk_size
    deadline_ticks = int(round(args.deadline_s * CONTROL_HZ))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(EPISODE_COLUMNS)
        for d in args.delays.split(','):
            for strategy in args.strategies.split(','):
                # a fresh env per cell, seeded like the plant, so scene k is the same everywhere
                built = build_env(EnvSpec(env_name=args.env, action_hz=CONTROL_HZ,
                                          max_episode_s=args.deadline_s + 1.0, seed=args.seed,
                                          absolute_actions=policy.absolute_actions))
                env, obs = built.env, built.obs
                delay = make_delay(d)
                worker = StepWorker(policy, delay)
                executor = make_strategy(strategy, worker, policy, args.s_wait)
                lab = label(strategy, d, k, args.world_speed, args.deadline_s)
                wins = 0
                for scene in range(args.episodes):
                    if isinstance(delay, TraceDelay):
                        delay.for_episode(scene)      # the same channel for every strategy
                    ep = run_episode(env, obs, policy, executor, worker,
                                     deadline_ticks=deadline_ticks,
                                     world_speed=args.world_speed,
                                     absolute=policy.absolute_actions)
                    wins += ep.success
                    w.writerow([lab, scene, int(ep.success), f'{ep.wall_s:.2f}',
                                f'{ep.sim_s:.3f}', ep.env_steps,
                                f'{ep.idle / max(ep.ticks, 1):.3f}', f'{ep.path_m * 1e3:.0f}',
                                ep.requests])
                    f.flush()
                    obs = env.reset()
                env.close()
                print(f'{lab}: {wins}/{args.episodes}', flush=True)


if __name__ == '__main__':
    main()
