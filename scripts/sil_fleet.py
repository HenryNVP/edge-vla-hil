#!/usr/bin/env python3
"""N robots sharing one policy server: delay past H/2 from queueing, not from a scaled trace.

Paper 1's replayed WiFi only separates the strategies once the network delay is multiplied by
4-6; at the recorded scale every strategy ties. A multiplier is an assumption. A shared server is
a cause: each robot keeps one request in flight (invariant 5), the server runs one inference at
a time, first come first served, so a robot's round trip is its own recorded WiFi delay (x1)
plus the wait behind the other robots' requests plus its own inference. How long the queue gets
depends on how often the fleet requests, which depends on the strategy: waiting requests once
per chunk, overlapping strategies far more often. That load is part of what is compared.

Everything else is sil_step: the same executors, policy, env, episode loop (`episode`, driven
here in lockstep, one tick for every robot per global tick) and trace replay. A fleet of one
with `--service-ms 80` is exactly sil_step's `trace:<session>:1` (tested).

Scenes: robot i's env is seeded `seed + 1000 i`, so robot 0 sees sil_step's scenes, and episode
(i, k) is the same scene, on the same channel window, for every strategy and fleet size. A
robot that has its `--episodes` keeps running unrecorded episodes until the last robot is done,
so the server load does not fall away at the end of a cell.

    MUJOCO_GL=egl python3 scripts/sil_fleet.py --strategies synchronous,rtc --fleets 1,4 \
        --channel far_nlos --episodes 10 --out outputs/sil/fleet.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np

from sil_step import (CONTROL_HZ, EPISODE_COLUMNS, StepWorker, TraceDelay, episode, make_delay,
                      make_strategy)

TICK_MS = 1e3 / CONTROL_HZ


class FleetServer:
    """One GPU: inferences run one at a time, in the order requests reach it."""

    def __init__(self, service_ms: float) -> None:
        self.service_ms = service_ms
        self.free_ms = 0.0                     # global time the current inference ends


class FleetLink:
    """One robot's delay: its own replayed WiFi (network part only) plus the server's queue.

    A StepWorker delay callable (`needs_tick`): called with the robot's EPISODE tick, which
    `start_episode` maps to the fleet's global clock. The network delay is split evenly between
    up- and downlink.
    shortcut: requests are queued in the order they are issued, not the order they reach the
    server; a request with a slow uplink can be served ahead of one issued later that arrives
    first. Exact with equal uplinks; replace with an event queue if uplinks differ a lot.
    """
    needs_tick = True

    def __init__(self, server: FleetServer, net: TraceDelay) -> None:
        self.server = server
        self.net = net
        self.g0 = 0
        self.delays_ms: list[float] = []

    def start_episode(self, g0: int, scene: int) -> None:
        self.g0 = g0
        self.net.for_episode(scene)
        self.delays_ms = []

    def __call__(self, t_issue: int = 0) -> int:
        net = max(self.net.rtt_ms_at(t_issue) - self.net.inference_ms, 0.0)
        sent = (self.g0 + t_issue) * TICK_MS
        start = max(sent + net / 2, self.server.free_ms)
        self.server.free_ms = start + self.server.service_ms
        rtt = self.server.free_ms + net / 2 - sent
        self.delays_ms.append(rtt)
        return max(1, math.ceil(rtt / TICK_MS - 1e-9))


def run_fleet(robots: list[dict], *, episodes: int, deadline_ticks: int, on_episode,
              episode_kw: dict | None = None) -> list[int]:
    """Drive every robot's episodes in lockstep until each has `episodes` recorded.

    `robots[i]` holds env, obs, policy, executor, worker, link, and `reset()` -> the next first
    observation. `on_episode(i, k, Episode, link)` gets each recorded episode. Returns how many
    episodes each robot ran, the unrecorded load-keeping ones included.
    """
    kw = episode_kw or {}
    n = len(robots)
    gens: list = [None] * n
    ran = [0] * n
    g = 0
    while min(ran) < episodes:
        for j in range(n):
            i = (g + j) % n                    # rotate who is first in the queue at a tie
            r = robots[i]
            if gens[i] is None:
                r['link'].start_episode(g, scene=1000 * i + ran[i])
                gens[i] = episode(r['env'], r['obs'], r['policy'], r['executor'], r['worker'],
                                  deadline_ticks=deadline_ticks, **kw)
            try:
                next(gens[i])
            except StopIteration as stop:
                if ran[i] < episodes:
                    on_episode(i, ran[i], stop.value, r['link'])
                ran[i] += 1
                gens[i] = None
                r['obs'] = r['reset']()
        g += 1
    return ran


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', required=True, help='episodes CSV (appended)')
    p.add_argument('--env', default='NutAssemblySquare')
    p.add_argument('--backend', default='dp')
    p.add_argument('--weights', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'checkpoints', 'dp_square_ph_image_cnn.ckpt'))
    p.add_argument('--denoise-steps', type=int, default=4)
    p.add_argument('--strategies', default='synchronous,noskip_async,rtc_noskip,rtc')
    p.add_argument('--fleets', default='1,2,4,8', help='robots per server, comma-separated')
    p.add_argument('--service-ms', type=float, default=80.0,
                   help='one inference on the server (80 = what the WiFi replay assumes)')
    p.add_argument('--channel', default='far_nlos', help='recorded WiFi session, replayed x1')
    p.add_argument('--world-speed', type=float, default=1.0)
    p.add_argument('--deadline-s', type=float, default=20.0)
    p.add_argument('--episodes', type=int, default=10, help='recorded per robot')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    import run_manifest
    run_manifest.record(args.out, {'kind': 'sil_fleet', 'config': vars(args)})

    from evh_controller.policy import make_policy
    from evh_plant.env_factory import EnvSpec, build_env

    policy = make_policy(args.backend, args.weights, denoise_steps=args.denoise_steps)
    deadline_ticks = int(round(args.deadline_s * CONTROL_HZ))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(EPISODE_COLUMNS + ['fleet', 'robot', 'rtt_p50_ms', 'rtt_p95_ms'])
        for n in (int(x) for x in args.fleets.split(',')):
            for strategy in args.strategies.split(','):
                server = FleetServer(args.service_ms)
                robots = []
                for i in range(n):
                    built = build_env(EnvSpec(env_name=args.env, action_hz=CONTROL_HZ,
                                              max_episode_s=args.deadline_s + 1.0,
                                              seed=args.seed + 1000 * i,
                                              absolute_actions=policy.absolute_actions))
                    link = FleetLink(server, make_delay(f'trace:{args.channel}:1'))
                    worker = StepWorker(policy, link)
                    robots.append(dict(env=built.env, obs=built.obs, policy=policy, link=link,
                                       worker=worker, reset=built.env.reset,
                                       executor=make_strategy(strategy, worker, policy)))
                lab = (f'strat={strategy}_fleet={n}_service={args.service_ms:g}'
                       f'_chan={args.channel}_alpha={args.world_speed:g}'
                       f'_deadline={args.deadline_s:g}')
                wins = []

                def record(i, k, ep, link):
                    rtt = np.asarray(link.delays_ms) if link.delays_ms else np.zeros(1)
                    wins.append(ep.success)
                    w.writerow([lab, k, int(ep.success), f'{ep.wall_s:.2f}', f'{ep.sim_s:.3f}',
                                ep.env_steps, f'{ep.idle / max(ep.ticks, 1):.3f}',
                                f'{ep.path_m * 1e3:.0f}', ep.requests, n, i,
                                f'{np.percentile(rtt, 50):.0f}', f'{np.percentile(rtt, 95):.0f}'])
                    f.flush()

                run_fleet(robots, episodes=args.episodes, deadline_ticks=deadline_ticks,
                          on_episode=record,
                          episode_kw=dict(world_speed=args.world_speed,
                                          absolute=policy.absolute_actions))
                for r in robots:
                    r['env'].close()
                print(f'{lab}: {sum(wins)}/{len(wins)}', flush=True)


if __name__ == '__main__':
    main()
