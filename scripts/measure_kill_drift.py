#!/usr/bin/env python3
"""Measure how far the arm drifts when the reactive layer dies mid-motion, watchdog on vs off.

The claim this backs (README, CLAUDE.md "Contracts"): in delta mode a cached /cmd/action is a
*velocity* command, so a plant that keeps re-applying it after its sender dies walks the arm away;
the deadline watchdog drops the cache and falls through to `_hold_action`. It used to rest on one
hand measurement (189 mm vs 8 mm in 4 s). This repeats it as a paired A/B:

  per trial k (scene seed k), for each arm (`action_watchdog` true / false on the plant):
    start the plant (delta mode) and the real reactive node as separate processes;
    stream delta waypoints at control_hz like the controller would, so the arm is moving;
    SIGKILL the reactive node's process group (a crash, not a clean shutdown);
    record /obs/ee_pose for `--window` s and report the EE displacement from the kill.

The speed at the kill is recorded too: without the watchdog the drift should be about speed x
window (until a joint limit), so a small drift with a small speed is not a pass. Needs ROS sourced:

    docker run --rm --gpus all --network host -e ROS_DOMAIN_ID=$RANDOM -v ~/edge-vla-hil:/ws \\
      edge-vla-hil:host python3 /ws/scripts/measure_kill_drift.py --kills 10
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import statistics
import subprocess
import sys
import time

import numpy as np

FIELDS = ['arm', 'trial', 'seed', 'speed_at_kill_mm_s', 'drift_mm', 'path_mm', 'samples']


def drift_after_kill(t: np.ndarray, xyz: np.ndarray, t_kill: float, window_s: float,
                     speed_span_s: float = 0.2) -> dict:
    """Displacement and path length of the EE over [t_kill, t_kill + window_s], in mm.

    `t` are receive times (s, monotonic), `xyz` the matching positions (m). The start point is the
    last sample at or before the kill; speed is averaged over the `speed_span_s` before it.
    """
    t, xyz = np.asarray(t, float), np.asarray(xyz, float)
    before = np.flatnonzero(t <= t_kill)
    if before.size == 0:
        raise ValueError('no EE sample before the kill')
    i0 = before[-1]
    after = np.flatnonzero((t > t_kill) & (t <= t_kill + window_s))
    if after.size == 0:
        raise ValueError('no EE sample after the kill')
    seg = xyz[i0:after[-1] + 1]
    j = np.flatnonzero(t >= t[i0] - speed_span_s)[0]
    dt = t[i0] - t[j]
    speed = float(np.linalg.norm(xyz[i0] - xyz[j]) / dt) if dt > 0 else 0.0
    return {'drift_mm': 1e3 * float(np.linalg.norm(seg[-1] - seg[0])),
            'path_mm': 1e3 * float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()),
            'speed_at_kill_mm_s': 1e3 * speed,
            'samples': int(after.size)}


def summarize(rows: list[dict]) -> dict:
    """Per arm: n, median and range of drift_mm and of speed at the kill."""
    out = {}
    for arm in sorted({r['arm'] for r in rows}):
        d = [float(r['drift_mm']) for r in rows if r['arm'] == arm]
        v = [float(r['speed_at_kill_mm_s']) for r in rows if r['arm'] == arm]
        out[arm] = {'n': len(d), 'drift_median': statistics.median(d), 'drift_min': min(d),
                    'drift_max': max(d), 'speed_median': statistics.median(v)}
    return out


def _spawn(cmd: list[str], log_path: str) -> subprocess.Popen:
    log = open(log_path, 'w')
    # own session: killpg reaches the `ros2 run` wrapper AND the python node under it
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def _stop(p: subprocess.Popen, sig=signal.SIGINT, wait_s: float = 10.0) -> None:
    if p.poll() is not None:
        return
    os.killpg(p.pid, sig)
    try:
        p.wait(wait_s)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()


def run_trial(args, arm: str, trial: int, log_dir: str) -> dict:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState

    from evh_reactive.reactive_node import WAYPOINT_QOS

    seed = args.seed + trial
    tag = f'{arm}_{trial}'
    plant = _spawn(['ros2', 'run', 'evh_plant', 'plant_node', '--ros-args',
                    '-p', 'absolute_actions:=false', '-p', f'env_name:={args.env}',
                    '-p', 'camera:=agentview', '-p', f'seed:={seed}',
                    '-p', f'max_episode_s:={float(args.warmup + args.window + 30)}',
                    '-p', f'action_watchdog:={"true" if arm == "watchdog" else "false"}'],
                   os.path.join(log_dir, f'plant_{tag}.log'))
    reactive = _spawn(['ros2', 'run', 'evh_reactive', 'reactive_node', '--ros-args',
                       '-p', 'absolute_waypoints:=false'],
                      os.path.join(log_dir, f'reactive_{tag}.log'))
    node = rclpy.create_node(f'kill_drift_{os.getpid()}_{tag}')
    ts, xyz = [], []

    def on_pose(m):
        p = m.pose.position
        ts.append(time.monotonic())
        xyz.append((p.x, p.y, p.z))

    node.create_subscription(PoseStamped, '/obs/ee_pose', on_pose, 10)
    pub = node.create_publisher(JointState, '/cmd/waypoint', WAYPOINT_QOS)
    wp = JointState()
    wp.position = [0.0, 0.0, args.delta, 0.0, 0.0, 0.0, -1.0]   # up: clear of the table
    try:
        deadline = time.monotonic() + args.boot_timeout
        while not ts:
            if time.monotonic() > deadline or plant.poll() is not None:
                raise RuntimeError(f'plant never published /obs/ee_pose ({tag}); see logs')
            rclpy.spin_once(node, timeout_sec=0.05)
        period, t_kill, t_end = 1.0 / args.control_hz, None, None
        t_start, t_next, home = None, 0.0, np.array(xyz[0])
        while True:
            now = time.monotonic()
            if now >= t_next:      # keep streaming after the kill, as a live controller would
                pub.publish(wp)
                t_next = now + period
            # warmup counts from motion onset: the reactive node takes seconds to boot and match,
            # and a kill before the arm moves measures nothing
            if t_start is None:
                if np.linalg.norm(np.array(xyz[-1]) - home) > 2e-3:
                    t_start = now
                elif now > deadline + args.boot_timeout:
                    raise RuntimeError(f'arm never moved ({tag}); is the reactive node up?')
            elif t_kill is None and now - t_start >= args.warmup:
                os.killpg(reactive.pid, signal.SIGKILL)
                t_kill, t_end = time.monotonic(), time.monotonic() + args.window + 0.1
            if t_end is not None and now >= t_end:
                break
            rclpy.spin_once(node, timeout_sec=0.002)
        reactive.wait(5)   # raises TimeoutExpired if it survived the SIGKILL
        row = drift_after_kill(np.array(ts), np.array(xyz), t_kill, args.window)
    finally:
        node.destroy_node()
        _stop(reactive, signal.SIGKILL)
        _stop(plant)
    return {'arm': arm, 'trial': trial, 'seed': seed, **row}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--kills', type=int, default=10, help='trials per arm')
    p.add_argument('--window', type=float, default=4.0, help='s of drift measured after the kill')
    p.add_argument('--warmup', type=float, default=1.0, help='s of motion before the kill')
    p.add_argument('--delta', type=float, default=0.2, help='waypoint z delta (unit action)')
    p.add_argument('--control_hz', type=float, default=20.0)
    p.add_argument('--env', default='Lift')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--boot_timeout', type=float, default=90.0)
    p.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'outputs', 'kill_drift', 'drift.csv'))
    args = p.parse_args(argv)

    import rclpy
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import run_manifest

    log_dir = os.path.join(os.path.dirname(args.out) or '.', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    run_manifest.record(args.out, {'kind': 'kill_drift', 'config': vars(args)})
    rclpy.init()
    rows = []
    try:
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for trial in range(args.kills):
                for arm in ('watchdog', 'no_watchdog'):     # paired: same seed, both arms
                    row = run_trial(args, arm, trial, log_dir)
                    rows.append(row)
                    w.writerow(row)
                    f.flush()
                    print(f"{arm:12s} trial {trial}: drift {row['drift_mm']:6.1f} mm, "
                          f"speed at kill {row['speed_at_kill_mm_s']:6.1f} mm/s", flush=True)
    finally:
        rclpy.shutdown()
    for arm, s in summarize(rows).items():
        print(f"{arm:12s} n={s['n']}: drift median {s['drift_median']:.1f} mm "
              f"[{s['drift_min']:.1f}, {s['drift_max']:.1f}], speed at kill median "
              f"{s['speed_median']:.1f} mm/s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
