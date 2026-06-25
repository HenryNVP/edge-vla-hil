"""Benchmark recorder + sweep driver — produces the headline plot.

Two modes:

  record  (default, run as a ROS node inside an already-launched graph):
      subscribes to /eval/success, /metrics/inference_ms, and /cmd/action, then writes one CSV row
      summarizing a fixed time window: success rate, mean/p95 inference latency, achieved control
      loop Hz. Tag the row with the current condition via --label.

  sweep   (orchestrator, launches the stack once per condition via `ros2 launch`):
      for each latency value, (re)launch hil.launch.py with/without the reactive layer, run the
      recorder for --duration seconds, tear down, append to CSV. This yields the
      success-rate-vs-latency curves (reactive ON vs OFF) that are the project's main result.

CSV columns: condition,latency_ms,jitter_ms,reactive,trials,success_rate,infer_ms_mean,
             infer_ms_p95,loop_hz
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import JointState


# --------------------------------------------------------------------- record
class Recorder(Node):
    def __init__(self) -> None:
        super().__init__('evh_benchmark_recorder')
        self._successes = 0
        self._trials = 0
        self._infer_ms: list[float] = []
        self._action_stamps: list[float] = []

        self.create_subscription(Bool, '/eval/success', self._on_success, 10)
        self.create_subscription(Float32, '/metrics/inference_ms', self._on_infer, 10)
        self.create_subscription(JointState, '/cmd/action', self._on_action, 10)

    def _on_success(self, msg: Bool) -> None:
        self._trials += 1
        self._successes += int(msg.data)

    def _on_infer(self, msg: Float32) -> None:
        self._infer_ms.append(float(msg.data))

    def _on_action(self, _msg: JointState) -> None:
        self._action_stamps.append(time.perf_counter())

    def summary(self) -> dict:
        infer = np.asarray(self._infer_ms) if self._infer_ms else np.array([np.nan])
        # loop Hz from inter-arrival of /cmd/action
        if len(self._action_stamps) > 1:
            dt = np.diff(self._action_stamps)
            loop_hz = 1.0 / float(np.mean(dt))
        else:
            loop_hz = float('nan')
        return {
            'trials': self._trials,
            'success_rate': (self._successes / self._trials) if self._trials else float('nan'),
            'infer_ms_mean': float(np.nanmean(infer)),
            'infer_ms_p95': float(np.nanpercentile(infer, 95)),
            'loop_hz': loop_hz,
        }


def run_record(args) -> dict:
    rclpy.init()
    node = Recorder()
    t_end = time.time() + args.duration
    try:
        while time.time() < t_end and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        s = node.summary()
        node.destroy_node()
        rclpy.shutdown()
    _append_csv(args.out, args, s)
    print(f'[record] label={args.label} {s}')
    return s


def _append_csv(path: str, args, s: dict) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['condition', 'strategy', 'latency_ms', 'jitter_ms', 'reactive', 'trials',
                        'success_rate', 'infer_ms_mean', 'infer_ms_p95', 'loop_hz'])
        w.writerow([args.label, getattr(args, 'strategy', ''), args.latency_ms, args.jitter_ms,
                    args.reactive, s['trials'], s['success_rate'], s['infer_ms_mean'],
                    s['infer_ms_p95'], s['loop_hz']])


# ---------------------------------------------------------------------- sweep
def run_sweep(args) -> None:
    """Wedge-A sweep: launch the stack once per (strategy x latency x reactive) cell and record.

    Yields the headline curves — success rate vs injected latency/jitter for each chunk-execution
    strategy, with and without the reactive layer.
    """
    values = [float(v) for v in args.values.split(',')]
    strategies = [s.strip() for s in args.strategies.split(',')]
    for strategy in strategies:
        for reactive in (True, False):
            for lat in values:
                passthrough = 'false' if reactive else 'true'
                label = f'strat={strategy}_reactive={reactive}_lat={lat}'
                print(f'[sweep] launching {label} ...')
                proc = subprocess.Popen(
                    ['ros2', 'launch', 'evh_bringup', 'hil.launch.py',
                     f'latency_ms:={lat}', f'jitter_ms:={args.jitter_ms}',
                     f'backend:={args.backend}', f'weights:={args.weights}',
                     f'strategy:={strategy}', f'passthrough:={passthrough}'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid)
                try:
                    time.sleep(args.warmup)   # let the graph come up
                    rec_args = argparse.Namespace(
                        out=args.out, duration=args.duration, label=label,
                        latency_ms=lat, jitter_ms=args.jitter_ms, reactive=reactive,
                        strategy=strategy)
                    run_record(rec_args)
                finally:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    proc.wait(timeout=15)
                time.sleep(2.0)
    print(f'[sweep] done -> {args.out}')


# ----------------------------------------------------------------------- main
def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description='EdgeVLA-HiL benchmark')
    p.add_argument('--sweep', choices=['latency'], help='run the orchestrated sweep')
    p.add_argument('--values', default='0,25,50,100,200', help='comma-separated latency_ms values')
    p.add_argument('--strategies', default='synchronous,temporal_ensemble,rtc',
                   help='comma-separated chunk-execution strategies to sweep (Wedge A)')
    p.add_argument('--duration', type=float, default=60.0, help='record window seconds')
    p.add_argument('--warmup', type=float, default=8.0, help='seconds to wait after launch')
    p.add_argument('--out', default='results/sweep.csv')
    p.add_argument('--backend', default='pytorch')
    p.add_argument('--weights', default='')
    p.add_argument('--jitter_ms', type=float, default=0.0)
    # record-mode-only tags
    p.add_argument('--label', default='manual')
    p.add_argument('--latency_ms', type=float, default=0.0)
    p.add_argument('--reactive', type=lambda s: s.lower() == 'true', default=True)
    p.add_argument('--strategy', default='', help='strategy tag for a manual record row')
    args = p.parse_args(argv)

    if args.sweep:
        run_sweep(args)
    else:
        run_record(args)


if __name__ == '__main__':
    main()
