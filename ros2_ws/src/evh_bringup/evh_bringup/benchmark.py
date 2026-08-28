"""Benchmark recorder + sweep driver — produces the headline plot.

Two modes:

  record  (default, run as a ROS node inside an already-launched graph):
      subscribes to /eval/success, /metrics/inference_ms, /cmd/waypoint and /cmd/action, then
      writes one CSV row summarizing a fixed time window: success rate, mean/p95 inference latency,
      cognitive command throughput (waypoint_hz) and the reactive layer's output rate (loop_hz).
      Tag the row with the current condition via --label.

      waypoint_hz is the metric the chunk-execution story turns on: the rate of NEW cognitive
      commands on /cmd/waypoint (the controller publishes only when its executor returns an
      action — a hold emits nothing), which collapses as latency grows under the synchronous
      strategy. loop_hz (the reactive layer's /cmd/action rate) is ~constant by design and is
      only a sanity check that the high-rate tracker is keeping up. Both are counts over the
      record window, so they read 0 rather than NaN when a condition starves.

  sweep   (orchestrator, launches the stack once per condition via `ros2 launch`):
      for each latency value, (re)launch hil.launch.py with/without the reactive layer, run the
      recorder for --duration seconds, tear down, append to CSV. This yields the
      success-rate-vs-latency curves (reactive ON vs OFF) that are the project's main result.
      --absolute/--backend must agree with the checkpoint (see resolve_absolute); the plant
      enforces this at runtime and aborts a mismatched cell rather than recording garbage.

CSV columns: condition,strategy,latency_ms,jitter_ms,reactive,trials,success_rate,infer_ms_mean,
             infer_ms_p95,waypoint_hz,loop_hz
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
        self._waypoint_stamps: list[float] = []

        self.create_subscription(Bool, '/eval/success', self._on_success, 10)
        self.create_subscription(Float32, '/metrics/inference_ms', self._on_infer, 10)
        self.create_subscription(JointState, '/cmd/waypoint', self._on_waypoint, 10)
        self.create_subscription(JointState, '/cmd/action', self._on_action, 10)

    def _on_success(self, msg: Bool) -> None:
        self._trials += 1
        self._successes += int(msg.data)

    def _on_infer(self, msg: Float32) -> None:
        self._infer_ms.append(float(msg.data))

    def _on_waypoint(self, _msg: JointState) -> None:
        self._waypoint_stamps.append(time.perf_counter())

    def _on_action(self, _msg: JointState) -> None:
        self._action_stamps.append(time.perf_counter())

    @staticmethod
    def _rate(stamps: list[float], window_s: float) -> float:
        """Throughput (Hz) = messages received / length of the record window.

        Deliberately count-over-window rather than 1/mean(inter-arrival): waypoint_hz has to stay
        well-defined when a condition is degraded enough that zero or one new commands arrive in
        the whole window — that is the interesting end of the curve, and a NaN there silently
        drops the point from the plot. The cost is that a graph still coming up during the window
        understates the rate; that is what --warmup is for.
        """
        if window_s <= 0.0:
            return float('nan')
        return len(stamps) / window_s

    def summary(self, window_s: float) -> dict:
        # NaN (not 0) with no samples: unlike a throughput, an unobserved latency has no value —
        # short-circuit rather than let numpy warn its way to the same answer
        infer = np.asarray(self._infer_ms) if self._infer_ms else None
        return {
            'trials': self._trials,
            'success_rate': (self._successes / self._trials) if self._trials else float('nan'),
            'infer_ms_mean': float(np.mean(infer)) if infer is not None else float('nan'),
            'infer_ms_p95': (float(np.percentile(infer, 95)) if infer is not None
                             else float('nan')),
            # cognitive command throughput (collapses with latency); the headline chunking metric
            'waypoint_hz': self._rate(self._waypoint_stamps, window_s),
            # reactive layer output rate — ~constant by design, sanity check only
            'loop_hz': self._rate(self._action_stamps, window_s),
        }


def run_record(args) -> dict:
    rclpy.init()
    node = Recorder()
    t0 = time.perf_counter()
    try:
        t_end = time.time() + args.duration
        while time.time() < t_end and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        # inside the finally: a window cut short (crashed graph, Ctrl-C) still gets its row, so a
        # long sweep never loses a cell's data to an exception on the way out
        s = node.summary(time.perf_counter() - t0)
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
                        'success_rate', 'infer_ms_mean', 'infer_ms_p95', 'waypoint_hz', 'loop_hz'])
        w.writerow([args.label, args.strategy, args.latency_ms, args.jitter_ms,
                    args.reactive, s['trials'], s['success_rate'], s['infer_ms_mean'],
                    s['infer_ms_p95'], s['waypoint_hz'], s['loop_hz']])


# ---------------------------------------------------------------------- sweep
def resolve_absolute(choice: str, backend: str) -> str:
    """Pick the launch's `absolute` value, as the lowercase string ros2 launch wants.

    The sweep must pass this EXPLICITLY. hil.launch.py defaults `absolute` to true (it is written
    for the abs-action DP checkpoint), so a sweep that leaves it alone while running the
    `pytorch` backend — whose policies emit deltas — silently produces a full CSV of garbage.
    'auto' derives it from the backend; the plant cross-checks the result against the loaded
    checkpoint and aborts on a mismatch, so a wrong guess here is loud rather than silent.
    """
    if choice != 'auto':
        return choice
    return 'true' if backend == 'dp' else 'false'


def run_sweep(args) -> None:
    """Wedge-A sweep: launch the stack once per (strategy x latency x reactive) cell and record.

    Yields the headline curves — success rate vs injected latency/jitter for each chunk-execution
    strategy, with and without the reactive layer.
    """
    values = [float(v) for v in args.values.split(',')]
    strategies = [s.strip() for s in args.strategies.split(',')]
    absolute = resolve_absolute(args.absolute, args.backend)
    print(f'[sweep] backend={args.backend} absolute={absolute} '
          f'({len(strategies)} strategies x 2 reactive x {len(values)} latencies)')
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
                     f'strategy:={strategy}', f'passthrough:={passthrough}',
                     f'absolute:={absolute}'],
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
    p.add_argument('--absolute', choices=['auto', 'true', 'false'], default='auto',
                   help="launch `absolute` arg; 'auto' = true for backend=dp, false otherwise")
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
