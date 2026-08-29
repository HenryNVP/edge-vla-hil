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

      Each cell waits for the graph to be READY (first /cmd/waypoint) rather than sleeping a
      fixed --warmup: loading the DP checkpoint takes ~7 s warm and far longer cold, and a cell
      that starts recording early reports a throughput the graph never had. --warmup is the
      timeout on that wait, not a delay. Launch output goes to a per-cell log and the driver
      checks the launch is still alive, so a cell whose nodes died is reported instead of
      silently contributing an empty row. --video_dir records an mp4 per cell.

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
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32


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

    @property
    def trials(self) -> int:
        """Episodes closed so far — the recorder's stopping condition in trial mode."""
        return self._trials

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
    """Record one condition. Stops at --trials episodes, or --duration seconds, whichever first.

    Trial mode is the honest one for comparing cells. Episodes get LONGER as conditions degrade
    (measured: ~7 s at 0 ms, ~15 s at 800 ms), so a fixed-seconds window hands the degraded
    cells — exactly the interesting end of the curve — roughly half the episodes of the easy
    ones, and the success rates being compared then carry very different uncertainties.
    --duration stays on as a wall-clock cap so a wedged cell still terminates; a row that hit the
    cap is flagged `truncated` rather than quietly reported as if it had reached its target.
    """
    rclpy.init()
    node = Recorder()
    target = int(getattr(args, 'trials_target', 0) or 0)
    t0 = time.perf_counter()
    truncated = False
    try:
        t_end = time.time() + args.duration
        while rclpy.ok():
            if time.time() >= t_end:
                truncated = bool(target) and node.trials < target
                break
            if target and node.trials >= target:
                break
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        # inside the finally: a window cut short (crashed graph, Ctrl-C) still gets its row, so a
        # long sweep never loses a cell's data to an exception on the way out
        s = node.summary(time.perf_counter() - t0)
        s['truncated'] = truncated
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
                        'success_rate', 'infer_ms_mean', 'infer_ms_p95', 'waypoint_hz', 'loop_hz',
                        'truncated'])
        w.writerow([args.label, args.strategy, args.latency_ms, args.jitter_ms,
                    args.reactive, s['trials'], s['success_rate'], s['infer_ms_mean'],
                    s['infer_ms_p95'], s['waypoint_hz'], s['loop_hz'],
                    s.get('truncated', False)])


# ---------------------------------------------------------------------- sweep
def _deaths_so_far(log_path: str) -> int:
    """How many nodes have logged "process has died" in this cell's launch log.

    Sampled BEFORE teardown, never after: SIGINT makes most of the graph log the same line, and
    how many do is not stable (the plant exits cleanly through destroy_node and does not).
    Comparing against a magic expected-deaths constant was wrong by one in practice, which would
    have hidden exactly one real mid-run death per cell.
    """
    with open(log_path) as log:
        return sum('process has died' in line for line in log)


def _shutdown(proc: subprocess.Popen) -> None:
    """SIGINT the launch process group, escalating only if it will not go.

    SIGINT and not SIGKILL: the plant finalizes its mp4 and closes the robosuite env in
    destroy_node(). Killing the group (not just the launcher) is what stops a node surviving as
    an orphan and polluting the next cell's ROS domain.
    """
    group = os.getpgid(proc.pid)
    os.killpg(group, signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(group, signal.SIGKILL)
        proc.wait(timeout=10)


def wait_until_ready(timeout_s: float) -> float:
    """Block until the graph produces its first /cmd/waypoint. Returns seconds waited, or -1.

    A true end-to-end readiness signal: a waypoint means the plant is publishing observations,
    the relay is forwarding them, the checkpoint is loaded and the policy has produced a chunk.
    Sleeping a fixed interval instead guesses at all four, and guessing short is not benign — the
    recorder counts messages over the whole window, so a graph that is still coming up drags
    waypoint_hz down and the cell looks degraded by the condition rather than by the clock.
    """
    # owns its own rclpy context: this runs before run_record(), which inits and shuts down its
    # own, and two live contexts in one process would clash
    rclpy.init()
    probe = rclpy.create_node('evh_benchmark_probe')
    seen: list[int] = []
    probe.create_subscription(JointState, '/cmd/waypoint', lambda _m: seen.append(1), 10)
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < timeout_s:
            rclpy.spin_once(probe, timeout_sec=0.1)
            if seen:
                return time.perf_counter() - t0
        return -1.0
    finally:
        probe.destroy_node()
        rclpy.shutdown()


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


_AXIS_PARAM = {'latency': 'latency_ms', 'jitter': 'jitter_ms', 'drop': 'drop_prob'}


def run_sweep(args) -> None:
    """Wedge-A sweep: launch the stack once per (strategy x latency x reactive) cell and record.

    Yields the headline curves — success rate vs injected latency/jitter for each chunk-execution
    strategy, with and without the reactive layer.
    """
    values = [float(v) for v in args.values.split(',')]
    strategies = [s.strip() for s in args.strategies.split(',')]
    axis = args.sweep      # which knob --values walks; the others stay at their flags
    reactive_modes = (True, False) if args.reactive_modes == 'both' else (
        (True,) if args.reactive_modes == 'on' else (False,))
    absolute = resolve_absolute(args.absolute, args.backend)

    log_dir = args.log_dir or os.path.join(os.path.dirname(args.out) or '.', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)

    cells = [(st, rx, lat) for st in strategies for rx in reactive_modes for lat in values]
    print(f'[sweep] axis={axis} backend={args.backend} absolute={absolute} '
          f'jitter_model={args.jitter_model} ({len(cells)} cells)')

    failures = []
    for n, (strategy, reactive, lat) in enumerate(cells, 1):
        label = f'strat={strategy}_reactive={reactive}_{axis}={lat}'
        log_path = os.path.join(log_dir, f'{label}.log'.replace('/', '_'))
        # every degradation knob is passed EXPLICITLY, swept or not. drop_prob used to be
        # omitted entirely, so it silently stayed at the launch default of 0.0 and no sweep
        # could ever reach it — the same class of landmine as the `absolute` default.
        knobs = {'latency_ms': args.latency_ms, 'jitter_ms': args.jitter_ms,
                 'drop_prob': args.drop_prob}
        knobs[_AXIS_PARAM[axis]] = lat
        cmd = ['ros2', 'launch', 'evh_bringup', 'hil.launch.py',
               *(f'{k}:={v}' for k, v in knobs.items()),
               f'jitter_model:={args.jitter_model}',
               f'backend:={args.backend}', f'weights:={args.weights}',
               f'strategy:={strategy}', f'passthrough:={"false" if reactive else "true"}',
               f'absolute:={absolute}']
        if args.video_dir:
            cmd += [f'video:={os.path.join(args.video_dir, label + ".mp4")}',
                    f'video_duration:={args.video_duration}']

        print(f'[sweep] {n}/{len(cells)} {label} ...', flush=True)
        with open(log_path, 'w') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    preexec_fn=os.setsid)
            try:
                waited = wait_until_ready(args.warmup)
                if waited < 0:
                    # never produced a waypoint: checkpoint load failed, a node died, or the
                    # plant aborted on an action-mode mismatch. Recording anyway would write a
                    # zero row indistinguishable from "this condition is simply too degraded".
                    failures.append((label, f'no /cmd/waypoint within {args.warmup:.0f}s'))
                    print(f'[sweep]   SKIPPED — not ready; see {log_path}')
                    continue
                if proc.poll() is not None:
                    failures.append((label, f'launch exited {proc.returncode}'))
                    print(f'[sweep]   SKIPPED — launch died; see {log_path}')
                    continue
                goal = (f'{args.trials} episodes (cap {args.duration:.0f}s)' if args.trials
                        else f'{args.duration:.0f}s')
                print(f'[sweep]   ready in {waited:.1f}s, recording {goal}')
                row = run_record(argparse.Namespace(
                    out=args.out, duration=args.duration, label=label,
                    latency_ms=knobs['latency_ms'], jitter_ms=knobs['jitter_ms'],
                    reactive=reactive, strategy=strategy, trials_target=args.trials))
                if row.get('truncated'):
                    failures.append((label, f"hit the {args.duration:.0f}s cap at "
                                            f"{row['trials']}/{args.trials} episodes"))
                    print(f'[sweep]   WARNING — truncated at {row["trials"]}/{args.trials}')
                # a node that died mid-window still produced a row; say so rather than let it pass
                died = _deaths_so_far(log_path)
                if died:
                    failures.append((label, f'{died} node(s) died mid-run'))
                    print(f'[sweep]   WARNING — nodes died mid-run; see {log_path}')
            finally:
                _shutdown(proc)
        time.sleep(2.0)

    if failures:
        print(f'[sweep] {len(failures)} cell(s) did not produce a trustworthy row:')
        for label, why in failures:
            print(f'[sweep]   {label}: {why}')
    print(f'[sweep] done -> {args.out}')


# ----------------------------------------------------------------------- main
def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description='EdgeVLA-HiL benchmark')
    p.add_argument('--sweep', choices=['latency', 'jitter', 'drop'],
                   help='run the orchestrated sweep, walking --values along this knob')
    p.add_argument('--drop_prob', type=float, default=0.0,
                   help='packet-loss probability held fixed (or swept with --sweep drop)')
    p.add_argument('--jitter_model', choices=['gaussian', 'uniform', 'lognormal'],
                   default='gaussian',
                   help='gaussian/uniform are light-tailed; lognormal supplies the heavy tail a '
                        'quantile delay forecast needs in order to differ from a max')
    p.add_argument('--values', default='0,25,50,100,200', help='comma-separated latency_ms values')
    p.add_argument('--strategies', default='synchronous,temporal_ensemble,rtc',
                   help='comma-separated chunk-execution strategies to sweep (Wedge A)')
    p.add_argument('--duration', type=float, default=60.0,
                   help='record window seconds; with --trials this is the wall-clock cap')
    p.add_argument('--trials', type=int, default=0,
                   help='stop each cell after N episodes instead of a fixed window (recommended: '
                        'degraded cells run longer episodes and a time window under-samples them)')
    p.add_argument('--warmup', type=float, default=90.0,
                   help='TIMEOUT (not a delay) on waiting for the first /cmd/waypoint')
    p.add_argument('--reactive_modes', choices=['both', 'on', 'off'], default='both',
                   help='which reactive-layer settings to sweep')
    p.add_argument('--video_dir', default='', help='record one mp4 per cell into this directory')
    p.add_argument('--video_duration', type=float, default=0.0,
                   help='seconds of video per cell (0 = the whole cell)')
    p.add_argument('--log_dir', default='', help='per-cell launch logs (default: <out dir>/logs)')
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
