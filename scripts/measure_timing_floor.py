#!/usr/bin/env python3
"""Measure the testbed's timing noise floor, so the injected-latency axis can be trusted.

The headline result is success-rate vs *injected* latency (0-200ms). That axis is only meaningful
if the testbed's own jitter is small compared to the smallest condition being distinguished --
and until this script existed, that had never been measured. Nothing here changes the system; it
subscribes and reports.

Run it on either machine while a run is live, on the same domain and CYCLONEDDS_URI:

    docker run --rm --network host -v ~/edge-vla-hil:/ws \\
      -e ROS_DOMAIN_ID=42 -e CYCLONEDDS_URI=file:///ws/docker/cyclonedds-jetson.xml \\
      --entrypoint bash edge-vla-hil:jetson -lc \\
      'source /ros_source.sh && python3 /ws/scripts/measure_timing_floor.py --seconds 30'

Two clocks, two different questions, and neither needs the machines to be time-synced:

  publisher period -- consecutive `header.stamp` differences. Every stamp on one topic comes from
      one machine's clock, so differencing them is immune to any offset between the two hosts.
      This is the timer's own jitter: how well `create_timer` actually holds its period under
      Python, the GIL, GC and the scheduler. This is the number that says whether the plant's
      simulated time (which is *defined* by whenever `_step_physics` fires -- there is no catch-up
      and no deadline accounting) tracks wall time.

  arrival period -- consecutive local receive times for the same messages, which adds DDS,
      the kernel network stack and, for a remote publisher, the wire.

Relay delay is measured separately and is offset-immune for a different reason: evh_latency
republishes the *same* message, stamp untouched, so pairing a raw message with its /delayed twin
by stamp and differencing their arrival times at a single observer gives the relay's own cost at
latency_ms=0 -- the floor under every injected value the sweep commands.

Caveat worth keeping in mind when reading the output: an observer on the far side of the link sees
arrival jitter that includes the network, while publisher-period jitter is measured at the source
either way. Run it on both machines if you want to separate the two.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy)
from sensor_msgs.msg import JointState

# topic -> nominal Hz. All JointState so one subscriber type covers them; the image topics carry
# the same stamps but at 20Hz would only restate what /obs/proprio already says.
TOPICS = {
    '/obs/proprio': 20.0,           # plant obs timer      (desktop)
    '/obs/proprio/delayed': 20.0,   # after the relay      (desktop)
    '/cmd/action': 250.0,           # reactive tracking    (desktop)
    '/cmd/waypoint': 20.0,          # controller chunk out (jetson)
}
RELAY_PAIRS = [('/obs/proprio', '/obs/proprio/delayed')]

# Permissive: BEST_EFFORT/VOLATILE subscribers match RELIABLE/TRANSIENT_LOCAL publishers too.
_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                  durability=DurabilityPolicy.VOLATILE,
                  history=HistoryPolicy.KEEP_LAST, depth=500)


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float('nan')
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


class Sampler(Node):
    def __init__(self) -> None:
        super().__init__('timing_floor')
        self.samples: dict[str, list[tuple[int, int]]] = defaultdict(list)  # (stamp_ns, recv_ns)
        for topic in TOPICS:
            self.create_subscription(
                JointState, topic, lambda m, t=topic: self._on(t, m), _QOS)

    def _on(self, topic: str, msg: JointState) -> None:
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.samples[topic].append((stamp_ns, time.monotonic_ns()))


def _report(topic: str, nominal_hz: float, rows: list[tuple[int, int]]) -> dict:
    nominal_ms = 1000.0 / nominal_hz
    stamps = [s for s, _ in rows]
    recvs = [r for _, r in rows]
    pub_dt = sorted((b - a) / 1e6 for a, b in zip(stamps, stamps[1:]) if b > a)
    arr_dt = sorted((b - a) / 1e6 for a, b in zip(recvs, recvs[1:]))
    # Two different failures live in this one series and must not be averaged together:
    #   a *skip* -- an interval of ~2x nominal or more, i.e. a whole period with no message. From
    #       stamps alone a skipped timer tick and a message lost on the wire look identical (both
    #       leave a hole in the stamp sequence), so this is reported as a rate, not a diagnosis;
    #       run the script on the publisher's own machine to rule the network out.
    #   jitter -- how much the periods that *did* fire deviate from nominal. Letting skips into
    #       this percentile makes a healthy timer look terrible and hides the skip rate entirely.
    kept = [d for d in pub_dt if d <= 1.5 * nominal_ms]
    skips = len(pub_dt) - len(kept)
    out = {
        'topic': topic, 'nominal_hz': nominal_hz, 'nominal_ms': nominal_ms,
        'n': len(rows), 'skips': skips,
        'skip_rate': skips / len(pub_dt) if pub_dt else float('nan'),
        'pub_p50_ms': _pct(kept, 0.50), 'pub_p99_ms': _pct(kept, 0.99),
        'pub_max_ms': kept[-1] if kept else float('nan'),
        'pub_max_incl_skips_ms': pub_dt[-1] if pub_dt else float('nan'),
        'arr_p99_ms': _pct(sorted(d for d in arr_dt if d <= 1.5 * nominal_ms), 0.99),
        'arr_max_ms': arr_dt[-1] if arr_dt else float('nan'),
    }
    out['pub_p99_jitter_ms'] = out['pub_p99_ms'] - nominal_ms
    return out


def _relay_delay(raw: list[tuple[int, int]], delayed: list[tuple[int, int]]) -> dict | None:
    by_stamp = {s: r for s, r in raw}
    deltas = sorted((r - by_stamp[s]) / 1e6 for s, r in delayed if s in by_stamp)
    if not deltas:
        return None
    return {'n': len(deltas), 'p50_ms': _pct(deltas, 0.50), 'p99_ms': _pct(deltas, 0.99),
            'max_ms': deltas[-1]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seconds', type=float, default=30.0, help='sampling window (default: 30)')
    ap.add_argument('--json', type=str, default='', help='also write the numbers to this path')
    args = ap.parse_args()

    rclpy.init()
    node = Sampler()
    deadline = time.monotonic() + args.seconds
    print(f'sampling {args.seconds:.0f}s...', flush=True)
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    results = [_report(t, hz, node.samples[t]) for t, hz in TOPICS.items()
               if len(node.samples[t]) > 2]
    silent = [t for t in TOPICS if len(node.samples[t]) <= 2]

    print(f'\n{"topic":<24}{"nominal":>9}{"p50":>8}{"p99":>8}{"jitter p99":>12}'
          f'{"max":>8}{"arr p99":>9}{"skips":>8}{"n":>7}')
    for r in results:
        print(f'{r["topic"]:<24}{r["nominal_ms"]:>8.1f}m{r["pub_p50_ms"]:>8.2f}'
              f'{r["pub_p99_ms"]:>8.2f}{r["pub_p99_jitter_ms"]:>+12.2f}{r["pub_max_ms"]:>8.2f}'
              f'{r["arr_p99_ms"]:>9.2f}{r["skip_rate"]:>7.2%}{r["n"]:>7}')
    print('  ms; period between consecutive header stamps (the publisher\'s own timer).')
    print('  jitter p99 and max EXCLUDE skipped periods; skips are the rightmost column.')
    print('  arr p99 = the same period measured on arrival here, so it adds DDS and the wire.')

    relays = {}
    for raw, delayed in RELAY_PAIRS:
        d = _relay_delay(node.samples[raw], node.samples[delayed])
        if d:
            relays[f'{raw} -> {delayed}'] = d
            print(f'\nrelay {raw} -> {delayed} at latency_ms=0: '
                  f'p50 {d["p50_ms"]:.2f}ms  p99 {d["p99_ms"]:.2f}ms  max {d["max_ms"]:.2f}ms '
                  f'(n={d["n"]})')

    if silent:
        print(f'\nno messages on: {", ".join(silent)} — is the whole graph up?')

    worst = max(results, key=lambda r: r['pub_p99_jitter_ms'])
    print(f'\nfloor: worst timer jitter is {worst["pub_p99_jitter_ms"]:+.2f}ms at p99 on '
          f'{worst["topic"]} — compare against the smallest injected-latency step the sweep must '
          f'resolve. Skipped periods are a separate question: a nonzero skip rate on a topic whose '
          f'publisher is doing per-tick work (inference, physics) is a missed deadline, not noise.')

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(
            {'window_s': args.seconds, 'topics': results, 'relay': relays}, indent=2))
        print(f'wrote {args.json}')

    node.destroy_node()
    rclpy.shutdown()
    return 0 if results else 1


if __name__ == '__main__':
    sys.exit(main())
