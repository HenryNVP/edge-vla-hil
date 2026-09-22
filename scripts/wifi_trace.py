#!/usr/bin/env python3
"""Measure a real link the way the testbed uses it, and fit the relay's channel model to it.

Experiment W of the short-term plan: before sweeping bursty loss, find out what a real WiFi link
between the robot side and the policy server actually does to THIS traffic, and turn that into
relay parameters (loss rate, Gilbert–Elliott burst length, delay quantiles). Generic ping/iperf
numbers do not answer it: what breaks here is 21 KB camera frames fragmenting into ~16 packets on
a best-effort transport, next to small proprio and waypoint messages on the same air.

Two modes.

  record   run on BOTH machines at once, same ROS_DOMAIN_ID, one `--role robot` and one
           `--role server`. Each side publishes its real traffic toward the other and logs every
           message it receives (stream, seq, send stamp, receive time) to a CSV:

             robot -> server   /trace/image, /trace/wrist  84x84x3 rgb8 at 20 Hz
                               /trace/proprio              9 floats at 20 Hz
             server -> robot   /trace/waypoint             7 floats at 20 Hz
                               /trace/chunk                16x7 floats at 5 Hz (robot-side execution)

           plus ping-pong probes (robot -> server -> robot, 10 Hz) for the clock offset, so
           one-way delay does not depend on chrony being right. All best-effort, depth 1, like
           the real topics.

             python3 scripts/wifi_trace.py record --role server --seconds 600 --out srv.csv
             python3 scripts/wifi_trace.py record --role robot  --seconds 600 --out rob.csv

  analyze  offline, no ROS: read both CSVs, estimate the clock offset from the probes (NTP's
           minimum-delay filter), and report per stream: one-way delay p50/p95/p99/max, loss
           rate, loss-burst length distribution, and the Gilbert–Elliott fit (loss, burst_ms)
           that evh_latency's `loss_model:=gilbert` takes.

             python3 scripts/wifi_trace.py analyze --robot rob.csv --server srv.csv \\
                 --label near_los --json near_los.json

A message counts as lost when its sequence number never arrives; a burst is a run of consecutive
lost sequence numbers, converted to milliseconds by the stream's period. The logic below is
importable without ROS (rclpy is imported only inside `record`), so the fast suite tests it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass

STREAMS = {
    # name: (direction, rate_hz, payload shape) — the testbed's real traffic
    'image': ('up', 20.0, (84, 84, 3)),
    'wrist': ('up', 20.0, (84, 84, 3)),
    'proprio': ('up', 20.0, (9,)),
    'waypoint': ('down', 20.0, (7,)),
    'chunk': ('down', 5.0, (16, 7)),
}
PROBE_HZ = 10.0


# --------------------------------------------------------------------- analysis
@dataclass
class Offset:
    """server_clock - robot_clock, and the round trip it was measured at."""
    offset_s: float
    rtt_s: float
    samples: int


def estimate_offset(probes: list[tuple[float, float, float, float]],
                    best_fraction: float = 0.1) -> Offset:
    """NTP-style offset from ping-pong probes (t1 robot send, t2 server recv, t3 server send,
    t4 robot recv). Uses the probes with the smallest round trip: queueing delay is one-sided,
    so the fastest exchanges are the ones where the two directions were most nearly symmetric."""
    if not probes:
        raise ValueError('no probes: cannot estimate the clock offset')
    rows = sorted(((t4 - t1) - (t3 - t2), ((t2 - t1) + (t3 - t4)) / 2.0)
                  for t1, t2, t3, t4 in probes)
    best = rows[:max(1, int(len(rows) * best_fraction))]
    return Offset(offset_s=statistics.median(o for _, o in best),
                  rtt_s=statistics.median(r for r, _ in best), samples=len(probes))


def loss_runs(received_seqs: list[int], first: int, last: int) -> tuple[int, list[int]]:
    """(messages sent, lengths of each run of consecutive missing sequence numbers)."""
    got = set(received_seqs)
    runs, n = [], 0
    for seq in range(first, last + 1):
        if seq in got:
            if n:
                runs.append(n)
                n = 0
        else:
            n += 1
    if n:
        runs.append(n)
    return last - first + 1, runs


def fit_gilbert(sent: int, runs: list[int], period_s: float) -> dict:
    """The relay's Gilbert–Elliott parameters from observed loss runs.

    loss = lost / sent; burst_ms = mean run length x the stream's period. Matches evh_latency's
    model, whose BAD periods are exponential with mean burst_ms and whose long-run BAD fraction
    is `loss`. With no loss at all there is nothing to fit, and it says so rather than inventing
    a burst length.
    """
    lost = sum(runs)
    if sent <= 0:
        raise ValueError('nothing was sent')
    if lost == 0:
        return {'loss': 0.0, 'burst_ms': None, 'bursts': 0}
    return {'loss': lost / sent, 'burst_ms': 1e3 * period_s * lost / len(runs),
            'bursts': len(runs)}


def quantiles_ms(delays_s: list[float]) -> dict:
    if not delays_s:
        return {'p50': None, 'p95': None, 'p99': None, 'max': None}
    d = sorted(delays_s)

    def q(p):
        return 1e3 * d[min(len(d) - 1, int(math.ceil(p * len(d))) - 1)]
    return {'p50': q(0.50), 'p95': q(0.95), 'p99': q(0.99), 'max': 1e3 * d[-1]}


def analyze(robot_rows: list[dict], server_rows: list[dict], label: str = '',
            trim_s: float = 2.0) -> dict:
    """Per-stream delay and loss from both sides' receive logs.

    The first and last `trim_s` of each stream are ignored: the two recorders start and stop a
    moment apart, and a message sent while the other side was not yet (or no longer) listening
    is a bookkeeping artefact, not loss on the link.
    """
    probes = [(float(r['t1']), float(r['t2']), float(r['t3']), float(r['t4']))
              for r in robot_rows if r['stream'] == 'probe']
    off = estimate_offset(probes)
    out = {'label': label, 'offset_ms': 1e3 * off.offset_s, 'min_rtt_ms': 1e3 * off.rtt_s,
           'probes': off.samples, 'streams': {}}

    for name, (direction, rate, _shape) in STREAMS.items():
        # up-streams are received (and logged) by the server, down-streams by the robot
        rows = [r for r in (server_rows if direction == 'up' else robot_rows)
                if r['stream'] == name]
        sent_rows = [r for r in (robot_rows if direction == 'up' else server_rows)
                     if r['stream'] == f'sent:{name}']
        if not sent_rows:
            continue
        t0 = min(float(r['sent']) for r in sent_rows) + trim_s
        t1 = max(float(r['sent']) for r in sent_rows) - trim_s
        window = [int(r['seq']) for r in sent_rows if t0 <= float(r['sent']) <= t1]
        if not window:
            continue
        first, last = min(window), max(window)
        rows = [r for r in rows if first <= int(r['seq']) <= last]
        # receive time is on the receiver's clock; convert to the sender's before differencing
        to_sender = off.offset_s if direction == 'up' else -off.offset_s
        delays = [float(r['recv']) - to_sender - float(r['sent']) for r in rows]
        sent, runs = loss_runs([int(r['seq']) for r in rows], first, last)
        out['streams'][name] = {
            'direction': direction, 'rate_hz': rate, 'sent': sent, 'received': len(rows),
            'delay_ms': quantiles_ms(delays),
            'burst_lengths': {str(k): runs.count(k) for k in sorted(set(runs))},
            'gilbert': fit_gilbert(sent, runs, 1.0 / rate),
        }
    return out


def read_csv(path: str) -> list[dict]:
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


# ----------------------------------------------------------------------- record
FIELDS = ['stream', 'seq', 'sent', 'recv', 't1', 't2', 't3', 't4']


def record(role: str, seconds: float, out: str) -> None:
    """Publish this side's streams, log everything received. Needs a sourced ROS 2."""
    import numpy as np
    import rclpy
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image, JointState

    qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                     history=QoSHistoryPolicy.KEEP_LAST)
    rclpy.init()
    node = rclpy.create_node(f'wifi_trace_{role}')
    fh = open(out, 'w', newline='')
    log = csv.DictWriter(fh, FIELDS)
    log.writeheader()
    mine = 'up' if role == 'robot' else 'down'
    seqs = dict.fromkeys(STREAMS, 0)

    def now():
        return time.time()

    def stamp(msg, t):
        msg.header.stamp.sec, msg.header.stamp.nanosec = int(t), int((t % 1) * 1e9)

    def on_msg(name):
        def cb(msg):
            sent = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            log.writerow({'stream': name, 'seq': int(msg.header.frame_id), 'sent': sent,
                          'recv': now()})
        return cb

    pubs = {}
    for name, (direction, rate, shape) in STREAMS.items():
        cls = Image if len(shape) == 3 else JointState
        if direction == mine:
            pubs[name] = (node.create_publisher(cls, f'/trace/{name}', qos), cls, shape)

            def send(name=name):
                pub, cls, shape = pubs[name]
                msg = cls()
                if cls is Image:
                    msg.height, msg.width, msg.encoding = shape[0], shape[1], 'rgb8'
                    msg.step = shape[1] * 3
                    msg.data = np.random.randint(0, 255, int(np.prod(shape)), np.uint8).tobytes()
                else:
                    msg.position = np.random.rand(int(np.prod(shape))).tolist()
                msg.header.frame_id = str(seqs[name])
                t = now()
                stamp(msg, t)
                pub.publish(msg)
                log.writerow({'stream': f'sent:{name}', 'seq': seqs[name], 'sent': t})
                seqs[name] += 1
            node.create_timer(1.0 / rate, send)
        else:
            node.create_subscription(cls, f'/trace/{name}', on_msg(name), qos)

    # ping-pong probes for the clock offset: robot t1 -> server (t2, t3) -> robot t4
    if role == 'robot':
        probe_pub = node.create_publisher(JointState, '/trace/probe', qos)

        def on_pong(msg):
            t1, t2, t3 = msg.position[:3]
            log.writerow({'stream': 'probe', 't1': t1, 't2': t2, 't3': t3, 't4': now()})
        node.create_subscription(JointState, '/trace/pong', on_pong, qos)
        node.create_timer(1.0 / PROBE_HZ, lambda: probe_pub.publish(JointState(position=[now()])))
    else:
        pong_pub = node.create_publisher(JointState, '/trace/pong', qos)

        def on_probe(msg):
            t2 = now()
            pong_pub.publish(JointState(position=[msg.position[0], t2, now()]))
        node.create_subscription(JointState, '/trace/probe', on_probe, qos)

    end = time.time() + seconds
    try:
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.01)
    finally:
        fh.close()
        node.destroy_node()
        rclpy.shutdown()
    print(f'[wifi_trace] {role}: wrote {out}')


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='mode', required=True)
    r = sub.add_parser('record')
    r.add_argument('--role', choices=['robot', 'server'], required=True)
    r.add_argument('--seconds', type=float, default=600.0)
    r.add_argument('--out', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('--robot', required=True)
    a.add_argument('--server', required=True)
    a.add_argument('--label', default='')
    a.add_argument('--json', default='')
    args = p.parse_args(argv)

    if args.mode == 'record':
        record(args.role, args.seconds, args.out)
        return
    result = analyze(read_csv(args.robot), read_csv(args.server), args.label)
    print(f"[{result['label']}] clock offset {result['offset_ms']:+.2f} ms, "
          f"min RTT {result['min_rtt_ms']:.2f} ms over {result['probes']} probes")
    for name, s in result['streams'].items():
        d, g = s['delay_ms'], s['gilbert']
        burst = f"{g['burst_ms']:.0f} ms" if g['burst_ms'] is not None else '-'
        p95 = f"{d['p95']:.1f}" if d['p95'] is not None else '-'
        p50 = f"{d['p50']:.1f}" if d['p50'] is not None else '-'
        print(f"  {name:9s} {s['direction']:4s} delay p50 {p50} p95 {p95} ms  "
              f"loss {100 * g['loss']:.2f}%  mean burst {burst}")
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(result, fh, indent=2)


if __name__ == '__main__':
    main()
