#!/usr/bin/env python3
"""Assert that the cross-machine HiL graph actually formed, and say what is missing if not.

Run this INSIDE a container on either machine while the other side is up, on the same
ROS_DOMAIN_ID and with the same CYCLONEDDS_URI the real run uses:

    docker run --rm --network host -v ~/edge-vla-hil:/ws \\
      -e ROS_DOMAIN_ID=42 -e CYCLONEDDS_URI=file:///ws/docker/cyclonedds-jetson.xml \\
      --entrypoint bash edge-vla-hil:jetson -lc \\
      'source /ros_source.sh && python3 /ws/scripts/check_hil_link.py'

Why this exists: every failure mode on this link is silent (see docker/cyclonedds-jetson.xml). A
half-formed graph looks identical to a healthy one from either terminal -- nodes start, no error
is printed, and the run produces a CSV of plausible numbers with no policy in the loop. Reading
`ros2 node list` by eye is how that gets missed, so this states the expectation instead: which
nodes belong to which machine, and which topic must have crossed.

Exit status is 0 only when every expected node is present. A mode-mismatch abort of evh_plant
(the intended outcome of the mismatch smoke test) reports as a distinct, expected failure.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

# Which side publishes what, in the controller-on-Jetson / host.launch.py-on-desktop split.
JETSON_NODES = ['/evh_controller']
DESKTOP_NODES = ['/evh_plant', '/evh_reactive', '/latency_image', '/latency_wrist',
                 '/latency_proprio']


def _ros2(*args: str, timeout: float = 20.0) -> str:
    out = subprocess.run(['ros2', *args], capture_output=True, text=True, timeout=timeout)
    return out.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--settle', type=float, default=10.0,
                    help='seconds to let discovery complete before looking (default: 10)')
    args = ap.parse_args()

    print(f'waiting {args.settle:.0f}s for discovery...', flush=True)
    time.sleep(args.settle)

    nodes = set(_ros2('node', 'list').split())
    topics = set(_ros2('topic', 'list').split())

    missing_local = [n for n in JETSON_NODES if n not in nodes]
    missing_remote = [n for n in DESKTOP_NODES if n not in nodes]

    for label, expected, missing in (('jetson ', JETSON_NODES, missing_local),
                                     ('desktop', DESKTOP_NODES, missing_remote)):
        for n in expected:
            print(f'  [{"ok" if n not in missing else "--"}] {label}  {n}')

    # The mode cross-check rides a latched topic, so it is readable at any time after the
    # controller starts -- and it is the one message that must have crossed the link for the
    # plant's abort-on-mismatch to be reachable at all.
    echoed = _ros2('topic', 'echo', '/policy/absolute', '--once', timeout=15.0)
    mode = next((ln.split(':', 1)[1].strip() for ln in echoed.splitlines()
                 if ln.startswith('data:')), '')
    print(f'  [{"ok" if mode else "--"}] link     /policy/absolute -> '
          f'{"absolute=" + mode if mode else "NOT RECEIVED"}')

    if not missing_local and not missing_remote:
        print('\nPASS: full cross-machine graph')
        return 0
    if missing_remote == ['/evh_plant'] and not missing_local:
        # Everything discovered and then the plant left: that is the mode cross-check firing,
        # which is a working link, not a broken one.
        print('\nEXPECTED FAILURE: the graph formed and evh_plant then exited — the abs/delta '
              'mode cross-check firing. Check the desktop terminal for the evh_plant error, and '
              'match the modes (host.launch.py absolute:=... vs the checkpoint) to go further.')
        return 2
    if len(missing_remote) == len(DESKTOP_NODES):
        print('\nFAIL: no nodes from the other machine — discovery did not cross the link. '
              'Check ROS_DOMAIN_ID and CYCLONEDDS_URI match on both sides, that both ends are on '
              'rmw_cyclonedds_cpp, and see docker/cyclonedds-jetson.xml for the tracing overlay.')
        return 1
    print(f'\nFAIL: missing {", ".join(missing_local + missing_remote)}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
