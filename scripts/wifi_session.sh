#!/usr/bin/env bash
# One WiFi trace session: run this on BOTH machines at once, one `robot` and one `server`.
#
# It is a wrapper around `scripts/wifi_trace.py record` (see that file for what is measured and
# why). All it adds is the part that is easy to get wrong across two terminals on two machines:
# the two IP env vars the CycloneDDS WiFi config substitutes, which are mirror images of each
# other, and which fail SILENTLY when swapped — discovery never completes, every stream reads as
# 100% loss, and nothing prints an error. So the addresses are checked against the machine's own
# interfaces before the container starts.
#
#   export EVH_SELF_IP=192.168.0.73 EVH_PEER_IP=192.168.0.50   # robot machine
#   scripts/wifi_session.sh robot near_los 600
#
#   export EVH_SELF_IP=192.168.0.50 EVH_PEER_IP=192.168.0.73   # server machine
#   scripts/wifi_session.sh server near_los 600
#
# Start the SERVER side first: the analyzer trims 2 s off each end, so a few seconds of one-sided
# start-up is absorbed, but a minute of it reads as loss.
#
# Afterwards, copy the robot CSV to wherever you analyze and run the command this script prints.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/wifi_session.sh <robot|server> <label> [seconds]

  label     names the condition and the output files (near_los, far_nlos, busy, smoke, ...)
  seconds   default 600; use 30 for a first smoke test

environment:
  EVH_SELF_IP   this machine's WiFi address            (required)
  EVH_PEER_IP   the other machine's WiFi address       (required)
  EVH_DOMAIN    ROS_DOMAIN_ID                          (default 77)
  EVH_IMAGE     docker image                           (default: jetson on arm64, host otherwise)
  EVH_OUT_DIR   output directory on the host           (default <repo>/outputs/wifi)
EOF
  exit 2
}

[ $# -ge 2 ] || usage
ROLE=$1
LABEL=$2
SECONDS_ARG=${3:-600}
case "$ROLE" in robot|server) ;; *) echo "bad role: $ROLE" >&2; usage ;; esac

REPO=$(cd "$(dirname "$0")/.." && pwd)
DOMAIN=${EVH_DOMAIN:-77}
OUT_DIR=${EVH_OUT_DIR:-$REPO/outputs/wifi}
if [ -n "${EVH_IMAGE:-}" ]; then
  IMAGE=$EVH_IMAGE
elif [ "$(uname -m)" = aarch64 ]; then
  IMAGE=edge-vla-hil:jetson
else
  IMAGE=edge-vla-hil:host
fi

# ------------------------------------------------------------------ preflight
die() { echo "[wifi_session] $*" >&2; exit 1; }

[ -n "${EVH_SELF_IP:-}" ] || die 'EVH_SELF_IP is not set (this machine)'
[ -n "${EVH_PEER_IP:-}" ] || die 'EVH_PEER_IP is not set (the other machine)'
[ "$EVH_SELF_IP" != "$EVH_PEER_IP" ] || die 'EVH_SELF_IP and EVH_PEER_IP are the same address'

# The swap that costs a whole session: SELF must be an address this machine actually holds.
if ! ip -o addr show | grep -q "inet ${EVH_SELF_IP}/"; then
  echo "[wifi_session] addresses on this machine:" >&2
  ip -br -4 addr show >&2
  die "EVH_SELF_IP=$EVH_SELF_IP is not on this machine — SELF/PEER are probably swapped"
fi
if ip -o addr show | grep -q "inet ${EVH_PEER_IP}/"; then
  die "EVH_PEER_IP=$EVH_PEER_IP is on THIS machine — SELF/PEER are swapped"
fi
# The sweeps run on domain 120; sharing a domain mixes two experiments' traffic, silently.
[ "$DOMAIN" != 120 ] || die 'EVH_DOMAIN=120 is the sweep domain — pick another'

ping -c 2 -W 2 "$EVH_PEER_IP" >/dev/null 2>&1 || die "cannot ping $EVH_PEER_IP — fix the link first"

IFACE=$(ip -o -4 addr show | awk -v ip="$EVH_SELF_IP" '$4 ~ "^"ip"/" {print $2; exit}')
mkdir -p "$OUT_DIR"
CSV=${LABEL}_$( [ "$ROLE" = robot ] && echo rob || echo srv ).csv

cat <<EOF
[wifi_session] role=$ROLE label=$LABEL ${SECONDS_ARG}s
               $EVH_SELF_IP ($IFACE) -> $EVH_PEER_IP, domain $DOMAIN, image $IMAGE
               writing $OUT_DIR/$CSV
EOF

# --------------------------------------------------------------------- record
# No --entrypoint override: both images' entrypoints source ROS and exec the command (they may
# colcon-build first if the mounted install/ is stale, which is slow but harmless here).
docker run --rm --network host \
  -v "$REPO":/ws \
  -e ROS_DOMAIN_ID="$DOMAIN" \
  -e EVH_SELF_IP="$EVH_SELF_IP" -e EVH_PEER_IP="$EVH_PEER_IP" \
  -e CYCLONEDDS_URI=file:///ws/docker/cyclonedds-wifi.xml \
  "$IMAGE" \
  python3 /ws/scripts/wifi_trace.py record \
    --role "$ROLE" --seconds "$SECONDS_ARG" --out "/ws/outputs/wifi/$CSV"

ROWS=$(wc -l < "$OUT_DIR/$CSV")
echo "[wifi_session] $CSV: $ROWS rows"
# ~95 rows/s on the robot (3 streams sent + 2 received + probes), ~85 on the server.
EXPECTED=$(awk -v s="$SECONDS_ARG" -v r="$ROLE" 'BEGIN{print int(s*(r=="robot"?95:85)*0.5)}')
[ "$ROWS" -ge "$EXPECTED" ] || echo "[wifi_session] WARNING: far fewer rows than expected \
(~$EXPECTED+) — the link may not have formed; check the analyze output before recording more"

cat <<EOF

[wifi_session] done. With BOTH csvs in one place:
  python3 scripts/wifi_trace.py analyze \\
    --robot ${OUT_DIR}/${LABEL}_rob.csv --server ${OUT_DIR}/${LABEL}_srv.csv \\
    --label ${LABEL} --json ${OUT_DIR}/${LABEL}.json
EOF
