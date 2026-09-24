#!/usr/bin/env bash
# One WiFi trace session, or the whole set: run this on BOTH machines at once, one `robot` and
# one `server`.
#
# It is a wrapper around `scripts/wifi_trace.py record` (see that file for what is measured and
# why). All it adds is the parts that are easy to get wrong across two terminals on two machines:
#
#   * the two IP env vars the CycloneDDS WiFi config substitutes, which are mirror images of each
#     other, and which fail SILENTLY when swapped — discovery never completes, every stream reads
#     as 100% loss, and nothing prints an error. The addresses are checked against the machine's
#     own interfaces before the container starts.
#   * the frame size. The testbed's raw 84x84x3 frames SATURATE the radio (measured: the sender
#     was held to 64-70% of its own cadence on an idle link, 32-39% on a busy one, while the same
#     recorder offers 100% on loopback), and a saturated trace measures the sender rather than the
#     channel. The default here is a compressed frame size, which is also what a deployed stack
#     sends: 2600 B is about JPEG q90 at 84x84 (measured on 200 real Square frames), which cuts
#     the uplink from 847 KB/s to ~98 and each frame from 16 DDS fragments to 2. Use
#     EVH_IMAGE_BYTES=21168 to reproduce the saturation deliberately.
#
#   export EVH_SELF_IP=192.168.0.73 EVH_PEER_IP=192.168.0.50   # robot machine
#   scripts/wifi_session.sh robot all
#
#   export EVH_SELF_IP=192.168.0.50 EVH_PEER_IP=192.168.0.73   # server machine
#   scripts/wifi_session.sh server all
#
# Start the SERVER side first for each condition: the analyzer trims 2 s off each end, so a few
# seconds of one-sided start-up is absorbed (the robot also waits EVH_LEAD_S before recording),
# but a minute of it reads as loss.
#
# Afterwards, copy the robot CSVs to wherever you analyze and run the commands this script prints.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/wifi_session.sh <robot|server> <label|all> [seconds]

  label     names the condition and the output files (near_los, far_nlos, busy, smoke, ...)
            or `all` to run every condition in ALL_CONDITIONS in turn, prompting between them
  seconds   default 600; use 30 for a first smoke test

environment:
  EVH_SELF_IP      this machine's WiFi address         (required)
  EVH_PEER_IP      the other machine's WiFi address    (required)
  EVH_IMAGE_BYTES  bytes per camera frame              (default 2600 ~= JPEG q90 at 84x84;
                                                        1900 ~= q80; 21168 = raw, which saturates)
  EVH_DOMAIN       ROS_DOMAIN_ID                       (default 77)
  EVH_IMAGE        docker image                        (default: jetson on arm64, host otherwise)
  EVH_OUT_DIR      output directory on the host        (default <repo>/outputs/wifi)
  EVH_LEAD_S       robot-side start delay, seconds     (default 2)
EOF
  exit 2
}

# label -> what the operator has to physically arrange before that condition runs
ALL_CONDITIONS=(near_los far_nlos busy)
describe() {
  case "$1" in
    near_los) echo 'both machines within a few metres of the AP, line of sight, no other traffic' ;;
    far_nlos) echo 'robot side one or two rooms away, through walls, ~10-15 m' ;;
    busy)     echo 'back near the AP, but with a video stream and a large file transfer on it' ;;
    *)        echo 'no setup notes for this label' ;;
  esac
}

[ $# -ge 2 ] || usage
ROLE=$1
LABEL=$2
SECONDS_ARG=${3:-600}
case "$ROLE" in robot|server) ;; *) echo "bad role: $ROLE" >&2; usage ;; esac

REPO=$(cd "$(dirname "$0")/.." && pwd)
DOMAIN=${EVH_DOMAIN:-77}
OUT_DIR=${EVH_OUT_DIR:-$REPO/outputs/wifi}
IMAGE_BYTES=${EVH_IMAGE_BYTES:-2600}
LEAD_S=${EVH_LEAD_S:-2}
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

echo "[wifi_session] $EVH_SELF_IP ($IFACE) -> $EVH_PEER_IP, domain $DOMAIN, image $IMAGE"
echo "[wifi_session] role=$ROLE, ${IMAGE_BYTES} B per frame, ${SECONDS_ARG}s per condition"

# -------------------------------------------------------------- one condition
run_one() {
  local label=$1
  local csv="${label}_$([ "$ROLE" = robot ] && echo rob || echo srv).csv"

  echo
  echo "[wifi_session] --- $label: $(describe "$label")"
  echo "[wifi_session]     writing $OUT_DIR/$csv"
  [ "$ROLE" = robot ] && sleep "$LEAD_S"   # let the server settle; matches the analyzer's trim

  # No --entrypoint override: both images' entrypoints source ROS and exec the command (they may
  # colcon-build first if the mounted install/ is stale, which is slow but harmless here).
  docker run --rm --network host \
    -v "$REPO":/ws \
    -e ROS_DOMAIN_ID="$DOMAIN" \
    -e EVH_SELF_IP="$EVH_SELF_IP" -e EVH_PEER_IP="$EVH_PEER_IP" \
    -e CYCLONEDDS_URI=file:///ws/docker/cyclonedds-wifi.xml \
    "$IMAGE" \
    python3 /ws/scripts/wifi_trace.py record \
      --role "$ROLE" --seconds "$SECONDS_ARG" --image-bytes "$IMAGE_BYTES" \
      --out "/ws/outputs/wifi/$csv"

  local rows expected
  rows=$(wc -l < "$OUT_DIR/$csv")
  # ~95 rows/s on the robot (3 streams sent + 2 received + probes), ~85 on the server.
  expected=$(awk -v s="$SECONDS_ARG" -v r="$ROLE" 'BEGIN{print int(s*(r=="robot"?95:85)*0.5)}')
  echo "[wifi_session]     $csv: $rows rows"
  [ "$rows" -ge "$expected" ] || echo "[wifi_session]     WARNING: far fewer rows than expected \
(~$expected+) — the link may not have formed; check the analyze output before recording more"
}

# ---------------------------------------------------------------------- drive
if [ "$LABEL" = all ]; then
  echo "[wifi_session] running ${#ALL_CONDITIONS[@]} conditions: ${ALL_CONDITIONS[*]}"
  for i in "${!ALL_CONDITIONS[@]}"; do
    label=${ALL_CONDITIONS[$i]}
    echo
    echo "[wifi_session] NEXT ($((i + 1))/${#ALL_CONDITIONS[@]}): $label"
    echo "[wifi_session]   set up: $(describe "$label")"
    echo "[wifi_session]   press Enter on the SERVER machine first, then on the ROBOT machine."
    read -r -p "[wifi_session]   ready? [Enter] " _ || true
    run_one "$label"
  done
  RAN=("${ALL_CONDITIONS[@]}")
else
  run_one "$LABEL"
  RAN=("$LABEL")
fi

echo
echo "[wifi_session] done. With BOTH machines' csvs in one place:"
for label in "${RAN[@]}"; do
  cat <<EOF
  python3 scripts/wifi_trace.py analyze \\
    --robot ${OUT_DIR}/${label}_rob.csv --server ${OUT_DIR}/${label}_srv.csv \\
    --label ${label} --json ${OUT_DIR}/${label}.json
EOF
done
