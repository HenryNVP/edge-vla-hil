#!/usr/bin/env bash
# Run an experiment directory's run.sh on a rented GPU box and bring the results home:
#
#   scripts/remote_run.sh <user@host> <ssh-port> outputs/<expt> [data dir ...]
#
# Pushes the working tree's code (scripts/, ros2_ws/src/, external DP and robomimic repos,
# uncommitted changes included: the box has no .git, so the commit and the diff against it are
# saved beside the results as code.commit / code.patch), the data dirs and <expt>/run.sh; runs
# scripts/remote_setup.sh; launches run.sh detached (it survives ssh drops), then pulls <expt>/
# every 10 min until run.sh writes <expt>/DONE. Rerunning reattaches: a box whose <expt>/run.log
# exists is not relaunched. The box is left running: destroy it once the results are home (Vast
# bills a stopped instance's disk).
set -uo pipefail
H=$1 P=$2 X=${3%/}
shift 3
R=edge-vla-hil-interp     # own dir, venv and env file: the box may also serve the query-planner runs
E="EVH_VENV=\$HOME/evh-interp-venv EVH_ENV=\$HOME/evh_interp_env.sh"
SSH="ssh -p $P -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=60"
cd "$(dirname "$0")/.."
[ -f $X/run.sh ] || { echo "no $X/run.sh"; exit 1; }
rs() { rsync -az -e "$SSH" "$@"; }
$SSH $H "command -v rsync > /dev/null || (apt-get update -qq && apt-get install -y -qq rsync)" \
    > /dev/null || { echo "cannot reach $H:$P"; exit 1; }
if ! $SSH $H "test -f $R/$X/run.log"; then
  git rev-parse HEAD > $X/code.commit
  git diff HEAD -- scripts ros2_ws > $X/code.patch
  rs --relative --copy-links --exclude __pycache__ --exclude .git scripts ros2_ws/src pyproject.toml external/diffusion_policy \
      external/robomimic $X/run.sh $X/code.commit $X/code.patch "$@" $H:$R/ || exit 1
  echo "$(date +%H:%M) pushed; setting up (log: $X/setup.log)"
  $SSH $H "cd $R && $E bash scripts/remote_setup.sh" > $X/setup.log 2>&1 \
      || { echo "setup failed, see $X/setup.log"; exit 1; }
  # -n: without it this ssh never returns (seen on Vast); not in $SSH, rsync needs ssh's stdin
  # timeout: on Vast this ssh can stay open after the job is detached, which would block the poll
  timeout 60 $SSH -n $H "cd $R && $E nohup setsid bash $X/run.sh > $X/run.log 2>&1 < /dev/null &"
  echo "$(date +%H:%M) launched"
fi
fails=0
while sleep 600; do
  # run.sh writes its own pid; a pgrep for the script name would match this ssh command itself
  s=$($SSH $H "cd $R/$X && if [ -f DONE ]; then echo done; \
      elif kill -0 \$(cat run.pid) 2> /dev/null; then echo running; else echo dead; fi") \
      || { fails=$((fails + 1)); [ $fails -lt 6 ] && continue; echo "lost $H for 1 h"; exit 1; }
  fails=0
  # pull results only: run.sh and code.* are pushed from here and must not overwrite local edits
  rs --exclude "*.resume*" --exclude run.sh --exclude "code.*" $H:$R/$X/ $X/
  case $s in
    done) echo "$(date +%H:%M) done; results in $X/"; exit 0 ;;
    dead) echo "$(date +%H:%M) run.sh died without DONE, see $X/run.log"; exit 1 ;;
  esac
done
