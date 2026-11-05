#!/usr/bin/env bash
# One-time setup of a rented GPU box (a Vast.ai / RunPod container on any Ubuntu + NVIDIA driver
# image) for the interp runs, without pushing our 19 GB Docker image: a Python 3.10 venv (uv) with
# the host image's pins plus LeRobot 0.3.3 with its SmolVLA extras (the image has no transformers),
# the GL libs, and a render smoke test that picks the MuJoCo GL backend. EGL needs the container to
# expose the driver's graphics capability (NVIDIA_DRIVER_CAPABILITIES with graphics or all);
# without it this falls back to OSMesa, which renders on the CPU. Writes ~/evh_env.sh for run
# scripts to source. Idempotent; run from the repo root after scripts/remote_run.sh has pushed the
# code. EVH_VENV picks the venv (default ~/evh-venv), EVH_ENV the env file (default ~/evh_env.sh);
# EVH_NO_APT=1 skips apt (a workstation).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if [ -z "${EVH_NO_APT:-}" ]; then
  SUDO=$([ "$(id -u)" = 0 ] || echo sudo)
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq --no-install-recommends libgl1 libegl1 libglib2.0-0 libosmesa6 \
      curl ca-certificates > /dev/null
fi
export PATH=$HOME/.local/bin:$PATH
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
V=${EVH_VENV:-$HOME/evh-venv}
ENVF=${EVH_ENV:-$HOME/evh_env.sh}
[ -x $V/bin/python ] || uv venv -q -p 3.10 $V
PIP="uv pip install -q -p $V/bin/python"
$PIP torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
# robosuite --no-deps: its pynput dep builds evdev from source and is only for teleop devices
# lerobot --no-deps too (its pynput is the same teleop-only evdev build): its requirements are
# installed below without pynput, with the SmolVLA extras
$PIP --no-deps robosuite==1.4.1 'lerobot==0.3.3'
$V/bin/python - > $V/lerobot_reqs.txt << 'PY'
import importlib.metadata as m, re
for r in m.requires('lerobot'):
    name = re.split(r'[<>=!;\[ ]', r, maxsplit=1)[0]
    extra = re.search(r'extra == "([^"]+)"', r)
    if name in ('pynput', 'cmake', 'lerobot') or (extra and extra[1] not in ('smolvla', 'transformers-dep')):
        continue
    print(r.split(';')[0].strip() if extra else r)
PY
$PIP numpy==1.26.4 mujoco==2.3.7 numba 'scipy<1.15' opencv-python==4.11.0.86 pillow pyyaml \
    termcolor h5py torch==2.7.1 torchvision==0.22.1 -r $V/lerobot_reqs.txt \
    --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
cat > $ENVF << EOF
source $V/bin/activate
export TORCH_HOME=\$HOME/.cache/torch OMP_NUM_THREADS=4 PYTHONPATH=\$PWD/ros2_ws/src/evh_controller:\$PWD/ros2_ws/src/evh_plant
EOF
source $ENVF
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name())"
python -c "import torchvision.models as m; m.resnet18(weights='IMAGENET1K_V1')"
python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy"

for gl in egl osmesa; do      # render one Lift frame through the same env the evals use
  if MUJOCO_GL=$gl PYOPENGL_PLATFORM=$gl python -c "
import sys; sys.path.insert(0, 'scripts')
from interp_gainmap import TASKS, make_env
env, obs = make_env(TASKS['lift']); assert obs['agentview_image'].std() > 1; env.close()"; then
    echo "export MUJOCO_GL=$gl PYOPENGL_PLATFORM=$gl" >> $ENVF
    echo "setup ok: MUJOCO_GL=$gl"
    exit 0
  fi
done
echo 'setup failed: a Lift frame rendered with neither EGL nor OSMesa' >&2
exit 1
