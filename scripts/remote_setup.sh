#!/usr/bin/env bash
# One-time setup of a rented GPU box (a Vast.ai / RunPod container on any Ubuntu + NVIDIA driver
# image) for bev_planner training and dp_mech evaluation, without pushing our 19 GB Docker image:
# a Python 3.10 venv (uv) with the host image's pins, the GL libs, the ImageNet backbones
# prefetched (parallel jobs would race on the download), then a smoke run (2 oracle demos, 200
# training steps on them, one eval episode; needs no pushed data) that picks the MuJoCo GL backend.
# EGL needs the container to expose the driver's graphics capability (NVIDIA_DRIVER_CAPABILITIES
# with graphics or all); without it this falls back to OSMesa, which renders on the CPU. Writes
# ~/evh_env.sh for run scripts to source. Idempotent; run from the repo root after
# scripts/remote_run.sh has pushed the code.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
SUDO=$([ "$(id -u)" = 0 ] || echo sudo)
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq --no-install-recommends libgl1 libegl1 libglib2.0-0 libosmesa6 \
    curl ca-certificates > /dev/null
export PATH=$HOME/.local/bin:$PATH
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
V=$HOME/evh-venv
[ -x $V/bin/python ] || uv venv -q -p 3.10 $V
PIP="uv pip install -q -p $V/bin/python"
$PIP torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
# robosuite --no-deps: its pynput dep builds evdev from source and is only for teleop devices
$PIP numpy==1.26.4 mujoco==2.3.7 numba 'scipy<1.15' opencv-python==4.11.0.86 pillow pyyaml \
    termcolor h5py
$PIP --no-deps robosuite==1.4.1
cat > ~/evh_env.sh << EOF
source $V/bin/activate
export TORCH_HOME=\$HOME/.cache/torch OMP_NUM_THREADS=4
EOF
source ~/evh_env.sh
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name())"
python -c "import torchvision.models as m; m.resnet18(weights='IMAGENET1K_V1'); \
m.mobilenet_v3_small(weights='IMAGENET1K_V1')"

S=$(mktemp -d)
for gl in egl osmesa; do      # collect 2 demos, train on them, run one eval episode
  if MUJOCO_GL=$gl PYOPENGL_PLATFORM=$gl python scripts/oracle_demos.py collect --task nut_goal \
      --top --npz-dir $S/npz --episodes 2 &&
    python scripts/bev_planner.py --data $S/npz --local-views agent --steps 200 --out $S/smoke.pt &&
    MUJOCO_GL=$gl PYOPENGL_PLATFORM=$gl python scripts/dp_mech.py --task nut_goal --backend bev \
      --weights $S/smoke.pt --strategies cloud_local --replan 4 --episodes 1 --delays 1 --speeds 0 \
      --deadline-s 40 --out $S/smoke.csv; then
    echo "export MUJOCO_GL=$gl PYOPENGL_PLATFORM=$gl" >> ~/evh_env.sh
    echo "setup ok: MUJOCO_GL=$gl"
    exit 0
  fi
done
echo 'setup failed: the smoke eval rendered with neither EGL nor OSMesa' >&2
exit 1
