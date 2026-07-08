#!/usr/bin/env bash
# Deps for loading real-stanford/diffusion_policy checkpoints (see evh_controller/dp_repo_policy.py).
# robomimic 0.2.0 is installed --no-deps: its egl_probe build dep fails on modern images and is
# only needed for its own env wrappers, which we don't use (the model code is what we import).
set -e
pip install --quiet hydra-core omegaconf "zarr<3"
pip install --quiet --no-deps robomimic==0.2.0
pip install --quiet h5py psutil termcolor tensorboardX
python3 -c "import robomimic, hydra, zarr; print('dp deps ok (robomimic', robomimic.__version__, ')')"
