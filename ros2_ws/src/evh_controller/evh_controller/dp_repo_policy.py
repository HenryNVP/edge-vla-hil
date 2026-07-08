"""Backend for real-stanford/diffusion_policy workspace checkpoints (robomimic image tasks).

Loads the published Lift/Can/Square image checkpoints (e.g.
diffusion-policy.cs.columbia.edu/data/experiments/image/lift_ph/diffusion_policy_cnn/) WITHOUT
their training workspace: the policy object is instantiated from the checkpoint's own hydra
config and the EMA weights are loaded directly. Bypassing the workspace avoids their
training-only imports (old-diffusers lr_scheduler), which don't survive a modern diffusers.

Requires the repo on sys.path (external/diffusion_policy — see EVH_DP_REPO) plus:
hydra-core, omegaconf, zarr<3, robomimic==0.2.0 (--no-deps), h5py, psutil, termcolor.

Obs contract (dict, values stacked over the last n_obs_steps ticks, oldest first):
    obs['agentview']  uint8 [To, H, W, 3]   (robosuite agentview_image, already flipped upright)
    obs['wrist']      uint8 [To, H, W, 3]   (robot0_eye_in_hand_image, upright)
    obs['proprio']    float [To, 9]         ([eef_pos(3), eef_quat(4, xyzw), gripper_qpos(2)])

Chunking: the policy's horizon is 16 with n_obs_steps=2; instead of the repo default of 8
actions per prediction we expose the full remaining tail (horizon - (n_obs_steps-1) = 15) and
let the chunk-execution strategy decide how much to use — longer chunks are what keeps the
executor out of the degenerate d >= H regime under real inference latency.

Actions: the published robomimic image checkpoints are the ABS-ACTION variant — 10-dim
[pos(3), rot_6d(6), gripper(1)] absolute EE pose targets, executed with OSC_POSE in
control_delta=False mode. This backend converts to the project's 7-dim convention
[pos(3), axis-angle(3), gripper] (porting their pytorch3d RotationTransformer inverse in
numpy) and sets `absolute_actions=True` so the plant/reactive layer switch OSC to absolute.

Inference speed: the checkpoint was trained with DDPM(100); `denoise_steps` < 100 swaps in a
DDIM scheduler with the same noise schedule (the standard DP eval trick) so the Jetson/host can
trade action quality for latency honestly.
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np

from evh_controller.policy import ChunkPolicy

logger = logging.getLogger(__name__)

_DP_KEYS = {   # our obs dict key -> checkpoint obs key
    'agentview': 'agentview_image',
    'wrist': 'robot0_eye_in_hand_image',
}


def _ensure_dp_repo_on_path() -> None:
    candidates = [
        os.environ.get('EVH_DP_REPO', ''),
        '/ws/external/diffusion_policy',
        os.path.join(os.path.dirname(__file__), '..', '..', '..', '..',
                     'external', 'diffusion_policy'),
    ]
    for cand in candidates:
        if cand and os.path.isdir(os.path.join(cand, 'diffusion_policy')):
            if cand not in sys.path:
                sys.path.insert(0, os.path.abspath(cand))
            return
    raise ImportError(
        'diffusion_policy repo not found; clone it to external/diffusion_policy '
        'or set EVH_DP_REPO')


def load_dp_checkpoint(ckpt_path: str, device: str = 'cuda',
                       denoise_steps: int | None = None):
    """Instantiate the policy from a workspace .ckpt and load its EMA weights.

    Returns (policy_module, cfg). The policy is in eval mode on `device`.
    """
    _ensure_dp_repo_on_path()
    import dill
    import hydra
    import torch

    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    policy = hydra.utils.instantiate(cfg.policy)
    state = payload['state_dicts'].get('ema_model') or payload['state_dicts']['model']
    policy.load_state_dict(state)
    policy.eval().to(device)

    if denoise_steps is not None and denoise_steps > 0:
        _swap_to_ddim(policy, denoise_steps)

    logger.info('loaded %s: task=%s horizon=%s n_obs_steps=%s inference_steps=%s',
                ckpt_path, cfg.task_name, cfg.horizon, cfg.n_obs_steps,
                policy.num_inference_steps)
    return policy, cfg


def _swap_to_ddim(policy, num_steps: int) -> None:
    """Replace the training DDPM scheduler with DDIM on the same noise schedule."""
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    old = policy.noise_scheduler
    policy.noise_scheduler = DDIMScheduler(
        num_train_timesteps=old.config.num_train_timesteps,
        beta_start=old.config.beta_start,
        beta_end=old.config.beta_end,
        beta_schedule=old.config.beta_schedule,
        clip_sample=old.config.clip_sample,
        prediction_type=old.config.prediction_type,
        set_alpha_to_one=True,
        steps_offset=0,
    )
    policy.num_inference_steps = int(num_steps)


def _rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """pytorch3d convention: d6 = first two ROWS of R; Gram-Schmidt the third."""
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2p = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2p / np.linalg.norm(a2p, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def _matrix_to_axisangle(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle, robust near 0 and pi (via quaternion, xyzw)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 0.0)) * 2.0
        q = np.zeros(4)
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    if q[3] < 0.0:
        q = -q
    v = np.linalg.norm(q[:3])
    if v < 1e-12:
        return np.zeros(3)
    return (q[:3] / v) * (2.0 * np.arctan2(v, q[3]))


class DiffusionPolicyRepoBackend(ChunkPolicy):
    """ChunkPolicy over a diffusion_policy-repo image checkpoint (dict obs, see module doc)."""

    needs_wrist = True

    def __init__(self, ckpt_path: str, device: str = 'cuda',
                 denoise_steps: int = 16) -> None:
        import torch

        if device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning('CUDA unavailable; running the DP backend on CPU.')
            device = 'cpu'
        self.device = device
        self.denoise_steps = denoise_steps
        self._policy, self._cfg = load_dp_checkpoint(ckpt_path, device, denoise_steps)

        self.n_obs_steps = int(self._cfg.n_obs_steps)
        raw_action_dim = int(self._cfg.shape_meta.action.shape[0])
        self.absolute_actions = raw_action_dim == 10   # [pos, rot_6d, gripper] abs variant
        self.action_dim = 7 if self.absolute_actions else raw_action_dim
        # full tail of the horizon, not the repo's n_action_steps=8 (see module docstring)
        self.chunk_size = int(self._cfg.horizon) - (self.n_obs_steps - 1)
        self._image_hw = tuple(self._cfg.shape_meta.obs.agentview_image.shape[1:])  # (H, W)

    # ------------------------------------------------------------------ obs
    def _to_policy_obs(self, obs: dict) -> dict:
        """Stacked-history obs dict -> checkpoint obs_dict of [1, To, ...] tensors."""
        import torch

        To = self.n_obs_steps
        out: dict = {}
        for our_key, dp_key in _DP_KEYS.items():
            imgs = np.asarray(obs[our_key])
            if imgs.ndim == 3:
                imgs = imgs[None]                       # single frame -> history of 1
            imgs = self._pad_history(imgs, To)
            t = torch.from_numpy(imgs.copy()).float().permute(0, 3, 1, 2) / 255.0
            if t.shape[-2:] != self._image_hw:
                t = torch.nn.functional.interpolate(
                    t, size=self._image_hw, mode='bilinear', align_corners=False)
            out[dp_key] = t.unsqueeze(0).to(self.device)

        prop = np.asarray(obs['proprio'], dtype=np.float32)
        if prop.ndim == 1:
            prop = prop[None]
        prop = self._pad_history(prop, To)
        out['robot0_eef_pos'] = self._t(prop[:, 0:3])
        out['robot0_eef_quat'] = self._t(prop[:, 3:7])
        out['robot0_gripper_qpos'] = self._t(prop[:, 7:9])
        return out

    @staticmethod
    def _pad_history(arr: np.ndarray, To: int) -> np.ndarray:
        """Left-pad by repeating the oldest frame; keep the newest To frames."""
        if len(arr) >= To:
            return arr[-To:]
        pad = np.repeat(arr[:1], To - len(arr), axis=0)
        return np.concatenate([pad, arr], axis=0)

    def _t(self, a: np.ndarray):
        import torch
        return torch.from_numpy(np.ascontiguousarray(a)).float().unsqueeze(0).to(self.device)

    # -------------------------------------------------------------- predict
    def predict(self, obs: dict) -> np.ndarray:
        import torch

        with torch.no_grad():
            result = self._policy.predict_action(self._to_policy_obs(obs))
        # full horizon prediction; execute from the current obs step onward
        action_pred = result['action_pred'][0].detach().cpu().numpy()
        chunk = np.asarray(action_pred[self.n_obs_steps - 1:], dtype=np.float32)
        if self.absolute_actions:
            chunk = self._undo_abs_transform(chunk)
        return chunk

    @staticmethod
    def _undo_abs_transform(chunk10: np.ndarray) -> np.ndarray:
        """[H, 10] abs [pos, rot_6d, gripper] -> [H, 7] abs [pos, axis-angle, gripper]."""
        out = np.empty((chunk10.shape[0], 7), dtype=np.float32)
        out[:, :3] = chunk10[:, :3]
        R = _rotation_6d_to_matrix(chunk10[:, 3:9].astype(np.float64))
        for i in range(chunk10.shape[0]):
            out[i, 3:6] = _matrix_to_axisangle(R[i])
        out[:, 6] = chunk10[:, 9]
        return out
