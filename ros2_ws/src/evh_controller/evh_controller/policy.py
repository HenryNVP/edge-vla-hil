"""Pluggable chunk-policy backends (diffusion / flow-matching).

We use a small diffusion/flow policy (not deterministic ACT) so that RTC and BID — which rely on
guided denoising / resampling — apply as first-class baselines. ACT+Temporal-Ensembling remains a
deterministic baseline elsewhere.

Interchangeable inference paths behind one interface so the rest of the system never changes:

  * PyTorchBackend  -- LeRobot diffusion checkpoint, runs anywhere (dev + Jetson fallback).
  * DiffusionPolicyRepoBackend (dp_repo_policy.py) -- real-stanford/diffusion_policy robomimic
                       image checkpoints (the project's actual Lift policy).
  * TensorRTBackend -- serialized .engine built from an exported ONNX policy (Jetson fast path).

Observation contract: a dict whose values may carry a history axis (stacked over the last
`n_obs_steps` control ticks, oldest first — the controller maintains the history):

  obs['agentview']  uint8 [To, H, W, 3] or [H, W, 3]
  obs['wrist']      uint8 [To, H, W, 3]           (optional; backends declare needs_wrist)
  obs['proprio']    float [To, D] or [D]          ([eef_pos(3), eef_quat(4), gripper_qpos(2)])

The interface exposes BOTH plain chunk prediction and *inpainting* prediction:

  predict(obs)                      -> action chunk [H, A]
  predict_inpaint(obs, prefix, w)   -> action chunk [H, A], guided so the first len(prefix)
                                       entries stay close to `prefix` with weights `w`

`predict_inpaint` is what the RTC strategy needs (freeze-d + soft-masked guidance).
Actions are end-effector (Cartesian) pose deltas (OSC_POSE convention).
"""
from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


def _import_diffusion_policy():
    """Load DiffusionPolicy across LeRobot 0.3.x package layouts."""
    tried: list[str] = []
    for modpath in (
        'lerobot.policies.diffusion.modeling_diffusion',
        'lerobot.common.policies.diffusion.modeling_diffusion',
    ):
        try:
            return importlib.import_module(modpath).DiffusionPolicy
        except Exception as exc:
            tried.append(f'  {modpath}: {type(exc).__name__}: {exc}')
    raise ImportError(
        'Could not import LeRobot DiffusionPolicy. Tried:\n'
        + '\n'.join(tried)
        + '\nInstall a compatible release: pip install lerobot==0.3.3'
    )


def _resolve_device(requested: str) -> str:
    import torch

    if requested.startswith('cuda') and not torch.cuda.is_available():
        logger.warning('CUDA requested but unavailable; using CPU for policy inference.')
        return 'cpu'
    return requested


class ChunkPolicy(ABC):
    """obs dict (see module docstring) -> action chunk [chunk_size, action_dim]."""

    action_dim: int
    chunk_size: int
    denoise_steps: int
    n_obs_steps: int = 1            # history depth the controller must maintain
    needs_wrist: bool = False       # whether obs['wrist'] is required
    absolute_actions: bool = False  # actions are absolute EE pose targets, not deltas

    @abstractmethod
    def predict(self, obs: dict) -> np.ndarray:
        ...

    def predict_inpaint(self, obs: dict, prefix: np.ndarray,
                        weights: np.ndarray) -> np.ndarray:
        """Guided generation with a soft-masked prefix (for RTC).

        Default: plain predict then soft-blend the frozen prefix. A real diffusion backend
        should instead inject `prefix`/`weights` into the denoising guidance (see RTCExecutor
        and the RTC paper's W_i masking). Override per backend.
        """
        chunk = self.predict(obs)
        if prefix is None or len(prefix) == 0:
            return chunk
        k = min(len(prefix), len(weights), chunk.shape[0])
        for i in range(k):
            w = float(weights[i])
            if w >= 1.0:
                chunk[i] = prefix[i]
            elif w > 0.0:
                chunk[i] = w * prefix[i] + (1.0 - w) * chunk[i]
        return chunk


def newest(obs_value: np.ndarray) -> np.ndarray:
    """Latest entry of a possibly history-stacked observation value."""
    arr = np.asarray(obs_value)
    return arr[-1] if arr.ndim in (2, 4) else arr


class PyTorchBackend(ChunkPolicy):
    def __init__(self, ckpt_path: str, device: str = 'cuda', denoise_steps: int = 5) -> None:
        self.ckpt_path = ckpt_path
        self.device = device
        self.action_dim = 7          # OSC_POSE: 6-DoF EE delta + gripper
        self.chunk_size = 16
        self.denoise_steps = denoise_steps
        self._model = None
        self._torch_device = 'cpu'
        self._image_keys: list[str] = []
        self._image_shapes: dict[str, tuple[int, ...]] = {}
        self._state_key = 'observation.state'
        self._state_dim = 7
        self._load()

    def _load(self) -> None:
        """Load a LeRobot DiffusionPolicy checkpoint (Hub id or local pretrained_model dir)."""
        if not self.ckpt_path:
            self._model = None
            return

        DiffusionPolicy = _import_diffusion_policy()

        self._torch_device = _resolve_device(self.device)
        logger.info('Loading diffusion policy from %s on %s', self.ckpt_path, self._torch_device)
        self._model = DiffusionPolicy.from_pretrained(self.ckpt_path)
        self._model.to(self._torch_device).eval()

        cfg = self._model.config
        self.chunk_size = int(getattr(cfg, 'n_action_steps', self.chunk_size))
        action_feat = cfg.output_features.get('action')
        if action_feat is not None and action_feat.shape:
            self.action_dim = int(action_feat.shape[0])

        self._image_keys = list(cfg.image_features.keys()) if cfg.image_features else []
        self._image_shapes = {
            key: tuple(cfg.input_features[key].shape)
            for key in self._image_keys
        }

        if self._state_key not in cfg.input_features:
            for key, feat in cfg.input_features.items():
                feat_type = getattr(feat, 'type', None)
                if feat_type is not None and str(feat_type).endswith('STATE'):
                    self._state_key = key
                    break
                if 'state' in key:
                    self._state_key = key
                    break

        state_shape = cfg.input_features[self._state_key].shape
        self._state_dim = int(state_shape[0]) if state_shape else self._state_dim

        if len(self._image_keys) > 1:
            logger.warning(
                'Policy expects %d image keys (%s); only the plant camera is wired today — '
                'duplicating the first view for all keys.',
                len(self._image_keys),
                ', '.join(self._image_keys),
            )

        infer_steps = getattr(cfg, 'num_inference_steps', None)
        if infer_steps is not None:
            self.denoise_steps = int(infer_steps)

        logger.info(
            'Loaded diffusion policy: chunk_size=%d action_dim=%d state_dim=%d image_keys=%s',
            self.chunk_size,
            self.action_dim,
            self._state_dim,
            self._image_keys or ['<none>'],
        )

    def _obs_to_batch(self, obs: dict) -> dict:
        """Map the newest observation (see module contract) to a LeRobot inference batch."""
        import torch
        import torch.nn.functional as F

        image = np.asarray(newest(obs['agentview']), dtype=np.uint8)
        state = newest(obs['proprio'])
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'expected HWC uint8 image, got shape {image.shape}')

        img = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        batch: dict[str, torch.Tensor] = {}

        for key in self._image_keys:
            c, h, w = self._image_shapes[key]
            resized = img
            if (img.shape[0], img.shape[1], img.shape[2]) != (c, h, w):
                resized = F.interpolate(
                    img.unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False,
                )[0]
            batch[key] = resized.unsqueeze(0).to(self._torch_device)

        state_vec = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_vec.size < self._state_dim:
            state_vec = np.pad(state_vec, (0, self._state_dim - state_vec.size))
        elif state_vec.size > self._state_dim:
            state_vec = state_vec[:self._state_dim]
        batch[self._state_key] = (
            torch.from_numpy(state_vec).unsqueeze(0).to(self._torch_device)
        )
        return batch

    def predict(self, obs: dict) -> np.ndarray:
        if self._model is None:
            return np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)

        import torch

        batch = self._obs_to_batch(obs)
        self._model.reset()
        chunk: list[np.ndarray] = []
        with torch.inference_mode():
            for _ in range(self.chunk_size):
                action = self._model.select_action({k: v.clone() for k, v in batch.items()})
                chunk.append(action[0].detach().cpu().numpy())
        return np.asarray(chunk, dtype=np.float32)


class TensorRTBackend(ChunkPolicy):
    def __init__(self, engine_path: str, denoise_steps: int = 5) -> None:
        self.engine_path = engine_path
        self.action_dim = 7
        self.chunk_size = 16
        self.denoise_steps = denoise_steps
        self._engine = None
        self._load()

    def _load(self) -> None:
        """TODO: deserialize TRT engine + allocate bindings (see scripts/build_trt_engine.py)."""
        self._engine = None  # STUB

    def predict(self, obs: dict) -> np.ndarray:
        if self._engine is None:
            return np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)  # STUB
        raise NotImplementedError


def make_policy(backend: str, weights_path: str, denoise_steps: int = 16) -> ChunkPolicy:
    backend = backend.lower()
    if backend in ('pytorch', 'torch', 'fallback'):
        return PyTorchBackend(weights_path)
    if backend in ('dp', 'diffusion_policy'):
        from evh_controller.dp_repo_policy import DiffusionPolicyRepoBackend
        return DiffusionPolicyRepoBackend(weights_path, denoise_steps=denoise_steps)
    if backend in ('tensorrt', 'trt'):
        return TensorRTBackend(weights_path)
    raise ValueError(f'unknown backend: {backend!r}')
