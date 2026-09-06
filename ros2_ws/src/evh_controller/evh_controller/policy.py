"""Pluggable chunk-policy backends (diffusion / flow-matching).

We use a small diffusion/flow policy (not deterministic ACT) so that RTC and BID — which rely on
guided denoising / resampling — apply as first-class baselines. ACT+Temporal-Ensembling remains a
deterministic baseline elsewhere.

Interchangeable inference paths behind one interface so the rest of the system never changes:

  * ACTBackend      -- LeRobot ACT checkpoint (single forward per chunk); a 10-dim head is the
                       abs-action [pos, rot_6d, gripper] layout and is converted to the 7-dim
                       contract like the DP one. Needs LeRobot, which
                       needs Python 3.10+ -- host/dev only. The Jetson controller image is
                       Python 3.8 (dustynv ROS Humble base), so this backend cannot load there;
                       use ONNXBackend on-device instead (see scripts/export_onnx.py).
  * ONNXBackend     -- ACT exported to ONNX (scripts/export_onnx.py) and run with ONNX Runtime.
                       No torch/lerobot at inference time, so it is the Jetson ACT fast path
                       (onnxruntime-gpu, CUDA execution provider) where pytorch diffusion
                       inference is too slow and lerobot's Python floor rules out ACTBackend.
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

from evh_controller.rotation import abs10_to_abs7

logger = logging.getLogger(__name__)


def _import_act_policy():
    """Load ACTPolicy across LeRobot 0.3.x package layouts."""
    tried: list[str] = []
    for modpath in (
        'lerobot.policies.act.modeling_act',
        'lerobot.common.policies.act.modeling_act',
    ):
        try:
            return importlib.import_module(modpath).ACTPolicy
        except Exception as exc:
            tried.append(f'  {modpath}: {type(exc).__name__}: {exc}')
    raise ImportError(
        'Could not import LeRobot ACTPolicy. Tried:\n'
        + '\n'.join(tried)
        + '\nInstall lerobot==0.3.3 (Python 3.10+).'
    )


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


POLICY_SIDECAR = 'evh_policy.json'


def sidecar_absolute(ckpt_dir: str) -> bool | None:
    """Read `evh_policy.json` next to a checkpoint: `{"absolute_actions": true}`, or None.

    Invariant 1 needs every backend to announce its action convention, but an ACT checkpoint
    cannot tell you: absolute and delta variants are both 7-dim, unlike the DP checkpoints where
    a 10-dim head (`[pos, rot_6d, gripper]`) gives the abs variant away. The training data knows,
    so the mode is stamped beside the weights at training time (see the README's ACT section) and
    read back here. Missing file -> None, and the caller falls back to delta with a warning
    rather than guessing: a wrong guess is exactly the silent garbage-motion failure the
    plant's `/policy/absolute` cross-check exists to catch.
    """
    import json
    import os

    path = os.path.join(ckpt_dir, POLICY_SIDECAR) if os.path.isdir(ckpt_dir) else ''
    if not path or not os.path.isfile(path):
        return None
    try:
        value = json.loads(open(path).read()).get('absolute_actions')
    except Exception as exc:                       # a malformed stamp must not be read as False
        raise ValueError(f'could not read {path}: {exc}') from exc
    return None if value is None else bool(value)


def act_action_dim(ckpt_dir: str) -> int | None:
    """Width of an ACT checkpoint's action head, from its config alone (no weights loaded).

    Lets an orchestrator see what the backend will see. Reading only `config.json` keeps this
    usable where loading the policy is not — the sweep driver picking a launch argument, for
    instance, which must decide before any node starts.
    """
    import json
    import os

    path = os.path.join(ckpt_dir, 'config.json') if os.path.isdir(ckpt_dir) else ''
    if not path or not os.path.isfile(path):
        return None
    try:
        shape = json.loads(open(path).read())['output_features']['action']['shape']
        return int(shape[0])
    except Exception:                       # not an ACT config, or a layout we do not know
        return None


def stamped_absolute(backend: str, weights_path: str) -> bool | None:
    """The action convention a checkpoint declares, read WITHOUT loading it (None = unstamped).

    Lets a caller that only orchestrates — the sweep driver picking `absolute:=` for a launch —
    ask the checkpoint the same question the backend will, instead of guessing from the backend
    name. `dp` is absent on purpose: it derives its mode from the head width at load time.
    """
    backend = backend.lower()
    if backend in ('act', 'act_lerobot'):
        head = act_action_dim(weights_path)
        if head == 10:                    # [pos, rot_6d, gripper] — abs, and says so itself
            return True
        return sidecar_absolute(weights_path)
    if backend in ('onnx', 'act_onnx'):
        import json
        import os

        meta_path = os.path.splitext(weights_path)[0] + '.json'
        if not weights_path or not os.path.isfile(meta_path):
            return None
        value = json.loads(open(meta_path).read()).get('absolute_actions')
        return None if value is None else bool(value)
    return None


def resolve_absolute(override: bool | None, stamped: bool | None, source: str) -> bool:
    """Settle a backend's action convention: explicit override, else the stamp, else delta.

    Falling back to delta is a choice, not a default: an unstamped abs-action policy then
    announces the wrong mode on `/policy/absolute` and the plant aborts — loud and immediate,
    rather than a run that completes with plausible-looking metrics and garbage motion.
    """
    if override is not None:
        return bool(override)
    if stamped is not None:
        return bool(stamped)
    logger.warning(
        '%s carries no action-convention stamp — assuming DELTA actions. If it was trained on '
        'absolute actions, stamp it (see the README) or the plant will abort the mode '
        'cross-check.', source,
    )
    return False


def newest(obs_value: np.ndarray) -> np.ndarray:
    """Latest entry of a possibly history-stacked observation value."""
    arr = np.asarray(obs_value)
    return arr[-1] if arr.ndim in (2, 4) else arr


class ACTBackend(ChunkPolicy):
    """LeRobot ACT checkpoint — one transformer forward per chunk (no denoise loop)."""

    def __init__(self, ckpt_path: str, device: str = 'cuda',
                 absolute: bool | None = None) -> None:
        self.ckpt_path = ckpt_path
        self.device = device
        self._absolute_override = absolute
        if absolute is not None:                    # applies even with no checkpoint to load
            self.absolute_actions = bool(absolute)
        self.denoise_steps = 1
        self.action_dim = 7
        self.chunk_size = 16
        self.n_obs_steps = 1
        self._model = None
        self._torch_device = 'cpu'
        self._image_keys: list[str] = []
        self._image_shapes: dict[str, tuple[int, ...]] = {}
        self._state_key = 'observation.state'
        self._state_dim = 7
        self._load()

    def _load(self) -> None:
        if not self.ckpt_path:
            self._model = None
            return

        ACTPolicy = _import_act_policy()
        self._torch_device = _resolve_device(self.device)
        logger.info('Loading ACT policy from %s on %s', self.ckpt_path, self._torch_device)
        self._model = ACTPolicy.from_pretrained(self.ckpt_path)
        self._model.to(self._torch_device).eval()

        cfg = self._model.config
        self.chunk_size = int(cfg.chunk_size)
        self.n_obs_steps = int(getattr(cfg, 'n_obs_steps', 1))
        if cfg.action_feature and cfg.action_feature.shape:
            raw_action_dim = int(cfg.action_feature.shape[0])
            # A 10-dim head is [pos(3), rot_6d(6), gripper] — the abs-action layout, and the same
            # tell the DP backend uses. Absolute rotation is trained as 6D because axis-angle is
            # discontinuous exactly where these targets live (see rotation.py), so a 10-dim ACT
            # announces its own mode and needs no stamp.
            self._rot6d = raw_action_dim == 10
            self.action_dim = 7 if self._rot6d else raw_action_dim

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
        self.needs_wrist = len(self._image_keys) > 1

        stamped = True if self._rot6d else sidecar_absolute(self.ckpt_path)
        self.absolute_actions = resolve_absolute(
            self._absolute_override, stamped, self.ckpt_path)

        logger.info(
            'Loaded ACT policy: chunk_size=%d action_dim=%d state_dim=%d image_keys=%s '
            'absolute=%s',
            self.chunk_size,
            self.action_dim,
            self._state_dim,
            self._image_keys or ['<none>'],
            self.absolute_actions,
        )

    def _obs_to_batch(self, obs: dict) -> dict:
        import torch
        import torch.nn.functional as F

        batch: dict[str, torch.Tensor] = {}
        for key in self._image_keys:
            if key == self._image_keys[0]:
                image = np.asarray(newest(obs['agentview']), dtype=np.uint8)
            elif len(self._image_keys) > 1 and key == self._image_keys[1]:
                image = np.asarray(newest(obs.get('wrist', obs['agentview'])), dtype=np.uint8)
            else:
                image = np.asarray(newest(obs['agentview']), dtype=np.uint8)

            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f'expected HWC uint8 image, got shape {image.shape}')
            img = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            c, h, w = self._image_shapes[key]
            if (img.shape[0], img.shape[1], img.shape[2]) != (c, h, w):
                img = F.interpolate(
                    img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False,
                )[0]
            batch[key] = img.unsqueeze(0).to(self._torch_device)

        state = newest(obs['proprio'])
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
        with torch.inference_mode():
            chunk = self._model.predict_action_chunk(batch)[0].cpu().numpy()
        n = min(self.chunk_size, chunk.shape[0])
        chunk = np.asarray(chunk[:n], dtype=np.float32)
        return abs10_to_abs7(chunk) if self._rot6d else chunk


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


class ONNXBackend(ChunkPolicy):
    """ACT exported to ONNX (scripts/export_onnx.py), run through ONNX Runtime.

    Pure numpy + onnxruntime at inference time -- no torch, no lerobot -- so this is the
    backend that actually loads on the Jetson controller's Python 3.8 image. Normalization and
    unnormalization of image/state/action are baked into the graph by the exporter; this class
    only resizes the raw HWC uint8 frame and feeds/reads the two named tensors ('image', 'state'
    in, 'action_chunk' out). Sidecar `<onnx>.json` (written by the exporter) carries chunk_size /
    action_dim / image_shape / state_dim so no checkpoint config needs to be read here.
    """

    def __init__(self, onnx_path: str, providers: list[str] | None = None,
                 absolute: bool | None = None) -> None:
        self.onnx_path = onnx_path
        self._absolute_override = absolute
        if absolute is not None:                    # applies even with no graph to load
            self.absolute_actions = bool(absolute)
        self.action_dim = 7
        self.chunk_size = 16
        self.denoise_steps = 1
        self._session = None
        self._image_shape = (3, 96, 96)
        self._state_dim = 7
        self._providers = providers
        self._load()

    def _load(self) -> None:
        if not self.onnx_path:
            self._session = None
            return

        import json
        from pathlib import Path

        import onnxruntime as ort

        onnx_path = Path(self.onnx_path)
        meta_path = onnx_path.with_suffix('.json')
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}

        providers = self._providers
        if not providers:
            available = ort.get_available_providers()
            providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider')
                        if p in available] or available[:1]

        logger.info('Loading ONNX policy from %s (providers=%s)', onnx_path, providers)
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)
        active = self._session.get_providers()
        if 'CUDAExecutionProvider' not in active:
            logger.warning(
                'ONNX policy running on %s, not CUDA -- check the GPU wheel/providers '
                '(see scripts/bench_onnx.py)', active,
            )

        self.chunk_size = int(meta.get('chunk_size', self.chunk_size))
        self.action_dim = int(meta.get('action_dim', self.action_dim))
        self._state_dim = int(meta.get('state_dim', self._state_dim))
        image_shape = meta.get('image_shape')
        if image_shape:
            self._image_shape = tuple(int(v) for v in image_shape)

        self.absolute_actions = resolve_absolute(
            self._absolute_override, meta.get('absolute_actions'), str(onnx_path))

        logger.info(
            'Loaded ONNX policy: chunk_size=%d action_dim=%d state_dim=%d image_shape=%s '
            'absolute=%s',
            self.chunk_size, self.action_dim, self._state_dim, self._image_shape,
            self.absolute_actions,
        )

    def _feeds(self, obs: dict) -> dict:
        image = np.asarray(newest(obs['agentview']), dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'expected HWC uint8 image, got shape {image.shape}')

        c, h, w = self._image_shape
        if (image.shape[0], image.shape[1]) != (h, w):
            import cv2
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        img = image.astype(np.float32).transpose(2, 0, 1) / 255.0

        state = newest(obs['proprio'])
        state_vec = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_vec.size < self._state_dim:
            state_vec = np.pad(state_vec, (0, self._state_dim - state_vec.size))
        elif state_vec.size > self._state_dim:
            state_vec = state_vec[:self._state_dim]

        return {
            'image': img[np.newaxis].astype(np.float32),
            'state': state_vec[np.newaxis].astype(np.float32),
        }

    def predict(self, obs: dict) -> np.ndarray:
        if self._session is None:
            return np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)

        out_name = self._session.get_outputs()[0].name
        chunk = self._session.run([out_name], self._feeds(obs))[0][0]
        n = min(self.chunk_size, chunk.shape[0])
        return np.asarray(chunk[:n], dtype=np.float32)


def make_policy(backend: str, weights_path: str, denoise_steps: int = 16,
                absolute: bool | None = None) -> ChunkPolicy:
    """`absolute` overrides the action convention a backend derives for itself (None = derive)."""
    backend = backend.lower()
    if backend in ('pytorch', 'torch', 'fallback'):
        return PyTorchBackend(weights_path)
    if backend in ('dp', 'diffusion_policy'):
        from evh_controller.dp_repo_policy import DiffusionPolicyRepoBackend
        return DiffusionPolicyRepoBackend(weights_path, denoise_steps=denoise_steps)
    if backend in ('act', 'act_lerobot'):
        return ACTBackend(weights_path, absolute=absolute)
    if backend in ('onnx', 'act_onnx'):
        return ONNXBackend(weights_path, absolute=absolute)
    if backend in ('tensorrt', 'trt'):
        return TensorRTBackend(weights_path)
    raise ValueError(f'unknown backend: {backend!r}')
