"""Diffusion Policy through ONNX Runtime — the Jetson's fast path for the DP checkpoints.

`ONNXBackend` (policy.py) covers ACT: one graph, one forward pass, newest frame only. A diffusion
policy needs none of those assumptions to hold. It sees TWO cameras over `n_obs_steps` of history,
and one prediction is `num_inference_steps` passes through a UNet with a DDIM update between them.

So the split that `scripts/export_dp_onnx.py` writes is: an encoder graph, a UNet graph, and the
loop HERE, in numpy. That is not a compromise — it is what keeps RTC working. `predict_inpaint`
applies its guidance between denoising steps, so a single unrolled graph would have nowhere to put
it and would quietly degrade to the soft blend RTC exists to replace (see `predict_inpaint` below,
which mirrors `dp_repo_policy.DiffusionPolicyRepoBackend.predict_inpaint` step for step).

Runtime dependencies are numpy + onnxruntime only. No torch, no diffusers, no diffusion_policy
repo — which is the point: the Jetson controller image is Python 3.8 and cannot load any of them.
Everything those libraries would have supplied (the beta schedule, the timestep sequence, the
normalizer's two vectors) is read off the checkpoint at export time and carried in `meta.json`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from evh_controller.policy import ChunkPolicy, resolve_absolute
from evh_controller.rotation import abs7_to_abs10, abs10_to_abs7

logger = logging.getLogger(__name__)

META_FILE = 'meta.json'


def ddim_sample(noise, global_cond, meta: dict, denoise, guide=None, mask=None):
    """Run the DDIM loop in numpy. `denoise(sample, timestep, global_cond) -> epsilon`.

    Shared by `predict` and `predict_inpaint`, and by the exporter's `--check`, so the loop that
    ships is the loop that was verified against diffusers rather than a second copy of it.

    `guide`/`mask` are RTC's soft-masked prefix, in NORMALIZED action units, re-imposed before
    every step and once more at the end — the same placement as the torch backend.
    """
    alphas = np.asarray(meta['alphas_cumprod'], dtype=np.float64)
    final_alpha = float(meta['final_alpha_cumprod'])
    stride = int(meta['num_train_timesteps']) // int(meta['num_inference_steps'])
    clip = bool(meta.get('clip_sample', False))
    clip_range = float(meta.get('clip_sample_range', 1.0))
    if str(meta.get('prediction_type', 'epsilon')) != 'epsilon':
        raise NotImplementedError(
            f"only epsilon-prediction schedulers are supported, meta says "
            f"{meta.get('prediction_type')!r}")

    traj = np.asarray(noise, dtype=np.float32)
    for t in meta['timesteps']:
        if mask is not None:
            traj = mask * guide + (1.0 - mask) * traj
        eps = denoise(traj, int(t), global_cond).astype(np.float64)

        # diffusers DDIMScheduler.step with eta=0 and use_clipped_model_output=False
        prev_t = int(t) - stride
        a_t = alphas[int(t)]
        a_prev = alphas[prev_t] if prev_t >= 0 else final_alpha
        pred_x0 = (traj - np.sqrt(1.0 - a_t) * eps) / np.sqrt(a_t)
        if clip:
            pred_x0 = np.clip(pred_x0, -clip_range, clip_range)
        traj = (np.sqrt(a_prev) * pred_x0 + np.sqrt(1.0 - a_prev) * eps).astype(np.float32)

    if mask is not None:
        traj = (mask * guide + (1.0 - mask) * traj).astype(np.float32)
    return traj


class DiffusionONNXBackend(ChunkPolicy):
    """A diffusion_policy checkpoint exported by scripts/export_dp_onnx.py.

    `weights_path` is the export DIRECTORY (encoder.onnx + unet.onnx + meta.json), not a single
    file — a diffusion policy is two graphs, and pointing at one of them would be a lie about
    what is loaded.
    """

    needs_wrist = True
    guided_resampling = True   # predict_inpaint steers inside the denoising loop

    def __init__(self, export_dir: str, providers: list[str] | None = None,
                 absolute: bool | None = None, denoise_steps: int = 0) -> None:
        self.export_dir = export_dir
        self._absolute_override = absolute
        # placeholders so a graph-less instance still answers the executor's metadata questions
        self.action_dim = 7
        self.chunk_size = 15
        self.n_obs_steps = 2
        self.denoise_steps = denoise_steps
        self._enc = self._unet = None
        self._meta: dict = {}
        self._providers = providers
        self._load(denoise_steps)

    # ------------------------------------------------------------------- load
    def _load(self, denoise_steps: int) -> None:
        if not self.export_dir:
            return

        import onnxruntime as ort

        root = Path(self.export_dir)
        meta_path = root / META_FILE
        if not meta_path.is_file():
            raise FileNotFoundError(
                f'{meta_path} not found — point weights_path at the directory written by '
                'scripts/export_dp_onnx.py, not at a single .onnx file')
        meta = json.loads(meta_path.read_text())
        if meta.get('kind') != 'diffusion_policy_onnx':
            raise ValueError(f'{meta_path} is not a diffusion-policy export (kind='
                             f'{meta.get("kind")!r})')

        if denoise_steps and denoise_steps != int(meta['num_inference_steps']):
            # the timestep sequence is baked in at export time; honouring the parameter here
            # would run the wrong schedule and still look like it worked
            logger.warning(
                'denoise_steps=%d ignored: this export was written for %d steps. Re-export with '
                '--steps %d to change it.', denoise_steps, meta['num_inference_steps'],
                denoise_steps)

        providers = self._providers
        if not providers:
            available = ort.get_available_providers()
            providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider')
                         if p in available] or available[:1]

        self._enc = ort.InferenceSession(str(root / meta['encoder']), providers=providers)
        self._unet = ort.InferenceSession(str(root / meta['unet']), providers=providers)
        active = self._unet.get_providers()
        if 'CUDAExecutionProvider' not in active:
            logger.warning(
                'DP ONNX running on %s, not CUDA — a 256M-param UNet x %d steps on CPU is not a '
                'control loop (see scripts/bench_onnx.py)', active, meta['num_inference_steps'])

        self._meta = meta
        self.n_obs_steps = int(meta['n_obs_steps'])
        self.chunk_size = int(meta['chunk_size'])
        self.action_dim = int(meta['action_dim'])
        self.denoise_steps = int(meta['num_inference_steps'])
        self._horizon = int(meta['horizon'])
        self._raw_action_dim = int(meta['raw_action_dim'])
        self._image_hw = tuple(int(v) for v in meta['image_shape'][1:])
        self._action_scale = np.asarray(meta['action_scale'], dtype=np.float32)
        self._action_offset = np.asarray(meta['action_offset'], dtype=np.float32)
        self.absolute_actions = resolve_absolute(
            self._absolute_override, meta.get('absolute_actions'), str(root))

        logger.info(
            'Loaded DP ONNX policy from %s: chunk_size=%d action_dim=%d n_obs=%d steps=%d '
            'absolute=%s providers=%s',
            root, self.chunk_size, self.action_dim, self.n_obs_steps, self.denoise_steps,
            self.absolute_actions, active)

    # -------------------------------------------------------------------- obs
    @staticmethod
    def _history(arr: np.ndarray, to: int) -> np.ndarray:
        """Left-pad by repeating the oldest frame; keep the newest `to`. Mirrors the torch path."""
        if arr.ndim == 1 or (arr.ndim == 3 and arr.shape[-1] == 3):
            arr = arr[None]                       # a bare frame is a history of one
        if len(arr) >= to:
            return arr[-to:]
        return np.concatenate([np.repeat(arr[:1], to - len(arr), axis=0), arr], axis=0)

    def _images(self, frames: np.ndarray) -> np.ndarray:
        """[To, H, W, 3] uint8 -> [To, 3, h, w] float in [0, 1] (the graph normalizes further)."""
        stack = self._history(np.asarray(frames), self.n_obs_steps)
        if stack.shape[1:3] != self._image_hw:
            import cv2
            h, w = self._image_hw
            stack = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
                              for f in stack])
        return np.ascontiguousarray(
            stack.astype(np.float32).transpose(0, 3, 1, 2) / 255.0)

    def _feeds(self, obs: dict) -> dict:
        prop = self._history(np.asarray(obs['proprio'], dtype=np.float32), self.n_obs_steps)
        return {
            'agentview': self._images(obs['agentview']),
            'wrist': self._images(obs['wrist']),
            'eef_pos': np.ascontiguousarray(prop[:, 0:3]),
            'eef_quat': np.ascontiguousarray(prop[:, 3:7]),
            'gripper': np.ascontiguousarray(prop[:, 7:9]),
        }

    def _denoise(self, sample, timestep, global_cond):
        return self._unet.run(['noise_pred'], {
            'sample': np.ascontiguousarray(sample, dtype=np.float32),
            'timestep': np.asarray([timestep], dtype=np.int64),
            'global_cond': global_cond})[0]

    def _unnormalize(self, traj: np.ndarray) -> np.ndarray:
        """Normalized trajectory -> the chunk the executor consumes, in the 7-dim contract."""
        action_pred = (traj[0] - self._action_offset) / self._action_scale
        chunk = np.asarray(action_pred[self.n_obs_steps - 1:], dtype=np.float32)
        return abs10_to_abs7(chunk) if self.absolute_actions else chunk

    def _zeros(self) -> np.ndarray:
        return np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)

    # ---------------------------------------------------------------- predict
    def predict(self, obs: dict) -> np.ndarray:
        if self._unet is None:
            return self._zeros()
        global_cond = self._enc.run(['global_cond'], self._feeds(obs))[0]
        noise = np.random.randn(1, self._horizon, self._raw_action_dim).astype(np.float32)
        traj = ddim_sample(noise, global_cond, self._meta, self._denoise)
        return self._unnormalize(traj)

    def predict_inpaint(self, obs: dict, prefix: np.ndarray,
                        weights: np.ndarray) -> np.ndarray:
        """RTC guided sampling, applied inside the loop — see the module docstring.

        Identical placement to the torch backend: the prefix is imposed before every denoising
        step and once more at the end, so the sample is drawn coherent with it rather than
        corrected into it afterwards.
        """
        if self._unet is None:
            return self._zeros()
        if prefix is None or len(prefix) == 0 or weights is None or len(weights) == 0:
            return self.predict(obs)

        start = self.n_obs_steps - 1
        guide_raw = (abs7_to_abs10(np.asarray(prefix, dtype=np.float32))
                     if self.absolute_actions else np.asarray(prefix, dtype=np.float32))
        n = min(len(guide_raw), len(weights), self._horizon - start)

        guide = np.zeros((1, self._horizon, self._raw_action_dim), dtype=np.float32)
        mask = np.zeros((1, self._horizon, 1), dtype=np.float32)
        guide[0, start:start + n] = (
            guide_raw[:n] * self._action_scale + self._action_offset)
        mask[0, start:start + n, 0] = np.clip(
            np.asarray(weights[:n], dtype=np.float32), 0.0, 1.0)

        global_cond = self._enc.run(['global_cond'], self._feeds(obs))[0]
        noise = np.random.randn(1, self._horizon, self._raw_action_dim).astype(np.float32)
        traj = ddim_sample(noise, global_cond, self._meta, self._denoise,
                           guide=guide, mask=mask)
        return self._unnormalize(traj)
