"""Pluggable chunk-policy backends (diffusion / flow-matching).

We use a small diffusion/flow policy (not deterministic ACT) so that RTC and BID — which rely on
guided denoising / resampling — apply as first-class baselines. ACT+Temporal-Ensembling remains a
deterministic baseline elsewhere.

Two interchangeable inference paths behind one interface so the rest of the system never changes:

  * PyTorchBackend  -- diffusion/flow checkpoint, runs anywhere (dev + Jetson fallback). Built
                       FIRST so a TensorRT stall never blocks downstream phases.
  * TensorRTBackend -- serialized .engine built from an exported ONNX policy (Jetson fast path).

The interface exposes BOTH plain chunk prediction and *inpainting* prediction:

  predict(image, state)                       -> action chunk [H, A]
  predict_inpaint(image, state, prefix, w)    -> action chunk [H, A], guided so the first len(w)
                                                 entries stay close to `prefix` with weights `w`

`predict_inpaint` is what the RTC strategy needs (freeze-d + soft-masked guidance). The plain
backends return zeros today (stub), but the contract is fixed so executors can be written against
it now. Actions are end-effector (Cartesian) pose deltas (OSC_POSE convention).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class ChunkPolicy(ABC):
    """obs (image HWC uint8 + state vector) -> action chunk [chunk_size, action_dim]."""

    action_dim: int
    chunk_size: int
    denoise_steps: int

    @abstractmethod
    def predict(self, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        ...

    def predict_inpaint(self, image: np.ndarray, state: np.ndarray,
                        prefix: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Guided generation with a soft-masked prefix (for RTC).

        Default: plain predict then hard-overwrite the frozen prefix. A real flow backend should
        instead inject `prefix`/`weights` into the denoising guidance (see RTCExecutor and the RTC
        paper's W_i masking). Override per backend.
        """
        chunk = self.predict(image, state)
        k = min(len(prefix), chunk.shape[0])
        chunk[:k] = prefix[:k]   # crude freeze; TODO real guided inpainting
        return chunk


class PyTorchBackend(ChunkPolicy):
    def __init__(self, ckpt_path: str, device: str = 'cuda', denoise_steps: int = 5) -> None:
        self.ckpt_path = ckpt_path
        self.device = device
        self.action_dim = 7          # OSC_POSE: 6-DoF EE delta + gripper
        self.chunk_size = 16
        self.denoise_steps = denoise_steps
        self._model = None
        self._load()

    def _load(self) -> None:
        """TODO: load a small diffusion/flow policy (e.g. Diffusion Policy on RoboMimic).

        Few-step / consistency sampling keeps the Orin Nano honest. Expose the denoiser so
        predict_inpaint can do guided sampling rather than the crude freeze fallback.
        """
        self._model = None  # STUB

    def predict(self, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)  # STUB
        raise NotImplementedError


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

    def predict(self, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        if self._engine is None:
            return np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)  # STUB
        raise NotImplementedError


def make_policy(backend: str, weights_path: str) -> ChunkPolicy:
    backend = backend.lower()
    if backend in ('pytorch', 'torch', 'fallback'):
        return PyTorchBackend(weights_path)
    if backend in ('tensorrt', 'trt'):
        return TensorRTBackend(weights_path)
    raise ValueError(f'unknown backend: {backend!r}')
