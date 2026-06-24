"""Pluggable ACT policy backends.

Two interchangeable inference paths behind one interface so the rest of the system never changes:

  * PyTorchBackend  -- LeRobot ACT checkpoint, runs anywhere (dev + Jetson fallback). Built FIRST
                       so a TensorRT stall never blocks downstream phases (see proposal Phase 3).
  * TensorRTBackend -- serialized .engine built from an exported ONNX policy. The Jetson fast path.

ACT predicts an action *chunk* (a horizon of future actions). The controller node consumes the
chunk; temporal ensembling / chunk bookkeeping lives there, not here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class ActionPolicy(ABC):
    """obs (image HWC uint8 + joint-state vector) -> action chunk [horizon, action_dim]."""

    action_dim: int
    chunk_size: int

    @abstractmethod
    def predict(self, image: np.ndarray, joint_state: np.ndarray) -> np.ndarray:
        ...


class PyTorchBackend(ActionPolicy):
    def __init__(self, ckpt_path: str, device: str = 'cuda') -> None:
        self.ckpt_path = ckpt_path
        self.device = device
        self.action_dim = 7
        self.chunk_size = 100
        self._model = None
        self._load()

    def _load(self) -> None:
        """TODO: load a LeRobot ACT policy.

            from lerobot.common.policies.act.modeling_act import ACTPolicy
            self._model = ACTPolicy.from_pretrained(self.ckpt_path).to(self.device).eval()
        """
        # STUB: no checkpoint yet — emit zero chunks so the graph runs.
        self._model = None

    def predict(self, image: np.ndarray, joint_state: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)  # STUB
        # TODO: normalize -> torch tensors -> self._model.select_action / forward -> numpy
        raise NotImplementedError


class TensorRTBackend(ActionPolicy):
    def __init__(self, engine_path: str) -> None:
        self.engine_path = engine_path
        self.action_dim = 7
        self.chunk_size = 100
        self._engine = None
        self._load()

    def _load(self) -> None:
        """TODO: deserialize the TRT engine + allocate I/O bindings (pycuda).

        See scripts/build_trt_engine.py for engine construction from ONNX.
        """
        self._engine = None  # STUB

    def predict(self, image: np.ndarray, joint_state: np.ndarray) -> np.ndarray:
        if self._engine is None:
            return np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)  # STUB
        raise NotImplementedError


def make_policy(backend: str, weights_path: str) -> ActionPolicy:
    backend = backend.lower()
    if backend in ('pytorch', 'torch', 'fallback'):
        return PyTorchBackend(weights_path)
    if backend in ('tensorrt', 'trt'):
        return TensorRTBackend(weights_path)
    raise ValueError(f'unknown backend: {backend!r}')
