"""Headless mp4 recording of the agentview camera — a debugging convenience, not part of the loop.

Split out of `plant_node.py` because it has nothing to do with the experiment: no episode, action,
or network semantics touch it. Keeping it here means the node's observation publisher is one line
(`self.recorder.record(frame)`) instead of three lifecycle methods and two counters.

An inactive recorder (no path configured) is a real object whose `record` is a no-op, so the node
never branches on whether recording is on. No ROS imports.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Writes frames to an mp4, optionally stopping after a fixed duration."""

    def __init__(self, path: str, fps: int, max_frames: int = 0) -> None:
        """`path` empty disables recording entirely; `max_frames` 0 records until close()."""
        self.path = str(path).strip()
        self.fps = max(1, int(fps))
        self.max_frames = max(0, int(max_frames))
        self.frames = 0
        self._writer = None
        if self.path:
            self._open()

    @classmethod
    def from_duration(cls, path: str, fps: float, duration_s: float) -> VideoRecorder:
        """Build one from a wall-clock duration; 0 s means 'until shutdown'."""
        fps_i = max(1, int(fps))
        return cls(path, fps_i, int(duration_s * fps_i) if duration_s > 0.0 else 0)

    @property
    def active(self) -> bool:
        return self._writer is not None

    def _open(self) -> None:
        import imageio

        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(str(path), fps=self.fps)
        limit = f'max {self.max_frames} frames' if self.max_frames else 'until shutdown'
        logger.info('recording %s @ %d Hz, %s', path, self.fps, limit)

    def record(self, frame: np.ndarray) -> None:
        """Append one frame; closes the file once max_frames is reached. No-op when inactive."""
        if self._writer is None:
            return
        self._writer.append_data(frame)
        self.frames += 1
        if self.max_frames and self.frames >= self.max_frames:
            self.close()

    def close(self) -> None:
        """Finalise the file. Idempotent — record() may already have closed it at max_frames."""
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None
        logger.info('wrote %d frames to %s', self.frames, self.path)
