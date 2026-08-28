"""Tests for the plant's mp4 recorder — pure Python, no ROS and no real imageio writer.

The recorder is a debugging convenience, but two of its behaviours matter to the node: an
unconfigured recorder must be a working no-op object (so the observation publisher never branches
on whether recording is on), and close() must be idempotent (record() closes the file itself at
max_frames, and destroy_node() closes it again).
"""
import sys
import types

import numpy as np
import pytest

from evh_plant.video import VideoRecorder


class FakeWriter:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, frame):
        self.frames.append(frame)

    def close(self):
        self.closed = True


@pytest.fixture()
def fake_imageio(monkeypatch):
    """Stub the imageio module so no file is ever written (and the host needs no imageio)."""
    written = []

    def get_writer(path, fps):
        writer = FakeWriter()
        written.append((path, fps, writer))
        return writer

    monkeypatch.setitem(sys.modules, 'imageio', types.SimpleNamespace(get_writer=get_writer))
    return written


def _frame():
    return np.zeros((8, 8, 3), np.uint8)


def test_an_unconfigured_recorder_is_an_inert_object(tmp_path):
    """No video_path: record() and close() must work and do nothing, so the node can call them
    unconditionally on the 20 Hz observation tick."""
    recorder = VideoRecorder('', fps=20)

    assert recorder.active is False
    recorder.record(_frame())
    recorder.close()
    assert recorder.frames == 0


def test_frames_are_written_until_close(fake_imageio, tmp_path):
    recorder = VideoRecorder(str(tmp_path / 'out.mp4'), fps=20)
    for _ in range(3):
        recorder.record(_frame())
    recorder.close()

    _path, fps, writer = fake_imageio[0]
    assert fps == 20
    assert len(writer.frames) == 3
    assert recorder.frames == 3
    assert writer.closed is True


def test_recording_stops_itself_at_max_frames(fake_imageio, tmp_path):
    recorder = VideoRecorder(str(tmp_path / 'out.mp4'), fps=20, max_frames=2)
    for _ in range(5):
        recorder.record(_frame())

    _path, _fps, writer = fake_imageio[0]
    assert len(writer.frames) == 2, 'kept writing past the limit'
    assert writer.closed is True
    assert recorder.active is False


def test_close_is_idempotent(fake_imageio, tmp_path):
    """record() closes at max_frames and destroy_node() closes again — the second must not raise
    (nor log a second 'wrote N frames')."""
    recorder = VideoRecorder(str(tmp_path / 'out.mp4'), fps=20, max_frames=1)
    recorder.record(_frame())
    recorder.close()
    recorder.close()


@pytest.mark.parametrize('duration,fps,expected', [
    (0.0, 20.0, 0),        # 0 = until shutdown
    (5.0, 20.0, 100),
    (2.5, 20.0, 50),
])
def test_duration_converts_to_a_frame_budget(fake_imageio, tmp_path, duration, fps, expected):
    recorder = VideoRecorder.from_duration(str(tmp_path / 'out.mp4'), fps, duration)
    assert recorder.max_frames == expected


def test_from_duration_without_a_path_stays_inert(tmp_path):
    assert VideoRecorder.from_duration('  ', 20.0, 5.0).active is False
