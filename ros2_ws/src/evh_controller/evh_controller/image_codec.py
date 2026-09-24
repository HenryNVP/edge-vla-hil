"""Read an observation image off the wire, whether it arrived raw or JPEG-compressed.

The plant's `evh_plant.messages` holds the ENCODER; this holds the decoder. They are deliberately
in different packages rather than shared: `evh_controller` is what ships to the Jetson and must not
depend on the simulator package. Duplication is the established pattern here (`phase.py`,
`MODE_QOS`, `EEF_TO_CONTROL_QUAT`), and `test_mode_crosscheck.py` pins the two halves in agreement
— including a round trip through a real JPEG, because the one way this fails is silent.

That failure is the channel order. cv2 works in BGR and the policy is trained on RGB, so an
encoder and decoder that disagree produce perfectly plausible images with red and blue exchanged,
which no shape check and no topic inspection would catch.
"""
from __future__ import annotations

import numpy as np

RAW_QUALITY = 0          # must match evh_plant.messages.RAW_QUALITY
JPEG_FORMAT = 'jpeg'


def decode_image(msg) -> np.ndarray:
    """Upright RGB uint8 [H, W, 3] from either a sensor_msgs/Image or a CompressedImage.

    Dispatches on the message having a `format` field rather than on a configured mode, so a topic
    whose type changed cannot be read with the wrong unpacking.
    """
    if hasattr(msg, 'format'):
        import cv2  # only on the compressed path; the fast suite imports this module without it
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f'could not decode a {len(msg.data)}-byte {msg.format!r} frame')
        return np.ascontiguousarray(frame[:, :, ::-1])
    return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)


def image_msg_type(quality: int) -> str:
    """The ROS type name the observation image topics carry at this quality setting."""
    return ('sensor_msgs/msg/Image' if int(quality) <= RAW_QUALITY
            else 'sensor_msgs/msg/CompressedImage')
