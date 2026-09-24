"""The plant's observation contract: robosuite obs dict in, ROS messages out.

Split out of `plant_node.py` so the contract other packages depend on lives somewhere you can read
top to bottom, instead of being interleaved with publisher calls. Two things are load-bearing:

  * Quaternions are `[x, y, z, w]` (robosuite / geometry_msgs order) everywhere.
  * `/obs/proprio` is `[eef_pos(3), eef_quat(4, xyzw), gripper_qpos(2)]` — the layout the policy's
    observation dict is built from. Changing the order here silently feeds the policy garbage.

`PlantObservation.from_robosuite` takes an obs dict the caller has ALREADY snapshotted into a
local. That is deliberate: the plant's 200 Hz physics thread rebinds `self._obs` underneath the
20 Hz publisher, so reading field-by-field off the attribute would pair an image with proprio from
a different sim step. Taking the dict as an argument makes the snapshot structural rather than a
comment someone has to honour.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, Image, JointState

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)   # [x, y, z, w]

# JPEG quality for the compressed observation path. 0 means "send raw", which is the default
# because compression is LOSSY: it changes what the policy consumes, not just what the link
# carries, so enabling it for a measured run is a change to the observation and needs its own
# validation. Raw 84x84x3 is 21168 B per frame; two of those at 20 Hz is 6.8 Mbit/s in ~640 UDP
# datagrams/s, which measurably saturates a WiFi link (see scripts/wifi_trace.py). q80 is ~2.6 KB.
RAW_QUALITY = 0
JPEG_FORMAT = 'jpeg'


def upright(frame) -> np.ndarray | None:
    """robosuite camera obs are bottom-up; flip to upright uint8."""
    if frame is None:
        return None
    return np.flipud(np.asarray(frame)).astype(np.uint8).copy()


@dataclass
class PlantObservation:
    """One coherent sim step's worth of observation, ready to publish."""
    frame: np.ndarray                  # agentview, upright uint8 [H, W, 3]
    joint_pos: np.ndarray              # [7]
    joint_vel: np.ndarray              # [7]
    ee_pos: np.ndarray                 # [3]
    ee_quat: np.ndarray                # [4] xyzw
    gripper_qpos: np.ndarray           # [2]
    wrist: np.ndarray | None = None    # second camera, when one is configured

    @classmethod
    def from_robosuite(cls, obs: dict, cameras: list[str],
                       want_wrist: bool) -> PlantObservation | None:
        """Build from a snapshotted obs dict. None when it carries no image (nothing to publish)."""
        frame = upright(obs.get(f'{cameras[0]}_image'))
        if frame is None:
            return None   # obs dict without images (shouldn't happen with use_camera_obs)
        wrist = upright(obs.get(f'{cameras[1]}_image')) if want_wrist and len(cameras) > 1 else None
        return cls(
            frame=frame,
            wrist=wrist,
            joint_pos=np.asarray(obs.get('robot0_joint_pos', np.zeros(7)), dtype=float),
            joint_vel=np.asarray(obs.get('robot0_joint_vel', np.zeros(7)), dtype=float),
            ee_pos=np.asarray(obs.get('robot0_eef_pos', np.zeros(3)), dtype=float),
            ee_quat=np.asarray(obs.get('robot0_eef_quat', IDENTITY_QUAT), dtype=float),
            gripper_qpos=np.asarray(obs.get('robot0_gripper_qpos', np.zeros(2)), dtype=float),
        )

    @classmethod
    def synthetic(cls, image_size: int, want_wrist: bool) -> PlantObservation:
        """Noise stand-in for the robosuite-less degraded mode: the graph still runs, no physics."""
        frame = np.random.randint(0, 255, (image_size, image_size, 3), np.uint8)
        return cls(
            frame=frame,
            wrist=frame if want_wrist else None,
            joint_pos=np.zeros(7), joint_vel=np.zeros(7),
            ee_pos=np.zeros(3), ee_quat=np.array(IDENTITY_QUAT), gripper_qpos=np.zeros(2),
        )

    # ----------------------------------------------------------------- packing
    def image_msg(self, stamp, quality: int = RAW_QUALITY):
        return _pack_frame(self.frame, stamp, quality)

    def wrist_msg(self, stamp, quality: int = RAW_QUALITY):
        return None if self.wrist is None else _pack_frame(self.wrist, stamp, quality)

    def joint_state_msg(self, stamp) -> JointState:
        msg = JointState()
        msg.header.stamp = stamp
        msg.position = list(self.joint_pos)
        msg.velocity = list(self.joint_vel)
        return msg

    def proprio_msg(self, stamp) -> JointState:
        """[eef_pos(3), eef_quat(4, xyzw), gripper_qpos(2)] — the policy's proprio contract."""
        msg = JointState()
        msg.header.stamp = stamp
        msg.position = list(self.ee_pos) + list(self.ee_quat) + list(self.gripper_qpos)
        return msg

    def ee_pose_msg(self, stamp, frame_id: str = 'base') -> PoseStamped:
        """The reactive layer's zero-delay local anchor — never routed through the relay."""
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, self.ee_pos)
        (msg.pose.orientation.x, msg.pose.orientation.y,
         msg.pose.orientation.z, msg.pose.orientation.w) = map(float, self.ee_quat)
        return msg


def to_image_msg(frame: np.ndarray, stamp) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.height, msg.width = frame.shape[0], frame.shape[1]
    msg.encoding = 'rgb8'
    msg.step = msg.width * 3
    msg.data = frame.tobytes()
    return msg


def _pack_frame(frame: np.ndarray, stamp, quality: int):
    """Raw Image at quality 0, JPEG CompressedImage otherwise."""
    return (to_image_msg(frame, stamp) if int(quality) <= RAW_QUALITY
            else to_compressed_image_msg(frame, stamp, quality))


def to_compressed_image_msg(frame: np.ndarray, stamp, quality: int) -> CompressedImage:
    """JPEG-encode an upright RGB frame. `format` is 'jpeg' so any ROS tool can read it.

    cv2 works in BGR, so the channels are swapped on the way in and back on the way out
    (`decode_image`); getting that wrong is silent — the policy sees plausible images with red and
    blue exchanged.
    """
    import cv2  # only needed on the compressed path; keeps the fast suite importable without it
    ok, buf = cv2.imencode('.jpg', frame[:, :, ::-1],
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError(f'JPEG encode failed for a {frame.shape} frame')
    msg = CompressedImage()
    msg.header.stamp = stamp
    msg.format = JPEG_FORMAT
    msg.data = buf.tobytes()
    return msg


def decode_image(msg) -> np.ndarray:
    """Upright RGB uint8 [H, W, 3] from either an Image or a CompressedImage.

    One decoder for both so the consumer does not branch on the transport, and so a topic that
    changed type cannot be read with the wrong unpacking.
    """
    if hasattr(msg, 'format'):
        import cv2
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f'could not decode a {len(msg.data)}-byte {msg.format!r} frame')
        return np.ascontiguousarray(frame[:, :, ::-1])
    return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)


def image_msg_type(quality: int) -> str:
    """The ROS type name the observation image topics carry at this quality setting."""
    return ('sensor_msgs/msg/Image' if int(quality) <= RAW_QUALITY
            else 'sensor_msgs/msg/CompressedImage')
