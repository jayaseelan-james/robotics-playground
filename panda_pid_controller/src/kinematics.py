"""
Forward kinematics and Jacobian of the Franka Panda.

Modified (Craig) DH convention - one link transform is
    T_{i-1,i} = Rot_x(alpha) @ Trans_x(a) @ Rot_z(theta) @ Trans_z(d), theta = q_i.
"""
from __future__ import annotations

import numpy as np

from .config import DH_FLANGE, DH_PARAMS, HAND_ROT_Z, TCP_OFFSET_Z
from .transforms import make_transform, rot_z


def dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """4x4 transform of one link in modified DH convention."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -sa * d],
        [st * sa, ct * sa, ca, ca * d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def joint_frames(q: np.ndarray) -> np.ndarray:
    """Poses of the 7 joint frames in the world frame.

    Returns an array of shape (7, 4, 4); entry ``i`` is ``T_world_{i+1}``.
    The z axis of frame ``i+1`` is the rotation axis of joint ``i+1``.
    """
    frames = np.zeros((7, 4, 4))
    T = np.eye(4)
    for i in range(7):
        a, d, alpha = DH_PARAMS[i]
        T = T @ dh_transform(a, d, alpha, q[i])
        frames[i] = T
    return frames


def forward_kinematics_flange(q: np.ndarray) -> np.ndarray:
    """``T_world_flange`` (4x4): the robot flange, i.e. DH frame 8 (fixed link after joint 7)."""
    a, d, alpha = DH_FLANGE
    return joint_frames(q)[-1] @ dh_transform(a, d, alpha, 0.0)


def flange_to_ee() -> np.ndarray:
    """Fixed transform ``T_flange_ee`` (4x4) from the flange to the TCP.

    The hand is mounted rotated by ``HAND_ROT_Z`` about the flange z axis, and the TCP lies
    ``TCP_OFFSET_Z`` along the (rotated) hand z axis.
    """
    T_flange_hand = make_transform(rot_z(HAND_ROT_Z), np.zeros(3))
    T_hand_ee = make_transform(np.eye(3), np.array([0.0, 0.0, TCP_OFFSET_Z]))
    return T_flange_hand @ T_hand_ee


def forward_kinematics_ee(q: np.ndarray) -> np.ndarray:
    """``T_world_ee`` (4x4): pose of the tool centre point in the world frame."""
    return forward_kinematics_flange(q) @ flange_to_ee()


def geometric_jacobian(q: np.ndarray) -> np.ndarray:
    """Geometric Jacobian of the TCP, shape (6, 7), expressed in the world frame."""
    frames = joint_frames(q)
    p_e = forward_kinematics_ee(q)[:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        z_i = frames[i][:3, 2]
        p_i = frames[i][:3, 3]
        J[:3, i] = np.cross(z_i, p_e - p_i)
        J[3:, i] = z_i
    return J
