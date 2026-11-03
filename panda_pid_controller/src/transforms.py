"""
Coordinate systems and rotations

Conventions
-----------
* Rotations are *active* and right-handed. ``rot_z(+90 deg)`` maps the x axis onto the y axis.
* Quaternions are ``[w, x, y, z]`` (MuJoCo order).
* A *rotation vector* (axis-angle) ``phi = angle * axis`` encodes ``R = Exp(phi)```.
"""
from __future__ import annotations

import numpy as np


def rot_x(angle: float) -> np.ndarray:
    """Rotation matrix for a rotation of ``angle`` [rad] about the x axis. Shape (3, 3)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    """Rotation matrix for a rotation of ``angle`` [rad] about the y axis. Shape (3, 3)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    """Rotation matrix for a rotation of ``angle`` [rad] about the z axis. Shape (3, 3)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Assemble the 4x4 homogeneous transform from a rotation ``R`` (3x3) and position ``p`` (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid-body transform, **without** calling ``np.linalg.inv``."""
    R, p = T[:3, :3], T[:3, 3]
    return make_transform(R.T, -R.T @ p)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply ``T`` to one point (shape (3,)) or many points (shape (N, 3)).

    The output has the same shape as the input. Example: if ``T = T_world_ee`` and ``points``
    are given in the end-effector frame, the result is expressed in the world frame.
    """
    points = np.asarray(points, dtype=float)
    return points @ T[:3, :3].T + T[:3, 3]


def skew(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix ``[w]_x`` such that ``skew(w) @ v == np.cross(w, v)``. Shape (3, 3)."""
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def vee(S: np.ndarray) -> np.ndarray:
    """Inverse of :func:`skew`: extract ``w`` (3,) from a skew-symmetric matrix."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def axis_angle_to_rotmat(rotvec: np.ndarray) -> np.ndarray:
    """Exponential map ``Exp(phi)``: rotation vector (3,) -> rotation matrix (3, 3).

    Use Rodrigues' formula ``R = I + sin(t) K + (1 - cos(t)) K^2`` with ``t = |phi|`` and
    ``K = skew(phi / t)``. Handle ``t ~ 0`` gracefully (return ``I + skew(phi)``).
    """
    rotvec = np.asarray(rotvec, dtype=float)
    theta = np.linalg.norm(rotvec)
    if theta < 1e-9:
        return np.eye(3) + skew(rotvec)
    K = skew(rotvec / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Logarithmic map ``Log(R)``: rotation matrix (3, 3) -> rotation vector (3,), angle in [0, pi]."""
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    w = vee(R - R.T)
    if theta < 1e-8:
        return w / 2.0
    if np.pi - theta < 1e-5:  # near the singularity: axis from the symmetric part (R + I)/2 = k k^T
        B = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(B)))
        k = B[:, i] / np.sqrt(B[i, i])
        return theta * k * (1.0 if k @ w >= 0 else -1.0)
    return theta / (2.0 * np.sin(theta)) * w


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Unit quaternion ``[w, x, y, z]`` -> rotation matrix (3, 3). Normalise the input first."""
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) -> unit quaternion ``[w, x, y, z]`` with ``w >= 0``.

    ``q`` and ``-q`` describe the same rotation; we fix the sign by requiring ``w >= 0``.
    Be careful with numerical robustness: do not divide by a (nearly) zero ``w``.
    """
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def orientation_error(R_des: np.ndarray, R: np.ndarray) -> np.ndarray:
    """World-frame rotation vector that rotates the current orientation ``R`` onto ``R_des``.

    Both matrices are expressed in the world frame (``R_des = R_world_desired``,
    ``R = R_world_current``). The result ``e`` satisfies ``Exp(e) @ R == R_des``.
    """
    return rotmat_to_axis_angle(R_des @ R.T)
