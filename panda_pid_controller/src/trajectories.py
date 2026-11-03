"""
Trajectory planning

A controller needs a reference that it can actually follow. A step in the target pose would
demand infinite acceleration and saturate the motors. Here you generate smooth point-to-point
motions in joint space and in Cartesian space.

    Joint space :  q(t) = q0 + s(t) (q1 - q0)
    Cartesian   :  p(t) = p0 + s(t) (p1 - p0)         (straight line)
                   R(t) = R0 Exp( s(t) Log(R0^T R1) ) (shortest rotation, constant axis)
"""
from __future__ import annotations

import numpy as np

from .transforms import axis_angle_to_rotmat, rotmat_to_axis_angle


def quintic_scaling(t: float, T: float):
    """Quintic time scaling ``s(t)`` with ``s(0)=0, s(T)=1`` and zero first/second derivative
    at both ends.

    Returns ``(s, ds, dds)``: the value and its first and second derivative **with respect to
    time** [1/s, 1/s^2]. For ``t < 0`` the motion has not started (``s = 0``), for ``t > T`` it is
    finished (``s = 1``); derivatives are 0 outside ``[0, T]``.
    """
    tau = float(np.clip(t / T, 0.0, 1.0))
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    ds = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / T
    dds = (60 * tau - 180 * tau ** 2 + 120 * tau ** 3) / T ** 2
    return s, ds, dds


class JointTrajectory:
    """Straight-line (in joint space) motion from ``q_start`` to ``q_goal`` in ``duration`` s."""

    def __init__(self, q_start: np.ndarray, q_goal: np.ndarray, duration: float):
        self.q_start = np.asarray(q_start, dtype=float)
        self.q_goal = np.asarray(q_goal, dtype=float)
        self.duration = float(duration)

    def evaluate(self, t: float):
        """Return ``(q, dq, ddq)``: desired joint positions, velocities, accelerations (each (7,)).

        After ``duration`` the trajectory stays at ``q_goal`` with zero velocity.
        """
        s, ds, dds = quintic_scaling(t, self.duration)
        delta = self.q_goal - self.q_start
        return self.q_start + s * delta, ds * delta, dds * delta


class CartesianLineTrajectory:
    """Straight-line motion of the TCP from pose ``T_start`` to pose ``T_goal`` (both 4x4,
    expressed in the world frame). Position moves on a line, orientation rotates about a
    single fixed axis by the shortest angle."""

    def __init__(self, T_start: np.ndarray, T_goal: np.ndarray, duration: float):
        self.duration = float(duration)
        self.p0, self.p1 = T_start[:3, 3].copy(), T_goal[:3, 3].copy()
        self.R0, self.R1 = T_start[:3, :3].copy(), T_goal[:3, :3].copy()
        # relative rotation expressed in the START frame, as a rotation vector
        self.phi = rotmat_to_axis_angle(self.R0.T @ self.R1)

    def evaluate(self, t: float):
        """Return ``(pos (3,), R (3,3), vel (3,), omega (3,))`` at time ``t``, in world frame."""
        s, ds, _ = quintic_scaling(t, self.duration)
        pos = self.p0 + s * (self.p1 - self.p0)
        R = self.R0 @ axis_angle_to_rotmat(s * self.phi)
        vel = ds * (self.p1 - self.p0)
        omega = ds * (self.R0 @ self.phi)
        return pos, R, vel, omega


class CircleTrajectory:
    """Circle in the world xy-plane (constant height and orientation).

    The TCP starts at ``start_pos`` and traverses one full revolution in ``duration`` seconds,
    with a smooth (quintic) speed profile: the circle's centre is ``start_pos - [radius, 0, 0]``.
    """

    def __init__(self, start_pos: np.ndarray, R: np.ndarray, radius: float, duration: float):
        self.start_pos = np.asarray(start_pos, dtype=float)
        self.R, self.radius, self.duration = np.asarray(R, dtype=float), float(radius), float(duration)

    def evaluate(self, t: float):
        tau = float(np.clip(t / self.duration, 0.0, 1.0))
        s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        ds = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / self.duration
        psi, dpsi = 2 * np.pi * s, 2 * np.pi * ds
        r = self.radius
        pos = self.start_pos + r * np.array([np.cos(psi) - 1.0, np.sin(psi), 0.0])
        vel = r * dpsi * np.array([-np.sin(psi), np.cos(psi), 0.0])
        return pos, self.R, vel, np.zeros(3)
