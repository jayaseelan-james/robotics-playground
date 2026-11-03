"""
Joint-space PID and Cartesian-space PID control

Joint space: tau = Kp (q_d - q) + Kd (dq_d - dq) + Ki * integral(q_d - q) + g(q)

Cartesian:   F   = Kp_pos e_x + Kd_pos (v_d - v) + Ki_pos * integral(e_x) (force   [N])
             M   = Kp_rot e_R + Kd_rot (w_d - w) + Ki_rot * integral(e_R) (moment  [Nm])
             tau = J(q)^T [F; M] + g(q)
where ``e_x = x_d - x`` is the position error, ``e_R`` the orientation error from
``transforms.orientation_error``, and ``v, w = J(q) dq`` the measured TCP twist.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import DT, N_JOINTS, TORQUE_LIMITS
from .transforms import orientation_error


def _as_vec(x, n: int = N_JOINTS) -> np.ndarray:
    """Broadcast a scalar / array to shape (n,)."""
    return np.broadcast_to(np.asarray(x, dtype=float), (n,)).copy()


class JointSpacePIDController:
    """Joint-space controller with gravity compensation and an optional integral term."""

    def __init__(self, kp, kd, ki=0.0, torque_limits=TORQUE_LIMITS, dt: float = DT, integral_limit: float = 0.5):
        self.kp, self.kd, self.ki = _as_vec(kp), _as_vec(kd), _as_vec(ki)
        self.torque_limits = _as_vec(torque_limits)
        self.dt = float(dt)
        self.integral_limit = float(integral_limit)
        self.reset()

    def reset(self) -> None:
        """Clear the integrator state. Called at the start of every rollout."""
        self.error_integral = np.zeros(N_JOINTS)

    def compute_torque(self, q: np.ndarray, dq: np.ndarray, q_des: np.ndarray,
                       dq_des: Optional[np.ndarray] = None,
                       gravity: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the joint torques for one control step.

        Returns:
        tau : (7,) torques, clipped to ``[-torque_limits, +torque_limits]``.
        """
        dq_des = np.zeros(N_JOINTS) if dq_des is None else dq_des
        gravity = np.zeros(N_JOINTS) if gravity is None else gravity
        e = q_des - q
        de = dq_des - dq
        self.error_integral = np.clip(self.error_integral + e * self.dt,
                                      -self.integral_limit, self.integral_limit)
        tau = self.kp * e + self.kd * de + self.ki * self.error_integral + gravity
        return np.clip(tau, -self.torque_limits, self.torque_limits)


class CartesianPIDController:
    """Cartesian-space controller (Jacobian transpose) with gravity compensation."""

    def __init__(self, kp_pos, kd_pos, kp_rot, kd_rot, ki_pos=0.0, ki_rot=0.0,
                 kp_null=0.0, kd_null=0.0, q_null=None, torque_limits=TORQUE_LIMITS,
                 dt: float = DT, integral_limit_pos: float = 0.05, integral_limit_rot: float = 0.2):
        self.kp_pos, self.kd_pos, self.ki_pos = float(kp_pos), float(kd_pos), float(ki_pos)
        self.kp_rot, self.kd_rot, self.ki_rot = float(kp_rot), float(kd_rot), float(ki_rot)
        self.kp_null, self.kd_null = float(kp_null), float(kd_null)
        self.q_null = None if q_null is None else np.asarray(q_null, dtype=float)
        self.torque_limits = _as_vec(torque_limits)
        self.dt = float(dt)
        self.integral_limit_pos = float(integral_limit_pos)
        self.integral_limit_rot = float(integral_limit_rot)
        self.reset()

    def reset(self) -> None:
        """Clear the integrator states. Called at the start of every rollout."""
        self.integral_pos = np.zeros(3)   # integral of (x_des - x) [m s]
        self.integral_rot = np.zeros(3)   # integral of orientation_error [rad s]

    def compute_torque(self, q, dq, J, x, R, x_des, R_des, v_des=None, w_des=None,
                       gravity=None) -> np.ndarray:
        """Compute the joint torques for one control step.

        Returns:
        tau : (7,) torques, clipped to ``[-torque_limits, +torque_limits]``.
        """
        v_des = np.zeros(3) if v_des is None else v_des
        w_des = np.zeros(3) if w_des is None else w_des
        gravity = np.zeros(N_JOINTS) if gravity is None else gravity

        e_pos = x_des - x
        e_rot = orientation_error(R_des, R)
        twist = J @ dq
        F = self.kp_pos * e_pos + self.kd_pos * (v_des - twist[:3])
        M = self.kp_rot * e_rot + self.kd_rot * (w_des - twist[3:])

        self.integral_pos = np.clip(self.integral_pos + e_pos * self.dt,
                                    -self.integral_limit_pos, self.integral_limit_pos)
        self.integral_rot = np.clip(self.integral_rot + e_rot * self.dt,
                                    -self.integral_limit_rot, self.integral_limit_rot)
        F = F + self.ki_pos * self.integral_pos
        M = M + self.ki_rot * self.integral_rot

        tau = J.T @ np.concatenate([F, M]) + gravity

        if (self.kp_null > 0.0 or self.kd_null > 0.0) and self.q_null is not None:
            tau0 = self.kp_null * (self.q_null - q) - self.kd_null * dq
            N = np.eye(N_JOINTS) - J.T @ np.linalg.pinv(J.T)
            tau = tau + N @ tau0

        return np.clip(tau, -self.torque_limits, self.torque_limits)
