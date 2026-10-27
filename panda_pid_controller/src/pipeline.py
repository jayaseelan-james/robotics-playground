"""
Closed-loop simulation.

``run_joint_space`` and ``run_cartesian`` execute the classic sense -> compute -> act cycle at
the simulation rate (500 Hz) and record everything in a :class:`RolloutLog`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .controllers import CartesianPIDController, JointSpacePIDController
from .env import PandaEnv
from .metrics import rotation_angle


@dataclass
class RolloutLog:
    """Time series recorded during a rollout (all arrays have the time axis first)."""
    t: np.ndarray = None          # (N,)
    q: np.ndarray = None          # (N, 7)  measured joint positions
    dq: np.ndarray = None         # (N, 7)  measured joint velocities
    tau: np.ndarray = None        # (N, 7)  torques returned by YOUR controller
    q_des: np.ndarray = None      # (N, 7)  desired joint positions   (joint-space runs)
    ee_pos: np.ndarray = None     # (N, 3)  measured TCP position (MuJoCo ground truth)
    ee_R: np.ndarray = None       # (N, 3, 3) measured TCP orientation
    x_des: np.ndarray = None      # (N, 3)  desired TCP position      (Cartesian runs)
    R_des: np.ndarray = None      # (N, 3, 3) desired TCP orientation (Cartesian runs)

    def joint_error(self) -> np.ndarray:
        """q_des - q, shape (N, 7)."""
        return self.q_des - self.q

    def position_error(self) -> np.ndarray:
        """Euclidean TCP position error [m], shape (N,)."""
        return np.linalg.norm(self.x_des - self.ee_pos, axis=1)

    def orientation_error(self) -> np.ndarray:
        """Geodesic TCP orientation error [rad], shape (N,)."""
        return np.array([rotation_angle(Rd, R) for Rd, R in zip(self.R_des, self.ee_R)])


def _finalize(buf: dict) -> RolloutLog:
    return RolloutLog(**{k: np.array(v) for k, v in buf.items() if len(v) > 0})


def run_joint_space(
    env: PandaEnv,
    controller: JointSpacePIDController,
    ref_fn: Callable[[float], tuple],
    duration: float,
    gravity_comp: bool = True,
    q0: Optional[np.ndarray] = None,
    callback: Optional[Callable[[PandaEnv], None]] = None,
    external_force: Optional[np.ndarray] = None,
) -> RolloutLog:
    """Closed-loop joint-space control."""
    env.reset(q0)
    env.set_external_force(external_force)
    controller.reset()
    buf = {k: [] for k in ("t", "q", "dq", "tau", "q_des", "ee_pos", "ee_R")}
    for _ in range(int(round(duration / env.dt))):
        t, q, dq = env.time, env.q, env.dq
        q_des, dq_des = ref_fn(t)
        g = env.gravity_torque() if gravity_comp else np.zeros(7)
        tau = np.asarray(controller.compute_torque(q, dq, q_des, dq_des, g), dtype=float)
        p, R = env.ee_pose_mujoco()
        for k, v in zip(buf, (t, q, dq, tau, q_des, p, R)):
            buf[k].append(v)
        env.step(tau)
        if callback is not None:
            callback(env)
    return _finalize(buf)


def run_cartesian(
    env: PandaEnv,
    controller: CartesianPIDController,
    ref_fn: Callable[[float], tuple],
    duration: float,
    gravity_comp: bool = True,
    q0: Optional[np.ndarray] = None,
    ee_pose_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    jacobian_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    callback: Optional[Callable[[PandaEnv], None]] = None,
    external_force: Optional[np.ndarray] = None,
) -> RolloutLog:
    """Closed-loop Cartesian-space control."""
    env.reset(q0)
    env.set_external_force(external_force)
    controller.reset()
    buf = {k: [] for k in ("t", "q", "dq", "tau", "ee_pos", "ee_R", "x_des", "R_des")}
    for _ in range(int(round(duration / env.dt))):
        t, q, dq = env.time, env.q, env.dq
        x_des, R_des, v_des, w_des = ref_fn(t)
        if ee_pose_fn is None:
            x, R = env.ee_pose_mujoco()
        else:
            T = ee_pose_fn(q)
            x, R = T[:3, 3], T[:3, :3]
        J = env.jacobian_mujoco() if jacobian_fn is None else jacobian_fn(q)
        g = env.gravity_torque() if gravity_comp else np.zeros(7)
        tau = np.asarray(controller.compute_torque(q, dq, J, x, R, x_des, R_des, v_des, w_des, g), dtype=float)
        p_true, R_true = env.ee_pose_mujoco()
        for k, v in zip(buf, (t, q, dq, tau, p_true, R_true, x_des, R_des)):
            buf[k].append(v)
        env.set_target_marker(x_des, R_des)
        env.step(tau)
        if callback is not None:
            callback(env)
    return _finalize(buf)
