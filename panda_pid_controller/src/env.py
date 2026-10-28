"""
Thin, torque-controlled wrapper around the MuJoCo Panda model.

The wrapper hides MuJoCo boilerplate and exposes exactly what a robot controller would get
on real hardware: joint positions/velocities in, joint torques out.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .config import DT, MODEL_PATH, N_JOINTS, Q_READY, TORQUE_LIMITS


class PandaEnv:
    """Simulated Franka Emika Panda driven by joint torques."""

    def __init__(self, model_path=MODEL_PATH):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)  # used for gravity / mass queries
        assert self.model.nv == N_JOINTS, "unexpected model: expected 7 DoF"
        assert abs(self.model.opt.timestep - DT) < 1e-12, "config.DT must equal XML timestep"
        self._ee_site = self.model.site("ee_site").id
        self._flange_site = self.model.site("flange_site").id
        self._target_mocap = self.model.body("target").mocapid[0]
        self._hand_body = self.model.body("hand").id
        self.torque_limits = TORQUE_LIMITS.copy()
        self.reset()

    @property
    def dt(self) -> float:
        """Simulation time step [s]."""
        return self.model.opt.timestep

    @property
    def time(self) -> float:
        """Simulated time since the last reset [s]."""
        return float(self.data.time)

    @property
    def q(self) -> np.ndarray:
        """Joint positions [rad], shape (7,)."""
        return self.data.qpos[:N_JOINTS].copy()

    @property
    def dq(self) -> np.ndarray:
        """Joint velocities [rad/s], shape (7,)."""
        return self.data.qvel[:N_JOINTS].copy()

    def reset(self, q=None, dq=None) -> None:
        """Reset the simulation to joint configuration ``q`` (default: ready pose) at rest."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:N_JOINTS] = Q_READY if q is None else np.asarray(q, dtype=float)
        if dq is not None:
            self.data.qvel[:N_JOINTS] = dq
        mujoco.mj_forward(self.model, self.data)

    def step(self, tau) -> None:
        """Apply joint torques ``tau`` [Nm] (clamped by the actuator limits) for one time step."""
        self.data.ctrl[:] = tau
        mujoco.mj_step(self.model, self.data)

    def set_external_force(self, force=None) -> None:
        """Apply a constant external force [N] (world frame) to the hand, e.g. an unknown payload.

        The controller's gravity compensation does NOT know about it - this is the classic
        situation in which an integral term pays off. ``None`` removes the force.
        ``reset()`` also removes it.
        """
        self.data.xfrc_applied[self._hand_body, :3] = 0.0 if force is None else np.asarray(force, dtype=float)

    def gravity_torque(self) -> np.ndarray:
        """Joint torques needed to hold the current configuration statically, g(q), shape (7,)."""
        s = self._scratch
        s.qpos[:] = self.data.qpos
        s.qvel[:] = 0.0
        mujoco.mj_forward(self.model, s)
        return s.qfrc_bias[:N_JOINTS].copy()

    def mass_matrix(self) -> np.ndarray:
        """Joint-space inertia matrix M(q), shape (7, 7)."""
        M = np.zeros((self.model.nv, self.model.nv))
        try:  # MuJoCo >= 3.3: mj_fullM(model, data, dst)
            mujoco.mj_fullM(self.model, self.data, M)
        except TypeError:  # older releases: mj_fullM(model, dst, data.qM)
            mujoco.mj_fullM(self.model, M, self.data.qM)
        return M

    def ee_pose_mujoco(self):
        """Ground-truth TCP pose in the world frame: ``(position (3,), rotation matrix (3,3))``."""
        d = self.data
        return d.site_xpos[self._ee_site].copy(), d.site_xmat[self._ee_site].reshape(3, 3).copy()

    def flange_pose_mujoco(self):
        """Ground-truth flange pose in the world frame: ``(position (3,), rotation (3,3))``."""
        d = self.data
        return d.site_xpos[self._flange_site].copy(), d.site_xmat[self._flange_site].reshape(3, 3).copy()

    def jacobian_mujoco(self) -> np.ndarray:
        """Ground-truth 6x7 Jacobian of the TCP: rows [v (3); omega (3)], world frame."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._ee_site)
        return np.vstack([jacp, jacr])

    def kinematics_at(self, q):
        """Ground truth ``(ee_pos, ee_R, flange_pos, flange_R, J)`` at an arbitrary ``q``.

        Uses a scratch copy of the simulation state, so the running simulation is untouched.
        """
        s = self._scratch
        s.qpos[:] = 0.0
        s.qpos[:N_JOINTS] = q
        s.qvel[:] = 0.0
        mujoco.mj_forward(self.model, s)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, s, jacp, jacr, self._ee_site)
        return (s.site_xpos[self._ee_site].copy(), s.site_xmat[self._ee_site].reshape(3, 3).copy(),
                s.site_xpos[self._flange_site].copy(), s.site_xmat[self._flange_site].reshape(3, 3).copy(),
                np.vstack([jacp, jacr]))

    def joint_frames_mujoco(self, q) -> np.ndarray:
        """Ground-truth poses ``T_world_link_i`` of link frames 1..7 at ``q``, shape (7, 4, 4).

        MuJoCo's link frames of the Panda coincide with the (modified) DH frames 1..7.
        """
        s = self._scratch
        s.qpos[:] = 0.0
        s.qpos[:N_JOINTS] = q
        s.qvel[:] = 0.0
        mujoco.mj_forward(self.model, s)
        frames = np.tile(np.eye(4), (N_JOINTS, 1, 1))
        for i in range(N_JOINTS):
            b = self.model.body(f"link{i + 1}").id
            frames[i, :3, :3] = s.xmat[b].reshape(3, 3)
            frames[i, :3, 3] = s.xpos[b]
        return frames

    def set_target_marker(self, pos, R=None) -> None:
        """Move the green target marker (mocap body) to a desired pose. Purely visual."""
        self.data.mocap_pos[self._target_mocap] = pos
        if R is not None:
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.asarray(R, dtype=float).reshape(9))
            self.data.mocap_quat[self._target_mocap] = quat
