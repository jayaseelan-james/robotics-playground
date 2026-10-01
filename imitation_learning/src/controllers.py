"""
Differential inverse kinematics for the Panda.

Method:
------
Damped least squares (a.k.a. Levenberg-Marquardt IK):

    dq = J^T (J J^T + lambda^2 I)^-1 * twist          (+ nullspace term)

`lambda` trades tracking accuracy against conditioning: near a kinematic singularity J loses rank,
the unregularised pseudo-inverse explodes, and the arm snaps. The damping term bounds ||dq||.
We additionally clamp per-joint steps and add a nullspace term that gently pulls the arm back 
towards its home posture, which keeps elbows out of the way and stops slow drift into joint
limits over a long episode.
"""

from __future__ import annotations

import mujoco
import numpy as np

from config import IKConfig

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))

# Down-facing TCP orientation that the Panda reaches at its home keyframe.
# Columns are the TCP x/y/z axes expressed in the world frame; z = (0,0,-1)
# means "gripper pointing at the floor".
DOWN_MAT = np.array([[0.0, 1.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 0.0, -1.0]])


class DiffIKController:
    """Cartesian pose target -> 7 joint position targets."""

    def __init__(
        self,
        model: mujoco.MjModel,
        cfg: IKConfig,
        site_name: str = "tcp",
        home_qpos: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise KeyError(f"site {site_name!r} not found")

        # Index bookkeeping: the cube's free joint also lives in qpos/qvel, so we
        # need explicit address arrays rather than assuming the arm is qpos[:7].
        self.joint_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        )
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in self.joint_ids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in self.joint_ids])

        self.jnt_range = model.jnt_range[self.joint_ids].copy()   # (7, 2)
        self.home_qpos = (
            np.asarray(home_qpos, dtype=np.float64)
            if home_qpos is not None
            else np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
        )

        # Scratch buffers (allocated once; MuJoCo writes into them in-place).
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._site_quat = np.zeros(4)
        self._err_quat = np.zeros(4)
        self._neg_quat = np.zeros(4)
        self._rot_err = np.zeros(3)
        self._target_quat = np.zeros(4)

        # A private MjData used purely as IK scratch space. Iterating the
        # linearised solve requires re-evaluating forward kinematics at a
        # candidate configuration; doing that on the live MjData would
        # teleport the simulated robot.
        self._ik_data = mujoco.MjData(model)

    def arm_qpos(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qpos_adr].copy()

    def tcp_pos(self, data: mujoco.MjData) -> np.ndarray:
        return data.site_xpos[self.site_id].copy()

    def tcp_mat(self, data: mujoco.MjData) -> np.ndarray:
        return data.site_xmat[self.site_id].reshape(3, 3).copy()

    def _pose_error(
        self, data: mujoco.MjData, target_pos: np.ndarray, target_mat: np.ndarray
    ) -> np.ndarray:
        """6-vector twist [v; w] that would take the TCP to the target."""
        pos_err = target_pos - data.site_xpos[self.site_id]

        # Orientation error as a rotation vector:
        #   q_err = q_target * conj(q_current);  w = 2*log(q_err)
        # mju_quat2Vel does the log map (and divides by dt=1 -> plain rotvec).
        mujoco.mju_mat2Quat(self._site_quat, data.site_xmat[self.site_id])
        mujoco.mju_mat2Quat(
            self._target_quat, np.ascontiguousarray(target_mat, dtype=np.float64).reshape(9)
        )
        mujoco.mju_negQuat(self._neg_quat, self._site_quat)
        mujoco.mju_mulQuat(self._err_quat, self._target_quat, self._neg_quat)
        mujoco.mju_quat2Vel(self._rot_err, self._err_quat, 1.0)

        return np.concatenate(
            [self.cfg.pos_gain * pos_err, self.cfg.rot_gain * self._rot_err]
        )

    def solve(
        self,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        target_mat: np.ndarray,
    ) -> np.ndarray:
        """
        Returns the joint position targets (7,) that move the TCP toward the target.

        The live `data` is never modified: all candidate configurations are evaluated on
        `self._ik_data`. Running the linear solve a couple of times per control step is
        cheap and noticeably improves tracking, because the Jacobian is only a first-order
        model of the kinematics.
        """
        ik = self._ik_data
        ik.qpos[:] = data.qpos
        ik.qvel[:] = 0.0
        mujoco.mj_kinematics(self.model, ik)
        mujoco.mj_comPos(self.model, ik)

        q = ik.qpos[self.qpos_adr].copy()

        for _ in range(max(1, self.cfg.iterations)):
            twist = self._pose_error(ik, target_pos, target_mat)

            mujoco.mj_jacSite(self.model, ik, self._jacp, self._jacr, self.site_id)
            J = np.vstack([self._jacp[:, self.dof_adr], self._jacr[:, self.dof_adr]])  # (6,7)

            lam2 = self.cfg.damping ** 2
            JJt = J @ J.T + lam2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, twist)

            # Nullspace posture regularisation
            # (I - J^+ J) projects onto joint motions that do not move the TCP,
            # so this term costs nothing in task space.
            if self.cfg.nullspace_gain > 0.0:
                J_pinv = J.T @ np.linalg.inv(JJt)
                null_proj = np.eye(7) - J_pinv @ J
                dq = dq + null_proj @ (self.cfg.nullspace_gain * (self.home_qpos - q))

            dq = np.clip(dq, -self.cfg.max_dq, self.cfg.max_dq)
            q = np.clip(q + dq, self.jnt_range[:, 0] + 1e-3, self.jnt_range[:, 1] - 1e-3)

            ik.qpos[self.qpos_adr] = q
            mujoco.mj_kinematics(self.model, ik)
            mujoco.mj_comPos(self.model, ik)

        return q
