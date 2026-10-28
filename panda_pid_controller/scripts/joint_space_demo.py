"""
Joint-space control: plan a quintic joint trajectory and track it with the joint-space PID controller.

A plot is written to ``results/joint_space_<timestamp>.png``.
"""
import time

import numpy as np

from _common import RESULTS, base_parser, viewer_callback

from src.config import Q_READY, JOINT_KI, JOINT_KP, JOINT_KD
from src.controllers import JointSpacePIDController
from src.env import PandaEnv
from src.pipeline import run_joint_space
from src.plotting import plot_joint_log
from src.trajectories import JointTrajectory


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--duration", type=float, default=2.0, help="duration of the planned motion [s]")
    ap.add_argument("--total", type=float, default=5.0, help="total simulated time [s]")
    ap.add_argument("--no-gravity-comp", action="store_true", help="pass gravity = 0 to the controller")
    ap.add_argument("--ki", type=float, default=None, help="override the integral gain of all joints")
    args = ap.parse_args()


    q_goal = Q_READY + np.array([0.6, -0.4, 0.5, 0.5, -0.6, 0.5, 0.8])
    ki = JOINT_KI if args.ki is None else np.full(7, args.ki)
    env = PandaEnv()
    ctrl = JointSpacePIDController(JOINT_KP, JOINT_KD, ki)
    traj = JointTrajectory(Q_READY, q_goal, args.duration)

    def ref(t):
        q, dq, _ = traj.evaluate(t)
        return q, dq

    try:
        with viewer_callback(env, args.viewer) as cb:
            log = run_joint_space(env, ctrl, ref, args.total, gravity_comp=not args.no_gravity_comp, callback=cb)
    except KeyboardInterrupt:
        return
    path = plot_joint_log(log, RESULTS / f"joint_space_{time.strftime('%Y%m%d-%H%M%S')}.png")
    err = np.abs(log.joint_error())
    print(f"Max tracking error : {err.max():.4f} rad")
    print(f"Final error        : {err[-1].max():.2e} rad")
    print(f"Max torque         : {np.abs(log.tau).max(axis=0).round(1)} Nm")
    print(f"Plot saved to {path}")


if __name__ == "__main__":
    main()
