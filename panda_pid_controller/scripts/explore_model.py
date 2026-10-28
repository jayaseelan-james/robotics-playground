"""Get to know the simulated Panda and its coordinate frames."""
import numpy as np

from _common import base_parser

from src import kinematics as K
from src.config import JOINT_LIMITS, Q_READY, TORQUE_LIMITS
from src.env import PandaEnv


def main():
    args = base_parser(__doc__).parse_args()
    env = PandaEnv()
    env.reset(Q_READY)
    np.set_printoptions(precision=4, suppress=True)

    print("== Model ==")
    print(f"Degrees of freedom : {env.model.nv}")
    print(f"Time step          : {env.dt} s ({1 / env.dt:.0f} Hz)")
    print(f"Torque limits [Nm] : {TORQUE_LIMITS}")
    print(f"Joint limits [rad] :\n{JOINT_LIMITS.T}")

    print("\n== Ready pose, ground truth from MuJoCo ==")
    p, R = env.ee_pose_mujoco()
    pf, Rf = env.flange_pose_mujoco()
    print(f"TCP position (world)    : {p}")
    print(f"TCP orientation (world) :\n{R}")
    print(f"Flange position (world) : {pf}")
    print(f"Gravity torque g(q)     : {env.gravity_torque()}")

    print("\n== My Forward Kinematics vs. MuJoCo (random configurations) ==")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(100):
        q = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        pe, Re, *_ = env.kinematics_at(q)
        T = K.forward_kinematics_ee(q)
        worst = max(worst, np.abs(T[:3, 3] - pe).max(), np.abs(T[:3, :3] - Re).max())
    print(f"Largest deviation over 100 random q: {worst:.2e}  ({'OK' if worst < 1e-6 else 'MISMATCH'})")

    if args.viewer:
        import mujoco.viewer
        print("\nViewer open: the red/blue spheres mark the TCP and the flange. Close the window to exit.")
        mujoco.viewer.launch(env.model, env.data)


if __name__ == "__main__":
    main()
