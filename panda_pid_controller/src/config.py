"""Global constants and controller gains."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "mujoco_experiments" / "panda_with_torque_actuator.xml"

N_JOINTS = 7
DT = 0.002  # simulation / control time step [s] (500 Hz)

# Franka's standard "ready" pose [rad]
Q_READY = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])

# Actuator torque limits [Nm]
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

# Joint position limits [rad]
JOINT_LIMITS = np.array([
    [-2.8973, 2.8973],
    [-1.7628, 1.7628],
    [-2.8973, 2.8973],
    [-3.0718, -0.0698],
    [-2.8973, 2.8973],
    [-0.0175, 3.7525],
    [-2.8973, 2.8973],
])

# Modified Denavit-Hartenberg parameters (Craig convention) of the Panda, taken from the Franka
# documentation.
DH_PARAMS = np.array([
    [0.0,     0.333,  0.0],
    [0.0,     0.0,   -np.pi / 2],
    [0.0,     0.316,  np.pi / 2],
    [0.0825,  0.0,    np.pi / 2],
    [-0.0825, 0.384, -np.pi / 2],
    [0.0,     0.0,    np.pi / 2],
    [0.088,   0.0,    np.pi / 2],
])

# DH row of the fixed "flange" frame (frame 8): [a, d, alpha], theta = 0.
DH_FLANGE = np.array([0.0, 0.107, 0.0])

# The hand/tool frame is rotated by this angle about the flange z axis.
HAND_ROT_Z = -np.pi / 4

# The tool centre point (TCP) lies this far along the hand's z axis.
TCP_OFFSET_Z = 0.1034

# Joint-space PID (one value per joint)
JOINT_KP = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])   # [Nm/rad]
JOINT_KD = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0])         # [Nm s/rad]
JOINT_KI = np.zeros(7)                                                  # [Nm/(rad s)]

# Cartesian-space PID
CART_KP_POS = 1000.0  # [N/m]
CART_KD_POS = 80.0    # [N s/m]
CART_KP_ROT = 100.0   # [Nm/rad]
CART_KD_ROT = 10.0    # [Nm s/rad]
CART_KI_POS = 1000.0  # [N/(m s)]
CART_KI_ROT = 100.0   # [Nm/(rad s)]
