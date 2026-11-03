"""Plot helpers for the demo scripts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import TORQUE_LIMITS
from .pipeline import RolloutLog


def plot_joint_log(log: RolloutLog, path, title: str = "Joint-space control") -> Path:
    """Desired vs. measured joint angles, tracking error and commanded torques (7 joints)."""
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    colors = plt.cm.tab10(np.arange(7))
    for i in range(7):
        ax[0].plot(log.t, log.q[:, i], color=colors[i], label=f"q{i + 1}")
        ax[0].plot(log.t, log.q_des[:, i], "--", color=colors[i], alpha=0.6)
        ax[1].plot(log.t, log.joint_error()[:, i], color=colors[i])
        ax[2].plot(log.t, log.tau[:, i], color=colors[i])
    ax[0].set(ylabel="joint angle [rad]", title=f"{title}  (dashed = desired)")
    ax[0].legend(ncol=7, fontsize=8, loc="upper center")
    ax[1].set(ylabel="error q_des - q [rad]")
    ax[2].set(ylabel="torque [Nm]", xlabel="time [s]")
    ax[2].axhline(TORQUE_LIMITS[0], color="k", lw=0.5, ls=":")
    ax[2].axhline(-TORQUE_LIMITS[0], color="k", lw=0.5, ls=":")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_cartesian_log(log: RolloutLog, path, title: str = "Cartesian-space control") -> Path:
    """Desired vs. measured TCP position, position/orientation error, torques and an xy top view."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for i, name in enumerate("xyz"):
        ax[0, 0].plot(log.t, log.ee_pos[:, i], label=name)
        ax[0, 0].plot(log.t, log.x_des[:, i], "--", color=f"C{i}", alpha=0.6)
    ax[0, 0].set(xlabel="time [s]", ylabel="TCP position [m]", title=f"{title}  (dashed = desired)")
    ax[0, 0].legend()
    ax[0, 1].plot(log.t, 1e3 * log.position_error(), label="position [mm]")
    ax[0, 1].plot(log.t, 1e3 * log.orientation_error(), label="orientation [mrad]")
    ax[0, 1].set(xlabel="time [s]", ylabel="error", title="tracking error")
    ax[0, 1].legend()
    for i in range(7):
        ax[1, 0].plot(log.t, log.tau[:, i], label=f"tau{i + 1}")
    ax[1, 0].set(xlabel="time [s]", ylabel="torque [Nm]", title="commanded torques")
    ax[1, 0].legend(ncol=4, fontsize=7)
    ax[1, 1].plot(log.x_des[:, 0], log.x_des[:, 1], "--", label="desired")
    ax[1, 1].plot(log.ee_pos[:, 0], log.ee_pos[:, 1], label="measured")
    ax[1, 1].set(xlabel="x [m]", ylabel="y [m]", title="top view", aspect="equal")
    ax[1, 1].legend()
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
