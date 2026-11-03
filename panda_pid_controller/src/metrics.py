"""Controller-performance metrics."""
from __future__ import annotations

import numpy as np


def rotation_angle(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle [rad] between two rotation matrices, in [0, pi]."""
    c = (np.trace(R_a @ R_b.T) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def rmse(x: np.ndarray) -> float:
    """Root mean square of an array."""
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


def overshoot(y: np.ndarray, y_start: float, y_goal: float) -> float:
    """Relative overshoot of a step response ``y`` going from ``y_start`` to ``y_goal``.

    0.0 means the goal was never exceeded; 0.1 means it was exceeded by 10 % of the step size.
    """
    step = y_goal - y_start
    if abs(step) < 1e-12:
        return 0.0
    return float(max(0.0, np.max((y - y_goal) * np.sign(step)) / abs(step)))


def settling_time(t: np.ndarray, err: np.ndarray, tol: float) -> float:
    """First time after which ``|err| <= tol`` holds for the rest of the record.

    Returns ``inf`` if the error never settles within the record.
    """
    bad = np.where(np.abs(err) > tol)[0]
    if bad.size == 0:
        return float(t[0])
    if bad[-1] == len(t) - 1:
        return float("inf")
    return float(t[bad[-1] + 1])
