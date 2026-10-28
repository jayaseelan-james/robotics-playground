# Joint-Space & Cartesian-Space PID Control of a Franka Emika Panda in MuJoCo

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/explore_model.py                     # should print the Panda's ready pose
python scripts/cartesian_demo.py --viewer --own-kinematics
```

*Viewer note:* `--viewer` needs a display. On macOS start scripts with `mjpython` instead of `python`.
Without a viewer all scripts save plots to `results/`.

## Conventions (used everywhere)

* World frame `W` = robot base: x forward, y left, z up. `T_a_b` is the pose of frame `b` in frame `a`, `p_a = T_a_b p_b`.
* Quaternions `[w, x, y, z]`. Rotation vectors, angular velocities and moments are in the **world frame**.
* Jacobians are **6×7**, rows `[linear; angular]`, world frame, point of reference = **TCP** (tool center point).
* Units: metres, radians, seconds, Newton, Newton-metres.
