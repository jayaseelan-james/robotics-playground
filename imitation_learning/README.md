# Imitation learning on a Franka Emika Panda in MuJoCo

Pick up a coloured cube, carry it over a fixed wall, drop it in a bin. A scripted expert produces
demonstrations; an MLP learns to imitate them with an MSE loss; the policy is then evaluated
closed-loop.

```
scene (MuJoCo)  ->  scripted expert  ->  demonstrations  ->  dataset -> MSE training -> evaluation
```


## Quick start

```bash
pip install -r requirements.txt

# ~2 minutes end to end; proves every stage is wired up
python src/run_pipeline.py --smoke --out output/smoke

# the real thing: 200 demos, 150 epochs, 50 evaluation episodes
python run_pipeline.py --out output/pipeline

# add the ablation table and the data-scaling curve
python run_pipeline.py --out output/pipeline --ablate --scaling
```

Headless machines need a GL backend for the videos:

```bash
MUJOCO_GL=egl    python run_pipeline.py     # headless with a GPU
MUJOCO_GL=osmesa python run_pipeline.py     # headless, software rendering (slow)
```


Each stage also runs standalone, which can be used for debugging:

```bash
python expert.py  --episodes 5 --debug     # watch the state machine
python collect_demos.py  --episodes 200
python train.py  --epochs 100
python evaluate.py  --episodes 50 --videos 2
python view.py      # watch the trained policy or the scripted expert run in real-time
```


## Files

| File | Responsibility |
|---|---|
| `config.py` | Every tunable number, in one dataclass tree |
| `controllers.py` | Damped-least-squares differential IK |
| `env.py` | Gymnasium-style environment; observation and action spaces |
| `expert.py` | Scripted state machine + demonstration rollout with DART noise |
| `collect_demos.py` | Demonstration collection, success filtering, dataset report |
| `dataset.py` | Normalisation, action chunking, episode-level train/val split |
| `policy.py` | Chunking MLP + temporal-ensembling runner + checkpoint I/O |
| `train.py` | BC training loop: masked weighted MSE, cosine schedule, EMA |
| `evaluate.py` | Closed-loop rollouts, per-colour breakdown, failure taxonomy |
| `view.py` | View the trained policy or scripted expert run in real-time |
| `run_pipeline.py` | Driver, ablation study, data-scaling study |


## Design decisions

### 1. Cartesian actions, not joint actions

- The action (`[dx, dy, dz, gripper]`) is a TCP displacement plus a gripper command. A classical IK
  solver converts it to joint targets.

- Cloning raw joint targets is a bad for two reasons:

   * **Conditioning:** A 0.05 rad error in joint 1 is harmless near the base and 4 cm of
     end-effector error at full extension. MSE in joint space does not correspond to MSE in the
     space we care about.
   * **Redundancy:** A 7-DoF arm has a 1-D nullspace: infinitely many joint configurations reach the
     same pose. A scripted expert picks among them essentially arbitrarily, so demonstrations are
     multi-modal in joint space — and the MSE average of two valid arm postures is usually an
     invalid one.

### 2. Actions integrate onto a *target* pose, not the measured pose

- `target += action * scale`, re-synced to the measured TCP only at reset.

- Integrating the *measured* pose creates a positive feedback loop: tracking lag shrinks the
  commanded step, the arm falls further behind, the step shrinks again, and the policy stalls. The
  target is clamped to a workspace box so a diverging policy cannot drive the IK solver into a
  singularity.

### 3. The expert sees exactly what the policy sees

- `ScriptedExpert.act()` takes the same flat observation vector the network receives, and never
  touches anything else.

### 4. Hand the network its relative features

- `cube_pos - tcp_pos` and `cube_pos - bin_pos` are trivial functions of quantities already in the
  observation. Including them anyway is worth a lot. A small MLP on ~20k samples will not reliably
  learn coordinate subtraction; handing it the difference converts the problem from "learn
  arithmetic" into "learn a servo law".
- Orientations enter as the **6-D rotation representation** (first two columns of the rotation
  matrix). Quaternions have a double cover (`q` and `-q` are the same rotation) and Euler angles
  wrap; both create discontinuities that an MSE regressor handles badly.

### 5. A P-controller expert

- The expert's Cartesian action is `clip(p_gain * error / scale, -1, 1)`, then exponentially
  smoothed (the gripper channel is excluded as smoothing a binary command just delays and blurs it).
- A proportional expert makes the regression target locally linear in the observation, which is
  exactly what MSE is good at fitting.


## Validation MSE is not the metric

- Validation MSE measures agreement with the expert **on the expert's own state distribution**.
  Success rate measures what happens when the policy's own mistakes move it off that distribution.
  The two correlate weakly and sometimes negatively.

- So `train.py` reports validation loss for diagnosis only; `evaluate.py` is what decides.
  Evaluation uses a **fixed seed list**, so two policies face identical cube poses, bin jitter and
  colours.


## When MSE stops being the right loss

- MSE regression fits `E[a | o]`. That is exactly right when the expert is a deterministic function
  of the observation, which the scripted expert nearly is.

- It is exactly wrong when demonstrations are **multi-modal** — a human who sometimes goes left and
  sometimes right around an obstacle. The mean of "left" and "right" is "straight into the wall".
  Symptoms: the policy freezes at decision points, or oscillates between modes.


## Extensions, roughly in order of value

1. **Gripper yaw as a fifth action:** Currently the TCP orientation is fixed "pointing down", and
   cube yaw (±23°) is absorbed by the 8 cm finger opening. Add `d_yaw`, have the controller maintain
   `Rz(θ) @ DOWN_MAT`, and let the expert align with the cube's yaw. Enables arbitrary cube
   orientations and non-cubic objects.
2. **Randomise the obstacle:** Its pose is currently constant, which is why the collection report
   flags those observation blocks as dead. Randomising position and height forces the policy to
   actually *use* them instead of memorising one transit height.
3. **Colour-conditioned goals:** Two bins; red cubes go left, blue go right. Turns colour from a
   nuisance variable into a task-relevant cue and makes the held-out-colour evaluation much more
   interesting.
4. **Pixel observations:** Swap the state vector for the wrist camera, add a small CNN (or a frozen
   ResNet-18 with a trained head).
5. **DAgger:** The expert is a function you can call at any state. Three or four rounds of on-policy
   relabelling typically beat doubling the dataset.
6. **RL fine-tuning:** The current environment class already returns a shaped reward that Behaviour
   Cloning never uses. Improvise by using Reinforcement learning.

---