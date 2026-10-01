"""
Central configuration for the whole project.

Units: metres, radians, seconds, kilograms. The world frame is the robot base
frame (Panda link0 sits at the origin).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

@dataclass
class EnvConfig:
    """Scene geometry, control rates and domain randomisation ranges."""

    # Simulation / control rates
    _control_hz: float = 20.0                 # policy/expert decision rate
    max_steps: int = 200                      # episode horizon in *control* steps

    # End-effector action scaling
    # A normalised action of 1.0 moves the TCP target by this much in one
    # control step.  0.025 m @ 20 Hz = 0.5 m/s, which the Panda tracks easily.
    action_pos_scale: float = 0.025

    # Cube size
    cube_half_size: float = 0.02              # 4 cm cube

    # Spawn region for the cube (x_min, x_max, y_min, y_max)
    cube_x_range: tuple[float, float] = (0.38, 0.54)
    cube_y_range: tuple[float, float] = (-0.38, -0.24)
    cube_yaw_range: tuple[float, float] = (-0.40, 0.40)   # +/-23 deg (see README)

    # Target bin
    bin_pos: tuple[float, float, float] = (0.5, 0.27, 0.04) 
    bin_inner_half: float = 0.065            # inner square half-width

    # Randomise the target bin a little so the policy cannot memorise one goal pose.
    bin_jitter: float = 0.04                 # +/- metres on x and y

    # Colour randomisation
    # `train_colors` are seen during demonstration collection;
    # `eval_colors` includes unseen hues to measure colour-invariance.
    train_colors: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "red":    (0.85, 0.15, 0.15),
            "green":  (0.15, 0.70, 0.25),
            "blue":   (0.15, 0.35, 0.85),
            "yellow": (0.90, 0.80, 0.15),
        }
    )
    heldout_colors: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "magenta": (0.85, 0.15, 0.75),
            "cyan":    (0.15, 0.80, 0.80),
        }
    )

    # Success / failure criteria
    success_height: float = 0.070            # cube centre must be below this ...
    success_settle_vel: float = 0.05         # ... and nearly at rest
    success_hold_steps: int = 10             # ... for this many control steps
    drop_height: float = -0.02               # cube fell off the world

    # Rendering
    render_width: int = 480
    render_height: int = 360
    render_camera: str = "track"

    @property
    def n_substeps(self) -> int:
        """Physics steps per control step."""
        return int(round(1.0 / (self._control_hz * 0.002)))

    @property
    def control_dt(self) -> float:
        return 1.0 / self._control_hz


@dataclass
class IKConfig:
    """Damped-least-squares differential inverse kinematics."""

    damping: float = 0.08          # lambda value
    pos_gain: float = 1.0          # fraction of position error taken per solve
    rot_gain: float = 0.6          # orientation is secondary -> lower gain
    max_dq: float = 0.12           # rad, per-joint clamp per control step
    nullspace_gain: float = 0.12   # pull towards the home posture
    iterations: int = 2            # DLS iterations per control step


@dataclass
class ExpertConfig:
    """Waypoint heights, tolerances and gains of the state-machine expert."""

    pre_grasp_height: float = 0.13   # TCP z when hovering above the cube
    grasp_z_offset: float = 0.002    # TCP z relative to cube centre at grasp
    transit_height: float = 0.32     # TCP z while crossing the obstacle
    release_height: float = 0.09     # TCP z when opening above the bin
    retreat_height: float = 0.35

    xy_tol: float = 0.008            # position tolerance for phase transitions
    z_tol: float = 0.006
    coarse_tol: float = 0.02

    close_steps: int = 12            # control steps spent squeezing
    open_steps: int = 8              # control steps spent releasing

    # Proportional gain converting a position error into a normalised action.
    # p_gain * error / action_pos_scale, then clipped to [-1, 1].
    p_gain: float = 0.9

    # Low-pass on the expert's own output: a_t = beta*a_t + (1-beta)*a_{t-1}.
    # Smooth expert actions are much easier to regress with an MSE loss.
    smoothing: float = 0.65


@dataclass
class CollectConfig:
    n_episodes: int = 200
    seed: int = 0

    # DART-style action noise. This widens the visited state distribution so
    # the learner also sees some recovery behaviour.
    action_noise_std: float = 0.06
    noise_prob: float = 0.8          # fraction of episodes that get noise
    keep_failures: bool = False      # only successful demos are written out
    out_path: str = "demos.npz"


@dataclass
class PolicyConfig:
    hidden_sizes: tuple[int, ...] = (512, 512, 512)
    dropout: float = 0.1
    layer_norm: bool = True
    # Action chunking: predict `chunk_len` future actions from one observation.
    chunk_len: int = 4
    # Temporal ensembling weight. Lower = smoother / more lag.
    ensemble_decay: float = 0.35
    obs_history: int = 1             # >1 stacks past observations


@dataclass
class TrainConfig:
    epochs: int = 150
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    val_frac: float = 0.1
    ema_decay: float = 0.999

    # The gripper dimension is effectively binary; up-weighting it stops the
    # regressor from smearing open/close transitions over many steps.
    gripper_loss_weight: float = 3.0
    seed: int = 0
    num_workers: int = 0
    device: str = "auto"
    ckpt_path: str = "output/policy.pt"
    log_every: int = 10


@dataclass
class EvalConfig:
    n_episodes: int = 50
    seed: int = 12345
    include_heldout_colors: bool = True
    temporal_ensemble: bool = True
    render_n_videos: int = 2
    out_dir: str = "output/eval"


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    debug: bool = False              # verbose per-step tracing

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Config":
        return Config(
            env=EnvConfig(**d["env"]),
            ik=IKConfig(**d["ik"]),
            expert=ExpertConfig(**d["expert"]),
            collect=CollectConfig(**d["collect"]),
            policy=PolicyConfig(**d["policy"]),
            train=TrainConfig(**d["train"]),
            eval=EvalConfig(**d["eval"]),
            debug=d.get("debug", False),
        )


def smoke_config() -> Config:
    """Tiny config used by the smoke test / CI: runs end-to-end in ~2 minutes."""
    cfg = Config()
    cfg.collect.n_episodes = 12
    cfg.train.epochs = 8
    cfg.train.batch_size = 128
    cfg.eval.n_episodes = 6
    cfg.eval.render_n_videos = 1
    return cfg
