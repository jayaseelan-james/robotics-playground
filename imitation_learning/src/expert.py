"""
A scripted, closed-loop expert for the pick-over-wall-place task.

Why a script?
-------------
Behavioural cloning needs (observation, action) pairs from something that can already do the task.
In a real lab that is a teleoperator. In simulation a short state machine is better for learning
the pipeline: it is deterministic and repeatable.

Run this file directly to watch the expert and print a phase trace:
    python expert.py --episodes 3 --debug
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from config import Config, EnvConfig, ExpertConfig
from env import ACTION_DIM, PandaPickPlaceEnv, obs_field

GRIP_OPEN = 1.0
GRIP_CLOSE = -1.0


class Phase(IntEnum):
    """Stages of the scripted plan (also used as an auxiliary training label)."""

    HOVER = 0       # move above the cube, gripper open
    DESCEND = 1     # lower onto the cube
    CLOSE = 2       # squeeze
    LIFT = 3        # straight up to transit height
    CROSS = 4       # travel to above the bin, clearing the wall
    LOWER = 5       # descend into the bin
    RELEASE = 6     # open the fingers
    RETREAT = 7     # back off so the cube can settle
    DONE = 8


N_PHASES = len(Phase)


class ScriptedExpert:
    """Closed-loop waypoint controller with a small finite state machine."""

    def __init__(self, cfg: ExpertConfig, env_cfg: EnvConfig, debug: bool = False) -> None:
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.debug = debug
        self.reset()

    def reset(self) -> None:
        self.phase = Phase.HOVER
        self._timer = 0
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._grasp_xy: np.ndarray | None = None

    def _servo(self, tcp: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Proportional position servo expressed in normalised action units.

        A P-controller makes the action magnitude a smooth function of the error,
        which is friendlier to an MSE regressor; the regression target becomes
        locally linear in the observation.
        """
        err = target - tcp
        return np.clip(self.cfg.p_gain * err / self.env_cfg.action_pos_scale, -1.0, 1.0)

    def act(self, obs: np.ndarray) -> tuple[np.ndarray, int]:
        """Return (action, phase) for one control step."""
        c = self.cfg
        tcp = obs_field(obs, "tcp_pos")
        cube = obs_field(obs, "cube_pos")
        bin_p = obs_field(obs, "bin_pos")
        width = float(obs_field(obs, "grip_width")[0])
        rel = obs_field(obs, "cube_rel_tcp")

        grip = GRIP_OPEN
        target = tcp.copy()

        # HOVER: get above the cube
        if self.phase == Phase.HOVER:
            target = np.array([cube[0], cube[1], c.pre_grasp_height])
            if (np.linalg.norm(tcp[:2] - cube[:2]) < c.xy_tol
                    and abs(tcp[2] - c.pre_grasp_height) < c.coarse_tol):
                self._advance(Phase.DESCEND)

        # DESCEND: lower onto the cube
        elif self.phase == Phase.DESCEND:
            target = np.array([cube[0], cube[1], cube[2] + c.grasp_z_offset])
            if np.linalg.norm(tcp - target) < c.z_tol:
                self._grasp_xy = tcp[:2].copy()
                self._advance(Phase.CLOSE)

        # CLOSE: squeeze
        elif self.phase == Phase.CLOSE:
            target = np.array([cube[0], cube[1], cube[2] + c.grasp_z_offset])
            grip = GRIP_CLOSE
            self._timer += 1
            if self._timer >= c.close_steps:
                self._advance(Phase.LIFT)

        # LIFT: straight up
        elif self.phase == Phase.LIFT:
            xy = self._grasp_xy if self._grasp_xy is not None else tcp[:2]
            target = np.array([xy[0], xy[1], c.transit_height])
            grip = GRIP_CLOSE
            if tcp[2] > c.transit_height - c.coarse_tol:
                self._advance(Phase.CROSS)

        # CROSS: over the wall
        elif self.phase == Phase.CROSS:
            target = np.array([bin_p[0], bin_p[1], c.transit_height])
            grip = GRIP_CLOSE
            if np.linalg.norm(tcp[:2] - bin_p[:2]) < c.coarse_tol:
                self._advance(Phase.LOWER)

        # LOWER: into the bin
        elif self.phase == Phase.LOWER:
            target = np.array([bin_p[0], bin_p[1], c.release_height])
            grip = GRIP_CLOSE
            if abs(tcp[2] - c.release_height) < c.coarse_tol:
                self._advance(Phase.RELEASE)

        # RELEASE: open
        elif self.phase == Phase.RELEASE:
            target = np.array([bin_p[0], bin_p[1], c.release_height])
            grip = GRIP_OPEN
            self._timer += 1
            if self._timer >= c.open_steps:
                self._advance(Phase.RETREAT)

        # RETREAT: get out of the way
        elif self.phase == Phase.RETREAT:
            target = np.array([bin_p[0], bin_p[1], c.retreat_height])
            grip = GRIP_OPEN
            if tcp[2] > c.retreat_height - c.coarse_tol:
                self._advance(Phase.DONE)

        else:  # DONE: hold still
            target = tcp.copy()
            grip = GRIP_OPEN

        # Recovery: if the cube slips out mid-transport, start over.
        # Without this the expert may place thin air, and those episodes
        # would be recorded as demonstrations of nothing.
        if self.phase in (Phase.LIFT, Phase.CROSS, Phase.LOWER):
            if np.linalg.norm(rel) > 0.06 or width > 0.075:
                if self.debug:
                    print(f"[expert] lost the cube in {self.phase.name}; re-approaching")
                self._advance(Phase.HOVER)

        action = np.concatenate([self._servo(tcp, target), [grip]]).astype(np.float32)

        # Exponential smoothing. The gripper channel is deliberately excluded:
        # smoothing a binary command just delays it and blurs the transition.
        beta = self.cfg.smoothing
        action[:3] = beta * action[:3] + (1.0 - beta) * self._prev_action[:3]
        self._prev_action = action.copy()
        return action, int(self.phase)

    def _advance(self, nxt: Phase) -> None:
        if self.debug:
            print(f"[expert] {self.phase.name} -> {nxt.name}")
        self.phase = nxt
        self._timer = 0


def perform_expert_action(
    env: PandaPickPlaceEnv,
    expert: ScriptedExpert,
    seed: int | None = None,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    options: dict | None = None,
):
    """
    Run one expert episode.

    DART-STYLE NOISE (noise_std > 0):
    Executes `expert(s) + noise` but labels the sample with `expert(s)`.
    The demonstrator therefore keeps making optimal decisions while the robot
    wanders into states a perfect expert would never visit. The dataset then
    contains the corrective action for those states.

    This is the cheapest available antidote to covariate shift, which is the
    failure mode of behavioural cloning: the policy makes a small error, lands
    in a state absent from the training distribution, makes a bigger error, and
    the errors compound quadratically in the horizon.
    """
    rng = rng or np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed, options=options)
    expert.reset()

    observations, actions, phases = [], [], []
    done = False
    info: dict = {}
    while not done:
        action, phase = expert.act(obs)
        observations.append(obs.copy())
        actions.append(action.copy())    # clean label
        phases.append(phase)

        exec_action = action
        if noise_std > 0.0:
            noisy = action.copy()
            noisy[:3] = np.clip(noisy[:3] + rng.normal(0.0, noise_std, 3), -1.0, 1.0)
            exec_action = noisy          # perturbed execution

        obs, _, terminated, truncated, info = env.step(exec_action)
        done = terminated or truncated

    ep = info["episode"]
    return {
        "obs": np.asarray(observations, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "phase": np.asarray(phases, dtype=np.int64),
        "success": bool(ep.success),
        "collided": bool(ep.collided),
        "length": len(observations),
        "color": ep.color_name,
    }


if __name__ == "__main__":
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    env = PandaPickPlaceEnv(cfg.env, cfg.ik, seed=0)
    env.debug = args.debug
    expert = ScriptedExpert(cfg.expert, cfg.env, debug=args.debug)

    stats = Counter()
    for i in range(args.episodes):
        ep = perform_expert_action(env, expert, seed=1000 + i, noise_std=args.noise)
        stats["success"] += ep["success"]
        stats["collided"] += ep["collided"]
        print(f"ep {i:2d}  success={ep['success']!s:<5} collided={ep['collided']!s:<5} "
              f"len={ep['length']:3d} color={ep['color']}")
    print(f"\n{stats['success']}/{args.episodes} succeeded, "
          f"{stats['collided']} touched the obstacle")
