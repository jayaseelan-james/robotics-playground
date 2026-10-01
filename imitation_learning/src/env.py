"""
A simulated environment with a Panda to pick a cube, carry it over a wall, and drop it in a bin.

Action Space:
4-D, each component in [-1, 1]:

    a[0:3]  desired TCP displacement, scaled by `action_pos_scale`
    a[3]    gripper:  +1 fully open, -1 fully closed

The TCP orientation is held at a fixed pointing down pose by the IK solver,
so the cube's yaw is absorbed by the 8 cm finger opening.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from config import EnvConfig, IKConfig
from controllers import DOWN_MAT, DiffIKController


SCENE_FILE = "panda_with_obstacle_scene.xml"
TCP_OFFSET_Z = 0.1034
ACTION_DIM = 4
ACTION_NAMES = ("dx", "dy", "dz", "grip")


# Observation layout
OBS_SPEC: tuple[tuple[str, int], ...] = (
    ("arm_qpos", 7),        # measured joint angles
    ("arm_qvel", 7),        # measured joint velocities
    ("grip_width", 1),      # distance between the fingers (0 .. 0.08)
    ("grip_vel", 1),
    ("tcp_pos", 3),
    ("tcp_rot6d", 6),
    ("target_pos", 3),      # the controller's *commanded* TCP target
    ("cube_pos", 3),
    ("cube_rot6d", 6),
    ("cube_rel_tcp", 3),    # cube_pos - tcp_pos            <- key feature
    ("cube_rgb", 3),        # colour, normalised to [0, 1]
    ("bin_pos", 3),
    ("cube_rel_bin", 3),    # cube_pos - bin_pos            <- key feature
    ("obstacle_pos", 3),
    ("obstacle_half", 3),
)
OBS_DIM = sum(n for _, n in OBS_SPEC)
_offset = 0
OBS_SLICES: dict[str, slice] = {}
for _name, _n in OBS_SPEC:
    OBS_SLICES[_name] = slice(_offset, _offset + _n)
    _offset += _n
del _offset, _name, _n


def obs_field(obs: np.ndarray, name: str) -> np.ndarray:
    """Read a named block out of a flat observation (works on batches too)."""
    return obs[..., OBS_SLICES[name]]


@dataclass
class EpisodeInfo:
    """Per-episode bookkeeping."""

    success: bool = False
    collided: bool = False
    dropped: bool = False
    grasped: bool = False
    color_name: str = ""
    steps: int = 0


class PandaPickPlaceEnv:
    """Gymnasium-style environment (reset/step/render), without the gym dependency."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        cfg: EnvConfig | None = None,
        ik_cfg: IKConfig | None = None,
        seed: int | None = None,
        render_mode: str | None = None,
        eval_mode: bool = False,
        verbose: bool = True,
        xml_file_path: str | None = None,
    ) -> None:
        self.env_config = cfg or EnvConfig()
        self.ik_config = ik_cfg or IKConfig()
        self.eval_mode = eval_mode
        self.render_mode = render_mode
        self.debug = False
        self.rng = np.random.default_rng(seed)

        if not xml_file_path:
            root_dir = Path(__file__).resolve().parent.parent.parent
            xml_file_path = root_dir / "mujoco_experiments" / SCENE_FILE

        spec = mujoco.MjSpec.from_file(str(xml_file_path))
        # Inject the TCP site into the existing `hand` body
        hand = spec.body("hand")
        tcp = hand.add_site()
        tcp.name = "tcp"
        tcp.pos = [0.0, 0.0, TCP_OFFSET_Z]
        tcp.size = [0.006, 0.006, 0.006]
        tcp.rgba = [1.0, 0.25, 0.1, 0.45]

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        if verbose:
            print(f"Compiled model: nq={self.model.nq} nv={self.model.nv} nu={self.model.nu} "
                f"ngeom={self.model.ngeom} nbody={self.model.nbody}")

        # Id lookups
        objects = mujoco.mjtObj
        self.tcp_sid = self.name2id(objects.mjOBJ_SITE, "tcp")
        self.cube_bid = self.name2id(objects.mjOBJ_BODY, "cube")
        self.bin_bid = self.name2id(objects.mjOBJ_BODY, "bin")
        self.obstacle_bid = self.name2id(objects.mjOBJ_BODY, "obstacle")
        self.cube_gid = self.name2id(objects.mjOBJ_GEOM, "cube_geom")
        self.obstacle_gid = self.name2id(objects.mjOBJ_GEOM, "obstacle_wall")
        self.floor_gid = self.name2id(objects.mjOBJ_GEOM, "floor")
        self.cube_qadr = self.model.jnt_qposadr[self.name2id(objects.mjOBJ_JOINT, "cube_joint")]
        self.cube_vadr = self.model.jnt_dofadr[self.name2id(objects.mjOBJ_JOINT, "cube_joint")]
        self.finger_qadr = [
            self.model.jnt_qposadr[self.name2id(objects.mjOBJ_JOINT, f"finger_joint{i}")] for i in (1, 2)
        ]
        self.finger_vadr = [
            self.model.jnt_dofadr[self.name2id(objects.mjOBJ_JOINT, f"finger_joint{i}")] for i in (1, 2)
        ]
        self.grip_act_id = self.name2id(objects.mjOBJ_ACTUATOR, "actuator8")
        self.mocap_bin = self.model.body_mocapid[self.bin_bid]

        # Every geom that belongs to the robot is used for obstacle-collision detection.
        link0 = self.name2id(objects.mjOBJ_BODY, "link0")
        self.robot_gids = {
            g for g in range(self.model.ngeom) if self.model.body_rootid[self.model.geom_bodyid[g]] == self.model.body_rootid[link0]
        }

        self.ctrl = DiffIKController(self.model, self.ik_config, "tcp")

        # Episode state
        self._target_pos = np.zeros(3)
        self._grip_cmd = 1.0
        self._step_count = 0
        self._success_counter = 0
        self._info = EpisodeInfo()
        self._renderer: mujoco.Renderer | None = None

        # Workspace clamp for the integrated target (safety, see module docstring).
        self.ws_low = np.array([0.22, -0.50, 0.012])
        self.ws_high = np.array([0.72, 0.50, 0.60])

        self.observation_dim = OBS_DIM
        self.action_dim = ACTION_DIM

        if verbose:
            print(f"obs_dim={OBS_DIM}, action_dim={ACTION_DIM}, substeps={self.env_config.n_substeps}")

    def name2id(self, objtype: mujoco.mjtObj, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, objtype, name)
        if idx < 0:
            raise KeyError(f"{objtype} named {name!r} not found in model")
        return idx

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}

        mujoco.mj_resetData(self.model, self.data)

        # arm to home 
        home = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
        self.data.qpos[self.ctrl.qpos_adr] = home
        for adr in self.finger_qadr:
            self.data.qpos[adr] = 0.04                       # fully open
        self.data.ctrl[:7] = home
        self.data.ctrl[self.grip_act_id] = 255.0             # open

        # cube pose
        c = self.env_config
        cube_xy = options.get("cube_xy")
        if cube_xy is None:
            cube_xy = np.array([
                self.rng.uniform(*c.cube_x_range),
                self.rng.uniform(*c.cube_y_range),
            ])
        yaw = options.get("cube_yaw", self.rng.uniform(*c.cube_yaw_range))
        self.data.qpos[self.cube_qadr: self.cube_qadr + 3] = [
            cube_xy[0], cube_xy[1], c.cube_half_size
        ]
        self.data.qpos[self.cube_qadr + 3: self.cube_qadr + 7] = [
            np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)
        ]
        self.data.qvel[self.cube_vadr: self.cube_vadr + 6] = 0.0

        # cube colour
        palette = dict(c.train_colors)
        if self.eval_mode:
            palette.update(c.heldout_colors)
        name = options.get("color")
        if name is None:
            name = str(self.rng.choice(sorted(palette.keys())))
        rgb = np.asarray(palette[name], dtype=np.float64)
        self.model.geom_rgba[self.cube_gid, :3] = rgb
        self._cube_rgb = rgb.astype(np.float32)
        self._color_name = name

        # bin pose (mocap body: set position directly)
        jitter = self.rng.uniform(-c.bin_jitter, c.bin_jitter, size=2)
        bin_xy = options.get("bin_xy", np.array(c.bin_pos[:2]) + jitter)
        self.data.mocap_pos[self.mocap_bin] = [bin_xy[0], bin_xy[1], c.bin_pos[2]]

        mujoco.mj_forward(self.model, self.data)

        # controller / episode state
        self._target_pos = self.data.site_xpos[self.tcp_sid].copy()
        self._grip_cmd = 1.0
        self._step_count = 0
        self._success_counter = 0
        self._info = EpisodeInfo(color_name=name)

        obs = self._get_obs()
        if self.debug:
            print(f"[env.reset] color={name:<8} cube={np.round(cube_xy,3)} "
                  f"yaw={yaw:+.2f} bin={np.round(bin_xy,3)} tcp={np.round(self._target_pos,3)}")
        return obs, {"episode": self._info, "color": name}

    def step(self, action: np.ndarray, render_callback=None):
        """Advance one control step."""
        act = np.clip(np.asarray(action, dtype=np.float64).reshape(ACTION_DIM), -1.0, 1.0)

        # 1. Integrate the Cartesian target
        self._target_pos = np.clip(
            self._target_pos + act[:3] * self.env_config.action_pos_scale, self.ws_low, self.ws_high
        )
        self._grip_cmd = act[3]

        # 2. Solve IK once per control step, then hold the joint targets for
        #    `n_substeps` physics steps.  Re-solving every physics step would be
        #    25x more expensive for almost no benefit: the position servo inside
        #    MuJoCo already interpolates.
        q_target = self.ctrl.solve(self.data, self._target_pos, DOWN_MAT)
        self.data.ctrl[:7] = q_target
        # actuator8 maps [0, 255] -> [0, 0.04] m per finger.
        self.data.ctrl[self.grip_act_id] = float(np.clip((act[3] + 1.0) * 0.5, 0.0, 1.0) * 255.0)

        collided = False
        for _ in range(self.env_config.n_substeps):
            mujoco.mj_step(self.model, self.data)
            collided |= self._check_obstacle_contact()
            if render_callback is not None:
                render_callback()

        self._step_count += 1
        self._info.steps = self._step_count
        self._info.collided |= collided
        self._info.grasped |= self._is_grasped()

        obs = self._get_obs()
        terminated, truncated, reward = self._evaluate_termination()

        info = {
            "episode": self._info,
            "success": self._info.success,
            "collided": self._info.collided,
            "grasped": self._info.grasped,
            "is_success": self._info.success,
        }
        if self.debug and self._step_count % 20 == 0:
            print(f"[env.step {self._step_count:3d}] tcp={np.round(self.tcp_pos,3)} "
                  f"cube={np.round(self.cube_pos,3)} grip={self.grip_width:.3f} "
                  f"a={np.round(act,2)} collided={self._info.collided}")
        return obs, reward, terminated, truncated, info

    @property
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_sid].copy()

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_bid].copy()

    @property
    def bin_pos(self) -> np.ndarray:
        return self.data.xpos[self.bin_bid].copy()

    @property
    def grip_width(self) -> float:
        return float(sum(self.data.qpos[a] for a in self.finger_qadr))

    def mat_to_rot6d(self, mat: np.ndarray) -> np.ndarray:
        """
        Flatten a 3x3 rotation into the continuous 6-D representation (first two columns).

        WHY: quaternions have a double cover (q and -q are the same rotation) and
        Euler angles wrap; both create discontinuities that an MSE-trained network
        handles badly. The 6-D representation is continuous and is the standard
        choice for regressing orientations.
        """
        m = np.asarray(mat, dtype=np.float64).reshape(3, 3)
        return np.concatenate([m[:, 0], m[:, 1]]).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        d, m = self.data, self.model
        blocks = {
            "arm_qpos": d.qpos[self.ctrl.qpos_adr],
            "arm_qvel": d.qvel[self.ctrl.dof_adr],
            "grip_width": [self.grip_width],
            "grip_vel": [sum(d.qvel[a] for a in self.finger_vadr)],
            "tcp_pos": d.site_xpos[self.tcp_sid],
            "tcp_rot6d": self.mat_to_rot6d(d.site_xmat[self.tcp_sid]),
            "target_pos": self._target_pos,
            "cube_pos": d.xpos[self.cube_bid],
            "cube_rot6d": self.mat_to_rot6d(d.xmat[self.cube_bid]),
            "cube_rel_tcp": d.xpos[self.cube_bid] - d.site_xpos[self.tcp_sid],
            "cube_rgb": self._cube_rgb,
            "bin_pos": d.xpos[self.bin_bid],
            "cube_rel_bin": d.xpos[self.cube_bid] - d.xpos[self.bin_bid],
            "obstacle_pos": d.xpos[self.obstacle_bid],
            "obstacle_half": m.geom_size[self.obstacle_gid],
        }
        return np.concatenate(
            [np.asarray(blocks[n], dtype=np.float32).ravel() for n, _ in OBS_SPEC]
        ).astype(np.float32)

    def _is_grasped(self) -> bool:
        """True when both fingertips are in contact with the cube."""
        touching = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self.cube_gid and g2 in self.robot_gids:
                touching.add(g2)
            elif g2 == self.cube_gid and g1 in self.robot_gids:
                touching.add(g1)
        return len(touching) >= 2 and self.grip_width < 0.075

    def _check_obstacle_contact(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = (c.geom1, c.geom2)
            if self.obstacle_gid in pair:
                other = pair[0] if pair[1] == self.obstacle_gid else pair[1]
                if other in self.robot_gids or other == self.cube_gid:
                    return True
        return False

    def _in_bin(self) -> bool:
        cube = self.cube_pos
        bin_p = self.bin_pos
        half = self.env_config.bin_inner_half
        inside_xy = np.all(np.abs(cube[:2] - bin_p[:2]) < half)
        low_enough = cube[2] < self.env_config.success_height
        at_rest = np.linalg.norm(self.data.qvel[self.cube_vadr: self.cube_vadr + 3]) < \
            self.env_config.success_settle_vel
        return bool(inside_xy and low_enough and at_rest)

    def _evaluate_termination(self) -> tuple[bool, bool, float]:
        """
        Returns (terminated, truncated, reward).

        NOTE: behavioural cloning never uses `reward`. It is computed anyway
        because it is a cheap, interpretable scalar for debugging.
        """
        if self._in_bin():
            self._success_counter += 1
        else:
            self._success_counter = 0

        if self._success_counter >= self.env_config.success_hold_steps:
            self._info.success = True
            return True, False, 1.0

        if self.cube_pos[2] < self.env_config.drop_height:
            self._info.dropped = True
            return True, False, -1.0

        truncated = self._step_count >= self.env_config.max_steps
        # dense shaping term, purely informational
        dist = float(np.linalg.norm(self.cube_pos - self.bin_pos))
        return False, truncated, -0.01 * dist

    def render(self, camera: str | None = None) -> np.ndarray:
        """
        Offscreen RGB frame. Requires a GL backend:
            export MUJOCO_GL=egl      # headless machine with a GPU
            export MUJOCO_GL=osmesa   # headless machine without a GPU
            export MUJOCO_GL=glfw     # desktop with a display
        """
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.env_config.render_height, width=self.env_config.render_width
            )
        self._renderer.update_scene(self.data, camera=camera or self.env_config.render_camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
