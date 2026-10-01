"""Watch the expert or trained policy control the Panda, in MuJoCo's interactive viewer window."""

from __future__ import annotations

import argparse
import time

import mujoco.viewer

from config import Config
from env import PandaPickPlaceEnv
from expert import ScriptedExpert
from policy import PolicyRunner, load_checkpoint


def watch(
    ckpt: str | None,
    episodes: int,
    seed: int,
    speed: float,
    colors: list[str] | None,
) -> None:
    runner, env, cfg = None, None, None

    if ckpt:
        net, obs_norm, act_norm, cfg, _ = load_checkpoint(ckpt)
        runner = PolicyRunner(
            net, obs_norm, act_norm, cfg.policy,
            temporal_ensemble=cfg.eval.temporal_ensemble,
        )

        print(f"Loaded policy from '{ckpt}'.")
    else:
        cfg = Config()
        runner = ScriptedExpert(cfg.expert, cfg.env)
        print("Initialised scripted expert.")

    env = PandaPickPlaceEnv(
        cfg.env, cfg.ik, seed=seed, eval_mode=True, verbose=True
    )
    print(f"Opening MuJoCo viewer...\n")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(episodes):
            if not viewer.is_running():
                break
            options = {"color": colors[ep % len(colors)]} if colors else None
            obs, info = env.reset(seed=seed + ep, options=options)
            runner.reset()
            print(f"[Episode {ep+1}: cube color = {info['color']}")

            done = False
            step_info: dict = {}
            while viewer.is_running() and not done:
                t0 = time.time()
                action = runner.act(obs)
                if type(runner) is ScriptedExpert:
                    action = action[0]

                # render_callback=viewer.sync redraws after every physics
                # substep (500 Hz), not just once per 20 Hz control step.
                obs, reward, terminated, truncated, step_info = env.step(
                    action, render_callback=viewer.sync
                )
                done = terminated or truncated

                # Pace to (roughly) real time. MuJoCo simulates far faster
                # than 20 Hz, so without this the whole episode would fly by
                # in a fraction of a second.
                target_dt = env.env_config.control_dt / max(speed, 1e-3)
                elapsed = time.time() - t0
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

            if step_info:
                outcome = "SUCCESS" if step_info["success"] else "did not succeed"
                print(f"           -> {outcome} in {step_info['episode'].steps} steps\n")

    env.close()
    print("MuJoCo viewer is closed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=str, default=None, help="omit to run the scripted expert")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time, 0.5 = half speed, 2.0 = double speed")
    ap.add_argument("--colors", type=str, nargs="*", default=None, help="cycle these cube colors, e.g. --colors magenta cyan")
    args = ap.parse_args()
    watch(args.ckpt, args.episodes, args.seed, args.speed, args.colors)
