"""
Closed-loop evaluation.

What is measured
----------------
* success rate: cube resting inside the bin, at rest, for several steps
* obstacle contact rate: did any robot link or the cube touch the wall
* grasp rate: did the fingers ever close on the cube
* steps to success: how much slower than the expert (~98 steps)
* per-colour breakdown, including the two hues held out of training

The held-out colours are the cheapest generalisation probe available. Colour
is a pure nuisance variable here: the correct policy is exactly invariant to
it. If success drops on magenta and cyan, the network has latched onto a
spurious feature, which is a miniature version of the failure that ruins real
robot policies (lighting, table texture, camera pose).

Fixed evaluation seeds
----------------------
Every configuration is evaluated on the *same* list of seeds, so two policies
face identical cube poses, bin jitter and colours.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import Config
from env import PandaPickPlaceEnv
from expert import ScriptedExpert, perform_expert_action
from policy import PolicyRunner, load_checkpoint


def rollout_policy(env: PandaPickPlaceEnv, runner: PolicyRunner, seed: int,
                   record: bool = False, camera: str | None = None):
    obs, _ = env.reset(seed=seed)
    runner.reset()
    frames, traj, actions = [], [], []
    done = False
    info: dict = {}
    while not done:
        a = runner.act(obs)
        actions.append(a.copy())
        traj.append(env.tcp_pos.copy())
        if record:
            frames.append(env.render(camera))
        obs, _, terminated, truncated, info = env.step(a)
        done = terminated or truncated
    ep = info["episode"]
    return {
        "success": bool(ep.success),
        "collided": bool(ep.collided),
        "grasped": bool(ep.grasped),
        "dropped": bool(ep.dropped),
        "length": ep.steps,
        "color": ep.color_name,
        "frames": frames,
        "tcp_traj": np.asarray(traj),
        "actions": np.asarray(actions),
        "cube_pos": env.cube_pos.copy(),
        "bin_pos": env.bin_pos.copy(),
    }


def evaluate(
    cfg: Config,
    ckpt: str | None = None,
    n_episodes: int | None = None,
    record_videos: int | None = None,
    verbose: bool = True,
    runner: PolicyRunner | None = None,
) -> dict:
    ec = cfg.eval
    n = n_episodes if n_episodes is not None else ec.n_episodes
    n_vid = record_videos if record_videos is not None else ec.render_n_videos

    if runner is None:
        net, obs_norm, act_norm, ck_cfg, _ = load_checkpoint(ckpt or cfg.train.ckpt_path)
        runner = PolicyRunner(net, obs_norm, act_norm, ck_cfg.policy,
                              temporal_ensemble=ec.temporal_ensemble)

    env = PandaPickPlaceEnv(
        cfg.env, cfg.ik, seed=ec.seed,
        eval_mode=ec.include_heldout_colors, verbose=verbose,
    )

    seeds = [ec.seed + i for i in range(n)]
    per_color: dict[str, list[bool]] = defaultdict(list)
    results, videos = [], []

    for i, s in enumerate(seeds):
        want_video = i < n_vid
        r = rollout_policy(env, runner, s, record=want_video)
        if want_video and r["frames"]:
            videos.append({"frames": r["frames"], "success": r["success"],
                           "color": r["color"], "seed": s})
        r.pop("frames")
        results.append(r)
        per_color[r["color"]].append(r["success"])
        if verbose and cfg.debug:
            print(f"[eval] seed={s} success={r['success']} grasped={r['grasped']} "
                  f"collided={r['collided']} len={r['length']} color={r['color']}")

    env.close()

    succ = np.array([r["success"] for r in results])
    grasp = np.array([r["grasped"] for r in results])
    coll = np.array([r["collided"] for r in results])
    lens = np.array([r["length"] for r in results])

    summary = {
        "n": n,
        "success_rate": float(succ.mean()),
        "grasp_rate": float(grasp.mean()),
        "collision_rate": float(coll.mean()),
        "mean_steps_on_success": float(lens[succ].mean()) if succ.any() else float("nan"),
        "per_color": {k: float(np.mean(v)) for k, v in sorted(per_color.items())},
        "per_color_n": {k: len(v) for k, v in sorted(per_color.items())},
        # Failure taxonomy: where in the pipeline does it break?
        "failed_to_grasp": float(np.mean(~grasp)),
        "grasped_but_failed": float(np.mean(grasp & ~succ)),
    }

    if verbose:
        _report(summary, cfg)
    return {"summary": summary, "results": results, "videos": videos, "runner": runner}


def _report(s: dict, cfg: Config) -> None:
    se = np.sqrt(s["success_rate"] * (1 - s["success_rate"]) / max(1, s["n"]))
    print("\n" + "=" * 72)
    print("POLICY EVALUATION")
    print("=" * 72)
    print(f"  episodes            : {s['n']}  (fixed seeds from {cfg.eval.seed})")
    print(f"  SUCCESS RATE        : {s['success_rate']:.1%}  +- {se:.1%} (1 sigma)")
    print(f"  grasp rate          : {s['grasp_rate']:.1%}")
    print(f"  obstacle contacts   : {s['collision_rate']:.1%}")
    print(f"  steps when successful: {s['mean_steps_on_success']:.1f}  (expert ~98)")
    print("\n  failure breakdown:")
    print(f"    never grasped the cube      : {s['failed_to_grasp']:.1%}")
    print(f"    grasped but did not place   : {s['grasped_but_failed']:.1%}")
    print("\n  per-colour success (held-out hues marked *):")
    held = set(cfg.env.heldout_colors)
    for k, v in s["per_color"].items():
        mark = "*" if k in held else " "
        print(f"    {mark}{k:<9} {v:6.1%}   (n={s['per_color_n'][k]})")
    print("=" * 72 + "\n")


def evaluate_expert(cfg: Config, n_episodes: int = 20, verbose: bool = True) -> dict:
    """Upper bound: run the scripted expert under the evaluation distribution."""
    env = PandaPickPlaceEnv(cfg.env, cfg.ik, seed=cfg.eval.seed,
                            eval_mode=True, verbose=False)
    expert = ScriptedExpert(cfg.expert, cfg.env)
    succ, lens = [], []
    for i in range(n_episodes):
        ep = perform_expert_action(env, expert, seed=cfg.eval.seed + i)
        succ.append(ep["success"])
        lens.append(ep["length"])
    env.close()
    out = {"success_rate": float(np.mean(succ)), "mean_steps": float(np.mean(lens))}
    if verbose:
        print(f"[eval] EXPERT reference: success={out['success_rate']:.1%} "
              f"mean_steps={out['mean_steps']:.1f} over {n_episodes} episodes")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--no-ensemble", action="store_true")
    ap.add_argument("--videos", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    cfg.debug = args.debug
    if args.no_ensemble:
        cfg.eval.temporal_ensemble = False
    out = evaluate(cfg, ckpt=args.ckpt, n_episodes=args.episodes,
                   record_videos=args.videos)
    Path(cfg.eval.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.eval.out_dir) / "summary.json", "w") as f:
        json.dump(out["summary"], f, indent=2)
