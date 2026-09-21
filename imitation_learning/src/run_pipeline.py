#!/usr/bin/env python3
"""
End-to-end driver: environment -> expert -> demos -> training -> evaluation -> plots.

Usage:
    python run_pipeline.py --smoke              # ~2 min, proves the wiring works
    python run_pipeline.py                      # full run (200 demos, 150 epochs)
    python run_pipeline.py --ablate             # full run + the ablation table

Rendering (videos/filmstrips) needs a GL backend:
    MUJOCO_GL=osmesa python run_pipeline.py     # headless, no GPU
    MUJOCO_GL=egl    python run_pipeline.py     # headless, with GPU
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from collect_demos import collect
from config import Config, smoke_config
from env import PandaPickPlaceEnv
from evaluate import evaluate, evaluate_expert, rollout_policy
from expert import ScriptedExpert, perform_expert_action
from train import train


def banner(title: str) -> None:
    print("\n" + "#" * 72)
    print(f"# {title}")
    print("#" * 72)


def run(cfg: Config, out_dir: Path, do_video: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    banner("1/5  COLLECT DEMONSTRATIONS")
    collect(cfg)

    banner("2/5  TRAIN THE POLICY (behavioural cloning, MSE)")
    tr = train(cfg)

    banner("3/5  EVALUATE CLOSED-LOOP")
    expert_ref = evaluate_expert(cfg, n_episodes=min(20, cfg.eval.n_episodes))
    ev = evaluate(cfg, record_videos=cfg.eval.render_n_videos if do_video else 0)

    banner("4/5  QUALITATIVE COMPARISON")
    env = PandaPickPlaceEnv(cfg.env, cfg.ik, seed=cfg.eval.seed,
                            eval_mode=True, verbose=False)
    expert = ScriptedExpert(cfg.expert, cfg.env)
    seed = cfg.eval.seed
    exp_ep = perform_expert_action(env, expert, seed=seed)
    exp_ep["tcp_traj"] = np.zeros((0, 3))     # filled below for a like-for-like plot
    # re-run the expert while logging the TCP path
    obs, _ = env.reset(seed=seed)
    expert.reset()
    traj, acts, done = [], [], False
    while not done:
        a, _ = expert.act(obs)
        traj.append(env.tcp_pos.copy())
        acts.append(a.copy())
        obs, _, term, trunc, _ = env.step(a)
        done = term or trunc
    exp_ep = {"tcp_traj": np.asarray(traj), "actions": np.asarray(acts),
              "bin_pos": env.bin_pos.copy()}

    pol_ep = rollout_policy(env, ev["runner"], seed=seed)
    env.close()

    banner("5/5  SUMMARY")
    s = ev["summary"]
    print(f"  expert success   : {expert_ref['success_rate']:.1%}")
    print(f"  POLICY success   : {s['success_rate']:.1%}")
    print(f"  grasp / collision: {s['grasp_rate']:.1%} / {s['collision_rate']:.1%}")
    print(f"  total wall clock : {time.time()-t_start:.1f}s")

    report = {
        "expert": expert_ref,
        "policy": s,
        "train_best": tr["best"],
        "config": cfg.to_dict(),
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def ablate(base: Config, out_dir: Path, n_eval: int = 40) -> None:
    """
    Retrain from the same demonstrations under several settings and compare
    closed-loop success. Each variant reuses the demo file, so the only thing
    that changes is the learning side.
    """
    banner("ABLATION STUDY")
    variants = {
        "full (chunk=12, EMA, grip_w=3)": {},
        "no action chunking (chunk=1)": {"policy.chunk_len": 1},
        "short chunk (chunk=4)": {"policy.chunk_len": 4},
        "no EMA": {"train.ema_decay": 0.0},
        "uniform action loss (grip_w=1)": {"train.gripper_loss_weight": 1.0},
        "with observation history (h=3)": {"policy.obs_history": 3},
    }
    rows = []
    for name, overrides in variants.items():
        cfg = Config.from_dict(json.loads(base.to_json()))
        for key, val in overrides.items():
            sec, field = key.split(".")
            setattr(getattr(cfg, sec), field, val)
        cfg.train.ckpt_path = str(out_dir / f"ablation_{len(rows)}.pt")
        print(f"\n--- {name}")
        train(cfg, verbose=False)
        ev = evaluate(cfg, n_episodes=n_eval, record_videos=0, verbose=False)
        s = ev["summary"]
        rows.append((name, s["success_rate"], s["grasp_rate"], s["collision_rate"]))
        print(f"    success={s['success_rate']:.1%}  grasp={s['grasp_rate']:.1%}")

    print("\n" + "=" * 78)
    print(f"{'variant':<34}{'success':>10}{'grasp':>10}{'wall hit':>10}")
    print("-" * 78)
    for name, su, gr, co in rows:
        print(f"{name:<34}{su:>9.1%}{gr:>10.1%}{co:>10.1%}")
    print("=" * 78)
    with open(out_dir / "ablation.json", "w") as f:
        json.dump([{"variant": n, "success": s, "grasp": g, "collision": c}
                   for n, s, g, c in rows], f, indent=2)


def data_scaling(base: Config, out_dir: Path, sizes=(25, 50, 100, 200), n_eval=40):
    """Success rate vs number of demonstrations."""
    banner("DATA SCALING")
    rows = []
    for n in sizes:
        cfg = Config.from_dict(json.loads(base.to_json()))
        cfg.collect.n_episodes = n
        cfg.collect.out_path = str(out_dir / f"demos_{n}.npz")
        cfg.train.ckpt_path = str(out_dir / f"policy_{n}.pt")
        collect(cfg, verbose=False)
        train(cfg, verbose=False)
        ev = evaluate(cfg, n_episodes=n_eval, record_videos=0, verbose=False)
        rows.append((n, ev["summary"]["success_rate"], ev["summary"]["grasp_rate"]))
        print(f"  {n:4d} demos -> success {rows[-1][1]:.1%}  grasp {rows[-1][2]:.1%}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", label="success")
    ax.plot([r[0] for r in rows], [r[2] for r in rows], "s--", label="grasp")
    ax.set_xlabel("number of demonstrations")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("Behavioural cloning data scaling")
    fig.tight_layout()
    fig.savefig(out_dir / "data_scaling.png", dpi=130)
    plt.close(fig)
    with open(out_dir / "data_scaling.json", "w") as f:
        json.dump([{"n": n, "success": s, "grasp": g} for n, s, g in rows], f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--out", type=str, default="output/pipeline")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = smoke_config() if args.smoke else Config()
    if args.episodes is not None:
        cfg.collect.n_episodes = args.episodes
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.eval_episodes is not None:
        cfg.eval.n_episodes = args.eval_episodes
    cfg.debug = args.debug

    out_dir = Path(args.out)
    cfg.collect.out_path = str(out_dir / "demos.npz")
    cfg.train.ckpt_path = str(out_dir / "policy.pt")
    cfg.eval.out_dir = str(out_dir)

    run(cfg, out_dir, do_video=not args.no_video)
    if args.ablate:
        ablate(cfg, out_dir)
    if args.scaling:
        data_scaling(cfg, out_dir)
