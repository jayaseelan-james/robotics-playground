"""
Demonstration collection.

A single compressed `.npz` holding the concatenation of all episodes plus an
index array so episode boundaries survive the flattening:

    obs        (N, obs_dim)   float32
    action     (N, act_dim)   float32
    phase      (N,)           int64   -- auxiliary label
    ep_start   (E,)           int64
    ep_len     (E,)           int64
    ep_color   (E,)           <U16
    config     ()             JSON string

Flat arrays + an index is deliberately chosen over a list of ragged arrays: it
loads with one mmap, indexes in O(1), and makes the chunking logic in
`dataset.py` a pure slicing operation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from config import Config
from env import ACTION_NAMES, OBS_SPEC, PandaPickPlaceEnv
from expert import N_PHASES, Phase, ScriptedExpert, perform_expert_action


def collect(cfg: Config, verbose: bool = True) -> dict:
    cc = cfg.collect
    env = PandaPickPlaceEnv(cfg.env, cfg.ik, seed=cc.seed, verbose=verbose)
    env.debug = cfg.debug
    expert = ScriptedExpert(cfg.expert, cfg.env, debug=cfg.debug)
    rng = np.random.default_rng(cc.seed)

    obs_chunks, act_chunks, phase_chunks = [], [], []
    ep_start, ep_len, ep_color = [], [], []
    n_fail = n_collide = 0
    cursor = 0
    t0 = time.time()

    for i in range(cc.n_episodes):
        # Only a fraction of episodes are perturbed, so the dataset keeps a core
        # of clean, canonical trajectories alongside the noisy ones.
        noise = cc.action_noise_std if rng.random() < cc.noise_prob else 0.0
        ep = perform_expert_action(env, expert, seed=cc.seed * 100_000 + i,
                            noise_std=noise, rng=rng)

        n_collide += ep["collided"]
        if not ep["success"]:
            n_fail += 1
            if not cc.keep_failures:
                if verbose:
                    print(f"[collect] ep {i+1:4d} DISCARDED (collided={ep['collided']}) color={ep['color']}")
                continue

        obs_chunks.append(ep["obs"])
        act_chunks.append(ep["action"])
        phase_chunks.append(ep["phase"])
        ep_start.append(cursor)
        ep_len.append(ep["length"])
        ep_color.append(ep["color"])
        cursor += ep["length"]

        if verbose and (i + 1) % max(1, cc.n_episodes // 10) == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"[collect] {i+1:4d}/{cc.n_episodes} episodes | kept={len(ep_len)} "
                  f"| {rate:5.1f} ep/s | transitions={cursor}")

    env.close()
    if not ep_len:
        raise RuntimeError("no successful demonstrations collected -- check the expert")

    data = {
        "obs": np.concatenate(obs_chunks).astype(np.float32),
        "action": np.concatenate(act_chunks).astype(np.float32),
        "phase": np.concatenate(phase_chunks).astype(np.int64),
        "ep_start": np.asarray(ep_start, dtype=np.int64),
        "ep_len": np.asarray(ep_len, dtype=np.int64),
        "ep_color": np.asarray(ep_color),
        "config": json.dumps(cfg.to_dict(), default=str),
    }

    out = Path(cc.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)

    if verbose:
        _report(data, cc.n_episodes, n_fail, n_collide, time.time() - t0, out)
    return data


def _report(data, n_req, n_fail, n_collide, dt, out: Path) -> None:
    obs, act, phase = data["obs"], data["action"], data["phase"]
    n_kept = len(data["ep_len"])
    print("\n" + "=" * 72)
    print(f"DEMONSTRATIONS  ->  {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print("=" * 72)
    print(f"  requested episodes : {n_req}")
    print(f"  expert failures    : {n_fail}  ({n_fail/max(1,n_req):.1%})")
    print(f"  obstacle contacts  : {n_collide}")
    print(f"  episodes kept      : {n_kept}")
    print(f"  transitions        : {len(obs)}  (mean {len(obs)/n_kept:.1f} steps/episode)")
    print(f"  wall clock         : {dt:.1f}s  ({n_req/dt:.1f} episodes/s)")

    print("\n  phase histogram (share of all transitions):")
    counts = np.bincount(phase, minlength=N_PHASES)
    for p in range(N_PHASES):
        if counts[p]:
            bar = "#" * int(40 * counts[p] / counts.max())
            print(f"    {Phase(p).name:<8} {counts[p]:6d}  {counts[p]/len(phase):5.1%} {bar}")

    print("\n  colour balance:")
    names, cnt = np.unique(data["ep_color"], return_counts=True)
    print("    " + "  ".join(f"{n}={c}" for n, c in zip(names, cnt)))

    print("\n  action statistics (these are the MSE regression targets):")
    for j, name in enumerate(ACTION_NAMES):
        a = act[:, j]
        sat = np.mean(np.abs(a) > 0.99)
        print(f"    {name:<5} mean={a.mean():+.3f} std={a.std():.3f} "
              f"min={a.min():+.2f} max={a.max():+.2f} saturated={sat:.1%}")

    # A near-zero-variance observation block is dead weight at best and a
    # divide-by-zero in normalisation at worst; flag it early.
    print("\n  low-variance observation blocks (candidates for removal):")
    off, flagged = 0, []
    for name, n in OBS_SPEC:
        blk = obs[:, off:off + n]
        if blk.std(axis=0).max() < 1e-4:
            flagged.append(name)
        off += n
    print("    " + (", ".join(flagged) if flagged else "none"))
    print("=" * 72 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    if args.episodes is not None:
        cfg.collect.n_episodes = args.episodes
    if args.noise is not None:
        cfg.collect.action_noise_std = args.noise
    if args.out:
        cfg.collect.out_path = args.out
    if args.seed is not None:
        cfg.collect.seed = args.seed
    cfg.debug = args.debug
    collect(cfg)
