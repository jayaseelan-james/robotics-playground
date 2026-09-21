"""
Data preparation: normalisation, observation history, action chunking.

Three Transformations:
----------------------

1. NORMALISATION: Raw observation components live on wildly different scales:
   joint velocities reach +-2 rad/s, `grip_width` spans 0.00-0.08 m, colour
   channels are 0-1. An MSE loss with a single learning rate implicitly assumes
   comparable scales; without normalisation the gradient is dominated by the
   large-magnitude blocks and the 8 mm of gripper travel that decides the whole
   task is numerically invisible.

2. OBSERVATION HISTORY: Stacking the previous `obs_history` observations adds
   velocity/intent information. It is OFF by default (history = 1). Briefly: with 
   history, the easiest way to predict a_t is to copy a_{t-1}, which is inferable
   from consecutive observations. This "copycat problem" is one of the most counter-
   intuitive results in imitation learning: more information makes the policy worse.

3. ACTION CHUNKING: Instead of regressing a single action a_t, the policy
   regresses the next `H` actions (a_t ... a_{t+H-1}) from one observation.

   Trade-off: the policy is less reactive. If the cube is nudged mid-chunk,
   the robot keeps executing a stale plan for up to H steps. Temporal
   ensembling recovers most of that reactivity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config


@dataclass
class Normalizer:
    """z-score normaliser that round-trips through a checkpoint."""

    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(x: np.ndarray, eps: float = 1e-3) -> "Normalizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        # Floor, not clip-to-eps: a constant feature becomes exactly 0 after
        # normalisation instead of a huge number.
        std = np.maximum(std, eps)
        return Normalizer(mean.astype(np.float32), std.astype(np.float32))

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            m = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
            s = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
            return (x - m) / s
        return (x - self.mean) / self.std

    def inverse(self, x):
        if isinstance(x, torch.Tensor):
            m = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
            s = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
            return x * s + m
        return x * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @staticmethod
    def from_state_dict(d: dict) -> "Normalizer":
        return Normalizer(np.asarray(d["mean"], np.float32), np.asarray(d["std"], np.float32))


class DemoDataset(Dataset):
    """
    One item = (stacked observation, action chunk, loss mask, phase label).
    """

    def __init__(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        phase: np.ndarray,
        ep_start: np.ndarray,
        ep_len: np.ndarray,
        chunk_len: int,
        obs_history: int,
        obs_norm: Normalizer,
        act_norm: Normalizer,
    ) -> None:
        self.obs = obs
        self.action = action
        self.phase = phase
        self.chunk_len = chunk_len
        self.obs_history = obs_history
        self.obs_norm = obs_norm
        self.act_norm = act_norm

        self.index = np.array(
            [(s, t) for s, L in zip(ep_start, ep_len) for t in range(L)], dtype=np.int64
        )
        self.ep_len_of = np.concatenate([np.full(L, L) for L in ep_len]).astype(np.int64)
        self.obs_dim = obs.shape[1] * obs_history
        self.action_dim = action.shape[1]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        start, t = self.index[i]
        L = self.ep_len_of[i]

        # Observation (+ optional history, clamped at the episode start)
        idx = [start + max(0, t - k) for k in range(self.obs_history - 1, -1, -1)]
        o = self.obs_norm(self.obs[idx]).reshape(-1).astype(np.float32)

        # Action chunk with padding
        H = self.chunk_len
        end = min(t + H, L)
        chunk = self.action[start + t: start + end]
        mask = np.ones(H, dtype=np.float32)
        if len(chunk) < H:
            pad = H - len(chunk)
            # Repeat the final action rather than padding with zeros: zeros are a
            # valid action ("hold still") and would be a wrong training signal.
            # The mask zeroes them out of the loss anyway; this just keeps the
            # target array smooth for debugging plots.
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, axis=0)])
            mask[len(chunk) - pad:] = 0.0
        a = self.act_norm(chunk).astype(np.float32)

        return (
            torch.from_numpy(o),
            torch.from_numpy(a),
            torch.from_numpy(mask),
            torch.tensor(self.phase[start + t], dtype=torch.long),
        )


def load_demos(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    d["config"] = json.loads(str(d["config"]))
    return d


def build_dataloaders(cfg: Config, path: str | Path | None = None, verbose: bool = True):
    """
    Returns (train_loader, val_loader, obs_norm, act_norm, meta).
    """
    d = load_demos(path or cfg.collect.out_path)
    obs, action, phase = d["obs"], d["action"], d["phase"]
    ep_start, ep_len = d["ep_start"], d["ep_len"]
    n_ep = len(ep_len)

    rng = np.random.default_rng(cfg.train.seed)
    order = rng.permutation(n_ep)
    n_val = max(1, int(round(cfg.train.val_frac * n_ep)))
    val_ep, train_ep = order[:n_val], order[n_val:]

    # Statistics from the training episodes only.
    tr_idx = np.concatenate([np.arange(ep_start[e], ep_start[e] + ep_len[e]) for e in train_ep])
    obs_norm = Normalizer.fit(obs[tr_idx])
    act_norm = Normalizer.fit(action[tr_idx])

    def make(eps):
        return DemoDataset(
            obs, action, phase, ep_start[eps], ep_len[eps],
            cfg.policy.chunk_len, cfg.policy.obs_history, obs_norm, act_norm,
        )

    train_ds, val_ds = make(train_ep), make(val_ep)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.train.num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    meta = {
        "obs_dim": train_ds.obs_dim,
        "action_dim": train_ds.action_dim,
        "n_train_ep": len(train_ep),
        "n_val_ep": len(val_ep),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    if verbose:
        print(f"[data] episodes: {meta['n_train_ep']} train / {meta['n_val_ep']} val")
        print(f"[data] transitions: {meta['n_train']} train / {meta['n_val']} val")
        print(f"[data] obs_dim={meta['obs_dim']} (history={cfg.policy.obs_history}) "
              f"action_dim={meta['action_dim']} chunk_len={cfg.policy.chunk_len}")
        rng_obs = obs_norm.std
        print(f"[data] obs std: min={rng_obs.min():.4f} max={rng_obs.max():.4f} "
              f"(constant blocks floored to the epsilon)")
    return train_loader, val_loader, obs_norm, act_norm, meta
