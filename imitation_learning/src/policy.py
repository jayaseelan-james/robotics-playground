"""
The policy network.

Architecture:
-------------
A plain MLP: LayerNorm -> Linear -> GELU -> Dropout, three hidden layers, and a
head that emits `chunk_len * action_dim` numbers reshaped to (H, 4).

* **LayerNorm on the input.**  Combined with the z-scoring in `dataset.py` this
  is belt-and-braces, but it makes the network robust to a distribution shift
  between the normaliser's statistics and the states actually visited at test
  time (which, by the nature of covariate shift, is guaranteed to happen).

* **Dropout only in the trunk, never on the output head.**  Dropping output
  units of a regression head injects noise directly into the predicted action.

Temporal Ensembling:
--------------------
At step t the policy has predictions for a_t from the chunk issued at t, at
t-1, ..., at t-H+1. Averaging them with exponentially decaying weights
(newest = highest weight) gives a much smoother command than executing one
chunk open-loop.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from config import PolicyConfig, Config
from dataset import Normalizer


class ChunkMLPPolicy(nn.Module):
    """obs -> (chunk_len, action_dim) in normalised action units."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: PolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk_len = cfg.chunk_len

        layers: list[nn.Module] = []
        if cfg.layer_norm:
            layers.append(nn.LayerNorm(obs_dim))
        d = obs_dim
        for h in cfg.hidden_sizes:
            layers += [nn.Linear(d, h), nn.GELU()]
            if cfg.layer_norm:
                layers.append(nn.LayerNorm(h))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            d = h
        self.mlp_trunk = nn.Sequential(*layers)                  # learned represenatation
        self.action_head = nn.Linear(d, cfg.chunk_len * action_dim)    # action prediction

        # Small output weights so the network starts near "predict the mean
        # action", which after normalisation is zero. Starting with large random
        # actions wastes the first epochs and can destabilise LayerNorm.
        nn.init.uniform_(self.action_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.mlp_trunk(obs)
        return self.action_head(h).view(-1, self.chunk_len, self.action_dim)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PolicyRunner:
    """
    Wraps a trained network for closed-loop control.
    """

    def __init__(
        self,
        net: ChunkMLPPolicy,
        obs_norm: Normalizer,
        act_norm: Normalizer,
        cfg: PolicyConfig,
        device: str = "cpu",
        temporal_ensemble: bool = True,
    ) -> None:
        self.net = net.to(device).eval()
        self.obs_norm = obs_norm
        self.act_norm = act_norm
        self.cfg = cfg
        self.device = device
        self.temporal_ensemble = temporal_ensemble
        self.reset()

    def reset(self) -> None:
        self._hist: deque[np.ndarray] = deque(maxlen=self.cfg.obs_history)
        # Row k holds the prediction for the current step made k steps ago.
        self._pending: list[np.ndarray] = []

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """One observation in, one executable action (range [-1, 1]) out."""
        o = self.obs_norm(obs.astype(np.float32))
        if not self._hist:
            for _ in range(self.cfg.obs_history):
                self._hist.append(o)
        else:
            self._hist.append(o)
        x = torch.from_numpy(np.concatenate(list(self._hist))[None]).float().to(self.device)

        chunk = self.net(x)[0].cpu().numpy()             # (H, A), normalised
        chunk = self.act_norm.inverse(chunk)             # back to [-1, 1] units

        if not self.temporal_ensemble:
            return np.clip(chunk[0], -1.0, 1.0)

        # Temporal ensembling
        self._pending.append(chunk)
        if len(self._pending) > self.cfg.chunk_len:
            self._pending.pop(0)

        # The chunk issued i steps ago contributes its element i.
        preds, weights = [], []
        n = len(self._pending)
        for age, ch in enumerate(reversed(self._pending)):   # Reversed to match index as age.
            if age < len(ch):
                preds.append(ch[age])
                weights.append(np.exp(-self.cfg.ensemble_decay * age))
        w = np.asarray(weights)[:, None]
        action = (np.asarray(preds) * w).sum(0) / w.sum()
        return np.clip(action, -1.0, 1.0)


def save_checkpoint(path, net, obs_norm, act_norm, cfg, extra=None) -> None:
    """A checkpoint must be self-sufficient: weights alone are useless here."""
    torch.save(
        {
            "model": net.state_dict(),
            "obs_norm": obs_norm.state_dict(),
            "act_norm": act_norm.state_dict(),
            "obs_dim": net.obs_dim,
            "action_dim": net.action_dim,
            "config": cfg.to_dict(),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ck["config"])
    net = ChunkMLPPolicy(ck["obs_dim"], ck["action_dim"], cfg.policy)
    net.load_state_dict(ck["model"])
    obs_norm = Normalizer.from_state_dict(ck["obs_norm"])
    act_norm = Normalizer.from_state_dict(ck["act_norm"])
    return net, obs_norm, act_norm, cfg, ck.get("extra", {})
