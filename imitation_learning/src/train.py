"""
Behavioural cloning with an MSE loss.

The Loss:
---------
    L = mean_over(batch, chunk, dim) [ w_dim * mask * (a_pred - a_expert)^2 ]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import Config
from dataset import build_dataloaders
from env import ACTION_NAMES
from policy import ChunkMLPPolicy, save_checkpoint


class EMA:
    """
    Exponential moving average of the weights.

    Free variance reduction: SGD on a noisy objective wanders around the
    optimum, and the average of the last few thousand iterates is usually a
    better model than the final iterate.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.step = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def _decay(self) -> float:
        # Warm up the decay: with a fixed 0.999 the shadow needs ~1000 updates
        # to forget its random initialisation, so a short run would ship noise.
        # This is the standard (1+t)/(10+t) schedule used by timm/diffusers.
        return min(self.decay, (1.0 + self.step) / (10.0 + self.step))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self._decay()
        self.step += 1
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def masked_weighted_mse(pred, target, mask, dim_w):
    """pred/target: (B,H,A); mask: (B,H); dim_w: (A,)."""
    se = (pred - target) ** 2 * dim_w            # (B,H,A)
    se = se.mean(-1) * mask                      # (B,H)
    return se.sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate_loss(net, loader, dim_w, device):
    net.eval()
    tot = n = 0.0
    per_dim = torch.zeros(len(ACTION_NAMES), device=device)
    for o, a, m, _ in loader:
        o, a, m = o.to(device), a.to(device), m.to(device)
        p = net(o)
        tot += masked_weighted_mse(p, a, m, dim_w).item() * m.sum().item()
        per_dim += (((p - a) ** 2).mean(1) * 1.0).sum(0)
        n += m.sum().item()
    net.train()
    return tot / max(n, 1.0), (per_dim / max(len(loader.dataset), 1)).cpu().numpy()


def train(cfg: Config, demo_path: str | None = None, verbose: bool = True) -> dict:
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = pick_device(cfg.train.device)

    train_loader, val_loader, obs_norm, act_norm, meta = build_dataloaders(
        cfg, demo_path, verbose=verbose
    )

    net = ChunkMLPPolicy(meta["obs_dim"], meta["action_dim"], cfg.policy).to(device)
    ema = EMA(net, cfg.train.ema_decay)

    # No weight decay on norms/biases: decaying a LayerNorm gain shrinks the
    # representation rather than regularising it.
    decay, no_decay = [], []
    for name, p in net.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)

    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.train.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.train.lr,
    )

    dim_w = torch.ones(meta["action_dim"], device=device)
    dim_w[-1] = cfg.train.gripper_loss_weight
    dim_w = dim_w / dim_w.mean()                  # keep the overall loss scale ~1

    total_steps = cfg.train.epochs * max(1, len(train_loader))
    warmup = int(cfg.train.warmup_frac * total_steps)

    if verbose:
        print(f"\n[train] device={device} params={net.n_params():,} "
              f"steps={total_steps} warmup={warmup}")
        print(f"[train] action loss weights: "
              f"{dict(zip(ACTION_NAMES, np.round(dim_w.cpu().numpy(), 3)))}\n")

    hist = {"epoch": [], "train_loss": [], "val_loss": [], "ema_val_loss": [], "lr": []}
    best = {"val": float("inf"), "epoch": -1}
    step = 0
    t0 = time.time()

    for epoch in range(cfg.train.epochs):
        net.train()
        running, nb = 0.0, 0
        for o, a, m, _ in train_loader:
            o, a, m = o.to(device), a.to(device), m.to(device)

            lr = cfg.train.lr * cosine_with_warmup(step, total_steps, warmup)
            for g in opt.param_groups:
                g["lr"] = lr

            loss = masked_weighted_mse(net(o), a, m, dim_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Gradient clipping is cheap insurance: a single outlier batch early
            # in training can otherwise blow the LayerNorm statistics apart.
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.train.grad_clip)
            opt.step()
            ema.update(net)

            running += loss.item()
            nb += 1
            step += 1

        tr = running / max(nb, 1)
        val, val_per_dim = evaluate_loss(net, val_loader, dim_w, device)
        ema_val, _ = evaluate_loss(ema.shadow, val_loader, dim_w, device)

        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr)
        hist["val_loss"].append(val)
        hist["ema_val_loss"].append(ema_val)
        hist["lr"].append(lr)

        if ema_val < best["val"]:
            best = {"val": ema_val, "epoch": epoch}

        if verbose and (epoch % cfg.train.log_every == 0 or epoch == cfg.train.epochs - 1):
            pd = "  ".join(f"{n}={v:.4f}" for n, v in zip(ACTION_NAMES, val_per_dim))
            print(f"[train] epoch {epoch:3d}/{cfg.train.epochs}  "
                  f"train={tr:.5f}  val={val:.5f}  ema_val={ema_val:.5f}  "
                  f"lr={lr:.2e}  |  per-dim val MSE: {pd}")

    # The final EMA weights are saved.
    final = ema.shadow
    ck_path = Path(cfg.train.ckpt_path)
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(ck_path, final, obs_norm, act_norm, cfg,
                    extra={"history": hist, "meta": meta, "best": best})

    if verbose:
        print(f"\n[train] done in {time.time()-t0:.1f}s | best ema val loss "
              f"{best['val']:.5f} @ epoch {best['epoch']}")
        print(f"[train] checkpoint -> {ck_path}")
        print("[train] NOTE: validation MSE is diagnostic only. "
              "Run evaluate.py for the number that matters.\n")

    return {"history": hist, "meta": meta, "ckpt": str(ck_path), "best": best}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--history", type=int, default=None)
    ap.add_argument("--grip-weight", type=float, default=None)
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()

    config = Config()
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.chunk is not None:
        config.policy.chunk_len = args.chunk
    if args.history is not None:
        config.policy.obs_history = args.history
    if args.grip_weight is not None:
        config.train.gripper_loss_weight = args.grip_weight
    if args.ckpt:
        config.train.ckpt_path = args.ckpt
    out = train(config, args.demos)
    print(json.dumps(out["best"], indent=2))
