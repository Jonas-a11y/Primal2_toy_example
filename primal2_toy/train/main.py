"""Entrypoint for training."""
from __future__ import annotations

import argparse
import time

from .config import TrainConfig
from .trainer import Trainer


# Paper (Section V.B.2): size ∈ [10, 70], density ∈ [0.2, 0.7], corridor ∈ [3, 21].
PAPER_SIZES = tuple(range(10, 71, 10))
PAPER_CORRIDORS = (3, 5, 7, 10, 15, 21)
PAPER_DENSITY_LOW = 0.2
PAPER_DENSITY_HIGH = 0.7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--deadline", type=str, default=None,
                   help="ISO datetime or unix ts to stop training.")
    p.add_argument("--deadline-hours", type=float, default=None,
                   help="Stop training after this many hours.")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--warmstart-weights", type=str, default=None,
                   help="Load only weights from this ckpt; keep fresh optimizer and episode counter.")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--n-agents", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--paper-ranges", action="store_true",
                   help="Use paper's env-randomization ranges (Section V.B.2): "
                        "sizes 10-70, density 0.2-0.7, corridor 3-21. Overrides individual flags.")
    p.add_argument("--sizes", type=int, nargs="+", default=None,
                   help="Explicit list of world sizes to sample from.")
    p.add_argument("--density-low", type=float, default=None)
    p.add_argument("--density-high", type=float, default=None)
    p.add_argument("--corridors", type=int, nargs="+", default=None,
                   help="Explicit list of typical corridor lengths to sample.")
    p.add_argument("--entropy-weight", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--il-warmup-episodes", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_kwargs = dict(
        total_episodes=args.episodes,
        device=args.device,
        log_every_episodes=args.log_every,
        ckpt_every_episodes=args.ckpt_every,
        n_agents=args.n_agents,
        num_workers=args.num_workers,
    )
    # Env-randomization overrides.
    if args.paper_ranges:
        cfg_kwargs["size_choices"] = PAPER_SIZES
        cfg_kwargs["density_low"] = PAPER_DENSITY_LOW
        cfg_kwargs["density_high"] = PAPER_DENSITY_HIGH
        cfg_kwargs["corridor_choices"] = PAPER_CORRIDORS
    if args.sizes:
        cfg_kwargs["size_choices"] = tuple(args.sizes)
    if args.corridors:
        cfg_kwargs["corridor_choices"] = tuple(args.corridors)
    if args.density_low is not None:
        cfg_kwargs["density_low"] = args.density_low
    if args.density_high is not None:
        cfg_kwargs["density_high"] = args.density_high
    if args.entropy_weight is not None:
        cfg_kwargs["entropy_weight"] = args.entropy_weight
    if args.lr is not None:
        cfg_kwargs["lr"] = args.lr
    if args.il_warmup_episodes is not None:
        cfg_kwargs["il_warmup_episodes"] = args.il_warmup_episodes
    cfg = TrainConfig(**cfg_kwargs)
    tr = Trainer(cfg, seed=args.seed)
    if args.resume:
        tr.load_checkpoint(args.resume)
        tr._log(f"resumed from {args.resume} at ep {tr.episode}")
    elif args.warmstart_weights:
        tr.load_checkpoint(args.warmstart_weights, weights_only=True)
        tr._log(f"warm-started weights from {args.warmstart_weights} (fresh optim, ep=0)")
    tr._log(f"env ranges: size={cfg.size_choices} density=[{cfg.density_low},{cfg.density_high}] "
            f"corridor={cfg.corridor_choices} n_agents={cfg.n_agents}")
    deadline_ts = None
    if args.deadline_hours is not None:
        deadline_ts = time.time() + args.deadline_hours * 3600
    tr.run(deadline_ts=deadline_ts)


if __name__ == "__main__":
    main()
