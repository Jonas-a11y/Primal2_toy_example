"""Entrypoint for training."""
from __future__ import annotations

import argparse
import time

from .config import TrainConfig
from .trainer import Trainer


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
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--n-agents", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        total_episodes=args.episodes,
        device=args.device,
        log_every_episodes=args.log_every,
        ckpt_every_episodes=args.ckpt_every,
        n_agents=args.n_agents,
        num_workers=args.num_workers,
    )
    tr = Trainer(cfg, seed=args.seed)
    if args.resume:
        tr.load_checkpoint(args.resume)
        tr._log(f"resumed from {args.resume} at ep {tr.episode}")
    deadline_ts = None
    if args.deadline_hours is not None:
        deadline_ts = time.time() + args.deadline_hours * 3600
    tr.run(deadline_ts=deadline_ts)


if __name__ == "__main__":
    main()
