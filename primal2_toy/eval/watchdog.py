"""Watchdog: whenever a fresh 'latest' checkpoint arrives, run a quick eval.

Runs indefinitely. Every N seconds:
  - stat checkpoints/primal2_latest.pt
  - if it's newer than the last eval, run headless eval on a fixed set of seeds
  - append the result to logs/eval_over_time.csv

Comparable throughput numbers over time = a training-progress curve.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
import numpy as np
import torch

from primal2_toy.policy.network import PolicyNet
from primal2_toy.obs.builder import ObsSpec
from primal2_toy.eval.rollout import Rollout, build_scenario


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def eval_ckpt(ckpt: str, device: torch.device, seeds: list[int], size: int, density: float,
              corridor_length: int, agents: int, steps: int) -> tuple[float, int, int]:
    obs_spec = ObsSpec(fov=11, n_pred=3)
    net = PolicyNet(
        n_spatial_channels=obs_spec.num_spatial_channels,
        n_scalars=obs_spec.num_scalar_features, n_actions=5, fov=11,
    ).to(device)
    s = torch.load(ckpt, map_location=device, weights_only=False)
    net.load_state_dict(s["model"])
    net.eval()
    total_arrivals = 0
    total_steps = 0
    episode_val = int(s.get("episode", 0))
    for seed in seeds:
        grid, task, corridors, cell_to_id = build_scenario(size, density, corridor_length, agents, seed)
        rollout = Rollout(net, grid, task, corridors, cell_to_id, device, obs_spec, greedy=True)
        for _ in range(steps):
            rollout.step()
        total_arrivals += rollout.total_arrivals
        total_steps += rollout.step_idx
    tp = total_arrivals / max(1, total_steps)
    return tp, total_arrivals, episode_val


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/primal2_latest.pt")
    p.add_argument("--out", type=str, default="logs/eval_over_time.csv")
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123, 555])
    p.add_argument("--size", type=int, default=15)
    p.add_argument("--density", type=float, default=0.3)
    p.add_argument("--corridor-length", type=int, default=5)
    p.add_argument("--agents", type=int, default=6)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--poll-s", type=int, default=30)
    args = p.parse_args()

    device = _resolve_device(args.device)
    last_mtime = 0.0
    if not os.path.exists(args.out):
        with open(args.out, "w", newline="") as f:
            csv.writer(f).writerow(["wall_time", "episode", "throughput", "arrivals", "seeds", "steps_per_seed"])
    while True:
        try:
            mt = os.path.getmtime(args.ckpt)
        except FileNotFoundError:
            time.sleep(args.poll_s); continue
        if mt <= last_mtime + 5:  # 5s dead-band
            time.sleep(args.poll_s); continue
        last_mtime = mt
        try:
            tp, arr, ep = eval_ckpt(args.ckpt, device, args.seeds, args.size, args.density,
                                    args.corridor_length, args.agents, args.steps)
        except Exception as e:
            print(f"eval failed: {e}"); time.sleep(args.poll_s); continue
        wall = time.time()
        with open(args.out, "a", newline="") as f:
            csv.writer(f).writerow([f"{wall:.1f}", ep, f"{tp:.4f}", arr, len(args.seeds), args.steps])
        print(f"[wall={time.strftime('%H:%M:%S')}] ep={ep} throughput={tp:.4f} arrivals={arr}")


if __name__ == "__main__":
    main()
