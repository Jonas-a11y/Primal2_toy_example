"""Headless rollout: load a checkpoint, run N steps, print metrics.

Useful for benchmarking a trained policy or as a smoke test in CI-free environments.
"""
from __future__ import annotations

import argparse
import time
import numpy as np
import torch

from primal2_toy.policy.network import PolicyNet
from primal2_toy.obs.builder import ObsSpec
from primal2_toy.eval.rollout import Rollout, build_scenario


def load_net(ckpt_path: str, device: torch.device, obs_spec: ObsSpec) -> PolicyNet:
    net = PolicyNet(
        n_spatial_channels=obs_spec.num_spatial_channels,
        n_scalars=obs_spec.num_scalar_features,
        n_actions=5, fov=obs_spec.fov,
    ).to(device)
    s = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(s["model"])
    net.eval()
    return net


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def evaluate(ckpt: str, size: int, density: float, corridor_length: int,
             agents: int, seeds: list[int], steps: int, device: str) -> dict:
    dev = resolve_device(device)
    obs_spec = ObsSpec(fov=11, n_pred=3)
    net = load_net(ckpt, dev, obs_spec)
    results = []
    for seed in seeds:
        grid, task, corridors, cell_to_id = build_scenario(size, density, corridor_length, agents, seed)
        rollout = Rollout(net, grid, task, corridors, cell_to_id, dev, obs_spec, greedy=True)
        t0 = time.time()
        for _ in range(steps):
            rollout.step()
        elapsed = time.time() - t0
        throughput = rollout.total_arrivals / max(1, rollout.step_idx)
        results.append({
            "seed": seed, "size": size, "density": density, "corridor_length": corridor_length,
            "agents": grid.n_agents, "steps": rollout.step_idx,
            "arrivals": rollout.total_arrivals,
            "throughput": throughput,
            "wall_s": elapsed,
        })
        print(f"seed={seed:5d} size={size:2d} n={grid.n_agents} d={density:.2f} cl={corridor_length} "
              f"steps={rollout.step_idx} arr={rollout.total_arrivals} throughput={throughput:.4f} time={elapsed:.1f}s")
    return {"results": results}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--size", type=int, default=15)
    p.add_argument("--density", type=float, default=0.3)
    p.add_argument("--corridor-length", type=int, default=5)
    p.add_argument("--agents", type=int, default=6)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123, 555, 2024])
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    r = evaluate(args.checkpoint, args.size, args.density, args.corridor_length,
                 args.agents, args.seeds, args.steps, args.device)
    total_arr = sum(x["arrivals"] for x in r["results"])
    total_steps = sum(x["steps"] for x in r["results"])
    print(f"\nAggregate: throughput={total_arr/max(1,total_steps):.4f} "
          f"({total_arr} arrivals / {total_steps} steps across {len(args.seeds)} seeds)")
