"""Random-policy and greedy-A* baselines for comparison against the learned policy.

Both baselines are decentralized to be fair against PRIMAL2.
  - random: uniformly sample a valid action per agent per step.
  - greedy_astar: pick the action along the single-agent shortest path to own goal,
                  ignoring other agents (falls back to stay if that action is invalid).
"""
from __future__ import annotations

import argparse
import numpy as np

from primal2_toy.env.grid import ACTIONS
from primal2_toy.env.corridor import analyze
from primal2_toy.eval.rollout import build_scenario
from primal2_toy.obs.astar import astar_path
from primal2_toy.train.validity import compute_valid_actions


def _random_step(rng, valid) -> np.ndarray:
    n, A = valid.shape
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        choices = np.where(valid[i] > 0)[0]
        if len(choices) == 0:
            out[i] = 0
        else:
            out[i] = int(rng.choice(choices))
    return out


def _greedy_astar_step(grid, task, valid) -> np.ndarray:
    n = grid.n_agents
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        pos = (int(grid.positions[i, 0]), int(grid.positions[i, 1]))
        goal = (int(task.goals[i, 0]), int(task.goals[i, 1]))
        path = astar_path(grid.obstacle_map, pos, goal)
        if not path or len(path) < 2:
            out[i] = 0
            continue
        nxt = path[1]
        delta = (nxt[0] - pos[0], nxt[1] - pos[1])
        for a, act in enumerate(ACTIONS):
            if act == delta:
                if valid[i, a] > 0:
                    out[i] = a
                else:
                    out[i] = 0
                break
        else:
            out[i] = 0
    return out


def run(baseline: str, size: int, density: float, corridor_length: int,
        agents: int, seed: int, steps: int) -> dict:
    grid, task, corridors, cell_to_id = build_scenario(size, density, corridor_length, agents, seed)
    rng = np.random.default_rng(seed + 100)
    total_arr = 0
    for _ in range(steps):
        valid = compute_valid_actions(grid, corridors, cell_to_id)
        if baseline == "random":
            actions = _random_step(rng, valid)
        elif baseline == "greedy_astar":
            actions = _greedy_astar_step(grid, task, valid)
        else:
            raise ValueError(baseline)
        _, arrived, _ = task.step(actions)
        total_arr += int(arrived.sum())
    return {"seed": seed, "size": size, "agents": grid.n_agents, "arrivals": total_arr,
            "steps": steps, "throughput": total_arr / max(1, steps)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=str, choices=["random", "greedy_astar"], required=True)
    p.add_argument("--size", type=int, default=15)
    p.add_argument("--density", type=float, default=0.3)
    p.add_argument("--corridor-length", type=int, default=5)
    p.add_argument("--agents", type=int, default=6)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123, 555, 2024])
    p.add_argument("--steps", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = []
    for s in args.seeds:
        r = run(args.baseline, args.size, args.density, args.corridor_length,
                args.agents, s, args.steps)
        results.append(r)
        print(f"[{args.baseline}] seed={s} arr={r['arrivals']} throughput={r['throughput']:.4f}")
    total_arr = sum(r["arrivals"] for r in results)
    total_steps = sum(r["steps"] for r in results)
    print(f"\nAggregate [{args.baseline}]: throughput={total_arr/max(1,total_steps):.4f}")
