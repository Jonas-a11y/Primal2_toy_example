"""Comprehensive evaluation: run PRIMAL2, random, greedy-A* on the same seeded
scenarios and produce a comparison table (Markdown + JSON).
"""
from __future__ import annotations

import argparse
import json
import time
import numpy as np
import torch

from primal2_toy.env.grid import ACTIONS
from primal2_toy.policy.network import PolicyNet
from primal2_toy.obs.builder import ObsSpec
from primal2_toy.eval.rollout import Rollout, build_scenario
from primal2_toy.obs.astar import astar_path
from primal2_toy.train.validity import compute_valid_actions


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def eval_learned(ckpt: str, device_spec: str, seeds: list[int], size: int, density: float,
                 corridor_length: int, agents: int, steps: int, greedy: bool = True) -> list[dict]:
    device = _resolve_device(device_spec)
    obs_spec = ObsSpec(fov=11, n_pred=3)
    net = PolicyNet(
        n_spatial_channels=obs_spec.num_spatial_channels,
        n_scalars=obs_spec.num_scalar_features, n_actions=5, fov=11,
    ).to(device)
    s = torch.load(ckpt, map_location=device, weights_only=False)
    net.load_state_dict(s["model"])
    net.eval()

    results = []
    for seed in seeds:
        grid, task, corridors, cell_to_id = build_scenario(size, density, corridor_length, agents, seed)
        rollout = Rollout(net, grid, task, corridors, cell_to_id, device, obs_spec, greedy=greedy)
        collisions = 0
        for _ in range(steps):
            _ = rollout.step()
        results.append({
            "kind": "PRIMAL2 (learned)" if greedy else "PRIMAL2 (sampled)", "seed": seed,
            "agents": grid.n_agents, "steps": rollout.step_idx,
            "arrivals": rollout.total_arrivals,
            "throughput": rollout.total_arrivals / max(1, rollout.step_idx),
        })
    return results


def _random_step(rng, valid) -> np.ndarray:
    n, _ = valid.shape
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        choices = np.where(valid[i] > 0)[0]
        out[i] = 0 if not len(choices) else int(rng.choice(choices))
    return out


def _greedy_astar_step(grid, task, valid) -> np.ndarray:
    n = grid.n_agents
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        pos = (int(grid.positions[i, 0]), int(grid.positions[i, 1]))
        goal = (int(task.goals[i, 0]), int(task.goals[i, 1]))
        path = astar_path(grid.obstacle_map, pos, goal)
        if not path or len(path) < 2:
            out[i] = 0; continue
        nxt = path[1]
        delta = (nxt[0] - pos[0], nxt[1] - pos[1])
        for a, act in enumerate(ACTIONS):
            if act == delta:
                out[i] = a if valid[i, a] > 0 else 0
                break
    return out


def eval_baseline(kind: str, seeds: list[int], size: int, density: float,
                  corridor_length: int, agents: int, steps: int) -> list[dict]:
    out = []
    for seed in seeds:
        grid, task, corridors, cell_to_id = build_scenario(size, density, corridor_length, agents, seed)
        rng = np.random.default_rng(seed + 100)
        total_arr = 0
        for _ in range(steps):
            valid = compute_valid_actions(grid, corridors, cell_to_id)
            if kind == "random":
                actions = _random_step(rng, valid)
            else:
                actions = _greedy_astar_step(grid, task, valid)
            _, arrived, _ = task.step(actions)
            total_arr += int(arrived.sum())
        out.append({
            "kind": kind, "seed": seed, "agents": grid.n_agents, "steps": steps,
            "arrivals": total_arr, "throughput": total_arr / max(1, steps),
        })
    return out


def summarize(results: list[dict]) -> dict[str, dict]:
    from collections import defaultdict
    by_kind = defaultdict(list)
    for r in results:
        by_kind[r["kind"]].append(r)
    summary = {}
    for k, rs in by_kind.items():
        summary[k] = {
            "n_scenarios": len(rs),
            "throughput_mean": float(np.mean([r["throughput"] for r in rs])),
            "throughput_median": float(np.median([r["throughput"] for r in rs])),
            "throughput_min": float(np.min([r["throughput"] for r in rs])),
            "throughput_max": float(np.max([r["throughput"] for r in rs])),
            "arrivals_total": int(sum(r["arrivals"] for r in rs)),
            "steps_total": int(sum(r["steps"] for r in rs)),
        }
    return summary


def as_markdown(summary: dict[str, dict], meta: dict) -> str:
    lines = ["# PRIMAL2 Toy — Evaluation Report", ""]
    lines.append(f"**Scenario:** size={meta['size']}, density={meta['density']}, "
                 f"corridor_length={meta['corridor_length']}, agents={meta['agents']}, "
                 f"steps/scenario={meta['steps']}, seeds={meta['seeds']}")
    lines.append(f"**Checkpoint:** `{meta['checkpoint']}` (episode {meta['ckpt_episode']})")
    lines.append("")
    lines.append("| Method | Throughput (mean) | Throughput (min–max) | Total arrivals | Total steps |")
    lines.append("|---|---|---|---|---|")
    order = ["PRIMAL2 (learned)", "PRIMAL2 (sampled)", "greedy_astar", "random"]
    for k in order:
        if k not in summary: continue
        s = summary[k]
        lines.append(
            f"| {k} | {s['throughput_mean']:.4f} | "
            f"{s['throughput_min']:.4f} – {s['throughput_max']:.4f} | "
            f"{s['arrivals_total']} | {s['steps_total']} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--size", type=int, default=15)
    p.add_argument("--density", type=float, default=0.3)
    p.add_argument("--corridor-length", type=int, default=5)
    p.add_argument("--agents", type=int, default=6)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[7, 42, 123, 555, 2024, 8, 91, 314, 777, 1000])
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--out-md", type=str, default="logs/report.md")
    p.add_argument("--out-json", type=str, default="logs/report.json")
    args = p.parse_args()

    results = []
    print("running learned (greedy)...")
    results.extend(eval_learned(args.checkpoint, args.device, args.seeds, args.size, args.density,
                                args.corridor_length, args.agents, args.steps, greedy=True))
    print("running learned (sampled)...")
    results.extend(eval_learned(args.checkpoint, args.device, args.seeds, args.size, args.density,
                                args.corridor_length, args.agents, args.steps, greedy=False))
    print("running greedy_astar...")
    results.extend(eval_baseline("greedy_astar", args.seeds, args.size, args.density,
                                 args.corridor_length, args.agents, args.steps))
    print("running random...")
    results.extend(eval_baseline("random", args.seeds, args.size, args.density,
                                 args.corridor_length, args.agents, args.steps))
    summary = summarize(results)
    ckpt_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = {
        "size": args.size, "density": args.density,
        "corridor_length": args.corridor_length, "agents": args.agents,
        "steps": args.steps, "seeds": args.seeds,
        "checkpoint": args.checkpoint,
        "ckpt_episode": int(ckpt_state.get("episode", 0)),
    }
    md = as_markdown(summary, meta)
    with open(args.out_md, "w") as f:
        f.write(md + "\n")
    with open(args.out_json, "w") as f:
        json.dump({"meta": meta, "summary": summary, "results": results}, f, indent=2)
    print(f"\n{md}\n")
    print(f"wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
