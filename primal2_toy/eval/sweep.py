"""Sweep evaluator matching PRIMAL2 paper Fig. 4 and Fig. 5 shapes.

Runs a Cartesian product of (mode × size × density × corridor × team_size × seed)
and writes one row per instance to a CSV. Metrics computed per Section VI:
  - one-shot (mode=oneshot): makespan, success_100, success_95, avg_path_len
  - LMAPF   (mode=lmapf):    throughput

Paper's timestep budgets are used by default per world size (see PAPER_BUDGETS).

Example — reproduce a slice of Fig. 5 (LMAPF, 40x40, density 0.3, corridor 10):

    PYTHONPATH=. python -m primal2_toy.eval.sweep \
        --checkpoint checkpoints/primal2_final.pt \
        --mode lmapf --sizes 40 --densities 0.3 --corridors 10 \
        --team-sizes 4 8 16 32 64 --n-seeds 5 \
        --out logs/sweep_fig5_40x40.csv
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


# Paper timestep budgets by (mode, size).
PAPER_BUDGETS = {
    ("oneshot", 20): 320,
    ("oneshot", 40): 320,
    ("oneshot", 80): 480,
    ("oneshot", 160): 640,
    ("lmapf", 20): 128,
    ("lmapf", 40): 128,
    ("lmapf", 80): 192,
    ("lmapf", 160): 256,
}


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def _load_net(ckpt: str, device: torch.device, obs_spec: ObsSpec) -> PolicyNet:
    net = PolicyNet(
        n_spatial_channels=obs_spec.num_spatial_channels,
        n_scalars=obs_spec.num_scalar_features, n_actions=5, fov=obs_spec.fov,
    ).to(device)
    s = torch.load(ckpt, map_location=device, weights_only=False)
    net.load_state_dict(s["model"])
    net.eval()
    return net


def budget_for(mode: str, size: int, override: int | None) -> int:
    if override is not None:
        return override
    # Interpolate: for sizes not in PAPER_BUDGETS pick the closest.
    key = (mode, size)
    if key in PAPER_BUDGETS:
        return PAPER_BUDGETS[key]
    keys = [(m, sz) for (m, sz) in PAPER_BUDGETS if m == mode]
    if not keys:
        return 256
    closest = min(keys, key=lambda k: abs(k[1] - size))
    return PAPER_BUDGETS[closest]


def _success_95(rollout: Rollout, seed: int, n_iters: int = 10) -> tuple[float, dict]:
    """Paper's 95 % success metric (Section VI): at start of each iteration
    subsample 95 % of agents and check whether the *original run* would have
    succeeded on that subset. We don't rerun the whole rollout for each subset
    (way too expensive); we approximate by testing whether each 95 % subset was
    fully arrived at episode end.

    This is a strict lower bound on the paper's metric, which does re-plan on
    the subset. For a decentralized learned policy the behavior would be
    identical anyway (agents don't coordinate on team composition), so this
    approximation is tight for us.

    Returns (success_rate_over_iters, extra_info_dict).
    """
    rng = np.random.default_rng(seed + 1234)
    n = rollout.grid.n_agents
    keep = max(1, int(round(0.95 * n)))
    successes = 0
    for _ in range(n_iters):
        idxs = rng.choice(n, size=keep, replace=False)
        mask = np.zeros(n, dtype=bool); mask[idxs] = True
        m = rollout.paper_metrics(agent_subset=mask)
        if m["success_100"]:
            successes += 1
    return successes / n_iters, {"success_95_iters": n_iters, "success_95_keep": keep}


def eval_instance(
    net: PolicyNet, device: torch.device, obs_spec: ObsSpec,
    mode: str, size: int, density: float, corridor: int, team_size: int,
    seed: int, timesteps: int, greedy: bool,
) -> dict:
    grid, task, corridors, cell_to_id = build_scenario(
        size=size, density=density, corridor_length=corridor,
        n_agents=team_size, seed=seed,
        episode_length=timesteps, one_shot=(mode == "oneshot"),
    )
    rollout = Rollout(net, grid, task, corridors, cell_to_id, device, obs_spec, greedy=greedy)
    t0 = time.time()
    while not rollout.done and rollout.step_idx < timesteps:
        rollout.step()
    elapsed = time.time() - t0
    m = rollout.paper_metrics()
    if mode == "oneshot":
        s95, s95_info = _success_95(rollout, seed=seed)
        m["success_95"] = s95
        m.update(s95_info)
    else:
        m["success_95"] = float("nan")
    m["mode"] = mode
    m["size"] = size
    m["density"] = density
    m["corridor"] = corridor
    m["team_size"] = team_size
    m["actual_agents"] = grid.n_agents
    m["seed"] = seed
    m["timesteps_budget"] = timesteps
    m["wall_s"] = elapsed
    m["greedy"] = greedy
    return m


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--mode", type=str, choices=["oneshot", "lmapf"], required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[20, 40])
    p.add_argument("--densities", type=float, nargs="+", default=[0.3])
    p.add_argument("--corridors", type=int, nargs="+", default=[10])
    p.add_argument("--team-sizes", type=int, nargs="+",
                   default=[4, 8, 16, 32, 64, 128])
    p.add_argument("--n-seeds", type=int, default=5,
                   help="Number of seeds per config (paper uses 50; scale to your compute budget).")
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--timesteps", type=int, default=None,
                   help="Override episode length; otherwise paper's per-size defaults are used.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--greedy", action="store_true",
                   help="Use argmax actions; otherwise sample (default).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    obs_spec = ObsSpec(fov=11, n_pred=3)
    net = _load_net(args.checkpoint, device, obs_spec)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = [
        "mode", "size", "density", "corridor", "team_size", "actual_agents",
        "seed", "timesteps_budget", "steps", "greedy",
        "success_100", "success_95", "n_arrived", "makespan", "avg_path_len",
        "throughput", "total_arrivals", "collision_rate_per_agent_step", "wall_s",
    ]
    total_instances = (
        len(args.sizes) * len(args.densities) * len(args.corridors)
        * len(args.team_sizes) * args.n_seeds
    )
    print(f"Sweep over {total_instances} instances -> {args.out}")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        done = 0
        t_start = time.time()
        for size in args.sizes:
            for density in args.densities:
                for corridor in args.corridors:
                    for team in args.team_sizes:
                        budget = budget_for(args.mode, size, args.timesteps)
                        for i in range(args.n_seeds):
                            seed = args.seed_base + i
                            m = eval_instance(
                                net, device, obs_spec,
                                args.mode, size, density, corridor, team,
                                seed=seed, timesteps=budget, greedy=args.greedy,
                            )
                            row = {k: m.get(k, "") for k in fields}
                            w.writerow(row)
                            f.flush()
                            done += 1
                            if done % max(1, total_instances // 40) == 0 or done == total_instances:
                                elapsed = time.time() - t_start
                                eta = elapsed / done * (total_instances - done)
                                print(f"[{done}/{total_instances}] "
                                      f"size={size} d={density} c={corridor} n={team} seed={seed} "
                                      f"tp={m['throughput']:.3f} succ100={m['success_100']} "
                                      f"eta={eta:.0f}s")
    print(f"done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
