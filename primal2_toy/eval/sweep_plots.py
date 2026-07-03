"""Aggregate a sweep CSV into paper-figure-shaped plots.

- Fig. 4 shape (one-shot): success rate + average path length vs. team size,
  one line per (size, density, corridor) config.
- Fig. 5 shape (LMAPF): throughput vs. team size, one line per config.

Assumes the CSV was produced by primal2_toy.eval.sweep.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: str) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("size", "corridor", "team_size", "actual_agents",
                  "seed", "timesteps_budget", "steps", "n_arrived",
                  "makespan", "total_arrivals"):
            if r.get(k) not in (None, "", "nan"):
                r[k] = int(float(r[k]))
        for k in ("density", "success_95", "avg_path_len",
                  "throughput", "collision_rate_per_agent_step", "wall_s"):
            if r.get(k) not in (None, "", "nan"):
                try:
                    r[k] = float(r[k])
                except ValueError:
                    pass
        r["success_100"] = str(r.get("success_100", "")).strip().lower() == "true"
    return rows


def _group_by(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def plot_lmapf(rows: list[dict], out_path: str) -> None:
    """Throughput vs team_size, one line per (size, density, corridor)."""
    groups = _group_by(rows, ("size", "density", "corridor"))
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, gr in sorted(groups.items()):
        size, density, corridor = key
        by_team = _group_by(gr, ("team_size",))
        team_sizes = sorted(by_team.keys())
        means = [float(np.mean([r["throughput"] for r in by_team[t]])) for t in team_sizes]
        stds = [float(np.std([r["throughput"] for r in by_team[t]])) for t in team_sizes]
        label = f"{size}×{size}, d={density:.2f}, c={corridor}"
        team_x = [t[0] for t in team_sizes]
        ax.errorbar(team_x, means, yerr=stds, marker="o", capsize=3, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("team size")
    ax.set_ylabel("throughput (arrivals / step)")
    ax.set_title("LMAPF throughput vs. team size (paper Fig. 5 shape)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"wrote {out_path}")


def plot_oneshot(rows: list[dict], out_prefix: str) -> None:
    """Success rate + average path length vs team_size (Fig. 4 shape).

    Emits two files: <prefix>_success.png and <prefix>_pathlen.png.
    """
    groups = _group_by(rows, ("size", "density", "corridor"))

    # Success rate.
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, gr in sorted(groups.items()):
        size, density, corridor = key
        by_team = _group_by(gr, ("team_size",))
        team_sizes = sorted(by_team.keys())
        s100 = [float(np.mean([1.0 if r["success_100"] else 0.0 for r in by_team[t]])) for t in team_sizes]
        s95 = [float(np.mean([r["success_95"] for r in by_team[t] if not np.isnan(r["success_95"])])) for t in team_sizes]
        team_x = [t[0] for t in team_sizes]
        label = f"{size}×{size}, d={density:.2f}, c={corridor}"
        line, = ax.plot(team_x, s100, marker="o", label=f"{label} (100%)")
        ax.plot(team_x, s95, marker="s", linestyle="--", color=line.get_color(),
                label=f"{label} (95%)")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("team size")
    ax.set_ylabel("success rate")
    ax.set_title("One-shot MAPF success rate (paper Fig. 4 shape)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=1)
    fig.tight_layout()
    p_succ = f"{out_prefix}_success.png"
    fig.savefig(p_succ, dpi=110)
    print(f"wrote {p_succ}")

    # Avg path length.
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, gr in sorted(groups.items()):
        size, density, corridor = key
        by_team = _group_by(gr, ("team_size",))
        team_sizes = sorted(by_team.keys())
        means = [float(np.mean([r["avg_path_len"] for r in by_team[t]])) for t in team_sizes]
        team_x = [t[0] for t in team_sizes]
        ax.plot(team_x, means, marker="o",
                label=f"{size}×{size}, d={density:.2f}, c={corridor}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("team size")
    ax.set_ylabel("average path length (moves)")
    ax.set_title("One-shot MAPF average path length (paper Fig. 4 shape)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p_len = f"{out_prefix}_pathlen.png"
    fig.savefig(p_len, dpi=110)
    print(f"wrote {p_len}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-csv", type=str, required=True)
    p.add_argument("--out", type=str, default="docs/images/sweep",
                   help="Output prefix for the plot files.")
    args = p.parse_args()

    rows = _read_rows(args.sweep_csv)
    if not rows:
        print("no rows in csv; aborting")
        return
    mode = rows[0]["mode"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if mode == "lmapf":
        plot_lmapf(rows, f"{args.out}_lmapf.png")
    elif mode == "oneshot":
        plot_oneshot(rows, args.out)
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
