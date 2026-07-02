"""Bar plot comparing learned policy vs. baselines."""
from __future__ import annotations

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=str, required=True, help="JSON output from compare.py")
    p.add_argument("--out", type=str, default="logs/plot_comparison.png")
    args = p.parse_args()

    with open(args.report) as f:
        data = json.load(f)
    summary = data["summary"]
    meta = data["meta"]

    order = ["random", "greedy_astar", "PRIMAL2 (learned)", "PRIMAL2 (sampled)"]
    order = [k for k in order if k in summary]
    means = [summary[k]["throughput_mean"] for k in order]
    mins = [summary[k]["throughput_min"] for k in order]
    maxs = [summary[k]["throughput_max"] for k in order]
    lower_err = [m - lo for m, lo in zip(means, mins)]
    upper_err = [hi - m for m, hi in zip(means, maxs)]

    color_map = {
        "random": "#c9c9c9",
        "greedy_astar": "#f5c518",
        "PRIMAL2 (learned)": "#2e7dd7",
        "PRIMAL2 (sampled)": "#8e44ad",
    }
    colors = [color_map[k] for k in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(order))
    bars = ax.bar(x, means, yerr=[lower_err, upper_err], capsize=6, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("throughput (arrivals / step)")
    ax.set_title(
        f"Held-out LMAPF throughput\n"
        f"{meta['size']}×{meta['size']}, "
        f"density {meta['density']:.2f}, corridor {meta['corridor_length']}, "
        f"{meta['agents']} agents, {len(meta['seeds'])} seeds × {meta['steps']} steps"
    )
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{m:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(args.out, dpi=100)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
