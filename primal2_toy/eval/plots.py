"""Plot training curves from the CSV logs."""
from __future__ import annotations

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _smooth(x: list[float], k: int = 20) -> list[float]:
    if not x:
        return x
    out = []
    for i in range(len(x)):
        lo = max(0, i - k + 1)
        out.append(float(np.mean(x[lo:i + 1])))
    return out


def plot(csv_path: str, out_prefix: str) -> None:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("empty csv"); return
    rl = [r for r in rows if r["kind"] == "RL"]
    il = [r for r in rows if r["kind"] == "IL"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # RL value_loss + entropy
    if rl:
        eps = [int(r["episode"]) for r in rl]
        vl = [float(r["value_loss"]) for r in rl]
        ent = [float(r["entropy"]) for r in rl]
        axes[0, 0].plot(eps, vl, alpha=0.25, color="tab:blue", label="value loss")
        axes[0, 0].plot(eps, _smooth(vl), color="tab:blue", label="value loss (smooth)")
        axes[0, 0].set_yscale("symlog")
        axes[0, 0].set_title("RL: value loss")
        axes[0, 0].set_xlabel("episode"); axes[0, 0].legend()
        axes[0, 1].plot(eps, ent, alpha=0.25, color="tab:orange")
        axes[0, 1].plot(eps, _smooth(ent), color="tab:orange", label="entropy (smooth)")
        axes[0, 1].set_title("RL: policy entropy")
        axes[0, 1].set_xlabel("episode"); axes[0, 1].legend()
    # IL bc_loss
    if il:
        eps = [int(r["episode"]) for r in il]
        bc = [float(r["bc_loss"]) for r in il]
        axes[1, 0].plot(eps, bc, alpha=0.25, color="tab:green")
        axes[1, 0].plot(eps, _smooth(bc), color="tab:green", label="BC loss (smooth)")
        axes[1, 0].set_title("IL: behavior-cloning loss")
        axes[1, 0].set_xlabel("episode"); axes[1, 0].legend()
    # Goals/ep across both
    if rl:
        eps = [int(r["episode"]) for r in rl]
        goals = [int(float(r["goals_reached"])) for r in rl]
        axes[1, 1].plot(eps, goals, alpha=0.25, color="tab:red")
        axes[1, 1].plot(eps, _smooth(goals), color="tab:red", label="RL goals/ep (smooth)")
        axes[1, 1].set_title("RL: goals reached per episode")
        axes[1, 1].set_xlabel("episode"); axes[1, 1].legend()

    fig.tight_layout()
    p = f"{out_prefix}_training.png"
    fig.savefig(p, dpi=100)
    print(f"wrote {p}")


def plot_eval_curve(eval_csv: str, out_path: str) -> None:
    with open(eval_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows: return
    eps = [int(r["episode"]) for r in rows]
    tp = [float(r["throughput"]) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eps, tp, marker="o")
    ax.set_title("Held-out eval throughput over training")
    ax.set_xlabel("training episode"); ax.set_ylabel("throughput (arrivals/step)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-csv", type=str, required=True)
    p.add_argument("--eval-csv", type=str, default="logs/eval_over_time.csv")
    p.add_argument("--out-prefix", type=str, default="logs/plot")
    args = p.parse_args()
    plot(args.metrics_csv, args.out_prefix)
    if os.path.exists(args.eval_csv):
        plot_eval_curve(args.eval_csv, f"{args.out_prefix}_eval.png")


if __name__ == "__main__":
    main()
