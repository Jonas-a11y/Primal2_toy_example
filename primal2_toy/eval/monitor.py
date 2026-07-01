"""Live training monitor — tail the CSV log and print smoothed metrics.

Usage:
    python -m primal2_toy.eval.monitor logs/train_metrics_XXX.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from collections import deque


def _tail_csv(path: str, window: int):
    seen = 0
    last_size = 0
    while True:
        try:
            size = os.path.getsize(path)
        except FileNotFoundError:
            time.sleep(1); continue
        if size > last_size:
            with open(path) as f:
                rows = list(csv.DictReader(f))
            last_size = size
            new = rows[seen:]
            seen = len(rows)
            for r in new:
                yield r
        time.sleep(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=str)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--every", type=int, default=25)
    args = p.parse_args()

    rl_val = deque(maxlen=args.window)
    rl_actor = deque(maxlen=args.window)
    rl_ent = deque(maxlen=args.window)
    rl_goals = deque(maxlen=args.window)
    il_bc = deque(maxlen=args.window)
    il_valid = deque(maxlen=args.window)
    n = 0
    for r in _tail_csv(args.csv, args.window):
        n += 1
        try:
            if r["kind"] == "RL":
                rl_val.append(float(r["value_loss"]))
                rl_actor.append(float(r["actor_loss"]))
                rl_ent.append(float(r["entropy"]))
                rl_goals.append(int(float(r["goals_reached"])))
            elif r["kind"] == "IL":
                il_bc.append(float(r["bc_loss"]))
                il_valid.append(float(r["valid_loss"]))
        except (KeyError, ValueError):
            continue
        if n % args.every == 0:
            def _avg(d): return (sum(d) / len(d)) if d else float("nan")
            print(
                f"ep {r['episode']:>5} | "
                f"RL val={_avg(rl_val):+.3f} act={_avg(rl_actor):+.3f} ent={_avg(rl_ent):.3f} goals/ep={_avg(rl_goals):.2f} | "
                f"IL bc={_avg(il_bc):.3f} valid={_avg(il_valid):.3f}"
            )


if __name__ == "__main__":
    main()
