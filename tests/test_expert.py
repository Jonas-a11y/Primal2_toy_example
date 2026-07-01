"""Sanity test for the prioritized-planning expert."""
import numpy as np
from primal2_toy.env.maze import generate_maze
from primal2_toy.expert.prioritized_astar import plan


def main():
    rng = np.random.default_rng(2)
    m = generate_maze(12, 0.25, 4, rng)
    empties = np.argwhere(m == 0)
    idx = rng.choice(len(empties), size=8, replace=False)
    positions = [tuple(empties[i]) for i in idx]
    # Goals: another random subset.
    idx2 = rng.choice(len(empties), size=8, replace=False)
    goals = [tuple(empties[i]) for i in idx2]
    plans = plan(m, positions, goals, horizon=64, rng=rng)
    assert len(plans) == 8
    for i, seq in enumerate(plans):
        assert len(seq) == 64
    print("expert_smoke OK")


if __name__ == "__main__":
    main()
