"""Maze generation for corridor-heavy worlds.

Produces obstacle maps parameterized by (size, density, typical_corridor_length),
in line with Section III.A of PRIMAL2. We construct maps by:

1. Start with a fully-empty grid.
2. Iteratively drop rectangular "wall segments" of length drawn from a Poisson-ish
   distribution centered on `typical_corridor_length`, oriented horizontally or
   vertically at random, at random positions, until the target obstacle density
   is reached.
3. Reject/regenerate if the empty cells are not a single connected component
   (paper requires that every agent has at least one path to its goal).

The corridor length parameter controls the average length of contiguous wall
segments; longer segments produce longer, narrower corridors in the free space.
"""
from __future__ import annotations

import numpy as np


def _connected(obstacle_map: np.ndarray) -> bool:
    h, w = obstacle_map.shape
    # Find first empty cell.
    idx = np.argwhere(obstacle_map == 0)
    if len(idx) == 0:
        return False
    start = tuple(idx[0])
    stack = [start]
    visited = np.zeros_like(obstacle_map, dtype=bool)
    visited[start] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not obstacle_map[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                stack.append((nr, nc))
    return int(visited.sum()) == int((obstacle_map == 0).sum())


def generate_maze(
    size: int,
    density: float,
    typical_corridor_length: int,
    rng: np.random.Generator,
    max_attempts: int = 200,
) -> np.ndarray:
    """Generate a connected obstacle map.

    Parameters
    ----------
    size : int — H = W = size.
    density : float in [0, 1] — target fraction of obstacle cells.
    typical_corridor_length : int — average length of drawn wall segments.
    rng : numpy random Generator.
    max_attempts : int — after this many failed connectivity attempts,
        drop density by 5% and retry.

    Returns
    -------
    (size, size) uint8 array.
    """
    target_obstacles = int(density * size * size)
    for attempt in range(max_attempts):
        m = np.zeros((size, size), dtype=np.uint8)
        # Border walls make the maze feel like a bounded world; optional.
        # We leave the border empty to avoid wasting cells on very small maps.
        placed = 0
        tries = 0
        max_tries = target_obstacles * 20
        while placed < target_obstacles and tries < max_tries:
            tries += 1
            length = max(1, int(rng.poisson(typical_corridor_length)))
            horizontal = rng.random() < 0.5
            r = int(rng.integers(0, size))
            c = int(rng.integers(0, size))
            for k in range(length):
                rr, cc = (r, c + k) if horizontal else (r + k, c)
                if 0 <= rr < size and 0 <= cc < size and m[rr, cc] == 0:
                    m[rr, cc] = 1
                    placed += 1
                    if placed >= target_obstacles:
                        break
        if _connected(m):
            return m
        # Retry with fresh randomness
    # Fallback: nudge density down and try one more time.
    return generate_maze(size, max(0.05, density * 0.9), typical_corridor_length, rng, max_attempts=1)
