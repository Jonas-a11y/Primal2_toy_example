"""Single-source shortest-path utilities.

For the path-length observation channel (Section IV.A) and for the future-position
maps, we need single-agent shortest paths on the *static* obstacle map, ignoring
other agents. Since we always need the whole distance field to a goal (for the
path-length channel), Dijkstra on a 4-connected uniform grid is the natural fit —
which reduces to BFS since all edges have unit cost.
"""
from __future__ import annotations

from collections import deque
import numpy as np


def bfs_distance_field(obstacle_map: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    """Return a (H, W) int32 array where each empty cell has the BFS distance
    (in 4-connected moves) from that cell to `goal`. Unreachable cells and
    obstacles are set to a large sentinel (-1)."""
    h, w = obstacle_map.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    gr, gc = goal
    if not (0 <= gr < h and 0 <= gc < w) or obstacle_map[gr, gc]:
        return dist
    dist[gr, gc] = 0
    q = deque([(gr, gc)])
    while q:
        r, c = q.popleft()
        d = dist[r, c]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and obstacle_map[nr, nc] == 0 and dist[nr, nc] == -1:
                dist[nr, nc] = d + 1
                q.append((nr, nc))
    return dist


def astar_path(
    obstacle_map: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    max_steps: int = 10_000,
) -> list[tuple[int, int]] | None:
    """Single-source shortest path (BFS suffices on uniform grid) from start to goal.

    Returns the list of cells [start, ..., goal] or None if unreachable.
    """
    if start == goal:
        return [start]
    h, w = obstacle_map.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w) or obstacle_map[start[0], start[1]]:
        return None
    if not (0 <= goal[0] < h and 0 <= goal[1] < w) or obstacle_map[goal[0], goal[1]]:
        return None
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    visited = np.zeros((h, w), dtype=bool)
    visited[start] = True
    q = deque([start])
    steps = 0
    while q and steps < max_steps:
        r, c = q.popleft()
        steps += 1
        if (r, c) == goal:
            path = [(r, c)]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return list(reversed(path))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and obstacle_map[nr, nc] == 0 and not visited[nr, nc]:
                visited[nr, nc] = True
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
    return None
