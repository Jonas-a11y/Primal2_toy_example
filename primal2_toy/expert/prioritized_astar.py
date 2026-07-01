"""Prioritized-planning multi-agent A* over a space-time grid.

Standard prioritized planning:
  1. Fix a priority order over agents (deterministic, seedable).
  2. For each agent in order, run single-agent A* in the space-time graph,
     treating already-planned agents' cell occupancies as time-varying obstacles.
  3. Add the resulting trajectory to the reservation table.

Actions returned per timestep per agent use the same action encoding as
`env.grid.ACTIONS`: 0=stay, 1=up, 2=down, 3=left, 4=right.

If an agent cannot find a path (e.g., blocked by higher-priority reservations),
we return "stay forever" for that agent — training code treats such episodes
as partial demonstrations and either uses them as-is or discards them.
"""
from __future__ import annotations

import heapq
import numpy as np

from ..env.grid import ACTIONS


def _action_of(delta: tuple[int, int]) -> int:
    for i, a in enumerate(ACTIONS):
        if a == delta:
            return i
    raise ValueError(f"Unknown delta {delta}")


def _neighbors(cell: tuple[int, int], obstacle_map: np.ndarray) -> list[tuple[tuple[int, int], int]]:
    h, w = obstacle_map.shape
    r, c = cell
    out = []
    for aidx, (dr, dc) in enumerate(ACTIONS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and obstacle_map[nr, nc] == 0:
            out.append(((nr, nc), aidx))
    return out


def _spacetime_astar(
    obstacle_map: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    reservations_cell: dict[int, set[tuple[int, int]]],
    reservations_edge: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]],
    horizon: int,
    max_expansions: int = 50_000,
) -> list[int] | None:
    """Return list of action indices of length `horizon`, or None if no path.

    `reservations_cell[t]` = set of cells occupied by higher-priority agents at time t.
    `reservations_edge[t]` = set of (from, to) edges used by higher-priority
        agents between t-1 and t (blocks swaps).
    """
    # Heuristic: Manhattan distance.
    def h_est(c: tuple[int, int]) -> int:
        return abs(c[0] - goal[0]) + abs(c[1] - goal[1])

    # State = (cell, t). Cost = t.
    open_heap: list[tuple[int, int, tuple[int, int], int]] = []  # (f, g, cell, t)
    heapq.heappush(open_heap, (h_est(start), 0, start, 0))
    came_from: dict[tuple[tuple[int, int], int], tuple[tuple[tuple[int, int], int], int]] = {}
    g_score: dict[tuple[tuple[int, int], int], int] = {(start, 0): 0}
    expansions = 0
    goal_state: tuple[tuple[int, int], int] | None = None

    while open_heap and expansions < max_expansions:
        _, g, cell, t = heapq.heappop(open_heap)
        expansions += 1
        if cell == goal and t >= 1:
            goal_state = (cell, t)
            break
        if t >= horizon:
            continue
        for nxt, aidx in _neighbors(cell, obstacle_map):
            nt = t + 1
            # Vertex conflict.
            if nt in reservations_cell and nxt in reservations_cell[nt]:
                continue
            # Edge (swap) conflict.
            if nt in reservations_edge and (nxt, cell) in reservations_edge[nt]:
                continue
            state = (nxt, nt)
            new_g = g + 1
            if new_g < g_score.get(state, 10**9):
                g_score[state] = new_g
                came_from[state] = ((cell, t), aidx)
                heapq.heappush(open_heap, (new_g + h_est(nxt), new_g, nxt, nt))
    if goal_state is None:
        # No path within horizon that reaches goal — try best-effort: pick the
        # state with lowest heuristic that's reachable, then extend with stays.
        # We instead just return None here and let the caller reserve "stay".
        return None
    # Reconstruct actions.
    actions_rev = []
    state = goal_state
    while state != (start, 0):
        prev, aidx = came_from[state]
        actions_rev.append(aidx)
        state = prev
    actions = list(reversed(actions_rev))
    # Extend with stays until horizon.
    while len(actions) < horizon:
        actions.append(0)
    return actions


def plan(
    obstacle_map: np.ndarray,
    starts: list[tuple[int, int]],
    goals: list[tuple[int, int]],
    horizon: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Plan actions for each agent using prioritized planning.

    Returns a list of length n_agents; each entry is a length-`horizon` list of
    action indices. Agents that couldn't plan get an all-zero (stay) sequence.
    """
    n = len(starts)
    order = list(range(n))
    rng.shuffle(order)

    reservations_cell: dict[int, set[tuple[int, int]]] = {}
    reservations_edge: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]] = {}

    def _reserve(traj: list[tuple[int, int]]) -> None:
        for t, cell in enumerate(traj):
            reservations_cell.setdefault(t, set()).add(cell)
        for t in range(1, len(traj)):
            reservations_edge.setdefault(t, set()).add((traj[t - 1], traj[t]))

    plans: list[list[int]] = [[0] * horizon for _ in range(n)]
    for i in order:
        actions = _spacetime_astar(
            obstacle_map, starts[i], goals[i],
            reservations_cell, reservations_edge, horizon,
        )
        if actions is None:
            # Reserve "stay at start forever"; skip this agent's plan.
            traj = [starts[i]] * (horizon + 1)
        else:
            # Reconstruct trajectory from start + actions.
            traj = [starts[i]]
            for a in actions:
                dr, dc = ACTIONS[a]
                prev = traj[-1]
                traj.append((prev[0] + dr, prev[1] + dc))
            plans[i] = actions
        _reserve(traj)
    return plans
