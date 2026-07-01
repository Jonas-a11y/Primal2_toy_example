"""Corridor-specific observation channels: ΔX, ΔY, blocking.

Layout choice (per design): full 11×11 sparse channels; values are written only
into cells corresponding to corridor endpoints. Zero everywhere else.

Blocking map (Section IV.A): for each corridor endpoint, marks 1 iff at least
one agent is currently inside that corridor moving *toward* the endpoint.
"""
from __future__ import annotations

import numpy as np

from ..env.corridor import Corridor


def _agent_dir_toward(cell: tuple[int, int], endpoint: tuple[int, int]) -> tuple[int, int]:
    """Direction (dr, dc) from cell to endpoint at unit magnitude along dominant axis.

    We use a simple sign check on the corridor cells' progression order rather
    than 2D bearing, since corridors are 1D locally.
    """
    return (int(np.sign(endpoint[0] - cell[0])), int(np.sign(endpoint[1] - cell[1])))


def build_blocking(
    corridors: list[Corridor],
    cell_to_corridor: np.ndarray,
    agent_positions: np.ndarray,
    agent_prev_positions: np.ndarray,
) -> dict[tuple[int, int], int]:
    """Return blocking value {endpoint -> 0/1} for each corridor endpoint.

    An endpoint E is 'blocked' if any agent currently inside the corridor is
    moving toward E (i.e., its last move increased its proximity to E along
    the corridor's axis).
    """
    blocking: dict[tuple[int, int], int] = {}
    for c in corridors:
        for e in c.endpoints:
            blocking[e] = 0
    for i in range(agent_positions.shape[0]):
        r, c = int(agent_positions[i, 0]), int(agent_positions[i, 1])
        cid = int(cell_to_corridor[r, c])
        if cid < 0:
            continue
        corr = corridors[cid]
        pr, pc = int(agent_prev_positions[i, 0]), int(agent_prev_positions[i, 1])
        # Which endpoint is this agent moving toward?
        for endpoint in corr.endpoints:
            # Manhattan distance to endpoint from cur vs. prev.
            d_now = abs(endpoint[0] - r) + abs(endpoint[1] - c)
            d_prev = abs(endpoint[0] - pr) + abs(endpoint[1] - pc)
            if d_now < d_prev:
                blocking[endpoint] = 1
                break
        else:
            # No movement toward an endpoint (e.g., first tick, or standing still).
            # Do nothing; endpoint stays at whatever previous agents set.
            pass
    return blocking


def paint_corridor_channels(
    obstacle_map: np.ndarray,
    corridors: list[Corridor],
    blocking: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dx_map, dy_map, blocking_map) full-size grids, non-zero only at endpoints."""
    h, w = obstacle_map.shape
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)
    bl = np.zeros((h, w), dtype=np.float32)
    for corr in corridors:
        for e in corr.endpoints:
            d = corr.delta_at(e)
            dx[e[0], e[1]] = d[0]
            dy[e[0], e[1]] = d[1]
            bl[e[0], e[1]] = float(blocking.get(e, 0))
    return dx, dy, bl
