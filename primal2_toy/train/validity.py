"""Compute per-step per-agent action validity flags.

Following Section IV.B + V.A.1 of PRIMAL2:
  - Move into obstacle → invalid.
  - Move into a cell currently occupied by another agent → invalid.
  - Immediate reversal (move back to prev cell) → invalid, unless staying.
  - Corridor conventions:
     * Don't enter a corridor at an endpoint whose 'blocking' flag is 1
       (another agent is inside heading toward you).
     * Don't reverse direction while inside a corridor.
"""
from __future__ import annotations

import numpy as np

from ..env.grid import ACTIONS, GridWorld
from ..env.corridor import Corridor
from ..obs.corridor_maps import build_blocking


def compute_valid_actions(
    grid: GridWorld,
    corridors: list[Corridor],
    cell_to_corridor: np.ndarray,
) -> np.ndarray:
    """Return (n_agents, 5) float array of 1/0 flags.

    Stay is always valid.
    """
    n = grid.n_agents
    valid = np.ones((n, 5), dtype=np.float32)

    blocking = build_blocking(corridors, cell_to_corridor, grid.positions, grid.prev_positions)

    # Precompute occupied cells set.
    occ = {(int(grid.positions[j, 0]), int(grid.positions[j, 1])): j for j in range(n)}

    for i in range(n):
        r, c = int(grid.positions[i, 0]), int(grid.positions[i, 1])
        pr, pc = int(grid.prev_positions[i, 0]), int(grid.prev_positions[i, 1])
        in_corridor_i = cell_to_corridor[r, c] >= 0
        for a, (dr, dc) in enumerate(ACTIONS):
            if a == 0:
                continue  # stay always valid
            nr, nc = r + dr, c + dc
            # Bounds/obstacle.
            if not grid.in_bounds(nr, nc) or grid.obstacle_map[nr, nc]:
                valid[i, a] = 0.0
                continue
            # Move into currently-occupied cell.
            if (nr, nc) in occ and occ[(nr, nc)] != i:
                valid[i, a] = 0.0
                continue
            # Immediate reversal.
            if (nr, nc) == (pr, pc) and (pr, pc) != (r, c):
                valid[i, a] = 0.0
                continue
            # Corridor convention: entering blocked corridor.
            neighbor_cid = int(cell_to_corridor[nr, nc])
            if neighbor_cid >= 0 and not in_corridor_i:
                # Entering a corridor. Which endpoint would we enter from?
                # We are moving from (r, c) into corridor cell (nr, nc); that
                # corridor's endpoint we approach is (nr, nc) if (nr, nc) is an
                # endpoint of that corridor, else the endpoint reachable in the
                # direction we're moving.
                corr = corridors[neighbor_cid]
                # Simple check: if the neighbor cell is an endpoint and it is
                # blocked, forbid the move.
                if (nr, nc) in corr.endpoints and blocking.get((nr, nc), 0) == 1:
                    valid[i, a] = 0.0
                    continue
            # Corridor convention: reversing inside a corridor.
            if in_corridor_i:
                # If we're inside a corridor and this action moves us to prev cell,
                # already forbidden above. Additionally: if this action moves us
                # into a corridor cell that is our previous direction reversed —
                # covered by immediate-reversal check.
                pass
    return valid
