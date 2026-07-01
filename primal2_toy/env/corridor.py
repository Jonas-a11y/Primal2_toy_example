"""Corridor decomposition for a static obstacle map.

Following PRIMAL2 Section IV.A / Fig. 2:
  - Corridor cell = empty cell whose in-map, non-obstacle neighbors count is <= 2
    (i.e., movement is degenerate: stay + at most one direction).
  - A corridor is a maximal connected run of corridor cells.
  - Endpoints of a corridor are its two extreme cells (or one, if dead-end).
  - Decision points = the cells adjacent to endpoints that themselves are NOT
    corridor cells (i.e., have >=3 empty neighbors). These are where agents
    must decide whether to enter the corridor.
  - ΔX, ΔY for a non-dead-end corridor = signed differences between endpoint
    coordinates (recorded at each endpoint, with opposite sign at the other end).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Corridor:
    cells: list[tuple[int, int]] = field(default_factory=list)
    endpoints: list[tuple[int, int]] = field(default_factory=list)  # 1 (dead-end) or 2
    decision_points: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)
    # decision_points maps endpoint -> adjacent decision cell (outside corridor).

    def delta_at(self, endpoint: tuple[int, int]) -> tuple[int, int]:
        """Return (dx, dy) recorded at this endpoint.

        For a dead-end corridor, returns (0, 0).
        Otherwise, returns the displacement to the OTHER endpoint.
        Following the paper, dx = col-difference, dy = row-difference.
        """
        if len(self.endpoints) < 2:
            return (0, 0)
        other = self.endpoints[1] if endpoint == self.endpoints[0] else self.endpoints[0]
        return (other[1] - endpoint[1], other[0] - endpoint[0])


def _empty_neighbors(obstacle_map: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = obstacle_map.shape
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and obstacle_map[nr, nc] == 0:
            out.append((nr, nc))
    return out


def analyze(obstacle_map: np.ndarray) -> tuple[list[Corridor], np.ndarray]:
    """Return (corridors, cell_to_corridor_id).

    cell_to_corridor_id[r, c] = int id of the corridor containing (r, c), or -1
    if (r, c) is not in any corridor (i.e., not a corridor cell, or an obstacle).
    """
    h, w = obstacle_map.shape
    is_corridor_cell = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if obstacle_map[r, c]:
                continue
            if len(_empty_neighbors(obstacle_map, r, c)) <= 2:
                is_corridor_cell[r, c] = True

    cell_to_id = np.full((h, w), -1, dtype=np.int32)
    corridors: list[Corridor] = []
    for r in range(h):
        for c in range(w):
            if not is_corridor_cell[r, c] or cell_to_id[r, c] != -1:
                continue
            # BFS to collect this corridor's cells (only through corridor cells).
            cid = len(corridors)
            corridor = Corridor()
            stack = [(r, c)]
            while stack:
                rr, cc = stack.pop()
                if cell_to_id[rr, cc] != -1:
                    continue
                cell_to_id[rr, cc] = cid
                corridor.cells.append((rr, cc))
                for nr, nc in _empty_neighbors(obstacle_map, rr, cc):
                    if is_corridor_cell[nr, nc] and cell_to_id[nr, nc] == -1:
                        stack.append((nr, nc))
            # Endpoints = cells with <=1 corridor-neighbor (i.e., they attach to
            # a decision point or are a dead-end tip).
            for (rr, cc) in corridor.cells:
                cn = [(nr, nc) for (nr, nc) in _empty_neighbors(obstacle_map, rr, cc)
                      if is_corridor_cell[nr, nc]]
                if len(cn) <= 1:
                    corridor.endpoints.append((rr, cc))
                    # Find adjacent decision point (non-corridor empty neighbor).
                    dps = [(nr, nc) for (nr, nc) in _empty_neighbors(obstacle_map, rr, cc)
                           if not is_corridor_cell[nr, nc]]
                    if dps:
                        corridor.decision_points[(rr, cc)] = dps[0]
            # Deduplicate endpoints just in case of degenerate 1-cell corridors.
            corridor.endpoints = list(dict.fromkeys(corridor.endpoints))
            corridors.append(corridor)
    return corridors, cell_to_id
