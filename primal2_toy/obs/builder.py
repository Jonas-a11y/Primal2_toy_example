"""Assemble the 13-channel FOV observation + 3 non-spatial scalars per agent.

Channel order (indexing into the output tensor's C dim):
    0  obstacle map
    1  own goal map (1 where own goal falls in FOV)
    2  other agents map
    3  other goals map
    4  path-length map (normalized single-agent A* distance to own goal)
    5  ΔX at corridor endpoints
    6  ΔY at corridor endpoints
    7  blocking at corridor endpoints
    8  future-position map t+1
    9  future-position map t+2
    10 future-position map t+3

FOV is (F, F), centered on the agent. Cells outside the world are zero-padded.
The FOV size is `fov` (paper: 11). `n_pred` is the number of future-step maps
(paper: 3, hence channels 8..10 above).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..env.grid import GridWorld
from ..env.corridor import Corridor
from .astar import bfs_distance_field, astar_path
from .corridor_maps import build_blocking, paint_corridor_channels


@dataclass
class ObsSpec:
    fov: int = 11
    n_pred: int = 3

    @property
    def num_spatial_channels(self) -> int:
        # 5 basic (obs, own goal, others, others' goals, path length)
        # + 3 corridor (dx, dy, blocking)
        # + n_pred future-position channels
        return 5 + 3 + self.n_pred

    @property
    def num_scalar_features(self) -> int:
        # unit vec dx, dy + magnitude
        return 3


def _slice_fov(grid_map: np.ndarray, center: tuple[int, int], fov: int) -> np.ndarray:
    """Return a (fov, fov) view of `grid_map` centered at `center`, zero-padded.

    grid_map may be float or int; we return float32.
    """
    h, w = grid_map.shape
    r, c = center
    half = fov // 2
    out = np.zeros((fov, fov), dtype=np.float32)
    # Compute intersection window.
    r0, r1 = max(0, r - half), min(h, r + half + 1)
    c0, c1 = max(0, c - half), min(w, c + half + 1)
    if r0 >= r1 or c0 >= c1:
        return out
    or0 = r0 - (r - half)
    oc0 = c0 - (c - half)
    out[or0:or0 + (r1 - r0), oc0:oc0 + (c1 - c0)] = grid_map[r0:r1, c0:c1].astype(np.float32)
    return out


class ObservationBuilder:
    """Builds per-agent observations. Keeps per-map caches (BFS fields, corridor decomposition)."""

    def __init__(
        self,
        grid: GridWorld,
        goals: np.ndarray,
        corridors: list[Corridor],
        cell_to_corridor: np.ndarray,
        spec: ObsSpec = ObsSpec(),
    ):
        self.grid = grid
        self.goals = goals
        self.corridors = corridors
        self.cell_to_corridor = cell_to_corridor
        self.spec = spec
        self._bfs_cache: dict[tuple[int, int], np.ndarray] = {}
        # Future-position paths cached per (agent_index, goal_tuple, start_tuple).
        self._future_paths: dict[int, tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]] = {}
        # Per-step cached corridor channels; refreshed by `refresh_step_cache()`.
        self._dx_full: np.ndarray | None = None
        self._dy_full: np.ndarray | None = None
        self._bl_full: np.ndarray | None = None
        # Per-step cached others/others'-goals maps.
        self._others: np.ndarray | None = None
        self._others_goals: np.ndarray | None = None

    def _bfs_field_for(self, goal: tuple[int, int]) -> np.ndarray:
        if goal not in self._bfs_cache:
            self._bfs_cache[goal] = bfs_distance_field(self.grid.obstacle_map, goal)
        return self._bfs_cache[goal]

    def invalidate_goal(self, goal: tuple[int, int]) -> None:
        self._bfs_cache.pop(goal, None)

    def refresh_step_cache(self) -> None:
        """Recompute per-step shared state (corridor blocking, others maps).

        Call this once per timestep before building observations for any agent.
        """
        blocking = build_blocking(
            self.corridors, self.cell_to_corridor,
            self.grid.positions, self.grid.prev_positions,
        )
        self._dx_full, self._dy_full, self._bl_full = paint_corridor_channels(
            self.grid.obstacle_map, self.corridors, blocking
        )
        h, w = self.grid.obstacle_map.shape
        others = np.zeros((h, w), dtype=np.float32)
        others_goals = np.zeros((h, w), dtype=np.float32)
        for j in range(self.grid.n_agents):
            others[int(self.grid.positions[j, 0]), int(self.grid.positions[j, 1])] = 1.0
            others_goals[int(self.goals[j, 0]), int(self.goals[j, 1])] = 1.0
        self._others = others
        self._others_goals = others_goals

    def build(self, agent_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (spatial (C, F, F), scalars (3,)).

        Call `refresh_step_cache()` once per timestep before batch-building.
        """
        if self._dx_full is None:
            self.refresh_step_cache()
        F = self.spec.fov
        C = self.spec.num_spatial_channels
        pos = (int(self.grid.positions[agent_id, 0]), int(self.grid.positions[agent_id, 1]))
        goal = (int(self.goals[agent_id, 0]), int(self.goals[agent_id, 1]))

        spatial = np.zeros((C, F, F), dtype=np.float32)
        # Ch 0: obstacles in FOV.
        spatial[0] = _slice_fov(self.grid.obstacle_map, pos, F)
        # Ch 1: own goal in FOV.
        goal_map = np.zeros_like(self.grid.obstacle_map, dtype=np.float32)
        goal_map[goal[0], goal[1]] = 1.0
        spatial[1] = _slice_fov(goal_map, pos, F)
        # Ch 2/3: other agents and their goals (subtract self from precomputed maps).
        others = self._others.copy()  # type: ignore[union-attr]
        others_goals = self._others_goals.copy()  # type: ignore[union-attr]
        others[pos[0], pos[1]] = 0.0
        others_goals[goal[0], goal[1]] = 0.0
        spatial[2] = _slice_fov(others, pos, F)
        spatial[3] = _slice_fov(others_goals, pos, F)
        # Ch 4: normalized path-length map to own goal (non-obstacle cells only).
        dist = self._bfs_field_for(goal).astype(np.float32)
        fov_dist = _slice_fov(dist, pos, F).copy()
        # Set unreachable/obstacle cells (dist == -1) to 0.
        neg = fov_dist < 0
        fov_dist[neg] = 0.0
        # Normalize by max in FOV so gradient shape is preserved but scale is stable.
        m = float(fov_dist.max())
        if m > 0:
            fov_dist = fov_dist / m
        spatial[4] = fov_dist
        # Ch 5..7: corridor channels (sliced from per-step cache).
        spatial[5] = _slice_fov(self._dx_full, pos, F)  # type: ignore[arg-type]
        spatial[6] = _slice_fov(self._dy_full, pos, F)  # type: ignore[arg-type]
        spatial[7] = _slice_fov(self._bl_full, pos, F)  # type: ignore[arg-type]
        # Ch 8..8+n_pred-1: future-position predictions of neighbors in FOV.
        for k in range(self.spec.n_pred):
            spatial[8 + k] = np.zeros((F, F), dtype=np.float32)
        half = F // 2
        r0, c0 = pos[0] - half, pos[1] - half
        for j in range(self.grid.n_agents):
            if j == agent_id:
                continue
            nj = (int(self.grid.positions[j, 0]), int(self.grid.positions[j, 1]))
            if not (r0 <= nj[0] < r0 + F and c0 <= nj[1] < c0 + F):
                continue  # neighbor not in this agent's FOV
            nj_goal = (int(self.goals[j, 0]), int(self.goals[j, 1]))
            # Cache path prediction per (j, current position, goal). Recompute only
            # when the neighbor's start-or-goal changes.
            cached = self._future_paths.get(j)
            if cached is None or cached[0] != nj or cached[1] != nj_goal:
                path = astar_path(self.grid.obstacle_map, nj, nj_goal) or [nj]
                self._future_paths[j] = (nj, nj_goal, path)
            path = self._future_paths[j][2]
            # path[0] = current position; project positions t+1..t+n_pred.
            for k in range(self.spec.n_pred):
                idx = min(k + 1, len(path) - 1)
                pr, pc = path[idx]
                fr, fc = pr - r0, pc - c0
                if 0 <= fr < F and 0 <= fc < F:
                    spatial[8 + k, fr, fc] = 1.0

        # Non-spatial features: goal unit-vec + magnitude.
        dr = goal[0] - pos[0]
        dc = goal[1] - pos[1]
        mag = float(np.hypot(dr, dc))
        if mag > 0:
            udr, udc = dr / mag, dc / mag
        else:
            udr, udc = 0.0, 0.0
        scalars = np.array([udr, udc, mag / max(self.grid.h, self.grid.w)], dtype=np.float32)
        return spatial, scalars
