"""Grid state, actions, and step semantics.

The world is a 2D 4-connected grid. Cells are either empty (0) or obstacles (1).
Agents occupy exactly one empty cell each. Actions are 5-way:
    0 = stay, 1 = up, 2 = down, 3 = left, 4 = right
Action semantics follow PRIMAL2 (Section III.A + IV.C):
  - Moves into obstacles are invalid.
  - Moves into cells occupied by another agent are invalid.
  - Two agents may still collide when both attempt to enter the same empty cell
    (paper allows this and penalizes with -2).
  - Agents cannot return to the cell they occupied in the previous timestep
    (paper's "no immediate reversal" rule); allowed to stay still.
"""
from __future__ import annotations

import numpy as np

# Action encoding, in (dr, dc) form. r = row (grows downward), c = column.
ACTIONS: tuple[tuple[int, int], ...] = (
    (0, 0),   # stay
    (-1, 0),  # up
    (1, 0),   # down
    (0, -1),  # left
    (0, 1),   # right
)
NUM_ACTIONS = len(ACTIONS)

EMPTY = 0
OBSTACLE = 1


class GridWorld:
    """A 4-connected grid with a set of agents.

    Parameters
    ----------
    obstacle_map : (H, W) uint8 array. 1 = obstacle, 0 = empty.
    agent_positions : list of (r, c) tuples, one per agent.
    """

    def __init__(self, obstacle_map: np.ndarray, agent_positions: list[tuple[int, int]]):
        self.obstacle_map = obstacle_map.astype(np.uint8)
        self.h, self.w = self.obstacle_map.shape
        self.n_agents = len(agent_positions)
        self.positions = np.array(agent_positions, dtype=np.int32)  # (N, 2)
        self.prev_positions = self.positions.copy()

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.h and 0 <= c < self.w

    def is_obstacle(self, r: int, c: int) -> bool:
        return not self.in_bounds(r, c) or bool(self.obstacle_map[r, c])

    def occupied_by(self, r: int, c: int) -> int:
        """Return index of agent at (r, c), or -1 if empty."""
        matches = np.where((self.positions[:, 0] == r) & (self.positions[:, 1] == c))[0]
        if len(matches) == 0:
            return -1
        return int(matches[0])

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply per-agent actions.

        Returns
        -------
        moved : (N,) bool array — True if the agent's position changed.
        collided : (N,) bool array — True if the agent was involved in a swap or
            edge-conflict collision at this step.
        """
        assert actions.shape == (self.n_agents,)
        new_positions = self.positions.copy()

        # First pass: compute intended new positions, veto invalid single-agent moves.
        intended = self.positions.copy()
        for i, a in enumerate(actions):
            dr, dc = ACTIONS[int(a)]
            nr, nc = int(self.positions[i, 0] + dr), int(self.positions[i, 1] + dc)
            if not self.in_bounds(nr, nc) or self.obstacle_map[nr, nc]:
                # invalid move — stay
                nr, nc = int(self.positions[i, 0]), int(self.positions[i, 1])
            intended[i, 0], intended[i, 1] = nr, nc

        # Second pass: resolve conflicts.
        #   - Cell conflict: two agents intend the same empty cell -> both stay, collide.
        #   - Edge conflict: A and B swap -> both stay, collide.
        #   - Move-into-currently-occupied cell where that agent stays or moves elsewhere:
        #       If the occupant stays, the mover cannot enter; mover stays. Not a collision.
        collided = np.zeros(self.n_agents, dtype=bool)

        # Detect cell conflicts.
        cell_to_agents: dict[tuple[int, int], list[int]] = {}
        for i in range(self.n_agents):
            key = (int(intended[i, 0]), int(intended[i, 1]))
            cell_to_agents.setdefault(key, []).append(i)
        for cell, agents in cell_to_agents.items():
            if len(agents) > 1:
                # Multi-way conflict: everyone stays, everyone collides.
                for i in agents:
                    intended[i] = self.positions[i]
                    collided[i] = True

        # Detect swap conflicts.
        pos_tuple = {(int(self.positions[i, 0]), int(self.positions[i, 1])): i
                     for i in range(self.n_agents)}
        for i in range(self.n_agents):
            if collided[i]:
                continue
            tgt = (int(intended[i, 0]), int(intended[i, 1]))
            if tgt == (int(self.positions[i, 0]), int(self.positions[i, 1])):
                continue  # staying
            j = pos_tuple.get(tgt, -1)
            if j == -1 or j == i:
                continue
            # j is currently at intended[i]; is j moving to i's current cell?
            j_tgt = (int(intended[j, 0]), int(intended[j, 1]))
            if j_tgt == (int(self.positions[i, 0]), int(self.positions[i, 1])):
                # swap -> collide, both stay
                intended[i] = self.positions[i]
                intended[j] = self.positions[j]
                collided[i] = True
                collided[j] = True

        # Prevent moves into currently-occupied cells if that occupant stays.
        # We iterate until no changes propagate.
        changed = True
        while changed:
            changed = False
            for i in range(self.n_agents):
                tgt = (int(intended[i, 0]), int(intended[i, 1]))
                cur = (int(self.positions[i, 0]), int(self.positions[i, 1]))
                if tgt == cur:
                    continue
                j = pos_tuple.get(tgt, -1)
                if j == -1 or j == i:
                    continue
                # Check whether occupant j is actually leaving this cell.
                j_tgt = (int(intended[j, 0]), int(intended[j, 1]))
                if j_tgt == tgt:  # occupant staying
                    intended[i] = self.positions[i]
                    changed = True

        moved = np.any(intended != self.positions, axis=1)
        self.prev_positions = self.positions
        self.positions = intended
        return moved, collided
