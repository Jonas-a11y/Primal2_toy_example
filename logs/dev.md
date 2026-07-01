# PRIMAL2 Toy Example — Development Log

Rolling narrative of what I'm doing, decisions made, and results. Newest entries at the bottom.

Deadline: **2026-07-03 09:00 CEST**. Local hardware: Apple M2 Pro (MPS) initially, RTX 3060 Ti optional.

---

## 2026-07-02 00:39 — Kickoff

Read the PRIMAL2 paper (Damani et al., RA-L 2021, arXiv:2010.08184v3). Design pinned in `docs/superpowers/specs/2026-07-02-primal2-toy-example-design.md`. Scope:

- Full paper fidelity: 13-channel obs (incl. future-position A* maps), paper's exact network (VGG+LSTM+residual), all four losses (value, actor+entropy, valid, BC), 50/50 RL/IL, prioritized-A* expert stand-in for ODrM*, env randomization.
- Scaled down: 4 workers, size ∈ {10,15,20}, 8 agents/env.

## 2026-07-02 00:40 — Environment scaffolded

Wrote `primal2_toy/env/{grid,maze,corridor,lmapf}.py`. Design choices:

- **Grid step semantics:** attempted-position resolution in three passes — invalid single-agent moves → veto; multi-agent same-cell → all stay + collide; A↔B swap → both stay + collide; move-into-occupied-that-stays → mover stays (fixed-point iteration). Matches the paper's move validity plus the "collisions still possible" wording in Section IV.C.
- **Maze generator:** wall-segment drop from Poisson(typical_corridor_length), horizontal/vertical at random. Reject/retry if not connected. Paper's exact algorithm isn't specified, only its parameters.
- **Corridor analyzer:** BFS over cells with ≤2 empty neighbors. Endpoints = corridor cells with ≤1 corridor-neighbor. Decision points = adjacent non-corridor empty neighbors.

Smoke tests passing on 15×15, density 0.3, corridor length 5 (found 3 corridors on a hand-crafted map).

