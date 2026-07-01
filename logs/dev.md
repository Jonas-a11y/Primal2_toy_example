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


## 2026-07-02 00:47 — Observation builder shipped

Wrote `obs/{astar, corridor_maps, builder}`, 13-channel FOV + 3 scalars. Design:
- BFS distance field (single-agent shortest path field) cached per goal.
- Corridor channels (dx, dy, blocking) as full-size grids (11×11 sparse choice from Section 3 approval).
- Future-position maps: per-neighbor A* recomputed when `(neighbor_position, neighbor_goal)` cache key changes.

## 2026-07-02 00:48 — Expert planner shipped

Prioritized-planning space-time A* with cell + edge reservations. Deterministic priority order per RNG shuffle. Falls back to "stay forever" for agents that can't find a path.

## 2026-07-02 00:50 — Policy net + losses shipped

Exact PRIMAL2 architecture: 2× VGG blocks → 1×1 conv → concat with goal-scalar FC → 2× FC → LSTM w/ residual → π (softmax), V. Losses: value + actor+entropy + valid (BCE-with-logits on sigmoid(logits)) + BC. Loss weights: α=0.5, β=1.0, ζ=0.5, σ_H=0.01. All from Eq. 1-4 in Section IV.D.

## 2026-07-02 00:52 — Training loop shipped

Single-worker A3C-style: env randomized per episode (size ∈ {10,15,20}, density ∈ [0.2,0.5], corridor ∈ {1,3,5,7}, n_agents=6-8). Coin-flip 0.5 IL vs. RL per episode. Actions masked+sampled from valid list at rollout. NAdam optimizer, lr=2e-5 with inverse-sqrt decay (paper V.B.1). Grad clip = 5.0.

Smoke on 10 episodes CPU: works. Profiled: ~5.7ms/step (obs 1.85ms, forward 1.75ms, valid 0.02ms, env 0.03ms). After obs refactor (per-step cache): obs down to 1.04ms/step, forward now the dominant cost.

## 2026-07-02 00:56 — Kicked off 6-hour training run

Started `python -m primal2_toy.train.main --episodes 50000 --deadline-hours 6 --device mps --n-agents 6 --seed 42` in background (PID 89832). Logs to `logs/train_stdout.log` + CSV metrics per episode.

At 125 episodes: IL loss dropping 11.7 → 5.0 = model is learning to imitate. RL loss noisy (expected).

## 2026-07-02 01:00 — Eval + demo shipped

Wrote `eval/rollout.py`, `eval/visualizer.py` (pygame), `eval/headless.py` (metrics), `eval/baselines.py` (random / greedy A*). Demo entry point at `demo.py`.

Baselines on seed=7,42 with 4 agents, 64 steps: random 0.016 throughput, greedy A* 0.07 (seed=7 deadlocked at 0 — this is the failure mode PRIMAL2's convention learning fixes).


## 2026-07-02 01:15 — Run 1 stopped at ep 450, diagnosing collapse

After ~450 episodes of run 1 (config: lr=2e-5, entropy=0.01 = paper defaults):
- IL BC loss: 1.59 → 0.54 (good).
- RL value loss: 18.5 → 0.07 (good — value fit).
- RL entropy: 1.61 → 0.29 (falling fast).
- RL goals/ep: 2.25 → 0.20 (**collapse**: agents standing still).

Hypothesis: single-worker training lacks the rollout diversity that the paper's
9-worker A3C provides, so low entropy weight leads to premature convergence to
"stay-forever" policy (which trivially avoids collisions and stays valid).

Also noticed: `valid_rate=1.000` was misleading because we mask before sampling —
metric measured nothing about the policy. Fixed to compare unmasked argmax.

## 2026-07-02 01:19 — Iteration 2 config

- `lr` 2e-5 → **5e-5**
- `entropy_weight` 0.01 → **0.05**
- Added `il_warmup_episodes = 500` (IL-only for first 500 eps to bootstrap a policy).
- `valid_rate` metric now uses unmasked argmax (Section V.A.1 spirit).
- Also fixed the accidental 2× on value loss (removed internal 0.5 factor).

Restarted training as run 2 with seed=43. `deadline-hours=5.5`.

Archived run 1 checkpoint as `checkpoints/primal2_run1_stopped_ep450.pt` for A/B comparison later.

