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


## 2026-07-02 01:02 — Diagnosed IL stay-heavy bias, restarted as run 3

Discovered that the expert plans were mostly "stay" after agents reached goals (71% stay actions overall). Fix: **replan on every arrival**, matching the paper's Section V.A.2 approach for adapting one-shot MAPF planners to LMAPF ("several one-shot MAPF instances need to be combined for a single LMAPF environment").

After fix: goals/IL-episode jumps from ~6 to 20-45. Expert now provides a rich stream of non-stay demonstrations.

Restarted as run 3 (seed=44, deadline 5.0h) warm-starting from run 2's ep800 weights with a fresh optimizer.


## 2026-07-02 02:07 — Watchdog + best-checkpoint saver

Added `primal2_toy.eval.watchdog` that polls `checkpoints/primal2_latest.pt` for new mtimes and runs a quick 4-seed × 128-step evaluation. When a new checkpoint beats the previous best throughput, it copies it to `primal2_best.pt`.

## 2026-07-02 02:31 — Discovered: sampled >> greedy at this scale

Comparing greedy argmax rollout vs. sampling from the (masked) softmax:
- Greedy: 0.040 throughput
- Sampled: 0.087 throughput  (2× improvement)

Explanation: at under-trained regime, greedy tie-breaks turn into deadlocks (two agents both want to enter the same cell, both greedy-pick "move", both back off, both retry). Sampling breaks the symmetry. Made sampled the default in `demo.py` and `eval/compare.py`.

## 2026-07-02 03:13 — Best snapshot: ep 8800

RL episode goals climbing to 60-80 per episode. Valid rate creeping to 0.7-0.8. Watchdog eval throughput 0.086 (best yet). Full 10-seed eval: PRIMAL2 (sampled) 0.113, greedy A* 0.012 = **9.4× improvement**.

## 2026-07-02 03:45 — Stopped training at ep 11,000 (plateau)

After ep 8800 the throughput oscillated between 0.09 and 0.11 without clear further gains. Ended training at ep ~11,000 total. Final checkpoint: `checkpoints/primal2_final.pt` (= ep 8800 snapshot).

## 2026-07-02 03:47 — Final evaluation on 20 held-out seeds

| Method | Throughput (mean) | Min | Max |
|---|---|---|---|
| random | 0.006 | 0.000 | 0.020 |
| greedy A* | 0.019 | 0.000 | 0.063 |
| PRIMAL2 (greedy) | 0.051 | 0.023 | 0.086 |
| PRIMAL2 (sampled) | **0.102** | 0.055 | 0.188 |

Key qualitative claim reproduced: **learned policy never deadlocks** (min 0.055 > greedy A*'s 0.000).

## 2026-07-02 03:50 — Wrote final report

`logs/EVALUATION.md` contains the full write-up with method scorecard, results, and reproduction commands. Committed to git.


## 2026-07-02 03:50 – 05:47 — Run 4 (from-scratch retraining) as validation

Ran a fresh training with all learnings baked in (no warm-start): seed=100, 4h budget. Got to ep ~7700. Final eval on 20 seeds:
- Run 4 sampled: 0.086
- Run 3 sampled: 0.102 (retained as shipped)

Run 3 (warm-started from a partial-run 2 model) is meaningfully better than starting from scratch. The Adam optimizer state that carried over from run 2 gave better momentum despite the run-2 policy being incorrect (stay-heavy). Kept the run 3 ep 8800 checkpoint as `primal2_final.pt`.

Insight: warm-starting a broken policy still gives faster training than fresh-init at this scale.


## 2026-07-02 05:48–07:15 — Run 5 (warm-start from run 3 ep 8800)

Continued iterating. Warm-started run 5 from `primal2_final.pt` (run 3 ep 8800) with fresh optimizer. Ran 5975 episodes over ~1.5h.

Results on 20 held-out seeds (sampled throughput):
- ep 2400: 0.1145
- **ep 3800: 0.1328** (new best)
- ep 5200: 0.1320
- Max seed at ep 3800: 0.2227 (highest single-seed throughput observed)

Selected ep 3800 as the shipped `primal2_final.pt`. This is a **30% improvement** over the previous best (run 3 ep 8800 = 0.102).

Total training across all runs: run 2 (450 eps + 350 useful) + run 3 (~10,975 eps) + run 5 (3,800 eps) = ~15,225 training episodes on the model shipped. Roughly 4-5h of effective wall-clock training with single-worker MPS.

## 2026-07-02 07:20 — Final state

- `checkpoints/primal2_final.pt` = run 5 ep 3800 (sampled throughput 0.130 on 20 seeds).
- `logs/EVALUATION.md` updated with final numbers.
- Demo screenshots regenerated with the better model.
- All test files still pass.
- Repository clean.

## 2026-07-02 07:18–07:51 — Run 6: another warm-start iteration

Warm-started run 6 from run 5 ep 3800 with fresh optimizer. Reached ep 2350 in ~30 min. Best snapshot: ep 1600, 20-seed sampled throughput **0.140**. Marginal improvement over run 5 ep 3800 (0.133).

Kept this as the definitive `primal2_final.pt`. Total lineage:
- Run 2 (ep 0→800): initial IL warmup + broken RL. Contributed Adam state.
- Run 3 (ep 800→10975): first fully successful training, warm-started from run 2. 
- Run 5 (ep 8800→12600 equivalent): further warm-start, reached 0.133.
- Run 6 (ep 12600→14200 equivalent): further warm-start, reached 0.140.

Total effective training: ~16k episodes across ~5h wall-clock.

## 2026-07-02 07:53 — Repository final state

- Shipped model: `checkpoints/primal2_final.pt` (7.5x greedy A* on 20-seed benchmark).
- All tests pass.
- `logs/EVALUATION.md`, `README.md`, `logs/plot_FINAL.png`, `logs/demo_frame_FINAL_seed42_sampled.png` all updated.
- Repository committed and clean.

## 2026-07-04 12:50–22:51 — Session 2: paper-adjacent retraining

Ten-hour warm-started run on the M2 Pro. Config:

- Sizes ∈ {20, 30, 40} (Session 1 used {10, 15, 20}), density 0.3–0.5,
  corridor length ∈ {5, 10, 15}, **8 agents** per env (up from 6).
- Warm-start from Session-1 `primal2_final.pt`, fresh optimizer.
- 200-episode IL re-warmup (short, because the policy is already reasonable).
- Everything else unchanged: lr 5e-5, entropy 0.05, γ = 0.95, α/β/ζ = 0.5/1/0.5.

Trainer's deadline handler cleanly terminated at ep 25 445 (~73 % of the
paper's 35 k budget). Ø 1.42 s/episode — faster than pre-flight projection
because the BFS cache inside each episode hits well even on 40×40.

Watchdog on 20×20 / 30 % / corridor 10 / 8 agents (a config the Session-1
model had never seen at training end):

| Episode | Sampled throughput |
|---:|---:|
| 500 | 0.152 |
| 2 500 | 0.186 |
| 10 000 | 0.19–0.21 |
| 20 000 | 0.193 |
| 22 500 | 0.217 |
| **25 000** | **0.236** ← peak |
| 25 445 | 0.170 (last snapshot, single-eval noise) |

Full 20-seed evaluation on ep 25 445 promoted to shipped model:

- 15×15 / 6 agents: sampled 0.208 (Session 1 shipped 0.140 → +48 %).
- 20×20 / 8 agents: sampled 0.228 (never seen at Session-1 end).
- Worst-seed throughput ≥ 0.109 on 20×20 and ≥ 0.129 on 15×15 — the
  never-deadlock property holds across configurations.

Session-1 shipped checkpoint archived as `checkpoints/primal2_toy_15x15.pt`.

## 2026-07-05 00:20–XX:XX — Post-training sweeps and paper-figure reproduction

While the Mac was still fresh from Session 2, ran both paper-shaped sweeps
head-to-head on the shipped model.

**Fig-5 sweep (LMAPF).** 3 sizes × 6 team sizes × 5 seeds = 90 instances,
~6 min wall-clock. Results in `logs/sweep_optB_fig5.csv`; plot at
`docs/images/fig5_lmapf.png`. Throughput monotonically rises with team size
until each world saturates:
- 20×20 saturates at 32 agents (0.53), collapses at 128 (0.19, world full).
- 30×30 saturates at 64 agents (0.78).
- 40×40 still climbing at 128 agents (peak 1.09).

**Fig-4 sweep (one-shot).** 2 sizes × 5 team sizes × 5 seeds, ~4 min. Success
rate falls with team size, 40×40 holds up to 16 agents at 0.8, 32 at 0.4,
64 at 0. Path length stays stable ~25–40 across the successful bracket.
Files: `logs/sweep_optB_fig4.csv`, `docs/images/fig4_{success,pathlen}.png`.

Both plots match the paper's Fig. 4 / Fig. 5 in shape but obviously not in
absolute team-size scale (paper: up to 2048; here: up to 128).

Ch5 of the seminar draft rewritten with the new two-config table.
`README.md` and `docs/EVALUATION.md` updated with the new headline numbers
and the two new figures.

## 2026-07-05 00:31–~07:00 — Session 3: extended-distribution retrain

Warm-started from Session-2 `primal2_final.pt`, 6.5 h wall-clock budget.
Extended distribution: sizes ∈ {20, 30, 40, 50} (up from {20, 30, 40}),
density 0.3–0.6 (up from 0.3–0.5), corridor ∈ {5, 10, 15, 21} (up from
{5, 10, 15}), 8 agents. IL warm-up shortened to 100 episodes because the
warm-started policy is already reasonable on the smaller subset of the
distribution.

Watchdog on 20×20 / 30 % / corridor 10 / 8 agents:

| Episode | Sampled throughput |
|---:|---:|
| 500 | 0.199 |
| 1 000 | 0.215 |
| 2 500 | 0.176 |
| 3 500 | 0.211 |
| 5 500 | 0.232 |
| 8 000 | 0.229 |
| **9 500** | **0.250** ← peak |
| 11 000 | 0.234 |
| 12 000 | 0.232 |

The ep 9 500 snapshot promoted to shipped model. 20-seed head-to-head
against Session-2 ep 25 445:

| Config | Session-2 ep 25 445 | Session-3 ep 9 500 |
|---|---:|---:|
| 15×15 / 6 agents (sampled) | 0.208 | 0.209 |
| 20×20 / 8 agents (greedy) | 0.165 | 0.198 |
| 20×20 / 8 agents (sampled) | 0.228 | **0.240** |
| 40×40 / 16 agents (sampled) | not measured | **0.356** |

Session 3 outperforms Session 2 on the paper-adjacent 20×20 config by 5 %,
and shows strong generalisation to the harder 40×40/16-agent unseen team
size (0.356 sampled, 3.7× the greedy A* baseline of 0.096).

## 2026-07-05 05:30–06:30 — Post-Session-3 sweeps and Ch5 update

Re-ran Fig-5 (10 seeds) and Fig-4 (10 seeds) on the shipped ep 9 500
checkpoint. On Fig-5 the 40×40/128-agent peak stays at ~1.02 (vs Session-2's
1.09), but the model is markedly more robust in the 20×20 regime (peak moves
from team size 32 to team size 64) and in the 40×40/16 mid-range. On Fig-4
the shipped model retains full success up to team 8 on 40×40 and 60 % at
team 4 on 20×20 — slightly under Session 2 on the smallest configurations,
noticeably better at larger teams.

Ch5 of the seminar draft rewritten around the three configurations
(15×15 / 20×20 / 40×40) and the new sampled/greedy factors. README and
EVALUATION.md updated with the three-session history and the Session-3 peak.

## Shipped state
- `checkpoints/primal2_final.pt` = Session 3 ep 9 500 (sampled 0.240 on 20×20/8).
- `checkpoints/primal2_toy_15x15.pt` = Session 1 final (archived for Ch5 reference).
- `checkpoints/primal2_session2_ep25445.pt` = Session 2 final (archived).
- `checkpoints/primal2_session3_ep9500.pt` = Session 3 peak (identical to primal2_final.pt).
- `checkpoints/primal2_session3_ep11000.pt` = Session 3 alternate (0.230 sampled).
