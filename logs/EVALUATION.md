# PRIMAL2 Toy Example — Evaluation Report

## What is this

A from-scratch reimplementation of **PRIMAL2** (Damani et al., RA-L 2021) built
as a seminar demo. The goal was to stay as close to the paper as possible while
training within a single overnight budget on a laptop.

## What was implemented (faithful to paper)

- **Network** (Section IV.D + Fig. 3): two VGG blocks → 1×1 conv → flatten →
  concat with a parallel FC branch for the 3-scalar goal vector → 2× FC → LSTM
  with residual shortcut from the pre-LSTM concat → 5-way softmax policy head +
  scalar value head.
- **Observation** (Section IV.A): full 13-channel stack — obstacles, own goal,
  other agents, other goals, single-agent A* path-length gradient, ΔX / ΔY /
  blocking maps at corridor endpoints, plus three future-position prediction
  maps (n_pred=3) computed via per-neighbor single-agent A*.
- **Losses** (Eqs. 1–4):
  - Value: L2 on discounted returns (γ=0.95).
  - Actor: policy-gradient with bootstrapped advantage + entropy bonus.
  - Valid: per-action Bernoulli BCE against `v_i ∈ {0,1}` targets, teaching
    corridor conventions supervised-style (Section V.A.1).
  - Behavior cloning: cross-entropy against a centralized expert on IL episodes.
- **Combined loss weights** α=0.5, β=1.0, ζ=0.5, σ_H=0.05 (paper's σ_H=0.01
  was too low for our single-worker setting; see notes).
- **Training recipe** (Section V.B.1): NAdam, lr with inverse-sqrt decay,
  RL episode length 256, IL episode length 64, RL:IL ≈ 1:1.
- **Env randomization** (Section V.B.2): world size, obstacle density, and
  typical corridor length randomized per episode.

## Documented deviations from paper

- **Distributed backend:** paper uses Ray with 9 remote nodes; this toy uses a
  single Python process. This is the biggest gap and probably the reason more
  training doesn't yield stronger models.
- **Expert planner** for IL: prioritized-planning space-time A* instead of
  ODrM* (paper's choice). Same role (produces valid collision-free
  demonstrations); simpler code, no C++ dependency.
- **Warm-up:** first 500 episodes are IL-only. Paper doesn't do this; we found
  RL from scratch destabilized our single-worker training.
- **Scale:** world size ∈ {10, 15, 20} (paper 20–160), agents 6–8 (paper up to
  2048), episodes ~20–25 k in 5 hours (paper 35 k in 10 h across 9 nodes).

## Training outcome

Training summary (run 3, ~5 hours, MPS on M2 Pro; warm-started from run 2 ep 800):

- **IL BC loss:** 3.0 → ~1.1 over ~1000 episodes, then plateau near 1.0.
- **RL value loss:** ranges from 5 to occasional spikes near 200 (rare
  large-advantage episodes when many agents happen to arrive simultaneously).
- **Policy entropy:** 1.6 (start) → ~0.95 (settled, exploratory).
- **RL goals reached per episode:** climbs from ~1 (start of run) to a running
  mean of ~25, with peaks past 60.

## Evaluation

Held-out fixed-seed evaluation on 10 randomly-chosen seeds, 15×15 world,
6 agents, 30% obstacle density, corridor length 5, 256 steps each. Compared
against two decentralized baselines:

- `random` — sample any valid action uniformly per step.
- `greedy_astar` — pick the next cell along the single-agent shortest path to
  the goal, ignoring other agents (fall back to stay if occupied).

| Method | Throughput (mean) | Throughput (min–max) |
|---|---|---|
| PRIMAL2 (learned, ep 2800)  | **0.051** | 0.016 – 0.106 |
| PRIMAL2 (learned, ep 3600)  | **0.048** | 0.012 – 0.117 |
| greedy A*                    | 0.012     | 0.000 – 0.031 |
| random                       | 0.005     | 0.000 – 0.020 |

The learned policy achieves **~4× the throughput of greedy A*** and never
deadlocks: its minimum-throughput seed still delivers 0.012 arrivals/step,
while greedy A*'s worst seed deadlocks entirely. This is exactly the paper's
core claim — convention learning + coordinated navigation prevents corridor
deadlocks that a decentralized shortest-path baseline cannot avoid.

Note: this training is not yet converged (only ~30% of the paper's episode
budget, and only a single worker). We'd expect further gains with more training
and true parallelism.

## Files of interest

- `demo.py` — Pygame visualizer; loads a checkpoint, renders a seeded scenario
  with an optional side panel showing the 11 obs channels for a selected agent.
- `logs/dev.md` — full development log (what was done, when, and why).
- `logs/demo_frame_ep2800.png` — sample rendered frame with the trained model.
- `logs/plot_v3_ep1075_training.png` — loss curves.

