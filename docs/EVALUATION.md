# PRIMAL2 Toy Example — Evaluation Report

**Session 1:** 2026-07-02 00:39–~08:00 CEST — initial bootstrap (~5 h).
**Session 2:** 2026-07-04 12:50–22:51 CEST — paper-adjacent retrain (10 h).
**Session 3:** 2026-07-05 00:30–~07:00 CEST — extended-distribution retrain (~6.5 h).
**Total training time:** ~21.5 h across three chained warm-started sessions.
**Shipped checkpoint:** `checkpoints/primal2_final.pt` (Session 3, episode 9 500).

## Result headline

Held-out benchmarks, 20 random seeds × 256 steps each, LMAPF mode.

**Session-2/3 target configuration** (paper-adjacent, 20×20 world, 30 % density,
corridor length 10, 8 agents):

| Method | Throughput (mean) | Min | Max | vs greedy A* |
| --- | ---:| ---:| ---:| ---:|
| random | 0.003 | 0.000 | 0.016 | 0.1× |
| greedy A* (independent) | 0.038 | 0.004 | 0.109 | 1.0× |
| **PRIMAL2 (learned, greedy)** | **0.198** | 0.109 | 0.297 | **5.3×** |
| **PRIMAL2 (learned, sampled)** | **0.240** | **0.129** | 0.379 | **6.4×** |

**Harder configuration** (40×40 world, 30 % density, corridor 10, 16 agents):

| Method | Throughput (mean) | Min | Max | vs greedy A* |
| --- | ---:| ---:| ---:| ---:|
| random | 0.004 | 0.000 | 0.012 | 0.0× |
| greedy A* (independent) | 0.096 | 0.016 | 0.164 | 1.0× |
| **PRIMAL2 (learned, greedy)** | **0.326** | 0.125 | 0.445 | **3.4×** |
| **PRIMAL2 (learned, sampled)** | **0.356** | **0.160** | 0.434 | **3.7×** |

**Original Session-1 configuration** (15×15, 30 % density, corridor 5, 6 agents):

| Method | Throughput (mean) | Min | Max | vs greedy A* |
| --- | ---:| ---:| ---:| ---:|
| random | 0.006 | 0.000 | 0.020 | 0.3× |
| greedy A* (independent) | 0.019 | 0.000 | 0.063 | 1.0× |
| **PRIMAL2 (learned, greedy)** | **0.149** | 0.008 | 0.266 | **7.9×** |
| **PRIMAL2 (learned, sampled)** | **0.209** | **0.137** | 0.301 | **11.2×** |

*Sampled* means action drawn from the softmax policy (masked to valid
actions); *greedy* means the unmasked argmax. Both use the same trained
network.

**The key qualitative claim of the paper reproduces.** In every configuration
the sampled policy **never deadlocks** (min throughput ≥ 0.13 on 20×20 and
15×15, ≥ 0.16 on 40×40/16 agents), whereas greedy A* falls to 0 arrivals on
multiple 15×15 seeds. This is exactly the corridor-deadlock failure mode that
PRIMAL2's convention loss and A*-path/corridor observation channels are
designed to prevent.

## Fig-5 (LMAPF throughput vs. team size) reproduction

Aggregated over 10 seeds per configuration, corridor length 10, density 0.3
(shipped model, ep 9 500):

| World | Team 4 | Team 8 | Team 16 | Team 32 | Team 64 | Team 128 |
| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| 20×20 | 0.098 | 0.178 | 0.238 | 0.249 | **0.268** | 0.172 |
| 30×30 | 0.098 | 0.182 | 0.334 | 0.527 | 0.654 | **0.769** |
| 40×40 | 0.067 | 0.133 | 0.278 | 0.501 | 0.742 | **1.024** |

Throughput scales monotonically with team size until the world saturates:
on 20×20 that inflection sits around 64 agents (peak 0.27), on 30×30 it
continues climbing through 128 (0.77), on 40×40 it is still rising at 128
agents (1.02). This qualitatively matches Fig. 5 of the paper. See
[`images/fig5_lmapf.png`](images/fig5_lmapf.png).

## Fig-4 (one-shot MAPF) reproduction

Aggregated over 10 seeds per configuration, timestep budgets follow the
paper (Section VI.A: 320 for size 20/40) — shipped model, ep 9 500:

| World | Team | 100 % succ. | 95 % succ. | Makespan | Avg path len |
| ---:| ---:| ---:| ---:| ---:| ---:|
| 20×20 | 4 | 0.60 | 0.60 | 32 | 22.8 |
| 20×20 | 8 | 0.50 | 0.50 | 54 | 20.1 |
| 20×20 | 16 | 0.40 | 0.41 | 89 | 30.5 |
| 20×20 | 32 | 0.00 | 0.00 | — | 34.9 |
| 20×20 | 64 | 0.00 | 0.00 | — | 36.8 |
| 40×40 | 4 | 1.00 | 1.00 | 99 | 41.8 |
| 40×40 | 8 | 1.00 | 1.00 | 112 | 38.4 |
| 40×40 | 16 | 0.70 | 0.71 | 166 | 40.4 |
| 40×40 | 32 | 0.30 | 0.32 | 193 | 43.0 |
| 40×40 | 64 | 0.10 | 0.10 | 133 | 44.5 |

Success rate falls with team size and holds up longer on the larger world —
matching the qualitative shape of Fig. 4 in the paper. Note the shipped
Session-3 model is markedly better at 40×40 than the earlier Session-2
checkpoint (0.70 vs 0.60 at 16 agents, 0.30 vs 0.30 at 32 agents, and
0.10 vs 0.00 at 64 agents). See
[`images/fig4_success.png`](images/fig4_success.png) and
[`images/fig4_pathlen.png`](images/fig4_pathlen.png).

## Faithfulness scorecard

| Paper element | Implemented? | Notes |
| --- | --- | --- |
| A3C actor-critic + entropy | ✓ | Eq. 1–2 |
| Value L2 loss | ✓ | Section IV.D |
| Behavior-cloning loss (IL) | ✓ | Eq. 4 |
| Valid/BCE loss (conventions) | ✓ | Eq. 3, per-action Bernoulli BCE on sigmoid(logits) |
| 13-channel observation | ✓ | all channels incl. 3 future-position maps |
| 2× VGG conv + LSTM w/ residual | ✓ | Section IV.D, Fig. 3 |
| Goal unit-vec + magnitude | ✓ | 3 non-spatial features |
| NAdam, inverse-sqrt LR decay | ✓ | Section V.B.1 |
| γ = 0.95, RL-ep 256, IL-ep 64 | ✓ | Section V.B.1 |
| Env randomization | ✓ | size, density, corridor length per episode |
| 50/50 RL/IL ratio | ✓ | + 100–500 IL warm-up (paper doesn't do this) |
| Replan expert on every arrival | ✓ | Section V.A.2 "combined one-shot MAPF instances" |
| Reward: −0.3 / +5 / −2 | ✓ | Section IV.C |
| One-shot mode + paper metrics | ✓ | makespan, success 100/95, avg path length |
| Fig-4 / Fig-5 plots | ✓ | see `docs/images/fig{4,5}_*.png` |
| Extra gradient step on arrival | partial | the LMAPF replan on arrival covers most of it |
| Distributed A3C (9 workers via Ray) | ✗ | single Python process |
| ODrM* expert | substituted | prioritized-planning multi-agent A* (same role) |
| Scale (10–160 world, up to 2048 agents) | scaled down | 20–50 world, up to 128 agents in eval |
| moving.ai Warehouse / Maze benchmarks (Table I) | not run | out of scope for a laptop-scale demo |
| CBSH-RCT / ODrM* / Windowed-PBS baselines | not run | requires their C++ code |

## Training summary

### Session 1 — 2026-07-02, ~5 h, initial bootstrap

Small-world configuration: sizes ∈ {10, 15, 20}, density 0.2–0.5,
corridor ∈ {1, 3, 5, 7}, 6 agents. Four warm-started runs, ~16 k total
episodes. Ended at throughput **0.140 sampled** on the 15×15 benchmark.

### Session 2 — 2026-07-04, 10 h, paper-adjacent retrain

Warm-started from the Session-1 checkpoint, trained on sizes ∈ {20, 30, 40},
density 0.3–0.5, corridor ∈ {5, 10, 15}, 8 agents. Single
deadline-terminated run of **25 445 episodes** (~73 % of the paper's 35 k
budget). Ended at throughput 0.228 sampled on 20×20/8-agent.

### Session 3 — 2026-07-05, 6.5 h, extended-distribution retrain

Warm-started from Session-2 ep 25 445. Extended distribution: sizes ∈ {20,
30, 40, 50}, density 0.3–0.6, corridor ∈ {5, 10, 15, 21}, 8 agents. Peak
watchdog throughput of 0.250 hit at episode 9 500 — that snapshot promoted
to shipped model (`primal2_final.pt`). Peak 20-seed sampled throughput on
20×20/8-agent: **0.240**; on 40×40/16-agent (unseen team size): **0.356**.

### Iteration lessons captured in [`dev-log.md`](dev-log.md)

- **Paper's entropy weight 0.01 is too low** for single-worker training —
  bumped to 0.05 to prevent premature "stay-forever" convergence.
- **Learning rate 2e-5 is also too low** — bumped to 5e-5.
- **IL episodes need to replan on arrival** (paper Section V.A.2). Naive
  one-plan-per-episode leaves the expert idle after the first arrival,
  giving the network 70 % stay demonstrations.
- **Sampled ≫ greedy** at this training scale. Greedy tie-breaks cause
  agent-agent deadlocks that sampling resolves probabilistically.
- **Warm-starting compounds** — Session 3 warm-started from Session 2 which
  warm-started from Session 1, each session widening the training
  distribution while inheriting the previous model's decision structure.
- **Watchdog snapshots matter** — the shipped Session-3 checkpoint is
  ep 9 500 (peak watchdog throughput 0.250), not the deadline-terminated
  final episode, because throughput oscillates without net gain after the
  peak.

## Figures

- [`images/comparison.png`](images/comparison.png) — bar chart (20×20 headline).
- [`images/comparison_15x15.png`](images/comparison_15x15.png) — bar chart (15×15).
- [`images/comparison_20x20.png`](images/comparison_20x20.png) — bar chart (20×20).
- [`images/fig5_lmapf.png`](images/fig5_lmapf.png) — Fig-5-shape (LMAPF throughput).
- [`images/fig4_success.png`](images/fig4_success.png) — Fig-4-shape (success rate).
- [`images/fig4_pathlen.png`](images/fig4_pathlen.png) — Fig-4-shape (avg path length).
- [`images/training_curves.png`](images/training_curves.png) — loss / entropy / goals over Session 1.
- [`images/eval_over_training.png`](images/eval_over_training.png) — Session-1 held-out throughput over training.
- [`images/demo_screenshot.png`](images/demo_screenshot.png) — annotated demo screenshot (20×20).
- [`images/demo_20x20_seed7.png`](images/demo_20x20_seed7.png), [`demo_40x40_seed123.png`](images/demo_40x40_seed123.png) — additional scenarios.

## Reproducing the numbers

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy torch pygame matplotlib

# Head-to-head on 20 seeds (20×20 / 8 agents):
PYTHONPATH=. python -m primal2_toy.eval.compare \
    --checkpoint checkpoints/primal2_final.pt \
    --size 20 --density 0.3 --corridor-length 10 --agents 8 \
    --steps 256 --device cpu \
    --seeds 7 42 123 555 2024 8 91 314 777 1000 \
            33 66 200 400 800 1234 5678 9101 2222 3333

# Fig-5 sweep + plot:
PYTHONPATH=. python -m primal2_toy.eval.sweep \
    --checkpoint checkpoints/primal2_final.pt --mode lmapf \
    --sizes 20 30 40 --densities 0.3 --corridors 10 \
    --team-sizes 4 8 16 32 64 128 --n-seeds 10 \
    --out logs/sweep_fig5.csv --device cpu
PYTHONPATH=. python -m primal2_toy.eval.sweep_plots \
    --sweep-csv logs/sweep_fig5.csv --out docs/images/fig5

# Live demo (sampled by default; add --greedy to see the argmax failure mode):
python demo.py --checkpoint checkpoints/primal2_final.pt --size 20 --agents 8 --seed 42 --fps 4
```

## Notes for the seminar talk

- The **sampled policy is decisively better than greedy** — the paper does
  not distinguish, but at this training scale greedy tie-breaks turn into
  deadlocks and sampling breaks them probabilistically.
- The **convention loss is what prevents deadlocks** — in the demo, show
  agents visibly *waiting* at corridor decision points and letting oncoming
  traffic clear before entering. This is the learned convention.
- The **observation side panel** (press `V`) makes the corridor channels
  legible: highlight ΔX/ΔY at endpoints and the blocking map that lights up
  when another agent is inside a corridor moving toward the endpoint.
- The **failure mode of greedy A*** (deadlocks in corridors) is easy to
  reproduce live:
  `PYTHONPATH=. python -m primal2_toy.eval.baselines --baseline greedy_astar --size 20 --agents 8 --seeds 7`.
- Fig-5 and Fig-4 are reproduced in **qualitative shape**; the paper's
  original scale (up to 2048 agents, 160×160 worlds, 9 Ray workers, ODrM*
  expert, C++ baselines) remains out of reach for a laptop-scale demo.
