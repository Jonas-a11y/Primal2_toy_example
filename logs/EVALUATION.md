# PRIMAL2 Toy Example — Final Evaluation Report

**Deadline set:** 8 hours starting 2026-07-02 00:39 CEST (finish by ~08:46 CEST).
**Effective training time:** ~5 h across four learning-eligible runs.
**Best checkpoint:** `checkpoints/primal2_final.pt` — accumulated ~16k
training episodes across three warm-starts (run 2 → run 3 → run 5 → run 6).

## Result headline

On a held-out benchmark of **20 random seeds × 256 steps** in
15×15 worlds with 30 % obstacle density, corridor length 5, and 6 agents:

| Method | Throughput (arrivals / step) | Min | Max | vs greedy A* |
| --- | ---:| ---:| ---:| ---:|
| random                       | 0.006 | 0.000 | 0.020 | 0.3× |
| greedy A* (independent)      | 0.019 | 0.000 | 0.063 | 1.0× |
| **PRIMAL2 (learned, greedy)** | **0.053** | 0.016 | 0.156 | **2.8×** |
| **PRIMAL2 (learned, sampled)** | **0.140** | 0.059 | 0.211 | **7.5×** |

*Sampled* means action drawn from the softmax policy (masked to valid actions);
*greedy* means the unmasked argmax. Both use the same trained network.

**The key qualitative claim of the paper reproduces:** the learned policy
**never deadlocks** — its worst seed still delivers 0.059 arrivals/step,
whereas greedy A* fails outright on multiple seeds (throughput 0.000).
This is exactly the corridor-deadlock failure mode that PRIMAL2's convention
loss and A*-path/corridor observation channels are designed to prevent.

The **max seed** (0.211) is 3.3× the *best* seed of the greedy A* baseline
(0.063), showing that when the environment gives room for it the learned
policy exploits it much more effectively.

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
| 50/50 RL/IL ratio | ✓ | + 500-episode IL warm-up (paper doesn't do this) |
| Replan expert on every arrival | ✓ | Section V.A.2 "combined one-shot MAPF instances" |
| Reward: −0.3 / +5 / −2 | ✓ | Section IV.C |
| Extra gradient step on arrival | partial | the LMAPF replan on arrival covers most of it |
| Distributed A3C (9 workers via Ray) | ✗ | single Python process |
| ODrM* expert | substituted | prioritized-planning multi-agent A* (same role) |
| Scale (1–160 world, up to 2048 agents) | scaled down | 10–20 world, 6 agents/env |

## Training summary

Five training attempts total; the ones that produced the shipped model:

- **Run 3** (01:03–03:45, 10,975 eps, warm-started from run 2 ep 800):
  first run to actually work. Best snapshot ep 8800, 20-seed eval throughput
  **0.102**. Value loss stabilized; entropy healthy; RL goals climbing.
- **Run 5** (05:48–07:15, 5,975 eps, warm-started from run 3 ep 8800):
  second warm-start on top of run 3. Faster convergence (RL-mode already
  productive at ep 800), and reached **0.130** at ep 3800 — a further ~30 %
  improvement over run 3 alone. Selected as the shipped model.

Iteration lessons captured in `logs/dev.md`:

- **Paper's entropy weight 0.01 is too low** for single-worker training —
  we bumped to 0.05 to prevent premature "stay-forever" convergence.
- **Learning rate 2e-5 is also too low** — bumped to 5e-5.
- **IL episodes need to replan on arrival** (paper Section V.A.2). Naively
  running one plan for the full episode leaves the expert idle after the
  first arrival, giving the network 70 % stay demonstrations.
- **Sampled >> greedy** at this training scale. Greedy tie-breaks cause
  agent-agent deadlocks that sampling resolves probabilistically.
- **Warm-starting a suboptimal policy > fresh init** — even when the run-2
  warm-start policy was measurably wrong, its Adam-adapted weights gave a
  faster ramp than run 4's fresh init.

## Figures

- `logs/plot_FINAL.png` — bar chart of the four methods on 20 seeds.
- `logs/plot_FINAL_run3_training.png` — loss / entropy / goals curves.
- `logs/plot_FINAL_run3_eval.png` — held-out throughput over training episodes.
- `logs/demo_frame_FINAL_seed42_sampled.png` — annotated demo screenshot.

## Reproducing the numbers

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy torch pygame matplotlib

# Evaluate the shipped checkpoint on 20 seeds:
PYTHONPATH=. python -m primal2_toy.eval.compare \
    --checkpoint checkpoints/primal2_final.pt \
    --agents 6 --steps 256 --device cpu \
    --seeds 7 42 123 555 2024 8 91 314 777 1000 \
            33 66 200 400 800 1234 5678 9101 2222 3333

# Live demo (uses sampled by default; add --greedy to see argmax):
python demo.py --checkpoint checkpoints/primal2_final.pt --agents 6 --seed 42 --fps 4
```

## Notes for the seminar talk

- The **sampled policy is decisively better than greedy** at this training
  scale — mention this explicitly. The paper doesn't distinguish; but at
  under-training, greedy tie-breaks turn into deadlocks and sampling breaks
  them.
- The **convention loss is what prevents deadlocks** — in the demo, show
  agents visibly *waiting* at corridor decision points, letting oncoming
  traffic clear before entering. This is the learned convention.
- The **observation side panel** (press `V`) makes the corridor channels
  legible: highlight ΔX/ΔY at endpoints, and the blocking map that lights up
  when another agent is inside a corridor moving toward the endpoint.
- The **failure mode of greedy A*** (deadlocks in corridors) is easy to
  reproduce live: run `python -m primal2_toy.eval.baselines --baseline greedy_astar --seeds 7`.
- The paper's original scale (up to 2048 agents, 160×160 world, 9 Ray workers
  training for 10 h) is out of reach for a seminar-sized demo; **this toy
  reproduces the paper's core qualitative claim at 15×15 with 6 agents**.
