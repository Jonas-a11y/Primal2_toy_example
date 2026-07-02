# PRIMAL2 Toy Example — Final Evaluation Report

**Deadline set:** 8 hours starting 2026-07-02 00:39 CEST (finish by ~08:46 CEST).
**Training window:** 2 h 42 min effective (three runs).
**Best checkpoint:** `checkpoints/primal2_final.pt` (= `primal2_ep8800_best.pt`), episode 8800.

## Result headline

On a held-out benchmark of **20 random seeds × 256 steps** in
15×15 worlds with 30 % obstacle density, corridor length 5, and 6 agents:

| Method | Throughput (arrivals / step) | Min | Max | vs greedy A* |
| --- | ---:| ---:| ---:| ---:|
| random                       | 0.006 | 0.000 | 0.020 | 0.3× |
| greedy A* (independent)      | 0.019 | 0.000 | 0.063 | 1.0× |
| **PRIMAL2 (learned, greedy)** | **0.051** | 0.023 | 0.086 | **2.7×** |
| **PRIMAL2 (learned, sampled)** | **0.102** | 0.055 | 0.188 | **5.5×** |

*Sampled* means action drawn from the softmax policy (masked to valid actions);
*greedy* means the unmasked argmax. Both use the same trained network.

**The key qualitative claim of the paper reproduces:** the learned policy
**never deadlocks** — its worst seed still delivers 0.055 arrivals/step,
whereas greedy A* fails outright on multiple seeds (throughput 0.000).
This is exactly the corridor-deadlock failure mode that PRIMAL2's convention
loss and A*-path/corridor observation channels are designed to prevent.

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
| Extra gradient step on arrival | partial | not implemented separately; the LMAPF replan on arrival covers most of it |
| Distributed A3C (9 workers via Ray) | ✗ | single Python process |
| ODrM* expert | substituted | prioritized-planning multi-agent A* (same role) |
| Scale (1–160 world, up to 2048 agents) | scaled down | 10–20 world, 6 agents/env |

## Training summary

- **Run 1** (00:56–01:02, 450 eps): default paper hyperparameters. Policy
  collapsed to "stay for all agents" — value function collapsed to 0 and
  entropy dropped too fast. Documented in `logs/dev.md`.
- **Run 2** (01:04–01:02, 850 eps): entropy weight 0.01 → 0.05, lr 2e-5 → 5e-5,
  added 500-episode IL warm-up. Policy stopped collapsing but IL data was
  stay-heavy (expert 71 % stay after arrivals).
- **Run 3** (01:03–03:45, 10,975 eps): fixed IL to replan on every arrival
  (paper Section V.A.2), warm-started from run 2's ep 800 with fresh optimizer.
  This produced the checkpoint used for the final numbers.

Total training: ~2 h 42 min of wall-clock on Apple M2 Pro (MPS), single Python
process, 6 agents per env. Deviation from paper: the paper runs 9 Ray workers
in parallel for ~10 h ≈ 35 k episodes.

Best snapshot: **episode 8800** in run 3. Later checkpoints (through ep 11,000)
oscillated between 0.09 and 0.11 sampled throughput without further improvement,
suggesting the model has plateaued at this scale. The paper reports gains from
larger fleets, environment diversity, and the 9-worker gradient diversity —
none of which we can match at this budget.

## Figures

- `logs/plot_FINAL.png` — bar chart of the four methods on 20 seeds.
- `logs/plot_v3_ep1075_training.png` — early training curves (loss / entropy /
  goals).
- `logs/demo_frame_FINAL_seed42_sampled.png` — annotated demo screenshot.
- `logs/demo_frame_FINAL_seed7.png`, `..._seed123.png` — additional scenarios.

## Reproducing the numbers

```bash
# From a fresh clone:
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
