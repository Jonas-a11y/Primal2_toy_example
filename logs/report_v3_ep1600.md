# PRIMAL2 Toy — Evaluation Report

**Scenario:** size=15, density=0.3, corridor_length=5, agents=6, steps/scenario=256, seeds=[7, 42, 123, 555, 2024]
**Checkpoint:** `checkpoints/primal2_latest.pt` (episode 1600)

| Method | Throughput (mean) | Throughput (min–max) | Total arrivals | Total steps |
|---|---|---|---|---|
| PRIMAL2 (learned) | 0.0133 | 0.0078 – 0.0156 | 17 | 1280 |
| greedy_astar | 0.0148 | 0.0078 – 0.0312 | 19 | 1280 |
| random | 0.0039 | 0.0000 – 0.0078 | 5 | 1280 |
