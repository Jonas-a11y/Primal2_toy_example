# PRIMAL2 Toy — Evaluation Report

**Scenario:** size=15, density=0.3, corridor_length=5, agents=6, steps/scenario=128, seeds=[7, 42, 123]
**Checkpoint:** `checkpoints/primal2_latest.pt` (episode 400)

| Method | Throughput (mean) | Throughput (min–max) | Total arrivals | Total steps |
|---|---|---|---|---|
| PRIMAL2 (learned) | 0.0208 | 0.0156 – 0.0234 | 8 | 384 |
| greedy_astar | 0.0234 | 0.0156 – 0.0312 | 9 | 384 |
| random | 0.0026 | 0.0000 – 0.0078 | 1 | 384 |
