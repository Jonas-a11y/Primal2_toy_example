# PRIMAL2 Toy — Evaluation Report

**Scenario:** size=15, density=0.3, corridor_length=5, agents=6, steps/scenario=256, seeds=[7, 42, 123, 555, 2024, 8, 91, 314, 777, 1000]
**Checkpoint:** `checkpoints/primal2_latest.pt` (episode 5600)

| Method | Throughput (mean) | Throughput (min–max) | Total arrivals | Total steps |
|---|---|---|---|---|
| PRIMAL2 (learned) | 0.0398 | 0.0117 – 0.1016 | 102 | 2560 |
| PRIMAL2 (sampled) | 0.0867 | 0.0234 – 0.1562 | 222 | 2560 |
| greedy_astar | 0.0121 | 0.0000 – 0.0312 | 31 | 2560 |
| random | 0.0051 | 0.0000 – 0.0195 | 13 | 2560 |
