# PRIMAL2 Toy — Evaluation Report

**Scenario:** size=15, density=0.3, corridor_length=5, agents=6, steps/scenario=256, seeds=[7, 42, 123, 555, 2024, 8, 91, 314, 777, 1000, 33, 66, 200, 400, 800, 1234, 5678, 9101, 2222, 3333]
**Checkpoint:** `checkpoints/primal2_latest.pt` (episode 2400)

| Method | Throughput (mean) | Throughput (min–max) | Total arrivals | Total steps |
|---|---|---|---|---|
| PRIMAL2 (learned) | 0.0521 | 0.0117 – 0.1211 | 267 | 5120 |
| PRIMAL2 (sampled) | 0.1324 | 0.0703 – 0.2266 | 678 | 5120 |
| greedy_astar | 0.0187 | 0.0000 – 0.0625 | 96 | 5120 |
| random | 0.0059 | 0.0000 – 0.0195 | 30 | 5120 |
