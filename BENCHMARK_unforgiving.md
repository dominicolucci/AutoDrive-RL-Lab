| Policy | Cell | Mean return | Safe completion | Collision | At fault | Stalled | Off-road | Distance | Mean speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `autodrive_dqn_best` | dense | 275.7 ± 88 | 96% | 0% | 0% | 4% | 0% | 879 m | 9.9 m/s |
| `autodrive_dqn_best` | unforgiving | -2.1 ± 277 | 45% | 16% | 13% | 39% | 0% | 623 m | 9.7 m/s |
| `clone` | dense | -219.2 ± 349 | 32% | 58% | 58% | 9% | 1% | 509 m | 9.1 m/s |
| `clone` | unforgiving | -303.8 ± 241 | 6% | 56% | 53% | 37% | 1% | 380 m | 10.0 m/s |
| `heuristic` | dense | -207.1 ± 223 | 80% | 20% | 20% | 0% | 0% | 1,193 m | 15.0 m/s |
| `heuristic` | unforgiving | -408.9 ± 189 | 39% | 61% | 59% | 0% | 0% | 623 m | 16.9 m/s |
| `random` | dense | -309.4 ± 59 | 0% | 2% | 2% | 93% | 5% | 100 m | 6.4 m/s |
| `random` | unforgiving | -348.0 ± 103 | 0% | 27% | 21% | 70% | 3% | 92 m | 7.0 m/s |

100 held-out episodes per cell, seeds `350000`–`350099`, 900-step limit. Every policy is evaluated on the identical set of worlds, so differences between rows are differences in driving, not in luck.

Difficulty cells: **dense** = 16 cars / 0 obstacles / 50% lane-changers, **unforgiving** = 16 cars / 0 obstacles / 50% lane-changers / traffic from behind / 60% inattentive drivers.

"Safe completion" means the episode reached the step limit without a collision, an off-road event, or a stall. A stall is the ego sitting below the minimum speed in a live lane for longer than the grace period — blocking a motorway lane is a failure, not a safe outcome. "At fault" counts only the collisions the ego caused: running into something ahead, merging into someone, or being struck from behind while stopped. Return is shown as mean ± population standard deviation across episodes.

Regenerate with:

```bash
python -m autodrive_rl.benchmark \
  --policy autodrive_dqn_best \
  --policy clone \
  --policy heuristic \
  --policy random \
  --cells dense unforgiving \
  --episodes 100 --seed-start 350000 --markdown
```
