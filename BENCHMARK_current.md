| Policy | Cell | Mean return | Safe completion | Collision | At fault | Stalled | Off-road | Distance | Mean speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `autodrive_dqn_best` | sparse | 390.0 ± 56 | 100% | 0% | 0% | 0% | 0% | 1,522 m | 16.9 m/s |
| `autodrive_dqn_best` | normal | 358.4 ± 48 | 100% | 0% | 0% | 0% | 0% | 1,216 m | 13.5 m/s |
| `autodrive_dqn_best` | dense | 275.7 ± 88 | 96% | 0% | 0% | 4% | 0% | 879 m | 9.9 m/s |
| `clone` | sparse | -415.3 ± 219 | 6% | 94% | 94% | 0% | 0% | 455 m | 22.6 m/s |
| `clone` | normal | -317.2 ± 369 | 23% | 76% | 76% | 1% | 0% | 490 m | 16.0 m/s |
| `clone` | dense | -219.2 ± 349 | 32% | 58% | 58% | 9% | 1% | 509 m | 9.1 m/s |
| `heuristic` | sparse | 480.5 ± 153 | 99% | 1% | 1% | 0% | 0% | 2,341 m | 26.2 m/s |
| `heuristic` | normal | 132.6 ± 284 | 89% | 11% | 11% | 0% | 0% | 1,906 m | 22.4 m/s |
| `heuristic` | dense | -207.1 ± 223 | 80% | 20% | 20% | 0% | 0% | 1,193 m | 15.0 m/s |
| `random` | sparse | -299.9 ± 32 | 0% | 0% | 0% | 95% | 5% | 102 m | 6.4 m/s |
| `random` | normal | -299.9 ± 32 | 0% | 0% | 0% | 95% | 5% | 102 m | 6.4 m/s |
| `random` | dense | -309.4 ± 59 | 0% | 2% | 2% | 93% | 5% | 100 m | 6.4 m/s |

100 held-out episodes per cell, seeds `350000`–`350099`, 900-step limit. Every policy is evaluated on the identical set of worlds, so differences between rows are differences in driving, not in luck.

Difficulty cells: **sparse** = 4 cars / 0 obstacles / 0% lane-changers, **normal** = 9 cars / 0 obstacles / 0% lane-changers, **dense** = 16 cars / 0 obstacles / 50% lane-changers.

"Safe completion" means the episode reached the step limit without a collision, an off-road event, or a stall. A stall is the ego sitting below the minimum speed in a live lane for longer than the grace period — blocking a motorway lane is a failure, not a safe outcome. "At fault" counts only the collisions the ego caused: running into something ahead, merging into someone, or being struck from behind while stopped. Return is shown as mean ± population standard deviation across episodes.

Regenerate with:

```bash
python -m autodrive_rl.benchmark \
  --policy autodrive_dqn_best \
  --policy clone \
  --policy heuristic \
  --policy random \
  --episodes 100 --seed-start 350000 --markdown
```
