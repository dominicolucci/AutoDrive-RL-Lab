| Policy | Cell | Mean return | Safe completion | Collision | Off-road | Distance | Mean speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `autodrive_dqn_best` | sparse | 390.0 ± 56 | 100% | 0% | 0% | 1,522 m | 16.9 m/s |
| `autodrive_dqn_best` | normal | 358.4 ± 48 | 100% | 0% | 0% | 1,216 m | 13.5 m/s |
| `autodrive_dqn_best` | dense | 61.0 ± 93 | 100% | 0% | 0% | 269 m | 3.0 m/s |
| `clone` | sparse | -415.3 ± 219 | 6% | 94% | 0% | 455 m | 22.6 m/s |
| `clone` | normal | -318.9 ± 369 | 24% | 76% | 0% | 497 m | 16.0 m/s |
| `clone` | dense | -482.7 ± 110 | 2% | 97% | 1% | 230 m | 10.2 m/s |
| `heuristic` | sparse | 480.5 ± 153 | 99% | 1% | 0% | 2,341 m | 26.2 m/s |
| `heuristic` | normal | 132.6 ± 284 | 89% | 11% | 0% | 1,906 m | 22.4 m/s |
| `heuristic` | dense | -508.2 ± 178 | 12% | 88% | 0% | 572 m | 17.8 m/s |
| `random` | sparse | -401.3 ± 157 | 38% | 0% | 62% | 115 m | 2.4 m/s |
| `random` | normal | -380.6 ± 176 | 43% | 0% | 57% | 122 m | 2.5 m/s |
| `random` | dense | -541.7 ± 149 | 16% | 57% | 27% | 92 m | 4.9 m/s |

100 held-out episodes per cell, seeds `350000`–`350099`, 900-step limit. Every policy is evaluated on the identical set of worlds, so differences between rows are differences in driving, not in luck.

Difficulty cells: **sparse** = 4 cars / 0 obstacles / 0% lane-changers, **normal** = 9 cars / 0 obstacles / 0% lane-changers, **dense** = 14 cars / 2 obstacles / 50% lane-changers.

"Safe completion" means the episode reached the step limit without a collision or off-road event. Return is shown as mean ± population standard deviation across episodes.

Regenerate with:

```bash
python -m autodrive_rl.benchmark \
  --policy autodrive_dqn_best \
  --policy clone \
  --policy heuristic \
  --policy random \
  --episodes 100 --seed-start 350000 --markdown
```
