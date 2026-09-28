| Policy | Cell | Mean return | Safe completion | Collision | Off-road | Distance | Mean speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `autodrive_dqn_best` | dense | 61.0 ± 93 | 100% | 0% | 0% | 269 m | 3.0 m/s |
| `autodrive_dqn_best` | unforgiving | -43.0 ± 236 | 88% | 12% | 0% | 272 m | 3.8 m/s |
| `clone` | dense | -482.7 ± 110 | 2% | 97% | 1% | 230 m | 10.2 m/s |
| `clone` | unforgiving | -484.3 ± 95 | 2% | 98% | 0% | 207 m | 12.2 m/s |
| `heuristic` | dense | -508.2 ± 178 | 12% | 88% | 0% | 572 m | 17.8 m/s |
| `heuristic` | unforgiving | -581.1 ± 82 | 2% | 98% | 0% | 431 m | 19.3 m/s |
| `random` | dense | -564.8 ± 193 | 13% | 58% | 29% | 96 m | 5.3 m/s |
| `random` | unforgiving | -579.5 ± 103 | 3% | 80% | 17% | 92 m | 5.6 m/s |

100 held-out episodes per cell, seeds `350000`–`350099`, 900-step limit. Every policy is evaluated on the identical set of worlds, so differences between rows are differences in driving, not in luck.

Difficulty cells: **dense** = 14 cars / 2 obstacles / 50% lane-changers, **unforgiving** = 14 cars / 2 obstacles / 50% lane-changers / traffic from behind / 40% inattentive drivers.

"Safe completion" means the episode reached the step limit without a collision or off-road event. Return is shown as mean ± population standard deviation across episodes.

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
