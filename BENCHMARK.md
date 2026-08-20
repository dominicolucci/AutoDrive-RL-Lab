# Included Training Benchmark

> **These numbers are historical.** They were measured before the traffic model
> gained physics-based following, speed limits, gap-aware merging and obstacles,
> so they describe an environment this repository no longer simulates. They are
> kept for continuity, not as a current claim.
>
> Regenerate current, reproducible results for any policy with:
>
> ```bash
> python -m autodrive_rl.benchmark \
>   --policy models/autodrive_dqn_best.npz \
>   --policy models/clone.npz \
>   --policy heuristic \
>   --policy random \
>   --episodes 100 --markdown
> ```
>
> That reports every policy on identical held-out worlds, broken out by
> difficulty cell, with collision and off-road rates alongside the safe-completion
> rate — the table below reports safe completion alone, which hides *how* a
> policy stays safe.

The included checkpoints were produced by a 400-episode curriculum run:

```bash
python -m autodrive_rl.train \
  --episodes 400 \
  --max-steps 900 \
  --eval-every 50 \
  --eval-episodes 10
```

The curriculum used lane keeping, light traffic, and then the full nine-car
traffic environment. The comparison below uses 100 held-out environment seeds
(`350000` through `350099`) that were not used for training.

| Policy | Mean return | Mean distance | Safe completion |
| --- | ---: | ---: | ---: |
| Best Double DQN checkpoint | 551.1 | 1,201.2 m | 100% |
| Final Double DQN checkpoint | 468.5 | 1,140.7 m | 92% |
| Random actions | -343.0 | 127.8 m | 35% |
| Rule-based heuristic | 521.0 | 1,882.0 m | 99% |

"Safe completion" means the episode reached its 900-step time limit without a
collision or off-road event. The random policy's 35% rate is not competent
driving: its very low distance shows that many of those episodes survived by
slowing dramatically or making little progress.

## What the result means

The DQN clearly learned beyond random behavior. The best checkpoint safely
completed every held-out episode, but traveled less far than the heuristic. It
learned a conservative policy that prioritizes the current safety penalties
over maximum throughput.

That is a useful first result rather than a reason to hide the benchmark. The
next controlled experiment should vary the progress/speed reward while keeping
the safety specification fixed, then compare:

- safe completion rate
- distance traveled
- collision rate
- minimum time-to-collision
- mean speed

The best checkpoint is `models/autodrive_dqn_best.npz`. The complete per-episode
history is `runs/training_metrics.csv`.

