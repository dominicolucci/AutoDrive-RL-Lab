# Next Steps

Working notes for picking this project back up. Last updated 2026-07-20.

## 1. ~~Realistic traffic behavior~~ — DONE (2026-07-23)

Implemented, roughly option B plus the speed-limit part of C:

- Every moving car (cruiser and reactive) now keeps a physics-based safe
  following distance from whatever is ahead — cars, obstacles, and the ego —
  via a shared `_follow_safely`, with an emergency-braking rate
  (`REACTIVE_EMERGENCY_BRAKE_MPS2`) when comfortable braking cannot shed the
  closing speed in the available gap.
- A posted speed limit (`EnvConfig.speed_limit_mps`, 25 m/s) that no traffic
  car ever exceeds; over-limit cars settle back down to it.
- Lane changes reject rear gaps with a fast-closing follower (closing-speed
  aware `_lane_change_ok`), so nobody cuts off the ego or another car.
- Recycled cars re-enter traffic like real merging drivers: only into a gap
  with room (`_spawn_gap_ok`), at a speed the flow ahead allows
  (`_entry_speed`); if no safe gap exists they wait out of range.
- Cruisers still never change lanes; `reactive_fraction` now means "fraction
  of drivers who will change lanes when blocked".

Regression tests cover: speed-limit adherence over rollouts, no
traffic-vs-traffic rear-ends, no rear-ending of the ego, cut-off rejection,
and emergency braking. Chosen benchmark option: **(i) accept and
re-benchmark** — BENCHMARK.md is marked historical until the robustness run
below replaces it.

**Follow-up (2026-07-23): hard no-overlap guarantee.** Traffic cars can no
longer occupy the same physical space, ever:

- Mid-lane-change cars count as occupying BOTH lanes (`_occupied_lanes`) in
  every gap check — following, merging, spawning, and obstacle placement.
- Lane changes abort and smoothly reverse (`_resolve_lane_change_conflicts`,
  `_abort_lane_change`) when the target gap collapses mid-animation or when
  two drivers merge toward the same spot (the less-committed one yields).
  Past 50% progress a change is committed and others yield instead.
- Recycled obstacles now also check moving cars — an obstacle never appears
  on top of a car or inside a car's emergency stopping distance.

Pinned by `test_no_two_cars_ever_overlap_geometrically` (rectangle-overlap
invariant over rollouts with a random ego) plus merge-conflict and abort
tests.

Not yet done from option C: lane discipline (slower traffic keeps right) and
courtesy yielding to merging vehicles.

## 2. Queued experiment: the robustness run

Never run yet. Existing checkpoints have never seen an obstacle or a
lane-changing car, so they fail the `dense` preset.

```bash
python -m autodrive_rl.train --episodes 1000 --seed 1 --output models/robust_s1.npz --metrics runs/robust_s1.csv
```

Run 3-5 seeds in parallel terminals (16 cores available, ~15 min wall clock
for all of them), then compare on held-out worlds per difficulty cell and
update BENCHMARK.md with a per-condition table. Watch progress live with
`mlflow ui`.

## 3. Other experiments available with current code

- Ablations: `--no-curriculum`; `--scenario-preset normal` vs `random`
  (robustness vs specialization); `hidden_sizes` `(32,32)` vs `(64,64)` vs
  `(128,128)` in `config.py`.
- Reward surgery: raise the speed coefficient (0.20) in `_reward`
  (`environment.py`) and retrain for an aggressive driver personality.
- Deliberate breakage for insight: `gamma=0.5` (short-sighted),
  `epsilon_end=0.5` (never stops exploring).

## 4. Learning roadmap toward production-grade ML

Ordered; step 1 is done.

1. ~~Experiment tracking (MLflow)~~ — done, merged.
2. **Config-driven runs** — hyperparameters move from code/CLI into
   versioned YAML config files. Teaches reproducibility discipline.
3. **PyTorch port** — reimplement the same DQN in the industry framework and
   verify parity with the NumPy version. Highest employability value, and a
   prerequisite for anything CNN/transformer.
4. **Warm starts / fine-tuning (`--init-from`)** — load a trained checkpoint
   and continue training (lower starting epsilon, no curriculum). Unlocks
   the transfer-learning experiment: fine-tuned vs cold-start race.
5. **Hyperparameter search** — Optuna or a grid script over the eval matrix.
6. **Serving** — FastAPI endpoint: POST 16 sensor values, return an action.
7. **Monitoring** — continuous safety metrics with alerting on degradation.
8. **CI** — GitHub Actions running the test suite plus a smoke train.

## 5. Longer-term architecture exploration

Requires the PyTorch port first.

- Noisy/dropout sensors before any pixel work (README expansion step 6).
- Small transformer for trajectory prediction of neighboring cars
  (supervised, labels free from the sim, CPU-friendly) — best first
  modern-architecture project here.
- Rendered bird's-eye image + CNN trained supervised to recover the 16
  sensor values — builds a perception layer without touching the RL loop.
- Only then: end-to-end pixels-to-control (needs a GPU and patience).

## 6. Imitation learning side quest

Record `(observation, action)` pairs while driving manually, train the same
16→64→64→5 network with a supervised loss to clone the human driver, then
race the clone against the DQN on held-out worlds. Adding a correction loop
afterward turns it into DAgger. Completes the paradigm map (supervised vs
reinforcement) with working code.