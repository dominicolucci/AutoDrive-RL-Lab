# Next Steps

Working notes for picking this project back up. Last updated 2026-07-20.

## 1. In flight: realistic traffic behavior (design paused mid-brainstorm)

**The problem observed:** in the hard presets, traffic does not drive with the
AI driver's safety in mind. Root cause confirmed in code: cars with
`behavior="cruiser"` (100% of traffic in the default world, ~50% in `dense`)
hold constant speed and are blind to every moving vehicle including the ego
car. They brake only for static obstacles. Only `reactive` cars follow safely
and check clearance before merging.

**Paused at this question — pick one to resume:**

- **A. Safe following only** — every car maintains a safe gap and brakes for
  whatever is ahead (cars, obstacles, ego). Smallest change that fixes the
  observed problem.
- **B. Following + safe merging** (was the recommendation) — all cars follow
  safely AND every lane change requires proper front/rear clearance
  (including the ego car), with more natural triggering than today's rare
  0.5%-per-step random roll.
- **C. Full realistic model** — B plus lane discipline (slower traffic keeps
  right), speed-limit adherence, and courtesy yielding to merging vehicles.

**Known decision that comes next, whichever is chosen:** making cruisers
competent changes the DEFAULT world (`DrivingEnv()` with no scenario_spec),
which invalidates the existing BENCHMARK.md numbers and makes old checkpoint
comparisons apples-to-oranges. Options are (i) accept it and re-benchmark,
or (ii) keep a `blind` behavior for the legacy default and make competent
driving the new presets' behavior.

**Relevant code:** `autodrive_rl/environment.py` — `_update_traffic`,
`_update_reactive`, `_update_cruiser`, `_front_gap_for`, `_rear_gap_for`,
`_maybe_start_lane_change`, and the `REACTIVE_*` / `LANE_CHANGE_*` constants.

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
