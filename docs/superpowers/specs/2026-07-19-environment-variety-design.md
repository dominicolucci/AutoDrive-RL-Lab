# Environment Variety Expansion — Design

Date: 2026-07-19
Status: approved by user (Approach A: extend existing modules in place)

## Goal

Expand the simulation so the trained agent must stay safe across varied
conditions, not one fixed world. Three new capabilities — traffic density
levels, reactive traffic, and static obstacles — combined through per-episode
domain randomization during training. Robustness ("safe no matter what") is
the explicit target: checkpoint selection and benchmarking move from a single
fixed scenario to a matrix of scenarios.

Out of scope: renderer/UI modernization (separate follow-up design), changes
to the DQN algorithm, curved roads, camera/pixel observations.

## Constraints

- The 16-value observation is unchanged. Obstacles are represented as
  zero-speed traffic cars, so existing sensors report them with no new inputs.
  Old checkpoints must load and drive unmodified.
- Default behavior is identical to today: running existing commands with no
  new flags reproduces current results. All changes are additive.
- Keep the project's single-file-per-concept readability. No new runtime
  dependencies; NumPy only.

## 1. Reactive traffic (`environment.py`)

`TrafficCar` gains `behavior: str = "cruiser"` with values
`"cruiser" | "reactive" | "obstacle"`, plus fields to support animated lane
changes: `target_lane: int` and `lane_change_progress: float` (0..1), and a
`cruise_speed_mps` it tries to return to.

Per step, a reactive car:

- **Time-headway braking:** find the nearest thing ahead in its lane
  (traffic, obstacle, or the ego car). Brake (up to 4.0 m/s^2) when
  gap < 2.0 m + 1.5 s x current speed + closing_speed^2 / (2 x 4.0 m/s^2)
  (the 2.0 m standstill margin prevents creeping into contact as speed
  approaches zero); otherwise
  accelerate up to 2.0 m/s^2 back toward `cruise_speed_mps`. The second
  term is the stopping distance at the current closing speed — a pure
  headway rule brakes too late against stopped obstacles (at 22 m/s it
  would trigger 33 m out but stopping takes ~60 m). Cruisers keep today's
  constant-speed behavior toward other cars, but brake for static obstacles
  using the same threshold (user-approved amendment: without this, cruisers
  drive through obstacles, since traffic-vs-traffic contact is not
  physically resolved).
- **Lane changes:** eligible only when held below 80% of cruise speed. With
  a per-step probability of 0.5%, evaluate adjacent lanes: candidate
  must have a larger front gap than the current lane and safe front/rear
  clearance (rear clearance check includes the ego car). If accepted, begin a
  lane change that animates linearly over 1.2 s. A car mid-change uses its
  interpolated x for both collision detection and sensor lane attribution
  (nearest lane to actual x).
- Collision detection between the ego and traffic switches from
  lane-center x to the car's actual interpolated x. Traffic-vs-traffic
  contact is not resolved physically; headway braking is what prevents it.

## 2. Static obstacles (`environment.py`)

An obstacle is a `TrafficCar` with `speed_mps = 0.0`,
`behavior = "obstacle"`. Sensors need zero new code; relative speed reads as
full closing speed.

Spawning rules:

- Spawn ahead of the ego at episode start; recycle far ahead after being
  passed (same lifecycle as traffic).
- Never spawn within 60 m directly ahead of the ego's starting position.
- Survivability guarantee: within any 30 m longitudinal window, at least one
  lane is free of obstacles, so every layout has an escape path.
- Reactive traffic brakes and lane-changes around obstacles via the same
  headway/lane-change rules.

## 3. Scenario specification and randomization (`config.py`)

New frozen dataclass `ScenarioSpec`:

```
traffic_count: int          obstacle_count: int          reactive_fraction: float
```

and `ScenarioRanges` with `(min, max)` bounds for each field, defaults:
traffic_count (4, 14), obstacle_count (0, 3), reactive_fraction (0.0, 1.0).

Named presets:

| Preset   | traffic | obstacles | reactive |
| -------- | ------: | --------: | -------: |
| sparse   | 4       | 0         | 0.0      |
| normal   | 9       | 0         | 0.0      |
| dense    | 14      | 2         | 0.5      |
| random   | rolled per episode from `ScenarioRanges` |

`normal` matches today's world exactly. Rolls use a dedicated
`np.random.Generator` derived from the run seed (offset stream), so identical
seeds produce identical scenario sequences and the environment's own RNG
stream is undisturbed.

`DrivingEnv` accepts an optional `scenario_spec` and builds its traffic and
obstacles from it; omitted means today's defaults.

## 4. Training integration (`train.py`)

- Warm-up curriculum unchanged: lane-keeping (~15%) then light traffic
  (~35%). After that, in randomization mode, each episode samples a fresh
  `ScenarioSpec` from the ranges.
- CSV gains three columns: `traffic_count`, `obstacle_count`,
  `reactive_fraction`, recording the episode's actual conditions.
- New CLI flags on `train` and `play`:
  - `--scenario-preset {sparse,normal,dense,random}` (train default:
    `random`; play default: `normal`)
  - Overrides: `--traffic N`, `--obstacles N`, `--reactive F` (0.0-1.0);
    any override wins over the preset for that field.
- Existing flags and defaults keep working; `--scenario lane` still exists
  for the pure lane-keeping world.

## 5. Evaluation (`train.py`)

- `evaluate()` accepts a `ScenarioSpec`.
- Periodic training exams run a fixed three-cell matrix: sparse, normal,
  dense. Per-cell return/distance/safe-rate go to the CSV
  (`eval_<cell>_return`, `eval_<cell>_safe_rate`, ...); the existing
  aggregate columns keep their meaning as the mean across cells.
- Best-checkpoint selection uses the mean return across the matrix, so
  "best" means "best across varied worlds".
- The held-out benchmark (100 seeds, 350000-350099) is run per matrix cell
  when re-benchmarking; BENCHMARK.md gains a per-condition table.

## 6. Renderer (minimal touch, `renderer.py`)

- Obstacles drawn as striped hazard blocks, visually distinct from cars.
- Traffic drawn at interpolated x during lane changes.
- No other visual changes; the UI modernization is a separate design.

## 7. Testing (`tests/`)

New tests in the existing unittest style:

- A reactive car approaching a stopped obstacle brakes and does not overlap
  it over a long horizon.
- A lane change never begins into a lane whose front/rear clearance check
  fails (including ego as the rear car).
- Obstacles never move over hundreds of steps.
- Randomized layouts always satisfy the survivability guarantee (property
  test across many seeds).
- Identical run seeds produce identical scenario sequences.
- A checkpoint saved before this change loads and selects actions in the
  new environment (observation contract unchanged).
- The full existing suite (12 tests) passes unmodified.

## Error handling

- Invalid preset names, negative counts, or `reactive_fraction` outside
  [0, 1] raise `ValueError` at argument-parsing/construction time.
- `traffic_count` beyond what spawn spacing can place degrades gracefully:
  the spawn loop already caps attempts; the realized count is what the CSV
  records.

## Success criteria

1. All new and existing tests pass.
2. A domain-randomized training run (1000 episodes, 3 seeds) reaches a
   matrix-mean safe rate on the held-out benchmark at or above the current
   single-world baseline's, and its per-cell spread is reported.
3. Watching `--scenario-preset dense` shows braking waves, occasional lane
   changes, and obstacle avoidance that are visibly plausible.
