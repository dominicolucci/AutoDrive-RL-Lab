# Environment Variety Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add traffic density presets, reactive traffic (headway braking + animated lane changes), and static obstacles to the driving environment, combined through per-episode domain randomization in training and matrix-based evaluation.

**Architecture:** Extend the existing modules in place (Approach A from the spec). Scenario configuration lives in `config.py` (`ScenarioSpec`, presets, sampling). `environment.py` gains obstacle spawning, reactive-driver rules, and interpolated lateral positions for lane-changing traffic. `train.py` samples a scenario per episode after the warm-up curriculum and evaluates on a fixed sparse/normal/dense matrix. Spec: `docs/superpowers/specs/2026-07-19-environment-variety-design.md`.

**Tech Stack:** Python 3.10+, NumPy only, stdlib `unittest`.

## Global Constraints

- No new runtime dependencies; NumPy only.
- The 16-value observation is unchanged; old checkpoints must load and drive.
- Defaults preserve today's behavior: `DrivingEnv()` with no `scenario_spec` behaves exactly as before; existing CLI commands keep working.
- All 12 existing tests must keep passing unmodified.
- Test style: stdlib `unittest`, classes like `class XTests(unittest.TestCase)`, run via `python -m unittest`.
- Behavior constants (headway 1.5 s, brake 4.0 m/s², accel 2.0 m/s², lane-change 1.2 s / 0.5% per step / 15 m front / 12 m rear, obstacle window 30 m, ego clear 60 m) are module-level constants in `environment.py`.
- Run all commands from the repo root: `c:\Users\domin\Downloads\AutoDrive-RL-Lab`.

---

### Task 1: ScenarioSpec, presets, and sampling

**Files:**
- Modify: `autodrive_rl/config.py`
- Test: `tests/test_scenarios.py` (create)

**Interfaces:**
- Consumes: `EnvConfig` (existing).
- Produces (used by every later task):
  - `ScenarioSpec(traffic_count: int = 9, obstacle_count: int = 0, reactive_fraction: float = 0.0)` — frozen dataclass, validates in `__post_init__`.
  - `ScenarioRanges(traffic_count=(4, 14), obstacle_count=(0, 3), reactive_fraction=(0.0, 1.0))` — frozen dataclass of inclusive bounds.
  - `SCENARIO_PRESETS: dict[str, ScenarioSpec]` with keys `"sparse"`, `"normal"`, `"dense"`.
  - `sample_scenario(ranges: ScenarioRanges, rng: np.random.Generator) -> ScenarioSpec`
  - `resolve_scenario(preset: str, *, traffic: int | None = None, obstacles: int | None = None, reactive: float | None = None, rng: np.random.Generator | None = None, ranges: ScenarioRanges | None = None) -> ScenarioSpec` — accepts preset names plus `"random"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scenarios.py`:

```python
from __future__ import annotations

import unittest

import numpy as np

from autodrive_rl.config import (
    SCENARIO_PRESETS,
    EnvConfig,
    ScenarioRanges,
    ScenarioSpec,
    resolve_scenario,
    sample_scenario,
)


class ScenarioSpecTests(unittest.TestCase):
    def test_presets_exist_and_normal_matches_default_world(self) -> None:
        self.assertEqual(set(SCENARIO_PRESETS), {"sparse", "normal", "dense"})
        normal = SCENARIO_PRESETS["normal"]
        self.assertEqual(normal.traffic_count, EnvConfig().traffic_count)
        self.assertEqual(normal.obstacle_count, 0)
        self.assertEqual(normal.reactive_fraction, 0.0)
        dense = SCENARIO_PRESETS["dense"]
        self.assertEqual((dense.traffic_count, dense.obstacle_count), (14, 2))
        self.assertEqual(dense.reactive_fraction, 0.5)

    def test_invalid_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            ScenarioSpec(traffic_count=-1)
        with self.assertRaises(ValueError):
            ScenarioSpec(obstacle_count=-1)
        with self.assertRaises(ValueError):
            ScenarioSpec(reactive_fraction=1.5)
        with self.assertRaises(ValueError):
            resolve_scenario("nope")

    def test_sampling_is_reproducible_and_in_bounds(self) -> None:
        ranges = ScenarioRanges()
        first = [sample_scenario(ranges, np.random.default_rng(5)) for _ in range(1)]
        second = [sample_scenario(ranges, np.random.default_rng(5)) for _ in range(1)]
        self.assertEqual(first, second)
        rng = np.random.default_rng(9)
        for _ in range(50):
            spec = sample_scenario(ranges, rng)
            self.assertTrue(4 <= spec.traffic_count <= 14)
            self.assertTrue(0 <= spec.obstacle_count <= 3)
            self.assertTrue(0.0 <= spec.reactive_fraction <= 1.0)

    def test_resolve_applies_overrides(self) -> None:
        spec = resolve_scenario("sparse", obstacles=2, reactive=0.25)
        self.assertEqual(spec.traffic_count, 4)
        self.assertEqual(spec.obstacle_count, 2)
        self.assertEqual(spec.reactive_fraction, 0.25)

    def test_random_preset_requires_rng(self) -> None:
        with self.assertRaises(ValueError):
            resolve_scenario("random")
        spec = resolve_scenario("random", rng=np.random.default_rng(3))
        self.assertTrue(4 <= spec.traffic_count <= 14)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_scenarios -v`
Expected: FAIL — `ImportError: cannot import name 'ScenarioSpec'`.

- [ ] **Step 3: Implement in `autodrive_rl/config.py`**

Append after the existing `DQNConfig` (and add `import numpy as np` at the top of the file below `from dataclasses import dataclass`):

```python
@dataclass(frozen=True)
class ScenarioSpec:
    """Concrete per-episode world conditions."""

    traffic_count: int = 9
    obstacle_count: int = 0
    reactive_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.traffic_count < 0:
            raise ValueError("traffic_count must be >= 0")
        if self.obstacle_count < 0:
            raise ValueError("obstacle_count must be >= 0")
        if not 0.0 <= self.reactive_fraction <= 1.0:
            raise ValueError("reactive_fraction must be in [0, 1]")


@dataclass(frozen=True)
class ScenarioRanges:
    """Inclusive sampling bounds for domain randomization."""

    traffic_count: tuple[int, int] = (4, 14)
    obstacle_count: tuple[int, int] = (0, 3)
    reactive_fraction: tuple[float, float] = (0.0, 1.0)


SCENARIO_PRESETS: dict[str, ScenarioSpec] = {
    "sparse": ScenarioSpec(traffic_count=4, obstacle_count=0, reactive_fraction=0.0),
    "normal": ScenarioSpec(traffic_count=9, obstacle_count=0, reactive_fraction=0.0),
    "dense": ScenarioSpec(traffic_count=14, obstacle_count=2, reactive_fraction=0.5),
}


def sample_scenario(ranges: ScenarioRanges, rng: np.random.Generator) -> ScenarioSpec:
    """Roll one episode's conditions from the given inclusive bounds."""

    return ScenarioSpec(
        traffic_count=int(
            rng.integers(ranges.traffic_count[0], ranges.traffic_count[1] + 1)
        ),
        obstacle_count=int(
            rng.integers(ranges.obstacle_count[0], ranges.obstacle_count[1] + 1)
        ),
        reactive_fraction=float(rng.uniform(*ranges.reactive_fraction)),
    )


def resolve_scenario(
    preset: str,
    *,
    traffic: int | None = None,
    obstacles: int | None = None,
    reactive: float | None = None,
    rng: np.random.Generator | None = None,
    ranges: ScenarioRanges | None = None,
) -> ScenarioSpec:
    """Turn a preset name plus optional overrides into a concrete spec."""

    if preset == "random":
        if rng is None:
            raise ValueError("preset 'random' requires an rng")
        base = sample_scenario(ranges or ScenarioRanges(), rng)
    elif preset in SCENARIO_PRESETS:
        base = SCENARIO_PRESETS[preset]
    else:
        raise ValueError(f"unknown scenario preset: {preset!r}")
    return ScenarioSpec(
        traffic_count=base.traffic_count if traffic is None else traffic,
        obstacle_count=base.obstacle_count if obstacles is None else obstacles,
        reactive_fraction=base.reactive_fraction if reactive is None else reactive,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_scenarios -v`
Expected: 5 tests PASS.

- [ ] **Step 5: Run the full suite, then commit**

Run: `python -m unittest discover -s tests`
Expected: 17 tests pass.

```bash
git add autodrive_rl/config.py tests/test_scenarios.py
git commit -m "feat: scenario specs, presets, and randomized sampling"
```

---

### Task 2: Environment accepts a scenario spec; obstacle spawning with survivability

**Files:**
- Modify: `autodrive_rl/environment.py`
- Test: `tests/test_scenarios.py` (append)

**Interfaces:**
- Consumes: `ScenarioSpec` from Task 1.
- Produces:
  - `DrivingEnv(config=None, *, scenario="traffic", seed=None, scenario_spec: ScenarioSpec | None = None)` — new keyword arg, stored as `self.scenario_spec`.
  - `TrafficCar` gains fields `behavior: str = "cruiser"` (`"cruiser" | "reactive" | "obstacle"`), `cruise_speed_mps: float | None = None` (defaults to `speed_mps` in `__post_init__`), `target_lane: int | None = None`, `lane_change_progress: float = 0.0`.
  - Module constants `OBSTACLE_WINDOW_M = 30.0`, `OBSTACLE_EGO_CLEAR_M = 60.0`.
  - Internal: `_spawn_obstacles()`, `_obstacle_position_ok(lane: int, y_m: float) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scenarios.py` (add these imports at the top: `from dataclasses import replace` and `from autodrive_rl.environment import DrivingEnv, TrafficCar`):

```python
class ObstacleSpawnTests(unittest.TestCase):
    def make_env(self, seed: int, spec: ScenarioSpec) -> DrivingEnv:
        config = replace(EnvConfig(), max_steps=50)
        return DrivingEnv(config, scenario="traffic", seed=seed, scenario_spec=spec)

    def test_spec_controls_counts(self) -> None:
        env = self.make_env(3, ScenarioSpec(traffic_count=5, obstacle_count=2))
        moving = [car for car in env.traffic if car.behavior != "obstacle"]
        obstacles = [car for car in env.traffic if car.behavior == "obstacle"]
        self.assertEqual(len(moving), 5)
        self.assertEqual(len(obstacles), 2)
        for obstacle in obstacles:
            self.assertEqual(obstacle.speed_mps, 0.0)

    def test_reactive_fraction_one_marks_all_moving_cars(self) -> None:
        env = self.make_env(4, ScenarioSpec(traffic_count=6, reactive_fraction=1.0))
        moving = [car for car in env.traffic if car.behavior != "obstacle"]
        self.assertTrue(all(car.behavior == "reactive" for car in moving))

    def test_no_obstacle_close_ahead_in_ego_start_lane(self) -> None:
        for seed in range(30):
            env = self.make_env(seed, ScenarioSpec(traffic_count=4, obstacle_count=3))
            for car in env.traffic:
                if car.behavior == "obstacle" and car.lane == env.current_lane:
                    self.assertGreaterEqual(car.y_m, 60.0)

    def test_layouts_always_leave_an_open_lane(self) -> None:
        for seed in range(50):
            env = self.make_env(seed, ScenarioSpec(traffic_count=6, obstacle_count=3))
            obstacles = [car for car in env.traffic if car.behavior == "obstacle"]
            for obstacle in obstacles:
                near_lanes = {
                    other.lane
                    for other in obstacles
                    if abs(other.y_m - obstacle.y_m) < 30.0
                }
                self.assertLess(len(near_lanes), env.config.lane_count)

    def test_default_env_unchanged(self) -> None:
        env = DrivingEnv(replace(EnvConfig(), max_steps=50), seed=1)
        self.assertEqual(len(env.traffic), env.config.traffic_count)
        self.assertTrue(all(car.behavior == "cruiser" for car in env.traffic))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_scenarios -v`
Expected: FAIL — `TypeError: DrivingEnv.__init__() got an unexpected keyword argument 'scenario_spec'`.

- [ ] **Step 3: Implement in `autodrive_rl/environment.py`**

Add the import (`from .config import EnvConfig` becomes `from .config import EnvConfig, ScenarioSpec`) and module constants near `ACTION_NAMES`:

```python
OBSTACLE_WINDOW_M = 30.0
OBSTACLE_EGO_CLEAR_M = 60.0
```

Replace the `TrafficCar` dataclass:

```python
@dataclass
class TrafficCar:
    """A traffic car (or static obstacle) represented relative to the ego car."""

    lane: int
    y_m: float
    speed_mps: float
    color_index: int = 0
    behavior: str = "cruiser"  # "cruiser" | "reactive" | "obstacle"
    cruise_speed_mps: float | None = None
    target_lane: int | None = None
    lane_change_progress: float = 0.0

    def __post_init__(self) -> None:
        if self.cruise_speed_mps is None:
            self.cruise_speed_mps = self.speed_mps
```

Extend `__init__` signature and store the spec (before `self.reset(seed=seed)`):

```python
    def __init__(
        self,
        config: EnvConfig | None = None,
        *,
        scenario: str = "traffic",
        seed: int | None = None,
        scenario_spec: ScenarioSpec | None = None,
    ) -> None:
        ...
        self.scenario_spec = scenario_spec
```

Replace `_spawn_initial_traffic` and add the two new methods:

```python
    def _spawn_initial_traffic(self) -> None:
        cfg = self.config
        spec = self.scenario_spec
        count = cfg.traffic_count if spec is None else spec.traffic_count
        reactive_fraction = 0.0 if spec is None else spec.reactive_fraction
        attempts = 0
        while len(self.traffic) < count and attempts < 500:
            attempts += 1
            lane = int(self.rng.integers(0, cfg.lane_count))
            y_m = float(self.rng.uniform(22.0, cfg.sensor_range_m + 55.0))
            if any(car.lane == lane and abs(car.y_m - y_m) < 24.0 for car in self.traffic):
                continue
            speed = float(self.rng.uniform(9.0, 26.0))
            color_index = int(self.rng.integers(0, 6))
            behavior = "cruiser"
            if reactive_fraction > 0.0 and self.rng.random() < reactive_fraction:
                behavior = "reactive"
            self.traffic.append(TrafficCar(lane, y_m, speed, color_index, behavior=behavior))
        self._spawn_obstacles()

    def _spawn_obstacles(self) -> None:
        spec = self.scenario_spec
        if spec is None or spec.obstacle_count == 0:
            return
        cfg = self.config
        ego_lane = self.current_lane
        placed = 0
        attempts = 0
        while placed < spec.obstacle_count and attempts < 200:
            attempts += 1
            lane = int(self.rng.integers(0, cfg.lane_count))
            y_m = float(self.rng.uniform(30.0, cfg.sensor_range_m + 55.0))
            if lane == ego_lane and y_m < OBSTACLE_EGO_CLEAR_M:
                continue
            if not self._obstacle_position_ok(lane, y_m):
                continue
            self.traffic.append(TrafficCar(lane, y_m, 0.0, behavior="obstacle"))
            placed += 1

    def _obstacle_position_ok(self, lane: int, y_m: float) -> bool:
        cfg = self.config
        blocked = {lane}
        for car in self.traffic:
            if car.lane == lane and abs(car.y_m - y_m) < 24.0:
                return False
            if car.behavior == "obstacle" and abs(car.y_m - y_m) < OBSTACLE_WINDOW_M:
                blocked.add(car.lane)
        return len(blocked) < cfg.lane_count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_scenarios -v`
Expected: 10 tests PASS.

- [ ] **Step 5: Run the full suite, then commit**

Run: `python -m unittest discover -s tests`
Expected: 22 tests pass (existing 12 untouched).

```bash
git add autodrive_rl/environment.py tests/test_scenarios.py
git commit -m "feat: scenario-driven spawning and survivable static obstacles"
```

---

### Task 3: Actual lateral position for traffic (interpolated x)

**Files:**
- Modify: `autodrive_rl/environment.py`
- Test: `tests/test_scenarios.py` (append)

**Interfaces:**
- Consumes: `TrafficCar.target_lane` / `lane_change_progress` from Task 2.
- Produces (used by Tasks 4, 5, and the renderer task):
  - `DrivingEnv.traffic_x_m(car: TrafficCar) -> float` — actual lateral position in meters, lane-change aware.
  - `DrivingEnv.traffic_lane(car: TrafficCar) -> int` — nearest lane to actual x.
  - `sensor_snapshot()` and `_has_collision()` use these instead of `car.lane` / lane-center x.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scenarios.py`:

```python
class LaneAttributionTests(unittest.TestCase):
    def make_env(self) -> DrivingEnv:
        config = replace(EnvConfig(), max_steps=50)
        return DrivingEnv(config, scenario="lane", seed=1)

    def test_traffic_x_interpolates_between_lane_centers(self) -> None:
        env = self.make_env()
        car = TrafficCar(0, 50.0, 20.0, target_lane=1, lane_change_progress=0.5)
        expected = (env.lane_center(0) + env.lane_center(1)) / 2.0
        self.assertAlmostEqual(env.traffic_x_m(car), expected)

    def test_mid_change_car_attributed_to_nearest_lane(self) -> None:
        env = self.make_env()
        car = TrafficCar(0, 50.0, 20.0, target_lane=1, lane_change_progress=0.1)
        self.assertEqual(env.traffic_lane(car), 0)
        car.lane_change_progress = 0.9
        self.assertEqual(env.traffic_lane(car), 1)

    def test_collision_uses_actual_x(self) -> None:
        env = self.make_env()
        ego_lane = env.current_lane
        car = TrafficCar(ego_lane - 1, 0.0, 12.0, target_lane=ego_lane)
        car.lane_change_progress = 0.0
        env.traffic = [car]
        self.assertFalse(env._has_collision())
        car.lane_change_progress = 0.95
        self.assertTrue(env._has_collision())

    def test_sensors_see_mid_change_car_in_target_lane(self) -> None:
        env = self.make_env()
        ego_lane = env.current_lane
        car = TrafficCar(ego_lane - 1, 40.0, 12.0, target_lane=ego_lane)
        car.lane_change_progress = 0.9
        env.traffic = [car]
        sensors = env.sensor_snapshot()
        front = np.asarray(sensors["front_gaps_m"])
        self.assertLess(front[ego_lane], env.config.sensor_range_m)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_scenarios -v`
Expected: FAIL — `AttributeError: 'DrivingEnv' object has no attribute 'traffic_x_m'`.

- [ ] **Step 3: Implement in `autodrive_rl/environment.py`**

Add two public methods (near `lane_center`):

```python
    def traffic_x_m(self, car: TrafficCar) -> float:
        """The car's actual lateral position, mid-lane-change aware."""

        x = self.lane_center(car.lane)
        if car.target_lane is not None:
            target = self.lane_center(car.target_lane)
            x += (target - x) * car.lane_change_progress
        return x

    def traffic_lane(self, car: TrafficCar) -> int:
        """The lane whose center is nearest the car's actual position."""

        return int(np.argmin(np.abs(self.lane_centers_m - self.traffic_x_m(car))))
```

In `sensor_snapshot`, replace every `car.lane` read inside the traffic loop with a local `lane = self.traffic_lane(car)` computed at the top of the loop (the gap arrays are indexed by that `lane`). In `_has_collision`, replace `car_x = self.lane_center(car.lane)` with `car_x = self.traffic_x_m(car)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_scenarios -v`
Expected: 14 tests PASS.

- [ ] **Step 5: Run the full suite, then commit**

Run: `python -m unittest discover -s tests`
Expected: 26 tests pass.

```bash
git add autodrive_rl/environment.py tests/test_scenarios.py
git commit -m "feat: lane-change-aware lateral positions for sensors and collision"
```

---

### Task 4: Reactive braking (time-headway rule)

**Files:**
- Modify: `autodrive_rl/environment.py`
- Test: `tests/test_scenarios.py` (append)

**Interfaces:**
- Consumes: `traffic_lane()` from Task 3, `behavior` field from Task 2.
- Produces (Task 5 extends `_update_reactive`):
  - Module constants `REACTIVE_TIME_HEADWAY_S = 1.5`, `REACTIVE_BRAKE_MPS2 = 4.0`, `REACTIVE_ACCEL_MPS2 = 2.0`.
  - `_update_traffic()` — called once per `step()` before traffic positions advance.
  - `_update_reactive(car: TrafficCar)` — brakes when `gap < 1.5 s x speed + closing_speed^2 / (2 x 4.0)`. The second term is the stopping distance at the current closing speed — without it, a car approaching a *stopped* obstacle would start braking far too late (at 22 m/s a pure headway rule triggers 33 m out but stopping takes ~60 m).
  - `_front_gap_for(car: TrafficCar, lane: int) -> tuple[float, float]` — `(gap, leader_speed)` for the nearest leader (traffic, obstacle, or ego) in a lane; gap floored at 0, `(inf, 0.0)` when the lane is empty ahead.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scenarios.py` (add `from autodrive_rl.environment import Action` to the imports):

```python
class ReactiveBrakingTests(unittest.TestCase):
    def make_env(self, max_steps: int = 400) -> DrivingEnv:
        config = replace(EnvConfig(), max_steps=max_steps)
        return DrivingEnv(config, scenario="lane", seed=2)

    def test_reactive_car_never_hits_obstacle(self) -> None:
        env = self.make_env()
        leader = TrafficCar(0, 80.0, 0.0, behavior="obstacle")
        follower = TrafficCar(0, 20.0, 15.0, behavior="reactive")
        env.traffic = [leader, follower]
        slowed = False
        for _ in range(300):
            env.step(Action.MAINTAIN)
            if follower.speed_mps < 14.0:
                slowed = True
            self.assertGreaterEqual(
                leader.y_m - follower.y_m, env.config.car_length_m
            )
        self.assertTrue(slowed)

    def test_reactive_car_recovers_toward_cruise_speed(self) -> None:
        env = self.make_env()
        car = TrafficCar(0, 60.0, 8.0, behavior="reactive", cruise_speed_mps=20.0)
        env.traffic = [car]
        for _ in range(200):
            env.step(Action.MAINTAIN)
        self.assertGreater(car.speed_mps, 18.0)

    def test_cruiser_behavior_is_unchanged(self) -> None:
        env = self.make_env()
        car = TrafficCar(0, 60.0, 15.0, behavior="cruiser")
        env.traffic = [car]
        for _ in range(50):
            env.step(Action.MAINTAIN)
        self.assertEqual(car.speed_mps, 15.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_scenarios.ReactiveBrakingTests -v`
Expected: `test_reactive_car_never_hits_obstacle` FAILS (the follower drives through the obstacle) and `test_reactive_car_recovers_toward_cruise_speed` FAILS (speed stays 8.0). `test_cruiser_behavior_is_unchanged` may already pass.

- [ ] **Step 3: Implement in `autodrive_rl/environment.py`**

Add module constants:

```python
REACTIVE_TIME_HEADWAY_S = 1.5
REACTIVE_BRAKE_MPS2 = 4.0
REACTIVE_ACCEL_MPS2 = 2.0
```

In `step()`, insert `self._update_traffic()` immediately before the loop that moves traffic (`for car in self.traffic: car.y_m += ...`). Add the methods:

```python
    def _update_traffic(self) -> None:
        for car in self.traffic:
            if car.behavior == "reactive":
                self._update_reactive(car)

    def _update_reactive(self, car: TrafficCar) -> None:
        cfg = self.config
        lane = self.traffic_lane(car)
        gap, leader_speed = self._front_gap_for(car, lane)
        assert car.cruise_speed_mps is not None
        closing = max(0.0, car.speed_mps - leader_speed)
        threshold = (
            REACTIVE_TIME_HEADWAY_S * car.speed_mps
            + closing**2 / (2.0 * REACTIVE_BRAKE_MPS2)
        )
        if gap < threshold:
            car.speed_mps = max(
                0.0, car.speed_mps - REACTIVE_BRAKE_MPS2 * cfg.dt_seconds
            )
        elif car.speed_mps < car.cruise_speed_mps:
            car.speed_mps = min(
                car.cruise_speed_mps,
                car.speed_mps + REACTIVE_ACCEL_MPS2 * cfg.dt_seconds,
            )

    def _front_gap_for(self, car: TrafficCar, lane: int) -> tuple[float, float]:
        """Gap to the nearest leader in a lane, plus that leader's speed."""

        cfg = self.config
        gap = float("inf")
        leader_speed = 0.0
        for other in self.traffic:
            if other is car:
                continue
            if self.traffic_lane(other) == lane and other.y_m > car.y_m:
                candidate = other.y_m - car.y_m - cfg.car_length_m
                if candidate < gap:
                    gap = candidate
                    leader_speed = other.speed_mps
        if lane == self.current_lane and car.y_m < 0.0:
            candidate = -car.y_m - cfg.car_length_m
            if candidate < gap:
                gap = candidate
                leader_speed = self.ego_speed_mps
        return max(0.0, gap), leader_speed
```

(The ego car sits at relative y = 0; a traffic car with `y_m < 0` is behind the ego, so the ego is its leader.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_scenarios -v`
Expected: 17 tests PASS.

- [ ] **Step 5: Run the full suite, then commit**

Run: `python -m unittest discover -s tests`
Expected: 29 tests pass.

```bash
git add autodrive_rl/environment.py tests/test_scenarios.py
git commit -m "feat: reactive traffic brakes on time headway and recovers to cruise"
```

---

### Task 5: Reactive lane changes (clearance check + animation)

**Files:**
- Modify: `autodrive_rl/environment.py`
- Test: `tests/test_scenarios.py` (append)

**Interfaces:**
- Consumes: `_update_traffic` / `_update_reactive` / `_front_gap_for` from Task 4; `traffic_x_m` from Task 3.
- Produces:
  - Module constants `LANE_CHANGE_DURATION_S = 1.2`, `LANE_CHANGE_PROBABILITY = 0.005`, `LANE_CHANGE_MIN_FRONT_GAP_M = 15.0`, `LANE_CHANGE_MIN_REAR_GAP_M = 12.0`.
  - `_rear_gap_for(car: TrafficCar, lane: int) -> float` (ego-aware, mirror of `_front_gap_for`).
  - `_lane_change_ok(car: TrafficCar, candidate: int, current_gap: float) -> bool`.
  - `_maybe_start_lane_change(car: TrafficCar, lane: int)`.
  - `_update_traffic` additionally advances `lane_change_progress` and finalizes `car.lane` on completion.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scenarios.py`:

```python
class LaneChangeTests(unittest.TestCase):
    def make_env(self, max_steps: int = 2000) -> DrivingEnv:
        config = replace(EnvConfig(), max_steps=max_steps)
        return DrivingEnv(config, scenario="lane", seed=6)

    def test_lane_change_rejects_occupied_lane(self) -> None:
        env = self.make_env()
        car = TrafficCar(0, 50.0, 10.0, behavior="reactive", cruise_speed_mps=25.0)
        blocker = TrafficCar(1, 55.0, 10.0)
        env.traffic = [car, blocker]
        current_gap, _ = env._front_gap_for(car, 0)
        self.assertFalse(env._lane_change_ok(car, 1, current_gap))

    def test_lane_change_accepts_clear_lane(self) -> None:
        env = self.make_env()
        leader = TrafficCar(0, 60.0, 0.0, behavior="obstacle")
        car = TrafficCar(0, 50.0, 10.0, behavior="reactive", cruise_speed_mps=25.0)
        env.traffic = [leader, car]
        current_gap, _ = env._front_gap_for(car, 0)
        self.assertTrue(env._lane_change_ok(car, 1, current_gap))

    def test_rear_gap_check_includes_ego(self) -> None:
        env = self.make_env()
        ego_lane = env.current_lane
        car = TrafficCar(ego_lane - 1, 8.0, 10.0, behavior="reactive", cruise_speed_mps=25.0)
        env.traffic = [car]
        self.assertLess(env._rear_gap_for(car, ego_lane), 12.0)
        current_gap, _ = env._front_gap_for(car, ego_lane - 1)
        self.assertFalse(env._lane_change_ok(car, ego_lane, current_gap))

    def test_lane_change_animates_then_completes(self) -> None:
        env = self.make_env()
        car = TrafficCar(0, 50.0, 15.0, target_lane=1)
        env.traffic = [car]
        env.step(Action.MAINTAIN)
        self.assertIsNotNone(car.target_lane)
        self.assertGreater(car.lane_change_progress, 0.0)
        for _ in range(15):
            env.step(Action.MAINTAIN)
        self.assertEqual(car.lane, 1)
        self.assertIsNone(car.target_lane)

    def test_blocked_reactive_car_eventually_changes_lane(self) -> None:
        env = self.make_env()
        leader = TrafficCar(0, 90.0, 0.0, behavior="obstacle")
        car = TrafficCar(0, 30.0, 18.0, behavior="reactive", cruise_speed_mps=24.0)
        env.traffic = [leader, car]
        for _ in range(1500):
            env.step(Action.MAINTAIN)
            if env.traffic_lane(car) != 0:
                break
        self.assertNotEqual(env.traffic_lane(car), 0)
```

Note on the last test: the trigger is stochastic (0.5% per eligible step) but the env RNG is seeded, so the outcome is deterministic for seed 6. If it fails only because the roll never fires within 1500 steps, adjust the env seed in `make_env` until it passes and note the seed dependency in a comment — do not change the probability constant.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_scenarios.LaneChangeTests -v`
Expected: FAIL — `AttributeError: 'DrivingEnv' object has no attribute '_lane_change_ok'` (and the animation test fails because progress never advances).

- [ ] **Step 3: Implement in `autodrive_rl/environment.py`**

Add module constants:

```python
LANE_CHANGE_DURATION_S = 1.2
LANE_CHANGE_PROBABILITY = 0.005
LANE_CHANGE_MIN_FRONT_GAP_M = 15.0
LANE_CHANGE_MIN_REAR_GAP_M = 12.0
```

Replace `_update_traffic` with:

```python
    def _update_traffic(self) -> None:
        cfg = self.config
        for car in self.traffic:
            if car.behavior == "reactive":
                self._update_reactive(car)
            if car.target_lane is not None:
                car.lane_change_progress += cfg.dt_seconds / LANE_CHANGE_DURATION_S
                if car.lane_change_progress >= 1.0:
                    car.lane = car.target_lane
                    car.target_lane = None
                    car.lane_change_progress = 0.0
```

Extend `_update_reactive` — append at the end of the method:

```python
        if (
            car.target_lane is None
            and car.speed_mps < 0.8 * car.cruise_speed_mps
            and self.rng.random() < LANE_CHANGE_PROBABILITY
        ):
            self._maybe_start_lane_change(car, lane)
```

Add the three new methods:

```python
    def _maybe_start_lane_change(self, car: TrafficCar, lane: int) -> None:
        current_gap, _ = self._front_gap_for(car, lane)
        candidates = [c for c in (lane - 1, lane + 1) if 0 <= c < self.config.lane_count]
        if len(candidates) == 2 and self.rng.random() < 0.5:
            candidates.reverse()
        for candidate in candidates:
            if self._lane_change_ok(car, candidate, current_gap):
                car.target_lane = candidate
                car.lane_change_progress = 0.0
                return

    def _lane_change_ok(self, car: TrafficCar, candidate: int, current_gap: float) -> bool:
        front_gap, _ = self._front_gap_for(car, candidate)
        rear_gap = self._rear_gap_for(car, candidate)
        return (
            front_gap > current_gap
            and front_gap >= LANE_CHANGE_MIN_FRONT_GAP_M
            and rear_gap >= LANE_CHANGE_MIN_REAR_GAP_M
        )

    def _rear_gap_for(self, car: TrafficCar, lane: int) -> float:
        cfg = self.config
        gap = float("inf")
        for other in self.traffic:
            if other is car:
                continue
            if self.traffic_lane(other) == lane and other.y_m < car.y_m:
                gap = min(gap, car.y_m - other.y_m - cfg.car_length_m)
        if lane == self.current_lane and car.y_m > 0.0:
            gap = min(gap, car.y_m - cfg.car_length_m)
        return max(0.0, gap)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_scenarios -v`
Expected: 22 tests PASS.

- [ ] **Step 5: Run the full suite, then commit**

Run: `python -m unittest discover -s tests`
Expected: 34 tests pass.

```bash
git add autodrive_rl/environment.py tests/test_scenarios.py
git commit -m "feat: reactive traffic changes lanes with clearance checks and animation"
```

---

### Task 6: Obstacle recycling and old-checkpoint compatibility

**Files:**
- Modify: `autodrive_rl/environment.py` (`_recycle_traffic`)
- Test: `tests/test_scenarios.py` (append)

**Interfaces:**
- Consumes: `_obstacle_position_ok` from Task 2.
- Produces: obstacles recycle far ahead (like traffic) while keeping the survivability guarantee; no new public API.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scenarios.py` (add these imports at the top: `from pathlib import Path` and `from autodrive_rl.dqn import DQNAgent`):

```python
class ObstacleLifecycleTests(unittest.TestCase):
    def test_obstacles_never_move_and_recycle_ahead(self) -> None:
        config = replace(EnvConfig(), max_steps=900)
        spec = ScenarioSpec(traffic_count=0, obstacle_count=2)
        env = DrivingEnv(config, scenario="traffic", seed=9, scenario_spec=spec)
        for _ in range(600):
            before = {id(car): car.y_m for car in env.traffic}
            env.step(Action.ACCELERATE)
            for car in env.traffic:
                self.assertEqual(car.behavior, "obstacle")
                self.assertEqual(car.speed_mps, 0.0)
                moved = car.y_m - before[id(car)]
                recycled = moved > env.config.sensor_range_m / 2.0
                drifted_back = moved < 0.0
                self.assertTrue(recycled or drifted_back)
        self.assertEqual(
            sum(1 for car in env.traffic if car.behavior == "obstacle"), 2
        )


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_old_checkpoint_drives_in_new_environment(self) -> None:
        path = Path("models/autodrive_dqn_best.npz")
        if not path.exists():
            self.skipTest("no shipped checkpoint available")
        agent = DQNAgent.load(path)
        config = replace(EnvConfig(), max_steps=100)
        spec = ScenarioSpec(traffic_count=14, obstacle_count=2, reactive_fraction=0.5)
        env = DrivingEnv(config, scenario="traffic", seed=12, scenario_spec=spec)
        observation, _ = env.reset(seed=12)
        for _ in range(50):
            action = agent.act(observation, explore=False)
            self.assertIn(action, range(env.action_size))
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_scenarios.ObstacleLifecycleTests -v`
Expected: FAIL — a recycled obstacle lands via the moving-traffic recycle path, which assigns it a nonzero `speed_mps` (the `assertEqual(car.speed_mps, 0.0)` breaks after enough steps). `CheckpointCompatibilityTests` may already pass — that is fine; it locks the contract.

- [ ] **Step 3: Implement in `autodrive_rl/environment.py`**

In `_recycle_traffic`, handle obstacles before the existing reassignment logic — the loop body becomes:

```python
        for car in self.traffic:
            if -45.0 <= car.y_m <= cfg.sensor_range_m + 90.0:
                continue
            if car.behavior == "obstacle":
                for _ in range(50):
                    lane = int(self.rng.integers(0, cfg.lane_count))
                    y_m = float(
                        self.rng.uniform(cfg.sensor_range_m + 20.0, cfg.sensor_range_m + 80.0)
                    )
                    if self._obstacle_position_ok(lane, y_m):
                        car.lane = lane
                        car.y_m = y_m
                        break
                continue
            car.lane = self._least_crowded_spawn_lane()
            car.y_m = float(self.rng.uniform(cfg.sensor_range_m + 20.0, cfg.sensor_range_m + 80.0))
            car.speed_mps = float(self.rng.uniform(9.0, 26.0))
            car.color_index = int(self.rng.integers(0, 6))
```

Also reset lane-change state when a moving car recycles — add after `car.color_index = ...`:

```python
            car.cruise_speed_mps = car.speed_mps
            car.target_lane = None
            car.lane_change_progress = 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_scenarios -v`
Expected: 24 tests PASS.

- [ ] **Step 5: Run the full suite, then commit**

Run: `python -m unittest discover -s tests`
Expected: 36 tests pass.

```bash
git add autodrive_rl/environment.py tests/test_scenarios.py
git commit -m "feat: obstacles recycle ahead safely; old checkpoints verified compatible"
```

---

### Task 7: Training integration — randomization, CSV columns, CLI flags

**Files:**
- Modify: `autodrive_rl/train.py`
- Test: `tests/test_scenarios.py` (append)

**Interfaces:**
- Consumes: `resolve_scenario`, `ScenarioSpec` from Task 1; `DrivingEnv(scenario_spec=...)` from Task 2.
- Produces (Task 8 builds on this):
  - `train(..., scenario_preset: str = "random", traffic: int | None = None, obstacles: int | None = None, reactive: float | None = None)` — new keyword params.
  - CLI flags `--scenario-preset {sparse,normal,dense,random}` (default `random`), `--traffic`, `--obstacles`, `--reactive`.
  - Per-episode CSV columns `traffic_count`, `obstacle_count`, `reactive_fraction`.
  - A dedicated scenario RNG: `np.random.default_rng(seed + 777_777)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scenarios.py` (add imports: `import tempfile` and `from autodrive_rl.train import build_parser, train`):

```python
class TrainingIntegrationTests(unittest.TestCase):
    def test_parser_has_scenario_flags(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.scenario_preset, "random")
        self.assertIsNone(args.traffic)
        args = build_parser().parse_args(
            ["--scenario-preset", "dense", "--traffic", "6", "--reactive", "0.3"]
        )
        self.assertEqual(args.scenario_preset, "dense")
        self.assertEqual(args.traffic, 6)
        self.assertEqual(args.reactive, 0.3)

    def test_randomized_training_records_conditions_reproducibly(self) -> None:
        def run(directory: str) -> list[dict[str, object]]:
            base = Path(directory)
            _, records = train(
                episodes=6,
                env_config=replace(EnvConfig(), max_steps=60),
                seed=11,
                curriculum=False,
                eval_every=0,
                eval_episodes=1,
                log_every=100,
                scenario_preset="random",
                output_path=base / "m.npz",
                metrics_path=base / "m.csv",
            )
            return records

        with tempfile.TemporaryDirectory() as first_dir:
            first = run(first_dir)
        with tempfile.TemporaryDirectory() as second_dir:
            second = run(second_dir)
        for record in first:
            self.assertIn("traffic_count", record)
            self.assertIn("obstacle_count", record)
            self.assertIn("reactive_fraction", record)
            self.assertTrue(4 <= int(record["traffic_count"]) <= 14)
        first_conditions = [
            (r["traffic_count"], r["obstacle_count"], r["reactive_fraction"]) for r in first
        ]
        second_conditions = [
            (r["traffic_count"], r["obstacle_count"], r["reactive_fraction"]) for r in second
        ]
        self.assertEqual(first_conditions, second_conditions)
```

Note: `eval_every=0` skips evaluation entirely (the existing `should_evaluate` already guards on `eval_every > 0`), keeping this test fast (~10 s).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_scenarios.TrainingIntegrationTests -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'scenario_preset'` and `TypeError: train() got an unexpected keyword argument 'scenario_preset'`.

- [ ] **Step 3: Implement in `autodrive_rl/train.py`**

Import additions:

```python
from .config import DQNConfig, EnvConfig, ScenarioSpec, resolve_scenario
```

In `build_parser()`, after the `--scenario` argument:

```python
    parser.add_argument(
        "--scenario-preset",
        choices=("sparse", "normal", "dense", "random"),
        default="random",
        help="world conditions for full-traffic episodes (default: random)",
    )
    parser.add_argument("--traffic", type=int, default=None, help="override car count")
    parser.add_argument("--obstacles", type=int, default=None, help="override obstacle count")
    parser.add_argument("--reactive", type=float, default=None, help="override reactive fraction 0..1")
```

Extend the `train()` signature (after `curriculum: bool = True`):

```python
    scenario_preset: str = "random",
    traffic: int | None = None,
    obstacles: int | None = None,
    reactive: float | None = None,
```

Before the episode loop, create the dedicated RNG:

```python
    scenario_rng = np.random.default_rng(seed + 777_777)
```

In the episode loop, initialize `episode_spec: ScenarioSpec | None = None` at the top of each iteration; in the final `else` branch (full-traffic stage) add:

```python
            if scenario == "traffic":
                episode_spec = resolve_scenario(
                    scenario_preset,
                    traffic=traffic,
                    obstacles=obstacles,
                    reactive=reactive,
                    rng=scenario_rng,
                )
```

Change the env construction to pass it:

```python
        env = DrivingEnv(
            episode_config, scenario=env_scenario, seed=seed + episode, scenario_spec=episode_spec
        )
```

In the `record` dict, insert after `"scenario": episode_scenario,`:

```python
            "traffic_count": sum(1 for car in env.traffic if car.behavior != "obstacle"),
            "obstacle_count": sum(1 for car in env.traffic if car.behavior == "obstacle"),
            "reactive_fraction": round(episode_spec.reactive_fraction, 4)
            if episode_spec is not None
            else 0.0,
```

In `main()`, pass the new args through to `train(...)`:

```python
        scenario_preset=args.scenario_preset,
        traffic=args.traffic,
        obstacles=args.obstacles,
        reactive=args.reactive,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_scenarios.TrainingIntegrationTests -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Run the full suite, then commit**

Run: `python -m unittest discover -s tests`
Expected: 38 tests pass.

```bash
git add autodrive_rl/train.py tests/test_scenarios.py
git commit -m "feat: domain-randomized training with per-episode condition logging"
```

---

### Task 8: Matrix evaluation and robust best-checkpoint selection

**Files:**
- Modify: `autodrive_rl/train.py` (`evaluate`, the `should_evaluate` block)
- Test: `tests/test_scenarios.py` (append)

**Interfaces:**
- Consumes: `SCENARIO_PRESETS` from Task 1, `evaluate()` and the record dict from Task 7.
- Produces:
  - `evaluate(agent, env_config, *, episodes, seed, scenario="traffic", scenario_spec: ScenarioSpec | None = None) -> dict[str, float]`.
  - Module constant `EVAL_CELLS = ("sparse", "normal", "dense")` in `train.py`.
  - CSV columns `eval_<cell>_return` and `eval_<cell>_safe_rate` for each cell; the existing `eval_return` / `eval_distance_m` / `eval_safe_rate` become means across cells (traffic scenario only; the pure `lane` scenario keeps single-cell behavior).
  - Best checkpoint selected on the matrix-mean return.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scenarios.py`:

```python
class MatrixEvaluationTests(unittest.TestCase):
    def test_matrix_eval_fills_cells_and_mean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _, records = train(
                episodes=2,
                env_config=replace(EnvConfig(), max_steps=40),
                seed=13,
                curriculum=False,
                eval_every=2,
                eval_episodes=1,
                log_every=100,
                scenario_preset="normal",
                output_path=base / "m.npz",
                metrics_path=base / "m.csv",
            )
        final = records[-1]
        cell_returns = []
        for cell in ("sparse", "normal", "dense"):
            self.assertNotEqual(final[f"eval_{cell}_return"], "")
            self.assertNotEqual(final[f"eval_{cell}_safe_rate"], "")
            cell_returns.append(float(final[f"eval_{cell}_return"]))
        mean_return = sum(cell_returns) / len(cell_returns)
        self.assertAlmostEqual(float(final["eval_return"]), mean_return, places=3)
        first = records[0]
        self.assertEqual(first["eval_sparse_return"], "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_scenarios.MatrixEvaluationTests -v`
Expected: FAIL — `KeyError: 'eval_sparse_return'`.

- [ ] **Step 3: Implement in `autodrive_rl/train.py`**

Add near the imports:

```python
from .config import SCENARIO_PRESETS  # extend the existing config import line

EVAL_CELLS = ("sparse", "normal", "dense")
```

Extend `evaluate()` — new signature and env construction:

```python
def evaluate(
    agent: DQNAgent,
    env_config: EnvConfig,
    *,
    episodes: int,
    seed: int,
    scenario: str = "traffic",
    scenario_spec: ScenarioSpec | None = None,
) -> dict[str, float]:
    ...
        env = DrivingEnv(
            env_config,
            scenario=scenario,
            seed=seed + episode_index,
            scenario_spec=scenario_spec,
        )
```

(Only the constructor call changes inside; the loop body is untouched.)

In the record dict (right after the three existing empty `eval_*` fields), add the per-cell placeholders:

```python
        for cell in EVAL_CELLS:
            record[f"eval_{cell}_return"] = ""
            record[f"eval_{cell}_safe_rate"] = ""
```

Replace the body of the `if should_evaluate:` block:

```python
        if should_evaluate:
            if scenario == "traffic":
                cell_results: dict[str, dict[str, float]] = {}
                for cell_index, cell in enumerate(EVAL_CELLS):
                    cell_results[cell] = evaluate(
                        agent,
                        env_config,
                        episodes=eval_episodes,
                        seed=seed + 100_000 + episode * eval_episodes + 10_000 * cell_index,
                        scenario=scenario,
                        scenario_spec=SCENARIO_PRESETS[cell],
                    )
                mean_return = fmean(result["return"] for result in cell_results.values())
                record["eval_return"] = round(mean_return, 5)
                record["eval_distance_m"] = round(
                    fmean(result["distance_m"] for result in cell_results.values()), 3
                )
                record["eval_safe_rate"] = round(
                    fmean(result["safe_rate"] for result in cell_results.values()), 4
                )
                for cell in EVAL_CELLS:
                    record[f"eval_{cell}_return"] = round(cell_results[cell]["return"], 5)
                    record[f"eval_{cell}_safe_rate"] = round(cell_results[cell]["safe_rate"], 4)
            else:
                evaluation = evaluate(
                    agent,
                    env_config,
                    episodes=eval_episodes,
                    seed=seed + 100_000 + episode * eval_episodes,
                    scenario=scenario,
                )
                mean_return = evaluation["return"]
                record["eval_return"] = round(evaluation["return"], 5)
                record["eval_distance_m"] = round(evaluation["distance_m"], 3)
                record["eval_safe_rate"] = round(evaluation["safe_rate"], 4)
            if mean_return > best_eval_return:
                best_eval_return = mean_return
                agent.save(best_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_scenarios.MatrixEvaluationTests -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite, then commit**

Run: `python -m unittest discover -s tests`
Expected: 39 tests pass.

```bash
git add autodrive_rl/train.py tests/test_scenarios.py
git commit -m "feat: sparse/normal/dense evaluation matrix drives best-checkpoint choice"
```

---

### Task 9: Play flags and renderer support for the new world

**Files:**
- Modify: `autodrive_rl/play.py`, `autodrive_rl/renderer.py`
- Test: `tests/test_scenarios.py` (append; renderer verified manually)

**Interfaces:**
- Consumes: `resolve_scenario` from Task 1, `env.traffic_x_m()` from Task 3, `behavior` field from Task 2.
- Produces: `play` CLI flags `--scenario-preset` (default `normal`), `--traffic`, `--obstacles`, `--reactive`; renderer draws obstacles as striped hazard blocks and traffic at interpolated x.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scenarios.py` (add import: `from autodrive_rl.play import build_parser as build_play_parser`):

```python
class PlayParserTests(unittest.TestCase):
    def test_play_parser_scenario_flags(self) -> None:
        args = build_play_parser().parse_args([])
        self.assertEqual(args.scenario_preset, "normal")
        args = build_play_parser().parse_args(
            ["--scenario-preset", "dense", "--obstacles", "3"]
        )
        self.assertEqual(args.scenario_preset, "dense")
        self.assertEqual(args.obstacles, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_scenarios.PlayParserTests -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'scenario_preset'`.

- [ ] **Step 3: Implement**

In `autodrive_rl/play.py` — add imports `import numpy as np` and `from .config import resolve_scenario`; in `build_parser()` after the `--scenario` argument:

```python
    parser.add_argument(
        "--scenario-preset",
        choices=("sparse", "normal", "dense", "random"),
        default="normal",
        help="world conditions to drive in (default: normal, today's world)",
    )
    parser.add_argument("--traffic", type=int, default=None, help="override car count")
    parser.add_argument("--obstacles", type=int, default=None, help="override obstacle count")
    parser.add_argument("--reactive", type=float, default=None, help="override reactive fraction 0..1")
```

In `main()`, replace the env construction:

```python
    scenario_spec = resolve_scenario(
        args.scenario_preset,
        traffic=args.traffic,
        obstacles=args.obstacles,
        reactive=args.reactive,
        rng=np.random.default_rng(args.seed),
    )
    env = DrivingEnv(scenario=args.scenario, seed=args.seed, scenario_spec=scenario_spec)
```

In `autodrive_rl/renderer.py` — in `_draw_world`, replace the traffic-drawing loop body:

```python
        for car in visible_cars:
            screen_y = self.ego_screen_y - car.y_m * self.longitudinal_scale
            if -60 <= screen_y <= self.height + 60:
                x = self._world_x_to_screen(env, env.traffic_x_m(car))
                if car.behavior == "obstacle":
                    self._draw_obstacle(x, screen_y)
                    continue
                color = self.traffic_colors[car.color_index % len(self.traffic_colors)]
                self._draw_car(x, screen_y, color, label=f"{car.speed_mps * 2.236936:.0f}")
```

Add the new method after `_draw_car`:

```python
    def _draw_obstacle(self, x: float, y: float) -> None:
        half_width = 25
        half_length = 30
        self.canvas.create_rectangle(
            x - half_width,
            y - half_length,
            x + half_width,
            y + half_length,
            fill="#3b3f46",
            outline="#ffca3a",
            width=3,
        )
        for offset in range(-half_length + 8, half_length, 14):
            self.canvas.create_line(
                x - half_width + 4,
                y + offset + 10,
                x + half_width - 4,
                y + offset - 4,
                fill="#ffca3a",
                width=4,
            )
```

- [ ] **Step 4: Run test, then verify visually**

Run: `python -m unittest tests.test_scenarios.PlayParserTests -v`
Expected: PASS.

Manual check (opens a window; close with Q):
`python -m autodrive_rl.play --policy heuristic --scenario-preset dense --seed 3`
Expected: 14 cars, 2 yellow-striped hazard blocks, visible braking waves behind obstacles, and occasional traffic lane changes that slide smoothly rather than teleporting.

- [ ] **Step 5: Run the full suite, then commit**

Run: `python -m unittest discover -s tests`
Expected: 40 tests pass.

```bash
git add autodrive_rl/play.py autodrive_rl/renderer.py tests/test_scenarios.py
git commit -m "feat: scenario flags for play and hazard-block obstacle rendering"
```

---

### Task 10: Documentation and end-to-end smoke run

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: user-facing documentation of the new flags; a verified short randomized training run.

- [ ] **Step 1: Document the new capability in `README.md`**

Insert a new section after "Compare policies":

```markdown
## Vary the world

Scenario presets control traffic density, static obstacles, and how many
drivers react (brake on short headway, change lanes when blocked):

```bash
python -m autodrive_rl.play --policy heuristic --scenario-preset dense
python -m autodrive_rl.play --policy dqn --model models/autodrive_dqn_best.npz --scenario-preset sparse
```

Presets: `sparse` (4 cars), `normal` (today's 9-car world), `dense`
(14 cars, 2 obstacles, half the drivers reactive), `random`. Override any
field with `--traffic N`, `--obstacles N`, or `--reactive F` (0 to 1).

Training now uses domain randomization by default: after the warm-up
curriculum, every episode rolls fresh conditions from the `random` ranges,
and periodic evaluations run a sparse/normal/dense matrix. The best
checkpoint is the one with the highest mean return across that matrix. Use
`--scenario-preset normal` to reproduce the old fixed-world training.
```

- [ ] **Step 2: End-to-end smoke run**

Run: `python -m autodrive_rl.train --episodes 30 --output models/smoke_variety.npz --metrics runs/smoke_variety.csv`
Expected: completes without error; the CSV contains `traffic_count`, `obstacle_count`, `reactive_fraction` columns varying across post-curriculum episodes, plus `eval_sparse_return` … `eval_dense_safe_rate` columns filled on evaluation rows.

Then delete the smoke artifacts: `models/smoke_variety.npz`, `models/smoke_variety_best.npz`, `runs/smoke_variety.csv`.

- [ ] **Step 3: Full suite one last time, then commit**

Run: `python -m unittest discover -s tests`
Expected: 40 tests pass.

```bash
git add README.md
git commit -m "docs: document scenario presets and randomized training"
```

---

## Follow-up (not in this plan)

- Re-run the 100-seed held-out benchmark per matrix cell and update `BENCHMARK.md` with a per-condition table (spec section 5) — this needs a real multi-hundred-episode training run, so it is an experiment to run after implementation, not an implementation task.
- UI modernization (separate design).
