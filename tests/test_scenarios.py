from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from autodrive_rl.config import (
    SCENARIO_PRESETS,
    EnvConfig,
    ScenarioRanges,
    ScenarioSpec,
    resolve_scenario,
    sample_scenario,
)
from autodrive_rl.clone import DemoRecorder, load_demos, save_clone, train_clone
from autodrive_rl.dqn import DQNAgent
from autodrive_rl.environment import Action, DrivingEnv, TrafficCar, grade_letter
from autodrive_rl.play import build_parser as build_play_parser
from autodrive_rl.train import build_parser, train


class ScenarioSpecTests(unittest.TestCase):
    def test_presets_exist_and_normal_matches_default_world(self) -> None:
        self.assertEqual(
            set(SCENARIO_PRESETS), {"sparse", "normal", "dense", "unforgiving"}
        )
        normal = SCENARIO_PRESETS["normal"]
        self.assertEqual(normal.traffic_count, EnvConfig().traffic_count)
        self.assertEqual(normal.obstacle_count, 0)
        self.assertEqual(normal.reactive_fraction, 0.0)
        dense = SCENARIO_PRESETS["dense"]
        self.assertEqual((dense.traffic_count, dense.slow_vehicle_count), (16, 2))
        self.assertEqual(dense.reactive_fraction, 0.5)

    def test_no_traffic_preset_uses_static_obstacles(self) -> None:
        """A permanent immovable block in a live motorway lane, with a queue
        stacked behind it, is not a problem a highway driver ever solves. Slow
        vehicles create the same lane-change pressure realistically."""
        for name, spec in SCENARIO_PRESETS.items():
            self.assertEqual(spec.obstacle_count, 0, name)

    def test_only_unforgiving_puts_traffic_behind_the_ego(self) -> None:
        """The three original presets are frozen: `BENCHMARK_current.md` was
        measured on them, so a world-model change must not touch them."""
        for name in ("sparse", "normal", "dense"):
            spec = SCENARIO_PRESETS[name]
            self.assertFalse(spec.rear_traffic, name)
            self.assertEqual(spec.inattentive_fraction, 0.0, name)
        hard = SCENARIO_PRESETS["unforgiving"]
        self.assertTrue(hard.rear_traffic)
        self.assertGreater(hard.inattentive_fraction, 0.0)

    def test_invalid_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            ScenarioSpec(traffic_count=-1)
        with self.assertRaises(ValueError):
            ScenarioSpec(obstacle_count=-1)
        with self.assertRaises(ValueError):
            ScenarioSpec(reactive_fraction=1.5)
        with self.assertRaises(ValueError):
            ScenarioSpec(inattentive_fraction=1.5)
        with self.assertRaises(ValueError):
            ScenarioSpec(slow_vehicle_count=-1)
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

    def test_dense_preset_places_all_requested_obstacles(self) -> None:
        for seed in range(30):
            env = self.make_env(
                seed,
                ScenarioSpec(traffic_count=14, obstacle_count=2, reactive_fraction=0.5),
            )
            obstacles = sum(1 for car in env.traffic if car.behavior == "obstacle")
            self.assertEqual(obstacles, 2)


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
            if follower.target_lane is not None or env.traffic_lane(follower) != 0:
                # The car legitimately escaped by merging out of the blocked
                # lane; the same-lane following invariant no longer applies.
                break
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

    def test_cruiser_brakes_for_slower_car_ahead(self) -> None:
        env = self.make_env()
        leader = TrafficCar(0, 70.0, 5.0)
        car = TrafficCar(0, 40.0, 20.0, behavior="cruiser")
        env.traffic = [leader, car]
        slowed = False
        for _ in range(300):
            env.step(Action.MAINTAIN)
            if car.speed_mps < 19.0:
                slowed = True
            self.assertGreaterEqual(
                leader.y_m - car.y_m, env.config.car_length_m
            )
        self.assertTrue(slowed)

    def test_cruiser_never_rear_ends_ego(self) -> None:
        env = self.make_env()
        car = TrafficCar(env.current_lane, -35.0, 24.0, behavior="cruiser")
        env.traffic = [car]
        slowed = False
        for _ in range(300):
            env.step(Action.MAINTAIN)
            if car.speed_mps < 23.0:
                slowed = True
            self.assertLessEqual(car.y_m, -env.config.car_length_m)
        self.assertTrue(slowed)

    def test_no_moving_car_ever_exceeds_speed_limit(self) -> None:
        limit = EnvConfig().speed_limit_mps
        for seed in range(10):
            spec = ScenarioSpec(traffic_count=10, obstacle_count=1, reactive_fraction=0.5)
            env = DrivingEnv(
                replace(EnvConfig(), max_steps=300),
                scenario="traffic",
                seed=seed,
                scenario_spec=spec,
            )
            for car in env.traffic:
                self.assertLessEqual(car.speed_mps, limit + 1e-9)
                self.assertLessEqual(car.cruise_speed_mps, limit + 1e-9)
            for _ in range(200):
                env.step(Action.MAINTAIN)
                for car in env.traffic:
                    self.assertLessEqual(car.speed_mps, limit + 1e-9)

    def test_traffic_cars_never_rear_end_each_other(self) -> None:
        for seed in range(3):
            spec = ScenarioSpec(traffic_count=12, obstacle_count=2, reactive_fraction=0.5)
            env = DrivingEnv(
                replace(EnvConfig(), max_steps=300),
                scenario="traffic",
                seed=seed,
                scenario_spec=spec,
            )
            for _ in range(300):
                _, _, terminated, truncated, _ = env.step(Action.MAINTAIN)
                movers = [c for c in env.traffic if c.behavior != "obstacle"]
                for i, first in enumerate(movers):
                    for second in movers[i + 1:]:
                        if env.traffic_lane(first) == env.traffic_lane(second):
                            self.assertGreaterEqual(
                                abs(first.y_m - second.y_m),
                                env.config.car_length_m,
                            )
                if terminated or truncated:
                    break

    def test_no_two_cars_ever_overlap_geometrically(self) -> None:
        """The hard invariant: no pair of traffic cars (moving or obstacle)
        may ever occupy the same physical space, even with a wild ego."""

        for seed in range(3):
            spec = ScenarioSpec(traffic_count=12, obstacle_count=2, reactive_fraction=0.6)
            env = DrivingEnv(
                replace(EnvConfig(), max_steps=400),
                scenario="traffic",
                seed=seed,
                scenario_spec=spec,
            )
            action_rng = np.random.default_rng(seed + 500)
            for _ in range(400):
                _, _, terminated, truncated, _ = env.step(
                    int(action_rng.integers(0, env.action_size))
                )
                for i, first in enumerate(env.traffic):
                    for second in env.traffic[i + 1:]:
                        longitudinal = abs(first.y_m - second.y_m)
                        lateral = abs(env.traffic_x_m(first) - env.traffic_x_m(second))
                        overlapping = (
                            longitudinal < env.config.car_length_m
                            and lateral < env.config.car_width_m
                        )
                        self.assertFalse(
                            overlapping,
                            f"seed {seed}: {first} overlaps {second}",
                        )
                if terminated or truncated:
                    break

    def test_simultaneous_merges_into_same_gap_resolve(self) -> None:
        env = self.make_env()
        first = TrafficCar(
            0, 50.0, 15.0, behavior="reactive", target_lane=1, lane_change_progress=0.2
        )
        second = TrafficCar(
            2, 51.0, 15.0, behavior="reactive", target_lane=1, lane_change_progress=0.2
        )
        env.traffic = [first, second]
        for _ in range(30):
            env.step(Action.MAINTAIN)
            longitudinal = abs(first.y_m - second.y_m)
            lateral = abs(env.traffic_x_m(first) - env.traffic_x_m(second))
            self.assertFalse(
                longitudinal < env.config.car_length_m
                and lateral < env.config.car_width_m
            )
        # They cannot both have completed the merge into lane 1 on top of
        # each other; at least one must have pulled back.
        self.assertNotEqual(env.traffic_lane(first), env.traffic_lane(second))

    def test_lane_change_aborts_when_target_gap_collapses(self) -> None:
        env = self.make_env()
        changer = TrafficCar(
            0, 50.0, 15.0, behavior="reactive", target_lane=1, lane_change_progress=0.2
        )
        blocker = TrafficCar(1, 53.0, 15.0)
        env.traffic = [changer, blocker]
        env.step(Action.MAINTAIN)
        # The abort reverses the animation: the origin lane becomes the new
        # target so the car glides back where it came from.
        self.assertEqual(changer.target_lane, 0)
        for _ in range(15):
            env.step(Action.MAINTAIN)
        self.assertEqual(env.traffic_lane(changer), 0)
        self.assertIsNone(changer.target_lane)

    def test_speeding_car_settles_back_to_the_limit(self) -> None:
        env = self.make_env()
        car = TrafficCar(0, 60.0, 29.0, behavior="cruiser")
        env.traffic = [car]
        for _ in range(50):
            env.step(Action.MAINTAIN)
        self.assertLessEqual(car.speed_mps, env.config.speed_limit_mps + 1e-9)

    def test_cruiser_brakes_for_obstacle(self) -> None:
        env = self.make_env()
        obstacle = TrafficCar(0, 80.0, 0.0, behavior="obstacle")
        car = TrafficCar(0, 20.0, 15.0, behavior="cruiser")
        env.traffic = [obstacle, car]
        slowed = False
        for _ in range(300):
            env.step(Action.MAINTAIN)
            if car.speed_mps < 14.0:
                slowed = True
            self.assertGreaterEqual(
                obstacle.y_m - car.y_m, env.config.car_length_m
            )
        self.assertTrue(slowed)


class RearTrafficTests(unittest.TestCase):
    """Standing still has to cost something.

    Every policy in `BENCHMARK_current.md` scores well on safety by crawling,
    and the reason is geometric: all traffic used to spawn *ahead* of the ego,
    so braking to a stop made the world drive away and leave it alone. These
    tests pin the two halves of the fix — traffic that arrives from behind, and
    drivers too slow to avoid a car that stops in front of them.
    """

    def _episode_ends_in_a_collision(self, preset: str, seed: int) -> bool:
        env = DrivingEnv(
            replace(EnvConfig(), max_steps=400),
            scenario="traffic",
            seed=seed,
            scenario_spec=resolve_scenario(preset),
        )
        env.reset(seed=seed)
        for _ in range(400):
            _, _, terminated, truncated, info = env.step(Action.BRAKE)
            if terminated or truncated:
                return bool(info["collision"])
        return False

    def test_frozen_presets_keep_every_car_ahead_of_the_ego(self) -> None:
        for preset in ("sparse", "normal", "dense"):
            for seed in range(8):
                env = DrivingEnv(
                    EnvConfig(), scenario="traffic", seed=seed,
                    scenario_spec=resolve_scenario(preset),
                )
                for car in env.traffic:
                    self.assertGreater(car.y_m, 0.0, f"{preset}/{seed}: {car}")

    def test_unforgiving_puts_cars_behind_the_ego(self) -> None:
        found = False
        for seed in range(8):
            env = DrivingEnv(
                EnvConfig(), scenario="traffic", seed=seed,
                scenario_spec=resolve_scenario("unforgiving"),
            )
            found = found or any(car.y_m < 0.0 for car in env.traffic)
        self.assertTrue(found)

    def test_no_car_ever_spawns_on_top_of_the_ego(self) -> None:
        """Rear spawning must not become a free collision at step zero."""
        for seed in range(12):
            env = DrivingEnv(
                EnvConfig(), scenario="traffic", seed=seed,
                scenario_spec=resolve_scenario("unforgiving"),
            )
            for car in env.traffic:
                self.assertGreaterEqual(abs(car.y_m), env.config.car_length_m)

    def test_an_attentive_driver_sees_the_road_as_it_is(self) -> None:
        env = DrivingEnv(EnvConfig(), scenario="lane", seed=1)
        car = TrafficCar(0, 40.0, 20.0, behavior="cruiser", attentive=True)
        self.assertEqual(env._perceive(car, 30.0, 12.0), (30.0, 12.0))
        self.assertEqual(env._perceive(car, 5.0, 0.0), (5.0, 0.0))

    def test_an_inattentive_driver_acts_on_a_stale_reading(self) -> None:
        """This is the mechanism that makes stopping dangerous: by the time an
        inattentive driver perceives the gap closing, it is already too late to
        brake all the way out of it."""
        env = DrivingEnv(EnvConfig(), scenario="lane", seed=1)
        car = TrafficCar(0, 40.0, 20.0, behavior="cruiser", attentive=False)
        self.assertEqual(env._perceive(car, 60.0, 20.0), (60.0, 20.0))
        stale, _ = env._perceive(car, 2.0, 0.0)
        self.assertEqual(stale, 60.0)

    def test_a_zero_delay_config_disables_the_lag_entirely(self) -> None:
        env = DrivingEnv(replace(EnvConfig(), reaction_delay_s=0.0),
                         scenario="lane", seed=1)
        car = TrafficCar(0, 40.0, 20.0, behavior="cruiser", attentive=False)
        env._perceive(car, 60.0, 20.0)
        self.assertEqual(env._perceive(car, 2.0, 0.0), (2.0, 0.0))

    def test_standing_still_is_punished_on_unforgiving_and_not_on_dense(self) -> None:
        """The headline claim, asserted rather than described: an ego that
        brakes and sits there survives `dense` and gets hit on `unforgiving`."""
        seeds = range(360000, 360040)
        dense = sum(self._episode_ends_in_a_collision("dense", s) for s in seeds)
        hard = sum(self._episode_ends_in_a_collision("unforgiving", s) for s in seeds)
        self.assertEqual(dense, 0)
        self.assertGreaterEqual(hard, 6)


class StallAndFaultTests(unittest.TestCase):
    """Two rules that decide what the agent is actually being scored on."""

    def _spec(self, **kw):
        return ScenarioSpec(**kw)

    def test_an_obstructing_car_stalls_out(self) -> None:
        """Sitting still on an open motorway lane ends the episode."""
        env = DrivingEnv(
            replace(EnvConfig(), max_steps=400), scenario="traffic", seed=3,
            scenario_spec=resolve_scenario("normal"),
        )
        env.reset(seed=3)
        for _ in range(400):
            _, _, terminated, _, info = env.step(Action.BRAKE)
            if terminated:
                break
        self.assertTrue(info["stalled"])
        self.assertFalse(info["collision"])

    def test_slowing_behind_traffic_is_not_a_stall(self) -> None:
        """Ordinary driving includes stopping behind something. Only an open
        lane ahead makes it obstruction."""
        env = DrivingEnv(replace(EnvConfig(), max_steps=400), scenario="lane", seed=4)
        env.traffic = [TrafficCar(env.current_lane, 8.0, 0.0, behavior="obstacle")]
        env.ego_speed_mps = 0.0
        stall_limit = round(env.config.stall_grace_s / env.config.dt_seconds)
        for _ in range(stall_limit * 3):
            _, _, _, _, info = env.step(Action.BRAKE)
            self.assertFalse(info["stalled"])

    def test_the_rule_based_driver_never_obstructs(self) -> None:
        """If the heuristic stalls, the threshold is wrong rather than the
        driver — so this is the calibration check for the whole rule."""
        from autodrive_rl.benchmark import benchmark

        results = benchmark(
            ["heuristic"], cells=("normal", "dense"), episodes=6,
            seed_start=350_000, max_steps=400, progress=False,
        )
        for r in results:
            self.assertEqual(r.stall_rate, 0.0, r.cell)

    def test_running_into_the_car_ahead_is_the_egos_fault(self) -> None:
        env = DrivingEnv(EnvConfig(), scenario="lane", seed=5)
        ahead = TrafficCar(env.current_lane, 5.0, 0.0, behavior="obstacle")
        env.traffic = [ahead]
        self.assertTrue(env._ego_at_fault(ahead))

    def test_being_struck_from_behind_while_driving_is_not(self) -> None:
        """A follower owns its own stopping distance."""
        env = DrivingEnv(EnvConfig(), scenario="lane", seed=5)
        env.ego_speed_mps = 20.0
        env.ego_lateral_speed_mps = 0.0
        behind = TrafficCar(env.current_lane, -4.0, 26.0)
        env.traffic = [behind]
        self.assertFalse(env._ego_at_fault(behind))

    def test_being_struck_from_behind_while_stopped_is_the_egos_fault(self) -> None:
        """Stopping in a live lane is the unreasonable act that caused it."""
        env = DrivingEnv(EnvConfig(), scenario="lane", seed=5)
        env.ego_speed_mps = 0.0
        env.ego_lateral_speed_mps = 0.0
        behind = TrafficCar(env.current_lane, -4.0, 26.0)
        env.traffic = [behind]
        self.assertTrue(env._ego_at_fault(behind))

    def test_merging_into_someone_is_the_egos_fault(self) -> None:
        env = DrivingEnv(EnvConfig(), scenario="lane", seed=5)
        env.ego_speed_mps = 20.0
        env.ego_lateral_speed_mps = 2.0  # actively moving sideways
        alongside = TrafficCar(env.current_lane, -1.0, 20.0)
        env.traffic = [alongside]
        self.assertTrue(env._ego_at_fault(alongside))

    def test_slow_vehicles_move_but_stay_below_the_flow(self) -> None:
        for seed in range(6):
            env = DrivingEnv(
                EnvConfig(), scenario="traffic", seed=seed,
                scenario_spec=ScenarioSpec(traffic_count=4, slow_vehicle_count=2),
            )
            slow = sorted(c.cruise_speed_mps for c in env.traffic)[:2]
            for speed in slow:
                self.assertGreater(speed, 0.0)
                self.assertLess(speed, env.config.traffic_min_speed_mps + 0.1)


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
        rear_gap, follower_speed = env._rear_gap_for(car, ego_lane)
        self.assertLess(rear_gap, 12.0)
        self.assertEqual(follower_speed, env.ego_speed_mps)
        current_gap, _ = env._front_gap_for(car, ego_lane - 1)
        self.assertFalse(env._lane_change_ok(car, ego_lane, current_gap))

    def test_lane_change_rejects_fast_closing_follower(self) -> None:
        env = self.make_env()
        blocker = TrafficCar(0, 60.0, 0.0, behavior="obstacle")
        car = TrafficCar(0, 30.0, 10.0, behavior="reactive", cruise_speed_mps=24.0)
        fast_follower = TrafficCar(2, 10.0, 24.0)
        env.traffic = [blocker, car, fast_follower]
        current_gap, _ = env._front_gap_for(car, 0)
        # A 15 m rear gap would have passed the old fixed threshold, but the
        # follower closes at 14 m/s: merging would cut it off.
        self.assertFalse(env._lane_change_ok(car, 2, current_gap))
        fast_follower.speed_mps = 10.0
        self.assertTrue(env._lane_change_ok(car, 2, current_gap))

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

    def test_invalid_override_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with self.assertRaises(ValueError):
                train(
                    episodes=5,
                    env_config=replace(EnvConfig(), max_steps=40),
                    seed=1,
                    curriculum=True,
                    eval_every=0,
                    eval_episodes=1,
                    log_every=100,
                    traffic=-3,
                    output_path=base / "m.npz",
                    metrics_path=base / "m.csv",
                )


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


class PlayParserTests(unittest.TestCase):
    def test_play_parser_scenario_flags(self) -> None:
        args = build_play_parser().parse_args([])
        self.assertEqual(args.scenario_preset, "normal")
        args = build_play_parser().parse_args(
            ["--scenario-preset", "dense", "--obstacles", "3"]
        )
        self.assertEqual(args.scenario_preset, "dense")
        self.assertEqual(args.obstacles, 3)


class BehaviorCloningTests(unittest.TestCase):
    """Recording human driving and cloning it with supervised learning."""

    def test_recorder_drops_crashed_episodes(self) -> None:
        recorder = DemoRecorder()
        recorder.add(np.zeros(16, dtype=np.float32), 1)
        recorder.end_episode(crashed=True)
        recorder.add(np.ones(16, dtype=np.float32), 2)
        recorder.add(np.ones(16, dtype=np.float32), 3)
        recorder.end_episode(crashed=False)
        with tempfile.TemporaryDirectory() as directory:
            path = recorder.save(Path(directory) / "demo.npz")
            observations, actions = load_demos([path])
        self.assertEqual(list(actions), [2, 3])
        self.assertEqual(observations.shape, (2, 16))
        self.assertEqual(recorder.dropped_episodes, 1)

    def test_recorder_refuses_to_save_nothing(self) -> None:
        recorder = DemoRecorder()
        recorder.add(np.zeros(16, dtype=np.float32), 0)
        recorder.end_episode(crashed=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                recorder.save(Path(directory) / "demo.npz")

    def test_clone_learns_and_is_a_drop_in_dqn_policy(self) -> None:
        rng = np.random.default_rng(0)
        observations = rng.uniform(-1, 1, size=(400, 16)).astype(np.float32)
        actions = np.where(observations[:, 0] > 0.0, 2, 1).astype(np.int64)
        network, result = train_clone(
            observations, actions, epochs=40, batch_size=64, seed=1, verbose=False
        )
        self.assertGreater(result.train_accuracy, 0.9)
        with tempfile.TemporaryDirectory() as directory:
            path = save_clone(
                network,
                Path(directory) / "clone.npz",
                observation_size=16,
                action_size=5,
            )
            agent = DQNAgent.load(path, seed=0)
        sample = observations[:20]
        network_choice = list(np.argmax(np.asarray(network.forward(sample)), axis=1))
        agent_choice = [agent.act(row, explore=False) for row in sample]
        self.assertEqual(network_choice, agent_choice)

    def test_play_parser_has_record_flag(self) -> None:
        args = build_play_parser().parse_args([])
        self.assertIsNone(args.record)
        args = build_play_parser().parse_args(["--record", "demos/me.npz"])
        self.assertEqual(args.record, Path("demos/me.npz"))


class QValueTests(unittest.TestCase):
    def test_q_values_shape_and_greedy_consistency(self) -> None:
        agent = DQNAgent(DrivingEnv.observation_size, DrivingEnv.action_size, seed=0)
        env = DrivingEnv(scenario="traffic", seed=0)
        observation, _ = env.reset(seed=0)
        q = agent.q_values(observation)
        self.assertEqual(q.shape, (DrivingEnv.action_size,))
        self.assertEqual(int(np.argmax(q)), agent.act(observation, explore=False))


class FastFlowSafetyTests(unittest.TestCase):
    """Safety margins that must hold at the 65 mph traffic flow."""

    def test_entry_speed_assumes_leader_may_stop(self) -> None:
        # A car merging behind a leader that is mid-emergency-brake must
        # enter slowly enough to stop before the leader's halting point —
        # never at a speed that only works if the flow stays fast.
        env = DrivingEnv(scenario="traffic", seed=0)
        env.traffic = [TrafficCar(1, 180.0, 14.0)]
        entry = env._entry_speed(1, 160.0, env.config.speed_limit_mps)
        gap = 180.0 - 160.0 - env.config.car_length_m
        # Comfortable stop from the entry speed must fit within the gap
        # plus the leader's own emergency stopping distance.
        available = (gap - 3.0) + 14.0**2 / (2.0 * 9.0)
        self.assertLessEqual(entry**2 / (2.0 * 4.0), available + 1e-6)
        self.assertLess(entry, env.config.speed_limit_mps)

    def test_lane_change_rejects_fast_closing_follower_at_speed(self) -> None:
        env = DrivingEnv(scenario="traffic", seed=0)
        slow = TrafficCar(0, 40.0, 10.0, behavior="reactive", cruise_speed_mps=12.0)
        follower = TrafficCar(1, 20.0, env.config.speed_limit_mps)
        env.traffic = [slow, follower]
        # 20 m rear gap but ~19 m/s closing: merging would force the
        # follower into an emergency stop, so the change must be refused.
        self.assertFalse(env._lane_change_ok(slow, 1, 5.0))


class AutopilotHandoverTests(unittest.TestCase):
    """Randomized start states and the manual-mode autopilot plumbing."""

    def test_randomized_start_varies_and_stays_on_road(self) -> None:
        cfg = EnvConfig()
        limit = cfg.road_half_width_m - cfg.car_width_m / 2.0
        positions = set()
        for seed in range(12):
            env = DrivingEnv(scenario="traffic", seed=seed)
            env.reset(seed=seed, options={"randomize_start": True})
            self.assertLessEqual(abs(env.ego_x_m), limit)
            self.assertTrue(6.0 <= env.ego_speed_mps <= cfg.speed_limit_mps)
            self.assertFalse(env._has_collision())
            positions.add(round(env.ego_x_m, 3))
        self.assertGreater(len(positions), 6)

    def test_randomized_start_is_reproducible(self) -> None:
        env_a = DrivingEnv(scenario="traffic", seed=3)
        env_a.reset(seed=3, options={"randomize_start": True})
        env_b = DrivingEnv(scenario="traffic", seed=3)
        env_b.reset(seed=3, options={"randomize_start": True})
        self.assertEqual(env_a.ego_x_m, env_b.ego_x_m)
        self.assertEqual(env_a.ego_speed_mps, env_b.ego_speed_mps)

    def test_plain_reset_is_unchanged(self) -> None:
        env = DrivingEnv(scenario="traffic", seed=5)
        env.reset(seed=5)
        self.assertEqual(env.ego_x_m, env.lane_center(env.config.lane_count // 2))
        self.assertEqual(env.ego_speed_mps, 12.0)

    def test_train_parser_has_handover_flag(self) -> None:
        args = build_parser().parse_args([])
        self.assertFalse(args.handover)
        args = build_parser().parse_args(["--handover"])
        self.assertTrue(args.handover)

    def test_play_parser_has_autopilot_model(self) -> None:
        args = build_play_parser().parse_args([])
        self.assertEqual(args.autopilot_model, Path("models/autopilot.npz"))

    def test_handover_training_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _, records = train(
                episodes=3,
                env_config=replace(EnvConfig(), max_steps=40),
                seed=2,
                curriculum=False,
                handover=True,
                eval_every=0,
                eval_episodes=1,
                log_every=100,
                output_path=base / "m.npz",
                metrics_path=base / "m.csv",
            )
            self.assertEqual(len(records), 3)
            self.assertTrue((base / "m.npz").exists())


class DrivingGradeTests(unittest.TestCase):
    """The real-time report card: laws, courtesy, and physical clearances."""

    def _empty_env(self) -> DrivingEnv:
        return DrivingEnv(scenario="lane", seed=0)

    def test_riding_the_divider_is_penalized(self) -> None:
        env = self._empty_env()
        env.ego_x_m = env.lane_center(1) - env.config.lane_width_m / 2.0
        env.ego_lateral_speed_mps = 0.0
        env.ego_speed_mps = 20.0
        env.step(Action.MAINTAIN)
        self.assertLess(env.last_reward_terms["divider_straddle"], -0.5)

    def test_centered_driving_is_not_penalized(self) -> None:
        env = self._empty_env()
        env.ego_speed_mps = 20.0
        env.step(Action.MAINTAIN)
        self.assertEqual(env.last_reward_terms["divider_straddle"], 0.0)

    def test_brisk_crossing_is_not_penalized(self) -> None:
        env = self._empty_env()
        env.ego_speed_mps = 20.0
        env.ego_x_m = env.lane_center(1) + env.config.lane_width_m / 2.0
        env.ego_lateral_speed_mps = env.config.max_lateral_speed_mps
        env.step(Action.STEER_RIGHT)
        self.assertEqual(env.last_reward_terms["divider_straddle"], 0.0)

    def test_speeding_beyond_grace_is_penalized(self) -> None:
        env = self._empty_env()
        env.ego_speed_mps = env.config.max_speed_mps
        env.step(Action.MAINTAIN)
        self.assertLess(env.last_reward_terms["speeding"], -0.5)

    def test_at_the_limit_is_not_penalized(self) -> None:
        env = self._empty_env()
        env.ego_speed_mps = env.config.speed_limit_mps
        env.step(Action.MAINTAIN)
        self.assertEqual(env.last_reward_terms["speeding"], 0.0)

    def test_slight_touch_during_merge_is_a_collision(self) -> None:
        env = DrivingEnv(scenario="traffic", seed=0)
        env.traffic = [TrafficCar(1, env.config.car_length_m, env.ego_speed_mps)]
        self.assertTrue(env._has_collision())

    def test_side_brush_is_a_collision(self) -> None:
        env = DrivingEnv(scenario="traffic", seed=0)
        env.traffic = [TrafficCar(2, 0.0, env.ego_speed_mps)]
        env.ego_x_m = env.lane_center(2) - env.config.car_width_m
        self.assertTrue(env._has_collision())

    def test_clear_gap_is_not_a_collision(self) -> None:
        env = DrivingEnv(scenario="traffic", seed=0)
        env.traffic = [
            TrafficCar(1, env.config.car_length_m + 0.5, env.ego_speed_mps)
        ]
        self.assertFalse(env._has_collision())

    def test_cutting_off_a_follower_is_penalized(self) -> None:
        env = DrivingEnv(scenario="traffic", seed=0)
        env.ego_speed_mps = 20.0
        env.traffic = [TrafficCar(2, -8.0, 20.0)]
        env.ego_x_m = env.lane_center(2)  # ego lands here mid-merge
        env.step(Action.MAINTAIN)
        self.assertLess(env.last_reward_terms["cut_off"], 0.0)

    def test_merge_with_room_is_not_a_cut_off(self) -> None:
        env = DrivingEnv(scenario="traffic", seed=0)
        env.ego_speed_mps = 20.0
        env.traffic = [TrafficCar(2, -40.0, 20.0)]
        env.ego_x_m = env.lane_center(2)
        env.step(Action.MAINTAIN)
        self.assertEqual(env.last_reward_terms["cut_off"], 0.0)

    def test_threading_a_tight_gap_is_penalized_without_contact(self) -> None:
        env = DrivingEnv(scenario="traffic", seed=0)
        env.ego_speed_mps = 20.0
        env.traffic = [TrafficCar(2, 0.0, 20.0)]
        env.ego_x_m = env.lane_center(2) - env.config.car_width_m - 0.3
        env.step(Action.MAINTAIN)
        self.assertLess(env.last_reward_terms["near_miss"], 0.0)

    def test_grade_starts_at_100_and_drops_when_speeding(self) -> None:
        env = self._empty_env()
        self.assertEqual(env.grade_score, 100.0)
        for _ in range(30):
            env.ego_speed_mps = env.config.max_speed_mps
            env.step(Action.MAINTAIN)
        self.assertLess(env.grade_score, 95.0)

    def test_crash_zeroes_the_grade(self) -> None:
        env = DrivingEnv(scenario="traffic", seed=0)
        env.traffic = [TrafficCar(1, 2.0, 10.0)]
        env.step(Action.MAINTAIN)
        self.assertEqual(env.grade_score, 0.0)

    def test_grade_letters_map_sensibly(self) -> None:
        self.assertEqual(grade_letter(100.0), "A+")
        self.assertEqual(grade_letter(85.0), "B")
        self.assertEqual(grade_letter(50.0), "F")

    def test_info_reports_the_grade(self) -> None:
        env = self._empty_env()
        _, _, _, _, info = env.step(Action.MAINTAIN)
        self.assertIn("driving_grade", info)
        self.assertIn("grade_letter", info)


if __name__ == "__main__":
    unittest.main()