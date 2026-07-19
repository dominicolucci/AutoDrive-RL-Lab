from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from autodrive_rl.config import (
    SCENARIO_PRESETS,
    EnvConfig,
    ScenarioRanges,
    ScenarioSpec,
    resolve_scenario,
    sample_scenario,
)
from autodrive_rl.environment import DrivingEnv, TrafficCar


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


if __name__ == "__main__":
    unittest.main()
