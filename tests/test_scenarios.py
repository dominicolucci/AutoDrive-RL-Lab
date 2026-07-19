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
from autodrive_rl.dqn import DQNAgent
from autodrive_rl.environment import Action, DrivingEnv, TrafficCar
from autodrive_rl.train import build_parser, train


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


if __name__ == "__main__":
    unittest.main()
