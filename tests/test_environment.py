from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from autodrive_rl.config import EnvConfig
from autodrive_rl.environment import Action, DrivingEnv, TrafficCar
from autodrive_rl.heuristic import HeuristicDriver


class DrivingEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = replace(EnvConfig(), max_steps=50)

    def test_observation_shape_range_and_types(self) -> None:
        env = DrivingEnv(self.config, seed=11)
        observation, info = env.reset(seed=11)
        self.assertEqual(observation.shape, (env.observation_size,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(observation)))
        self.assertTrue(np.all(observation >= -1.0))
        self.assertTrue(np.all(observation <= 1.0))
        self.assertFalse(info["collision"])

    def test_seed_reproduces_traffic(self) -> None:
        first = DrivingEnv(self.config, seed=22)
        second = DrivingEnv(self.config, seed=22)
        first_observation, _ = first.reset(seed=22)
        second_observation, _ = second.reset(seed=22)
        first_cars = [(car.lane, car.y_m, car.speed_mps) for car in first.traffic]
        second_cars = [(car.lane, car.y_m, car.speed_mps) for car in second.traffic]
        np.testing.assert_allclose(first_observation, second_observation)
        self.assertEqual(first_cars, second_cars)

    def test_acceleration_and_steering_change_vehicle_state(self) -> None:
        env = DrivingEnv(self.config, scenario="lane", seed=1)
        env.reset(seed=1)
        initial_speed = env.ego_speed_mps
        env.step(Action.ACCELERATE)
        self.assertGreater(env.ego_speed_mps, initial_speed)
        initial_x = env.ego_x_m
        for _ in range(5):
            env.step(Action.STEER_LEFT)
        self.assertLess(env.ego_x_m, initial_x)

    def test_collision_terminates_and_has_large_penalty(self) -> None:
        env = DrivingEnv(self.config, scenario="lane", seed=2)
        env.reset(seed=2)
        env.traffic = [TrafficCar(lane=env.current_lane, y_m=0.0, speed_mps=12.0)]
        _, reward, terminated, _, info = env.step(Action.MAINTAIN)
        self.assertTrue(terminated)
        self.assertTrue(info["collision"])
        self.assertLess(reward, -480.0)

    def test_leaving_road_terminates(self) -> None:
        env = DrivingEnv(self.config, scenario="lane", seed=3)
        env.reset(seed=3)
        env.ego_x_m = env.config.road_half_width_m
        _, reward, terminated, _, info = env.step(Action.MAINTAIN)
        self.assertTrue(terminated)
        self.assertTrue(info["off_road"])
        self.assertLess(reward, -330.0)

    def test_dangerous_following_has_dense_penalty_before_collision(self) -> None:
        env = DrivingEnv(self.config, scenario="lane", seed=8)
        env.reset(seed=8)
        env.ego_speed_mps = 22.0
        env.traffic = [TrafficCar(lane=env.current_lane, y_m=16.0, speed_mps=8.0)]
        _, _, terminated, _, _ = env.step(Action.MAINTAIN)
        self.assertFalse(terminated)
        self.assertLess(env.last_reward_terms["unsafe_following"], -1.0)

    def test_stopped_car_cannot_farm_positive_centering_reward(self) -> None:
        env = DrivingEnv(self.config, scenario="lane", seed=6)
        env.reset(seed=6)
        env.ego_speed_mps = 0.0
        _, reward, _, _, _ = env.step(Action.MAINTAIN)
        self.assertLess(reward, 0.0)

    def test_heuristic_returns_valid_action(self) -> None:
        env = DrivingEnv(self.config, seed=4)
        driver = HeuristicDriver()
        for _ in range(20):
            action = driver.act(env)
            self.assertIn(action, range(env.action_size))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break


if __name__ == "__main__":
    unittest.main()
