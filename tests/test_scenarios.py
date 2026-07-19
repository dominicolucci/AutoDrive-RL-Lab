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
