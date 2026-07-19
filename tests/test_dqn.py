from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from autodrive_rl.config import DQNConfig
from autodrive_rl.dqn import DQNAgent, MLP, ReplayBuffer


class DQNTests(unittest.TestCase):
    def test_network_forward_shape(self) -> None:
        network = MLP(16, 5, (32, 16), seed=1)
        batch = np.zeros((7, 16), dtype=np.float32)
        output = network.forward(batch)
        self.assertEqual(np.asarray(output).shape, (7, 5))

    def test_replay_buffer_wraps_and_samples(self) -> None:
        buffer = ReplayBuffer(capacity=5, observation_size=3)
        for index in range(8):
            observation = np.full(3, index, dtype=np.float32)
            buffer.add(observation, index % 2, float(index), observation + 1, False)
        self.assertEqual(buffer.size, 5)
        sample = buffer.sample(4, np.random.default_rng(2))
        self.assertEqual(sample[0].shape, (4, 3))

    def test_learning_updates_online_network(self) -> None:
        config = DQNConfig(
            hidden_sizes=(16,),
            batch_size=8,
            replay_capacity=100,
            warmup_steps=8,
            target_update_steps=2,
        )
        agent = DQNAgent(4, 3, config, seed=3)
        rng = np.random.default_rng(4)
        for _ in range(12):
            observation = rng.normal(size=4).astype(np.float32)
            next_observation = rng.normal(size=4).astype(np.float32)
            agent.observe(observation, int(rng.integers(0, 3)), 1.0, next_observation, False)
        before = [weight.copy() for weight in agent.online.weights]
        result = agent.learn()
        self.assertIsNotNone(result)
        self.assertTrue(np.isfinite(result.loss))  # type: ignore[union-attr]
        self.assertTrue(any(not np.array_equal(old, new) for old, new in zip(before, agent.online.weights)))

    def test_saved_agent_reloads_same_policy_values(self) -> None:
        agent = DQNAgent(4, 3, DQNConfig(hidden_sizes=(8,)), seed=5)
        observation = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
        expected = np.asarray(agent.online.forward(observation))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.npz"
            agent.save(path)
            loaded = DQNAgent.load(path, seed=6)
            actual = np.asarray(loaded.online.forward(observation))
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()

