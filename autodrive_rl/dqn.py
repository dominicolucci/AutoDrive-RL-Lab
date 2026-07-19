"""A small Deep Q-Network implemented directly with NumPy.

The code is intentionally explicit. There is no hidden trainer: replay memory,
epsilon-greedy exploration, the target network, Huber loss, backpropagation,
and Adam updates are all visible in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import DQNConfig


class ReplayBuffer:
    """Fixed-size circular storage for observed transitions."""

    def __init__(self, capacity: int, observation_size: int) -> None:
        self.capacity = int(capacity)
        self.observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.dones[index] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size > self.size:
            raise ValueError("cannot sample more transitions than the buffer contains")
        indices = rng.integers(0, self.size, size=batch_size)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
        )


class MLP:
    """Fully connected ReLU network with a linear output layer."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Iterable[int] = (64, 64),
        *,
        seed: int | None = None,
    ) -> None:
        layer_sizes = [input_size, *hidden_sizes, output_size]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        rng = np.random.default_rng(seed)

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(
                rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32)
            )
            self.biases.append(np.zeros(fan_out, dtype=np.float32))

    def forward(
        self, x: np.ndarray, *, cache: bool = False
    ) -> np.ndarray | tuple[np.ndarray, tuple[list[np.ndarray], list[np.ndarray]]]:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]

        activations = [x]
        preactivations: list[np.ndarray] = []
        activation = x
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = activation @ weight + bias
            preactivations.append(z)
            if layer_index < len(self.weights) - 1:
                activation = np.maximum(z, 0.0)
            else:
                activation = z
            activations.append(activation)

        if cache:
            return activation, (activations, preactivations)
        return activation

    def backward(
        self,
        cache: tuple[list[np.ndarray], list[np.ndarray]],
        output_gradient: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations, preactivations = cache
        grad_weights: list[np.ndarray] = [np.empty_like(weight) for weight in self.weights]
        grad_biases: list[np.ndarray] = [np.empty_like(bias) for bias in self.biases]
        grad_z = output_gradient.astype(np.float32, copy=False)

        for layer_index in range(len(self.weights) - 1, -1, -1):
            grad_weights[layer_index] = activations[layer_index].T @ grad_z
            grad_biases[layer_index] = grad_z.sum(axis=0)
            if layer_index > 0:
                grad_activation = grad_z @ self.weights[layer_index].T
                grad_z = grad_activation * (preactivations[layer_index - 1] > 0.0)

        return grad_weights, grad_biases

    def copy_from(self, other: "MLP") -> None:
        for destination, source in zip(self.weights, other.weights):
            destination[...] = source
        for destination, source in zip(self.biases, other.biases):
            destination[...] = source


class Adam:
    """Minimal Adam optimizer for the network's weight and bias arrays."""

    def __init__(
        self,
        parameters: list[np.ndarray],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first_moments = [np.zeros_like(parameter) for parameter in parameters]
        self.second_moments = [np.zeros_like(parameter) for parameter in parameters]
        self.step_count = 0

    def step(self, gradients: list[np.ndarray]) -> None:
        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1**self.step_count
        bias_correction2 = 1.0 - self.beta2**self.step_count

        for parameter, gradient, first, second in zip(
            self.parameters, gradients, self.first_moments, self.second_moments
        ):
            first *= self.beta1
            first += (1.0 - self.beta1) * gradient
            second *= self.beta2
            second += (1.0 - self.beta2) * np.square(gradient)
            first_hat = first / bias_correction1
            second_hat = second / bias_correction2
            parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + self.epsilon)


@dataclass(frozen=True)
class LearnResult:
    loss: float
    mean_q: float
    mean_target: float


class DQNAgent:
    """Deep Q-learning agent with replay and a periodically copied target net."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: DQNConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.observation_size = observation_size
        self.action_size = action_size
        self.config = config or DQNConfig()
        self.rng = np.random.default_rng(seed)
        self.online = MLP(
            observation_size,
            action_size,
            self.config.hidden_sizes,
            seed=seed,
        )
        self.target = MLP(
            observation_size,
            action_size,
            self.config.hidden_sizes,
            seed=None if seed is None else seed + 1,
        )
        self.target.copy_from(self.online)
        parameters: list[np.ndarray] = []
        for weight, bias in zip(self.online.weights, self.online.biases):
            parameters.extend([weight, bias])
        self.optimizer = Adam(parameters, self.config.learning_rate)
        self.replay = ReplayBuffer(self.config.replay_capacity, observation_size)
        self.environment_steps = 0
        self.gradient_steps = 0

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.environment_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def act(self, observation: np.ndarray, *, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.action_size))
        q_values = self.online.forward(observation)
        return int(np.argmax(np.asarray(q_values)[0]))

    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.add(observation, action, reward, next_observation, done)
        self.environment_steps += 1

    def learn(self) -> LearnResult | None:
        cfg = self.config
        if self.replay.size < max(cfg.warmup_steps, cfg.batch_size):
            return None

        observations, actions, rewards, next_observations, dones = self.replay.sample(
            cfg.batch_size, self.rng
        )
        # Double DQN: the online network selects the next action while the
        # slower target network evaluates it. This reduces optimistic Q-value
        # feedback without hiding any part of the learning loop.
        next_online_q = np.asarray(self.online.forward(next_observations))
        next_actions = np.argmax(next_online_q, axis=1)
        next_target_q = np.asarray(self.target.forward(next_observations))
        bootstrap_values = next_target_q[np.arange(cfg.batch_size), next_actions]
        targets = rewards + cfg.gamma * (1.0 - dones) * bootstrap_values

        q_values, cache = self.online.forward(observations, cache=True)
        q_values = np.asarray(q_values)
        selected_q = q_values[np.arange(cfg.batch_size), actions]
        errors = selected_q - targets
        absolute_errors = np.abs(errors)
        losses = np.where(
            absolute_errors <= 1.0,
            0.5 * np.square(errors),
            absolute_errors - 0.5,
        )

        selected_gradient = np.where(absolute_errors <= 1.0, errors, np.sign(errors))
        selected_gradient /= cfg.batch_size
        output_gradient = np.zeros_like(q_values, dtype=np.float32)
        output_gradient[np.arange(cfg.batch_size), actions] = selected_gradient

        grad_weights, grad_biases = self.online.backward(cache, output_gradient)
        gradients: list[np.ndarray] = []
        for grad_weight, grad_bias in zip(grad_weights, grad_biases):
            gradients.extend([grad_weight, grad_bias])
        self._clip_gradients(gradients, cfg.gradient_clip_norm)
        self.optimizer.step(gradients)

        self.gradient_steps += 1
        if self.gradient_steps % cfg.target_update_steps == 0:
            self.target.copy_from(self.online)

        return LearnResult(
            loss=float(np.mean(losses)),
            mean_q=float(np.mean(selected_q)),
            mean_target=float(np.mean(targets)),
        )

    @staticmethod
    def _clip_gradients(gradients: list[np.ndarray], maximum_norm: float) -> None:
        total_squared = sum(float(np.sum(np.square(gradient))) for gradient in gradients)
        total_norm = np.sqrt(total_squared)
        if total_norm > maximum_norm:
            scale = maximum_norm / (total_norm + 1e-8)
            for gradient in gradients:
                gradient *= scale

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "observation_size": np.array(self.observation_size, dtype=np.int64),
            "action_size": np.array(self.action_size, dtype=np.int64),
            "hidden_sizes": np.asarray(self.config.hidden_sizes, dtype=np.int64),
            "environment_steps": np.array(self.environment_steps, dtype=np.int64),
        }
        for index, (weight, bias) in enumerate(zip(self.online.weights, self.online.biases)):
            payload[f"weight_{index}"] = weight
            payload[f"bias_{index}"] = bias
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path, *, seed: int | None = None) -> "DQNAgent":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            observation_size = int(data["observation_size"])
            action_size = int(data["action_size"])
            hidden_sizes = tuple(int(value) for value in data["hidden_sizes"])
            config = DQNConfig(hidden_sizes=hidden_sizes)
            agent = cls(observation_size, action_size, config, seed=seed)
            for index, (weight, bias) in enumerate(
                zip(agent.online.weights, agent.online.biases)
            ):
                weight[...] = data[f"weight_{index}"]
                bias[...] = data[f"bias_{index}"]
            agent.environment_steps = int(data["environment_steps"])
        agent.target.copy_from(agent.online)
        return agent
