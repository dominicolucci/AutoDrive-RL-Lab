"""Configuration objects for the environment and DQN agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    """Physical and episode settings for the highway simulation."""

    lane_count: int = 3
    lane_width_m: float = 3.7
    car_width_m: float = 1.9
    car_length_m: float = 4.6
    max_speed_mps: float = 30.0
    target_speed_mps: float = 24.0
    max_lateral_speed_mps: float = 2.6
    acceleration_mps2: float = 3.2
    braking_mps2: float = 6.5
    rolling_drag_mps2: float = 0.12
    steering_response: float = 5.0
    dt_seconds: float = 0.10
    sensor_range_m: float = 120.0
    traffic_count: int = 9
    max_steps: int = 900

    @property
    def road_width_m(self) -> float:
        return self.lane_count * self.lane_width_m

    @property
    def road_half_width_m(self) -> float:
        return self.road_width_m / 2.0


@dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters for the deliberately small, educational DQN."""

    hidden_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 64
    replay_capacity: int = 50_000
    warmup_steps: int = 750
    target_update_steps: int = 750
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 90_000
    gradient_clip_norm: float = 10.0

