"""Configuration objects for the environment and DQN agent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
