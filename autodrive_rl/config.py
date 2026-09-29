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
    max_speed_mps: float = 49.0   # ~110 mph flat-out
    target_speed_mps: float = 28.0
    speed_limit_mps: float = 29.0  # ~65 mph posted limit
    #: How stale an inattentive driver's view of the road is. At 0.9 s a
    #: follower closing at 20 m/s covers ~18 m before it starts reacting —
    #: harmless behind a car moving with the flow, dangerous behind one that
    #: has stopped. Only applies to drivers marked inattentive. 1.5 s is the
    #: figure commonly cited for a distracted driver, versus roughly 0.7 s for
    #: an alert one; at 20 m/s that is 30 m covered before the brake goes on.
    reaction_delay_s: float = 1.5

    #: Stopping in a live freeway lane is not a neutral choice — most
    #: interstates carry minimum-speed laws, and a vehicle immobilised in a
    #: travel lane is a reportable incident for a real autonomous fleet. Below
    #: `stall_speed_mps` for longer than `stall_grace_s`, the episode ends as a
    #: failure in its own right. The grace period is what keeps ordinary
    #: slowing — and stop-and-go behind a queue — legal.
    #: A stall is only counted when the road ahead is actually clear. Crawling
    #: because the car in front has slowed is ordinary driving; crawling with
    #: `stall_clear_gap_m` of empty lane ahead is obstructing traffic, which is
    #: the thing real minimum-speed laws exist to stop.
    stall_speed_mps: float = 2.0
    stall_grace_s: float = 2.0
    stall_clear_gap_m: float = 25.0
    traffic_min_speed_mps: float = 9.0
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
    #: Static hazards. Retained for the lane-keeping scenario and for anyone
    #: who wants them, but no traffic preset uses them any more: a permanent
    #: immovable block in a live freeway lane, with a queue stacked behind it,
    #: is not a situation a highway driver ever has to solve.
    obstacle_count: int = 0
    #: The realistic replacement. A car doing 6-9 m/s in a 29 m/s flow forces
    #: the same decision — plan ahead, find a gap, change lanes — without
    #: pretending a wall can appear mid-motorway.
    slow_vehicle_count: int = 0
    reactive_fraction: float = 0.0
    #: Whether traffic also approaches from behind. Off by default: turning it
    #: on changes what every existing result means, so the original presets stay
    #: frozen and only new ones opt in.
    rear_traffic: bool = False
    #: Fraction of drivers who perceive the road with a reaction delay rather
    #: than instantaneously. Defaults to 0.0, which reproduces the original
    #: behaviour exactly. Above zero, a slow or stopped ego becomes genuinely
    #: dangerous, because some followers notice too late to shed the closing
    #: speed.
    inattentive_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.traffic_count < 0:
            raise ValueError("traffic_count must be >= 0")
        if self.obstacle_count < 0:
            raise ValueError("obstacle_count must be >= 0")
        if self.slow_vehicle_count < 0:
            raise ValueError("slow_vehicle_count must be >= 0")
        if not 0.0 <= self.reactive_fraction <= 1.0:
            raise ValueError("reactive_fraction must be in [0, 1]")
        if not 0.0 <= self.inattentive_fraction <= 1.0:
            raise ValueError("inattentive_fraction must be in [0, 1]")


@dataclass(frozen=True)
class ScenarioRanges:
    """Inclusive sampling bounds for domain randomization."""

    traffic_count: tuple[int, int] = (4, 14)
    obstacle_count: tuple[int, int] = (0, 3)
    slow_vehicle_count: tuple[int, int] = (0, 0)
    reactive_fraction: tuple[float, float] = (0.0, 1.0)
    # Zero by default: widening this silently would change what every existing
    # 'random' training run means.
    inattentive_fraction: tuple[float, float] = (0.0, 0.0)
    rear_traffic: bool = False


#: World model v2 (2026-09-28). Static obstacles were removed from every
#: traffic preset and replaced with slow-moving vehicles; see `BENCHMARK.md`
#: for the v1 numbers, which describe a different simulation and are kept only
#: as history.
SCENARIO_PRESETS: dict[str, ScenarioSpec] = {
    "sparse": ScenarioSpec(traffic_count=4, reactive_fraction=0.0),
    "normal": ScenarioSpec(traffic_count=9, reactive_fraction=0.0),
    "dense": ScenarioSpec(
        traffic_count=16, slow_vehicle_count=2, reactive_fraction=0.5
    ),
    # Same world as "dense", except traffic also arrives from behind and most
    # of those drivers react late. A policy that survives by crawling gets
    # rear-ended here, which the other three presets cannot express.
    "unforgiving": ScenarioSpec(
        traffic_count=16, slow_vehicle_count=2, reactive_fraction=0.5,
        rear_traffic=True, inattentive_fraction=0.6,
    ),
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
        slow_vehicle_count=int(
            rng.integers(
                ranges.slow_vehicle_count[0], ranges.slow_vehicle_count[1] + 1
            )
        ),
        reactive_fraction=float(rng.uniform(*ranges.reactive_fraction)),
        inattentive_fraction=float(rng.uniform(*ranges.inattentive_fraction)),
        rear_traffic=ranges.rear_traffic,
    )


def resolve_scenario(
    preset: str,
    *,
    traffic: int | None = None,
    obstacles: int | None = None,
    slow_vehicles: int | None = None,
    reactive: float | None = None,
    inattentive: float | None = None,
    rear_traffic: bool | None = None,
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
        slow_vehicle_count=(
            base.slow_vehicle_count if slow_vehicles is None else slow_vehicles
        ),
        reactive_fraction=base.reactive_fraction if reactive is None else reactive,
        inattentive_fraction=(
            base.inattentive_fraction if inattentive is None else inattentive
        ),
        rear_traffic=base.rear_traffic if rear_traffic is None else rear_traffic,
    )