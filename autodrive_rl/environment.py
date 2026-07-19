"""A compact, Gym-style three-lane highway environment.

The environment intentionally uses numerical sensors instead of camera pixels.
That keeps the first project focused on the reinforcement-learning loop: state,
action, transition, reward, replay, and policy improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from .config import EnvConfig, ScenarioSpec


class Action(IntEnum):
    """The five discrete controls available to the learning agent."""

    MAINTAIN = 0
    ACCELERATE = 1
    BRAKE = 2
    STEER_LEFT = 3
    STEER_RIGHT = 4


ACTION_NAMES = {
    Action.MAINTAIN: "maintain",
    Action.ACCELERATE: "accelerate",
    Action.BRAKE: "brake",
    Action.STEER_LEFT: "steer left",
    Action.STEER_RIGHT: "steer right",
}

OBSTACLE_WINDOW_M = 30.0
OBSTACLE_EGO_CLEAR_M = 60.0

REACTIVE_TIME_HEADWAY_S = 1.5
REACTIVE_BRAKE_MPS2 = 4.0
REACTIVE_ACCEL_MPS2 = 2.0
REACTIVE_MIN_GAP_M = 2.0

LANE_CHANGE_DURATION_S = 1.2
LANE_CHANGE_PROBABILITY = 0.005
LANE_CHANGE_MIN_FRONT_GAP_M = 15.0
LANE_CHANGE_MIN_REAR_GAP_M = 12.0


@dataclass
class TrafficCar:
    """A traffic car (or static obstacle) represented relative to the ego car."""

    lane: int
    y_m: float
    speed_mps: float
    color_index: int = 0
    behavior: str = "cruiser"  # "cruiser" | "reactive" | "obstacle"
    cruise_speed_mps: float | None = None
    target_lane: int | None = None
    lane_change_progress: float = 0.0

    def __post_init__(self) -> None:
        if self.cruise_speed_mps is None:
            self.cruise_speed_mps = self.speed_mps


class DrivingEnv:
    """Straight-highway RL environment with moving traffic.

    ``reset`` and ``step`` follow Gymnasium's return convention, but Gymnasium
    itself is not required. This makes the learning loop easy to inspect and
    lets the MVP run with only NumPy installed.
    """

    observation_size = 16
    action_size = len(Action)

    def __init__(
        self,
        config: EnvConfig | None = None,
        *,
        scenario: str = "traffic",
        seed: int | None = None,
        scenario_spec: ScenarioSpec | None = None,
    ) -> None:
        if scenario not in {"lane", "traffic"}:
            raise ValueError("scenario must be 'lane' or 'traffic'")
        self.config = config or EnvConfig()
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.traffic: list[TrafficCar] = []
        self.ego_x_m = 0.0
        self.ego_lateral_speed_mps = 0.0
        self.ego_speed_mps = 0.0
        self.distance_m = 0.0
        self.steps = 0
        self.previous_action = Action.MAINTAIN
        self.last_reward_terms: dict[str, float] = {}
        self.scenario_spec = scenario_spec
        self.reset(seed=seed)

    @property
    def lane_centers_m(self) -> np.ndarray:
        cfg = self.config
        leftmost = -cfg.road_half_width_m + cfg.lane_width_m / 2.0
        return leftmost + np.arange(cfg.lane_count) * cfg.lane_width_m

    @property
    def current_lane(self) -> int:
        return int(np.argmin(np.abs(self.lane_centers_m - self.ego_x_m)))

    def lane_center(self, lane: int) -> float:
        if not 0 <= lane < self.config.lane_count:
            raise ValueError(f"invalid lane {lane}")
        return float(self.lane_centers_m[lane])

    def traffic_x_m(self, car: TrafficCar) -> float:
        """The car's actual lateral position, mid-lane-change aware."""

        x = self.lane_center(car.lane)
        if car.target_lane is not None:
            target = self.lane_center(car.target_lane)
            x += (target - x) * car.lane_change_progress
        return x

    def traffic_lane(self, car: TrafficCar) -> int:
        """The lane whose center is nearest the car's actual position."""

        return int(np.argmin(np.abs(self.lane_centers_m - self.traffic_x_m(car))))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.ego_x_m = self.lane_center(self.config.lane_count // 2)
        self.ego_lateral_speed_mps = 0.0
        self.ego_speed_mps = 12.0
        self.distance_m = 0.0
        self.steps = 0
        self.previous_action = Action.MAINTAIN
        self.last_reward_terms = {}
        self.traffic = []
        if self.scenario == "traffic":
            self._spawn_initial_traffic()

        observation = self._observation()
        return observation, self._info(collision=False, off_road=False)

    def step(
        self, action: int | Action
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        try:
            action = Action(int(action))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"action must be an integer from 0 to {self.action_size - 1}") from exc

        cfg = self.config
        acceleration = 0.0
        target_lateral_speed = 0.0

        if action == Action.ACCELERATE:
            acceleration = cfg.acceleration_mps2
        elif action == Action.BRAKE:
            acceleration = -cfg.braking_mps2
        elif action == Action.STEER_LEFT:
            target_lateral_speed = -cfg.max_lateral_speed_mps
        elif action == Action.STEER_RIGHT:
            target_lateral_speed = cfg.max_lateral_speed_mps

        acceleration -= cfg.rolling_drag_mps2
        self.ego_speed_mps = float(
            np.clip(
                self.ego_speed_mps + acceleration * cfg.dt_seconds,
                0.0,
                cfg.max_speed_mps,
            )
        )

        steering_alpha = min(1.0, cfg.steering_response * cfg.dt_seconds)
        self.ego_lateral_speed_mps += steering_alpha * (
            target_lateral_speed - self.ego_lateral_speed_mps
        )
        self.ego_x_m += self.ego_lateral_speed_mps * cfg.dt_seconds

        forward_step = self.ego_speed_mps * cfg.dt_seconds
        self.distance_m += forward_step
        self._update_traffic()
        for car in self.traffic:
            car.y_m += (car.speed_mps - self.ego_speed_mps) * cfg.dt_seconds
        self._recycle_traffic()

        self.steps += 1
        collision = self._has_collision()
        off_road = self._is_off_road()
        terminated = collision or off_road
        truncated = self.steps >= cfg.max_steps

        reward = self._reward(
            action=action,
            forward_step=forward_step,
            collision=collision,
            off_road=off_road,
        )
        self.previous_action = action
        observation = self._observation()
        return observation, reward, terminated, truncated, self._info(collision, off_road)

    def sensor_snapshot(self) -> dict[str, np.ndarray | float | int]:
        """Return human-readable sensor values in meters and meters/second."""

        cfg = self.config
        front_gaps = np.full(cfg.lane_count, cfg.sensor_range_m, dtype=np.float32)
        rear_gaps = np.full(cfg.lane_count, cfg.sensor_range_m, dtype=np.float32)
        front_relative = np.zeros(cfg.lane_count, dtype=np.float32)
        rear_relative = np.zeros(cfg.lane_count, dtype=np.float32)

        half_length = cfg.car_length_m
        for car in self.traffic:
            lane = self.traffic_lane(car)
            if car.y_m >= 0.0:
                gap = max(0.0, car.y_m - half_length)
                if gap < front_gaps[lane] and gap <= cfg.sensor_range_m:
                    front_gaps[lane] = gap
                    front_relative[lane] = car.speed_mps - self.ego_speed_mps
            else:
                gap = max(0.0, -car.y_m - half_length)
                if gap < rear_gaps[lane] and gap <= cfg.sensor_range_m:
                    rear_gaps[lane] = gap
                    rear_relative[lane] = car.speed_mps - self.ego_speed_mps

        lane = self.current_lane
        lane_offset = self.ego_x_m - self.lane_center(lane)
        return {
            "front_gaps_m": front_gaps,
            "rear_gaps_m": rear_gaps,
            "front_relative_speeds_mps": front_relative,
            "rear_relative_speeds_mps": rear_relative,
            "lane": lane,
            "lane_offset_m": float(lane_offset),
        }

    def _observation(self) -> np.ndarray:
        cfg = self.config
        sensors = self.sensor_snapshot()
        safe_center_limit = cfg.road_half_width_m - cfg.car_width_m / 2.0
        lane_offset_scale = cfg.lane_width_m / 2.0

        observation = np.concatenate(
            [
                np.array(
                    [
                        self.ego_x_m / safe_center_limit,
                        self.ego_lateral_speed_mps / cfg.max_lateral_speed_mps,
                        self.ego_speed_mps / cfg.max_speed_mps,
                        float(sensors["lane_offset_m"]) / lane_offset_scale,
                    ],
                    dtype=np.float32,
                ),
                np.asarray(sensors["front_gaps_m"], dtype=np.float32) / cfg.sensor_range_m,
                np.asarray(sensors["rear_gaps_m"], dtype=np.float32) / cfg.sensor_range_m,
                np.asarray(sensors["front_relative_speeds_mps"], dtype=np.float32)
                / cfg.max_speed_mps,
                np.asarray(sensors["rear_relative_speeds_mps"], dtype=np.float32)
                / cfg.max_speed_mps,
            ]
        )
        return np.clip(observation, -1.0, 1.0).astype(np.float32)

    def _reward(
        self,
        *,
        action: Action,
        forward_step: float,
        collision: bool,
        off_road: bool,
    ) -> float:
        cfg = self.config
        sensors = self.sensor_snapshot()
        lane_offset = abs(float(sensors["lane_offset_m"]))
        center_score = max(0.0, 1.0 - lane_offset / (cfg.lane_width_m / 2.0))

        progress = forward_step / (cfg.max_speed_mps * cfg.dt_seconds)
        speed = 0.20 * min(self.ego_speed_mps / cfg.target_speed_mps, 1.0)
        # Centering is useful only while moving. Without this gate, a stopped
        # car could earn positive reward forever simply by sitting in a lane.
        movement_fraction = min(self.ego_speed_mps / 6.0, 1.0)
        lane_centering = 0.12 * center_score * movement_fraction
        living_cost = -0.04

        current_gap = float(np.asarray(sensors["front_gaps_m"])[self.current_lane])
        current_relative_speed = float(
            np.asarray(sensors["front_relative_speeds_mps"])[self.current_lane]
        )
        desired_gap = max(10.0, 1.25 * self.ego_speed_mps)
        unsafe_following = 0.0
        if current_gap < desired_gap:
            unsafe_following = -2.0 * (1.0 - current_gap / desired_gap)
        closing_speed = max(0.0, -current_relative_speed)
        if closing_speed > 0.1:
            time_to_collision = current_gap / closing_speed
            if time_to_collision < 4.0:
                unsafe_following += -2.0 * (1.0 - time_to_collision / 4.0)

        control_cost = -0.015 if action in {Action.STEER_LEFT, Action.STEER_RIGHT} else 0.0
        unsafe_lane_change = 0.0
        if action in {Action.STEER_LEFT, Action.STEER_RIGHT}:
            direction = -1 if action == Action.STEER_LEFT else 1
            target_lane = self.current_lane + direction
            if not 0 <= target_lane < cfg.lane_count:
                unsafe_lane_change = -2.5
            else:
                target_front = float(np.asarray(sensors["front_gaps_m"])[target_lane])
                target_rear = float(np.asarray(sensors["rear_gaps_m"])[target_lane])
                front_risk = max(0.0, 1.0 - target_front / 14.0)
                rear_risk = max(0.0, 1.0 - target_rear / 12.0)
                unsafe_lane_change = -2.0 * max(front_risk, rear_risk)

        safe_center_limit = cfg.road_half_width_m - cfg.car_width_m / 2.0
        edge_fraction = abs(self.ego_x_m) / safe_center_limit
        road_edge = 0.0
        if edge_fraction > 0.82:
            road_edge = -2.0 * min(1.0, (edge_fraction - 0.82) / 0.18)

        terminal = -500.0 if collision else (-350.0 if off_road else 0.0)

        self.last_reward_terms = {
            "progress": progress,
            "speed": speed,
            "lane_centering": lane_centering,
            "unsafe_following": unsafe_following,
            "unsafe_lane_change": unsafe_lane_change,
            "road_edge": road_edge,
            "control_cost": control_cost,
            "living_cost": living_cost,
            "terminal": terminal,
        }
        return float(sum(self.last_reward_terms.values()))

    def _spawn_initial_traffic(self) -> None:
        cfg = self.config
        spec = self.scenario_spec
        count = cfg.traffic_count if spec is None else spec.traffic_count
        reactive_fraction = 0.0 if spec is None else spec.reactive_fraction
        self._spawn_obstacles()
        moving_target = count + len(self.traffic)
        attempts = 0
        while len(self.traffic) < moving_target and attempts < 500:
            attempts += 1
            lane = int(self.rng.integers(0, cfg.lane_count))
            y_m = float(self.rng.uniform(22.0, cfg.sensor_range_m + 55.0))
            if any(car.lane == lane and abs(car.y_m - y_m) < 24.0 for car in self.traffic):
                continue
            speed = float(self.rng.uniform(9.0, 26.0))
            color_index = int(self.rng.integers(0, 6))
            behavior = "cruiser"
            if reactive_fraction > 0.0 and self.rng.random() < reactive_fraction:
                behavior = "reactive"
            self.traffic.append(TrafficCar(lane, y_m, speed, color_index, behavior=behavior))

    def _spawn_obstacles(self) -> None:
        spec = self.scenario_spec
        if spec is None or spec.obstacle_count == 0:
            return
        cfg = self.config
        ego_lane = self.current_lane
        placed = 0
        attempts = 0
        while placed < spec.obstacle_count and attempts < 200:
            attempts += 1
            lane = int(self.rng.integers(0, cfg.lane_count))
            y_m = float(self.rng.uniform(30.0, cfg.sensor_range_m + 55.0))
            if lane == ego_lane and y_m < OBSTACLE_EGO_CLEAR_M:
                continue
            if not self._obstacle_position_ok(lane, y_m):
                continue
            self.traffic.append(TrafficCar(lane, y_m, 0.0, behavior="obstacle"))
            placed += 1

    def _obstacle_position_ok(self, lane: int, y_m: float) -> bool:
        cfg = self.config
        obstacles = [(car.lane, car.y_m) for car in self.traffic if car.behavior == "obstacle"]
        for other_lane, other_y in obstacles:
            if other_lane == lane and abs(other_y - y_m) < 24.0:
                return False
        # Checking only the candidate's own window is not enough: two
        # obstacles can each individually clear an existing obstacle's
        # window yet jointly box it in (a chain, e.g. A-B close, B-C close,
        # A-C far apart). Re-check every obstacle's window with the
        # candidate hypothetically added so no lane position ever ends up
        # fully blocked.
        candidates = obstacles + [(lane, y_m)]
        for center_lane, center_y in candidates:
            blocked = {
                other_lane
                for other_lane, other_y in candidates
                if abs(other_y - center_y) < OBSTACLE_WINDOW_M
            }
            if len(blocked) >= cfg.lane_count:
                return False
        return True

    def _recycle_traffic(self) -> None:
        if self.scenario != "traffic":
            return
        cfg = self.config
        for car in self.traffic:
            if -45.0 <= car.y_m <= cfg.sensor_range_m + 90.0:
                continue
            if car.behavior == "obstacle":
                for _ in range(50):
                    lane = int(self.rng.integers(0, cfg.lane_count))
                    y_m = float(
                        self.rng.uniform(cfg.sensor_range_m + 20.0, cfg.sensor_range_m + 80.0)
                    )
                    if self._obstacle_position_ok(lane, y_m):
                        car.lane = lane
                        car.y_m = y_m
                        break
                continue
            car.lane = self._least_crowded_spawn_lane()
            car.y_m = float(self.rng.uniform(cfg.sensor_range_m + 20.0, cfg.sensor_range_m + 80.0))
            car.speed_mps = float(self.rng.uniform(9.0, 26.0))
            car.color_index = int(self.rng.integers(0, 6))
            car.cruise_speed_mps = car.speed_mps
            car.target_lane = None
            car.lane_change_progress = 0.0

    def _least_crowded_spawn_lane(self) -> int:
        cfg = self.config
        scores: list[tuple[float, int]] = []
        spawn_y = cfg.sensor_range_m + 45.0
        for lane in range(cfg.lane_count):
            nearest = min(
                (abs(car.y_m - spawn_y) for car in self.traffic if car.lane == lane),
                default=999.0,
            )
            scores.append((nearest + float(self.rng.uniform(0.0, 3.0)), lane))
        return max(scores)[1]

    def _update_traffic(self) -> None:
        cfg = self.config
        for car in self.traffic:
            if car.behavior == "reactive":
                self._update_reactive(car)
            elif car.behavior == "cruiser":
                self._update_cruiser(car)
            if car.target_lane is not None:
                car.lane_change_progress += cfg.dt_seconds / LANE_CHANGE_DURATION_S
                if car.lane_change_progress >= 1.0:
                    car.lane = car.target_lane
                    car.target_lane = None
                    car.lane_change_progress = 0.0

    def _update_reactive(self, car: TrafficCar) -> None:
        cfg = self.config
        lane = self.traffic_lane(car)
        gap, leader_speed = self._front_gap_for(car, lane)
        assert car.cruise_speed_mps is not None
        closing = max(0.0, car.speed_mps - leader_speed)
        threshold = (
            REACTIVE_MIN_GAP_M
            + REACTIVE_TIME_HEADWAY_S * car.speed_mps
            + closing**2 / (2.0 * REACTIVE_BRAKE_MPS2)
        )
        if gap < threshold:
            car.speed_mps = max(
                0.0, car.speed_mps - REACTIVE_BRAKE_MPS2 * cfg.dt_seconds
            )
        elif car.speed_mps < car.cruise_speed_mps:
            car.speed_mps = min(
                car.cruise_speed_mps,
                car.speed_mps + REACTIVE_ACCEL_MPS2 * cfg.dt_seconds,
            )
        if (
            car.target_lane is None
            and car.speed_mps < 0.8 * car.cruise_speed_mps
            and self.rng.random() < LANE_CHANGE_PROBABILITY
        ):
            self._maybe_start_lane_change(car, lane)

    def _update_cruiser(self, car: TrafficCar) -> None:
        """Cruisers are blind to other cars but brake for static obstacles."""

        cfg = self.config
        lane = self.traffic_lane(car)
        gap = float("inf")
        for other in self.traffic:
            if other is car or other.behavior != "obstacle":
                continue
            if self.traffic_lane(other) == lane and other.y_m > car.y_m:
                gap = min(gap, other.y_m - car.y_m - cfg.car_length_m)
        assert car.cruise_speed_mps is not None
        threshold = (
            REACTIVE_MIN_GAP_M
            + REACTIVE_TIME_HEADWAY_S * car.speed_mps
            + car.speed_mps**2 / (2.0 * REACTIVE_BRAKE_MPS2)
        )
        if gap < threshold:
            car.speed_mps = max(
                0.0, car.speed_mps - REACTIVE_BRAKE_MPS2 * cfg.dt_seconds
            )
        elif car.speed_mps < car.cruise_speed_mps:
            car.speed_mps = min(
                car.cruise_speed_mps,
                car.speed_mps + REACTIVE_ACCEL_MPS2 * cfg.dt_seconds,
            )

    def _front_gap_for(self, car: TrafficCar, lane: int) -> tuple[float, float]:
        """Gap to the nearest leader in a lane, plus that leader's speed."""

        cfg = self.config
        gap = float("inf")
        leader_speed = 0.0
        for other in self.traffic:
            if other is car:
                continue
            if self.traffic_lane(other) == lane and other.y_m > car.y_m:
                candidate = other.y_m - car.y_m - cfg.car_length_m
                if candidate < gap:
                    gap = candidate
                    leader_speed = other.speed_mps
        if lane == self.current_lane and car.y_m < 0.0:
            candidate = -car.y_m - cfg.car_length_m
            if candidate < gap:
                gap = candidate
                leader_speed = self.ego_speed_mps
        return max(0.0, gap), leader_speed

    def _maybe_start_lane_change(self, car: TrafficCar, lane: int) -> None:
        current_gap, _ = self._front_gap_for(car, lane)
        candidates = [c for c in (lane - 1, lane + 1) if 0 <= c < self.config.lane_count]
        if len(candidates) == 2 and self.rng.random() < 0.5:
            candidates.reverse()
        for candidate in candidates:
            if self._lane_change_ok(car, candidate, current_gap):
                car.target_lane = candidate
                car.lane_change_progress = 0.0
                return

    def _lane_change_ok(self, car: TrafficCar, candidate: int, current_gap: float) -> bool:
        front_gap, _ = self._front_gap_for(car, candidate)
        rear_gap = self._rear_gap_for(car, candidate)
        return (
            front_gap > current_gap
            and front_gap >= LANE_CHANGE_MIN_FRONT_GAP_M
            and rear_gap >= LANE_CHANGE_MIN_REAR_GAP_M
        )

    def _rear_gap_for(self, car: TrafficCar, lane: int) -> float:
        cfg = self.config
        gap = float("inf")
        for other in self.traffic:
            if other is car:
                continue
            if self.traffic_lane(other) == lane and other.y_m < car.y_m:
                gap = min(gap, car.y_m - other.y_m - cfg.car_length_m)
        if lane == self.current_lane and car.y_m > 0.0:
            gap = min(gap, car.y_m - cfg.car_length_m)
        return max(0.0, gap)

    def _has_collision(self) -> bool:
        cfg = self.config
        for car in self.traffic:
            car_x = self.traffic_x_m(car)
            longitudinal_overlap = abs(car.y_m) < cfg.car_length_m
            lateral_overlap = abs(car_x - self.ego_x_m) < cfg.car_width_m
            if longitudinal_overlap and lateral_overlap:
                return True
        return False

    def _is_off_road(self) -> bool:
        return (
            abs(self.ego_x_m) + self.config.car_width_m / 2.0
            > self.config.road_half_width_m
        )

    def _info(self, collision: bool, off_road: bool) -> dict[str, Any]:
        return {
            "collision": collision,
            "off_road": off_road,
            "distance_m": self.distance_m,
            "speed_mps": self.ego_speed_mps,
            "speed_mph": self.ego_speed_mps * 2.236936,
            "lane": self.current_lane,
            "steps": self.steps,
            "action": ACTION_NAMES[self.previous_action],
            "reward_terms": self.last_reward_terms.copy(),
        }
