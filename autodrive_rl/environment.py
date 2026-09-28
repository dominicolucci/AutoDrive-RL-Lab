"""A compact, Gym-style three-lane highway environment.

The environment intentionally uses numerical sensors instead of camera pixels.
That keeps the first project focused on the reinforcement-learning loop: state,
action, transition, reward, replay, and policy improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

# Rear traffic ---------------------------------------------------------------
# Without these, every car spawns ahead of the ego and braking to a halt makes
# the whole world drive away — which is exactly how a safety-only score gets
# won by not moving. Only scenarios with `rear_traffic` set use them.
REAR_SPAWN_RANGE_M = 150.0   # farthest a car enters behind the ego
REAR_SPAWN_MIN_M = 60.0      # nearest it may enter, so nothing appears on top
SPAWN_EGO_CLEAR_M = 22.0     # keep-out bubble around the ego at spawn time

REACTIVE_TIME_HEADWAY_S = 1.5
REACTIVE_BRAKE_MPS2 = 4.0
REACTIVE_EMERGENCY_BRAKE_MPS2 = 9.0
REACTIVE_ACCEL_MPS2 = 2.0
# Sized for the 65 mph flow: at higher speeds the discrete 0.1 s tick moves
# cars up to ~2 m per step, so the standstill margin must absorb a full
# step of closing motion on top of a real-world bumper gap.
REACTIVE_MIN_GAP_M = 3.0

LANE_CHANGE_DURATION_S = 1.2
LANE_CHANGE_PROBABILITY = 0.005
LANE_CHANGE_MIN_FRONT_GAP_M = 15.0
LANE_CHANGE_MIN_REAR_GAP_M = 12.0
LANE_CHANGE_ABORT_FRONT_M = 6.0
LANE_CHANGE_ABORT_REAR_M = 4.0
LANE_CHANGE_CONFLICT_WINDOW_M = 18.0

# Real-time driving grade ---------------------------------------------------
# The ego is scored every step like a driving instructor riding along:
# lane discipline, speed compliance, following distance, merge courtesy,
# and physical clearances all feed both the reward and a rolling report
# card (0-100, mapped to letter grades for the dashboard).
CONTACT_EPSILON_M = 0.05          # any physical touch counts as a crash
SPEEDING_GRACE_MPS = 1.0          # ~2 mph of real-world enforcement grace
DIVIDER_CROSSING_SPEED_MPS = 0.6  # lateral speed marking a deliberate crossing
CUT_OFF_SETTLE_STEPS = 20         # post-merge window judged for courtesy
NEAR_MISS_LONG_M = 2.0            # bumper clearance considered a close call
NEAR_MISS_LAT_M = 0.6             # side clearance considered a close call
GRADE_EMA_ALPHA = 0.03            # ~3 s rolling window at dt = 0.1 s
GRADE_PENALTY_SCALE = 5.0         # step penalty that drags a step score to 0

GRADE_BANDS: tuple[tuple[float, str], ...] = (
    (97.0, "A+"), (93.0, "A"), (90.0, "A-"),
    (87.0, "B+"), (83.0, "B"), (80.0, "B-"),
    (77.0, "C+"), (73.0, "C"), (70.0, "C-"),
    (60.0, "D"),
)


def grade_letter(score: float) -> str:
    """Map a 0-100 driving score to a report-card letter."""

    for cutoff, letter in GRADE_BANDS:
        if score >= cutoff:
            return letter
    return "F"


@dataclass
class TrafficCar:
    """A traffic car (or static obstacle) represented relative to the ego car."""

    lane: int
    y_m: float
    speed_mps: float
    color_index: int = 0
    # "cruiser": keeps a safe following distance, never changes lanes.
    # "reactive": keeps a safe following distance, changes lanes when blocked.
    # "obstacle": static hazard, never moves.
    # All moving drivers respect the posted speed limit.
    behavior: str = "cruiser"
    cruise_speed_mps: float | None = None
    target_lane: int | None = None
    lane_change_progress: float = 0.0
    #: An attentive driver perceives the road instantly and can always avoid a
    #: rear-end collision. An inattentive one acts on a stale view, so it can
    #: fail to shed the closing speed behind a car that has stopped.
    attentive: bool = True
    #: Ring of recent (gap, leader_speed) readings; the oldest retained entry
    #: is what an inattentive driver is currently acting on.
    perception_log: list[tuple[float, float]] = field(default_factory=list)

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

    def _occupied_lanes(self, car: TrafficCar) -> tuple[int, ...]:
        """Every lane a car physically claims right now.

        A car mid-lane-change straddles both its origin and target lane, so
        all gap logic must treat it as present in both. This is central to
        the no-overlap guarantee: no driver ever plans around only half of a
        lane-changing car.
        """

        if car.target_lane is None:
            return (car.lane,)
        return (car.lane, car.target_lane)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        randomize_start = bool(options and options.get("randomize_start"))
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.ego_x_m = self.lane_center(self.config.lane_count // 2)
        self.ego_lateral_speed_mps = 0.0
        self.ego_speed_mps = 12.0
        if randomize_start:
            # Handover training: an autopilot engages mid-drive, in whatever
            # state the human hands it — off-center, on a divider, slow or
            # near the limit, drifting sideways. Starting episodes in such
            # states teaches the policy to recover from all of them.
            cfg = self.config
            margin = cfg.car_width_m / 2.0 + 0.15
            self.ego_x_m = float(
                self.rng.uniform(
                    -cfg.road_half_width_m + margin,
                    cfg.road_half_width_m - margin,
                )
            )
            self.ego_speed_mps = float(self.rng.uniform(6.0, cfg.speed_limit_mps))
            self.ego_lateral_speed_mps = float(
                self.rng.uniform(-0.5, 0.5) * cfg.max_lateral_speed_mps
            )
        self.distance_m = 0.0
        self.steps = 0
        self.previous_action = Action.MAINTAIN
        self.last_reward_terms = {}
        self.grade_score = 100.0
        self._graded_lane = self.current_lane
        self._cut_off_timer = 0
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

        # Track completed lane changes: the courtesy grade judges the gap
        # the ego leaves its new follower for a short window after a merge.
        lane_now = self.current_lane
        if lane_now != self._graded_lane:
            self._graded_lane = lane_now
            self._cut_off_timer = CUT_OFF_SETTLE_STEPS
        elif self._cut_off_timer > 0:
            self._cut_off_timer -= 1

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

        # Lane discipline: a safe driver never rides the divider line.
        # Crossing briskly during a deliberate lane change is fine (the
        # lateral-speed gate); lingering on the paint is what gets graded.
        interior_lines = self.lane_centers_m[:-1] + cfg.lane_width_m / 2.0
        line_distance = float(np.min(np.abs(interior_lines - self.ego_x_m)))
        half_width = cfg.car_width_m / 2.0
        divider_straddle = 0.0
        if (
            line_distance < half_width
            and abs(self.ego_lateral_speed_mps) < DIVIDER_CROSSING_SPEED_MPS
        ):
            divider_straddle = -0.8 * (1.0 - line_distance / half_width)

        # Speed compliance: a small real-world grace, then a growing fine.
        over_limit = self.ego_speed_mps - (cfg.speed_limit_mps + SPEEDING_GRACE_MPS)
        speeding = 0.0
        if over_limit > 0.0:
            over_span = max(
                0.1, cfg.max_speed_mps - cfg.speed_limit_mps - SPEEDING_GRACE_MPS
            )
            speeding = -1.0 * min(1.0, over_limit / over_span)

        # Merge courtesy: for a short window after every completed lane
        # change, the gap left to the new follower is judged. Squeezing in
        # front of someone with no room is a cut-off even without contact.
        cut_off = 0.0
        if self._cut_off_timer > 0:
            rear_gap = float(np.asarray(sensors["rear_gaps_m"])[self.current_lane])
            rear_relative = float(
                np.asarray(sensors["rear_relative_speeds_mps"])[self.current_lane]
            )
            follower_speed = max(0.0, self.ego_speed_mps + rear_relative)
            needed_gap = max(6.0, 0.8 * follower_speed)
            if rear_gap < needed_gap:
                cut_off = -1.5 * (1.0 - rear_gap / needed_gap)

        # Near miss: physical clearance to any car shrinking in both axes
        # at once — threading a gap with inches to spare is graded down
        # even when nothing actually touches.
        near_miss = 0.0
        for car in self.traffic:
            longitudinal_clear = abs(car.y_m) - cfg.car_length_m
            lateral_clear = (
                abs(self.traffic_x_m(car) - self.ego_x_m) - cfg.car_width_m
            )
            if longitudinal_clear < NEAR_MISS_LONG_M and lateral_clear < NEAR_MISS_LAT_M:
                squeeze = (
                    1.0 - max(0.0, longitudinal_clear) / NEAR_MISS_LONG_M
                ) * (1.0 - max(0.0, lateral_clear) / NEAR_MISS_LAT_M)
                near_miss = min(near_miss, -1.2 * squeeze)

        terminal = -500.0 if collision else (-350.0 if off_road else 0.0)

        self.last_reward_terms = {
            "progress": progress,
            "speed": speed,
            "lane_centering": lane_centering,
            "unsafe_following": unsafe_following,
            "unsafe_lane_change": unsafe_lane_change,
            "road_edge": road_edge,
            "divider_straddle": divider_straddle,
            "speeding": speeding,
            "cut_off": cut_off,
            "near_miss": near_miss,
            "control_cost": control_cost,
            "living_cost": living_cost,
            "terminal": terminal,
        }

        # Roll the safety terms into the live report card. Crashing or
        # leaving the road zeroes it; otherwise it tracks a ~3 s window of
        # how law-abiding and courteous the driving has been.
        safety_penalty = (
            unsafe_following
            + unsafe_lane_change
            + road_edge
            + divider_straddle
            + speeding
            + cut_off
            + near_miss
        )
        step_score = 100.0 * max(0.0, 1.0 + safety_penalty / GRADE_PENALTY_SCALE)
        if collision or off_road:
            self.grade_score = 0.0
        else:
            self.grade_score += GRADE_EMA_ALPHA * (step_score - self.grade_score)

        return float(sum(self.last_reward_terms.values()))

    def _spawn_initial_traffic(self) -> None:
        cfg = self.config
        spec = self.scenario_spec
        count = cfg.traffic_count if spec is None else spec.traffic_count
        reactive_fraction = 0.0 if spec is None else spec.reactive_fraction
        inattentive_fraction = 0.0 if spec is None else spec.inattentive_fraction
        rear_traffic = False if spec is None else spec.rear_traffic
        self._spawn_obstacles()
        moving_target = count + len(self.traffic)
        attempts = 0
        while len(self.traffic) < moving_target and attempts < 500:
            attempts += 1
            lane = int(self.rng.integers(0, cfg.lane_count))
            # Traffic exists behind as well as ahead — but only where the
            # scenario asks for it. The low bound is the only thing that
            # changes, so the draw sequence is identical when it is off.
            low = -REAR_SPAWN_RANGE_M if rear_traffic else SPAWN_EGO_CLEAR_M
            y_m = float(self.rng.uniform(low, cfg.sensor_range_m + 55.0))
            if abs(y_m) < SPAWN_EGO_CLEAR_M:
                continue
            if not self._spawn_gap_ok(lane, y_m):
                continue
            cruise = self._sample_legal_speed()
            speed = self._entry_speed(lane, y_m, cruise)
            if not self._entry_rear_ok(lane, y_m, speed):
                continue
            color_index = int(self.rng.integers(0, 6))
            behavior = "cruiser"
            if reactive_fraction > 0.0 and self.rng.random() < reactive_fraction:
                behavior = "reactive"
            # Gated so a preset with no inattentive drivers consumes exactly
            # the random numbers it did before this feature existed.
            attentive = not (
                inattentive_fraction > 0.0
                and self.rng.random() < inattentive_fraction
            )
            self.traffic.append(
                TrafficCar(
                    lane, y_m, speed, color_index,
                    attentive=attentive,
                    behavior=behavior, cruise_speed_mps=cruise,
                )
            )

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
        # An obstacle must never appear on top of a moving car, nor so close
        # ahead of one that even emergency braking could not avoid it. Movers
        # ahead of the candidate only need clear spacing, since they drive
        # away from it.
        for car in self.traffic:
            if car.behavior == "obstacle" or lane not in self._occupied_lanes(car):
                continue
            if car.y_m <= y_m:
                gap = y_m - car.y_m - cfg.car_length_m
                stopping_distance = car.speed_mps**2 / (2.0 * REACTIVE_BRAKE_MPS2)
                if gap < REACTIVE_MIN_GAP_M + stopping_distance + cfg.car_length_m:
                    return False
            elif car.y_m - y_m < 24.0:
                return False
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
        spec = self.scenario_spec
        rear_traffic = spec is not None and spec.rear_traffic
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
            # Re-enter traffic like a real merging driver: only into a gap
            # with room, and never faster than the flow ahead allows.
            for _ in range(50):
                lane = self._least_crowded_spawn_lane()
                if rear_traffic:
                    # A car quicker than the ego belongs behind it, closing;
                    # a slower one belongs ahead. Deciding needs the cruise
                    # speed first, so it is sampled early — but ONLY on this
                    # branch. Sampling it early unconditionally would change
                    # how many draws every seeded world consumes and silently
                    # invalidate every published result.
                    cruise = self._sample_legal_speed()
                    if cruise > self.ego_speed_mps + 0.5:
                        y_m = float(
                            self.rng.uniform(-REAR_SPAWN_RANGE_M, -REAR_SPAWN_MIN_M)
                        )
                    else:
                        y_m = float(
                            self.rng.uniform(
                                cfg.sensor_range_m + 20.0, cfg.sensor_range_m + 80.0
                            )
                        )
                else:
                    y_m = float(
                        self.rng.uniform(cfg.sensor_range_m + 20.0, cfg.sensor_range_m + 80.0)
                    )
                if not self._spawn_gap_ok(lane, y_m, ignore=car):
                    continue
                if not rear_traffic:
                    cruise = self._sample_legal_speed()
                entry = self._entry_speed(lane, y_m, cruise, ignore=car)
                if not self._entry_rear_ok(lane, y_m, entry, ignore=car):
                    continue
                car.lane = lane
                car.y_m = y_m
                car.cruise_speed_mps = cruise
                car.speed_mps = entry
                car.color_index = int(self.rng.integers(0, 6))
                car.target_lane = None
                car.lane_change_progress = 0.0
                # A recycled car is a different driver. Re-roll attentiveness
                # and discard the previous driver's stale readings — gated so
                # a preset without inattentive drivers draws nothing extra.
                fraction = 0.0 if spec is None else spec.inattentive_fraction
                if fraction > 0.0:
                    car.attentive = self.rng.random() >= fraction
                car.perception_log.clear()
                break
            # If no safe gap exists this step, the car simply stays out of
            # range and tries again on a later step.

    def _spawn_gap_ok(
        self, lane: int, y_m: float, *, ignore: TrafficCar | None = None
    ) -> bool:
        """True when no other car occupies the entry gap in this lane."""

        return not any(
            car is not ignore
            and lane in self._occupied_lanes(car)
            and abs(car.y_m - y_m) < 24.0
            for car in self.traffic
        )

    def _entry_rear_ok(
        self,
        lane: int,
        y_m: float,
        entry_speed: float,
        *,
        ignore: TrafficCar | None = None,
    ) -> bool:
        """True when every follower can absorb this merge with normal braking.

        A car may never enter the flow in front of a follower whose closing
        speed cannot be shed at the comfortable braking rate within the gap
        left over — that would force an emergency stop (or a rear-end) the
        entering driver caused. The ego also counts as a follower.
        """

        cfg = self.config
        for other in self.traffic:
            if other is ignore or other.behavior == "obstacle":
                continue
            if lane in self._occupied_lanes(other) and other.y_m < y_m:
                gap = y_m - other.y_m - cfg.car_length_m
                closing = max(0.0, other.speed_mps - entry_speed)
                needed = (
                    REACTIVE_MIN_GAP_M + closing**2 / (2.0 * REACTIVE_BRAKE_MPS2)
                )
                if gap < needed:
                    return False
        if lane == self.current_lane and y_m > 0.0:
            gap = y_m - cfg.car_length_m
            closing = max(0.0, self.ego_speed_mps - entry_speed)
            needed = REACTIVE_MIN_GAP_M + closing**2 / (2.0 * cfg.braking_mps2)
            if gap < needed:
                return False
        return True

    def _entry_speed(
        self,
        lane: int,
        y_m: float,
        sampled: float,
        *,
        ignore: TrafficCar | None = None,
    ) -> float:
        """A safe speed for a car joining the flow at (lane, y_m).

        Real drivers match the traffic ahead of them. The cap is the highest
        speed from which comfortable braking can settle to the leader's speed
        within the available gap, so an entering car never has to slam its
        brakes or rear-end anyone.
        """

        cfg = self.config
        gap = float("inf")
        leader_speed = 0.0
        for other in self.traffic:
            if other is ignore:
                continue
            if lane in self._occupied_lanes(other) and other.y_m > y_m:
                candidate = other.y_m - y_m - cfg.car_length_m
                if candidate < gap:
                    gap = candidate
                    leader_speed = other.speed_mps
        if not np.isfinite(gap):
            return sampled
        # Pessimistic: assume the leader might be braking to a dead stop
        # (a shockwave in the queue). The entering car must be able to stop
        # at a comfortable rate before the point where the leader would
        # halt, so no entry speed ever relies on the flow staying fast.
        stopping_gap = max(gap - REACTIVE_MIN_GAP_M, 0.0)
        leader_stop_distance = leader_speed**2 / (2.0 * REACTIVE_EMERGENCY_BRAKE_MPS2)
        safe_speed = np.sqrt(
            2.0 * REACTIVE_BRAKE_MPS2 * (stopping_gap + leader_stop_distance)
        )
        return float(min(sampled, safe_speed))

    def _sample_legal_speed(self) -> float:
        """A cruise speed for a law-abiding driver: never above the limit."""

        cfg = self.config
        return float(
            self.rng.uniform(cfg.traffic_min_speed_mps, cfg.speed_limit_mps)
        )

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
        self._resolve_lane_change_conflicts()
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

    def _resolve_lane_change_conflicts(self) -> None:
        """Abort lane changes that have become unsafe mid-animation.

        Real drivers glance again mid-merge and pull back when the gap has
        gone away. Two situations trigger an abort while a change is still
        in its first half: another car is merging into the same lane at
        nearly the same position (the less-committed driver yields), or the
        target-lane gap has collapsed below a critical margin. Past the
        halfway point the car is already mostly in the new lane, so it
        commits and everyone else keeps distance instead.
        """

        changers = [car for car in self.traffic if car.target_lane is not None]
        for car in changers:
            if car.lane_change_progress >= 0.5:
                continue
            conflict = False
            for other in changers:
                if other is car or other.target_lane != car.target_lane:
                    continue
                if abs(other.y_m - car.y_m) >= LANE_CHANGE_CONFLICT_WINDOW_M:
                    continue
                yields = (car.lane_change_progress, car.y_m) <= (
                    other.lane_change_progress,
                    other.y_m,
                )
                if yields:
                    conflict = True
                    break
            if not conflict:
                assert car.target_lane is not None
                front_gap, _ = self._front_gap_for(car, car.target_lane)
                rear_gap, _ = self._rear_gap_for(car, car.target_lane)
                if (
                    front_gap >= LANE_CHANGE_ABORT_FRONT_M
                    and rear_gap >= LANE_CHANGE_ABORT_REAR_M
                ):
                    continue
            self._abort_lane_change(car)

    def _abort_lane_change(self, car: TrafficCar) -> None:
        """Smoothly reverse an in-progress lane change back to its origin."""

        assert car.target_lane is not None
        car.lane, car.target_lane = car.target_lane, car.lane
        car.lane_change_progress = 1.0 - car.lane_change_progress

    def _update_reactive(self, car: TrafficCar) -> None:
        """Reactive drivers follow safely and change lanes when blocked."""

        lane = self.traffic_lane(car)
        gap, leader_speed = self._nearest_front_gap(car)
        gap, leader_speed = self._perceive(car, gap, leader_speed)
        self._follow_safely(car, gap, leader_speed)
        assert car.cruise_speed_mps is not None
        if (
            car.target_lane is None
            and car.speed_mps < 0.8 * car.cruise_speed_mps
            and self.rng.random() < LANE_CHANGE_PROBABILITY
        ):
            self._maybe_start_lane_change(car, lane)

    def _update_cruiser(self, car: TrafficCar) -> None:
        """Cruisers drive like ordinary careful drivers.

        They keep a physics-based safe following distance behind whatever is
        ahead of them in their lane (traffic cars, static obstacles, and the
        ego car alike) and hold their cruise speed otherwise. Unlike reactive
        drivers they never change lanes.
        """

        lane = self.traffic_lane(car)
        gap, leader_speed = self._front_gap_for(car, lane)
        gap, leader_speed = self._perceive(car, gap, leader_speed)
        self._follow_safely(car, gap, leader_speed)

    def _perceive(
        self, car: TrafficCar, gap: float, leader_speed: float
    ) -> tuple[float, float]:
        """What this driver believes the road ahead looks like right now.

        Attentive drivers see the truth, which is why they can always avoid a
        car that stops in front of them — the safe-following model brakes in
        time by construction. Inattentive ones act on a reading from
        ``reaction_delay_s`` ago, so by the time a closing gap registers there
        may no longer be room to shed the speed. That delay, not the traffic
        density, is what makes standing still in a live lane dangerous.
        """

        if car.attentive or self.config.reaction_delay_s <= 0.0:
            return gap, leader_speed
        lag_steps = max(1, round(self.config.reaction_delay_s / self.config.dt_seconds))
        car.perception_log.append((gap, leader_speed))
        if len(car.perception_log) > lag_steps:
            del car.perception_log[0]
        # Before the log fills, the oldest entry is the most recent one, so a
        # driver starts out effectively attentive and degrades into the full
        # delay as the episode runs.
        return car.perception_log[0]

    def _follow_safely(self, car: TrafficCar, gap: float, leader_speed: float) -> None:
        """Brake when the safe-stopping envelope is violated, else cruise.

        The threshold combines a minimum standstill gap, a time headway, and
        the extra distance needed to shed any closing speed at a comfortable
        braking rate. Cruise speeds are additionally clamped to the posted
        speed limit so no simulated driver ever speeds.
        """

        cfg = self.config
        assert car.cruise_speed_mps is not None
        legal_cruise = min(car.cruise_speed_mps, cfg.speed_limit_mps)
        closing = max(0.0, car.speed_mps - leader_speed)
        threshold = (
            REACTIVE_MIN_GAP_M
            + REACTIVE_TIME_HEADWAY_S * car.speed_mps
            + closing**2 / (2.0 * REACTIVE_BRAKE_MPS2)
        )
        if gap < threshold:
            # Comfortable braking is the norm, but when the remaining gap is
            # too short to shed the closing speed at the comfortable rate, a
            # real driver brakes hard instead of causing a rear-end crash.
            stopping_gap = max(gap - REACTIVE_MIN_GAP_M, 0.1)
            required_decel = closing**2 / (2.0 * stopping_gap)
            brake = (
                REACTIVE_EMERGENCY_BRAKE_MPS2
                if required_decel > REACTIVE_BRAKE_MPS2
                else REACTIVE_BRAKE_MPS2
            )
            car.speed_mps = max(0.0, car.speed_mps - brake * cfg.dt_seconds)
        elif car.speed_mps < legal_cruise:
            car.speed_mps = min(
                legal_cruise,
                car.speed_mps + REACTIVE_ACCEL_MPS2 * cfg.dt_seconds,
            )
        elif car.speed_mps > cfg.speed_limit_mps:
            car.speed_mps = max(
                cfg.speed_limit_mps,
                car.speed_mps - REACTIVE_BRAKE_MPS2 * cfg.dt_seconds,
            )

    def _front_gap_for(self, car: TrafficCar, lane: int) -> tuple[float, float]:
        """Gap to the nearest leader in a lane, plus that leader's speed."""

        cfg = self.config
        gap = float("inf")
        leader_speed = 0.0
        for other in self.traffic:
            if other is car:
                continue
            if lane in self._occupied_lanes(other) and other.y_m > car.y_m:
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

    def _nearest_front_gap(self, car: TrafficCar) -> tuple[float, float]:
        """The tightest front gap over every lane the car occupies."""

        gap = float("inf")
        leader_speed = 0.0
        for lane in self._occupied_lanes(car):
            lane_gap, lane_speed = self._front_gap_for(car, lane)
            if lane_gap < gap:
                gap = lane_gap
                leader_speed = lane_speed
        return gap, leader_speed

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
        """A courteous lane change: enough room ahead, and no cut-off behind.

        The rear requirement grows with how fast the trailing car (or the ego)
        is closing, so a driver never merges into a gap that forces someone
        behind to brake hard.
        """

        front_gap, leader_speed = self._front_gap_for(car, candidate)
        rear_gap, follower_speed = self._rear_gap_for(car, candidate)
        rear_closing = max(0.0, follower_speed - car.speed_mps)
        # The follower must be able to absorb the merge at a comfortable
        # braking rate — a time-headway margin alone is too thin at highway
        # closing speeds.
        required_rear_gap = LANE_CHANGE_MIN_REAR_GAP_M + rear_closing**2 / (
            2.0 * REACTIVE_BRAKE_MPS2
        )
        front_closing = max(0.0, car.speed_mps - leader_speed)
        required_front_gap = LANE_CHANGE_MIN_FRONT_GAP_M + front_closing**2 / (
            2.0 * REACTIVE_BRAKE_MPS2
        )
        return (
            front_gap > current_gap
            and front_gap >= required_front_gap
            and rear_gap >= required_rear_gap
        )

    def _rear_gap_for(self, car: TrafficCar, lane: int) -> tuple[float, float]:
        """Gap to the nearest follower in a lane, plus that follower's speed."""

        cfg = self.config
        gap = float("inf")
        follower_speed = 0.0
        for other in self.traffic:
            if other is car:
                continue
            if lane in self._occupied_lanes(other) and other.y_m < car.y_m:
                candidate = car.y_m - other.y_m - cfg.car_length_m
                if candidate < gap:
                    gap = candidate
                    follower_speed = other.speed_mps
        if lane == self.current_lane and car.y_m > 0.0:
            candidate = car.y_m - cfg.car_length_m
            if candidate < gap:
                gap = candidate
                follower_speed = self.ego_speed_mps
        return max(0.0, gap), follower_speed

    def _has_collision(self) -> bool:
        # Touch-inclusive: even a slight brush against another car's bumper
        # or door panel counts as a crash, exactly like a rear-end. A merge
        # only "works" if the ego never makes contact with the cars that
        # frame the gap.
        cfg = self.config
        for car in self.traffic:
            car_x = self.traffic_x_m(car)
            longitudinal_overlap = abs(car.y_m) <= cfg.car_length_m + CONTACT_EPSILON_M
            lateral_overlap = (
                abs(car_x - self.ego_x_m) <= cfg.car_width_m + CONTACT_EPSILON_M
            )
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
            "driving_grade": float(self.grade_score),
            "grade_letter": grade_letter(self.grade_score),
            "distance_m": self.distance_m,
            "speed_mps": self.ego_speed_mps,
            "speed_mph": self.ego_speed_mps * 2.236936,
            "lane": self.current_lane,
            "steps": self.steps,
            "action": ACTION_NAMES[self.previous_action],
            "reward_terms": self.last_reward_terms.copy(),
        }