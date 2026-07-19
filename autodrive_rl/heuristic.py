"""A simple rule-based driver used to inspect the simulation before training."""

from __future__ import annotations

import numpy as np

from .environment import Action, DrivingEnv


class HeuristicDriver:
    """Keep speed, change lanes around slow traffic, and avoid nearby cars."""

    def __init__(self, target_speed_mps: float = 23.0) -> None:
        self.target_speed_mps = target_speed_mps
        self.target_lane: int | None = None

    def reset(self) -> None:
        self.target_lane = None

    def act(self, env: DrivingEnv) -> int:
        sensors = env.sensor_snapshot()
        front = np.asarray(sensors["front_gaps_m"])
        rear = np.asarray(sensors["rear_gaps_m"])
        lane = env.current_lane

        if self.target_lane is not None:
            target_x = env.lane_center(self.target_lane)
            difference = target_x - env.ego_x_m
            if abs(difference) < 0.18:
                self.target_lane = None
            else:
                return int(Action.STEER_RIGHT if difference > 0.0 else Action.STEER_LEFT)

        desired_gap = max(18.0, env.ego_speed_mps * 1.35)
        if front[lane] < desired_gap:
            candidates: list[tuple[float, int]] = []
            for candidate in (lane - 1, lane + 1):
                if not 0 <= candidate < env.config.lane_count:
                    continue
                if front[candidate] > desired_gap + 8.0 and rear[candidate] > 14.0:
                    candidates.append((float(front[candidate]), candidate))
            if candidates:
                self.target_lane = max(candidates)[1]
                target_x = env.lane_center(self.target_lane)
                return int(
                    Action.STEER_RIGHT if target_x > env.ego_x_m else Action.STEER_LEFT
                )
            if front[lane] < max(9.0, env.ego_speed_mps * 0.8):
                return int(Action.BRAKE)
            return int(Action.MAINTAIN)

        if env.ego_speed_mps < self.target_speed_mps:
            return int(Action.ACCELERATE)
        return int(Action.MAINTAIN)

