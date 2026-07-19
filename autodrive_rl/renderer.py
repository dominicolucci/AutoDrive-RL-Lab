"""Tkinter renderer for the top-down driving simulation."""

from __future__ import annotations

import time
import tkinter as tk
from typing import Any

import numpy as np

from .environment import ACTION_NAMES, Action, DrivingEnv


class TopDownRenderer:
    """Draw the highway, traffic, sensor ranges, and learning dashboard."""

    width = 1000
    height = 720
    road_left = 65
    road_right = 600
    ego_screen_y = 570
    longitudinal_scale = 4.25

    traffic_colors = ("#ff595e", "#ffca3a", "#8ac926", "#6a4c93", "#f28482", "#84a59d")

    def __init__(self, *, fps: int = 30, title: str = "AutoDrive RL Lab") -> None:
        self.fps = max(1, fps)
        self.closed = False
        self.paused = False
        self.reset_requested = False
        self.keys_down: set[str] = set()
        self.last_frame_time = time.perf_counter()

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{self.width}x{self.height}")
        self.root.resizable(False, False)
        self.root.configure(bg="#0d1520")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.canvas = tk.Canvas(
            self.root,
            width=self.width,
            height=self.height,
            bg="#0d1520",
            highlightthickness=0,
        )
        self.canvas.pack()
        self.root.focus_force()

    def _on_key_press(self, event: tk.Event[Any]) -> None:
        key = str(event.keysym).lower()
        self.keys_down.add(key)
        if key in {"escape", "q"}:
            self.close()
        elif key == "p":
            self.paused = not self.paused
        elif key == "r":
            self.reset_requested = True

    def _on_key_release(self, event: tk.Event[Any]) -> None:
        self.keys_down.discard(str(event.keysym).lower())

    def process_events(self) -> None:
        if self.closed:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True

    def manual_action(self) -> int:
        if {"left", "a"} & self.keys_down:
            return int(Action.STEER_LEFT)
        if {"right", "d"} & self.keys_down:
            return int(Action.STEER_RIGHT)
        if {"down", "s"} & self.keys_down:
            return int(Action.BRAKE)
        if {"up", "w"} & self.keys_down:
            return int(Action.ACCELERATE)
        return int(Action.MAINTAIN)

    def render(
        self,
        env: DrivingEnv,
        *,
        policy_name: str,
        action: int,
        episode: int,
        episode_reward: float,
        epsilon: float | None = None,
        message: str | None = None,
    ) -> None:
        if self.closed:
            return
        self.canvas.delete("all")
        self._draw_world(env)
        self._draw_dashboard(
            env,
            policy_name=policy_name,
            action=action,
            episode=episode,
            episode_reward=episode_reward,
            epsilon=epsilon,
        )
        if self.paused:
            self._draw_overlay("PAUSED", "Press P to continue")
        elif message:
            self._draw_overlay(message, "R: restart   Q/Esc: quit")
        self.process_events()

    def _draw_world(self, env: DrivingEnv) -> None:
        canvas = self.canvas
        canvas.create_rectangle(0, 0, self.road_left, self.height, fill="#173f2b", outline="")
        canvas.create_rectangle(
            self.road_right, 0, self.road_right + 18, self.height, fill="#173f2b", outline=""
        )
        canvas.create_rectangle(
            self.road_left,
            0,
            self.road_right,
            self.height,
            fill="#252a31",
            outline="",
        )
        canvas.create_line(self.road_left + 7, 0, self.road_left + 7, self.height, fill="#f4f1de", width=4)
        canvas.create_line(self.road_right - 7, 0, self.road_right - 7, self.height, fill="#f4f1de", width=4)

        lane_pixel_width = (self.road_right - self.road_left) / env.config.lane_count
        dash_offset = (env.distance_m * self.longitudinal_scale) % 64.0
        for lane_boundary in range(1, env.config.lane_count):
            x = self.road_left + lane_boundary * lane_pixel_width
            y = -64.0 + dash_offset
            while y < self.height:
                canvas.create_line(x, y, x, y + 34, fill="#d8dbe2", width=3)
                y += 64.0

        self._draw_sensors(env)
        visible_cars = sorted(env.traffic, key=lambda car: car.y_m, reverse=True)
        for car in visible_cars:
            screen_y = self.ego_screen_y - car.y_m * self.longitudinal_scale
            if -60 <= screen_y <= self.height + 60:
                x = self._world_x_to_screen(env, env.lane_center(car.lane))
                color = self.traffic_colors[car.color_index % len(self.traffic_colors)]
                self._draw_car(x, screen_y, color, label=f"{car.speed_mps * 2.236936:.0f}")

        ego_x = self._world_x_to_screen(env, env.ego_x_m)
        self._draw_car(ego_x, self.ego_screen_y, "#3a86ff", label="AI", ego=True)

        canvas.create_text(
            self.road_left + 18,
            18,
            text="NUMERICAL RANGE SENSORS",
            anchor="nw",
            fill="#a9d6e5",
            font=("Arial", 10, "bold"),
        )

    def _draw_sensors(self, env: DrivingEnv) -> None:
        sensors = env.sensor_snapshot()
        front = np.asarray(sensors["front_gaps_m"])
        rear = np.asarray(sensors["rear_gaps_m"])
        ego_x = self._world_x_to_screen(env, env.ego_x_m)
        for lane in range(env.config.lane_count):
            lane_x = self._world_x_to_screen(env, env.lane_center(lane))
            front_y = self.ego_screen_y - float(front[lane]) * self.longitudinal_scale
            rear_y = self.ego_screen_y + float(rear[lane]) * self.longitudinal_scale
            color = "#62d9ff" if lane == env.current_lane else "#476d7c"
            canvas_width = 2 if lane == env.current_lane else 1
            self.canvas.create_line(
                ego_x,
                self.ego_screen_y,
                lane_x,
                max(10.0, front_y),
                fill=color,
                width=canvas_width,
                dash=(5, 5),
            )
            if rear[lane] < env.config.sensor_range_m:
                self.canvas.create_line(
                    ego_x,
                    self.ego_screen_y,
                    lane_x,
                    min(self.height - 8.0, rear_y),
                    fill="#374f5b",
                    width=1,
                    dash=(3, 6),
                )

    def _draw_car(
        self,
        x: float,
        y: float,
        color: str,
        *,
        label: str,
        ego: bool = False,
    ) -> None:
        half_width = 25
        half_length = 43
        outline = "#c7f9ff" if ego else "#111820"
        width = 3 if ego else 2
        self.canvas.create_rectangle(
            x - half_width,
            y - half_length,
            x + half_width,
            y + half_length,
            fill=color,
            outline=outline,
            width=width,
        )
        self.canvas.create_rectangle(
            x - 18,
            y - 23,
            x + 18,
            y + 9,
            fill="#152535",
            outline="#8ecae6",
            width=1,
        )
        self.canvas.create_rectangle(x - 29, y - 29, x - 23, y - 11, fill="#080b0f", outline="")
        self.canvas.create_rectangle(x + 23, y - 29, x + 29, y - 11, fill="#080b0f", outline="")
        self.canvas.create_rectangle(x - 29, y + 13, x - 23, y + 31, fill="#080b0f", outline="")
        self.canvas.create_rectangle(x + 23, y + 13, x + 29, y + 31, fill="#080b0f", outline="")
        self.canvas.create_text(x, y + 25, text=label, fill="white", font=("Arial", 9, "bold"))

    def _draw_dashboard(
        self,
        env: DrivingEnv,
        *,
        policy_name: str,
        action: int,
        episode: int,
        episode_reward: float,
        epsilon: float | None,
    ) -> None:
        left = 635
        self.canvas.create_text(
            left,
            36,
            text="AUTODRIVE RL LAB",
            anchor="w",
            fill="#f8f9fa",
            font=("Arial", 19, "bold"),
        )
        self.canvas.create_text(
            left,
            67,
            text="A small car learning a big idea",
            anchor="w",
            fill="#8da9c4",
            font=("Arial", 11),
        )

        rows = [
            ("POLICY", policy_name.upper()),
            ("ACTION", ACTION_NAMES[Action(action)].upper()),
            ("SPEED", f"{env.ego_speed_mps * 2.236936:5.1f} mph"),
            ("LANE", f"{env.current_lane + 1} / {env.config.lane_count}"),
            ("DISTANCE", f"{env.distance_m:,.0f} m"),
            ("EPISODE", str(episode)),
            ("RETURN", f"{episode_reward:,.1f}"),
        ]
        if epsilon is not None:
            rows.append(("EPSILON", f"{epsilon:.3f}"))

        y = 115
        for label, value in rows:
            self.canvas.create_text(
                left, y, text=label, anchor="w", fill="#6f8ca3", font=("Arial", 9, "bold")
            )
            self.canvas.create_text(
                955, y, text=value, anchor="e", fill="#f8f9fa", font=("Arial", 11, "bold")
            )
            self.canvas.create_line(left, y + 18, 955, y + 18, fill="#1e3346")
            y += 40

        sensors = env.sensor_snapshot()
        front = np.asarray(sensors["front_gaps_m"])
        y += 8
        self.canvas.create_text(
            left,
            y,
            text="FRONT RANGE BY LANE",
            anchor="w",
            fill="#6f8ca3",
            font=("Arial", 9, "bold"),
        )
        y += 30
        for lane, gap in enumerate(front):
            active = lane == env.current_lane
            self.canvas.create_text(
                left,
                y,
                text=f"L{lane + 1}",
                anchor="w",
                fill="#62d9ff" if active else "#8da9c4",
                font=("Arial", 10, "bold"),
            )
            bar_left = left + 36
            bar_right = 905
            fraction = min(1.0, float(gap) / env.config.sensor_range_m)
            self.canvas.create_rectangle(bar_left, y - 7, bar_right, y + 8, fill="#172533", outline="")
            self.canvas.create_rectangle(
                bar_left,
                y - 7,
                bar_left + fraction * (bar_right - bar_left),
                y + 8,
                fill="#3a86ff" if active else "#3f5f6f",
                outline="",
            )
            self.canvas.create_text(
                955,
                y,
                text=f"{gap:4.0f} m",
                anchor="e",
                fill="#f8f9fa",
                font=("Arial", 9),
            )
            y += 28

        self.canvas.create_text(
            left,
            677,
            text="P: pause    R: restart    Q/Esc: quit",
            anchor="w",
            fill="#6f8ca3",
            font=("Arial", 10),
        )

    def _draw_overlay(self, heading: str, subheading: str) -> None:
        self.canvas.create_rectangle(160, 285, 505, 410, fill="#0a1119", outline="#62d9ff", width=2)
        self.canvas.create_text(
            332,
            330,
            text=heading,
            fill="#f8f9fa",
            font=("Arial", 22, "bold"),
        )
        self.canvas.create_text(332, 372, text=subheading, fill="#8da9c4", font=("Arial", 11))

    def _world_x_to_screen(self, env: DrivingEnv, x_m: float) -> float:
        fraction = (x_m + env.config.road_half_width_m) / env.config.road_width_m
        return self.road_left + fraction * (self.road_right - self.road_left)

    def tick(self) -> None:
        frame_duration = 1.0 / self.fps
        elapsed = time.perf_counter() - self.last_frame_time
        if elapsed < frame_duration:
            time.sleep(frame_duration - elapsed)
        self.last_frame_time = time.perf_counter()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

