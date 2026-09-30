"""Tkinter renderer for the top-down driving simulation.

Visual design notes
-------------------
The scene uses a layered, modern semi-realistic style drawn entirely with
Tkinter canvas primitives:

- Cars are composite sprites: a soft drop shadow, tires tucked under the
  fenders, a crisp straight-sided sedan silhouette with chamfered corners,
  hood and trunk panels, tinted glass with a highlight, side mirrors, and
  wide light bars. The ego car wears the accent blue and a halo ring so
  the eye finds it instantly.
- Cars are drawn near true-to-scale lane fill (a real car is about half a
  lane wide), with a closer camera so the traffic feels substantial. A
  dashed footprint box under each car marks the exact rectangle the
  physics uses for collisions, so what you see is what the crash check
  sees.
- The road is built from layers (verge, guardrail, shoulder, asphalt,
  edge lines, scrolling lane dashes and asphalt speckle) to create motion
  and depth without hurting the frame rate.
- The dashboard is a card-based panel with a speedometer arc, metric
  chips, per-lane clearance bars, and keycap-styled controls.
"""

from __future__ import annotations

import math
import time
import tkinter as tk
from typing import Any

import numpy as np

from .environment import ACTION_NAMES, Action, DrivingEnv, grade_letter

# Palette ------------------------------------------------------------------
BG = "#0b1017"
PANEL_BG = "#0e141d"
CARD_BG = "#141c27"
CARD_EDGE = "#1f2c3b"
TEXT_MAIN = "#f1f3f5"
TEXT_DIM = "#6f8ca3"
ACCENT = "#4cc9f0"
ACCENT_DEEP = "#3a86ff"
ASPHALT = "#2b2f36"
ASPHALT_SPECKLE = "#24272e"
SHOULDER = "#23262d"
EDGE_LINE = "#eef0f3"
LANE_DASH = "#dfe3e8"
VERGE = "#12281b"
VERGE_BAND = "#112617"
GUARDRAIL = "#55606c"
GLASS = "#1d2a38"
GLASS_EDGE = "#12202c"
GLASS_SHINE = "#46647e"
TIRE = "#0c0f13"
HEADLIGHT = "#f8f3d6"
TAILLIGHT = "#e5383b"
EGO_BODY = "#3a86ff"
EGO_HALO = "#7fd8ff"

TRAFFIC_COLORS = ("#d64550", "#e9a13b", "#7cb464", "#8d6fb8", "#d9776f", "#5f9ea0")


class TopDownRenderer:
    """Draw the highway, traffic, sensor ranges, and learning dashboard."""

    width = 1000
    height = 720
    road_left = 110
    road_right = 470
    ego_screen_y = 560
    longitudinal_scale = 13.0

    traffic_colors = TRAFFIC_COLORS

    # Honest proportions: sprite width is the car's true share of the lane
    # (1.9 m of 3.7 m), and sprite length exceeds the physical car by only
    # ~1.2 m of road — so two sprites visually touch just as the physics is
    # about to call the crash. The dashed footprint box marks the exact
    # collision rectangle.
    CAR_HALF_W = 30
    CAR_HALF_L = 38

    #: Selectable playback rates. A 900-step episode is 90 s of wall clock at
    #: 1x, which is far too slow to watch a policy improve over a training
    #: run; 20x turns it into four and a half seconds.
    SPEEDS: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)

    def __init__(
        self,
        *,
        fps: int = 30,
        title: str = "AutoDrive RL Lab",
        speed: float = 1.0,
    ) -> None:
        self.fps = max(1, fps)
        self.speed = self._nearest_speed(speed)
        self.skip_requested = False
        self.closed = False
        self.paused = False
        self.reset_requested = False
        self.autopilot_toggle_requested = False
        self.keys_down: set[str] = set()
        self.last_frame_time = time.perf_counter()

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{self.width}x{self.height}")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.canvas = tk.Canvas(
            self.root,
            width=self.width,
            height=self.height,
            bg=BG,
            highlightthickness=0,
        )
        self.canvas.pack()
        self.root.focus_force()

    # Event handling -------------------------------------------------------

    def _on_key_press(self, event: tk.Event[Any]) -> None:
        key = str(event.keysym).lower()
        self.keys_down.add(key)
        if key in {"escape", "q"}:
            self.close()
        elif key == "p":
            self.paused = not self.paused
        elif key == "r":
            self.reset_requested = True
        elif key == "space":
            self.autopilot_toggle_requested = True
        elif key in {"plus", "equal", "kp_add"}:
            self.change_speed(1)
        elif key in {"minus", "underscore", "kp_subtract"}:
            self.change_speed(-1)
        elif key == "n":
            self.skip_requested = True

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

    # Drawing helpers ------------------------------------------------------

    @staticmethod
    def _grade_color(score: float) -> str:
        """Report-card color: green for honor roll, red for flunking."""

        if score >= 90.0:
            return "#57cc99"
        if score >= 80.0:
            return "#a7c957"
        if score >= 70.0:
            return "#e9c46a"
        if score >= 60.0:
            return "#f4a261"
        return "#e5383b"

    @staticmethod
    def _shade(color: str, factor: float) -> str:
        """Darken (<1) or lighten (>1) a #rrggbb color."""

        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        if factor <= 1.0:
            r, g, b = (int(c * factor) for c in (r, g, b))
        else:
            blend = factor - 1.0
            r, g, b = (int(c + (255 - c) * blend) for c in (r, g, b))
        return f"#{min(r, 255):02x}{min(g, 255):02x}{min(b, 255):02x}"

    def _rounded_rect(
        self, x1: float, y1: float, x2: float, y2: float, radius: float, **kwargs: Any
    ) -> int:
        r = min(radius, (x2 - x1) / 2.0, (y2 - y1) / 2.0)
        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
        ]
        return self.canvas.create_polygon(points, smooth=True, **kwargs)

    # Frame ---------------------------------------------------------------

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
        autopilot: str | None = None,
        q_values: np.ndarray | None = None,
        caption: str | None = None,
    ) -> None:
        if self.closed:
            return
        self.canvas.delete("all")
        self._draw_world(env)
        self._draw_autopilot_chip(autopilot)
        if q_values is not None:
            self._draw_q_panel(np.asarray(q_values, dtype=float), action)
        self._draw_dashboard(
            env,
            policy_name=policy_name,
            action=action,
            episode=episode,
            episode_reward=episode_reward,
            epsilon=epsilon,
            caption=caption,
        )
        if self.paused:
            self._draw_overlay("PAUSED", "Press P to continue")
        elif message:
            self._draw_overlay(message, "R: restart   Q/Esc: quit")
        self.process_events()

    # World ----------------------------------------------------------------

    def _draw_world(self, env: DrivingEnv) -> None:
        canvas = self.canvas
        scroll = env.distance_m * self.longitudinal_scale

        # Grass verge with scrolling mow bands for a sense of motion.
        scene_right = 620
        canvas.create_rectangle(0, 0, self.road_left, self.height, fill=VERGE, outline="")
        canvas.create_rectangle(
            self.road_right, 0, scene_right, self.height, fill=VERGE, outline=""
        )
        band_offset = scroll % 96.0
        y = -96.0 + band_offset
        while y < self.height:
            canvas.create_rectangle(0, y, self.road_left, y + 34, fill=VERGE_BAND, outline="")
            canvas.create_rectangle(
                self.road_right, y, scene_right, y + 34, fill=VERGE_BAND, outline=""
            )
            y += 96.0

        # Guardrails: rail plus posts.
        for rail_x in (self.road_left - 11.0, self.road_right + 14.0):
            canvas.create_line(rail_x, 0, rail_x, self.height, fill=GUARDRAIL, width=4)
            canvas.create_line(
                rail_x, 0, rail_x, self.height, fill=self._shade(GUARDRAIL, 1.25), width=1
            )
            post_offset = scroll % 84.0
            y = -84.0 + post_offset
            while y < self.height:
                canvas.create_rectangle(
                    rail_x - 3, y, rail_x + 3, y + 9, fill="#3a434d", outline=""
                )
                y += 84.0

        # Shoulders and asphalt.
        canvas.create_rectangle(
            self.road_left, 0, self.road_right, self.height, fill=SHOULDER, outline=""
        )
        canvas.create_rectangle(
            self.road_left + 14, 0, self.road_right - 14, self.height, fill=ASPHALT, outline=""
        )

        # Subtle scrolling asphalt speckle for texture. A deterministic
        # pseudo-random x per row keeps the pattern stable frame to frame
        # while it scrolls past.
        row_height = 26.0
        rows = int(self.height / row_height) + 2
        for row in range(rows):
            noise = ((row * 2654435761) % 1000) / 1000.0
            x = self.road_left + 24 + noise * (self.road_right - self.road_left - 62)
            speck_y = (row * row_height + scroll) % (self.height + row_height) - row_height
            canvas.create_oval(
                x, speck_y, x + 14, speck_y + 3, fill=ASPHALT_SPECKLE, outline=""
            )

        # Solid edge lines.
        canvas.create_line(
            self.road_left + 14, 0, self.road_left + 14, self.height, fill=EDGE_LINE, width=3
        )
        canvas.create_line(
            self.road_right - 14, 0, self.road_right - 14, self.height, fill=EDGE_LINE, width=3
        )

        # Scrolling dashed lane separators with rounded caps.
        lane_pixel_width = (self.road_right - self.road_left) / env.config.lane_count
        dash_period = 64.0
        dash_offset = scroll % dash_period
        for lane_boundary in range(1, env.config.lane_count):
            x = self.road_left + lane_boundary * lane_pixel_width
            y = -dash_period + dash_offset
            while y < self.height:
                canvas.create_line(
                    x, y, x, y + 30, fill=LANE_DASH, width=3, capstyle=tk.ROUND
                )
                y += dash_period

        self._draw_sensors(env)

        # True collision-rectangle extents in pixels, from the physics.
        lat_scale = (self.road_right - self.road_left) / env.config.road_width_m
        foot_hw = env.config.car_width_m / 2.0 * lat_scale
        foot_hl = env.config.car_length_m / 2.0 * self.longitudinal_scale

        visible_cars = sorted(env.traffic, key=lambda car: car.y_m, reverse=True)
        for car in visible_cars:
            screen_y = self.ego_screen_y - car.y_m * self.longitudinal_scale
            if -140 <= screen_y <= self.height + 140:
                x = self._world_x_to_screen(env, env.traffic_x_m(car))
                if car.behavior == "obstacle":
                    self._draw_obstacle(x, screen_y)
                    self._draw_footprint(x, screen_y, foot_hw, foot_hl)
                    continue
                color = self.traffic_colors[car.color_index % len(self.traffic_colors)]
                self._draw_car(
                    x, screen_y, color, label=f"{car.speed_mps * 2.236936:.0f}"
                )
                self._draw_footprint(x, screen_y, foot_hw, foot_hl)

        ego_x = self._world_x_to_screen(env, env.ego_x_m)
        self._draw_car(ego_x, self.ego_screen_y, EGO_BODY, label="AI", ego=True)
        self._draw_footprint(ego_x, self.ego_screen_y, foot_hw, foot_hl)

        # Sensor legend chip.
        self._rounded_rect(
            self.road_left + 14, 12, self.road_left + 152, 32, 9,
            fill=PANEL_BG, outline=CARD_EDGE,
        )
        canvas.create_text(
            self.road_left + 83,
            22,
            text="RANGE SENSORS",
            fill="#7fb3c8",
            font=("Arial", 8, "bold"),
        )

    def _draw_sensors(self, env: DrivingEnv) -> None:
        sensors = env.sensor_snapshot()
        front = np.asarray(sensors["front_gaps_m"])
        rear = np.asarray(sensors["rear_gaps_m"])
        ego_x = self._world_x_to_screen(env, env.ego_x_m)
        for lane in range(env.config.lane_count):
            lane_x = self._world_x_to_screen(env, env.lane_center(lane))
            front_y = self.ego_screen_y - float(front[lane]) * self.longitudinal_scale
            active = lane == env.current_lane
            color = ACCENT if active else "#2e4a5c"
            self.canvas.create_line(
                ego_x,
                self.ego_screen_y - self.CAR_HALF_L,
                lane_x,
                max(10.0, front_y),
                fill=color,
                width=2 if active else 1,
                dash=(6, 6),
            )
            if float(front[lane]) < env.config.sensor_range_m:
                hit_y = max(10.0, front_y)
                self.canvas.create_oval(
                    lane_x - 3, hit_y - 3, lane_x + 3, hit_y + 3,
                    outline=color, width=2, fill="",
                )
            if rear[lane] < env.config.sensor_range_m:
                rear_y = self.ego_screen_y + float(rear[lane]) * self.longitudinal_scale
                self.canvas.create_line(
                    ego_x,
                    self.ego_screen_y + self.CAR_HALF_L,
                    lane_x,
                    min(self.height - 8.0, rear_y),
                    fill="#223744",
                    width=1,
                    dash=(2, 8),
                )

    # Sprites --------------------------------------------------------------

    def _draw_footprint(self, x: float, y: float, hw: float, hl: float) -> None:
        """Corner brackets marking the exact rectangle the physics crashes on."""

        color = "#f4a6b0"
        arm = 7.0
        for sx in (-1, 1):
            for sy in (-1, 1):
                corner_x, corner_y = x + sx * hw, y + sy * hl
                self.canvas.create_line(
                    corner_x, corner_y, corner_x - sx * arm, corner_y,
                    fill=color, width=2,
                )
                self.canvas.create_line(
                    corner_x, corner_y, corner_x, corner_y - sy * arm,
                    fill=color, width=2,
                )

    def _car_silhouette(self, x: float, y: float, hw: float, hl: float) -> list[float]:
        """Vertices for a crisp top-down sedan, nose pointing up.

        Straight parallel sides with chamfered fender corners — drawn
        unsmoothed so the body reads as sheet metal, not a beetle shell.
        """

        return [
            x - hw * 0.58, y - hl,           # front bumper, left edge
            x + hw * 0.58, y - hl,           # front bumper, right edge
            x + hw * 0.90, y - hl * 0.82,    # right front fender chamfer
            x + hw, y - hl * 0.52,           # right side begins
            x + hw, y + hl * 0.60,           # right side ends (straight)
            x + hw * 0.92, y + hl * 0.86,    # right rear fender chamfer
            x + hw * 0.62, y + hl,           # rear bumper, right edge
            x - hw * 0.62, y + hl,           # rear bumper, left edge
            x - hw * 0.92, y + hl * 0.86,
            x - hw, y + hl * 0.60,
            x - hw, y - hl * 0.52,
            x - hw * 0.90, y - hl * 0.82,
        ]

    def _draw_car(
        self,
        x: float,
        y: float,
        color: str,
        *,
        label: str,
        ego: bool = False,
    ) -> None:
        canvas = self.canvas
        hw, hl = float(self.CAR_HALF_W), float(self.CAR_HALF_L)

        # Drop shadow, offset toward the lower-right light direction.
        canvas.create_polygon(
            self._car_silhouette(x + 4, y + 6, hw, hl),
            fill="#000000",
            stipple="gray50",
            outline="",
        )

        # Halo ring makes the learning agent easy to track.
        if ego:
            self._rounded_rect(
                x - hw - 7, y - hl - 7, x + hw + 7, y + hl + 7, 16,
                fill="", outline=EGO_HALO, width=2,
            )

        # Tires, mostly tucked under the fenders.
        for wheel_y in (y - hl * 0.62, y + hl * 0.38):
            for side in (-1, 1):
                wheel_x = x + side * (hw - 1)
                canvas.create_rectangle(
                    wheel_x - 4, wheel_y, wheel_x + 4, wheel_y + 18,
                    fill=TIRE, outline="",
                )

        # Body: crisp, straight-sided sedan.
        canvas.create_polygon(
            self._car_silhouette(x, y, hw, hl),
            fill=color,
            outline=self._shade(color, 0.45),
            width=1,
        )

        # Front bumper trim.
        canvas.create_line(
            x - hw * 0.52, y - hl + 3, x + hw * 0.52, y - hl + 3,
            fill=self._shade(color, 0.7), width=2,
        )

        # Hood panel with crease lines.
        canvas.create_polygon(
            [
                x - hw * 0.70, y - hl * 0.76,
                x + hw * 0.70, y - hl * 0.76,
                x + hw * 0.80, y - hl * 0.40,
                x - hw * 0.80, y - hl * 0.40,
            ],
            fill=self._shade(color, 1.12),
            outline="",
        )
        for side in (-1, 1):
            canvas.create_line(
                x + side * hw * 0.42, y - hl * 0.74,
                x + side * hw * 0.52, y - hl * 0.42,
                fill=self._shade(color, 0.85), width=1,
            )

        # Windshield.
        canvas.create_polygon(
            [
                x - hw * 0.80, y - hl * 0.38,
                x + hw * 0.80, y - hl * 0.38,
                x + hw * 0.68, y - hl * 0.08,
                x - hw * 0.68, y - hl * 0.08,
            ],
            fill=GLASS,
            outline=GLASS_EDGE,
        )
        canvas.create_line(
            x - hw * 0.70, y - hl * 0.30, x + hw * 0.70, y - hl * 0.30,
            fill=GLASS_SHINE, width=1,
        )

        # Roof.
        self._rounded_rect(
            x - hw * 0.72, y - hl * 0.08, x + hw * 0.72, y + hl * 0.30, 7,
            fill=self._shade(color, 0.90), outline="",
        )

        # Rear window.
        canvas.create_polygon(
            [
                x - hw * 0.66, y + hl * 0.32,
                x + hw * 0.66, y + hl * 0.32,
                x + hw * 0.76, y + hl * 0.54,
                x - hw * 0.76, y + hl * 0.54,
            ],
            fill=GLASS,
            outline=GLASS_EDGE,
        )

        # Trunk lid.
        canvas.create_polygon(
            [
                x - hw * 0.78, y + hl * 0.58,
                x + hw * 0.78, y + hl * 0.58,
                x + hw * 0.70, y + hl * 0.90,
                x - hw * 0.70, y + hl * 0.90,
            ],
            fill=self._shade(color, 1.06),
            outline="",
        )

        # Side mirrors.
        for side in (-1, 1):
            canvas.create_rectangle(
                x + side * (hw + 1), y - hl * 0.34,
                x + side * (hw + 6), y - hl * 0.34 + 6,
                fill=self._shade(color, 0.75), outline="",
            )

        # Wide light bars: headlights up front, tail bars in back.
        for side in (-1, 1):
            self._rounded_rect(
                x + side * hw * 0.42 - hw * 0.20, y - hl + 4,
                x + side * hw * 0.42 + hw * 0.20, y - hl + 9, 3,
                fill=HEADLIGHT, outline="",
            )
            self._rounded_rect(
                x + side * hw * 0.55 - hw * 0.25, y + hl - 9,
                x + side * hw * 0.55 + hw * 0.25, y + hl - 4, 2,
                fill=TAILLIGHT, outline="",
            )

        # Label badge under the car.
        badge_fill = ACCENT_DEEP if ego else "#0d141d"
        badge_w = 15 + 4 * len(label)
        self._rounded_rect(
            x - badge_w / 2, y + hl + 6, x + badge_w / 2, y + hl + 20, 7,
            fill=badge_fill, outline=CARD_EDGE,
        )
        canvas.create_text(
            x, y + hl + 13, text=label, fill="white", font=("Arial", 8, "bold")
        )

    def _draw_obstacle(self, x: float, y: float) -> None:
        canvas = self.canvas
        hw, hl = 40.0, 16.0
        canvas.create_polygon(
            [
                x - hw + 4, y - hl + 6, x + hw + 4, y - hl + 6,
                x + hw + 4, y + hl + 6, x - hw + 4, y + hl + 6,
            ],
            fill="#000000", stipple="gray50", outline="",
        )
        # Barrier board with alternating hazard chevrons.
        self._rounded_rect(
            x - hw, y - hl, x + hw, y + hl, 5,
            fill="#f77f00", outline="#8a4a03", width=1,
        )
        stripe = 14
        inner_left = x - hw + 6
        inner_right = x + hw - 6
        sx = inner_left
        toggle = True
        while sx < inner_right:
            end = min(sx + stripe, inner_right)
            if toggle:
                canvas.create_polygon(
                    sx, y + hl - 5, end, y - hl + 5,
                    min(end + 7, inner_right), y - hl + 5, min(sx + 7, inner_right), y + hl - 5,
                    fill="#f4f1de", outline="",
                )
            toggle = not toggle
            sx += stripe
        # End posts.
        for side in (-1, 1):
            canvas.create_rectangle(
                x + side * hw - 3, y - hl - 5, x + side * hw + 3, y + hl + 5,
                fill="#3a434d", outline="",
            )

    def _draw_autopilot_chip(self, autopilot: str | None) -> None:
        """Status chip for the manual-mode autopilot (SPACE to toggle)."""

        if autopilot is None:
            return
        road_cx = (self.road_left + self.road_right) / 2.0
        if autopilot == "on":
            text, color = "AUTOPILOT ENGAGED", "#57cc99"
        elif autopilot == "missing":
            text, color = "NO AUTOPILOT MODEL - TRAIN WITH --handover", "#e5383b"
        else:
            text, color = "SPACE - AUTOPILOT", "#44586a"
        chip_w = 18 + 7.2 * len(text)
        self._rounded_rect(
            road_cx - chip_w / 2, 44, road_cx + chip_w / 2, 68, 11,
            fill=PANEL_BG, outline=color, width=2 if autopilot == "on" else 1,
        )
        self.canvas.create_text(
            road_cx, 56, text=text,
            fill=color if autopilot != "off" else TEXT_DIM,
            font=("Arial", 9, "bold"),
        )

    def _draw_q_panel(self, q_values: np.ndarray, action: int) -> None:
        """Live window into the value function: one bar per action.

        Bar lengths are min-max normalized within the current frame (Q-values
        are only meaningful relative to each other), so the longest bar is
        always the action the network likes most right now.
        """

        labels = ("MAINTAIN", "ACCEL", "BRAKE", "LEFT", "RIGHT")
        panel_left, panel_right = 8, 102
        panel_top = 464
        row_height = 38
        panel_bottom = panel_top + 34 + row_height * len(labels)
        self._rounded_rect(
            panel_left, panel_top, panel_right, panel_bottom, 10,
            fill=PANEL_BG, outline=CARD_EDGE,
        )
        self.canvas.create_text(
            (panel_left + panel_right) / 2, panel_top + 15,
            text="Q-VALUES", fill="#7fb3c8", font=("Arial", 8, "bold"),
        )
        low = float(np.min(q_values))
        span = float(np.max(q_values) - low)
        best = int(np.argmax(q_values))
        bar_left = panel_left + 8
        bar_max = panel_right - 8 - bar_left
        y = panel_top + 30
        for index, label in enumerate(labels):
            is_best = index == best
            is_chosen = index == int(action)
            name_color = ACCENT if is_best else TEXT_DIM
            self.canvas.create_text(
                bar_left, y + 6, text=label, anchor="w",
                fill=name_color, font=("Arial", 8, "bold"),
            )
            self.canvas.create_text(
                panel_right - 8, y + 6, text=f"{q_values[index]:.1f}", anchor="e",
                fill=TEXT_MAIN if is_best else TEXT_DIM, font=("Arial", 7),
            )
            fraction = (float(q_values[index]) - low) / span if span > 1e-9 else 0.5
            fill_w = 4 + fraction * (bar_max - 4)
            self._rounded_rect(
                bar_left, y + 14, bar_left + bar_max, y + 24, 5,
                fill=CARD_BG, outline="",
            )
            self._rounded_rect(
                bar_left, y + 14, bar_left + fill_w, y + 24, 5,
                fill=ACCENT_DEEP if is_best else "#33566b", outline="",
            )
            if is_chosen:
                self._rounded_rect(
                    bar_left - 3, y + 11, bar_left + bar_max + 3, y + 27, 7,
                    fill="", outline=ACCENT, width=1,
                )
            y += row_height

    # Dashboard ------------------------------------------------------------

    def _draw_dashboard(
        self,
        env: DrivingEnv,
        *,
        policy_name: str,
        action: int,
        episode: int,
        episode_reward: float,
        epsilon: float | None,
        caption: str | None = None,
    ) -> None:
        canvas = self.canvas
        panel_left = 620
        canvas.create_rectangle(panel_left, 0, self.width, self.height, fill=PANEL_BG, outline="")
        canvas.create_line(panel_left, 0, panel_left, self.height, fill=CARD_EDGE, width=2)

        left = panel_left + 18
        right = self.width - 18

        title = canvas.create_text(
            left, 34, text="AUTODRIVE", anchor="w", fill=TEXT_MAIN, font=("Arial", 20, "bold")
        )
        title_end = canvas.bbox(title)[2]
        canvas.create_text(
            title_end + 9, 34, text="RL LAB", anchor="w", fill=ACCENT,
            font=("Arial", 20, "bold"),
        )
        canvas.create_text(
            left, 60, text=caption or "A small car learning a big idea", anchor="w",
            fill=ACCENT if caption else TEXT_DIM,
            font=("Arial", 10, "bold") if caption else ("Arial", 10),
        )

        # Playback rate. Only worth the pixels when it is not real time, but
        # then it matters a great deal — at 20x it is the difference between
        # "the agent is broken" and "you are watching four seconds of a run".
        if self.speed != 1.0:
            chip = f"{self.speed:g}x".replace(".0x", "x")
            canvas.create_text(
                right, 34, text=chip, anchor="e", fill=ACCENT,
                font=("Arial", 13, "bold"),
            )
            canvas.create_text(
                right, 52, text="+ / -  speed", anchor="e", fill=TEXT_DIM,
                font=("Arial", 8),
            )

        # Speedometer arc.
        gauge_cx, gauge_cy, gauge_r = (left + right) / 2, 158, 66
        speed = env.ego_speed_mps
        fraction = min(1.0, speed / env.config.max_speed_mps)
        canvas.create_arc(
            gauge_cx - gauge_r, gauge_cy - gauge_r, gauge_cx + gauge_r, gauge_cy + gauge_r,
            start=-30, extent=240, style=tk.ARC, outline="#1a2530", width=10,
        )
        if fraction > 0.003:
            canvas.create_arc(
                gauge_cx - gauge_r, gauge_cy - gauge_r, gauge_cx + gauge_r, gauge_cy + gauge_r,
                start=210, extent=-240 * fraction, style=tk.ARC, outline=ACCENT, width=10,
            )
        canvas.create_text(
            gauge_cx, gauge_cy - 6, text=f"{speed * 2.236936:.0f}",
            fill=TEXT_MAIN, font=("Arial", 30, "bold"),
        )
        canvas.create_text(
            gauge_cx, gauge_cy + 22, text="mph", fill=TEXT_DIM, font=("Arial", 10, "bold")
        )
        # Red tick marking the posted speed limit on the gauge.
        limit_fraction = min(1.0, env.config.speed_limit_mps / env.config.max_speed_mps)
        limit_angle = math.radians(210.0 - 240.0 * limit_fraction)
        cos_a, sin_a = math.cos(limit_angle), math.sin(limit_angle)
        canvas.create_line(
            gauge_cx + (gauge_r - 9) * cos_a, gauge_cy - (gauge_r - 9) * sin_a,
            gauge_cx + (gauge_r + 8) * cos_a, gauge_cy - (gauge_r + 8) * sin_a,
            fill=TAILLIGHT, width=3,
        )

        card_w, card_h, gap = 172, 50, 10

        # Live driver's report card: letter grade plus rolling score.
        score = float(getattr(env, "grade_score", 100.0))
        letter = grade_letter(score)
        grade_color = self._grade_color(score)
        banner_w = 2 * card_w + gap
        self._rounded_rect(
            left, 240, left + banner_w, 292, 10, fill=CARD_BG, outline=grade_color
        )
        canvas.create_text(
            left + 38, 266, text=letter, fill=grade_color, font=("Arial", 26, "bold")
        )
        canvas.create_text(
            left + 76, 256, text="DRIVER GRADE", anchor="w", fill=TEXT_DIM,
            font=("Arial", 8, "bold"),
        )
        bar_left, bar_right = left + 76, left + banner_w - 62
        self._rounded_rect(bar_left, 268, bar_right, 280, 6, fill="#1a2530", outline="")
        fill_end = bar_left + (score / 100.0) * (bar_right - bar_left)
        if fill_end > bar_left + 6:
            self._rounded_rect(bar_left, 268, fill_end, 280, 6, fill=grade_color, outline="")
        canvas.create_text(
            left + banner_w - 14, 274, text=f"{score:3.0f}", anchor="e",
            fill=TEXT_MAIN, font=("Arial", 12, "bold"),
        )

        # Metric cards, two columns.
        metrics = [
            ("POLICY", policy_name.upper()),
            ("ACTION", ACTION_NAMES[Action(action)].upper()),
            ("LANE", f"{env.current_lane + 1} / {env.config.lane_count}"),
            ("DISTANCE", f"{env.distance_m:,.0f} m"),
            ("EPISODE", str(episode)),
            ("RETURN", f"{episode_reward:,.1f}"),
        ]
        top = 306
        for index, (label, value) in enumerate(metrics):
            column = index % 2
            row = index // 2
            cx1 = left + column * (card_w + gap)
            cy1 = top + row * (card_h + gap)
            self._rounded_rect(
                cx1, cy1, cx1 + card_w, cy1 + card_h, 9, fill=CARD_BG, outline=CARD_EDGE
            )
            canvas.create_text(
                cx1 + 12, cy1 + 15, text=label, anchor="w", fill=TEXT_DIM,
                font=("Arial", 8, "bold"),
            )
            canvas.create_text(
                cx1 + 12, cy1 + 34, text=value, anchor="w", fill=TEXT_MAIN,
                font=("Arial", 12, "bold"),
            )

        y = top + 3 * (card_h + gap) + 8
        if epsilon is not None:
            canvas.create_text(
                left, y, text="EXPLORATION ε", anchor="w", fill=TEXT_DIM,
                font=("Arial", 8, "bold"),
            )
            bar_left, bar_right = left + 110, right - 56
            self._rounded_rect(bar_left, y - 6, bar_right, y + 6, 6, fill=CARD_BG, outline="")
            fill_end = bar_left + epsilon * (bar_right - bar_left)
            if fill_end > bar_left + 6:
                self._rounded_rect(bar_left, y - 6, fill_end, y + 6, 6, fill="#8d6fb8", outline="")
            canvas.create_text(
                right, y, text=f"{epsilon:.3f}", anchor="e", fill=TEXT_MAIN,
                font=("Arial", 10, "bold"),
            )
            y += 30

        # Per-lane clearance bars.
        canvas.create_text(
            left, y, text="FRONT CLEARANCE BY LANE", anchor="w", fill=TEXT_DIM,
            font=("Arial", 8, "bold"),
        )
        y += 26
        sensors = env.sensor_snapshot()
        front = np.asarray(sensors["front_gaps_m"])
        for lane, gclearance in enumerate(front):
            active = lane == env.current_lane
            canvas.create_text(
                left, y, text=f"L{lane + 1}", anchor="w",
                fill=ACCENT if active else TEXT_DIM, font=("Arial", 10, "bold"),
            )
            bar_left, bar_right = left + 34, right - 62
            fraction = min(1.0, float(gclearance) / env.config.sensor_range_m)
            self._rounded_rect(bar_left, y - 7, bar_right, y + 7, 7, fill=CARD_BG, outline="")
            fill_end = bar_left + fraction * (bar_right - bar_left)
            if fill_end > bar_left + 7:
                self._rounded_rect(
                    bar_left, y - 7, fill_end, y + 7, 7,
                    fill=ACCENT_DEEP if active else "#33566b", outline="",
                )
            canvas.create_text(
                right, y, text=f"{gclearance:4.0f} m", anchor="e", fill=TEXT_MAIN,
                font=("Arial", 9, "bold"),
            )
            y += 30

        # Keycap-styled controls. Two rows now: the new keys matter most to
        # someone watching a long training replay, which is when they most
        # need to be discoverable without reading the README.
        controls = [("P", "pause"), ("R", "restart"), ("Q", "quit")]
        extras = [("N", "next"), ("+/-", "speed")]
        kx = left
        ky = self.height - 72
        for key, meaning in extras:
            width = 24 if len(key) == 1 else 34
            self._rounded_rect(kx, ky, kx + width, ky + 22, 6, fill=CARD_BG, outline=CARD_EDGE)
            canvas.create_text(
                kx + width / 2, ky + 11, text=key, fill=TEXT_MAIN,
                font=("Arial", 10, "bold"),
            )
            canvas.create_text(
                kx + width + 8, ky + 11, text=meaning, anchor="w", fill=TEXT_DIM,
                font=("Arial", 10),
            )
            kx += width + 16 + 9 * len(meaning)

        kx = left
        ky = self.height - 42
        for key, meaning in controls:
            self._rounded_rect(kx, ky, kx + 24, ky + 22, 6, fill=CARD_BG, outline=CARD_EDGE)
            canvas.create_text(
                kx + 12, ky + 11, text=key, fill=TEXT_MAIN, font=("Arial", 10, "bold")
            )
            canvas.create_text(
                kx + 32, ky + 11, text=meaning, anchor="w", fill=TEXT_DIM, font=("Arial", 10)
            )
            kx += 40 + 9 * len(meaning)

    def _draw_overlay(self, heading: str, subheading: str) -> None:
        canvas = self.canvas
        canvas.create_rectangle(
            0, 0, self.width, self.height, fill="#000000", stipple="gray50", outline=""
        )
        cx, cy = (self.road_left + self.road_right) / 2.0, 348
        head = canvas.create_text(
            cx, cy - 18, text=heading, fill=TEXT_MAIN, font=("Arial", 20, "bold")
        )
        sub = canvas.create_text(
            cx, cy + 22, text=subheading, fill=TEXT_DIM, font=("Arial", 11)
        )
        # Size the card to its content.
        hx1, hy1, hx2, hy2 = canvas.bbox(head)
        sx1, sy1, sx2, sy2 = canvas.bbox(sub)
        left = min(hx1, sx1) - 30
        right = max(hx2, sx2) + 30
        top = min(hy1, sy1) - 26
        bottom = max(hy2, sy2) + 26
        self._rounded_rect(
            left + 6, top + 6, right + 6, bottom + 6, 16,
            fill="#000000", stipple="gray50",
        )
        self._rounded_rect(left, top, right, bottom, 14, fill="#10161f", outline=ACCENT, width=2)
        canvas.tag_raise(head)
        canvas.tag_raise(sub)

    # Utilities ------------------------------------------------------------

    def _world_x_to_screen(self, env: DrivingEnv, x_m: float) -> float:
        fraction = (x_m + env.config.road_half_width_m) / env.config.road_width_m
        return self.road_left + fraction * (self.road_right - self.road_left)

    @classmethod
    def _nearest_speed(cls, value: float) -> float:
        return min(cls.SPEEDS, key=lambda option: abs(option - float(value)))

    def change_speed(self, direction: int) -> float:
        """Step one place up or down the rate ladder, clamped at the ends."""

        index = self.SPEEDS.index(self.speed)
        self.speed = self.SPEEDS[
            max(0, min(len(self.SPEEDS) - 1, index + direction))
        ]
        return self.speed

    def tick(self) -> None:
        frame_duration = 1.0 / (self.fps * self.speed)
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