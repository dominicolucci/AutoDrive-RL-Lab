"""Tkinter renderer for the top-down driving simulation.

Visual design notes
-------------------
The scene uses a layered, modern semi-realistic style drawn entirely with
Tkinter canvas primitives:

- Cars are composite sprites: a soft drop shadow, wheels tucked under the
  body, a spline-smoothed silhouette, tinted glass with a highlight, side
  mirrors, and head/tail lights. The ego car wears the accent blue and a
  halo ring so the eye finds it instantly.
- Sprites are drawn close to their physical proportions so two cars that
  look close on screen really are close in the simulation.
- The road is built from layers (verge, guardrail, shoulder, asphalt,
  edge lines, scrolling lane dashes and asphalt speckle) to create motion
  and depth without hurting the frame rate.
- The dashboard is a card-based panel with a speedometer arc, metric
  chips, per-lane clearance bars, and keycap-styled controls.
"""

from __future__ import annotations

import time
import tkinter as tk
from typing import Any

import numpy as np

from .environment import ACTION_NAMES, Action, DrivingEnv

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
    road_left = 65
    road_right = 600
    ego_screen_y = 560
    longitudinal_scale = 5.5

    traffic_colors = TRAFFIC_COLORS

    # Kept close to physical proportions relative to the longitudinal
    # scale, so cars that look close on screen really are close.
    CAR_HALF_W = 16
    CAR_HALF_L = 24

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

    # World ----------------------------------------------------------------

    def _draw_world(self, env: DrivingEnv) -> None:
        canvas = self.canvas
        scroll = env.distance_m * self.longitudinal_scale

        # Grass verge with scrolling mow bands for a sense of motion.
        canvas.create_rectangle(0, 0, self.road_left, self.height, fill=VERGE, outline="")
        canvas.create_rectangle(
            self.road_right, 0, self.road_right + 20, self.height, fill=VERGE, outline=""
        )
        band_offset = scroll % 96.0
        y = -96.0 + band_offset
        while y < self.height:
            canvas.create_rectangle(0, y, self.road_left, y + 34, fill=VERGE_BAND, outline="")
            canvas.create_rectangle(
                self.road_right, y, self.road_right + 20, y + 34, fill=VERGE_BAND, outline=""
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

        visible_cars = sorted(env.traffic, key=lambda car: car.y_m, reverse=True)
        for car in visible_cars:
            screen_y = self.ego_screen_y - car.y_m * self.longitudinal_scale
            if -70 <= screen_y <= self.height + 70:
                x = self._world_x_to_screen(env, env.traffic_x_m(car))
                if car.behavior == "obstacle":
                    self._draw_obstacle(x, screen_y)
                    continue
                color = self.traffic_colors[car.color_index % len(self.traffic_colors)]
                self._draw_car(
                    x, screen_y, color, label=f"{car.speed_mps * 2.236936:.0f}"
                )

        ego_x = self._world_x_to_screen(env, env.ego_x_m)
        self._draw_car(ego_x, self.ego_screen_y, EGO_BODY, label="AI", ego=True)

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

    def _car_silhouette(self, x: float, y: float, hw: float, hl: float) -> list[float]:
        """Spline control points for a sedan seen from above, nose up."""

        return [
            x, y - hl,                     # nose center
            x + hw * 0.62, y - hl + 2,     # nose corner
            x + hw, y - hl * 0.42,         # front shoulder
            x + hw, y + hl * 0.34,         # rear haunch
            x + hw * 0.78, y + hl - 2,     # tail corner
            x, y + hl,                     # tail center
            x - hw * 0.78, y + hl - 2,
            x - hw, y + hl * 0.34,
            x - hw, y - hl * 0.42,
            x - hw * 0.62, y - hl + 2,
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
            self._car_silhouette(x + 3, y + 5, hw, hl),
            smooth=True,
            splinesteps=12,
            fill="#000000",
            stipple="gray50",
            outline="",
        )

        # Halo ring makes the learning agent easy to track.
        if ego:
            self._rounded_rect(
                x - hw - 6, y - hl - 6, x + hw + 6, y + hl + 6, 14,
                fill="", outline=EGO_HALO, width=2,
            )

        # Wheels peeking from under the body.
        for wheel_y in (y - hl * 0.52, y + hl * 0.40):
            for side in (-1, 1):
                wheel_x = x + side * hw
                canvas.create_rectangle(
                    wheel_x - 4, wheel_y, wheel_x + 4, wheel_y + 13,
                    fill=TIRE, outline="",
                )

        # Body.
        canvas.create_polygon(
            self._car_silhouette(x, y, hw, hl),
            smooth=True,
            splinesteps=12,
            fill=color,
            outline=self._shade(color, 0.5),
            width=1,
        )

        # Hood sheen.
        canvas.create_polygon(
            [
                x, y - hl + 5,
                x + hw * 0.34, y - hl * 0.5,
                x, y - hl * 0.3,
                x - hw * 0.34, y - hl * 0.5,
            ],
            smooth=True,
            fill=self._shade(color, 1.18),
            outline="",
        )

        # Windshield.
        canvas.create_polygon(
            [
                x - hw * 0.64, y - hl * 0.34,
                x + hw * 0.64, y - hl * 0.34,
                x + hw * 0.52, y - hl * 0.02,
                x - hw * 0.52, y - hl * 0.02,
            ],
            fill=GLASS,
            outline=GLASS_EDGE,
        )
        canvas.create_line(
            x - hw * 0.52, y - hl * 0.27, x + hw * 0.52, y - hl * 0.27,
            fill=GLASS_SHINE, width=1,
        )

        # Roof panel.
        self._rounded_rect(
            x - hw * 0.58, y - hl * 0.02, x + hw * 0.58, y + hl * 0.42, 6,
            fill=self._shade(color, 0.9), outline="",
        )

        # Rear window.
        canvas.create_polygon(
            [
                x - hw * 0.50, y + hl * 0.44,
                x + hw * 0.50, y + hl * 0.44,
                x + hw * 0.58, y + hl * 0.64,
                x - hw * 0.58, y + hl * 0.64,
            ],
            fill=GLASS,
            outline=GLASS_EDGE,
        )

        # Side mirrors.
        for side in (-1, 1):
            canvas.create_rectangle(
                x + side * (hw + 1), y - hl * 0.30,
                x + side * (hw + 5), y - hl * 0.30 + 5,
                fill=self._shade(color, 0.75), outline="",
            )

        # Headlights and taillights, tucked just inside the body curve.
        for side in (-1, 1):
            canvas.create_oval(
                x + side * hw * 0.45 - 3, y - hl + 3,
                x + side * hw * 0.45 + 3, y - hl + 8,
                fill=HEADLIGHT, outline="",
            )
            canvas.create_rectangle(
                x + side * hw * 0.55 - 5, y + hl - 6,
                x + side * hw * 0.55 + 5, y + hl - 3,
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
        hw, hl = 24.0, 12.0
        canvas.create_polygon(
            [
                x - hw + 3, y - hl + 5, x + hw + 3, y - hl + 5,
                x + hw + 3, y + hl + 5, x - hw + 3, y + hl + 5,
            ],
            fill="#000000", stipple="gray50", outline="",
        )
        # Barrier board with alternating hazard chevrons.
        self._rounded_rect(
            x - hw, y - hl, x + hw, y + hl, 5,
            fill="#f77f00", outline="#8a4a03", width=1,
        )
        stripe = 12
        inner_left = x - hw + 5
        inner_right = x + hw - 5
        sx = inner_left
        toggle = True
        while sx < inner_right:
            end = min(sx + stripe, inner_right)
            if toggle:
                canvas.create_polygon(
                    sx, y + hl - 5, end, y - hl + 5,
                    min(end + 6, inner_right), y - hl + 5, min(sx + 6, inner_right), y + hl - 5,
                    fill="#f4f1de", outline="",
                )
            toggle = not toggle
            sx += stripe
        # End posts.
        for side in (-1, 1):
            canvas.create_rectangle(
                x + side * hw - 3, y - hl - 4, x + side * hw + 3, y + hl + 4,
                fill="#3a434d", outline="",
            )

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
            left, 60, text="A small car learning a big idea", anchor="w",
            fill=TEXT_DIM, font=("Arial", 10),
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

        # Metric cards, two columns.
        metrics = [
            ("POLICY", policy_name.upper()),
            ("ACTION", ACTION_NAMES[Action(action)].upper()),
            ("LANE", f"{env.current_lane + 1} / {env.config.lane_count}"),
            ("DISTANCE", f"{env.distance_m:,.0f} m"),
            ("EPISODE", str(episode)),
            ("RETURN", f"{episode_reward:,.1f}"),
        ]
        card_w, card_h, gap = 172, 50, 10
        top = 246
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

        # Keycap-styled controls.
        controls = [("P", "pause"), ("R", "restart"), ("Q", "quit")]
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
        cx, cy = 332, 348
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