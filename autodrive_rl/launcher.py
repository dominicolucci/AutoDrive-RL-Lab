"""Desktop launcher — the front door to AutoDrive RL Lab.

Everything the project can do from a terminal is a form here: drive the
simulation, train an agent, clone your own driving, benchmark policies against
each other. Pick settings, press Run, watch the output stream in.

The design rule is that this window builds command lines and runs them
(`jobrunner`), and the translation from widgets to arguments lives in
`commands`. Nothing about training, evaluation or cloning is reimplemented
here — so the GUI cannot drift out of step with the CLI, and the parts worth
testing are testable without a display.

The command being run is shown above the Run button. That is deliberate in a
teaching project: you can see exactly what your clicks produce, copy it, and
learn the command line by using the interface that replaces it.

Styling lives in `theme`, which borrows the simulation's own palette so the
launcher and the thing it launches look like one application.

    python -m autodrive_rl.launcher
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .commands import (
    BASELINE_POLICIES,
    BENCHMARK_CELL_CHOICES,
    BENCHMARK_CELLS,
    DRIVE_POLICIES,
    PRESETS,
    SCENARIOS,
    Assets,
    build_args,
    discover_assets,
    validate,
)
from .jobrunner import Job, JobState, format_command, module_command
from .theme import (
    ACCENT,
    BG,
    CARD_BG,
    CARD_EDGE,
    ERROR,
    OK,
    PANEL_BG,
    TEXT_DIM,
    TEXT_MAIN,
    WARN,
    apply_theme,
    enable_dpi_awareness,
    style_listbox,
    style_text,
)

POLL_MS = 120
LOG_LIMIT = 4000  # lines retained; older ones are trimmed so memory stays flat

MODE_BLURB = {
    "Overview": "What the simulation is, what the agent senses, and what each tab does.",
    "Drive": "Watch a policy drive, or take the wheel yourself.",
    "Train": "Teach an agent to drive by trial and reward.",
    "Clone": "Learn to drive by imitating recorded human demonstrations.",
    "Benchmark": "Score policies on identical held-out worlds, by difficulty.",
}


# ── Small widget helpers ─────────────────────────────────────────────────────


class Field:
    """One labelled row: label on the left, controls centre, hint on the right."""

    def __init__(self, parent: tk.Widget, row: int, label: str, hint: str = "") -> None:
        ttk.Label(parent, text=label, style="Card.TLabel").grid(
            row=row, column=0, sticky="w", padx=(0, 18), pady=5
        )
        self.holder = ttk.Frame(parent, style="Card.TFrame")
        self.holder.grid(row=row, column=1, sticky="ew", pady=5)
        parent.columnconfigure(1, weight=1)
        if hint:
            ttk.Label(parent, text=hint, style="Hint.TLabel").grid(
                row=row, column=2, sticky="w", padx=(20, 0)
            )


def spacer(parent: tk.Widget, text: str) -> None:
    ttk.Label(parent, text=text, style="Card.TLabel").pack(side="left", padx=(10, 6))


def combo(parent, values, initial, width=24, on_change=None) -> tk.StringVar:
    var = tk.StringVar(value=initial)
    box = ttk.Combobox(parent, textvariable=var, values=list(values), state="readonly", width=width)
    box.pack(side="left")
    if on_change:
        box.bind("<<ComboboxSelected>>", lambda _e: on_change())
    return var


def entry(parent, initial, width=12, on_change=None) -> tk.StringVar:
    var = tk.StringVar(value="" if initial is None else str(initial))
    ttk.Entry(parent, textvariable=var, width=width).pack(side="left")
    if on_change:
        var.trace_add("write", lambda *_: on_change())
    return var


def check(parent, text: str, initial: bool, on_change=None) -> tk.BooleanVar:
    var = tk.BooleanVar(value=initial)
    ttk.Checkbutton(parent, text=text, variable=var, command=on_change,
                    style="TCheckbutton").pack(side="left", padx=(0, 12))
    return var


def as_int(var: tk.StringVar, fallback: int | None = None) -> int | None:
    try:
        return int(var.get().strip())
    except (ValueError, AttributeError):
        return fallback


def as_float(var: tk.StringVar, fallback: float | None = None) -> float | None:
    try:
        return float(var.get().strip())
    except (ValueError, AttributeError):
        return fallback


# ── Panels ───────────────────────────────────────────────────────────────────


class Disclosure:
    """A collapsible block of secondary settings.

    Every panel used to show everything at once, which made the common case —
    pick a policy, pick a difficulty, press go — hard to find inside fifteen
    controls. The split is by *how often a setting is touched*, not by how
    advanced it is: seeds, paths and overrides are perfectly ordinary, they
    are just rarely changed twice in a row.
    """

    def __init__(self, parent: tk.Widget, row: int, label: str = "More settings") -> None:
        self.label = label
        self.open = False
        self.header = ttk.Label(parent, style="Toggle.TLabel", cursor="hand2")
        self.header.grid(row=row, column=0, columnspan=3, sticky="w", pady=(16, 2))
        self.header.bind("<Button-1>", lambda _e: self.toggle())
        self.header.bind(
            "<Enter>", lambda _e: self.header.configure(style="ToggleHover.TLabel")
        )
        self.header.bind(
            "<Leave>", lambda _e: self.header.configure(style="Toggle.TLabel")
        )

        self.body = ttk.Frame(parent, style="Card.TFrame")
        self.body.grid(row=row + 1, column=0, columnspan=3, sticky="ew")
        self.body.grid_columnconfigure(1, weight=1)
        self.body.grid_remove()
        self._paint()

    def _paint(self) -> None:
        arrow = "\u25be" if self.open else "\u25b8"
        self.header.configure(text=f"{arrow}  {self.label}")

    def toggle(self) -> None:
        self.open = not self.open
        if self.open:
            self.body.grid()
        else:
            self.body.grid_remove()
        self._paint()


class Panel(ttk.Frame):
    """Base for the four mode panels."""

    module = ""
    run_label = "Run"

    def __init__(self, parent: tk.Widget, app: "LauncherApp") -> None:
        super().__init__(parent, style="Card.TFrame", padding=(24, 16, 24, 18))
        self.app = app
        self.fonts = app.fonts
        self.grid_columnconfigure(1, weight=1)
        self.build()

    def build(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def values(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def refresh_assets(self, assets: Assets) -> None:
        """Re-point dropdowns at whatever is on disk now."""

    def changed(self) -> None:
        self.app.update_preview()

    def combo_widgets(self, tag: str) -> list[ttk.Combobox]:
        """Comboboxes registered under a tag, so assets can be refreshed."""
        return getattr(self, f"_combos_{tag}", [])

    def register_combo(self, tag: str, widget_parent: tk.Widget) -> None:
        boxes = [w for w in widget_parent.winfo_children() if isinstance(w, ttk.Combobox)]
        setattr(self, f"_combos_{tag}", boxes)


class OverviewPanel(Panel):
    """What this is, before any settings are touched.

    The other four tabs assume you already know what an observation is, what
    the presets change, and why a safety percentage on its own is misleading.
    This one does not.
    """

    module = ""
    run_label = "Open the Drive tab"

    SECTIONS: tuple[tuple[str, str], ...] = (
        (
            "The car",
            "Three lanes, no map, no route. Every tenth of a second it picks one "
            "of five controls - hold, accelerate, brake, steer left, steer right - "
            "from 16 sensor readings: its own speed and position, and the gap and "
            "closing speed to the nearest car ahead and behind in each lane.",
        ),
        (
            "The scoring",
            "Rewarded for covering ground at a sensible speed, centred in a lane. "
            "Penalised for tailgating, speeding, cutting people up, near misses, "
            "crashing, leaving the road, and blocking a live lane. Get that "
            "balance wrong and it finds a loophole instead of learning to drive.",
        ),
        (
            "The difficulties",
            "sparse is 4 cars. normal is 9. dense adds slow vehicles and lane "
            "changers. unforgiving adds traffic closing from behind, most of it "
            "driven by someone looking at their phone.",
        ),
        (
            "The tabs",
            "Drive watches a policy or hands you the keys. Train teaches one by "
            "trial and reward. Clone copies your own driving instead. Benchmark "
            "scores any of them on identical held-out worlds.",
        ),
        (
            "Seeing it learn",
            "Train can draw every Nth episode at up to 20x and save a checkpoint "
            "every N episodes. Point Drive's checkpoint sequence at those and the "
            "whole run replays as one clip: crashing, wandering, then driving.",
        ),
    )

    def build(self) -> None:
        ttk.Label(
            self,
            text="An agent learning to drive, and the instruments to see whether it did.",
            style="Intro.TLabel",
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        for index, (heading, body) in enumerate(self.SECTIONS, start=1):
            ttk.Label(self, text=heading, style="Card.TLabel").grid(
                row=index, column=0, sticky="nw", padx=(0, 18), pady=3
            )
            ttk.Label(
                self, text=body, style="Hint.TLabel", justify="left", wraplength=700
            ).grid(row=index, column=1, columnspan=2, sticky="w", pady=3)

        ttk.Label(
            self,
            text=(
                "In the simulation window:  W A S D or arrows to drive  ·  "
                "P pause  ·  R restart  ·  N next episode  ·  + / - speed  ·  Q quit"
            ),
            style="Hint.TLabel",
            justify="left",
        ).grid(row=len(self.SECTIONS) + 1, column=0, columnspan=3, sticky="w", pady=(12, 0))

    def values(self) -> dict[str, Any]:
        return {}


class DrivePanel(Panel):
    module = "play"
    run_label = "\u25b6  Drive"

    def build(self) -> None:
        change = self.changed

        # Everyday: who is driving, in what traffic, how fast to play it back.
        f = Field(self, 0, "Who drives", "manual puts you at the keyboard")
        self.policy = combo(f.holder, DRIVE_POLICIES, "heuristic", 20, change)

        f = Field(self, 1, "Model", "used by the 'dqn' policy")
        self.model = combo(f.holder, (), "", 36, change)
        self.register_combo("model", f.holder)

        f = Field(self, 2, "Difficulty", "unforgiving adds traffic from behind")
        self.preset = combo(f.holder, PRESETS, "normal", 16, change)

        f = Field(self, 3, "Speed", "1 is real time; + / - adjust it live")
        self.speed = combo(f.holder, ("0.5", "1", "2", "5", "10", "20"), "1", 6, change)
        spacer(f.holder, "x")

        adv = Disclosure(self, 4)
        body = adv.body
        self.advanced = adv

        f = Field(body, 0, "Replay a run", "plays a whole training run as one clip")
        self.sequence_enabled = check(f.holder, "checkpoint sequence", False, change)
        self.sequence = entry(f.holder, "models/autodrive_dqn_ep*.npz", 30, change)

        f = Field(body, 1, "Record", "captures demonstrations for cloning")
        self.record_enabled = check(f.holder, "save my driving to", False, change)
        self.record_path = entry(f.holder, "demos/me.npz", 22, change)

        f = Field(body, 2, "Autopilot", "hold SPACE to hand over control")
        self.autopilot_enabled = check(f.holder, "enable", False, change)
        self.autopilot_model = combo(f.holder, (), "", 28, change)
        self.register_combo("autopilot", f.holder)

        f = Field(body, 3, "World", "'lane' drops the traffic entirely")
        self.scenario = combo(f.holder, SCENARIOS, "traffic", 12, change)

        f = Field(body, 4, "Overrides", "leave blank to use the difficulty preset")
        spacer(f.holder, "cars")
        self.traffic = entry(f.holder, "", 5, change)
        spacer(f.holder, "slow")
        self.slow_vehicles = entry(f.holder, "", 5, change)
        spacer(f.holder, "reactive")
        self.reactive = entry(f.holder, "", 6, change)

        f = Field(body, 5, "Seed")
        self.seed = entry(f.holder, 7, 8, change)
        spacer(f.holder, "frames / sec")
        self.fps = entry(f.holder, 30, 8, change)

    def refresh_assets(self, assets: Assets) -> None:
        names = list(assets.model_names)
        for tag, var in (("model", self.model), ("autopilot", self.autopilot_model)):
            for box in self.combo_widgets(tag):
                box.configure(values=names)
            if var.get() not in names:
                var.set(names[0] if names else "")

    def values(self) -> dict[str, Any]:
        return {
            "policy": self.policy.get(),
            "model": self.model.get(),
            "scenario": self.scenario.get(),
            "preset": self.preset.get(),
            "traffic": as_int(self.traffic),
            "slow_vehicles": as_int(self.slow_vehicles),
            "reactive": as_float(self.reactive),
            "seed": as_int(self.seed, 7),
            "fps": as_int(self.fps, 30),
            "speed": as_float(self.speed, 1.0),
            "sequence_enabled": self.sequence_enabled.get(),
            "model_sequence": self.sequence.get().strip(),
            "record_enabled": self.record_enabled.get(),
            "record_path": self.record_path.get().strip(),
            "autopilot_enabled": self.autopilot_enabled.get(),
            "autopilot_model": self.autopilot_model.get(),
        }


class TrainPanel(Panel):
    module = "train"
    run_label = "\u25b6  Start training"

    def build(self) -> None:
        change = self.changed

        # Everyday: how long, how hard, and whether you want to watch.
        f = Field(self, 0, "Length", "300 episodes \u2248 a few minutes")
        self.episodes = entry(f.holder, 300, 8, change)
        spacer(f.holder, "episodes")

        f = Field(self, 1, "Difficulty", "'random' varies the world every episode")
        self.preset = combo(f.holder, PRESETS, "random", 16, change)

        f = Field(
            self, 2, "Watch it learn",
            "0 trains headless; it cannot change the run",
        )
        spacer(f.holder, "draw every")
        self.render_every = entry(f.holder, 0, 6, change)
        spacer(f.holder, "episodes at")
        self.render_speed = combo(f.holder, ("1", "2", "5", "10", "20"), "10", 5, change)
        spacer(f.holder, "x")

        f = Field(
            self, 3, "Snapshots",
            "replay the whole run later from the Drive tab",
        )
        spacer(f.holder, "save every")
        self.snapshot_every = entry(f.holder, 0, 6, change)
        spacer(f.holder, "episodes")

        adv = Disclosure(self, 4)
        body = adv.body
        self.advanced = adv

        f = Field(body, 0, "Episode length")
        self.max_steps = entry(f.holder, 900, 8, change)
        spacer(f.holder, "steps each")

        f = Field(body, 1, "Seed")
        self.seed = entry(f.holder, 7, 8, change)

        f = Field(body, 2, "World", "'lane' drops the traffic entirely")
        self.scenario = combo(f.holder, SCENARIOS, "traffic", 12, change)

        f = Field(body, 3, "Overrides", "leave blank to use the difficulty preset")
        spacer(f.holder, "cars")
        self.traffic = entry(f.holder, "", 5, change)
        spacer(f.holder, "slow")
        self.slow_vehicles = entry(f.holder, "", 5, change)
        spacer(f.holder, "reactive")
        self.reactive = entry(f.holder, "", 6, change)

        f = Field(body, 4, "Options")
        self.curriculum = check(f.holder, "curriculum", True, change)
        self.handover = check(f.holder, "handover", False, change)
        self.tracking = check(f.holder, "MLflow", True, change)

        f = Field(body, 5, "Run name", "labels the run in MLflow")
        self.run_name = entry(f.holder, "", 26, change)

        f = Field(body, 6, "Evaluate")
        spacer(f.holder, "every")
        self.eval_every = entry(f.holder, 25, 6, change)
        spacer(f.holder, "episodes, over")
        self.eval_episodes = entry(f.holder, 3, 6, change)
        spacer(f.holder, "worlds; log every")
        self.log_every = entry(f.holder, 5, 6, change)

        f = Field(body, 7, "Save model to")
        self.output = entry(f.holder, "models/autodrive_dqn.npz", 34, change)

        f = Field(body, 8, "Save metrics to")
        self.metrics = entry(f.holder, "runs/training_metrics.csv", 34, change)

    def values(self) -> dict[str, Any]:
        return {
            "episodes": as_int(self.episodes, 300),
            "max_steps": as_int(self.max_steps, 900),
            "seed": as_int(self.seed, 7),
            "scenario": self.scenario.get(),
            "preset": self.preset.get(),
            "traffic": as_int(self.traffic),
            "slow_vehicles": as_int(self.slow_vehicles),
            "reactive": as_float(self.reactive),
            "curriculum": self.curriculum.get(),
            "handover": self.handover.get(),
            "tracking": self.tracking.get(),
            "run_name": self.run_name.get().strip(),
            "eval_every": as_int(self.eval_every, 25),
            "eval_episodes": as_int(self.eval_episodes, 3),
            "log_every": as_int(self.log_every, 5),
            "render_every": as_int(self.render_every, 0),
            "render_speed": as_float(self.render_speed, 10.0),
            "snapshot_every": as_int(self.snapshot_every, 0),
            "output": self.output.get().strip(),
            "metrics": self.metrics.get().strip(),
        }


class ClonePanel(Panel):
    module = "clone"
    run_label = "▶  Clone my driving"

    def build(self) -> None:
        change = self.changed
        ttk.Label(
            self,
            text=(
                "The supervised counterpart to reinforcement learning: the same network, "
                "trained on what you did rather than on reward.\nRecord a demonstration "
                "first — Drive tab, policy 'manual', with recording switched on."
            ),
            style="Intro.TLabel",
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 16))

        f = Field(self, 1, "Recordings", "ctrl-click to combine several")
        self.demo_list = tk.Listbox(f.holder, selectmode="extended", height=5, width=42,
                                    exportselection=False)
        self.demo_list.pack(side="left")
        style_listbox(self.demo_list, self.fonts)
        self.demo_list.bind("<<ListboxSelect>>", lambda _e: change())
        ttk.Button(f.holder, text="Add…", command=self._add_demo).pack(side="left", padx=10)

        f = Field(self, 2, "Save model to")
        self.output = entry(f.holder, "models/clone.npz", 30, change)

        adv = Disclosure(self, 3)
        body = adv.body
        self.advanced = adv

        f = Field(body, 0, "Training")
        spacer(f.holder, "epochs")
        self.epochs = entry(f.holder, 60, 6, change)
        spacer(f.holder, "batch")
        self.batch_size = entry(f.holder, 128, 6, change)
        spacer(f.holder, "learning rate")
        self.learning_rate = entry(f.holder, 0.001, 8, change)

        f = Field(body, 1, "Validation")
        self.val_fraction = entry(f.holder, 0.1, 6, change)
        spacer(f.holder, "held back, seed")
        self.seed = entry(f.holder, 0, 6, change)
        spacer(f.holder, "")
        self.eval = check(f.holder, "score it afterwards", True, change)

    def _add_demo(self) -> None:
        chosen = filedialog.askopenfilenames(
            title="Select recordings", filetypes=[("NumPy archives", "*.npz")]
        )
        for path in chosen:
            try:
                shown = str(Path(path).relative_to(Path.cwd())).replace("\\", "/")
            except ValueError:
                shown = path
            if shown not in self.demo_list.get(0, "end"):
                self.demo_list.insert("end", shown)
        self.changed()

    def refresh_assets(self, assets: Assets) -> None:
        selected = {self.demo_list.get(i) for i in self.demo_list.curselection()}
        self.demo_list.delete(0, "end")
        for name in assets.demo_names:
            self.demo_list.insert("end", name)
        for index in range(self.demo_list.size()):
            if self.demo_list.get(index) in selected:
                self.demo_list.selection_set(index)
        if not self.demo_list.curselection() and self.demo_list.size():
            self.demo_list.selection_set(0)

    def values(self) -> dict[str, Any]:
        return {
            "demos": [self.demo_list.get(i) for i in self.demo_list.curselection()],
            "output": self.output.get().strip(),
            "epochs": as_int(self.epochs, 60),
            "batch_size": as_int(self.batch_size, 128),
            "learning_rate": as_float(self.learning_rate, 0.001),
            "val_fraction": as_float(self.val_fraction, 0.1),
            "seed": as_int(self.seed, 0),
            "eval": self.eval.get(),
        }


class BenchmarkPanel(Panel):
    module = "benchmark"
    run_label = "▶  Run benchmark"

    def build(self) -> None:
        change = self.changed
        ttk.Label(
            self,
            text=(
                "Every selected policy drives the same worlds, so differences between "
                "rows are differences in driving rather than luck.\nSelect several to "
                "compare imitation against reinforcement directly."
            ),
            style="Intro.TLabel",
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 16))

        f = Field(self, 1, "Policies", "ctrl-click to compare several")
        self.policy_list = tk.Listbox(f.holder, selectmode="extended", height=7, width=42,
                                      exportselection=False)
        self.policy_list.pack(side="left")
        style_listbox(self.policy_list, self.fonts)
        self.policy_list.bind("<<ListboxSelect>>", lambda _e: change())

        f = Field(self, 2, "Difficulty", "unforgiving adds traffic from behind")
        self.cell_vars: dict[str, tk.BooleanVar] = {}
        for cell in BENCHMARK_CELL_CHOICES:
            self.cell_vars[cell] = check(
                f.holder, cell, cell in BENCHMARK_CELLS, change
            )

        f = Field(self, 3, "Sample", "100 gives a publishable sample")
        self.episodes = entry(f.holder, 100, 8, change)
        spacer(f.holder, "episodes per cell")

        adv = Disclosure(self, 4)
        body = adv.body
        self.advanced = adv

        f = Field(body, 0, "Episode length")
        self.max_steps = entry(f.holder, 900, 8, change)
        spacer(f.holder, "steps")

        f = Field(body, 1, "Held-out seeds", "far from anything used in training")
        self.seed_start = entry(f.holder, 350000, 12, change)
        spacer(f.holder, "onwards")

        f = Field(body, 2, "Output")
        self.markdown = check(f.holder, "markdown table", True, change)
        spacer(f.holder, "write to")
        self.out = entry(f.holder, "BENCHMARK_current.md", 24, change)

    def refresh_assets(self, assets: Assets) -> None:
        selected = {self.policy_list.get(i) for i in self.policy_list.curselection()}
        self.policy_list.delete(0, "end")
        for name in (*BASELINE_POLICIES, *assets.model_names):
            self.policy_list.insert("end", name)
        for index in range(self.policy_list.size()):
            if self.policy_list.get(index) in selected:
                self.policy_list.selection_set(index)
        if not self.policy_list.curselection() and self.policy_list.size():
            self.policy_list.selection_set(0)

    def values(self) -> dict[str, Any]:
        return {
            "policies": [self.policy_list.get(i) for i in self.policy_list.curselection()],
            "cells": [c for c, v in self.cell_vars.items() if v.get()],
            "episodes": as_int(self.episodes, 100),
            "max_steps": as_int(self.max_steps, 900),
            "seed_start": as_int(self.seed_start, 350000),
            "markdown": self.markdown.get(),
            "out": self.out.get().strip(),
        }


# ── Application ──────────────────────────────────────────────────────────────


class LauncherApp:
    def __init__(self, root: tk.Tk, project_root: Path | None = None) -> None:
        self.root = root
        self.project_root = Path(project_root or Path.cwd())
        self.job: Job | None = None
        # Panels fire change callbacks while being constructed, before the
        # preview widget exists. Nothing may touch widgets until assembly ends.
        self._ready = False

        enable_dpi_awareness()
        self.fonts = apply_theme(root)

        root.title("AutoDrive RL Lab")
        root.geometry("1120x900")
        root.minsize(960, 760)
        root.configure(background=BG)

        self._build_header()
        self._build_tabs()
        self._build_command_bar()
        self._build_controls()
        self._build_log()

        self._ready = True
        self.refresh_assets()
        self.update_preview()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(POLL_MS, self._poll)

    # ── layout ───────────────────────────────────────────────────────────────

    def _build_header(self) -> None:
        bar = tk.Frame(self.root, background=PANEL_BG)
        bar.pack(fill="x")
        inner = ttk.Frame(bar, style="Bar.TFrame", padding=(24, 18, 24, 16))
        inner.pack(fill="x")

        left = ttk.Frame(inner, style="Bar.TFrame")
        left.pack(side="left")
        ttk.Label(left, text="AutoDrive RL Lab", style="Title.TLabel").pack(anchor="w")
        self.subtitle = ttk.Label(left, text="", style="Subtitle.TLabel")
        self.subtitle.pack(anchor="w", pady=(3, 0))

        ttk.Button(inner, text="↻  Refresh files", command=self.refresh_assets).pack(side="right")

        tk.Frame(self.root, background=CARD_EDGE, height=1).pack(fill="x")

    def _build_tabs(self) -> None:
        holder = ttk.Frame(self.root, padding=(24, 14, 24, 0))
        holder.pack(fill="x")
        self.notebook = ttk.Notebook(holder)
        self.notebook.pack(fill="both", expand=True)
        self.panels: dict[str, Panel] = {}
        for title, panel_class in (
            ("Overview", OverviewPanel),
            ("Drive", DrivePanel),
            ("Train", TrainPanel),
            ("Clone", ClonePanel),
            ("Benchmark", BenchmarkPanel),
        ):
            panel = panel_class(self.notebook, self)
            self.notebook.add(panel, text=title)
            self.panels[title] = panel
        self.notebook.bind("<<NotebookTabChanged>>", lambda _e: self.update_preview())

    def _build_command_bar(self) -> None:
        frame = ttk.Frame(self.root, padding=(24, 16, 24, 0))
        frame.pack(fill="x")
        ttk.Label(frame, text="COMMAND", style="Caption.TLabel").pack(anchor="w", pady=(0, 5))
        self.preview = tk.Text(frame, height=2, wrap="word")
        self.preview.pack(fill="x")
        style_text(self.preview, self.fonts)
        self.preview.tag_configure("cmd", foreground=ACCENT)
        self.preview.configure(state="disabled")

    def _build_controls(self) -> None:
        frame = ttk.Frame(self.root, padding=(24, 14, 24, 10))
        frame.pack(fill="x")
        self.run_button = ttk.Button(frame, text="Run", style="Accent.TButton", command=self.run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(frame, text="■  Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=(10, 0))
        ttk.Button(frame, text="Clear log", command=self.clear_log).pack(side="left", padx=(10, 0))

        status_box = ttk.Frame(frame)
        status_box.pack(side="right")
        self.status_dot = tk.Canvas(status_box, width=10, height=10, highlightthickness=0,
                                    background=BG)
        self.status_dot.pack(side="left", padx=(0, 8))
        self._dot = self.status_dot.create_oval(1, 1, 9, 9, fill=TEXT_DIM, outline="")
        self.status = ttk.Label(status_box, text="idle", style="Status.TLabel")
        self.status.pack(side="left")

    def _build_log(self) -> None:
        frame = ttk.Frame(self.root, padding=(24, 0, 24, 22))
        frame.pack(fill="both", expand=True)
        self.log = tk.Text(frame, wrap="none", height=14)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.log.yview,
                               style="Vertical.TScrollbar")
        self.log.configure(yscrollcommand=scroll.set)
        style_text(self.log, self.fonts, log=True)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # Colour by meaning so a long run is skimmable rather than a wall of grey.
        self.log.tag_configure("echo", foreground=ACCENT)
        self.log.tag_configure("ok", foreground=OK)
        self.log.tag_configure("warn", foreground=WARN)
        self.log.tag_configure("error", foreground=ERROR)
        self.log.tag_configure("dim", foreground=TEXT_DIM)
        self.log.tag_configure("plain", foreground=TEXT_MAIN)
        self.log.configure(state="disabled")

    # ── state ────────────────────────────────────────────────────────────────

    @property
    def current_panel(self) -> Panel:
        return self.panels[self.notebook.tab(self.notebook.select(), "text").strip()]

    def refresh_assets(self) -> None:
        assets = discover_assets(self.project_root)
        for panel in self.panels.values():
            panel.refresh_assets(assets)
        self.update_preview()

    def current_command(self) -> list[str]:
        panel = self.current_panel
        if not panel.module:
            raise ValueError("this tab has nothing to run")
        return module_command(panel.module, build_args(panel.module, panel.values()))

    def update_preview(self) -> None:
        if not self._ready:
            return
        panel_now = self.current_panel
        if not panel_now.module:
            self.preview.configure(state="normal")
            self.preview.delete("1.0", "end")
            self.preview.insert(
                "1.0", "Nothing to run from this tab - pick Drive, Train, "
                "Clone or Benchmark.", "cmd",
            )
            self.preview.configure(state="disabled")
            self.subtitle.configure(text=MODE_BLURB.get("Overview", ""))
            if not (self.job and self.job.is_running):
                self.run_button.configure(text=panel_now.run_label)
            return
        try:
            text = format_command(self.current_command())
        except Exception as error:  # a half-typed field should never crash the window
            text = f"(cannot build command: {error})"
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", text, "cmd")
        self.preview.configure(state="disabled")

        panel = self.current_panel
        self.subtitle.configure(text=MODE_BLURB.get(
            self.notebook.tab(self.notebook.select(), "text").strip(), ""
        ))
        if not (self.job and self.job.is_running):
            self.run_button.configure(text=panel.run_label)

    # ── log ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _classify(line: str) -> str:
        stripped = line.strip()
        if stripped.startswith("$"):
            return "echo"
        lowered = stripped.lower()
        if stripped.startswith(("✓",)) or "succeeded" in lowered:
            return "ok"
        if stripped.startswith(("✗",)) or lowered.startswith(("error", "traceback")) \
                or "error:" in lowered:
            return "error"
        if stripped.startswith("■") or lowered.startswith("warning"):
            return "warn"
        if stripped.startswith(("…", "evaluating", "  evaluating")):
            return "dim"
        return "plain"

    def append(self, *lines: str) -> None:
        self.log.configure(state="normal")
        for line in lines:
            self.log.insert("end", line + "\n", self._classify(line))
        excess = int(self.log.index("end-1c").split(".")[0]) - LOG_LIMIT
        if excess > 0:
            self.log.delete("1.0", f"{excess}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # ── running ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        if self.job and self.job.is_running:
            return
        panel = self.current_panel
        if not panel.module:
            self.notebook.select(1)
            return
        problems = validate(panel.module, panel.values())
        if problems:
            messagebox.showwarning("Check the settings", "\n\n".join(problems))
            return

        argv = self.current_command()
        self.append("", f"$ {format_command(argv)}")
        self.job = Job(argv=argv, cwd=self.project_root)
        try:
            self.job.start()
        except OSError as error:
            self.append(f"error: could not start — {error}")
            self.job = None
            return
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._set_status("running", ACCENT)

    def stop(self) -> None:
        if self.job and self.job.is_running:
            self.append("… stopping")
            self.job.stop()

    def _poll(self) -> None:
        """Drain output on the Tk thread. The only place widgets get touched."""
        if self.job is not None:
            lines = self.job.drain()
            if lines:
                self.append(*lines)
            if not self.job.is_running:
                self._finish()
        self.root.after(POLL_MS, self._poll)

    def _finish(self) -> None:
        assert self.job is not None
        state, code = self.job.state, self.job.returncode
        if state is JobState.SUCCEEDED:
            self._set_status("finished", OK)
            self.append("✓ done")
        elif state is JobState.CANCELLED:
            self._set_status("stopped", WARN)
            self.append("■ stopped")
        else:
            self._set_status(f"failed · exit {code}", ERROR)
            self.append(f"✗ exited with code {code}")
        self.job = None
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.refresh_assets()  # a finished run may have written a new model

    def _set_status(self, text: str, colour: str) -> None:
        self.status.configure(text=text, foreground=colour)
        self.status_dot.itemconfigure(self._dot, fill=colour)

    def _on_close(self) -> None:
        if self.job and self.job.is_running:
            if not messagebox.askokcancel("Quit", "A job is still running. Stop it and quit?"):
                return
            self.job.stop()
        self.root.destroy()


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the AutoDrive RL Lab desktop UI.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="folder holding models/ and demos/ (default: current directory)",
    )
    args = parser.parse_args(argv)

    try:
        root = tk.Tk()
    except tk.TclError as error:  # headless machine, or no display forwarded
        print(f"Could not open a window: {error}", file=sys.stderr)
        print("Use the command line directly, e.g. python -m autodrive_rl.play", file=sys.stderr)
        raise SystemExit(1)

    LauncherApp(root, project_root=args.project_root)
    root.mainloop()


if __name__ == "__main__":
    main()
