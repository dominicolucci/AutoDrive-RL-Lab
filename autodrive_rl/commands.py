"""Turn launcher form values into command-line arguments.

Separated from the GUI on purpose: these are plain functions over plain
dictionaries, so the translation from "what the user picked" to "what runs" is
unit-testable with no display, no Tk, and no subprocess. Every bug class that
actually matters here — a flag misspelled, an override silently dropped, a
default emitted when it should have been omitted — is caught by a test rather
than by squinting at a window.

Two rules the builders follow:

* **Omit anything left at its default.** The preview line stays short and
  readable, and the underlying module keeps ownership of its own defaults.
* **Never invent values.** A blank optional field produces no flag at all, not
  a guessed one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# ── Choices, mirrored from the module parsers ────────────────────────────────

DRIVE_POLICIES = ("heuristic", "manual", "random", "dqn")
SCENARIOS = ("traffic", "lane")
PRESETS = ("sparse", "normal", "dense", "unforgiving", "random")

#: Cells the benchmark *offers*. The first three are what `BENCHMARK_current.md`
#: was measured on and stay the default selection, so an untouched form still
#: reproduces the published table; "unforgiving" is opt-in.
BENCHMARK_CELL_CHOICES = ("sparse", "normal", "dense", "unforgiving")
BENCHMARK_CELLS = ("sparse", "normal", "dense")
BASELINE_POLICIES = ("heuristic", "random")


@dataclass(frozen=True)
class Assets:
    """Checkpoints and demonstration files discovered on disk."""

    models: tuple[Path, ...] = ()
    demos: tuple[Path, ...] = ()

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(str(p).replace("\\", "/") for p in self.models)

    @property
    def demo_names(self) -> tuple[str, ...]:
        return tuple(str(p).replace("\\", "/") for p in self.demos)


def discover_assets(root: Path | str = ".") -> Assets:
    """Find `models/*.npz` and `demos/*.npz`, relative to the project root.

    Paths stay relative so the command preview reads like something a user
    would type, and so a copied command works from the project directory.
    """
    root = Path(root)
    def _find(folder: str) -> tuple[Path, ...]:
        directory = root / folder
        if not directory.is_dir():
            return ()
        return tuple(
            sorted(
                (p.relative_to(root) for p in directory.glob("*.npz")),
                key=lambda p: str(p).lower(),
            )
        )

    return Assets(models=_find("models"), demos=_find("demos"))


# ── Helpers ──────────────────────────────────────────────────────────────────


def _put(
    args: list[str],
    flag: str,
    value: Any,
    default: Any = None,
    *,
    cast: Callable[[Any], str] = str,
) -> None:
    """Append `--flag value` unless the value is empty or already the default."""
    if value is None or value == "" or value == default:
        return
    args.extend([flag, cast(value)])


def _toggle(args: list[str], flag: str, value: bool, default: bool) -> None:
    """Append `--flag` / `--no-flag` only when it differs from the default."""
    if bool(value) == default:
        return
    args.append(flag if value else f"--no-{flag.lstrip('-')}")


def _world_overrides(args: list[str], values: Mapping[str, Any]) -> None:
    """Car count / obstacle count / reactive fraction, shared by drive + train."""
    _put(args, "--traffic", values.get("traffic"))
    _put(args, "--obstacles", values.get("obstacles"))
    _put(args, "--slow-vehicles", values.get("slow_vehicles"))
    _put(args, "--reactive", values.get("reactive"))


# ── Builders ─────────────────────────────────────────────────────────────────


def build_drive_args(values: Mapping[str, Any]) -> list[str]:
    """`autodrive_rl.play` — watch or drive the simulation."""
    args: list[str] = []
    policy = values.get("policy", "heuristic")
    _put(args, "--policy", policy, default="heuristic")

    # The model only means something to the learned policy; emitting it for
    # manual or heuristic driving would be misleading in the preview.
    if policy == "dqn":
        _put(args, "--model", values.get("model"))

    _put(args, "--scenario", values.get("scenario"), default="traffic")
    _put(args, "--scenario-preset", values.get("preset"), default="normal")
    _world_overrides(args, values)
    _put(args, "--seed", values.get("seed"), default=7)
    _put(args, "--fps", values.get("fps"), default=30)

    if values.get("record_enabled") and values.get("record_path"):
        _put(args, "--record", values.get("record_path"))
    if values.get("autopilot_enabled") and values.get("autopilot_model"):
        _put(args, "--autopilot-model", values.get("autopilot_model"))
    return args


def build_train_args(values: Mapping[str, Any]) -> list[str]:
    """`autodrive_rl.train` — train a DQN agent."""
    args: list[str] = []
    _put(args, "--episodes", values.get("episodes"), default=300)
    _put(args, "--max-steps", values.get("max_steps"), default=900)
    _put(args, "--seed", values.get("seed"), default=7)
    _put(args, "--scenario", values.get("scenario"), default="traffic")
    _put(args, "--scenario-preset", values.get("preset"), default="random")
    _world_overrides(args, values)
    _toggle(args, "--curriculum", values.get("curriculum", True), default=True)
    _toggle(args, "--handover", values.get("handover", False), default=False)
    _toggle(args, "--tracking", values.get("tracking", True), default=True)
    _put(args, "--run-name", values.get("run_name"))
    _put(args, "--eval-every", values.get("eval_every"), default=25)
    _put(args, "--eval-episodes", values.get("eval_episodes"), default=3)
    _put(args, "--log-every", values.get("log_every"), default=5)
    _put(args, "--output", values.get("output"), default="models/autodrive_dqn.npz")
    _put(args, "--metrics", values.get("metrics"), default="runs/training_metrics.csv")
    return args


def build_clone_args(values: Mapping[str, Any]) -> list[str]:
    """`autodrive_rl.clone` — behaviour cloning from recorded demonstrations."""
    args: list[str] = []
    demos = values.get("demos") or []
    if isinstance(demos, (str, Path)):
        demos = [demos]
    if demos:
        args.append("--demos")
        args.extend(str(d) for d in demos)
    _put(args, "--output", values.get("output"), default="models/clone.npz")
    _put(args, "--epochs", values.get("epochs"), default=60)
    _put(args, "--batch-size", values.get("batch_size"), default=128)
    _put(args, "--learning-rate", values.get("learning_rate"), default=0.001)
    _put(args, "--val-fraction", values.get("val_fraction"), default=0.1)
    _put(args, "--seed", values.get("seed"), default=0)
    _toggle(args, "--eval", values.get("eval", True), default=True)
    return args


def build_benchmark_args(values: Mapping[str, Any]) -> list[str]:
    """`autodrive_rl.benchmark` — score policies on held-out worlds."""
    args: list[str] = []
    for policy in values.get("policies") or []:
        args.extend(["--policy", str(policy)])

    cells: Sequence[str] = values.get("cells") or []
    if cells and tuple(cells) != BENCHMARK_CELLS:
        args.append("--cells")
        args.extend(cells)

    _put(args, "--episodes", values.get("episodes"), default=100)
    _put(args, "--seed-start", values.get("seed_start"), default=350000)
    _put(args, "--max-steps", values.get("max_steps"), default=900)
    if values.get("markdown"):
        args.append("--markdown")
    _put(args, "--out", values.get("out"))
    return args


BUILDERS: dict[str, Callable[[Mapping[str, Any]], list[str]]] = {
    "play": build_drive_args,
    "train": build_train_args,
    "clone": build_clone_args,
    "benchmark": build_benchmark_args,
}


def build_args(module: str, values: Mapping[str, Any]) -> list[str]:
    try:
        return BUILDERS[module](values)
    except KeyError:
        raise ValueError(f"unknown module: {module!r}") from None


# ── Validation ───────────────────────────────────────────────────────────────


def validate(module: str, values: Mapping[str, Any]) -> list[str]:
    """Problems worth blocking a launch for, in plain language.

    Caught here rather than left to the subprocess, because an argparse error
    scrolling past in a log pane is a much worse experience than a sentence
    next to the button that would have caused it.
    """
    problems: list[str] = []

    if module == "play":
        if values.get("policy") == "dqn" and not values.get("model"):
            problems.append("Choose a model file to drive with the learned policy.")
        if values.get("record_enabled") and not values.get("record_path"):
            problems.append("Recording is on but no output file is set.")
        if values.get("record_enabled") and values.get("policy") != "manual":
            problems.append("Recording captures your own driving — set policy to manual.")

    elif module == "train":
        if int(values.get("episodes") or 0) <= 0:
            problems.append("Episodes must be at least 1.")
        if int(values.get("eval_episodes") or 0) <= 0:
            problems.append("Evaluation episodes must be at least 1.")

    elif module == "clone":
        if not values.get("demos"):
            problems.append(
                "Select at least one recording. Make one with Drive → policy "
                "'manual' and 'record my driving' enabled."
            )
        if int(values.get("epochs") or 0) <= 0:
            problems.append("Epochs must be at least 1.")

    elif module == "benchmark":
        if not values.get("policies"):
            problems.append("Select at least one policy to evaluate.")
        if not values.get("cells"):
            problems.append("Select at least one difficulty cell.")
        if int(values.get("episodes") or 0) <= 0:
            problems.append("Episodes per cell must be at least 1.")

    return problems
