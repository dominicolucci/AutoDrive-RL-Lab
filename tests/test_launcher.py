"""Tests for the launcher's testable core: command building and job running.

The window itself is thin — widgets wired to two modules. Those two modules are
where a bug would be silent and damaging, so that is where the tests are:

* `commands` — a mistranslated flag would run the wrong experiment while the
  preview line looked plausible.
* `jobrunner` — a job that cannot be cancelled, or whose output never arrives,
  makes the GUI worse than the terminal it replaces.

Both run headless. A display is only needed by the smoke test at the bottom,
which is skipped when there isn't one.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from autodrive_rl.commands import (
    BENCHMARK_CELL_CHOICES,
    BENCHMARK_CELLS,
    Assets,
    build_args,
    build_benchmark_args,
    build_clone_args,
    build_drive_args,
    build_train_args,
    discover_assets,
    validate,
)
from autodrive_rl.jobrunner import (
    Job,
    JobState,
    format_command,
    module_command,
)


# ── Command building: defaults are omitted ───────────────────────────────────


def test_drive_defaults_produce_no_flags():
    """An untouched form should run the plain command, not a wall of defaults."""
    assert build_drive_args(
        {"policy": "heuristic", "scenario": "traffic", "preset": "normal",
         "seed": 7, "fps": 30}
    ) == []


def test_train_defaults_produce_no_flags():
    assert build_train_args(
        {"episodes": 300, "max_steps": 900, "seed": 7, "scenario": "traffic",
         "preset": "random", "curriculum": True, "handover": False,
         "tracking": True, "eval_every": 25, "eval_episodes": 3, "log_every": 5,
         "output": "models/autodrive_dqn.npz", "metrics": "runs/training_metrics.csv"}
    ) == []


def test_clone_defaults_reduce_to_the_demos_flag():
    assert build_clone_args(
        {"demos": ["demos/me.npz"], "output": "models/clone.npz", "epochs": 60,
         "batch_size": 128, "learning_rate": 0.001, "val_fraction": 0.1,
         "seed": 0, "eval": True}
    ) == ["--demos", "demos/me.npz"]


def test_benchmark_defaults_reduce_to_policies():
    args = build_benchmark_args(
        {"policies": ["heuristic"], "cells": list(BENCHMARK_CELLS),
         "episodes": 100, "seed_start": 350000, "max_steps": 900}
    )
    assert args == ["--policy", "heuristic"]


# ── Command building: changed values are carried through ─────────────────────


def test_drive_carries_policy_model_and_world():
    args = build_drive_args(
        {"policy": "dqn", "model": "models/clone.npz", "scenario": "traffic",
         "preset": "dense", "traffic": 12, "obstacles": 2, "reactive": 0.5,
         "seed": 99, "fps": 60}
    )
    assert args == [
        "--policy", "dqn", "--model", "models/clone.npz",
        "--scenario-preset", "dense",
        "--traffic", "12", "--obstacles", "2", "--reactive", "0.5",
        "--seed", "99", "--fps", "60",
    ]


def test_drive_omits_model_unless_the_learned_policy_is_selected():
    """A model path next to 'manual' would misrepresent what is about to run."""
    args = build_drive_args({"policy": "manual", "model": "models/clone.npz"})
    assert "--model" not in args


def test_drive_recording_requires_the_checkbox():
    base = {"policy": "manual", "record_path": "demos/me.npz"}
    assert "--record" not in build_drive_args({**base, "record_enabled": False})
    assert build_drive_args({**base, "record_enabled": True})[-2:] == [
        "--record", "demos/me.npz"
    ]


def test_train_negative_toggles_use_the_no_prefix():
    args = build_train_args({"curriculum": False, "tracking": False, "handover": True})
    assert "--no-curriculum" in args
    assert "--no-tracking" in args
    assert "--handover" in args


def test_train_carries_paths_and_eval_settings():
    args = build_train_args(
        {"episodes": 1000, "seed": 1, "eval_every": 50, "eval_episodes": 10,
         "output": "models/robust_s1.npz", "metrics": "runs/robust_s1.csv"}
    )
    assert args[:2] == ["--episodes", "1000"]
    assert "--output" in args and "models/robust_s1.npz" in args
    assert "--metrics" in args and "runs/robust_s1.csv" in args


def test_benchmark_repeats_the_policy_flag_for_a_comparison():
    args = build_benchmark_args(
        {"policies": ["models/autodrive_dqn_best.npz", "models/clone.npz",
                      "heuristic", "random"],
         "cells": list(BENCHMARK_CELLS), "episodes": 100}
    )
    assert args.count("--policy") == 4
    assert args.index("--policy") == 0


def test_benchmark_only_names_cells_when_a_subset_is_chosen():
    full = build_benchmark_args({"policies": ["random"], "cells": list(BENCHMARK_CELLS)})
    subset = build_benchmark_args({"policies": ["random"], "cells": ["dense"]})
    assert "--cells" not in full
    assert subset[-2:] == ["--cells", "dense"]


def test_unforgiving_is_offered_but_off_by_default():
    """It has to be reachable without editing a command line, and it has to be
    off unless asked for — `BENCHMARK_current.md` is the other three cells, and
    a silently widened default would make the next run incomparable."""
    assert "unforgiving" in BENCHMARK_CELL_CHOICES
    assert "unforgiving" not in BENCHMARK_CELLS
    args = build_benchmark_args(
        {"policies": ["random"], "cells": list(BENCHMARK_CELL_CHOICES)}
    )
    assert args[-5:] == ["--cells", "sparse", "normal", "dense", "unforgiving"]


def test_benchmark_markdown_and_output():
    args = build_benchmark_args(
        {"policies": ["random"], "cells": list(BENCHMARK_CELLS),
         "markdown": True, "out": "BENCHMARK_current.md"}
    )
    assert "--markdown" in args
    assert args[-2:] == ["--out", "BENCHMARK_current.md"]


def test_build_args_dispatches_by_module():
    assert build_args("play", {"policy": "random"}) == ["--policy", "random"]
    with pytest.raises(ValueError):
        build_args("nope", {})


# ── Validation ───────────────────────────────────────────────────────────────


def test_learned_policy_needs_a_model():
    assert validate("play", {"policy": "dqn", "model": ""})
    assert not validate("play", {"policy": "dqn", "model": "models/x.npz"})


def test_recording_only_makes_sense_while_driving_manually():
    problems = validate(
        "play", {"policy": "dqn", "model": "m.npz", "record_enabled": True,
                 "record_path": "demos/me.npz"}
    )
    assert any("manual" in p for p in problems)


def test_cloning_without_a_recording_is_blocked_with_a_useful_hint():
    problems = validate("clone", {"demos": [], "epochs": 60})
    assert problems and "record" in problems[0].lower()


def test_benchmark_needs_policies_and_cells():
    assert validate("benchmark", {"policies": [], "cells": ["dense"], "episodes": 10})
    assert validate("benchmark", {"policies": ["random"], "cells": [], "episodes": 10})
    assert not validate("benchmark", {"policies": ["random"], "cells": ["dense"], "episodes": 10})


def test_nonpositive_counts_are_rejected():
    assert validate("train", {"episodes": 0, "eval_episodes": 3})
    assert validate("benchmark", {"policies": ["random"], "cells": ["dense"], "episodes": 0})


# ── Asset discovery ──────────────────────────────────────────────────────────


def test_discover_assets_finds_checkpoints_and_demos(tmp_path: Path):
    (tmp_path / "models").mkdir()
    (tmp_path / "demos").mkdir()
    (tmp_path / "models" / "b.npz").write_bytes(b"")
    (tmp_path / "models" / "a.npz").write_bytes(b"")
    (tmp_path / "models" / "notes.txt").write_text("ignored")
    (tmp_path / "demos" / "me.npz").write_bytes(b"")

    assets = discover_assets(tmp_path)
    assert assets.model_names == ("models/a.npz", "models/b.npz")  # sorted
    assert assets.demo_names == ("demos/me.npz",)


def test_discover_assets_tolerates_missing_folders(tmp_path: Path):
    assert discover_assets(tmp_path) == Assets(models=(), demos=())


def test_discovered_paths_use_forward_slashes_for_display():
    """Command previews should look the same on every platform."""
    assets = Assets(models=(Path("models") / "a.npz",))
    assert assets.model_names == ("models/a.npz",)


# ── Job runner ───────────────────────────────────────────────────────────────


def test_module_command_uses_the_running_interpreter():
    argv = module_command("play", ["--policy", "random"])
    assert argv[0] == sys.executable
    assert argv[1:4] == ["-u", "-m", "autodrive_rl.play"]
    assert argv[4:] == ["--policy", "random"]


def test_format_command_is_readable():
    assert "autodrive_rl.play" in format_command(module_command("play", []))


def test_job_streams_output_and_reports_success():
    job = Job(argv=[sys.executable, "-u", "-c", "print('alpha'); print('beta')"])
    job.start()
    lines = job.read_all(timeout=30)
    assert lines == ["alpha", "beta"]
    assert job.state is JobState.SUCCEEDED
    assert job.returncode == 0


def test_job_merges_stderr_into_the_stream():
    """A terminal interleaves them; the log pane should too."""
    job = Job(argv=[sys.executable, "-u", "-c",
                    "import sys; print('out'); print('err', file=sys.stderr)"])
    job.start()
    assert set(job.read_all(timeout=30)) == {"out", "err"}


def test_job_reports_failure_with_the_exit_code():
    job = Job(argv=[sys.executable, "-c", "raise SystemExit(3)"])
    job.start()
    job.read_all(timeout=30)
    assert job.state is JobState.FAILED
    assert job.returncode == 3


def test_job_can_be_cancelled_and_says_so():
    """Without this, a mistyped 1000-episode run can only be killed by force."""
    job = Job(argv=[sys.executable, "-u", "-c",
                    "import time\nwhile True:\n    print('tick', flush=True)\n    time.sleep(0.05)"])
    job.start()
    deadline = time.time() + 10
    while not job.drain() and time.time() < deadline:
        time.sleep(0.05)
    job.stop(timeout=10)
    job.read_all(timeout=15)
    assert job.state is JobState.CANCELLED
    assert not job.is_running


def test_drain_never_blocks_on_a_silent_job():
    """The UI polls every ~120ms; a blocking drain would freeze the window."""
    job = Job(argv=[sys.executable, "-u", "-c", "import time; time.sleep(2)"])
    job.start()
    started = time.time()
    assert job.drain() == []
    assert time.time() - started < 0.5
    job.stop()
    job.wait(timeout=10)


def test_drain_is_bounded_so_one_tick_cannot_hog_the_ui():
    job = Job(argv=[sys.executable, "-u", "-c", "\n".join(f"print({i})" for i in range(50))])
    job.start()
    job.wait(timeout=30)
    assert len(job.drain(max_lines=10)) <= 10


def test_starting_a_running_job_twice_is_refused():
    job = Job(argv=[sys.executable, "-u", "-c", "import time; time.sleep(2)"])
    job.start()
    with pytest.raises(RuntimeError):
        job.start()
    job.stop()
    job.wait(timeout=10)


def test_job_runs_in_the_requested_directory(tmp_path: Path):
    job = Job(argv=[sys.executable, "-u", "-c", "import os; print(os.getcwd())"], cwd=tmp_path)
    job.start()
    lines = job.read_all(timeout=30)
    assert Path(lines[0]).resolve() == tmp_path.resolve()


# ── GUI smoke test (needs a display) ─────────────────────────────────────────


def _display_available() -> bool:
    try:
        import tkinter

        root = tkinter.Tk()
    except Exception:
        return False
    root.destroy()
    return True


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_launcher_window_builds_and_previews_every_tab():
    """Construct the real window and walk its tabs, without showing it."""
    import tkinter

    from autodrive_rl.launcher import LauncherApp

    root = tkinter.Tk()
    root.withdraw()
    try:
        app = LauncherApp(root, project_root=Path.cwd())
        assert set(app.panels) == {"Drive", "Train", "Clone", "Benchmark"}
        for index in range(len(app.panels)):
            app.notebook.select(index)
            root.update_idletasks()
            app.update_preview()
            argv = app.current_command()
            assert argv[0] == sys.executable
            assert "autodrive_rl." in argv[3]
    finally:
        root.destroy()
