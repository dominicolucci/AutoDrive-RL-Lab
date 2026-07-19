# Experiment Tracking (MLflow) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log every training run's params, per-episode metrics, eval-matrix results, git commit, and model artifacts to a local MLflow store behind a project-owned tracker seam.

**Architecture:** New module `autodrive_rl/tracking.py` is the ONLY file that imports mlflow; it exposes `NullTracker` (no-op) and `MlflowTracker` behind a five-method interface plus a `create_tracker` factory. `train.py` gains `tracking`/`tracking_uri`/`run_name` params (API default: tracking OFF; CLI default: ON) and calls the tracker at run start, per episode, and at run end. Spec: `docs/superpowers/specs/2026-07-19-experiment-tracking-design.md`.

**Tech Stack:** Python 3.10+, NumPy, MLflow >= 2.9 (new required dependency), stdlib `unittest`.

## Global Constraints

- `mlflow>=2.9` becomes a required dependency: added to `requirements.txt` AND `pyproject.toml` dependencies.
- `train.py` must never import mlflow — only `autodrive_rl/tracking.py` may.
- Python API default `tracking=False`: all 44 existing tests run unchanged with zero MLflow involvement. CLI default: tracking ON.
- `--no-tracking` runs behave byte-identically to today (same console output, same files, no mlruns/ writes).
- Tracker method failures during a run must never crash training (fail-soft with a single printed warning).
- `mlruns/` is gitignored.
- Run tests ONLY with the discover form: `python -m unittest discover -s tests` (the dotted module form is broken on this machine by an unrelated site-packages `tests` package).
- Work from the repo root: `c:\Users\domin\Downloads\AutoDrive-RL-Lab`, branch `experiment-tracking`.

---

### Task 1: The tracker seam (`tracking.py`), dependency, and factory tests

**Files:**
- Create: `autodrive_rl/tracking.py`
- Create: `tests/test_tracking.py`
- Modify: `requirements.txt`, `pyproject.toml:11`, `.gitignore`

**Interfaces:**
- Consumes: nothing from the codebase (standalone module).
- Produces (Task 2 relies on these exact signatures):
  - `class NullTracker` — methods `log_params(params: dict) -> None`, `set_tags(tags: dict) -> None`, `log_metrics(metrics: dict, step: int) -> None`, `log_artifact(path: str | Path) -> None`, `end(status: str = "FINISHED") -> None`; all no-ops.
  - `class MlflowTracker(run_name: str, *, experiment: str = "autodrive-rl", tracking_uri: str | None = None)` — same five methods, mlflow-backed, fail-soft.
  - `create_tracker(enabled: bool, run_name: str, *, tracking_uri: str | None = None)` — returns `NullTracker` when disabled; raises `SystemExit` with an install hint when enabled but mlflow is missing.
  - `git_commit_tag() -> dict` — `{"git_commit": "<short-hash>"}` or `{}` (best-effort, never raises).

- [ ] **Step 1: Install the dependency and declare it**

Run: `python -m pip install "mlflow>=2.9"`
Expected: installs successfully (takes a minute; many transitive packages).

Edit `requirements.txt` to:

```text
numpy>=1.26,<3
mlflow>=2.9
```

Edit `pyproject.toml` line 11 to:

```toml
dependencies = ["numpy>=1.26,<3", "mlflow>=2.9"]
```

Append to `.gitignore`:

```text
mlruns/
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_tracking.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autodrive_rl.tracking import (
    MlflowTracker,
    NullTracker,
    create_tracker,
    git_commit_tag,
)

try:
    import mlflow  # noqa: F401

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class TrackerFactoryTests(unittest.TestCase):
    def test_disabled_returns_null_tracker(self) -> None:
        self.assertIsInstance(create_tracker(False, "any-name"), NullTracker)

    def test_null_tracker_methods_are_silent_no_ops(self) -> None:
        tracker = NullTracker()
        tracker.log_params({"a": 1})
        tracker.set_tags({"t": "x"})
        tracker.log_metrics({"m": 1.0, "text": "ignored"}, step=1)
        tracker.log_artifact("does/not/exist.npz")
        tracker.end("FAILED")

    @unittest.skipUnless(HAS_MLFLOW, "mlflow not installed")
    def test_enabled_returns_working_mlflow_tracker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store_uri = (Path(directory) / "store").as_uri()
            tracker = create_tracker(
                True, "factory-test", tracking_uri=store_uri
            )
            self.assertIsInstance(tracker, MlflowTracker)
            tracker.log_params({"seed": 7})
            tracker.log_metrics({"return": -1.5, "skipped": ""}, step=1)
            tracker.end()

    def test_git_commit_tag_shape(self) -> None:
        tag = git_commit_tag()
        if tag:
            self.assertEqual(set(tag), {"git_commit"})
            self.assertGreaterEqual(len(tag["git_commit"]), 7)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m unittest discover -s tests -k TrackerFactoryTests -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'autodrive_rl.tracking'`.

- [ ] **Step 4: Implement `autodrive_rl/tracking.py`**

```python
"""Experiment tracking behind a project-owned seam.

This is the only module that imports mlflow. Training code talks to a
tracker object, so swapping vendors or disabling tracking never touches
the training loop.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class NullTracker:
    """No-op tracker used when tracking is disabled (and in most tests)."""

    def log_params(self, params: dict) -> None:
        del params

    def set_tags(self, tags: dict) -> None:
        del tags

    def log_metrics(self, metrics: dict, step: int) -> None:
        del metrics, step

    def log_artifact(self, path: str | Path) -> None:
        del path

    def end(self, status: str = "FINISHED") -> None:
        del status


class MlflowTracker:
    """MLflow-backed tracker. Every mlflow call is fail-soft: a tracking
    failure prints one warning and never crashes training."""

    def __init__(
        self,
        run_name: str,
        *,
        experiment: str = "autodrive-rl",
        tracking_uri: str | None = None,
    ) -> None:
        import mlflow

        self._mlflow = mlflow
        self._warned = False
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)

    def _guard(self, operation) -> None:
        try:
            operation()
        except Exception as error:
            if not self._warned:
                self._warned = True
                print(f"warning: experiment tracking call failed: {error}")

    def log_params(self, params: dict) -> None:
        self._guard(
            lambda: self._mlflow.log_params(
                {key: str(value) for key, value in params.items()}
            )
        )

    def set_tags(self, tags: dict) -> None:
        if tags:
            self._guard(lambda: self._mlflow.set_tags(tags))

    def log_metrics(self, metrics: dict, step: int) -> None:
        numeric = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if numeric:
            self._guard(lambda: self._mlflow.log_metrics(numeric, step=step))

    def log_artifact(self, path: str | Path) -> None:
        path = Path(path)
        if path.exists():
            self._guard(lambda: self._mlflow.log_artifact(str(path)))

    def end(self, status: str = "FINISHED") -> None:
        self._guard(lambda: self._mlflow.end_run(status=status))


def git_commit_tag() -> dict:
    """Best-effort {"git_commit": <short-hash>}; empty dict if unavailable."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return {"git_commit": commit} if commit else {}
    except Exception:
        return {}


def create_tracker(
    enabled: bool,
    run_name: str,
    *,
    tracking_uri: str | None = None,
) -> NullTracker | MlflowTracker:
    """Build the tracker for a training run."""

    if not enabled:
        return NullTracker()
    try:
        import mlflow  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "mlflow is not installed. Run: python -m pip install -r "
            "requirements.txt, or pass --no-tracking."
        ) from error
    return MlflowTracker(run_name, tracking_uri=tracking_uri)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m unittest discover -s tests -k TrackerFactoryTests -v`
Expected: 4 tests PASS (the mlflow-backed one takes ~10 s on first run — mlflow import is slow).

- [ ] **Step 6: Run the full suite, then commit**

Run: `python -m unittest discover -s tests`
Expected: 48 tests pass (44 existing + 4 new).

```bash
git add autodrive_rl/tracking.py tests/test_tracking.py requirements.txt pyproject.toml .gitignore
git commit -m "feat: MLflow tracker seam with no-op twin and factory"
```

---

### Task 2: Wire tracking into `train.py`

**Files:**
- Modify: `autodrive_rl/train.py`
- Test: `tests/test_tracking.py` (append)

**Interfaces:**
- Consumes from Task 1: `create_tracker(enabled, run_name, *, tracking_uri=None)`, `git_commit_tag()`, and the five tracker methods (`log_params`, `set_tags`, `log_metrics(metrics, step)`, `log_artifact(path)`, `end(status)`).
- Produces: `train(..., tracking: bool = False, tracking_uri: str | None = None, run_name: str | None = None)`; CLI flags `--tracking/--no-tracking` (default on) and `--run-name`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tracking.py` (add imports at the top of the file: `from dataclasses import replace`, `from autodrive_rl.config import EnvConfig`, `from autodrive_rl.train import build_parser, train`):

```python
class UntrackedTrainingTests(unittest.TestCase):
    def test_parser_has_tracking_flags(self) -> None:
        args = build_parser().parse_args([])
        self.assertTrue(args.tracking)
        self.assertIsNone(args.run_name)
        args = build_parser().parse_args(["--no-tracking", "--run-name", "x"])
        self.assertFalse(args.tracking)
        self.assertEqual(args.run_name, "x")

    def test_untracked_training_creates_no_mlruns(self) -> None:
        cwd_mlruns = Path("mlruns")
        existed_before = cwd_mlruns.exists()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            train(
                episodes=2,
                env_config=replace(EnvConfig(), max_steps=30),
                seed=5,
                curriculum=False,
                eval_every=0,
                eval_episodes=1,
                log_every=100,
                output_path=base / "m.npz",
                metrics_path=base / "m.csv",
            )
            self.assertFalse((base / "mlruns").exists())
        self.assertEqual(cwd_mlruns.exists(), existed_before)


@unittest.skipUnless(HAS_MLFLOW, "mlflow not installed")
class TrackedTrainingTests(unittest.TestCase):
    def test_tracked_run_records_params_metrics_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            store_uri = (base / "store").as_uri()
            train(
                episodes=2,
                env_config=replace(EnvConfig(), max_steps=30),
                seed=5,
                curriculum=False,
                eval_every=0,
                eval_episodes=1,
                log_every=100,
                output_path=base / "m.npz",
                metrics_path=base / "m.csv",
                tracking=True,
                tracking_uri=store_uri,
            )
            client = mlflow.tracking.MlflowClient(tracking_uri=store_uri)
            experiment = client.get_experiment_by_name("autodrive-rl")
            self.assertIsNotNone(experiment)
            runs = client.search_runs([experiment.experiment_id])
            self.assertEqual(len(runs), 1)
            run = runs[0]
            self.assertEqual(run.info.run_name, "m-seed5")
            self.assertEqual(run.data.params["seed"], "5")
            self.assertEqual(run.data.params["gamma"], "0.99")
            self.assertEqual(run.data.params["scenario_preset"], "random")
            history = client.get_metric_history(run.info.run_id, "return")
            self.assertEqual(len(history), 2)
            self.assertEqual(
                run.data.metrics["outcome_complete"]
                + run.data.metrics["outcome_collision"]
                + run.data.metrics["outcome_off_road"],
                1.0,
            )
            artifacts = [
                artifact.path
                for artifact in client.list_artifacts(run.info.run_id)
            ]
            self.assertIn("m.npz", artifacts)
            self.assertIn("m.csv", artifacts)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest discover -s tests -k TrackedTrainingTests -k UntrackedTrainingTests -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'tracking'` and `TypeError: train() got an unexpected keyword argument 'tracking'`.

- [ ] **Step 3: Implement in `autodrive_rl/train.py`**

Import changes at the top: extend the dataclasses import to `from dataclasses import asdict, replace`, and add:

```python
from .tracking import create_tracker, git_commit_tag
```

In `build_parser()`, after the `--reactive` argument:

```python
    parser.add_argument(
        "--tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="log this run to the local MLflow store (default: on)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (default: <output-stem>-seed<seed>)",
    )
```

Extend the `train()` signature after `reactive: float | None = None,` (keep existing params untouched):

```python
    tracking: bool = False,
    tracking_uri: str | None = None,
    run_name: str | None = None,
```

After `best_eval_return = -np.inf` (i.e. once paths and the agent exist), create the tracker and log the run's identity:

```python
    tracker = create_tracker(
        tracking,
        run_name or f"{output_path.stem}-seed{seed}",
        tracking_uri=tracking_uri,
    )
    tracker.log_params(
        {
            **asdict(agent.config),
            **asdict(env_config),
            "episodes": episodes,
            "seed": seed,
            "scenario": scenario,
            "curriculum": curriculum,
            "scenario_preset": scenario_preset,
            "traffic": traffic,
            "obstacles": obstacles,
            "reactive": reactive,
        }
    )
    tracker.set_tags(git_commit_tag())
```

Wrap the episode loop and the final saves in try/except so a crash ends the run as FAILED. The structure becomes:

```python
    try:
        for episode in range(1, episodes + 1):
            ...existing loop body unchanged, except one addition below...
        agent.save(output_path)
        _write_metrics(metrics_path, records)
        tracker.log_artifact(output_path)
        tracker.log_artifact(best_path)
        tracker.log_artifact(metrics_path)
        tracker.end("FINISHED")
    except BaseException:
        tracker.end("FAILED")
        raise
```

(The three `print(f"\nSaved ...")` lines stay after `_write_metrics`, inside the try.)

The one addition inside the loop — immediately after `records.append(record)`:

```python
            tracker.log_metrics(
                {
                    **record,
                    "outcome_collision": float(outcome == "collision"),
                    "outcome_off_road": float(outcome == "off_road"),
                    "outcome_complete": float(outcome == "complete"),
                },
                step=episode,
            )
```

(The tracker's numeric filter drops the record's string fields — `scenario`, `outcome`, empty eval placeholders — so no massaging is needed here.)

In `main()`, pass the new args through to `train(...)`:

```python
        tracking=args.tracking,
        tracking_uri=None,
        run_name=args.run_name,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest discover -s tests -k TrackedTrainingTests -k UntrackedTrainingTests -v`
Expected: 3 tests PASS (the tracked one takes ~15 s).

- [ ] **Step 5: Run the full suite, then commit**

Run: `python -m unittest discover -s tests`
Expected: 51 tests pass.

```bash
git add autodrive_rl/train.py tests/test_tracking.py
git commit -m "feat: training runs log params, metrics, and artifacts to MLflow"
```

---

### Task 3: Docs and a real tracked run

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: user-facing documentation; a manually verified tracked run.

- [ ] **Step 1: Update README**

In the "Quick start" section, change the line "Install the one Python dependency:" to "Install the Python dependencies (NumPy for the simulation, MLflow for experiment tracking):".

Insert a new section after "## Train the agent". (Careful with the nested code fences below — reproduce the intent: a "## Track your experiments" heading, one prose paragraph, a bash block with `mlflow ui`, a sentence, and a bash block with the `--no-tracking` example; keep the README's markdown valid.)

```markdown
## Track your experiments

Every training run is logged to a local MLflow store in `mlruns/` (no
account, no network): the full configuration, the git commit, per-episode
curves (return, distance, crash rate, epsilon, loss), the evaluation
matrix, and the saved model files. Open the dashboard with:

```bash
mlflow ui
```

then visit http://localhost:5000 to browse runs, plot metrics, and compare
training runs side by side. Opt out of tracking for a single run with:

```bash
python -m autodrive_rl.train --episodes 300 --no-tracking
```
```

- [ ] **Step 2: End-to-end verification run**

Run: `python -m autodrive_rl.train --episodes 8 --output models/tracking_smoke.npz --metrics runs/tracking_smoke.csv --run-name tracking-smoke`
Expected: completes normally; an `mlruns/` directory appears in the repo root; git status shows it ignored.

Verify from the store:

```bash
python -c "import mlflow; c = mlflow.tracking.MlflowClient(); e = c.get_experiment_by_name('autodrive-rl'); runs = c.search_runs([e.experiment_id]); r = runs[0]; print(r.info.run_name, r.data.tags.get('git_commit'), len(c.list_artifacts(r.info.run_id)))"
```

Expected output: `tracking-smoke <short-hash> 3`.

Then delete the smoke artifacts (keep `mlruns/` — it demonstrates the dashboard): `models/tracking_smoke.npz`, `models/tracking_smoke_best.npz`, `runs/tracking_smoke.csv`.

- [ ] **Step 3: Full suite, commit**

Run: `python -m unittest discover -s tests`
Expected: 51 tests pass.

```bash
git add README.md
git commit -m "docs: experiment tracking with MLflow"
```

---

## Follow-up (not in this plan)

- Launch `mlflow ui` for the user and walk the dashboard together.
- The 3-seed robustness training runs, now tracked.
