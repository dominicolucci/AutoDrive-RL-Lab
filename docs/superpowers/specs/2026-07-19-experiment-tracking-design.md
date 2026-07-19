# Experiment Tracking (MLflow) — Design

Date: 2026-07-19
Status: approved by user (Approach B: thin tracker seam over local MLflow)

## Goal

Give every training run production-grade experiment tracking: parameters,
per-episode metrics, evaluation-matrix results, and model artifacts logged to
a local MLflow store, browsable and comparable in the `mlflow ui` dashboard.
Educational intent: teach the corporate pattern of wrapping a vendor tool
behind a project-owned seam.

Out of scope: hosted tracking servers, W&B, hyperparameter sweeps, CI
integration, tracking for `play` sessions.

## Decisions already made

- Tool: MLflow, fully local (no account, no network). Store: `./mlruns/`,
  gitignored.
- Dependency posture: required — `mlflow` joins `numpy` in
  `requirements.txt`; README updated (the project now has two dependencies).
- CLI default: tracking ON (`--tracking` / `--no-tracking` flags).
- Python API default: tracking OFF (`train(..., tracking=False)`), so the
  existing test suite (44 tests) runs exactly as today with no MLflow
  involvement.

## Architecture

New module `autodrive_rl/tracking.py` — the only file that imports mlflow.
`train.py` talks to a tracker object; everything MLflow-specific lives behind
it.

```
train.py ──> Tracker (protocol)
               ├── NullTracker   (no-op; tracking off / tests)
               └── MlflowTracker (real; wraps mlflow calls)
```

## Tracker interface (`autodrive_rl/tracking.py`)

```python
class NullTracker:
    def log_params(self, params: dict) -> None: ...      # no-op
    def set_tags(self, tags: dict) -> None: ...          # no-op
    def log_metrics(self, metrics: dict, step: int) -> None: ...  # no-op
    def log_artifact(self, path) -> None: ...            # no-op
    def end(self, status: str = "FINISHED") -> None: ... # no-op

class MlflowTracker:  # same five methods, backed by mlflow
    def __init__(self, run_name: str, *, experiment: str = "autodrive-rl",
                 tracking_uri: str | None = None) -> None: ...
```

- `MlflowTracker.__init__` calls `mlflow.set_tracking_uri` (when given),
  `mlflow.set_experiment(experiment)`, and `mlflow.start_run(run_name=...)`.
- `tracking_uri` parameter exists so tests can point the store at a temp
  directory; production callers omit it (default `./mlruns/`).
- `log_metrics` filters out non-numeric and empty-string values so the
  existing record dict can be passed with minimal massaging.
- `end` closes the run with the given status ("FINISHED" or "FAILED").
- A factory `create_tracker(enabled: bool, run_name: str, tracking_uri=None)`
  returns `MlflowTracker` when enabled, else `NullTracker`.

## What gets logged

**Params, once at run start** (`log_params`):
- All `DQNConfig` fields (hidden_sizes, learning_rate, gamma, batch_size,
  replay_capacity, warmup_steps, target_update_steps, epsilon_start,
  epsilon_end, epsilon_decay_steps, gradient_clip_norm)
- All `EnvConfig` fields (lane_count, lane_width_m, ..., traffic_count,
  max_steps)
- Run settings: episodes, seed, scenario, curriculum, scenario_preset, and
  the traffic/obstacles/reactive overrides (logged as "None" when unset)

**Tags** (`set_tags`): `git_commit` — short hash from
`git rev-parse --short HEAD` (best-effort: if git is unavailable the tag is
skipped, never an error).

**Per-episode metrics** (`log_metrics` with `step=episode`):
- return, rolling_return_20, steps, distance_m, mean_speed_mps, epsilon,
  mean_loss (skipped while empty)
- outcome as three 0/1 series: outcome_collision, outcome_off_road,
  outcome_complete
- traffic_count, obstacle_count, reactive_fraction (the episode's realized
  conditions)

**Evaluation metrics** (same call, eval episodes only): eval_return,
eval_distance_m, eval_safe_rate, and the six per-cell values
(eval_sparse_return, eval_sparse_safe_rate, ..., eval_dense_safe_rate).

**Artifacts, at run end** (`log_artifact`): the final model `.npz`, the best
checkpoint `.npz` (if it exists), and the metrics CSV.

## train.py integration

- `train()` gains keyword params: `tracking: bool = False`,
  `tracking_uri: str | None = None`, `run_name: str | None = None`
  (default run name: `f"{output_path.stem}-seed{seed}"`).
- Tracker created once before the episode loop via `create_tracker`;
  `log_params`/`set_tags` immediately after.
- One `tracker.log_metrics(...)` call per episode, built from the existing
  `record` dict (numeric fields only; empty strings filtered by the tracker).
- Artifacts logged after the final save; `tracker.end("FINISHED")`.
- The whole training loop is wrapped so that on an exception the tracker
  ends with status "FAILED" before the exception propagates (try/except/
  re-raise).
- CLI: `--tracking` / `--no-tracking` (argparse BooleanOptionalAction,
  default True); `--run-name` optional override. `main()` passes them
  through.
- CSV writing is unchanged — MLflow is additive, not a replacement.

## Storage and viewing

- Default store: `./mlruns/` in the project root, created by MLflow on first
  tracked run. Add `mlruns/` to `.gitignore` (run data is bulky and
  regenerable; models remain in git).
- View: `mlflow ui` from the project root → http://localhost:5000.

## Error handling

- `create_tracker(enabled=True)` when mlflow is not importable raises a
  clear `SystemExit`-style error: "mlflow is not installed. Run: python -m
  pip install -r requirements.txt, or pass --no-tracking."
- Git-commit tag lookup failing is silent (tag omitted).
- Tracker method failures during a run must not crash training: MlflowTracker
  methods wrap their mlflow calls in try/except and print a one-line warning
  on first failure, then stay silent.

## Testing

- Existing 44 tests unchanged and passing (tracking defaults off in the
  Python API).
- `NullTracker` test: train(episodes=2, tracking=False) creates no `mlruns/`
  directory in the metrics temp dir.
- Integration test (real MLflow, no mocks): train(episodes=2, tracking=True,
  tracking_uri=<temp dir as file URI>), then use the mlflow client API to
  assert: exactly one run exists; params include seed and gamma; metric
  history for "return" has 2 points; artifacts include the final model file.
- Factory test: create_tracker(False, ...) returns NullTracker;
  create_tracker(True, ...) returns MlflowTracker (skipped if mlflow
  missing).

## Documentation

- `requirements.txt`: add `mlflow>=2.9`.
- README: new "Track your experiments" section after "Train the agent":
  what tracking records, `--no-tracking` to opt out, and how to open the
  dashboard (`mlflow ui`). Update the "one dependency" phrasing in the
  Quick start.

## Success criteria

1. All existing tests pass untouched; new tests pass.
2. A tracked training run appears in `mlflow ui` with full params, episode
   curves, eval-matrix metrics, git commit tag, and downloadable model
   artifacts.
3. `--no-tracking` runs behave byte-identically to today (same console
   output, same files, no mlruns/ writes).
