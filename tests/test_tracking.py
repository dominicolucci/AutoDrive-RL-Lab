from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from autodrive_rl.config import EnvConfig
from autodrive_rl.train import build_parser, train
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


if __name__ == "__main__":
    unittest.main()
