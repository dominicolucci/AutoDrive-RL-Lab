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
