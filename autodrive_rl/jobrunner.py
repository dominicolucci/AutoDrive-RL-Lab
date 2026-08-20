"""Run a project command as a subprocess and stream its output.

The launcher never reimplements training, evaluation or cloning — it builds an
argument list and runs the existing module. That keeps the command line as the
single source of truth: change how training works and the GUI follows for free,
because it is literally calling the same code a terminal would.

Two constraints shape this module.

**The window must never freeze.** Training runs for minutes. So the child
process is drained by a background thread that only ever pushes lines onto a
queue; the GUI polls that queue from the Tk main thread. Nothing here touches a
widget, and no Tk object is ever passed in — which is also why this module is
importable and testable with no display attached.

**Output must arrive while the job is running, not at the end.** Python buffers
stdout aggressively when it is not a terminal, so a long run would show nothing
for minutes and look hung. `PYTHONUNBUFFERED` plus `-u` forces line-by-line
delivery.

Stopping a job kills the whole process group on POSIX, because a child may have
spawned its own children; Windows gets an equivalent via a new process group.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

IS_WINDOWS = os.name == "nt"


class JobState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


def module_command(module: str, args: Sequence[str]) -> list[str]:
    """Build `python -u -m autodrive_rl.<module> ...` using *this* interpreter.

    `sys.executable` rather than a bare "python" so the job runs in the same
    environment as the launcher — the usual cause of "works in my terminal,
    fails in the app" is a different interpreter on PATH.
    """
    return [sys.executable, "-u", "-m", f"autodrive_rl.{module}", *args]


def format_command(argv: Sequence[str]) -> str:
    """Render argv the way a user would type it, for the command preview."""
    if IS_WINDOWS:
        return subprocess.list2cmdline(list(argv))
    import shlex

    return shlex.join(argv)


@dataclass
class Job:
    """One subprocess, its output queue, and its reader thread."""

    argv: list[str]
    cwd: Path | None = None
    state: JobState = JobState.IDLE
    returncode: int | None = None

    _process: subprocess.Popen[str] | None = field(default=None, repr=False)
    _lines: "queue.Queue[str | None]" = field(default_factory=queue.Queue, repr=False)
    _reader: threading.Thread | None = field(default=None, repr=False)
    _cancelled: bool = field(default=False, repr=False)
    _finished: threading.Event = field(default_factory=threading.Event, repr=False)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self.state is JobState.RUNNING:
            raise RuntimeError("job already running")

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"

        # A separate process group means Stop can reach grandchildren too.
        if IS_WINDOWS:
            spawn_kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        else:
            spawn_kwargs = {"start_new_session": True}

        self._process = subprocess.Popen(
            self.argv,
            cwd=str(self.cwd) if self.cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # one interleaved stream, as a terminal shows it
            text=True,
            bufsize=1,
            **spawn_kwargs,
        )
        self.state = JobState.RUNNING
        self._cancelled = False
        self._finished.clear()

        self._reader = threading.Thread(
            target=self._pump, name="jobrunner-reader", daemon=True
        )
        self._reader.start()

    def _pump(self) -> None:
        """Background thread: move child output into the queue, then signal EOF."""
        assert self._process is not None and self._process.stdout is not None
        try:
            for line in self._process.stdout:
                self._lines.put(line.rstrip("\n"))
        except (ValueError, OSError):
            # Pipe closed underneath us during a stop — expected, not an error.
            pass
        finally:
            self._process.wait()
            self._lines.put(None)  # sentinel: no more output is coming
            self._finished.set()

    # ── consumption (called from the GUI thread) ─────────────────────────────

    def drain(self, max_lines: int = 500) -> list[str]:
        """Take whatever output has arrived. Never blocks.

        Bounded so a chatty job cannot monopolise a single UI tick and make the
        window unresponsive — the remainder simply arrives on the next poll.
        """
        out: list[str] = []
        for _ in range(max_lines):
            try:
                line = self._lines.get_nowait()
            except queue.Empty:
                break
            if line is None:
                self._settle()
                break
            out.append(line)
        return out

    def _settle(self) -> None:
        """Record the final state once the output stream has ended."""
        if self._process is None:
            return
        self.returncode = self._process.returncode
        if self._cancelled:
            self.state = JobState.CANCELLED
        elif self.returncode == 0:
            self.state = JobState.SUCCEEDED
        else:
            self.state = JobState.FAILED

    @property
    def is_running(self) -> bool:
        return self.state is JobState.RUNNING

    # ── cancellation ─────────────────────────────────────────────────────────

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the job to stop, then insist."""
        if self._process is None or self._process.poll() is not None:
            return
        self._cancelled = True
        try:
            if IS_WINDOWS:
                self._process.terminate()
            else:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            return

        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                if IS_WINDOWS:
                    self._process.kill()
                else:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    def wait(self, timeout: float | None = None) -> int | None:
        """Block until the job ends. For tests and headless use, not the GUI."""
        if self._process is None:
            return None
        self._finished.wait(timeout=timeout)
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        return self._process.returncode

    def read_all(self, timeout: float | None = None) -> list[str]:
        """Run to completion and return every line. Test helper."""
        self.wait(timeout=timeout)
        collected: list[str] = []
        while True:
            batch = self.drain()
            if not batch:
                if not self.is_running:
                    break
                continue
            collected.extend(batch)
        return collected
