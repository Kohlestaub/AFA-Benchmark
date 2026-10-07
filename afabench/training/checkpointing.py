"""
Time-based training checkpoints that survive Slurm time limits.

A `TrainingCheckpointer` is consulted once per training step. It saves the
training state after fixed amounts of cumulative training time (by default
after 10, 20, 30 and 60 minutes, then every hour) and once more when the job
is about to reach its time limit or receives SIGTERM/SIGUSR1. Training time
accumulates across resumed runs, so the schedule refers to the total training
time, not to the time spent in the current Slurm job.

Checkpoints are written to a temporary file that is then renamed, so a job
that is killed while saving leaves the previous checkpoint intact.

Typical use in a training loop::

    checkpointer = TrainingCheckpointer.from_config(
        cfg.checkpoint, save_path=Path(cfg.save_path), fingerprint=fingerprint
    )
    start_step = 0
    resumed = checkpointer.load_latest()
    if resumed is not None:
        model.load_state_dict(resumed["model"])
        start_step = resumed["step"]
    with checkpointer:  # handles SIGTERM/SIGUSR1 while training
        for step in range(start_step, n_steps):
            train_one_step()
            if checkpointer.step(lambda: {"model": model.state_dict(), ...}):
                stop_for_time_limit(checkpointer, cfg.checkpoint.on_timeout)
                break
    save_final_result()
    checkpointer.cleanup()
"""

import hashlib
import json
import logging
import math
import os
import random
import signal
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType, TracebackType
from typing import Any, Self

import numpy as np
import torch

log = logging.getLogger(__name__)

DEFAULT_SCHEDULE_MINUTES: tuple[float, ...] = (10.0, 20.0, 30.0, 60.0)
DEFAULT_INTERVAL_MINUTES = 60.0
DEFAULT_STOP_MARGIN_MINUTES = 10.0

# Exit status of a training script that stopped early because of the time
# limit and wants to be resubmitted. 75 is EX_TEMPFAIL ("try again later").
EXIT_CODE_RESUME = 75

ON_TIMEOUT_RESUME = "resume"
ON_TIMEOUT_FINALIZE = "finalize"

_CHECKPOINT_FORMAT_VERSION = 1
_CHECKPOINT_PREFIX = "checkpoint-"
_CHECKPOINT_SUFFIX = ".pt"
_STOP_SIGNALS = (signal.SIGTERM, signal.SIGUSR1)


@dataclass
class CheckpointConfig:
    """
    Checkpoint settings shared by training scripts.

    Attributes:
        enabled: Save checkpoints and resume from them.
        dir: Where to keep checkpoints. Defaults to `<save_path>.checkpoints`
            next to the script's output, which Snakemake does not delete when
            a job fails. Each configuration gets its own subdirectory.
        schedule_minutes: Cumulative training times (minutes) at which to save.
        interval_minutes: After the last scheduled time, save this often.
        stop_margin_minutes: Stop and save this long before the time limit.
        time_limit_minutes: Time limit of the current run, counted from the
            start of the script. Defaults to the Slurm job end time
            (`SLURM_JOB_END_TIME`) when that is set.
        on_timeout: What to do when the time limit is reached. "resume" saves
            and exits with status 75 so that a resubmitted job continues;
            "finalize" stops training and saves the result as usual.
        keep_last: How many checkpoints to keep. `None` keeps all of them.
        keep_after_success: Keep the checkpoints once training has finished.
    """

    enabled: bool = True
    dir: str | None = None
    schedule_minutes: list[float] = field(
        default_factory=lambda: list(DEFAULT_SCHEDULE_MINUTES)
    )
    interval_minutes: float = DEFAULT_INTERVAL_MINUTES
    stop_margin_minutes: float = DEFAULT_STOP_MARGIN_MINUTES
    time_limit_minutes: float | None = None
    on_timeout: str = ON_TIMEOUT_RESUME
    keep_last: int | None = 2
    keep_after_success: bool = False


def next_checkpoint_minute(
    elapsed_minutes: float,
    schedule_minutes: Sequence[float] = DEFAULT_SCHEDULE_MINUTES,
    interval_minutes: float = DEFAULT_INTERVAL_MINUTES,
) -> float:
    """Return the first checkpoint time strictly after `elapsed_minutes`."""
    for minute in sorted(schedule_minutes):
        if minute > elapsed_minutes:
            return float(minute)
    if interval_minutes <= 0:
        return math.inf
    last = max(schedule_minutes, default=0.0)
    n_intervals = math.floor((elapsed_minutes - last) / interval_minutes) + 1
    return float(last + n_intervals * interval_minutes)


def resolve_deadline(
    time_limit_minutes: float | None = None,
    environ: Mapping[str, str] | None = None,
    now: float | None = None,
) -> float | None:
    """
    Return the Unix time at which the current job will be stopped, if known.

    An explicit `time_limit_minutes` (counted from `now`) takes precedence
    over the Slurm job end time in `SLURM_JOB_END_TIME`.
    """
    now = time.time() if now is None else now
    if time_limit_minutes is not None:
        return now + 60.0 * time_limit_minutes
    environ = os.environ if environ is None else environ
    raw_end_time = environ.get("SLURM_JOB_END_TIME")
    if raw_end_time is None:
        return None
    try:
        end_time = float(raw_end_time)
    except ValueError:
        log.warning(f"Ignoring unparsable SLURM_JOB_END_TIME={raw_end_time!r}")
        return None
    # Slurm reports 0 for jobs without a time limit.
    return end_time if end_time > 0 else None


def config_fingerprint(
    config: Mapping[str, Any],
    exclude: Sequence[str] = ("device", "use_wandb", "checkpoint"),
) -> str:
    """Hash the settings that determine a training run's result."""
    relevant = {k: v for k, v in config.items() if k not in exclude}
    encoded = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def capture_rng_state() -> dict[str, Any]:
    """Return the state of every random number generator used in training."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore random number generators from `capture_rng_state` output."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_state)
        except (RuntimeError, IndexError) as error:
            log.warning(f"Could not restore CUDA RNG state: {error}")


class TrainingCheckpointer:
    """Save and restore training state on a wall-clock schedule."""

    def __init__(
        self,
        directory: Path,
        *,
        schedule_minutes: Sequence[float] = DEFAULT_SCHEDULE_MINUTES,
        interval_minutes: float = DEFAULT_INTERVAL_MINUTES,
        stop_margin_minutes: float = DEFAULT_STOP_MARGIN_MINUTES,
        deadline: float | None = None,
        keep_last: int | None = 2,
        fingerprint: str | None = None,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ):
        if keep_last is not None and keep_last < 1:
            msg = f"keep_last must be at least 1 or None, got {keep_last}"
            raise ValueError(msg)
        # Runs with different settings get separate subdirectories, so that a
        # changed configuration never resumes from, or prunes, another run's
        # checkpoints.
        self.base_directory: Path = directory
        self.directory: Path = (
            directory / fingerprint if fingerprint is not None else directory
        )
        self.schedule_minutes: tuple[float, ...] = tuple(schedule_minutes)
        self.interval_minutes: float = interval_minutes
        self.stop_margin_minutes: float = stop_margin_minutes
        self.deadline: float | None = deadline
        self.keep_last: int | None = keep_last
        self.fingerprint: str | None = fingerprint
        self.enabled: bool = enabled
        self._clock: Callable[[], float] = clock
        self._wall_clock: Callable[[], float] = wall_clock
        self._start: float = clock()
        self._elapsed_offset: float = 0.0
        self._next_minute: float = next_checkpoint_minute(
            0.0, self.schedule_minutes, self.interval_minutes
        )
        self._stop_requested: bool = False
        self._previous_handlers: dict[int, Any] = {}
        self.time_limit_reached: bool = False
        self.last_saved_path: Path | None = None

    @classmethod
    def from_config(
        cls,
        cfg: CheckpointConfig,
        *,
        save_path: Path,
        fingerprint: str | None = None,
    ) -> Self:
        if cfg.on_timeout not in (ON_TIMEOUT_RESUME, ON_TIMEOUT_FINALIZE):
            msg = (
                f"checkpoint.on_timeout must be '{ON_TIMEOUT_RESUME}' or "
                f"'{ON_TIMEOUT_FINALIZE}', got {cfg.on_timeout!r}"
            )
            raise ValueError(msg)
        directory = (
            Path(cfg.dir)
            if cfg.dir is not None
            else save_path.with_name(save_path.name + ".checkpoints")
        )
        deadline = resolve_deadline(cfg.time_limit_minutes)
        if cfg.enabled:
            log.info(
                f"Checkpoints go to {directory} after "
                f"{list(cfg.schedule_minutes)} minutes of training, then "
                f"every {cfg.interval_minutes} minutes."
            )
            if deadline is None:
                log.info("No time limit known; saving on schedule only.")
            else:
                minutes_left = (deadline - time.time()) / 60
                log.info(
                    f"Time limit in {minutes_left:.1f} minutes; stopping "
                    f"{cfg.stop_margin_minutes} minutes before it."
                )
        return cls(
            directory,
            schedule_minutes=cfg.schedule_minutes,
            interval_minutes=cfg.interval_minutes,
            stop_margin_minutes=cfg.stop_margin_minutes,
            deadline=deadline,
            keep_last=cfg.keep_last,
            fingerprint=fingerprint,
            enabled=cfg.enabled,
        )

    @property
    def elapsed_seconds(self) -> float:
        """Cumulative training time, including resumed runs."""
        return self._elapsed_offset + (self._clock() - self._start)

    @property
    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds / 60.0

    def checkpoint_paths(self) -> list[Path]:
        """Existing checkpoint files, oldest first."""
        if not self.directory.is_dir():
            return []
        return sorted(
            path
            for path in self.directory.iterdir()
            if path.name.startswith(_CHECKPOINT_PREFIX)
            and path.name.endswith(_CHECKPOINT_SUFFIX)
        )

    def load_latest(
        self, map_location: torch.device | str | None = None
    ) -> dict[str, Any] | None:
        """
        Load the newest usable checkpoint and continue its training clock.

        Checkpoints written for a different fingerprint, or that cannot be
        read, are skipped. Returns the saved state, or None to start fresh.
        """
        if not self.enabled:
            return None
        for path in reversed(self.checkpoint_paths()):
            try:
                payload = torch.load(
                    path, map_location=map_location, weights_only=False
                )
            except Exception as error:  # noqa: BLE001
                log.warning(f"Skipping unreadable checkpoint {path}: {error}")
                continue
            if payload.get("fingerprint") != self.fingerprint:
                log.warning(
                    f"Skipping checkpoint {path}: it was written for a "
                    "different configuration."
                )
                continue
            self._elapsed_offset = float(payload["elapsed_seconds"])
            self._start = self._clock()
            self._next_minute = next_checkpoint_minute(
                self.elapsed_minutes,
                self.schedule_minutes,
                self.interval_minutes,
            )
            log.info(
                f"Resuming from {path} after {self.elapsed_minutes:.1f} "
                "minutes of training."
            )
            return payload["state"]
        return None

    def should_stop(self) -> bool:
        """Whether training should stop now because of the time limit."""
        if self._stop_requested:
            return True
        if self.deadline is None:
            return False
        stop_at = self.deadline - 60.0 * self.stop_margin_minutes
        return self._wall_clock() >= stop_at

    def step(self, make_state: Callable[[], dict[str, Any]]) -> bool:
        """
        Save on schedule; return True if training should stop for time.

        `make_state` is only called when a checkpoint is actually written, so
        building the state dict costs nothing on most steps.
        """
        stop = self.should_stop()
        if not self.enabled:
            self.time_limit_reached = stop
            return stop
        if stop:
            self.save(make_state(), reason="time limit")
            self.time_limit_reached = True
        elif self.elapsed_minutes >= self._next_minute:
            self.save(make_state(), reason="schedule")
        return stop

    def save(self, state: dict[str, Any], reason: str = "manual") -> Path:
        """Write a checkpoint atomically and prune old ones."""
        self.directory.mkdir(parents=True, exist_ok=True)
        elapsed = self.elapsed_seconds
        path = (
            self.directory
            / f"{_CHECKPOINT_PREFIX}{int(elapsed):09d}s{_CHECKPOINT_SUFFIX}"
        )
        payload = {
            "format_version": _CHECKPOINT_FORMAT_VERSION,
            "state": state,
            "elapsed_seconds": elapsed,
            "fingerprint": self.fingerprint,
            "reason": reason,
            "saved_at": self._wall_clock(),
        }
        tmp_path = path.with_name(f".tmp-{os.getpid()}-{path.name}")
        try:
            with tmp_path.open("wb") as f:
                torch.save(payload, f)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)
        self.last_saved_path = path
        self._next_minute = next_checkpoint_minute(
            elapsed / 60.0, self.schedule_minutes, self.interval_minutes
        )
        log.info(
            f"Saved checkpoint ({reason}) after {elapsed / 60:.1f} minutes of "
            f"training to {path}"
        )
        self._prune()
        return path

    def _prune(self) -> None:
        if self.keep_last is None:
            return
        for old_path in self.checkpoint_paths()[: -self.keep_last]:
            old_path.unlink(missing_ok=True)

    def cleanup(self) -> None:
        """Delete this run's checkpoints, for example after training."""
        for path in self.checkpoint_paths():
            path.unlink(missing_ok=True)
        for directory in (self.directory, self.base_directory):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()

    def request_stop(self) -> None:
        """Ask the training loop to save and stop at the next step."""
        self._stop_requested = True

    def _handle_signal(self, signum: int, _frame: FrameType | None) -> None:
        log.warning(
            f"Received {signal.Signals(signum).name}; saving a checkpoint "
            "and stopping at the end of this training step."
        )
        self.request_stop()

    def __enter__(self) -> Self:
        for signum in _STOP_SIGNALS:
            try:
                self._previous_handlers[signum] = signal.signal(
                    signum, self._handle_signal
                )
            except ValueError:
                # Signal handlers can only be installed in the main thread.
                log.debug(f"Could not install a handler for {signum}")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)
        self._previous_handlers.clear()


def stop_for_time_limit(
    checkpointer: TrainingCheckpointer, on_timeout: str
) -> None:
    """
    Act on a time-limit stop according to `CheckpointConfig.on_timeout`.

    With "resume", exit with `EXIT_CODE_RESUME` so that the step fails and
    runs again later (in the next Slurm job, or through Snakemake
    `--retries`); the next run continues from the checkpoint that was just
    saved. With "finalize", return so the caller can save its result from
    the state reached so far.
    """
    if on_timeout == ON_TIMEOUT_RESUME:
        log.warning(
            f"Stopping after {checkpointer.elapsed_minutes:.1f} minutes of "
            "training because of the time limit. Run the job again to "
            f"continue from {checkpointer.last_saved_path}."
        )
        raise SystemExit(EXIT_CODE_RESUME)
    log.warning(
        f"Stopping after {checkpointer.elapsed_minutes:.1f} minutes of "
        "training because of the time limit; saving the result reached so "
        "far."
    )
