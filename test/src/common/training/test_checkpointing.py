import os
import random
import signal
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from afabench.training.checkpointing import (
    EXIT_CODE_RESUME,
    CheckpointConfig,
    TrainingCheckpointer,
    capture_rng_state,
    config_fingerprint,
    next_checkpoint_minute,
    resolve_deadline,
    restore_rng_state,
    stop_for_time_limit,
)


class FakeClock:
    """Monotonic and wall clock that only move when told to."""

    def __init__(self, start: float = 1_000_000.0):
        self.now: float = start

    def __call__(self) -> float:
        return self.now

    def advance_minutes(self, minutes: float) -> None:
        self.now += 60.0 * minutes


def make_checkpointer(
    directory: Path, clock: FakeClock, **kwargs: object
) -> TrainingCheckpointer:
    return TrainingCheckpointer(
        directory,
        clock=clock,
        wall_clock=clock,
        **kwargs,  # pyright: ignore[reportArgumentType]
    )


def saved_minutes(checkpointer: TrainingCheckpointer) -> list[int]:
    return [
        int(path.name.removeprefix("checkpoint-").removesuffix("s.pt")) // 60
        for path in checkpointer.checkpoint_paths()
    ]


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        (0.0, 10.0),
        (9.9, 10.0),
        (10.0, 20.0),
        (25.0, 30.0),
        (30.0, 60.0),
        (60.0, 120.0),
        (61.0, 120.0),
        (179.9, 180.0),
        (180.0, 240.0),
    ],
)
def test_next_checkpoint_minute_follows_default_schedule(
    elapsed: float, expected: float
) -> None:
    assert next_checkpoint_minute(elapsed) == expected


def test_saves_at_10_20_30_60_minutes_then_hourly(tmp_path: Path) -> None:
    clock = FakeClock()
    checkpointer = make_checkpointer(tmp_path, clock, keep_last=None)
    states_built: list[float] = []

    def make_state() -> dict[str, Any]:
        states_built.append(checkpointer.elapsed_minutes)
        return {"elapsed": checkpointer.elapsed_minutes}

    for _ in range(4 * 60):  # four hours, one step per minute
        clock.advance_minutes(1)
        assert not checkpointer.step(make_state)

    assert states_built == [10, 20, 30, 60, 120, 180, 240]
    assert saved_minutes(checkpointer) == [10, 20, 30, 60, 120, 180, 240]


def test_keeps_only_the_newest_checkpoints(tmp_path: Path) -> None:
    clock = FakeClock()
    checkpointer = make_checkpointer(tmp_path, clock, keep_last=2)
    for _ in range(130):
        clock.advance_minutes(1)
        checkpointer.step(dict)
    assert saved_minutes(checkpointer) == [60, 120]


def test_resume_continues_state_and_training_clock(tmp_path: Path) -> None:
    clock = FakeClock()
    first = make_checkpointer(tmp_path, clock, fingerprint="abc")
    clock.advance_minutes(30)
    first.step(lambda: {"step": 42})

    # A resubmitted job starts with a fresh process and clock.
    new_clock = FakeClock(start=5.0)
    second = make_checkpointer(tmp_path, new_clock, fingerprint="abc")
    state = second.load_latest()
    assert state == {"step": 42}
    assert second.elapsed_minutes == pytest.approx(30)

    # The next scheduled save is at 60 minutes of total training time.
    new_clock.advance_minutes(29)
    assert second.step(lambda: {"step": 43}) is False
    assert saved_minutes(second) == [30]
    new_clock.advance_minutes(1)
    second.step(lambda: {"step": 44})
    assert saved_minutes(second) == [30, 60]


def test_other_configuration_does_not_resume_or_prune(tmp_path: Path) -> None:
    clock = FakeClock()
    old = make_checkpointer(tmp_path, clock, fingerprint="old")
    clock.advance_minutes(60)
    old.save({"step": 1})

    new = make_checkpointer(tmp_path, clock, fingerprint="new", keep_last=1)
    assert new.load_latest() is None
    for _ in range(2):
        clock.advance_minutes(10)
        new.save({"step": 2})
    assert len(old.checkpoint_paths()) == 1
    assert len(new.checkpoint_paths()) == 1


def test_unreadable_newest_checkpoint_falls_back_to_previous(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    checkpointer = make_checkpointer(tmp_path, clock, keep_last=None)
    clock.advance_minutes(10)
    checkpointer.save({"step": 10})
    clock.advance_minutes(10)
    newest = checkpointer.save({"step": 20})
    newest.write_bytes(b"truncated")

    resumed = make_checkpointer(tmp_path, FakeClock())
    assert resumed.load_latest() == {"step": 10}


def test_failed_save_keeps_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    checkpointer = make_checkpointer(tmp_path, clock, keep_last=1)
    clock.advance_minutes(10)
    previous = checkpointer.save({"step": 10})

    def failing_save(*_args: object, **_kwargs: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(torch, "save", failing_save)
    clock.advance_minutes(10)
    with pytest.raises(OSError, match="disk full"):
        checkpointer.save({"step": 20})

    assert checkpointer.checkpoint_paths() == [previous]
    assert not list(tmp_path.glob(".tmp-*"))
    monkeypatch.undo()
    assert make_checkpointer(tmp_path, FakeClock()).load_latest() == {
        "step": 10
    }


def test_stops_and_saves_before_the_deadline(tmp_path: Path) -> None:
    clock = FakeClock()
    deadline = clock.now + 60.0 * 45  # job ends in 45 minutes
    checkpointer = make_checkpointer(
        tmp_path, clock, deadline=deadline, stop_margin_minutes=10
    )
    stopped_at = None
    for minute in range(1, 60):
        clock.advance_minutes(1)
        if checkpointer.step(lambda: {"step": 0}):
            stopped_at = minute
            break
    assert stopped_at == 35
    assert checkpointer.time_limit_reached
    assert saved_minutes(checkpointer) == [30, 35]


def test_disabled_checkpointer_never_writes(tmp_path: Path) -> None:
    clock = FakeClock()
    checkpointer = make_checkpointer(tmp_path, clock, enabled=False)
    clock.advance_minutes(120)
    assert checkpointer.step(dict) is False
    assert checkpointer.load_latest() is None
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_resolve_deadline_sources() -> None:
    now = 1000.0
    assert resolve_deadline(5, environ={}, now=now) == now + 300
    assert resolve_deadline(
        None, environ={"SLURM_JOB_END_TIME": "4242"}, now=now
    ) == pytest.approx(4242)
    assert (
        resolve_deadline(5, environ={"SLURM_JOB_END_TIME": "4242"}, now=now)
        == now + 300
    )
    assert resolve_deadline(None, environ={}, now=now) is None
    assert (
        resolve_deadline(None, environ={"SLURM_JOB_END_TIME": "0"}, now=now)
        is None
    )
    assert (
        resolve_deadline(None, environ={"SLURM_JOB_END_TIME": "x"}, now=now)
        is None
    )


def test_signal_requests_stop_and_handlers_are_restored(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    checkpointer = make_checkpointer(tmp_path, clock)
    previous = signal.getsignal(signal.SIGUSR1)
    with checkpointer:
        os.kill(os.getpid(), signal.SIGUSR1)
        assert checkpointer.step(lambda: {"step": 7})
    assert signal.getsignal(signal.SIGUSR1) == previous
    assert checkpointer.time_limit_reached
    assert len(checkpointer.checkpoint_paths()) == 1


def test_stop_for_time_limit_modes(tmp_path: Path) -> None:
    checkpointer = make_checkpointer(tmp_path, FakeClock())
    with pytest.raises(SystemExit) as exit_info:
        stop_for_time_limit(checkpointer, "resume")
    assert exit_info.value.code == EXIT_CODE_RESUME
    stop_for_time_limit(checkpointer, "finalize")


def test_from_config_uses_directory_next_to_output(tmp_path: Path) -> None:
    save_path = tmp_path / "model.bundle"
    checkpointer = TrainingCheckpointer.from_config(
        CheckpointConfig(), save_path=save_path, fingerprint="f00"
    )
    assert checkpointer.directory == tmp_path / "model.bundle.checkpoints/f00"
    checkpointer.save({"step": 1})
    checkpointer.cleanup()
    assert not (tmp_path / "model.bundle.checkpoints").exists()

    with pytest.raises(ValueError, match="on_timeout"):
        TrainingCheckpointer.from_config(
            CheckpointConfig(on_timeout="sometimes"), save_path=save_path
        )


def test_config_fingerprint_ignores_runtime_only_settings() -> None:
    base = {"lr": 1e-3, "device": "cpu", "use_wandb": False, "checkpoint": 1}
    moved = base | {"device": "cuda", "use_wandb": True, "checkpoint": 2}
    changed = base | {"lr": 1e-4}
    assert config_fingerprint(base) == config_fingerprint(moved)
    assert config_fingerprint(base) != config_fingerprint(changed)


def test_rng_state_roundtrip() -> None:
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1).item())  # noqa: NPY002
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(1).item())  # noqa: NPY002
    assert actual == expected
