"""Tests for bringing sources back after a restart.

Workers are child processes and die with the app, so every start begins by
marking all sources stopped. The ones flagged ``auto_start`` are then started
again by ``_resume_auto_start_sources`` -- which is the whole of what an
unattended install has after a power cut, since nobody is there to press Start.

Pinned down here: which sources that picks, that it leaves the rest alone, that
one source refusing to spawn does not strand the ones behind it, and that a
shutdown part-way through stops it launching the remainder (a resume is
staggered over seconds, and Ctrl-C during one used to keep spawning workers the
manager was already tearing down).

    cd backend && PYTHONPATH=. python tests/test_auto_start.py
"""
from __future__ import annotations

import os
import shutil
import tempfile

# A temp DATA_DIR *before* app.config is imported. A real environment variable
# beats the .env file, so this keeps the test off the developer's own app.db.
_TMP_DATA_DIR = tempfile.mkdtemp(prefix="auto-start-test-")
os.environ["DATA_DIR"] = _TMP_DATA_DIR

from app import main  # noqa: E402
from app.api import sources as sources_api  # noqa: E402
from app.database import SessionLocal, init_db  # noqa: E402
from app.detection.manager import SHUTTING_DOWN  # noqa: E402
from app.models import Source  # noqa: E402
from app.schemas import SourceCreate, SourceUpdate  # noqa: E402


class FakeManager:
    """Records what would have been spawned, without spawning anything."""

    def __init__(self):
        self.started: list[int] = []

    def start(self, source_cfg: dict) -> None:
        self.started.append(source_cfg["id"])

    def is_running(self, source_id: int) -> bool:
        return False


def setup(*auto_start_flags: int) -> FakeManager:
    """One source per flag, in order, and a manager that records starts."""
    init_db()
    db = SessionLocal()
    try:
        db.query(Source).delete()
        for index, flag in enumerate(auto_start_flags):
            db.add(
                Source(
                    name=f"cam-{index}",
                    type="rtsp",
                    url=f"rtsp://camera-{index}/stream",
                    enabled_classes=["car"],
                    auto_start=flag,
                    status="stopped",
                )
            )
        db.commit()
    finally:
        db.close()

    fake = FakeManager()
    main.get_manager = lambda: fake  # type: ignore[assignment]
    sources_api.get_manager = lambda: fake  # type: ignore[assignment]
    # No stagger: the delay between two real workers is there to spread the
    # model loading, and waiting for it here would only make the test slow.
    main.AUTO_START_STAGGER_SECONDS = 0
    SHUTTING_DOWN.clear()
    return fake


def resume(fake: FakeManager) -> None:
    thread = main._resume_auto_start_sources()
    if thread is not None:
        thread.join(10)
        assert not thread.is_alive(), "resume thread did not finish"


def statuses() -> list[tuple[int, str]]:
    db = SessionLocal()
    try:
        return [(s.auto_start, s.status) for s in db.query(Source).order_by(Source.id)]
    finally:
        db.close()


def ids() -> list[int]:
    db = SessionLocal()
    try:
        return [s.id for s in db.query(Source).order_by(Source.id)]
    finally:
        db.close()


# --- what gets resumed ------------------------------------------------------

def test_only_the_flagged_sources_are_started():
    fake = setup(1, 0, 1)
    expected = ids()
    resume(fake)
    assert fake.started == [expected[0], expected[2]], fake.started
    print("OK only_the_flagged_sources_are_started")


def test_a_resumed_source_is_marked_starting():
    fake = setup(1, 0)
    resume(fake)
    # "starting", not "running": the worker says so itself once frames arrive,
    # exactly as it does for a source started from the dashboard.
    assert statuses() == [(1, "starting"), (0, "stopped")], statuses()
    print("OK a_resumed_source_is_marked_starting")


def test_nothing_flagged_means_no_thread_and_no_starts():
    fake = setup(0, 0)
    assert main._resume_auto_start_sources() is None
    assert fake.started == []
    print("OK nothing_flagged_means_no_thread_and_no_starts")


def test_a_source_that_fails_to_start_is_skipped_not_fatal():
    fake = setup(1, 1, 1)
    expected = ids()

    # The middle camera blows up the way a spawn that runs out of handles
    # would. Nobody is at an unattended machine to start the third one by
    # hand, so it has to be started anyway.
    def start_but_fail_on_the_second(source_cfg: dict) -> None:
        if source_cfg["id"] == expected[1]:
            raise OSError("could not spawn worker")
        fake.started.append(source_cfg["id"])

    fake.start = start_but_fail_on_the_second  # type: ignore[assignment]
    resume(fake)
    assert fake.started == [expected[0], expected[2]], fake.started
    # The one that failed says so, rather than sitting on a "starting" that
    # never becomes "running".
    assert statuses() == [(1, "starting"), (1, "error"), (1, "starting")], statuses()
    print("OK a_source_that_fails_to_start_is_skipped_not_fatal")


# --- the flag itself --------------------------------------------------------

def test_the_api_round_trips_the_flag():
    setup()
    db = SessionLocal()
    try:
        created = sources_api.create_source(
            SourceCreate(
                name="cam",
                type="rtsp",
                url="rtsp://camera/stream",
                enabled_classes=["car"],
                auto_start=True,
            ),
            db,
        )
        assert created.auto_start is True
        assert sources_api.update_source(
            created.id, SourceUpdate(auto_start=False), db
        ).auto_start is False
        # An update that says nothing about auto_start must leave it alone.
        assert sources_api.update_source(
            created.id, SourceUpdate(auto_start=True), db
        ).auto_start is True
        assert sources_api.update_source(
            created.id, SourceUpdate(name="cam-renamed"), db
        ).auto_start is True
    finally:
        db.close()
    print("OK the_api_round_trips_the_flag")


# --- shutdown while resuming ------------------------------------------------

def test_a_shutdown_stops_the_resume_before_the_first_start():
    fake = setup(1, 1)
    SHUTTING_DOWN.set()
    try:
        resume(fake)
    finally:
        SHUTTING_DOWN.clear()
    assert fake.started == [], fake.started
    print("OK a_shutdown_stops_the_resume_before_the_first_start")


def test_a_shutdown_mid_resume_stops_the_rest():
    fake = setup(1, 1, 1)
    first = ids()[0]

    # Trip the shutdown from inside the first start, i.e. the same instant
    # Ctrl-C would: everything after it must be abandoned.
    def start_then_shut_down(source_cfg: dict) -> None:
        fake.started.append(source_cfg["id"])
        SHUTTING_DOWN.set()

    fake.start = start_then_shut_down  # type: ignore[assignment]
    # A stagger long enough that a resume ignoring the event would visibly
    # hang the test rather than quietly pass it.
    main.AUTO_START_STAGGER_SECONDS = 30
    try:
        resume(fake)
    finally:
        SHUTTING_DOWN.clear()
    assert fake.started == [first], fake.started
    print("OK a_shutdown_mid_resume_stops_the_rest")


if __name__ == "__main__":
    try:
        test_only_the_flagged_sources_are_started()
        test_a_resumed_source_is_marked_starting()
        test_nothing_flagged_means_no_thread_and_no_starts()
        test_a_source_that_fails_to_start_is_skipped_not_fatal()
        test_the_api_round_trips_the_flag()
        test_a_shutdown_stops_the_resume_before_the_first_start()
        test_a_shutdown_mid_resume_stops_the_rest()
        print("\nAll auto-start tests passed.")
    finally:
        shutil.rmtree(_TMP_DATA_DIR, ignore_errors=True)
