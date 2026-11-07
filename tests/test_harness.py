"""Tests for the test harness itself (``tests/_runner.py``).

The harness is what keeps a non-terminating sampler from wedging a 4-core board, so it
gets tested like any other code: the timeout must fire, the abort file must be
honoured, and the suite must stay single-threaded.

Run with::

    python -m tests.test_harness
"""
from __future__ import annotations

import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

# Allow `python tests/test_harness.py` as well as `python -m tests.test_harness`: run by path,
# `tests/` is on sys.path but the repo root is not, so `from tests import ...`
# below would fail with ModuleNotFoundError before a single test ran.
import os as _os
import sys as _sys

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from tests import _runner


# --------------------------------------------------------------------------- #
#  Single-threaded                                                            #
# --------------------------------------------------------------------------- #
def test_numeric_thread_pools_are_pinned_to_one():
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert os.environ.get(key) == "1", (key, os.environ.get(key))
    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"
    assert os.environ.get("ANTISLOP_REGEX_MAX_WORKERS") == "1"


def test_regex_pool_is_lazy_and_single_worker():
    from tests import _stubs

    _stubs.install()
    from validators import regex_validator

    # Importing the module must not fork workers.
    assert regex_validator._REGEX_POOL is None
    assert regex_validator.REGEX_MAX_WORKERS == 1


def test_importing_the_sampler_starts_no_threads():
    from tests import _stubs

    _stubs.install()
    before = threading.active_count()
    import core.sampler  # noqa: F401  (import for the side effect under test)

    assert threading.active_count() == before, [
        t.name for t in threading.enumerate()
    ]


# --------------------------------------------------------------------------- #
#  Timeout                                                                    #
# --------------------------------------------------------------------------- #
def test_time_limit_aborts_a_non_terminating_loop():
    def spin():
        with _runner._time_limit(0.2):
            while True:
                pass

    started = time.monotonic()
    try:
        spin()
    except _runner.TestTimeout as exc:
        elapsed = time.monotonic() - started
        assert "budget" in str(exc), exc
    else:
        raise AssertionError("the timer did not fire")
    assert elapsed < 5.0, elapsed


def test_time_limit_does_not_disturb_a_test_that_finishes():
    with _runner._time_limit(5.0):
        time.sleep(0.05)
    # The SIGALRM handler must be restored, so the default disposition is back.
    assert signal.getsignal(signal.SIGALRM) is not None


def test_time_limit_is_reusable():
    for _ in range(3):
        with _runner._time_limit(5.0):
            pass


# --------------------------------------------------------------------------- #
#  Abort                                                                      #
# --------------------------------------------------------------------------- #
def test_abort_file_is_detected_and_can_be_cleared():
    # Redirect to a scratch file: the real one is polled by tests/run_all.py, which
    # would (correctly) abort the whole run if this test created it.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ".abort"
        _runner.set_abort_file(path)
        try:
            assert not _runner.abort_requested()
            path.write_text("stop\n")
            assert _runner.abort_requested()
            assert _runner.clear_abort_file() is True
            assert not _runner.abort_requested()
            assert _runner.clear_abort_file() is False
        finally:
            _runner.set_abort_file(None)


def test_abort_file_interrupts_a_running_test():
    def wait_for_abort():
        with _runner._time_limit(5.0):
            while True:
                pass

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ".abort"
        _runner.set_abort_file(path)
        timer = threading.Timer(0.2, lambda: path.write_text("stop\n"))
        timer.daemon = True
        timer.start()
        started = time.monotonic()
        try:
            wait_for_abort()
        except _runner.AbortRequested as exc:
            elapsed = time.monotonic() - started
            assert "abort file" in str(exc), exc
        else:
            raise AssertionError("the abort file was not noticed mid-test")
        finally:
            timer.cancel()
            _runner.set_abort_file(None)
    # Noticed on a tick, not after the full 5s budget.
    assert elapsed < 3.0, elapsed


def test_budgets_are_short_by_default():
    # A hung test must not be allowed to sit on a core for minutes.
    assert _runner.TEST_TIMEOUT_SEC <= 30
    assert _runner.SUITE_TIMEOUT_SEC <= 300


# --------------------------------------------------------------------------- #
#  Missing optional dependencies must SKIP, not fail the suite                  #
# --------------------------------------------------------------------------- #
def test_require_module_raises_skip_for_a_missing_module():
    """A missing optional dep is a skip, not a failure.

    `test_audit_fixes` used to `from api_server import server` at module scope, so
    on a box without `fastapi` the whole suite died with ModuleNotFoundError and
    reported 3 failures out of 24 -- the other 21 never ran.
    """
    try:
        _runner.require_module("definitely_not_a_real_module_xyz")
    except _runner.SkipTest as exc:
        assert "definitely_not_a_real_module_xyz" in exc.reason, exc.reason
        assert "install" in exc.reason.lower(), exc.reason
    else:
        raise AssertionError("require_module did not raise SkipTest")


def test_require_module_returns_an_installed_module():
    assert _runner.require_module("sys") is sys


def test_a_skip_does_not_fail_the_suite():
    """Run a throw-away suite: one pass, one skip; expect EXIT_SKIPPED (3).

    Executes `run_suite` in a subprocess so its output and exit code are observed
    exactly as `run_all` sees them, rather than trusting the branch.
    """
    import subprocess
    import textwrap

    script = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, {root!r})
        from tests import _runner

        def test_passes():
            pass

        def test_skips():
            _runner.require_module("definitely_not_a_real_module_xyz")

        sys.exit(_runner.run_suite("throwaway", globals()))
        """
    ).format(root=str(_runner.Path(__file__).resolve().parents[1]))

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    assert "PASS  test_passes" in proc.stdout, proc.stdout
    assert "SKIP  test_skips" in proc.stdout, proc.stdout
    assert "FAIL" not in proc.stdout, proc.stdout
    assert proc.returncode == _runner.EXIT_SKIPPED, (proc.returncode, proc.stdout)


# --------------------------------------------------------------------------- #
#  Run lock (concurrent agents)                                                #
# ---------------------------------------------------------------------------
def test_second_run_is_refused_while_the_lock_is_held():
    """Two agents in one checkout must not run the suite at the same time.

    There is no per-agent isolation: a second run would interleave its output and
    compete for the same four cores, which is exactly what the single-threaded and
    short-timer design exists to prevent.
    """
    from tests import run_all

    with tempfile.TemporaryDirectory() as tmp:
        lock_path = Path(tmp) / ".runlock"
        first = run_all.acquire_run_lock(lock_path)
        try:
            assert first is not None, "the first run should take the lock"
            assert run_all.acquire_run_lock(lock_path) is None, (
                "a concurrent run must be refused, not queued"
            )
        finally:
            if first is not None:
                run_all.release_run_lock(first)


def test_lock_is_released_after_a_run():
    from tests import run_all

    with tempfile.TemporaryDirectory() as tmp:
        lock_path = Path(tmp) / ".runlock"
        first = run_all.acquire_run_lock(lock_path)
        assert first is not None
        run_all.release_run_lock(first)
        again = run_all.acquire_run_lock(lock_path)
        assert again is not None, "the lock was not released"
        run_all.release_run_lock(again)


def test_lock_survives_a_killed_holder():
    """A SIGKILLed run must not leave the lock stuck forever.

    ``flock`` is used rather than a plain lockfile for exactly this reason: the
    kernel drops it when the process dies, so a killed run cannot block every later
    run in the checkout.
    """
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        lock_path = Path(tmp) / ".runlock"
        holder = subprocess.Popen(
            [sys.executable, "-c",
             "import fcntl, sys, time;\n"
             f"fh = open({str(lock_path)!r}, 'w');\n"
             "fcntl.flock(fh, fcntl.LOCK_EX);\n"
             "print('locked', flush=True);\n"
             "time.sleep(60)"],
            stdout=subprocess.PIPE,
        )
        try:
            assert holder.stdout.readline().strip() == b"locked"
            from tests import run_all
            assert run_all.acquire_run_lock(lock_path) is None, (
                "the live holder's lock was not respected"
            )
            holder.kill()
            holder.wait(timeout=10)
            from tests import run_all as ra
            held = ra.acquire_run_lock(lock_path)
            assert held is not None, "a killed run left the lock stuck"
            ra.release_run_lock(held)
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.wait(timeout=10)
                holder.stdout.close()


def main() -> int:
    return _runner.run_suite("test_harness", globals())


if __name__ == "__main__":
    sys.exit(main())
