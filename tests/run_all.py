"""Run the regression suites one at a time, under a hard kill watchdog.

This is the entry point that is safe to run on a small board (Orange Pi 5+,
RK3588): it never runs two suites at once, gives each suite a short wall-clock
budget, and ``SIGKILL``s the suite's **whole process group** when the budget runs
out.  Killing the group matters — the regex validator owns a ``ProcessPoolExecutor``,
and its workers are children of the suite process; killing only the parent leaves
them running and the board stays pegged.

Abort at any time, without waiting for the current suite::

    touch tests/.abort          # checked by the runner and inside every test
    Ctrl-C                      # or SIGTERM to the runner

Several agents may share this checkout, so the run takes an exclusive lock on
``tests/.runlock``.  A second concurrent run refuses to start rather than
interleaving output and competing for the same cores.  ``flock`` is used, not a plain
lockfile, so a killed run cannot leave the lock stuck.

Usage::

    python -m tests.run_all                 # everything
    python -m tests.run_all early_eos       # one or more suites by name
    python -m tests.run_all --list          # what would run
    python -m tests.run_all --timeout 30    # hard per-suite budget, seconds

Exit codes: 0 all passed, 1 failures, 130 aborted.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from typing import IO, Optional

from tests import _runner

# Suites in the order they run.  Kept explicit rather than discovered so the run is
# reproducible and a half-written file cannot join the run by accident.
SUITES = (
    "test_harness",
    "test_early_eos",
    "test_partial_word_bans",
    "test_config_and_sse",
    "test_sampler_filters",
)

# Grace period between SIGTERM and SIGKILL when a suite overruns.
_KILL_GRACE_SEC = 2.0
_POLL_SEC = 0.05

# Exclusive lock so two agents sharing this checkout do not run at the same time.
_RUN_LOCK_NAME = ".runlock"


def run_lock_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), _RUN_LOCK_NAME)


def acquire_run_lock(path: Optional[str] = None) -> Optional[IO[str]]:
    """Take the exclusive run lock. Returns the handle, or None if it is held.

    Non-blocking on purpose: a second agent should be told to wait, not silently
    queue up and run minutes later. ``flock`` is released by the kernel when the
    holder dies, including on ``SIGKILL``, so a killed run cannot wedge the
    checkout the way a plain lockfile would.
    """
    path = path or run_lock_path()
    handle = open(path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def release_run_lock(handle: IO[str]) -> None:
    """Release the run lock. Safe to call twice."""
    try:
        fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _terminate(proc: subprocess.Popen, reason: str) -> None:
    """Stop *proc* and every process it spawned. Never raises."""
    print(f"\nKILL  {reason}", flush=True)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=_KILL_GRACE_SEC)
        return
    except subprocess.TimeoutExpired:
        print("      SIGTERM ignored; sending SIGKILL", flush=True)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_suite(name: str, timeout: float) -> tuple[int, bool]:
    """Run one suite in its own process group. Returns (exit_code, timed_out)."""
    print(f"\n=== {name}  (hard budget {timeout:g}s)", flush=True)

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    # The suite's own SIGALRM budget should fire first and give a named failure; the
    # hard kill here is the backstop for a blocked C call or a stuck thread.
    env.setdefault("ANTISLOP_TEST_TIMEOUT", str(_runner.TEST_TIMEOUT_SEC))

    proc = subprocess.Popen(
        [sys.executable, "-m", f"tests.{name}"],
        cwd=_repo_root(),
        env=env,
        start_new_session=True,  # own process group -> killpg takes the pool too
    )

    deadline = time.monotonic() + timeout
    try:
        while True:
            code = proc.poll()
            if code is not None:
                return code, False

            if _runner.abort_requested():
                _terminate(proc, f"abort: {_runner.abort_reason()}")
                return _runner.EXIT_ABORTED, False

            if time.monotonic() >= deadline:
                _terminate(proc, f"{name} exceeded its {timeout:g}s hard budget")
                return _runner.EXIT_FAILURES, True

            time.sleep(_POLL_SEC)
    except KeyboardInterrupt:
        _runner.request_abort("KeyboardInterrupt")
        _terminate(proc, "abort: KeyboardInterrupt")
        return _runner.EXIT_ABORTED, False
    finally:
        if proc.poll() is None:
            _terminate(proc, "cleanup")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.run_all",
        description="Run the regression suites sequentially under a hard kill watchdog.",
    )
    parser.add_argument(
        "suites",
        nargs="*",
        metavar="SUITE",
        help=f"suite names to run (default: all). Known: {', '.join(SUITES)}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_runner.SUITE_TIMEOUT_SEC,
        help="hard per-suite budget in seconds (default: %(default)g)",
    )
    parser.add_argument("--list", action="store_true", help="list suites and exit")
    parser.add_argument(
        "--keep-abort-file",
        action="store_true",
        help="do not delete a stale abort file before starting",
    )
    args = parser.parse_args(argv)

    if args.list:
        for name in SUITES:
            print(name)
        return _runner.EXIT_OK

    selected = list(args.suites) or list(SUITES)
    unknown = [name for name in selected if name not in SUITES]
    if unknown:
        parser.error(
            f"unknown suite(s): {', '.join(unknown)}. Known: {', '.join(SUITES)}"
        )

    if not args.keep_abort_file and _runner.clear_abort_file():
        print(f"NOTE  removed stale abort file {_runner.abort_file()}")

    _runner.install_abort_signals()

    lock = acquire_run_lock()
    if lock is None:
        print(
            f"BUSY  another run holds {run_lock_path()}."
        )
        print("      Another agent is using this checkout; wait for it to finish.")
        print("      Do not delete the lock file while a run is in progress.")
        return _runner.EXIT_BUSY

    try:
        return _run(args, selected)
    finally:
        release_run_lock(lock)


def _run(args: argparse.Namespace, selected: list[str]) -> int:
    print(
        f"Running {len(selected)} suite(s) sequentially; "
        f"per-test {_runner.TEST_TIMEOUT_SEC:g}s, "
        f"per-suite hard kill {args.timeout:g}s."
    )
    print(f"Abort at any time: touch {_runner.abort_file()}")

    started = time.monotonic()
    results: list[tuple[str, int, bool]] = []

    for name in selected:
        if _runner.abort_requested():
            print(f"\nABORT before {name}: {_runner.abort_reason()}")
            break
        code, timed_out = run_suite(name, args.timeout)
        results.append((name, code, timed_out))
        if code == _runner.EXIT_ABORTED:
            break

    elapsed = time.monotonic() - started
    aborted = _runner.abort_requested() or any(
        code == _runner.EXIT_ABORTED for _, code, _ in results
    )

    print("\n" + "-" * 62)
    for name, code, timed_out in results:
        if code == _runner.EXIT_OK:
            status = "PASS"
        elif timed_out:
            status = "FAIL (killed: hard budget exceeded)"
        elif code == _runner.EXIT_ABORTED:
            status = "ABORTED"
        else:
            status = f"FAIL (exit {code})"
        print(f"  {status:<38} {name}")

    skipped = len(selected) - len(results)
    if skipped:
        print(f"  {'NOT RUN':<38} {skipped} suite(s)")

    print("-" * 62)
    print(f"Finished in {elapsed:.1f}s, single-threaded, one suite at a time.")

    if aborted:
        print(f"Run aborted: {_runner.abort_reason()}")
        print(f"Clear it with: rm -f {_runner.abort_file()}")
        return _runner.EXIT_ABORTED

    failures = [name for name, code, _ in results if code != _runner.EXIT_OK]
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return _runner.EXIT_FAILURES

    print("All suites passed.")
    return _runner.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
