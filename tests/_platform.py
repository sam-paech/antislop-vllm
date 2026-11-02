"""Platform guard for the test suite.

The regression tests are pure-Python and normally run anywhere.  They were made
skippable on request for small ARM boards, because some of them deliberately
exercise code paths that used to **loop forever** (the back-tracking storm).  On a
constrained board a bug that fails to terminate is far more disruptive: it pegs a
core, and the surrounding test runner has no timeout of its own.

Detection is by hardware, not by architecture: ``aarch64`` alone is far too broad
(it also covers Graviton, Apple silicon under a VM, Ampere, …).  The board is
recognised from ``/proc/device-tree/model`` and ``/proc/cpuinfo``.

Overrides, highest priority first::

    ANTISLOP_SKIP_TESTS=1   skip unconditionally
    ANTISLOP_RUN_TESTS=1    run unconditionally (also overrides detection)

So CI on such a board can force a run with ``ANTISLOP_RUN_TESTS=1``.
"""
from __future__ import annotations

import contextlib
import os
import platform
import signal

# Model strings that identify a board we skip on.  Matched case-insensitively.
_SKIP_BOARD_MARKERS = (
    "orange pi 5",
    "orangepi 5",
    "rk3588",
)

_ENV_SKIP = "ANTISLOP_SKIP_TESTS"
_ENV_RUN = "ANTISLOP_RUN_TESTS"


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def detected_board() -> str:
    """Return the device-tree model string, or "" when unavailable."""
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        model = _read(path).strip().strip("\x00")
        if model:
            return model
    return ""


def skip_reason() -> str | None:
    """Return a human-readable reason to skip, or None to run normally."""
    if os.environ.get(_ENV_SKIP) == "1":
        return f"skipped because {_ENV_SKIP}=1"

    if os.environ.get(_ENV_RUN) == "1":
        return None

    model = detected_board().lower()
    haystack = model
    if not haystack:
        # Fall back to cpuinfo, which carries the SoC name on some kernels.
        haystack = _read("/proc/cpuinfo").lower()

    for marker in _SKIP_BOARD_MARKERS:
        if marker in haystack:
            shown = detected_board() or "RK3588-class board"
            return (
                f"skipped on {shown!r}: this board is excluded from the test suite "
                f"by default. Set {_ENV_RUN}=1 to run anyway."
            )
    return None


def report_skip(suite: str) -> bool:
    """Print the skip notice for *suite* and return True if it should skip."""
    reason = skip_reason()
    if reason is None:
        return False
    print(f"SKIP  {suite}: {reason}")
    return True


# Per-test wall-clock budget.  Several of these tests exist precisely because the
# code under test used to loop forever, so a regression shows up as a *hang*, not
# as a failed assertion.  Generous enough that a slow machine is never a false
# failure.
TEST_TIMEOUT_SEC = 30


class _TestTimeout(Exception):
    """Raised by the SIGALRM handler when a test exceeds its budget."""


@contextlib.contextmanager
def _time_limit(seconds: int):
    """Abort the current test after *seconds*, in-process.

    Deliberately **not** implemented with ``multiprocessing``: forking a process
    that already owns threads and a ``ProcessPoolExecutor`` (the regex validator
    keeps one) can deadlock in the child, which looks exactly like the hang the
    timeout is meant to catch.  ``SIGALRM`` needs no fork and interrupts a
    pure-Python loop reliably on Linux.  On a platform without ``SIGALRM`` the
    limit is skipped rather than silently failing the test.
    """
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise _TestTimeout(
            f"test exceeded its {seconds}s budget - the code under test did not "
            f"terminate. This is the failure mode these tests exist to catch "
            f"(an unbounded back-track loop), not a slow machine."
        )

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def run_suite(suite: str, namespace: dict) -> int:
    """Run every ``test_*`` callable in *namespace*; return a process exit code.

    Each test gets :data:`TEST_TIMEOUT_SEC` seconds, so a regression that makes
    the sampler loop forever is reported as a failure instead of wedging the
    runner.
    """
    import traceback

    tests = [
        (name, obj)
        for name, obj in sorted(namespace.items())
        if name.startswith("test_") and callable(obj)
    ]

    failures = 0
    for name, fn in tests:
        try:
            with _time_limit(TEST_TIMEOUT_SEC):
                fn()
            print(f"PASS  {name}")
        except _TestTimeout as exc:
            failures += 1
            print(f"FAIL  {name}")
            print(f"      {exc}")
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            print(traceback.format_exc())

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":  # convenience: `python -m tests._platform`
    print("board:  ", detected_board() or "(unknown)")
    print("machine:", platform.machine())
    print("skip:   ", skip_reason() or "no")
