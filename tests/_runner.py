"""Test harness: one thread, short per-test timers, abortable from outside.

These suites exist because the code under test used to loop forever, so a regression
shows up as a *hang* rather than as a failed assertion.  The harness is therefore
built around three properties, in order of importance:

1. **It cannot wedge the host.**
   Two independent limits contain a hang:

   * a per-test *soft* budget, :data:`TEST_TIMEOUT_SEC` (default 10 s), enforced
     in-process with ``SIGALRM``;
   * a per-suite *hard* budget, :data:`SUITE_TIMEOUT_SEC` (default 60 s), enforced
     by ``tests/run_all.py``, which ``SIGKILL``s the suite's whole process group.
     This one is not optional: ``SIGALRM`` cannot interrupt a blocking C call or a
     stuck thread, so only a kill is guaranteed to reclaim the machine.  Killing the
     *process group* also takes out the ``ProcessPoolExecutor`` workers the regex
     validator spawns, which would otherwise survive as orphans and keep pegging
     cores.

2. **Single-threaded.**
   Suites run one after another, tests run one at a time in the main thread, and the
   numeric / tokenizer thread pools are pinned to one thread at import time
   (see :func:`pin_single_threaded`).  ``run_suite`` also reports any thread still
   alive at the end of the suite, because on a 4-core board a leaked thread is the
   difference between a test run and an unresponsive machine.

3. **Abortable.**
   Whoever is driving the tests — a human, or an LLM that started them and changed
   its mind — can stop the run at any point, without waiting for the current test:

   * ``touch tests/.abort`` (or the file named by ``ANTISLOP_TEST_ABORT_FILE``);
   * ``Ctrl-C`` / ``SIGTERM`` / ``SIGHUP`` to the runner.

   The abort is checked every few milliseconds inside a test and between tests.  The
   runner turns it into a clean ``ABORTED`` summary and exit code
   :data:`EXIT_ABORTED` instead of a traceback.

Environment overrides::

    ANTISLOP_TEST_TIMEOUT=10        per-test budget, seconds (float)
    ANTISLOP_SUITE_TIMEOUT=60       per-suite hard budget, seconds (float)
    ANTISLOP_TEST_ABORT_FILE=path   abort file (default ``tests/.abort``)
"""
from __future__ import annotations

import contextlib
import math
import os
import signal
import threading
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Single-threaded: pin the numeric / tokenizer thread pools at import time.     #
# --------------------------------------------------------------------------- #
# Imported before ``core.sampler`` by every suite, so setting these here is early
# enough to take effect.  ``setdefault`` keeps an explicit caller choice intact.
_SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    # The regex validator keeps a process pool; one worker, created lazily.
    "ANTISLOP_REGEX_MAX_WORKERS": "1",
}


def pin_single_threaded() -> dict[str, str]:
    """Pin the numeric libraries to one thread. Returns what was actually set."""
    applied = {}
    for key, value in _SINGLE_THREAD_ENV.items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


pin_single_threaded()


# --------------------------------------------------------------------------- #
#  Budgets                                                                    #
# --------------------------------------------------------------------------- #
def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        print(f"WARN  ignoring {name}={raw!r}: not a number; using {default:g}")
        return default
    if value <= 0:
        print(f"WARN  ignoring {name}={raw!r}: must be > 0; using {default:g}")
        return default
    return value


# Short on purpose: the whole suite runs in about a second, so anything that takes
# this long is not a slow machine, it is a loop that does not terminate.
TEST_TIMEOUT_SEC: float = _env_float("ANTISLOP_TEST_TIMEOUT", 10.0)

# Backstop used by tests/run_all.py.  Must stay well above the sum of the per-test
# budgets only if you want per-test reports; keep it tight so a wedged suite dies.
SUITE_TIMEOUT_SEC: float = _env_float("ANTISLOP_SUITE_TIMEOUT", 60.0)

# How often the in-process timer re-arms itself to look at the abort file.
_TICK_SEC = 0.05

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_BUSY = 2
EXIT_ABORTED = 130


class TestTimeout(Exception):
    """Raised when a test exceeds its per-test budget."""


class AbortRequested(Exception):
    """Raised when an abort was requested while a test was running."""


# --------------------------------------------------------------------------- #
#  Abort                                                                      #
# --------------------------------------------------------------------------- #
_ENV_ABORT_FILE = "ANTISLOP_TEST_ABORT_FILE"
_DEFAULT_ABORT_FILE = Path(__file__).with_name(".abort")

_abort_lock = threading.Lock()
_abort_reason: str | None = None
_abort_file_override: Path | None = None


def abort_file() -> Path:
    """Path of the sentinel file; its existence means "stop the run"."""
    if _abort_file_override is not None:
        return _abort_file_override
    raw = os.environ.get(_ENV_ABORT_FILE)
    return Path(raw) if raw else _DEFAULT_ABORT_FILE


def set_abort_file(path: "str | Path | None") -> None:
    """Point the abort check at a different file (``None`` restores the default).

    Used by ``tests/test_harness.py`` so that exercising the abort mechanism does not
    trip the watchdog in ``tests/run_all.py``, which polls the real path.  A test must
    put this back when it is done.
    """
    global _abort_file_override
    _abort_file_override = Path(path) if path is not None else None


def abort_requested() -> bool:
    """True if an abort was signalled by file or by a signal handler."""
    if _abort_reason is not None:
        return True
    return abort_file().exists()


def abort_reason() -> str:
    if _abort_reason is not None:
        return _abort_reason
    path = abort_file()
    if path.exists():
        return f"abort file {path} exists"
    return "abort requested"


def request_abort(reason: str) -> None:
    """Record an abort request. Safe to call from a signal handler."""
    global _abort_reason
    with _abort_lock:
        if _abort_reason is None:
            _abort_reason = reason


def clear_abort_file() -> bool:
    """Delete a stale abort file. Returns True if one was removed."""
    path = abort_file()
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        print(f"WARN  could not remove stale abort file {path}: {exc}")
        return False


def _on_abort_signal(signum, frame):  # noqa: ARG001 - signal handler signature
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = str(signum)
    request_abort(f"received {name}")


def install_abort_signals() -> None:
    """Turn Ctrl-C / SIGTERM / SIGHUP into an :class:`AbortRequested`."""
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _on_abort_signal)
        except (ValueError, OSError):  # not the main thread, or not supported
            pass


# --------------------------------------------------------------------------- #
#  Per-test timer                                                             #
# --------------------------------------------------------------------------- #
def _arm(seconds: float) -> None:
    seconds = max(seconds, 0.001)
    if hasattr(signal, "setitimer"):
        signal.setitimer(signal.ITIMER_REAL, seconds)
    else:  # pragma: no cover - no sub-second timers on this platform
        signal.alarm(max(1, math.ceil(seconds)))


def _disarm() -> None:
    if hasattr(signal, "setitimer"):
        signal.setitimer(signal.ITIMER_REAL, 0)
    else:  # pragma: no cover
        signal.alarm(0)


@contextlib.contextmanager
def _time_limit(seconds: float):
    """Abort the current test after *seconds*, or as soon as an abort is asked.

    Deliberately **not** implemented with ``multiprocessing``: forking a process
    that already owns threads and a ``ProcessPoolExecutor`` (the regex validator
    keeps one) can deadlock in the child, which looks exactly like the hang the
    timeout is meant to catch.  ``SIGALRM`` needs no fork.

    The timer re-arms itself every :data:`_TICK_SEC` so the abort file is noticed
    mid-test rather than only between tests.  Where ``SIGALRM`` is unavailable the
    limit is skipped rather than silently failing; ``tests/run_all.py``'s hard kill is
    then the only protection.
    """
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    deadline = time.monotonic() + seconds

    def _handler(signum, frame):  # noqa: ARG001
        if abort_requested():
            raise AbortRequested(abort_reason())
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TestTimeout(
                f"test exceeded its {seconds:g}s budget - the code under test did "
                f"not terminate. This is the failure mode these tests exist to "
                f"catch (an unbounded back-track loop), not a slow machine."
            )
        _arm(min(_TICK_SEC, remaining))

    previous = signal.signal(signal.SIGALRM, _handler)
    _arm(min(_TICK_SEC, seconds))
    try:
        yield
    finally:
        _disarm()
        signal.signal(signal.SIGALRM, previous)


# --------------------------------------------------------------------------- #
#  Suite                                                                      #
# --------------------------------------------------------------------------- #
def _collect_tests(namespace: dict) -> list[tuple[str, object]]:
    return [
        (name, obj)
        for name, obj in sorted(namespace.items())
        if name.startswith("test_") and callable(obj)
    ]


def _live_threads() -> list[str]:
    return sorted(
        thread.name
        for thread in threading.enumerate()
        if thread is not threading.main_thread()
    )


def shutdown_regex_pool() -> bool:
    """Tear down the regex validator's ``ProcessPoolExecutor``, if one exists.

    The pool is created lazily on the first ``RegexValidator.check`` and is kept for
    the life of the process, which is right for a server and wrong for a suite: the
    executor's ``QueueFeederThread`` outlives the last test and :func:`run_suite`
    (correctly) reports it as a leaked thread.  Any suite that exercises the regex
    path would therefore end with a WARN that is not a bug in the code under test.

    Shutting the pool down here keeps that warning meaningful: after this runs, a
    live thread really is a leak.  Returns True if a pool was shut down.
    """
    try:
        import validators.regex_validator as regex_module
    except Exception:  # pragma: no cover - suite does not import validators
        return False

    pool = getattr(regex_module, "_REGEX_POOL", None)
    if pool is None:
        return False

    lock = getattr(regex_module, "_REGEX_POOL_LOCK", None)
    try:
        if lock is not None:
            with lock:
                regex_module._REGEX_POOL = None
        else:  # pragma: no cover - defensive
            regex_module._REGEX_POOL = None
        pool.shutdown(wait=True)
    except Exception as exc:  # pragma: no cover - shutdown must never fail a run
        print(f"WARN  could not shut down the regex process pool: {exc}")
    return True


def run_suite(suite: str, namespace: dict) -> int:
    """Run every ``test_*`` callable in *namespace*; return a process exit code.

    Tests run one at a time, in the main thread, each with
    :data:`TEST_TIMEOUT_SEC` seconds and each preceded by an abort check, so a
    regression that makes the sampler loop forever is reported as a failure instead
    of wedging the runner.
    """
    import traceback

    install_abort_signals()

    tests = _collect_tests(namespace)
    total = len(tests)
    passed = failed = aborted = 0
    started = time.monotonic()

    print(
        f"SUITE {suite}: {total} test(s), one at a time, "
        f"{TEST_TIMEOUT_SEC:g}s each, abort file {abort_file()}"
    )

    for name, fn in tests:
        if abort_requested():
            aborted = total - passed - failed
            break
        try:
            with _time_limit(TEST_TIMEOUT_SEC):
                fn()
        except AbortRequested:
            aborted = total - passed - failed
            print(f"ABORT {name}")
            break
        except TestTimeout as exc:
            failed += 1
            print(f"FAIL  {name}")
            print(f"      {exc}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            print(traceback.format_exc())
        else:
            passed += 1
            print(f"PASS  {name}")

    elapsed = time.monotonic() - started
    print()

    if aborted:
        print(
            f"ABORTED {suite}: {passed} passed, {failed} failed, "
            f"{aborted} not run ({elapsed:.2f}s)"
        )
        print(f"        reason: {abort_reason()}")
        print(f"        clear it with: rm -f {abort_file()}")
        return EXIT_ABORTED

    print(f"{passed}/{total} passed in {elapsed:.2f}s (single-threaded)")

    # The regex validator's process pool is created on first use and never closed,
    # so it would otherwise be reported below as a leak.  Closing it here is what
    # makes the check meaningful (see shutdown_regex_pool).
    shutdown_regex_pool()

    leaked = _live_threads()
    if leaked:
        print(
            f"WARN  {len(leaked)} thread(s) still alive at exit: {', '.join(leaked)}"
        )

    return EXIT_FAILURES if failed else EXIT_OK


if __name__ == "__main__":  # convenience: `python -m tests._runner`
    print("per-test budget:", f"{TEST_TIMEOUT_SEC:g}s")
    print("per-suite budget:", f"{SUITE_TIMEOUT_SEC:g}s")
    print("abort file:      ", abort_file(), "(present)" if abort_file().exists() else "")
    print("pinned env:      ", ", ".join(f"{k}={v}" for k, v in _SINGLE_THREAD_ENV.items()))
    print("abort requested: ", abort_requested())
