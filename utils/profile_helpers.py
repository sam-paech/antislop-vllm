# utils/profile_helpers.py
"""
Light-weight function-level timing that can be enabled at run-time with
`export ANTISLOP_PROFILE=1`.

Usage
-----
    from utils.profile_helpers import profile_zone

    @profile_zone          # decorator
    def some_hot_function(...):
        ...

or

    with profile_zone("custom-label"):
        ...

At the end of the program a table is printed to stderr — only if the
env-var is set.
"""
import os, sys, time, threading, atexit
from functools import wraps
from collections import defaultdict

_ENABLED = os.getenv("ANTISLOP_PROFILE") == "1"
_lock    = threading.Lock()
_stats   = defaultdict(lambda: [0.0, 0])   # label -> [accum_sec, calls]

def _record(label: str, elapsed: float):
    with _lock:
        s, n = _stats[label]
        _stats[label] = [s + elapsed, n + 1]

def _print():
    if not _stats:                                       # nothing recorded
        return
    sys.stderr.write("\n── timing summary ─────────────────────────────\n")
    sys.stderr.write("   ms/call    calls   label\n")
    for lbl, (tot, n) in sorted(_stats.items(),
                                key=lambda kv: kv[1][0], reverse=True):
        sys.stderr.write(f"{tot*1000/n:10.2f}  {n:7d}   {lbl}\n")
    sys.stderr.write("───────────────────────────────────────────────\n")

if _ENABLED:
    atexit.register(_print)

class profile_zone:
    def __init__(self, label_or_func):
        self.label = label_or_func if isinstance(label_or_func, str) else label_or_func.__qualname__
        self.func  = None if isinstance(label_or_func, str) else label_or_func

    def __enter__(self):
        self._t0 = time.perf_counter()

    def __exit__(self, exc_type, exc, tb):
        _record(self.label, time.perf_counter() - self._t0)

    def __call__(self, func):
        if not _ENABLED:
            return func
        @wraps(func)
        def wrapper(*args, **kw):
            t0 = time.perf_counter()
            try:
                return func(*args, **kw)
            finally:
                _record(self.label, time.perf_counter() - t0)
        return wrapper if self.func is None else wrapper(self.func)
