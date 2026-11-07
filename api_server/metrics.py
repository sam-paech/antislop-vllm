"""Prometheus metrics for the antislop OpenAI-compatible API server.

Why this exists
---------------
The dataset generator's 📊 Metrics tab polls ``<host>:<port>/metrics`` for every configured API
slot.  A plain vLLM serves that endpoint; this proxy did not, so every slot pointed at the proxy
was reported as *unreachable* and **no** figure appeared for it -- even though the proxy forwards
to a vLLM that does serve ``/metrics``.

``GET /metrics`` on this server therefore does two things:

1. **Proxies the upstream vLLM body verbatim.**  The generator's existing ``vllm:`` parsing keeps
   working unchanged, which is the backwards-compatibility half of the fix.
2. **Appends this server's own ``antislop:`` metrics** to the same body: requests, in-flight
   generations, tokens, ban events by validator and outcome, and upstream failures.  The proxy's
   whole purpose is the ban/back-track loop, and until now none of it was observable.

A dead or absent upstream must never blank the endpoint: the ``antislop:`` figures are still
served, with ``antislop:upstream_metrics_scrape_total{result="error"}`` saying why the ``vllm:``
ones are missing.  That is the difference between "the proxy is unreachable" and "the proxy is up
but its backend is not", which is exactly the distinction the Metrics tab needs.

No third-party dependency
-------------------------
``prometheus_client`` is not in ``requirements.txt`` and this project targets a small board, so the
exposition format is written out here: ``# HELP`` / ``# TYPE`` once per metric, then one sample per
label set.  It is ~150 lines, thread-safe, and testable without a server.

Importing this module must never issue a request or start anything: the tests, the CLI and every
headless run import it.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import requests

# The upstream scrape is a cheap local call; a short timeout keeps a dead backend from stalling
# the generator's poll (which uses 3 s of its own).
UPSTREAM_TIMEOUT_SECONDS = 2.0

METRICS_PATH = "/metrics"


# --------------------------------------------------------------------------- #
#  Exposition helpers                                                        #
# --------------------------------------------------------------------------- #
def _escape_label_value(value: object) -> str:
    """Escape a label value the way the Prometheus text format requires.

    Backslash, double quote and newline are the only three characters that must be escaped.
    """
    text = str(value)
    return (text.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n"))


def _format_value(value: float) -> str:
    """Render a sample value.

    ``repr``-style float formatting keeps integers readable (``3.0``, not ``3``) and matches what
    vLLM emits, so a parser written against vLLM sees the same shapes here.
    """
    try:
        return repr(float(value))
    except (TypeError, ValueError):
        return "0.0"


def _labels_key(labels: Optional[Dict[str, object]], names: Tuple[str, ...]) -> Tuple[str, ...]:
    """Normalise a label dict to the metric's declared label order, filling absent ones with ''."""
    if not labels:
        return tuple("" for _ in names)
    return tuple(str(labels.get(name, "")) for name in names)


def _render_labels(names: Tuple[str, ...], values: Tuple[str, ...]) -> str:
    if not names:
        return ""
    pairs = ",".join(
        '{}="{}"'.format(name, _escape_label_value(value))
        for name, value in zip(names, values)
    )
    return "{" + pairs + "}"


# --------------------------------------------------------------------------- #
#  Metric types                                                              #
# --------------------------------------------------------------------------- #
class _Metric:
    """One metric family: a name, a help string, its label names and its series."""

    kind = "untyped"

    def __init__(self, name: str, help_text: str, label_names: Tuple[str, ...] = ()) -> None:
        self.name = name
        self.help_text = help_text
        self.label_names = tuple(label_names)
        # Insertion-ordered so the output is stable between renders (a diff is readable, and a
        # test can assert on a whole body).
        self._series: Dict[Tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    # -- series access ---------------------------------------------------- #
    def _key(self, labels: Optional[Dict[str, object]]) -> Tuple[str, ...]:
        return _labels_key(labels, self.label_names)

    def _add(self, labels: Optional[Dict[str, object]], amount: float) -> None:
        key = self._key(labels)
        with self._lock:
            self._series[key] = self._series.get(key, 0.0) + float(amount)

    def _set(self, labels: Optional[Dict[str, object]], value: float) -> None:
        key = self._key(labels)
        with self._lock:
            self._series[key] = float(value)

    def get(self, labels: Optional[Dict[str, object]] = None) -> float:
        """Current value of one series (0.0 when it has never been touched)."""
        with self._lock:
            return self._series.get(self._key(labels), 0.0)

    def clear(self) -> None:
        with self._lock:
            self._series.clear()

    # -- rendering -------------------------------------------------------- #
    def samples(self) -> List[Tuple[str, Tuple[str, ...], float]]:
        """``[(sample_name, label_values, value), ...]`` for the exposition body."""
        with self._lock:
            items = list(self._series.items())
        return [(self.name, key, value) for key, value in items]

    def render(self) -> str:
        lines = [
            "# HELP {} {}".format(self.name, self.help_text),
            "# TYPE {} {}".format(self.name, self.kind),
        ]
        for sample_name, key, value in self.samples():
            lines.append("{}{} {}".format(
                sample_name, _render_labels(self.label_names, key), _format_value(value)))
        return "\n".join(lines) + "\n"


class Counter(_Metric):
    """A monotonically increasing value.  ``inc()`` adds; there is no ``dec()``."""

    kind = "counter"

    def inc(self, labels: Optional[Dict[str, object]] = None, amount: float = 1.0) -> None:
        self._add(labels, amount)


class Gauge(_Metric):
    """A value that can go up and down (in-flight requests, last-scrape timestamp)."""

    kind = "gauge"

    def inc(self, labels: Optional[Dict[str, object]] = None, amount: float = 1.0) -> None:
        self._add(labels, amount)

    def dec(self, labels: Optional[Dict[str, object]] = None, amount: float = 1.0) -> None:
        self._add(labels, -float(amount))

    def set(self, value: float, labels: Optional[Dict[str, object]] = None) -> None:
        self._set(labels, value)


class Summary(_Metric):
    """A sum/count pair (``_sum`` and ``_count``), which is all a duration needs here.

    Deliberately not a histogram: the generator reads rates and totals, and quantile buckets would
    be a lot of exposition for a figure nobody plots.
    """

    kind = "summary"

    def __init__(self, name: str, help_text: str, label_names: Tuple[str, ...] = ()) -> None:
        super().__init__(name, help_text, label_names)
        # A summary needs two numbers per series, while the other types keep one float in
        # ``_series``; two parallel dicts are simpler than a second class hierarchy.
        self._sums: Dict[Tuple[str, ...], float] = {}
        self._counts: Dict[Tuple[str, ...], float] = {}

    def observe(self, value: float, labels: Optional[Dict[str, object]] = None) -> None:
        key = self._key(labels)
        with self._lock:
            self._sums[key] = self._sums.get(key, 0.0) + float(value)
            self._counts[key] = self._counts.get(key, 0.0) + 1.0

    def get(self, labels: Optional[Dict[str, object]] = None) -> float:
        """The sum of the observed values (0.0 when nothing was observed)."""
        with self._lock:
            return self._sums.get(self._key(labels), 0.0)

    def clear(self) -> None:
        with self._lock:
            self._sums.clear()
            self._counts.clear()

    def samples(self) -> List[Tuple[str, Tuple[str, ...], float]]:
        with self._lock:
            keys = list(self._sums.keys())
            sum_items = [(self.name + "_sum", key, self._sums[key]) for key in keys]
            count_items = [(self.name + "_count", key, self._counts.get(key, 0.0))
                           for key in keys]
        return sum_items + count_items


# --------------------------------------------------------------------------- #
#  The registry                                                              #
# --------------------------------------------------------------------------- #
class MetricsRegistry:
    """A tiny registry: create-or-return a metric by name, and render the whole body."""

    def __init__(self) -> None:
        self._metrics: Dict[str, _Metric] = {}
        self._lock = threading.Lock()

    def _get_or_create(self, cls, name, help_text, label_names):
        with self._lock:
            existing = self._metrics.get(name)
            if existing is not None:
                return existing
            metric = cls(name, help_text, tuple(label_names or ()))
            self._metrics[name] = metric
            return metric

    def counter(self, name: str, help_text: str, label_names=()) -> Counter:
        return self._get_or_create(Counter, name, help_text, label_names)

    def gauge(self, name: str, help_text: str, label_names=()) -> Gauge:
        return self._get_or_create(Gauge, name, help_text, label_names)

    def summary(self, name: str, help_text: str, label_names=()) -> Summary:
        return self._get_or_create(Summary, name, help_text, label_names)

    def render(self) -> str:
        """The whole exposition body, or ``""`` when nothing has been registered."""
        with self._lock:
            metrics = list(self._metrics.values())
        return "".join(metric.render() for metric in metrics)

    def reset(self) -> None:
        """Clear every series but keep the metric definitions (used by the tests)."""
        with self._lock:
            metrics = list(self._metrics.values())
        for metric in metrics:
            metric.clear()


# --------------------------------------------------------------------------- #
#  This server's metric set                                                  #
# --------------------------------------------------------------------------- #
SERVER_METRICS = MetricsRegistry()

# --- requests ---------------------------------------------------------------- #
REQUESTS_TOTAL = SERVER_METRICS.counter(
    "antislop:requests_total",
    "Chat completion requests handled by this proxy, by endpoint and HTTP status.",
    ("endpoint", "status"),
)
REQUESTS_IN_FLIGHT = SERVER_METRICS.gauge(
    "antislop:requests_in_flight",
    "Generations currently running in this proxy.",
)
REQUEST_DURATION = SERVER_METRICS.summary(
    "antislop:request_duration_seconds",
    "Wall-clock time one request spent inside this proxy.",
)
STREAMING_TOTAL = SERVER_METRICS.counter(
    "antislop:streaming_requests_total",
    "Requests split by whether the client asked for an SSE stream.",
    ("stream",),
)

# --- generations ------------------------------------------------------------- #
GENERATIONS_TOTAL = SERVER_METRICS.counter(
    "antislop:generations_total",
    "Completed generations by the finish_reason the sampler ended for.",
    ("finish_reason",),
)
PROMPT_TOKENS_TOTAL = SERVER_METRICS.counter(
    "antislop:prompt_tokens_total",
    "Prompt tokens counted for the responses this proxy returned.",
)
GENERATION_TOKENS_TOTAL = SERVER_METRICS.counter(
    "antislop:generation_tokens_total",
    "Completion tokens counted for the responses this proxy returned.",
)

# --- the ban / back-track loop ---------------------------------------------- #
BAN_EVENTS_TOTAL = SERVER_METRICS.counter(
    "antislop:ban_events_total",
    "Validator violations, by validator type and whether the back-track fixed them.",
    ("validator", "fixed"),
)
BACKTRACKS_TOTAL = SERVER_METRICS.counter(
    "antislop:backtracks_total",
    "Back-track attempts made while resolving a violation.",
)

# --- failures ---------------------------------------------------------------- #
UPSTREAM_ERRORS_TOTAL = SERVER_METRICS.counter(
    "antislop:upstream_errors_total",
    "Requests that ended in an upstream/backend failure rather than an answer.",
)
UPSTREAM_SCRAPES_TOTAL = SERVER_METRICS.counter(
    "antislop:upstream_metrics_scrape_total",
    "Upstream /metrics scrapes attempted by this proxy, by result.",
    ("result",),
)
UPSTREAM_LAST_SCRAPE = SERVER_METRICS.gauge(
    "antislop:upstream_metrics_last_scrape_timestamp_seconds",
    "Unix timestamp of the last successful upstream /metrics scrape (0 = never).",
)


# --------------------------------------------------------------------------- #
#  Upstream scrape                                                           #
# --------------------------------------------------------------------------- #
def upstream_metrics_url(api_base_url: Optional[str]) -> Optional[str]:
    """The ``/metrics`` URL of the vLLM behind this proxy's ``api_base_url``.

    vLLM serves Prometheus metrics on the same host and port as its OpenAI API at the root path,
    so the configured path (``/v1``) is discarded -- the same rule the generator's
    ``vllm_metrics.metrics_url_for_api_url()`` applies.  ``None`` when there is no usable URL, so
    an unconfigured backend simply contributes no upstream body.
    """
    if not api_base_url:
        return None
    text = str(api_base_url).strip()
    if not text:
        return None
    if "://" not in text:
        text = "http://" + text
    try:
        parts = urlsplit(text)
    except ValueError:
        return None
    if not parts.scheme or not parts.netloc:
        return None
    if any(char.isspace() for char in parts.netloc):
        return None
    return "{}://{}{}".format(parts.scheme, parts.netloc, METRICS_PATH)


def scrape_upstream_metrics(api_base_url: Optional[str]) -> str:
    """Fetch the upstream ``/metrics`` body, or ``""`` when it cannot be read.

    Every failure is recorded in ``antislop:upstream_metrics_scrape_total`` and swallowed: a dead
    backend must still produce a body (the antislop figures), never an error page, because the
    generator treats a non-200 as "this endpoint is unreachable".
    """
    url = upstream_metrics_url(api_base_url)
    if url is None:
        UPSTREAM_SCRAPES_TOTAL.inc({"result": "unconfigured"})
        return ""
    try:
        response = requests.get(url, timeout=UPSTREAM_TIMEOUT_SECONDS)
    except Exception:
        UPSTREAM_SCRAPES_TOTAL.inc({"result": "error"})
        return ""
    if response.status_code != 200:
        UPSTREAM_SCRAPES_TOTAL.inc({"result": "error"})
        return ""
    UPSTREAM_SCRAPES_TOTAL.inc({"result": "ok"})
    UPSTREAM_LAST_SCRAPE.set(time.time())
    return response.text or ""


def render_metrics_body(api_base_url: Optional[str]) -> str:
    """The full ``/metrics`` body: the upstream vLLM text, then this server's own metrics.

    The upstream body is passed through **verbatim** (not re-parsed and re-emitted) so a newer
    vLLM metric this module has never heard of still reaches the generator.
    """
    upstream = scrape_upstream_metrics(api_base_url)
    own = SERVER_METRICS.render()
    if upstream and not upstream.endswith("\n"):
        upstream += "\n"
    return upstream + own
