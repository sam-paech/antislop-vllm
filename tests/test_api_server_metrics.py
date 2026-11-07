"""Regression tests for the API server's Prometheus ``/metrics`` endpoint.

The dataset generator's 📊 Metrics tab polls ``<host>:<port>/metrics`` for every API slot.
A plain vLLM serves that; this proxy did not, so every slot pointed at the proxy was reported
as *unreachable* and no figure at all appeared.  These tests pin the two halves of the fix:

1. **The proxy passes the upstream vLLM body through verbatim**, so the generator's existing
   vLLM parsing keeps working unchanged (backwards compatibility).
2. **The proxy adds its own ``antislop:`` metrics** -- requests, tokens, ban events, upstream
   errors -- to the same body, so the proxy itself can be seen.  A dead upstream must not blank
   the endpoint: the antislop figures are still served, with a counter saying the scrape failed.

Run with::

    python -m tests.test_api_server_metrics
"""
from __future__ import annotations

import logging
import sys

# Allow `python tests/test_api_server_metrics.py` as well as `python -m tests...`: run by
# path, `tests/` is on sys.path but the repo root is not.
import os as _os
import sys as _sys

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from tests import _runner, _stubs

_stubs.install()

from api_client.api_client import StreamCancelToken  # noqa: E402
from core.models import ApiChunkResult  # noqa: E402

import api_server.metrics as metrics_mod  # noqa: E402


# --------------------------------------------------------------------------- #
#  Fakes                                                                      #
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code


class _StubRequests:
    """Stand-in for the ``requests`` module inside ``api_server.metrics``."""

    def __init__(self, text: str = "", status_code: int = 200, error=None) -> None:
        self.text = text
        self.status_code = status_code
        self.error = error
        self.calls: list[str] = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if self.error is not None:
            raise self.error
        return _FakeResponse(self.text, self.status_code)


class _ScriptedClient:
    """A backend that yields one scripted token list, then a finish reason."""

    def __init__(self, tokens, reason="stop") -> None:
        self.tokens = list(tokens)
        self.reason = reason
        self.calls = []

    def new_stream_token(self):
        return StreamCancelToken()

    def cancel_stream(self, token):
        if token is not None:
            token.cancel()

    def cancel_current_stream(self):
        pass

    def reset_connection_pool(self):
        pass

    def generate_stream(self, **kw):
        self.calls.append(kw)
        for token in self.tokens:
            yield ApiChunkResult(token, [token], {0: [(token, 0.0)]}, None)
        yield ApiChunkResult("", [], {}, self.reason)


class _Fmt:
    model_id = "fake"
    tokenizer = None

    def build_base_prompt_from_messages(self, messages, **kw):
        return "|".join(m["content"] for m in messages)


UPSTREAM_BODY = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="acme/model"} 3.0
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="acme/model"} 500.0
"""


def _reset():
    metrics_mod.SERVER_METRICS.reset()


def _with_stub(stub):
    class _Ctx:
        def __enter__(self):
            self._saved = metrics_mod.requests
            metrics_mod.requests = stub
            return stub

        def __exit__(self, *exc):
            metrics_mod.requests = self._saved
            return False

    return _Ctx()


def _server_with(client, formatter=None, api_base_url="http://vllm:8000/v1"):
    """Build the real server module with shared resources, like the audit suite does."""
    server = _runner.require_module(
        "api_server.server",
        why="The API-server tests need `fastapi` (pip install fastapi).",
    )
    server.setup_shared_resources(
        {
            "generation_params": {
                "request_mode": "stream",
                "probe_stream_logprobs": False,
                "threads": 4,
                "max_new_tokens": 40,
                "api_retry_delay_sec": 0,
            },
            "threads": 1,
            "filter_tail_banned_prefix_tokens": False,
            "api_base_url": api_base_url,
        },
        [],
        client,
        formatter if formatter is not None else _Fmt(),
        logging.getLogger("t"),
    )
    return server


# --------------------------------------------------------------------------- #
#  1. The exposition format                                                  #
# --------------------------------------------------------------------------- #
def test_a_counter_renders_help_type_and_a_sample():
    _reset()
    metrics_mod.SERVER_METRICS.counter("antislop:thing_total", "A thing.").inc()
    text = metrics_mod.SERVER_METRICS.render()
    assert "# HELP antislop:thing_total A thing." in text, text
    assert "# TYPE antislop:thing_total counter" in text, text
    assert "antislop:thing_total 1.0" in text, text


def test_a_labelled_counter_renders_one_sample_per_label_set():
    _reset()
    counter = metrics_mod.SERVER_METRICS.counter(
        "antislop:req_total", "Requests.", ("endpoint", "status"))
    counter.inc({"endpoint": "/v1/chat/completions", "status": "200"})
    counter.inc({"endpoint": "/v1/chat/completions", "status": "200"}, 2)
    counter.inc({"endpoint": "/v1/chat/completions", "status": "500"})
    text = metrics_mod.SERVER_METRICS.render()
    assert ('antislop:req_total{endpoint="/v1/chat/completions",status="200"} 3.0'
            in text), text
    assert ('antislop:req_total{endpoint="/v1/chat/completions",status="500"} 1.0'
            in text), text


def test_label_values_are_escaped():
    _reset()
    counter = metrics_mod.SERVER_METRICS.counter("antislop:esc_total", "Esc.", ("why",))
    counter.inc({"why": 'a "quoted" \\ word\nnext'})
    text = metrics_mod.SERVER_METRICS.render()
    assert 'why="a \\"quoted\\" \\\\ word\\nnext"' in text, text


def test_a_gauge_can_go_up_and_down():
    _reset()
    gauge = metrics_mod.SERVER_METRICS.gauge("antislop:in_flight", "In flight.")
    gauge.inc()
    gauge.inc()
    gauge.dec()
    text = metrics_mod.SERVER_METRICS.render()
    assert "antislop:in_flight 1.0" in text, text


def test_a_summary_renders_sum_and_count():
    _reset()
    summary = metrics_mod.SERVER_METRICS.summary("antislop:duration_seconds", "Duration.")
    summary.observe(0.25)
    summary.observe(0.75)
    text = metrics_mod.SERVER_METRICS.render()
    assert "antislop:duration_seconds_sum 1.0" in text, text
    assert "antislop:duration_seconds_count 2.0" in text, text


def test_reset_clears_every_series():
    """``reset()`` clears the SERIES but keeps the definitions, so the tab's shape is stable."""
    _reset()
    metrics_mod.SERVER_METRICS.counter("antislop:gone_total", "Gone.").inc()
    assert "antislop:gone_total 1.0" in metrics_mod.SERVER_METRICS.render()
    _reset()
    text = metrics_mod.SERVER_METRICS.render()
    assert "antislop:gone_total 1.0" not in text, text
    # The HELP/TYPE lines stay: a metric that disappears and reappears between scrapes is
    # worse for a dashboard than one that is present with no samples.
    assert "# TYPE antislop:gone_total counter" in text, text


def test_an_empty_registry_renders_an_empty_body():
    """A fresh registry has no metrics, so it renders nothing (not even a blank line).

    Deliberately a NEW registry, not ``SERVER_METRICS``: the module-level one always carries the
    server's metric definitions, so it is never empty.
    """
    assert metrics_mod.MetricsRegistry().render().strip() == ""


# --------------------------------------------------------------------------- #
#  2. The upstream URL derivation                                            #
# --------------------------------------------------------------------------- #
def test_upstream_metrics_url_discards_the_api_path():
    """Same rule the generator uses: /metrics lives at the server root."""
    assert metrics_mod.upstream_metrics_url("http://vllm:8000/v1") == \
        "http://vllm:8000/metrics"
    assert metrics_mod.upstream_metrics_url("http://vllm:8000/v1/chat/completions") == \
        "http://vllm:8000/metrics"
    assert metrics_mod.upstream_metrics_url("http://vllm:8000") == \
        "http://vllm:8000/metrics"


def test_upstream_metrics_url_is_none_without_a_backend():
    assert metrics_mod.upstream_metrics_url("") is None
    assert metrics_mod.upstream_metrics_url(None) is None
    assert metrics_mod.upstream_metrics_url("not a url") is None


# --------------------------------------------------------------------------- #
#  3. The /metrics endpoint                                                  #
# --------------------------------------------------------------------------- #
def _metrics_body(server) -> str:
    """Call the async ``/metrics`` route and return its body as text.

    The route is ``async def`` (it awaits the blocking scrape in the threadpool), so it cannot be
    called synchronously.  Driving it with ``asyncio.run`` keeps the test free of an ASGI
    transport and of ``httpx``, which the harness does not depend on.
    """
    import asyncio

    response = asyncio.run(server.metrics())
    body = response.body
    return body.decode() if isinstance(body, bytes) else body


def test_metrics_endpoint_passes_the_upstream_body_through_verbatim():
    """The generator's vLLM parsing must keep working: the vllm: samples survive."""
    _reset()
    server = _server_with(_ScriptedClient([]))
    stub = _StubRequests(UPSTREAM_BODY)
    with _with_stub(stub):
        body = _metrics_body(server)
    assert "vllm:generation_tokens_total" in body, body
    assert "vllm:num_requests_running" in body, body
    assert stub.calls == ["http://vllm:8000/metrics"], stub.calls


def test_metrics_endpoint_appends_the_antislop_metrics():
    _reset()
    server = _server_with(_ScriptedClient([]))
    with _with_stub(_StubRequests(UPSTREAM_BODY)):
        body = _metrics_body(server)
    assert "antislop:requests_in_flight" in body, body
    assert "antislop:upstream_metrics_scrape_total" in body, body


def test_a_dead_upstream_still_serves_the_antislop_metrics():
    """A dead vLLM must not blank the endpoint: the proxy's own figures still arrive."""
    _reset()
    server = _server_with(_ScriptedClient([]))
    with _with_stub(_StubRequests(error=IOError("connection refused"))):
        body = _metrics_body(server)
    assert "antislop:requests_in_flight" in body, body
    assert ('antislop:upstream_metrics_scrape_total{result="error"} 1.0' in body), body
    assert "vllm:" not in body, body


def test_a_non_200_upstream_scrape_is_recorded_as_an_error():
    _reset()
    server = _server_with(_ScriptedClient([]))
    with _with_stub(_StubRequests("nope", status_code=404)):
        body = _metrics_body(server)
    assert ('antislop:upstream_metrics_scrape_total{result="error"} 1.0' in body), body


def test_a_successful_scrape_is_recorded_and_timestamped():
    _reset()
    server = _server_with(_ScriptedClient([]))
    with _with_stub(_StubRequests(UPSTREAM_BODY)):
        body = _metrics_body(server)
    assert ('antislop:upstream_metrics_scrape_total{result="ok"} 1.0' in body), body
    assert "antislop:upstream_metrics_last_scrape_timestamp_seconds" in body, body


def test_metrics_endpoint_is_a_get_route():
    """The generator polls it with a plain GET, so it must not be POST-only."""
    server = _runner.require_module(
        "api_server.server",
        why="The API-server tests need `fastapi` (pip install fastapi).",
    )
    routes = {(getattr(r, "path", None), tuple(sorted(getattr(r, "methods", ()) or ())))
              for r in server.app.routes}
    assert ("/metrics", ("GET",)) in routes, routes


# --------------------------------------------------------------------------- #
#  4. Instrumentation of a generation                                      #
# --------------------------------------------------------------------------- #
def test_a_generation_records_requests_tokens_and_finish_reason():
    _reset()
    client = _ScriptedClient(["Hello", " there"])
    server = _server_with(client)
    result = server._run_generation_for_request(
        {"messages": [{"role": "user", "content": "U"}]})
    assert "error" not in result, result
    body = metrics_mod.SERVER_METRICS.render()
    assert ('antislop:generations_total{finish_reason="stop"} 1.0' in body), body
    assert "antislop:prompt_tokens_total" in body, body
    assert "antislop:generation_tokens_total" in body, body
    assert "antislop:request_duration_seconds_count 1.0" in body, body
    # The in-flight gauge must be back to zero once the request is done.
    assert "antislop:requests_in_flight 0.0" in body, body


def test_a_backend_failure_is_counted_as_an_upstream_error():
    _reset()
    client = _ScriptedClient([], reason="error")
    server = _server_with(client)
    server._run_generation_for_request({"messages": [{"role": "user", "content": "U"}]})
    body = metrics_mod.SERVER_METRICS.render()
    assert "antislop:upstream_errors_total" in body, body


def test_ban_events_are_counted_by_validator_and_outcome():
    """The proxy's whole purpose is the ban/back-track loop, so it must be visible.

    The events are read off the *sampler* the server builds per request, not off the API client,
    so the sampler is stubbed here (``api_server.server.ApiAntiSlopSampler``) and the real server
    code path is exercised unchanged.
    """
    _reset()
    server = _server_with(_ScriptedClient(["Hello"]))

    class _StubSampler:
        def __init__(self, *args, **kwargs):
            self.finish_reason = "stop"
            self.stop_sequences = None
            self.extra_request_params = {}
            self.ban_strength = 1.0
            self.tiktoken_encoding = None
            self.events = [
                {"type": "slop_phrase", "index": 1, "details": {"phrase": "very"},
                 "original_token_string": " very", "fixed": True},
                {"type": "regex", "index": 2, "details": {"pattern": "not \\w+, but"},
                 "original_token_string": " not", "fixed": False},
            ]

        def generate(self, **kwargs):
            yield "Hello"

    saved = server.ApiAntiSlopSampler
    server.ApiAntiSlopSampler = _StubSampler
    try:
        result = server._run_generation_for_request(
            {"messages": [{"role": "user", "content": "U"}]})
    finally:
        server.ApiAntiSlopSampler = saved
    assert "error" not in result, result
    body = metrics_mod.SERVER_METRICS.render()
    assert ('antislop:ban_events_total{validator="slop_phrase",fixed="true"} 1.0'
            in body), body
    assert ('antislop:ban_events_total{validator="regex",fixed="false"} 1.0'
            in body), body


def test_importing_the_module_does_not_touch_the_network():
    """Importing must issue no request: the tests and the CLI import this module."""
    stub = _StubRequests(UPSTREAM_BODY)
    with _with_stub(stub):
        _runner.require_module("api_server.metrics")
    assert stub.calls == [], stub.calls


def main() -> int:
    return _runner.run_suite("test_api_server_metrics", globals())


if __name__ == "__main__":
    sys.exit(main())
