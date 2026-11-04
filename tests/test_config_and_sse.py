"""Regression tests for the config merge and the SSE stream parser.

Run with::

    python -m tests.test_config_and_sse
"""
from __future__ import annotations

import argparse
import copy
import sys
import traceback

# Allow `python tests/test_config_and_sse.py` as well as `python -m tests.test_config_and_sse`: run by path,
# `tests/` is on sys.path but the repo root is not, so `from tests import ...`
# below would fail with ModuleNotFoundError before a single test ran.
import os as _os
import sys as _sys

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from tests import _runner, _stubs

_stubs.install()

from api_client.api_client import ApiClient  # noqa: E402


def _helpers():
    """Import ``utils.helpers``, or skip if ``pyyaml`` is missing.

    ``utils/helpers.py`` imports ``yaml`` at module scope, so a module-level
    import here aborted the entire suite -- including the four SSE tests, which do
    not need it at all -- on a machine without pyyaml.
    """
    _runner.require_module("yaml", why="Install it with: pip install pyyaml")
    return _runner.require_module("utils.helpers")


# --------------------------------------------------------------------------- #
#  config merge                                                               #
# --------------------------------------------------------------------------- #
def _cli(*argv):
    p = argparse.ArgumentParser()
    p.add_argument("--request-mode", choices=["chunk", "stream"])
    return p.parse_args(list(argv))


def test_request_mode_cli_override_reaches_generation_params():
    """`--request-mode stream` used to be written to a top-level key that the
    sampler never read, so the run silently stayed in chunk mode."""
    helpers = _helpers()
    cfg = helpers.merge_configs(
        helpers.load_config("config-example.yaml"), _cli("--request-mode", "stream")
    )
    assert cfg["generation_params"]["request_mode"] == "stream", cfg["generation_params"]


def test_request_mode_legacy_top_level_is_migrated():
    helpers = _helpers()
    base = helpers.load_config("config-example.yaml")
    base.pop("generation_params", None)
    base["request_mode"] = "stream"
    cfg = helpers.merge_configs(base, _cli())
    assert cfg["generation_params"]["request_mode"] == "stream"
    assert "request_mode" not in cfg, "stray top-level key left behind"


def test_merge_does_not_mutate_the_caller_config():
    helpers = _helpers()
    base = helpers.load_config("config-example.yaml")
    base["generation_params"]["request_mode"] = "chunk"
    snapshot = copy.deepcopy(base)
    helpers.merge_configs(base, _cli("--request-mode", "stream"))
    assert base == snapshot, "merge_configs mutated its input"


# --------------------------------------------------------------------------- #
#  SSE parsing                                                                #
# --------------------------------------------------------------------------- #
class _FakeResponse:
    """Minimal stand-in for a streamed ``requests`` response."""

    def __init__(self, raw: bytes):
        self._raw = raw

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self._raw), chunk_size):
            yield self._raw[i:i + chunk_size]


def test_sse_parsing_handles_utf8_split_across_packets():
    """A multi-byte character split over two TCP packets used to be mangled into
    U+FFFD, fail JSON parsing, and get silently skipped — dropping tokens."""
    text = "héllo wörld"
    line = ('data: {"choices":[{"text":"%s","logprobs":{"tokens":["%s"],'
            '"top_logprobs":[{"%s":0.0}]},"finish_reason":null}]}\n\n'
            % (text, text, text)).encode("utf-8")
    # split in the middle of the two-byte 'é'
    idx = line.index("é".encode("utf-8")) + 1
    resp = _FakeResponse(line[:idx] + line[idx:])

    payloads = list(ApiClient._iter_sse_data(resp))
    assert len(payloads) == 1, payloads
    assert "héllo wörld" in payloads[0], payloads[0]


def test_sse_parsing_accepts_data_without_space():
    resp = _FakeResponse(b'data:{"choices":[]}\n\ndata: [DONE]\n\n')
    assert list(ApiClient._iter_sse_data(resp)) == ['{"choices":[]}', "[DONE]"]


def test_sse_parsing_ignores_comments_and_joins_multiline():
    resp = _FakeResponse(b': keep-alive\n\ndata: line1\ndata: line2\n\n')
    assert list(ApiClient._iter_sse_data(resp)) == ["line1\nline2"]


def main() -> int:
    return _runner.run_suite("test_config_and_sse", globals())


if __name__ == "__main__":
    sys.exit(main())
