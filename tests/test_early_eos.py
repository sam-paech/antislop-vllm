"""Regression tests for the early-end-of-turn / truncation bugs.

Run with::

    python -m tests.test_early_eos

No pytest, no GPU, no network: the backend is a scripted fake and the heavy
optional imports are stubbed (see ``tests/_stubs.py``).
"""
from __future__ import annotations

import sys
import traceback

from tests import _platform, _stubs

_stubs.install()

from core.models import ApiChunkResult  # noqa: E402
from core.sampler import ApiAntiSlopSampler  # noqa: E402


# --------------------------------------------------------------------------- #
#  Fake backend                                                               #
# --------------------------------------------------------------------------- #
class FakeStreamClient:
    """Scripted ``generate_stream``.

    Each entry in *scripts* is the list of chunks one *attempt* yields.  An entry
    may also be an exception instance, which is raised mid-attempt.
    """

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = []
        self.suppress_tokens_unsupported = False

    def generate_stream(self, **kwargs):
        self.calls.append(kwargs)
        script = self.scripts.pop(0) if self.scripts else []
        for item in script:
            if isinstance(item, Exception):
                raise item
            yield item

    # used by the sampler when a validator fires
    def cancel_current_stream(self):
        pass

    def reset_connection_pool(self):
        pass


def _chunks(pieces, finish_reason="stop"):
    """Build chunks for one attempt: text tokens then a terminal chunk."""
    out = []
    for text in pieces:
        out.append(
            ApiChunkResult(
                generated_text=text,
                token_strings=[text],
                logprobs={0: [(text, 0.0)]},
                finish_reason=None,
            )
        )
    out.append(
        ApiChunkResult(
            generated_text="",
            token_strings=[],
            logprobs={},
            finish_reason=finish_reason,
        )
    )
    return out


def _make_sampler(client, **gen_overrides):
    gen = {
        "chunk_size": 20,
        "top_logprobs_count": 1,
        "max_new_tokens": 200,
        "temperature": 1.0,
        "top_p": None,
        "top_k": None,
        "min_p": None,
        "timeout": 30,
        "stop_sequences": [],
        "request_mode": "stream",
        "max_consecutive_stops": 3,
        "max_stream_attempts": 6,
        # The probe is a real backend round-trip; the fake client scripts its
        # stream attempts instead.
        "probe_stream_logprobs": False,
    }
    gen.update(gen_overrides)
    config = {
        "generation_params": gen,
        "backtracking": {"force_backtrack": False},
        "filter_tail_banned_prefix_tokens": False,
        "regex_max_span_tokens": 4,
    }
    return ApiAntiSlopSampler(
        api_client=client,
        validators=[],
        config=config,
        tiktoken_model_name_for_counting=None,
        chat_template_formatter=None,
    )


def _run(sampler, prompt="PROMPT"):
    return "".join(sampler.generate(prompt))


# --------------------------------------------------------------------------- #
#  Tests                                                                      #
# --------------------------------------------------------------------------- #
def test_premature_stop_without_marker_continues():
    """The reported bug: a mid-sentence `stop` with NO marker token truncated
    the answer.  It must now continue and produce the full sentence."""
    client = FakeStreamClient([
        _chunks(["The quick brown fox"], finish_reason="stop"),
        _chunks([" jumps over the lazy dog."], finish_reason="stop"),
    ])
    sampler = _make_sampler(client)
    text = _run(sampler)
    assert "jumps over the lazy dog." in text, text
    assert len(client.calls) >= 2, "sampler accepted the premature stop"


def test_clean_end_is_terminal():
    client = FakeStreamClient([
        _chunks(["A complete sentence."], finish_reason="stop"),
        _chunks([" SHOULD NOT APPEAR"], finish_reason="stop"),
    ])
    sampler = _make_sampler(client)
    text = _run(sampler)
    assert "SHOULD NOT APPEAR" not in text, text
    assert len(client.calls) == 1, client.calls


def test_stop_sequence_is_terminal():
    client = FakeStreamClient([
        _chunks(["partial and then END"], finish_reason="stop"),
        _chunks([" SHOULD NOT APPEAR"], finish_reason="stop"),
    ])
    sampler = _make_sampler(client, stop_sequences=["END"])
    text = _run(sampler)
    assert "SHOULD NOT APPEAR" not in text, text
    assert len(client.calls) == 1


def test_strike_counter_resets_on_progress():
    """Three premature stops separated by real progress must NOT truncate."""
    client = FakeStreamClient([
        _chunks(["First part"], finish_reason="stop"),
        _chunks([" second part"], finish_reason="stop"),
        _chunks([" third part"], finish_reason="stop"),
        _chunks([" fourth part done."], finish_reason="stop"),
    ])
    sampler = _make_sampler(client)
    text = _run(sampler)
    assert "fourth part done." in text, text


def test_no_progress_guard_terminates():
    """A backend that always returns nothing must not loop forever."""
    client = FakeStreamClient([[] for _ in range(50)])
    sampler = _make_sampler(client, max_stream_attempts=4)
    text = _run(sampler)
    assert text == ""
    assert len(client.calls) <= 8, len(client.calls)


def test_marker_split_across_chunks_is_removed():
    client = FakeStreamClient([
        _chunks(["Some text <tur", "n|> more text here."], finish_reason="stop"),
    ])
    sampler = _make_sampler(client)
    text = _run(sampler)
    assert "<tur" not in text and "n|>" not in text and "<turn|>" not in text, text
    assert "Some text" in text and "more text here." in text, text


def test_marker_mid_chunk_is_removed():
    client = FakeStreamClient([
        _chunks(["foo ", "<turn|>", "bar baz qux quux."], finish_reason="stop"),
    ])
    sampler = _make_sampler(client)
    text = _run(sampler)
    assert "<turn|>" not in text, text
    assert "foo " in text and "bar baz qux quux." in text, text


def test_unknown_special_token_is_removed():
    client = FakeStreamClient([
        _chunks(["<start_of_turn>user hello there friend<|im_end|>"], finish_reason="stop"),
    ])
    sampler = _make_sampler(client)
    text = _run(sampler)
    assert "<start_of_turn>" not in text, text
    assert "<|im_end|>" not in text, text
    assert "hello there friend" in text, text


def test_stream_error_is_retried_not_truncated():
    client = FakeStreamClient([
        _chunks(["Half a sent"]) + [RuntimeError("connection reset")],
        _chunks(["ence that finishes here."], finish_reason="stop"),
    ])
    sampler = _make_sampler(client)
    text = _run(sampler)
    assert "Half a sent" in text, text
    assert "ence that finishes here." in text, text
    assert len(client.calls) >= 2


def test_finish_reason_is_exposed():
    client = FakeStreamClient([_chunks(["Done here."], finish_reason="stop")])
    sampler = _make_sampler(client)
    _run(sampler)
    assert sampler.finish_reason == "stop"


def test_suppress_tokens_sent_on_restart_only():
    client = FakeStreamClient([
        _chunks(["First bit"], finish_reason="stop"),
        _chunks([" and the rest."], finish_reason="stop"),
    ])
    sampler = _make_sampler(client)
    _run(sampler)
    assert "suppress_tokens" not in client.calls[0]
    assert client.calls[1].get("suppress_tokens"), client.calls[1]


# --------------------------------------------------------------------------- #
#  Chunk mode                                                               #
# --------------------------------------------------------------------------- #
class FakeChunkClient:
    """Scripted ``generate_chunk``."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.calls = []

    def generate_chunk(self, **kwargs):
        self.calls.append(kwargs)
        if self.chunks:
            return self.chunks.pop(0)
        return ApiChunkResult("", [], {}, "stop")

    def cancel_current_stream(self):
        pass

    def reset_connection_pool(self):
        pass


def _chunk(text, reason=None):
    if not text:
        return ApiChunkResult("", [], {}, reason)
    return ApiChunkResult(text, [text], {0: [(text, 0.0)]}, reason)


def test_chunk_mode_premature_stop_continues():
    client = FakeChunkClient([_chunk("The quick brown fox"), _chunk(" jumps over the dog.")])
    sampler = _make_sampler(client, request_mode="chunk")
    text = _run(sampler)
    assert "jumps over the dog." in text, text


def test_chunk_mode_empty_chunk_is_retried():
    """An empty chunk with no finish_reason used to end the generation."""
    client = FakeChunkClient([_chunk("Hello there"), _chunk(""), _chunk(" friend.")])
    sampler = _make_sampler(client, request_mode="chunk")
    text = _run(sampler)
    assert "friend." in text, text


def test_chunk_mode_marker_is_removed():
    client = FakeChunkClient([_chunk("Some text <turn|> more words here")])
    sampler = _make_sampler(client, request_mode="chunk")
    text = _run(sampler)
    assert "<turn|>" not in text, text
    assert "Some text" in text


def test_chunk_mode_terminates_on_endless_empty_chunks():
    """A backend that only ever returns empty chunks must not loop forever."""
    client = FakeChunkClient([_chunk("") for _ in range(200)])
    sampler = _make_sampler(client, request_mode="chunk")
    text = _run(sampler)
    assert text == ""
    assert len(client.calls) <= 6, len(client.calls)


def main() -> int:
    if _platform.report_skip("test_early_eos"):
        return 0
    return _platform.run_suite("test_early_eos", globals())


if __name__ == "__main__":
    sys.exit(main())
