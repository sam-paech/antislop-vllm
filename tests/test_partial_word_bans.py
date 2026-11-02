"""Regression tests for the partial-word ban / backtracking storm.

Symptoms reproduced from a production log:

* ``Back-tracking @tok=2 orig=' la'`` repeating at the same token index, followed
  by a burst of ``GenerationState initialised`` lines (one per candidate token
  tested) and a ``Connection pool reset``.
* ``Stream closed without [DONE] or finish_reason`` on *other*, unrelated
  requests, because one request backtracked on a client shared by all of them.
* The banned phrase ``la`` firing when the model was still in the middle of a
  word such as ``land`` / ``last``.

Run with::

    python -m tests.test_partial_word_bans
"""
from __future__ import annotations

import sys
import traceback

from tests import _platform, _stubs

_stubs.install()

from core.models import ApiChunkResult  # noqa: E402
from core.sampler import ApiAntiSlopSampler  # noqa: E402
from state.generation_state import GenerationState  # noqa: E402
from utils.slop_helpers import detect_disallowed_sequence  # noqa: E402
from validators.slop_phrase_validator import SlopPhraseValidator  # noqa: E402


# --------------------------------------------------------------------------- #
#  helpers                                                                    #
# --------------------------------------------------------------------------- #
BANLIST = {"la": 1.0, "the soft glow": 1.0}


def _phrase_validator():
    return SlopPhraseValidator(dict(BANLIST), app_config=None)


def _state_with(text):
    st = GenerationState("PROMPT", log_init=False)
    st.generated_token_strings = [text]
    st.rebuild_decoded_buffer()
    return st


def _check(text, final):
    return _phrase_validator().check(_state_with(text), final=final)


# --------------------------------------------------------------------------- #
#  Validator level                                                            #
# --------------------------------------------------------------------------- #
def test_partial_word_is_not_banned_while_streaming():
    """``la`` is the first half of ``land``; it must not fire mid-stream."""
    assert _check("the old la", final=False) is None


def test_completed_banned_word_is_banned_while_streaming():
    """Once a word boundary exists the ban fires immediately."""
    vio = _check("the old la now", final=False)
    assert vio is not None and vio.details["phrase"] == "la"


def test_partial_word_is_banned_in_the_final_sweep():
    """At end of generation there is no next token, so the match is definitive."""
    vio = _check("the old la", final=True)
    assert vio is not None and vio.details["phrase"] == "la"


def test_banned_phrase_inside_a_word_still_ignored():
    """Word-boundary matching is unchanged: ``land`` never matches ``la``."""
    assert _check("the old land is far", final=True) is None


def test_detect_disallowed_sequence_defers_end_of_text_match():
    keys = {"la"}
    # Streaming: match ends at the end of the text -> provisional, deferred.
    assert detect_disallowed_sequence(
        "the old la", keys, 2, 2, check_n_chars_back=10,
        require_word_boundaries=True, allow_end_of_text_match=False,
    ) == (None, None)
    # Final pass: same text is now a real match.
    phrase, pos = detect_disallowed_sequence(
        "the old la", keys, 2, 2, check_n_chars_back=10,
        require_word_boundaries=True, allow_end_of_text_match=True,
    )
    assert phrase == "la" and pos == 8


def test_detect_disallowed_sequence_keeps_non_word_tail_immediate():
    """A ban that ends in punctuation needs no lookahead, so it fires at once."""
    phrase, pos = detect_disallowed_sequence(
        "cat[more", {"cat["}, 4, 4, check_n_chars_back=9,
        require_word_boundaries=True, allow_end_of_text_match=False,
    )
    assert phrase == "cat[" and pos == 0


# --------------------------------------------------------------------------- #
#  Sampler level                                                              #
# --------------------------------------------------------------------------- #
class FakeStreamClient:
    """Scripted stream client that honours the per-stream cancel token."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = []
        self.suppress_tokens_unsupported = False
        self.cancelled = 0

    def new_stream_token(self):
        from api_client.api_client import StreamCancelToken
        return StreamCancelToken()

    def cancel_stream(self, token):
        self.cancelled += 1
        if token is not None:
            token.cancel()

    def cancel_current_stream(self):
        # Deliberately *not* what the sampler should call any more.
        self.cancelled += 1

    def reset_connection_pool(self):
        pass

    def generate_stream(self, **kwargs):
        self.calls.append(kwargs)
        token = kwargs.get("cancel_token")
        script = self.scripts.pop(0) if self.scripts else []
        for item in script:
            if token is not None and token.cancelled:
                return
            if isinstance(item, Exception):
                raise item
            yield item


def _chunks(pieces, finish_reason="stop", alternatives=None):
    """Build one attempt: text tokens, then a terminal chunk.

    *alternatives* maps a piece's text to the top-logprob list the backend
    reported for it, so the back-tracker has something to resample from.  Without
    it every position has exactly one candidate (the banned token) and no
    replacement is possible.
    """
    alternatives = alternatives or {}
    out = []
    for text in pieces:
        alts = alternatives.get(text, [(text, 0.0)])
        out.append(ApiChunkResult(text, [text], {0: alts}, None))
    out.append(ApiChunkResult("", [], {}, finish_reason))
    return out


def _make_sampler(client, validators, **gen_overrides):
    gen = {
        "chunk_size": 20,
        "top_logprobs_count": 4,
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
        "probe_stream_logprobs": False,
    }
    gen.update(gen_overrides)
    config = {
        "generation_params": gen,
        "backtracking": {"force_backtrack": True, "max_retries_per_position": 5},
        "filter_tail_banned_prefix_tokens": False,
        "regex_max_span_tokens": 4,
    }
    return ApiAntiSlopSampler(
        api_client=client,
        validators=validators,
        config=config,
        tiktoken_model_name_for_counting=None,
        chat_template_formatter=None,
    )


def test_partial_word_does_not_trigger_backtracking():
    """``the old la`` must be allowed to grow into ``the old land``."""
    client = FakeStreamClient([
        _chunks(["The old la", "nd was quiet here."], finish_reason="stop"),
    ])
    sampler = _make_sampler(client, [_phrase_validator()])
    text = "".join(sampler.generate("PROMPT"))

    assert "The old land was quiet here." in text, text
    # No backtrack happened at all.
    assert not sampler.events, sampler.events


def test_banned_word_mid_stream_is_backtracked_once():
    """A real violation is fixed, and the loop does not spin on that index."""
    client = FakeStreamClient([
        # token 0 = "The old ", token 1 = "la", token 2 = " then more text."
        _chunks(
            ["The old ", "la", " then more text."],
            finish_reason="stop",
            alternatives={"la": [("la", -0.1), ("lake", -1.5), ("lane", -2.0)]},
        ),
    ])
    sampler = _make_sampler(client, [_phrase_validator()])
    text = "".join(sampler.generate("PROMPT"))

    assert "la then" not in text, text
    # Exactly one backtrack, at one position.  The old code re-entered the same
    # index indefinitely because the replacement was re-banned by the same
    # end-of-text match, producing the `Back-tracking @tok=2 orig=' la'` loop.
    assert len(sampler.events) == 1, sampler.events
    assert sampler.events[0]["index"] == 1
    assert sampler.events[0]["fixed"] is True


def test_cancellation_is_scoped_to_one_stream():
    """Backtracking must not cancel unrelated streams on a shared client."""
    client = FakeStreamClient([
        _chunks(["The old ", "la", " then more text."], finish_reason="stop"),
    ])
    sampler = _make_sampler(client, [_phrase_validator()])
    "".join(sampler.generate("PROMPT"))

    # Exactly one stream was cancelled, and only through the per-stream token.
    assert client.cancelled == 1, client.cancelled
    assert all("cancel_token" in call for call in client.calls), client.calls


def test_hypothetical_state_does_not_log(caplog_records=None):
    """Building candidate states must not emit a log line each."""
    import logging

    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Collect()
    logger = logging.getLogger("state.generation_state")
    logger.addHandler(handler)
    try:
        GenerationState.hypothetical("PROMPT", ["a", "b"])
    finally:
        logger.removeHandler(handler)
    assert not [r for r in records if "initialised" in r.getMessage()], records


def test_backtrack_budget_is_enforced():
    """An unfixable position must stop, not loop forever."""
    # Every candidate token in the logprobs is the banned one, so no replacement
    # can be found; the budget must end it.
    client = FakeStreamClient([
        _chunks(["The old ", "la", " and then a lot more text follows here."],
                finish_reason="stop"),
    ])
    sampler = _make_sampler(client, [_phrase_validator()])
    text = "".join(sampler.generate("PROMPT"))

    # The run terminated and produced text rather than hanging.
    assert isinstance(text, str)
    # The unfixable position was reported once, not in a loop.
    assert len([e for e in sampler.events if e["index"] == 1]) == 1, sampler.events


def main() -> int:
    if _platform.report_skip("test_partial_word_bans"):
        return 0
    return _platform.run_suite("test_partial_word_bans", globals())


if __name__ == "__main__":
    sys.exit(main())
