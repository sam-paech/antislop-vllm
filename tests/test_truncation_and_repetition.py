"""Regression tests for the two ways a record gets cut off mid-turn.

Both symptoms come from the same production dataset:

1.  **Truncation.** Assistant turns end mid-word (``...the terror``,
    ``...she whim``).  Every one of those records has ``finish_reason='length'``,
    because the sampler's *cumulative* attempt counter hit
    ``max_stream_attempts``.  Two unrelated things consume that counter: a
    premature end-of-turn that needs a restart, and a back-track that needs a
    restart.  A response that back-tracks more than a dozen times — routine
    against the shipped slop list — therefore ended as if it had run out of
    tokens, mid-sentence, with text still to come.

2.  **Repetition.** One turn contains ``of course.`` 327 times.  After a backend
    ``stop`` the sampler rejects the stop (it looks premature) and asks for a
    continuation; the model has nothing left to say, so it repeats a stock phrase.
    Because the rejected stop is reported with ``produced_tokens=True``, the
    no-progress strike counter is reset on every attempt, so the loop runs until
    the attempt budget is gone and the repetition is emitted verbatim.  The same
    reset also means the ``max_consecutive_stops`` safety valve documented in
    README can never fire on a backend that always emits *something*.

Both are bounded-work bugs, not hangs, so each test also asserts that the run
terminates and that the attempt budget is still respected.

Run with::

    python -m tests.test_truncation_and_repetition
"""
from __future__ import annotations

import sys
import traceback

from tests import _runner, _stubs

_stubs.install()

from core.models import ApiChunkResult  # noqa: E402
from core.sampler import ApiAntiSlopSampler  # noqa: E402
from validators.regex_validator import RegexValidator  # noqa: E402
from validators.slop_phrase_validator import SlopPhraseValidator  # noqa: E402


# --------------------------------------------------------------------------- #
#  Fake backend                                                               #
# --------------------------------------------------------------------------- #
class FakeStreamClient:
    """Scripted ``generate_stream``; one script entry per attempt."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = []
        self.suppress_tokens_unsupported = False

    def new_stream_token(self):
        from api_client.api_client import StreamCancelToken
        return StreamCancelToken()

    def cancel_stream(self, token):
        if token is not None:
            token.cancel()

    def cancel_current_stream(self):
        pass

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
    alternatives = alternatives or {}
    out = []
    for text in pieces:
        alts = alternatives.get(text, [(text, 0.0)])
        out.append(ApiChunkResult(text, [text], {0: alts}, None))
    out.append(ApiChunkResult("", [], {}, finish_reason))
    return out


def _make_sampler(client, validators=(), **gen_overrides):
    gen = {
        "chunk_size": 20,
        "top_logprobs_count": 6,
        "max_new_tokens": 400,
        "temperature": 1.0,
        "top_p": None,
        "top_k": None,
        "min_p": None,
        "timeout": 30,
        "stop_sequences": [],
        "request_mode": "stream",
        "max_consecutive_stops": 3,
        "max_stream_attempts": 12,
        "probe_stream_logprobs": False,
    }
    gen.update(gen_overrides)
    config = {
        "generation_params": gen,
        "backtracking": {"force_backtrack": True, "max_retries_per_position": 100},
        "filter_tail_banned_prefix_tokens": False,
        "regex_max_span_tokens": 0,
        # Suites that need a violation to survive until the post-generation sweep
        # set this high so the live regex pass never sees it.
        "regex_validation_interval": 10 ** 9,
    }
    # These keys live at the top level of the config, not under generation_params.
    for key in ("regex_validation_interval", "regex_max_span_tokens"):
        if key in gen_overrides:
            config[key] = gen_overrides.pop(key)
    return ApiAntiSlopSampler(
        api_client=client,
        validators=list(validators),
        config=config,
        tiktoken_model_name_for_counting=None,
        chat_template_formatter=None,
    )


SLOP_WORDS = [
    "gripping", "rhythmic", "shaky", "stale", "scent", "gasps", "tightening",
    "fragile", "drifting", "shaking", "jagged", "jarring", "vibrating",
    "terrified", "trembling", "spine", "fabric", "hum", "ragged", "brushing",
    "jolt", "shivers",
]


def _slop_validator():
    return SlopPhraseValidator({w: 1.0 for w in SLOP_WORDS}, app_config=None)


def _slop_validator_for(words):
    return SlopPhraseValidator({w: 1.0 for w in words}, app_config=None)


# --------------------------------------------------------------------------- #
#  1. Truncation: back-tracks must not spend the restart budget              #
# --------------------------------------------------------------------------- #
def test_backtracks_do_not_exhaust_the_restart_budget():
    """A response that back-tracks many times must still finish.

    Each attempt here produces one more slop word plus a clean replacement, so
    every attempt makes real progress.  On the old code the twelfth back-track
    tripped ``max_stream_attempts`` and the generation was cut off with
    ``finish_reason='length'`` even though the backend had more to say — the
    ``...she whim`` / ``...the terror`` truncations in the dataset.

    A back-track is *progress by definition*: it rewrote a token and resumed
    generation.  Only attempts that add nothing new may consume the budget.
    """
    scripts = []
    for i in range(30):
        word = SLOP_WORDS[i % len(SLOP_WORDS)]
        scripts.append(
            _chunks(
                [f" filler{i}", f" {word}", f" ok{i}."],
                "stop",
                {f" {word}": [(f" {word}", -0.01), (f" better{i}", -1.0)]},
            )
        )
    scripts.append(_chunks([" final words here."], "stop"))

    client = FakeStreamClient(scripts)
    sampler = _make_sampler(client, [_slop_validator()])
    text = "".join(sampler.generate("PROMPT"))

    # The run went well past the twelve back-tracks the old budget allowed.
    assert len(sampler.events) > 12, len(sampler.events)
    # ...and it did not stop for want of attempts.
    assert sampler.finish_reason != "length", sampler.finish_reason
    # The last thing the backend said actually reached the output.
    assert "final words here." in text, text
    # Nothing banned survived.
    for word in SLOP_WORDS:
        assert word not in text.lower(), (word, text)


def test_restart_budget_still_bounds_a_backend_that_never_ends():
    """The budget must still bound a stream that never terminates.

    Relaxing the budget for back-tracks must not remove the guarantee that a
    backend which keeps returning text forever is eventually stopped.
    """
    # Each attempt ends mid-sentence, so no stop is ever accepted as clean and the
    # only thing that can end the run is the attempt budget.
    client = FakeStreamClient([_chunks([f" more{i}"], "stop") for i in range(200)])
    sampler = _make_sampler(client, max_stream_attempts=6)
    text = "".join(sampler.generate("PROMPT"))

    assert len(client.calls) <= 6, len(client.calls)
    assert sampler.finish_reason == "length", sampler.finish_reason
    assert text  # it produced what it could


def test_stop_on_clean_end_is_not_a_progress_stop():
    """A clean stop that is *rejected* must not count as progress.

    ``stop_on_clean_end: false`` keeps generating past every backend stop.  The
    model has nothing left to add, so each attempt is a no-progress attempt and
    the safety valve must end the run — otherwise the tail of the response is
    whatever the model repeats to fill the space.
    """
    client = FakeStreamClient(
        [_chunks(["A complete sentence."], "stop")]
        + [_chunks([" of course."], "stop") for _ in range(40)]
    )
    sampler = _make_sampler(client, stop_on_clean_end=False, max_consecutive_stops=3)
    text = "".join(sampler.generate("PROMPT"))

    # The valve fired instead of running to the attempt cap.
    assert len(client.calls) <= 5, len(client.calls)
    assert text.count("of course.") <= 2, text


# --------------------------------------------------------------------------- #
#  2. Repetition: a rejected stop whose continuation only repeats itself      #
# --------------------------------------------------------------------------- #
def test_rejected_stop_that_only_repeats_itself_ends_the_turn():
    """The ``of course.`` x327 turn, reproduced.

    The first attempt ends on a clean boundary.  The sampler rejects the stop (the
    backend may have stopped early), asks for a continuation, and the model answers
    with the same stock phrase every time.  That is a *no-progress* continuation:
    it must not be emitted, and it must not reset the strike counter.
    """
    client = FakeStreamClient(
        [_chunks([" The room was cold and the door was locked."], "stop")]
        + [_chunks([" of course."], "stop") for _ in range(40)]
    )
    sampler = _make_sampler(client, stop_on_clean_end=False, max_consecutive_stops=3)
    text = "".join(sampler.generate("PROMPT"))

    # The *first* repetition is indistinguishable from a continuation, so one is
    # allowed through.  Every later one is recognised and the turn ends there.
    assert text.count("of course.") <= 2, (text.count("of course."), text[:400])
    assert len(client.calls) <= 5, len(client.calls)
    assert "The room was cold and the door was locked." in text, text


def test_repeated_continuation_is_dropped_from_the_output():
    """Text that only repeats the previous attempt must not be emitted at all."""
    client = FakeStreamClient(
        [_chunks([" She waited by the door."], "stop")]
        + [_chunks([" She waited by the door."], "stop") for _ in range(40)]
    )
    sampler = _make_sampler(client, stop_on_clean_end=False, max_consecutive_stops=3)
    text = "".join(sampler.generate("PROMPT"))

    assert text.count("She waited by the door.") == 1, text


def test_genuine_continuation_is_still_emitted():
    """A rejected stop whose continuation is new text is normal and must survive.

    This is the behaviour the early-EOS recovery exists for: the backend stopped
    mid-sentence and the rest of the sentence is still to come.
    """
    client = FakeStreamClient([
        _chunks(["The quick brown fox"], "stop"),
        _chunks([" jumps over the lazy dog."], "stop"),
    ])
    sampler = _make_sampler(client)
    text = "".join(sampler.generate("PROMPT"))

    assert "jumps over the lazy dog." in text, text
    assert len(client.calls) == 2, len(client.calls)


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


def test_chunk_mode_repeated_continuation_ends_the_turn():
    """Chunk mode has the same repetition loop as stream mode.

    ``request_mode: chunk`` re-sends the prompt for every chunk, so a refused stop
    costs a full round-trip and the model answers with the same stock phrase each
    time.  The guard has to exist in both loops; this covers the chunk one.
    """
    client = FakeChunkClient(
        [_chunk("A complete sentence.", "stop")]
        + [_chunk(" of course.", "stop") for _ in range(40)]
    )
    sampler = _make_sampler(
        client, request_mode="chunk", stop_on_clean_end=False,
        max_consecutive_stops=3,
    )
    text = "".join(sampler.generate("PROMPT"))

    assert text.count("of course.") <= 2, (text.count("of course."), text[:400])
    assert len(client.calls) <= 5, len(client.calls)
    assert "A complete sentence." in text, text


def test_tool_call_marker_is_scrubbed_from_the_text():
    """A leaked ``<tool_call|>`` must not reach the dataset.

    The turn in the dataset ends ``...She has a distinctive dyed-<tool_call|>``:
    the marker is a chat-template control token, not prose, and it survived because
    the default scrubber pattern did not cover the tool-call family.
    """
    client = FakeStreamClient([
        _chunks(["She has distinctive dyed-", "<tool_call|>", " hair."], "stop"),
    ])
    sampler = _make_sampler(client)
    text = "".join(sampler.generate("PROMPT"))

    assert "<tool_call|>" not in text, text
    assert "<tool_call>" not in text, text
    assert "dyed-" in text and "hair." in text, text


# --------------------------------------------------------------------------- #
#  3. The final sweep must not throw away the tail it was cleaning            #
# --------------------------------------------------------------------------- #
def test_final_sweep_keeps_the_text_after_a_repaired_violation():
    """A violation fixed after generation stops must not delete the rest of the turn.

    This is the third way a record gets cut off, and it looks different from the
    other two: the generation itself finished normally (``finish_reason='stop'``),
    the response is complete, and the text simply stops early.

    ``_perform_backtrack`` rewrites a token by calling ``state.truncate(idx + 1)``
    and then ``replace_token_string(idx, ...)``.  When it runs from the *final*
    sweep that is correct only if nothing follows ``idx``.  Mid-generation it is
    fine, because the backend is asked to regenerate everything after the rewrite.
    In the sweep the backend is never called again, so every token after ``idx`` is
    dropped from the state with nothing to replace it -- the turn ends at the
    repaired word.

    Against a 3 506-entry ban list this fires constantly: the sweep exists
    precisely because a ban at the very end of the text is only definitive once
    generation has stopped, so the last few tokens of a slop-heavy turn are
    routinely repaired.  The tail is whatever the model wrote after the last
    banned phrase.
    """
    pieces = [
        " She", " felt", " pale", " and", " cold", " and", " tired",
        " and", " alone", ".",
    ]
    # The banned span has a viable alternative, so the back-track succeeds.
    alternatives = {" pale": [(" pale", -0.01), (" happy", -1.0)]}
    client = FakeStreamClient([_chunks(pieces, "stop", alternatives)])
    sampler = _make_sampler(
        client,
        [RegexValidator([r"pale and"])],
        # The live pass is skipped so the violation is first seen by the sweep --
        # which is the normal case, because the sweep exists for matches that only
        # become definitive once the text stops growing.
        regex_validation_interval=10 ** 9,
        # Hold the whole response back until generation ends, exactly as the
        # shipped config does (``regex_max_span_tokens: 999999`` => tail_keep_tokens
        # is every token).  With the default of 0 the text is emitted as it is
        # generated, so a sweep repair can no longer be observed at all -- which is
        # why this bug survived a suite that already covered the sweep.
        regex_max_span_tokens=10 ** 6,
    )
    text = "".join(sampler.generate("PROMPT"))

    # The repair happened ...
    assert sampler.events, "the banned phrase was never reported"
    assert sampler.events[-1]["fixed"] is True, sampler.events
    # ... and the rest of the sentence survived it.
    assert "cold and tired and alone." in text, (
        f"final sweep truncated the turn at the repaired token: {text!r}"
    )
    assert "pale and" not in text.lower(), text


def test_final_sweep_keeps_the_tail_when_several_violations_are_repaired():
    """The sweep runs repeatedly; each pass must preserve what follows its repair.

    Fixing one violation can expose another earlier in the rewritten tail, so the
    sweep loops.  On the old code the *first* pass already destroyed everything
    after its own repair, which is why a slop-heavy turn lost more text the more
    violations it had.
    """
    pieces = [
        " She", " felt", " pale", " and", " stale", " and", " tired",
        " and", " utterly", " alone", ".",
    ]
    alternatives = {
        " pale": [(" pale", -0.01), (" warm", -1.0)],
        " stale": [(" stale", -0.01), (" quiet", -1.0)],
    }
    client = FakeStreamClient([_chunks(pieces, "stop", alternatives)])
    sampler = _make_sampler(
        client,
        [RegexValidator([r"pale and", r"stale and"])],
        regex_validation_interval=10 ** 9,
        regex_max_span_tokens=10 ** 6,
    )
    text = "".join(sampler.generate("PROMPT"))

    assert len([e for e in sampler.events if e["fixed"]]) >= 1, sampler.events
    assert "and utterly alone." in text, (
        f"final sweep truncated the turn at a repaired token: {text!r}"
    )
    assert "pale and" not in text.lower(), text
    assert "stale and" not in text.lower(), text


def test_live_backtrack_still_discards_the_invalidated_tail():
    """The sweep fix must not change what a *live* back-track does.

    Mid-generation the tail after a rewrite is invalidated on purpose: the sampler
    asks the backend to regenerate it, and the continuation arrives as new text.  If
    a live back-track stopped truncating, the stale tail would be kept *and* the
    regenerated text appended, duplicating the passage.

    This is the counterpart to the two sweep tests above.  Making the sweep keep its
    tail by removing ``truncate`` altogether would pass those two and break this one.
    """
    # Attempt 1: a banned span early in the text, then more text, then a mid-sentence
    # stop so the sampler asks for a continuation.
    first = _chunks(
        [" She", " felt", " pale", " and", " cold"],
        "stop",
        {" pale": [(" pale", -0.01), (" happy", -1.0)]},
    )
    # Attempt 2: the backend regenerates from the rewritten token onwards.
    second = _chunks([" and cold and tired."], "stop")
    client = FakeStreamClient([first, second])
    sampler = _make_sampler(
        client,
        [_slop_validator_for(["pale"])],
        stop_on_clean_end=False,
        regex_max_span_tokens=10 ** 6,
    )
    text = "".join(sampler.generate("PROMPT"))

    # The live back-track happened before the stop was accepted.
    assert sampler.events and sampler.events[0]["fixed"] is True, sampler.events
    # The stale " and cold" that followed the rewrite in attempt 1 is gone, so the
    # passage appears exactly once, from the regenerated continuation.  Assert on the
    # *whole* result, not just on a substring count: keeping the stale tail and then
    # appending the regenerated text yields "... happy and and cold ...", which still
    # contains exactly one "and cold".
    assert text == " She felt happy and cold and tired.", (
        f"live back-track kept the invalidated tail and duplicated it: {text!r}"
    )


def main() -> int:
    return _runner.run_suite("test_truncation_and_repetition", globals())


if __name__ == "__main__":
    sys.exit(main())
