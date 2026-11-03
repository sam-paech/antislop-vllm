"""Regression tests for the 2026-11-03 audit (dev @ aba6cfb).

Each test fails on aba6cfb.  The dataset symptoms they trace back to:

* repetition loops and long tails: a refused stop banned `<turn|>` for the rest
  of the generation, and roleplay endings (`*action*`, `~`, emoji) were refused;
* `word.word`-style joins and garbled tails: the final sweep spliced a
  replacement token in front of a tail sampled after the old token;
* turns that continue as a new turn: a back-track could pick `<turn|>` itself;
* B != A with the same client settings: the proxy dropped repetition_penalty.

Run with::

    python -m tests.test_audit_fixes
"""
from __future__ import annotations

import argparse
import logging
import sys

from tests import _runner, _stubs

_stubs.install()

from api_client.api_client import StreamCancelToken  # noqa: E402
from core.models import ApiChunkResult  # noqa: E402
from core.sampler import ApiAntiSlopSampler, _is_repetition  # noqa: E402
from state.generation_state import GenerationState  # noqa: E402
from utils.sampler_helpers import apply_sampling_filters  # noqa: E402
from validators.slop_phrase_validator import SlopPhraseValidator  # noqa: E402
from validators.regex_validator import RegexValidator  # noqa: E402


# --------------------------------------------------------------------------- #
#  Fake backend that honours bad_words like vLLM                              #
# --------------------------------------------------------------------------- #
class VllmLikeClient:
    """One script per attempt.  When a request bans `<turn|>`, the model cannot
    stop where it wanted to and writes `after_ban` until max_tokens."""

    def __init__(self, scripts, after_ban=(), alts=None, fail_first=False):
        self.scripts = [list(s) for s in scripts]
        self.after_ban = list(after_ban)
        self.alts = alts or {}
        self.fail_first = fail_first
        self.calls = []
        self.suppress_tokens_unsupported = False

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
        if self.fail_first and len(self.calls) == 1:
            yield ApiChunkResult("She", ["She"], {0: [("She", 0.0)]}, None)
            raise ConnectionError("connection reset")
        toks = self.scripts.pop(0) if self.scripts else []
        reason = "stop"
        if "<turn|>" in (kw.get("suppress_tokens") or []) and self.after_ban:
            filler = []
            while len(toks) + len(filler) < kw["max_tokens"]:
                filler.extend(self.after_ban)
            toks = toks + filler
        if len(toks) >= kw["max_tokens"]:
            toks, reason = toks[: kw["max_tokens"]], "length"
        for t in toks:
            yield ApiChunkResult(t, [t], {0: self.alts.get(t, [(t, 0.0)])}, None)
        yield ApiChunkResult("", [], {}, reason)


class FakeTokenizer:
    all_special_tokens = ["<bos>", "<eos>", "<turn|>", "<|turn>", "<|channel>",
                          "<channel|>", "<|think|>", '<|"|>']
    added_tokens_decoder = {}


class FakeFormatter:
    model_id = "fake"
    tokenizer = FakeTokenizer()


def _sampler(client, validators=(), formatter=None, **gen_overrides):
    gen = {
        "chunk_size": 20, "top_logprobs_count": 20, "max_new_tokens": 60,
        "temperature": 1.0, "top_p": None, "top_k": None, "min_p": None,
        "timeout": 30, "stop_sequences": [], "request_mode": "stream",
        "max_consecutive_stops": 3, "max_stream_attempts": 12,
        "probe_stream_logprobs": False, "api_retry_delay_sec": 0,
    }
    gen.update(gen_overrides)
    config = {
        "generation_params": gen,
        "backtracking": {"force_backtrack": True, "max_retries_per_position": 100},
        "filter_tail_banned_prefix_tokens": False,
        "regex_max_span_tokens": 10 ** 6,
        "regex_validation_interval": 20,
    }
    s = ApiAntiSlopSampler(client, list(validators), config, None, None, None, formatter)
    s.api_retry_delay_sec = 0
    return s


def _run(s):
    return "".join(s.generate("PROMPT"))


# --------------------------------------------------------------------------- #
#  Stop policy                                                                #
# --------------------------------------------------------------------------- #
def test_refused_stop_does_not_ban_end_of_turn_for_the_rest_of_the_run():
    client = VllmLikeClient(
        [["She", " nods", " and"], [" smiles", "."]],
        after_ban=[" Of", " course", "."],
    )
    s = _sampler(client, restart_suppress_markers=True, restart_suppress_window=32,
                 max_new_tokens=200)
    text = _run(s)
    bans = [bool(c.get("suppress_tokens")) for c in client.calls]
    # Only the attempt right after the refused stop is affected, and it is short.
    assert bans[0] is False and all(not b for b in bans[2:]), bans
    assert client.calls[1]["max_tokens"] <= s.restart_suppress_window
    assert text.count("Of course") <= s.restart_suppress_window // 3 + 1, text


def test_a_dropped_connection_does_not_ban_end_of_turn():
    client = VllmLikeClient([[" nods", "."]], after_ban=[" Of", " course", "."],
                            fail_first=True)
    s = _sampler(client, restart_suppress_markers=True, max_new_tokens=60)
    text = _run(s)
    assert not any(c.get("suppress_tokens") for c in client.calls), client.calls
    assert text == "She nods.", text


def test_roleplay_endings_are_clean_ends():
    s = _sampler(VllmLikeClient([]))
    for end in ["*She leans back.*", "She waits~", "Well—", "Good girl \U0001F60F",
                "Ready?**", "*leans in*", "Love you ♥"]:
        assert s._ends_cleanly(end), end
    assert not s._ends_cleanly("and then")


def test_action_ending_is_not_continued():
    client = VllmLikeClient([["She", " smiles", ".", " *", "She", " leans", " back", ".*"]],
                            after_ban=["\n\n", "The", " director", " calls", " cut", "."])
    s = _sampler(client, restart_suppress_markers=True)
    assert _run(s) == "She smiles. *She leans back.*"
    assert len(client.calls) == 1


# --------------------------------------------------------------------------- #
#  Back-tracking                                                              #
# --------------------------------------------------------------------------- #
def test_backtrack_never_picks_a_control_token():
    import random
    for seed in range(20):
        random.seed(seed)
        client = VllmLikeClient(
            [["She", " finished", ".", " She", " took", " a", " deep", " breath", "."],
             [" Her", " eyes", " closed", "."]],
            alts={" She": [(" She", -0.4), ("<turn|>", -1.2), (" Her", -2.0)]},
        )
        sv = SlopPhraseValidator({"she took a deep breath": 1.0}, app_config=None)
        _run(_sampler(client, [sv]))
        assert not any("<turn|>" in c["prompt_text"] for c in client.calls), seed


def test_suppressions_after_a_rewritten_token_are_forgotten():
    client = VllmLikeClient(
        [[" A", " pale", " and", " cold", " day", "."], [" and", " calm", "."]],
        alts={" pale": [(" pale", -0.01), (" warm", -1.0)]},
    )
    rv = RegexValidator([r"pale and"])
    rv._ignored = {(4, "stale"), (1, "x")}
    s = _sampler(client, [rv], regex_validation_interval=1)
    _run(s)
    assert all(k[0] <= 1 for k in rv._ignored), rv._ignored


# --------------------------------------------------------------------------- #
#  Control tokens                                                             #
# --------------------------------------------------------------------------- #
def test_tokenizer_control_tokens_are_scrubbed():
    client = VllmLikeClient([["She", " smiles", ".", "<|channel>", "thought", "<channel|>",
                              " Yes", ".", '<|"|>']])
    s = _sampler(client, formatter=FakeFormatter())
    text = _run(s)
    for marker in ("<|channel>", "<channel|>", '<|"|>'):
        assert marker not in text, text


def test_prose_angle_brackets_survive_when_the_tokenizer_is_known():
    client = VllmLikeClient([["The", " sign", " read", " <", "mask", ">", "."]])
    s = _sampler(client, formatter=FakeFormatter())
    assert _run(s) == "The sign read <mask>."


def test_gemma4_markers_in_the_fallback_pattern():
    client = VllmLikeClient([["Yes", ".", "<|channel>", "<|think|>", '<|"|>']])
    assert _run(_sampler(client)) == "Yes."


# --------------------------------------------------------------------------- #
#  Repetition check                                                           #
# --------------------------------------------------------------------------- #
def test_repetition_check_compares_words_not_characters():
    assert not _is_repetition("Then she said nothing.", "he said")
    assert _is_repetition("Of course. Of course.", "of course.")


# --------------------------------------------------------------------------- #
#  Validators                                                                 #
# --------------------------------------------------------------------------- #
def _state(text):
    toks = [w if i == 0 else " " + w for i, w in enumerate(text.split(" "))]
    st = GenerationState.hypothetical("P", toks)
    return st


def test_slop_window_does_not_start_mid_word():
    sv = SlopPhraseValidator({"grim": 1.0}, app_config=None)
    window = sv.max_phrase_len + sv.scan_window_base_size
    # The window covers the last `window` characters.  Make it start exactly on
    # the "g" of "pilgrim": text = "x" + rest with len(rest) == window + 4.
    filler_len = window + 4 - len(" pilgrim")
    filler = " a" * (filler_len // 2) + ("." if filler_len % 2 else "")
    text = "x pilgrim" + filler
    st = GenerationState.hypothetical(
        "P", ["x", " pil", "grim"] + [" a"] * (filler_len // 2) + (["."] if filler_len % 2 else []))
    gen = st.get_generated_text()
    assert gen == text
    assert gen[len(gen) - window:][:4] == "grim", gen[len(gen) - window - 3:len(gen) - window + 5]
    assert sv.check(st, final=True) is None


def test_a_suppressed_match_does_not_hide_a_later_one():
    sv = SlopPhraseValidator({"gaze": 1.0, "testament": 1.0}, app_config=None)
    st = _state("Her gaze was a testament to it .")
    first = sv.check(st, final=True)
    assert first.details["phrase"] == "gaze"
    sv.ignore_violation(first)
    second = sv.check(st, final=True)
    assert second is not None and second.details["phrase"] == "testament", second


def test_regex_suppression_does_not_hide_a_later_match():
    rv = RegexValidator([r"not \w+, but"])
    st = _state("It was not fear, but more . It was not love, but more .")
    first = rv.check(st)
    rv.ignore_violation(first)
    second = rv.check(st)
    assert second is not None and second.violation_index > first.violation_index


def test_typographic_apostrophes_match():
    sv = SlopPhraseValidator({"couldn't help but": 1.0}, app_config=None)
    assert sv.check(_state("She couldn’t help but smile ."), final=True) is not None
    rv = RegexValidator([r"\bit'?s\s+not"])
    assert rv.check(_state("And it’s not over .")) is not None


def test_top_k_disabled_values():
    pairs = [("a", 0.6), ("b", 0.4)]
    assert len(apply_sampling_filters(pairs, min_p=None, top_p=None, top_k=-1)) == 2
    assert len(apply_sampling_filters(pairs, min_p=None, top_p=None, top_k=0)) == 2


# --------------------------------------------------------------------------- #
#  Config and server                                                          #
# --------------------------------------------------------------------------- #
def test_force_backtrack_flag_reaches_the_sampler_key():
    from utils.helpers import merge_configs
    args = argparse.Namespace(force_backtrack=False)
    cfg = merge_configs({"backtracking": {"force_backtrack": True}}, args)
    assert cfg["backtracking"]["force_backtrack"] is False


def test_word_boundary_switch_under_generation_params():
    sv = SlopPhraseValidator({"bill": 1.0},
                             app_config={"generation_params": {"match_requires_word_boundaries": False}})
    assert sv.match_requires_word_boundaries is False


def _server_with(client, formatter):
    from api_server import server
    server.setup_shared_resources(
        {"generation_params": {"request_mode": "stream", "probe_stream_logprobs": False,
                               "threads": 40},
         "threads": 1, "filter_tail_banned_prefix_tokens": False},
        [], client, formatter, logging.getLogger("t"))
    return server


class _Fmt:
    model_id = "fake"
    tokenizer = None
    seen_kwargs = None

    def build_base_prompt_from_messages(self, messages, **kw):
        _Fmt.seen_kwargs = kw
        return "|".join(m["content"] for m in messages)


def test_server_forwards_penalties_and_template_kwargs():
    client = VllmLikeClient([["Hi", "."]])
    fmt = _Fmt()
    fmt.tokenizer = object()
    server = _server_with(client, fmt)
    assert server.SHARED_RESOURCES["max_concurrent_requests"] == 40
    server._run_generation_for_request({
        "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
        "repetition_penalty": 1.1, "presence_penalty": 0.2, "seed": 3,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    sent = client.calls[0]
    assert sent.get("repetition_penalty") == 1.1 and sent.get("presence_penalty") == 0.2, sent
    assert "seed" not in sent, sent
    assert _Fmt.seen_kwargs == {"enable_thinking": False}


def test_server_refuses_to_drop_the_conversation_without_a_template():
    client = VllmLikeClient([["Hi", "."]])
    server = _server_with(client, None)
    res = server._run_generation_for_request(
        {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]})
    assert "error" in res and not client.calls, res


# --------------------------------------------------------------------------- #
#  Forced phrases and backend failure (found in the real-model run)           #
# --------------------------------------------------------------------------- #
def test_a_phrase_the_context_forces_is_allowed_instead_of_looping():
    # The model keeps writing the character's name; the list bans the word.
    # Seen on Gemma 4 31B: a character called Mara against the entry "mara"
    # produced "**\\\n\\\n\\\n..." until the token cap.
    alts = {" Mara": [(" Mara", -0.01), ("\\", -4.0), ("\n", -4.5)]}
    first = ["Her", " name", " is", " Mara", "."]
    again = [" Mara", "."]
    client = VllmLikeClient([first] + [again] * 10, alts=alts)
    sv = SlopPhraseValidator({"mara": 1.0}, app_config=None)
    s = _sampler(client, [sv], max_new_tokens=200)
    text = _run(s)
    assert text == "Her name is Mara.", text
    assert len(client.calls) <= 5, len(client.calls)


def _dying_client(tokens_before_drop):
    class Dying(VllmLikeClient):
        def generate_stream(self, **kw):
            self.calls.append(kw)
            for t in tokens_before_drop:
                yield ApiChunkResult(t, [t], {0: [(t, 0.0)]}, None)
            raise ConnectionError("backend gone")
    return Dying([])


def test_backend_failure_is_an_error_not_a_200():
    # Seen in the real-model run: with the backend gone, every request came back
    # HTTP 200 with empty text and finish_reason "error".
    for tokens in ([], [" word"]):     # gone before any token / drops mid-stream
        client = _dying_client(tokens)
        fmt = _Fmt()
        fmt.tokenizer = object()
        server = _server_with(client, fmt)
        server.SHARED_RESOURCES["config"]["generation_params"]["api_retry_delay_sec"] = 0
        res = server._run_generation_for_request({"messages": [{"role": "user", "content": "U"}]})
        assert res.get("http_status") == 502, (tokens, res)


def main() -> int:
    return _runner.run_suite("test_audit_fixes", globals())


if __name__ == "__main__":
    sys.exit(main())
