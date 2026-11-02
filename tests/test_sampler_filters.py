"""Regression tests for sampling-filter interaction during back-tracking.

Run with::

    python -m tests.test_sampler_filters
"""
from __future__ import annotations

import math
import sys
from unittest.mock import patch

from tests import _runner, _stubs

_stubs.install()

from core.models import ViolationInfo  # noqa: E402
from core.sampler import ApiAntiSlopSampler  # noqa: E402


def _make_sampler(**gen_overrides) -> ApiAntiSlopSampler:
    """Build a sampler through the real constructor.

    This used to be ``object.__new__(ApiAntiSlopSampler)`` with a hand-written list
    of attributes.  That bypassed ``__init__``, so the test kept passing while the
    sampler gained attributes it needed: it broke on
    ``max_retries_per_position`` with an ``AttributeError`` that said nothing about
    the behaviour under test.  Going through ``__init__`` means a new required
    attribute is picked up for free, and a real misconfiguration shows up here.

    ``probe_stream_logprobs`` is off because the backend is a fake; everything else
    that matters to these tests comes from config.
    """
    gen = {
        "temperature": 1.0,
        "min_p": 0.10,
        "top_p": None,
        "top_k": None,
        "top_logprobs_count": 5,
        "max_new_tokens": 50,
        "request_mode": "stream",
        "probe_stream_logprobs": False,
    }
    gen.update(gen_overrides)
    config = {
        "generation_params": gen,
        "backtracking": {"max_retries_per_position": 5},
        "filter_tail_banned_prefix_tokens": False,
    }
    return ApiAntiSlopSampler(api_client=object(), validators=[], config=config)


def test_rejected_top_token_does_not_lower_min_p_floor():
    """min-p must be measured before the banned token is removed.

    ``banned`` is the top token at 0.80, and ``min_p=0.10``, so the floor is
    0.08: only ``banned`` (0.80) and ``good`` (0.10) survive.  ``banned`` is then
    excluded as a hard ban, leaving exactly ``good``.

    If the ban were applied first, the floor would instead be 0.10 × 0.10 = 0.01 and
    the tail tokens (0.05, 0.03, 0.02) would sneak back in — tokens plain min-p
    sampling could never have selected.
    """
    probabilities = {
        "banned": 0.80,
        "good": 0.10,
        "tail-1": 0.05,
        "tail-2": 0.03,
        "tail-3": 0.02,
    }

    class State:
        generated_token_strings = ["banned"]
        prompt_string = ""

        def get_logprobs(self, _index):
            return [(token, math.log(prob)) for token, prob in probabilities.items()]

        def truncate(self, _index):
            pass

        def replace_token_string(self, index, token):
            self.generated_token_strings[index] = token

    sampler = _make_sampler(top_logprobs_count=len(probabilities))
    # The candidate check is the seam: it decides which tokens are acceptable.
    # It must accept ``final``: the sampler passes it through to distinguish a
    # mid-stream candidate (a match touching the end of the text is provisional)
    # from a final-sweep one (it is definitive).
    sampler._check_hypothetical_state = lambda _state, _index, _token, final=False: (True, None)

    violation = ViolationInfo(
        validator_type="regex",
        violation_index=0,
        original_token_string="banned",
        details={},
    )

    def choose(tokens, weights, k):
        assert tokens == ("good",), tokens
        assert k == 1, k
        return [tokens[0]]

    state = State()
    with patch("core.sampler.random.choices", side_effect=choose):
        assert sampler._perform_backtrack(state, violation)

    assert state.generated_token_strings == ["good"]


def test_backtrack_stops_at_the_configured_budget():
    """The per-position budget must be read from config and enforced.

    A position whose candidates are all rejected used to be re-entered without limit:
    each pass resampled from the same cached logprobs and generation made no
    progress.  The budget is what ends it.
    """
    sampler = _make_sampler()

    class State:
        generated_token_strings = ["banned"]
        prompt_string = ""

        def get_logprobs(self, _index):
            return [("banned", math.log(1.0))]

        def truncate(self, _index):
            pass

        def replace_token_string(self, index, token):
            self.generated_token_strings[index] = token

    sampler._check_hypothetical_state = (
        lambda _state, _index, _token, final=False: (True, None)
    )
    sampler._suppress_violation = lambda _vio: None

    violation = ViolationInfo(
        validator_type="regex",
        violation_index=0,
        original_token_string="banned",
        details={},
    )

    state = State()
    # Every candidate is the banned token, so no replacement exists. The budget,
    # not the candidate list, is what has to end this.
    assert sampler._perform_backtrack(state, violation) is False
    assert state.generated_token_strings == ["banned"], state.generated_token_strings


def main() -> int:
    return _runner.run_suite("test_sampler_filters", globals())


if __name__ == "__main__":
    sys.exit(main())
