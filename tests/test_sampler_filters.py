import math
from unittest.mock import patch

from core.models import ViolationInfo
from core.sampler import ApiAntiSlopSampler


def test_rejected_top_token_does_not_lower_min_p_floor():
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

    sampler = object.__new__(ApiAntiSlopSampler)
    sampler.ban_strength = 1.0
    sampler._tried_alternatives = {}
    sampler.temperature = 1.0
    sampler.min_p = 0.10
    sampler.top_p = None
    sampler.top_k = None
    sampler.force_backtrack = False
    sampler.top_logprobs_count = len(probabilities)
    sampler.max_chosen_tokens = 0
    sampler.ftpo_samples = {}
    sampler._check_hypothetical_state = lambda _state, _index, _token: (True, None)

    violation = ViolationInfo(
        validator_type="regex",
        violation_index=0,
        original_token_string="banned",
        details={},
    )

    def choose(tokens, weights, k):
        assert tokens == ("good",)
        assert k == 1
        return [tokens[0]]

    state = State()
    with patch("core.sampler.random.choices", side_effect=choose):
        assert sampler._perform_backtrack(state, violation)

    assert state.generated_token_strings == ["good"]
