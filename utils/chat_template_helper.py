# utils/chat_template_helper.py
"""
Light-weight wrapper that turns a Hugging Face *chat template* into a
plain-text prompt suitable for the **completions** endpoint.

Call `build_prompt(user_prompt, assistant_so_far)` each time you want to
hit the API.  The helper guarantees that the returned string ends **just
after** the template’s assistant-header token(s) so the model can keep
generating.
"""
import threading
from typing import Tuple

from transformers import AutoTokenizer


class ChatTemplateFormatter:
    _cache = {}
    _lock = threading.Lock()

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

        # ── one-time tokenizer download / cache ─────────────────────────
        with self._lock:
            tok = self._cache.get(model_id)
            if tok is None:
                tok = AutoTokenizer.from_pretrained(
                    model_id, trust_remote_code=True
                )
                self._cache[model_id] = tok
        self.tokenizer = tok

        # ── dissect the template into   prefix + middle + suffix ───────
        self._prefix, self._middle = self._extract_segments()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                  #
    # ------------------------------------------------------------------ #
    def _extract_segments(self) -> Tuple[str, str]:
        """
        Return `(prefix, middle)` such that

            prompt = prefix + <user_prompt> + middle + <assistant_text>

        `suffix` (anything the template adds *after* assistant text,
        e.g. `<|end_of_turn|>`) is purposely discarded so the model keeps
        writing.
        """
        ph_user = "__PLACEHOLDER_USER__"
        ph_ass  = "__PLACEHOLDER_ASST__"

        tpl = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": ph_user},
                {"role": "assistant", "content": ph_ass},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        i_u = tpl.find(ph_user)
        i_a = tpl.find(ph_ass)
        if i_u == -1 or i_a == -1 or i_a <= i_u:
            raise ValueError(
                f"Cannot locate placeholders in chat template for {self.model_id}"
            )

        prefix  = tpl[:i_u]
        middle  = tpl[i_u + len(ph_user) : i_a]
        return prefix, middle

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #
    def build_prompt(
        self, user_prompt: str, assistant_so_far: str = ""
    ) -> str:
        """
        Compose the prompt for the *completions* call.
        """
        prompt = f"{self._prefix}{user_prompt}{self._middle}{assistant_so_far}"
        return prompt
