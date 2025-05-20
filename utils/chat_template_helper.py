# utils/chat_template_helper.py
"""
Turns a Hugging-Face chat template into a plain-text prompt compatible
with the **/v1/completions** endpoint.

If *system_prompt* is supplied it is injected as a system role message
exactly once, before the user message.
"""
import threading
from typing import Tuple
from transformers import AutoTokenizer


class ChatTemplateFormatter:
    _cache = {}
    _lock  = threading.Lock()

    def __init__(self,
                 model_id: str,
                 system_prompt: str = "") -> None:
        self.model_id      = model_id
        self.system_prompt = system_prompt or ""

        with self._lock:
            tok = self._cache.get(model_id)
            if tok is None:
                tok = AutoTokenizer.from_pretrained(model_id,
                                                    trust_remote_code=True)
                self._cache[model_id] = tok
        self.tokenizer = tok

        self._prefix, self._middle, self._sys_placeholder = \
            self._extract_segments()

    # ------------------------------------------------------------- #
    # internal helpers                                              #
    # ------------------------------------------------------------- #
    def _extract_segments(self) -> Tuple[str, str, str]:
        ph_user = "__PLACEHOLDER_USER__"
        ph_asst = "__PLACEHOLDER_ASST__"
        ph_sys  = "__PLACEHOLDER_SYS__"

        messages = []
        if self.system_prompt:
            messages.append({"role": "system",    "content": ph_sys})
        messages.extend([
            {"role": "user",      "content": ph_user},
            {"role": "assistant", "content": ph_asst},
        ])

        tpl = self.tokenizer.apply_chat_template(messages,
                                                 tokenize=False,
                                                 add_generation_prompt=False)

        i_user = tpl.find(ph_user)
        i_asst = tpl.find(ph_asst)
        if i_user == -1 or i_asst == -1 or i_asst <= i_user:
            raise ValueError("placeholders not found in template")

        prefix = tpl[:i_user]                       # up to user placeholder
        middle = tpl[i_user + len(ph_user): i_asst] # between user & asst
        return prefix, middle, ph_sys

    # ------------------------------------------------------------- #
    # public API                                                    #
    # ------------------------------------------------------------- #
    def build_prompt(self,
                     user_prompt: str,
                     assistant_so_far: str = "") -> str:
        """
        Returns *prefix + user_prompt + middle + assistant_so_far*,
        with the system prompt already inserted (if any).
        """
        prefix = self._prefix
        if self.system_prompt:
            prefix = prefix.replace(self._sys_placeholder, self.system_prompt)

        return f"{prefix}{user_prompt}{self._middle}{assistant_so_far}"
