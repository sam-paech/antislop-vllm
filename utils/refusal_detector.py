# utils/refusal_detector.py
from __future__ import annotations

import sys
import threading
from typing import Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig
)

# ---------------------------------------------------------------------- #
#  Tunables                                                              #
# ---------------------------------------------------------------------- #
MAX_LEN          = 512   # overall sequence budget (change to 200 if desired)
MIN_USER_TOKENS  = 40    # never shorten user_text below this many tokens

class RefusalDetector:
    """
    Thin wrapper around NousResearch/Minos-v1 (or any compatible classifier).

    • If the model or tokenizer fails to load, the error is printed loudly to
      stderr *once*.  Subsequent calls still print a short notice and return the
      sentinel values ("error", –1, 0.0) / (False, 0.0, "error").
    """

    _instance: "RefusalDetector | None" = None
    _model_id: tuple[str, str] | None = None
    _lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Construction (private – use .get())                               #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_len: int = MAX_LEN,
    ) -> None:
        # These attributes are always present so method calls never break,
        # even when the load fails.
        self.device      = device
        self.max_len     = max_len
        self.tokenizer   = None
        self.model       = None
        self.id2label: dict[int, str] = {}
        self._failed     = False
        self._error: str | None = None
        self._tok_lock = threading.Lock()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            # shrink the tokenizer’s default budget so downstream code stays honest
            self.tokenizer.model_max_length = max_len

            # one-shot truncation setup ─ avoids per-call writes
            if self.tokenizer.is_fast:                     # safe for slow tokenizers
                self.tokenizer._tokenizer.enable_truncation(max_length=max_len)

            cfg = AutoConfig.from_pretrained(model_id)
            # ModernBERT uses the *private* field _attn_implementation.
            # Set both the private and public names to maximise coverage.
            for attr in ("_attn_implementation", "attn_implementation"):
                if hasattr(cfg, attr):
                    setattr(cfg, attr, "eager")  # valid keys: "torch", "eager"

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization
                bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16
                bnb_4bit_use_double_quant=True  # Use double quantization for better accuracy
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                torch_dtype="bfloat16",
                quantization_config=bnb_config,
                reference_compile=False,
                attn_implementation="eager",
            )
            self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

        except Exception as exc:
            # ‼️ Loud error – printed once during first failure
            self._failed = True
            self._error = str(exc)
            print(
                f"\n[RefusalDetector ERROR] Failed to load '{model_id}': {exc}\n",
                file=sys.stderr,
                flush=True,
            )

    # ------------------------------------------------------------------ #
    #  Singleton accessor                                                #
    # ------------------------------------------------------------------ #
    @classmethod
    def get(
        cls,
        model_id: str = "NousResearch/Minos-v1",
        *,
        device: str | None = None,
        max_len: int = MAX_LEN,
    ) -> "RefusalDetector":
        """
        Thread-safe factory.  The underlying model is loaded at most once per
        Python process.  If loading fails, the same failed instance is returned
        on subsequent calls, so no further load attempts are made.
        """
        with cls._lock:
            want_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            cache_key   = (model_id, want_device, max_len)
            if cls._instance is None or cls._model_id != cache_key:
                cls._instance = RefusalDetector(
                    model_id, device=want_device, max_len=max_len
                )
                cls._model_id = cache_key
        return cls._instance

    # ------------------------------------------------------------------ #
    #  Core helpers                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _chat_wrap(user: str, assistant: str) -> str:
        return f"<|user|>\n{user}\n<|assistant|>\n{assistant}"

    def _truncate_inputs(
        self,
        user_text: str,
        assistant_text: str,
    ) -> tuple[str, str]:
        """
        Ensures the wrapped sequence fits within self.max_len:
        1. Compute total length.
        2. If too long, drop tokens from the *end* of user_text down to
           MIN_USER_TOKENS.
        3. Anything still over budget is handled by tokenizer-level truncation.
        """
        tok = self.tokenizer  # shorthand
        # token counts without special tokens
        user_ids       = tok.encode(user_text, add_special_tokens=False)
        assistant_ids  = tok.encode(assistant_text, add_special_tokens=False)
        static_ids     = tok.encode(self._chat_wrap("", ""), add_special_tokens=False)

        total_len = len(static_ids) + len(user_ids) + len(assistant_ids)
        if total_len <= self.max_len:
            return user_text, assistant_text

        excess = total_len - self.max_len
        # How many user tokens can we drop while respecting MIN_USER_TOKENS?
        drop = min(excess, max(0, len(user_ids) - MIN_USER_TOKENS))
        if drop:
            user_ids = user_ids[:-drop]
            user_text = tok.decode(user_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            excess -= drop

        # Any remaining excess will be handled by the tokenizer's own truncation.
        return user_text, assistant_text

    # -------- public API ------------------------------------------------#
    def classify(
        self,
        user_text: str,
        assistant_text: str,
    ) -> Tuple[str, int, float]:
        """
        Returns (label_text, label_id, confidence).  If the detector failed
        to load, a sentinel triple ("error", –1, 0.0) is returned.
        """
        if self._failed:
            # Secondary, less-intrusive notice.
            print(
                "[RefusalDetector] Model unavailable – returning sentinel.",
                file=sys.stderr,
            )
            return "error", -1, 0.0

        # --- 1. length control -----------------------------------------
        user_text, assistant_text = self._truncate_inputs(user_text, assistant_text)

        # --- 2. tokenise & run -----------------------------------------
        text = self._chat_wrap(user_text, assistant_text)

        print(text)

        with self._tok_lock:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=True,          # ← ensure truncation
                max_length=self.max_len,  # ← limit to model capacity
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits  # type: ignore[attr-defined]
                probs  = torch.softmax(logits, dim=-1)[0]
                label_id = int(torch.argmax(probs).item())
                conf     = float(probs[label_id])

            label_txt = self.id2label.get(label_id, str(label_id))

            del inputs, logits, probs  # free per-call tensors
            torch.cuda.empty_cache()
        return label_txt, label_id, conf

    def is_refusal(
        self,
        user_text: str,
        assistant_text: str,
        threshold: float = 0.8,
    ) -> Tuple[bool, float, str]:
        """
        Returns (is_refusal, confidence, raw_label).  If the detector failed
        to load, (False, 0.0, "error") is returned.
        """
        if self._failed:
            print(
                "[RefusalDetector] Model unavailable – returning sentinel.",
                file=sys.stderr,
            )
            return False, 0.0, "error"

        label, _id, conf = self.classify(user_text, assistant_text)
        is_ref = (label.lower().strip() == "refusal") and conf >= threshold
        print(is_ref, conf, label)
        return is_ref, conf, label
