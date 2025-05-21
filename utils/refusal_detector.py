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
)


class RefusalDetector:
    """
    Thin wrapper around NousResearch/Minos-v1 (or any compatible
    classifier).  Mirrors the usage pattern shown on the model card.

    • If the model or tokenizer fails to load, the error is printed
      loudly to stderr *once*.  Subsequent calls still print a short
      notice and return the sentinel values
      ("error", –1, 0.0) / (False, 0.0, "error").
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
    ) -> None:
        # These attributes are always present so method calls never break,
        # even when the load fails.
        self.device = device
        self.tokenizer = None
        self.model = None
        self.id2label: dict[int, str] = {}
        self._failed = False
        self._error: str | None = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)

            cfg = AutoConfig.from_pretrained(model_id)
            # ModernBERT uses the *private* field _attn_implementation.
            # We set both the private and public names to maximise coverage.
            for attr in ("_attn_implementation", "attn_implementation"):
                if hasattr(cfg, attr):
                    setattr(cfg, attr, "eager")  # valid keys: "torch", "eager"

            self.model = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    torch_dtype="bfloat16",
                    reference_compile=False,
                    attn_implementation="eager",
                )
                .to(device)
                .eval()
            )
            self.id2label = {
                int(k): v for k, v in self.model.config.id2label.items()
            }
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
    ) -> "RefusalDetector":
        """
        Thread-safe factory.  The underlying model is loaded at most once
        per Python process.  If loading fails, the same failed instance
        is returned on subsequent calls, so no further load attempts are
        made.
        """
        with cls._lock:
            want_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            if cls._instance is None or cls._model_id != (model_id, want_device):
                cls._instance = RefusalDetector(model_id, device=want_device)
                cls._model_id = (model_id, want_device)
        return cls._instance

    # ------------------------------------------------------------------ #
    #  Core helpers                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _chat_wrap(user: str, assistant: str) -> str:
        return f"<|user|>\n{user}\n<|assistant|>\n{assistant}"

    # -------- public API ------------------------------------------------#
    def classify(
        self,
        user_text: str,
        assistant_text: str,
    ) -> Tuple[str, int, float]:
        """
        Returns (label_text, label_id, confidence).  If the detector
        failed to load, a sentinel triple ("error", –1, 0.0) is returned.
        """
        if self._failed:
            # Secondary, less-intrusive notice.
            print(
                "[RefusalDetector] Model unavailable – returning sentinel.",
                file=sys.stderr,
            )
            return "error", -1, 0.0

        text = self._chat_wrap(user_text, assistant_text)
        inputs = self.tokenizer(  # type: ignore[call-arg]
            text,
            return_tensors="pt",
            truncation=True,  # delete if you never exceed 2 048 tokens
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits  # type: ignore[attr-defined]
            probs = torch.softmax(logits, dim=-1)[0]
            label_id = int(torch.argmax(probs).item())
            conf = float(probs[label_id])

        label_txt = self.id2label.get(label_id, str(label_id))
        return label_txt, label_id, conf

    def is_refusal(
        self,
        user_text: str,
        assistant_text: str,
        threshold: float = 0.5,
    ) -> Tuple[bool, float, str]:
        """
        Returns (is_refusal, confidence, raw_label).  If the detector
        failed to load, (False, 0.0, "error") is returned.
        """
        if self._failed:
            print(
                "[RefusalDetector] Model unavailable – returning sentinel.",
                file=sys.stderr,
            )
            return False, 0.0, "error"

        label, _id, conf = self.classify(user_text, assistant_text)
        label_lower = label.lower().strip()
        is_ref = (label_lower == "refusal") and conf >= threshold
        return is_ref, conf, label
