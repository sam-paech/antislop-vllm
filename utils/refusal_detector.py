# utils/refusal_detector.py
from __future__ import annotations

import threading
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


class RefusalDetector:
    """
    Thin wrapper around NousResearch/Minos-v1 (or any compatible
    classifier).  Mirrors the usage pattern shown on the model card.
    """

    _instance = None
    _model_id = None
    _lock     = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Construction (private – use .get())                               #
    # ------------------------------------------------------------------ #
    def __init__(self, model_id: str,
                 *,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"
                 ) -> None:

        self.device    = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        

        cfg = AutoConfig.from_pretrained(model_id)
        # ModernBERT uses the *private* field _attn_implementation.
        # We set both the private and public names to maximise coverage.
        for attr in ("_attn_implementation", "attn_implementation"):
            if hasattr(cfg, attr):
                setattr(cfg, attr, "eager")     # valid keys: "torch", "eager"

        # 3) model
        self.model = AutoModelForSequenceClassification.from_pretrained(
                "NousResearch/Minos-v1",
                torch_dtype="bfloat16",
                reference_compile=False,
                attn_implementation="eager",
        ).to(device).eval()
        self.id2label  = {int(k): v for k, v in
                          self.model.config.id2label.items()}

    # ------------------------------------------------------------------ #
    #  Singleton accessor                                                #
    # ------------------------------------------------------------------ #
    @classmethod
    def get(cls, model_id: str = "NousResearch/Minos-v1",
            *,
            device: str | None = None) -> "RefusalDetector":
        """
        Cheap thread-safe factory.  The model is loaded only once per
        Python process.
        """
        with cls._lock:
            want_device = (device if device is not None else
                           ("cuda" if torch.cuda.is_available() else "cpu"))
            if (cls._instance is None
                    or cls._model_id != (model_id, want_device)):
                cls._instance = RefusalDetector(model_id,
                                                device=want_device)
                cls._model_id = (model_id, want_device)
        return cls._instance

    # ------------------------------------------------------------------ #
    #  Core helpers                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _chat_wrap(user: str, assistant: str) -> str:
        return f"<|user|>\n{user}\n<|assistant|>\n{assistant}"

    def classify(self, user_text: str,
                 assistant_text: str) -> Tuple[str, int, float]:
        """
        Exactly the sequence shown in the model-card example.
        Returns (label_text, label_id, confidence).
        """
        text = self._chat_wrap(user_text, assistant_text)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,              # ← delete if you never exceed 2 048 t
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1)[0]
            label_id = int(torch.argmax(probs).item())
            conf     = float(probs[label_id])

        label_txt = self.id2label.get(label_id, str(label_id))
        return label_txt, label_id, conf

    def is_refusal(self, user_text: str,
                   assistant_text: str,
                   threshold: float = .5
                   ) -> Tuple[bool, float, str]:
        """
        Convenience: returns (is_refusal, confidence, raw_label).
        A refusal is any label whose text contains “refusal”
        (case-insensitive) **and** whose probability ≥ threshold.
        """
        label, _id, conf = self.classify(user_text, assistant_text)
        label_lower = label.lower().strip()
        is_ref = (label_lower == "refusal") and conf >= threshold
        return is_ref, conf, label
