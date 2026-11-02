"""Import-time stubs for the heavy optional dependencies.

The CI/target box has no GPU and no ``torch``/``transformers``/``tiktoken``/
``nltk``.  ``core.sampler`` imports them at module scope, so the tests install
minimal fakes into ``sys.modules`` *before* importing it.

Only what the sampler actually touches is implemented:

* ``tiktoken.encoding_for_model`` / ``tiktoken.get_encoding`` → an encoder whose
  ``encode`` splits on whitespace (token counts are only used for progress
  reporting in the tests).
* ``transformers.AutoTokenizer`` → only needed by the tail-prefix builder, which the
  tests disable via ``filter_tail_banned_prefix_tokens: false``.
"""
from __future__ import annotations

import sys
import types


class _FakeEncoding:
    def encode(self, text, *args, **kwargs):
        return (text or "").split()


def install() -> None:
    if "tiktoken" not in sys.modules:
        tiktoken = types.ModuleType("tiktoken")
        tiktoken.encoding_for_model = lambda name: _FakeEncoding()
        tiktoken.get_encoding = lambda name: _FakeEncoding()
        sys.modules["tiktoken"] = tiktoken

    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")

        class _AutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("tokenizer unavailable in tests")

        transformers.AutoTokenizer = _AutoTokenizer
        sys.modules["transformers"] = transformers

    # ``validators/__init__.py`` imports the n-gram validator eagerly, which
    # imports nltk.  These tests do not use it, so a bare stub is enough.
    if "nltk" not in sys.modules:
        nltk = types.ModuleType("nltk")
        nltk.word_tokenize = lambda text, **kwargs: (text or "").split()
        nltk.corpus = types.SimpleNamespace(
            stopwords=types.SimpleNamespace(words=lambda lang: []),
        )
        nltk.data = types.SimpleNamespace(find=lambda *a, **k: True)
        nltk.download = lambda *a, **k: True
        sys.modules["nltk"] = nltk
