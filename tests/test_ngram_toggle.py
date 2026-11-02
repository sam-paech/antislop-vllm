"""Regression tests for disabling the N-gram validator from ``config.yaml``.

``ngram_validator.enabled: false`` must stop ``main._setup_validators`` from
building an :class:`NGramValidator` at all, even when ``banned_file`` /
``banned_list`` are configured.  Before this option existed the only way to turn
n-gram banning off was to empty the ban list, which silently re-enabled itself as
soon as a file was configured again.

Run with::

    python -m tests.test_ngram_toggle
"""
from __future__ import annotations

import contextlib
import logging
import sys

from tests import _runner, _stubs

_stubs.install()

import main as main_module  # noqa: E402  (aliased: this module defines main())
import validators.ngram_validator as ngram_module  # noqa: E402
from utils.helpers import load_config  # noqa: E402


# --------------------------------------------------------------------------- #
#  helpers                                                                    #
# --------------------------------------------------------------------------- #
BANNED_FILE = "banlists/banned_ngrams.json"


@contextlib.contextmanager
def _spy_ngram_validator():
    """Replace ``NGramValidator`` with a recorder.

    The real class needs NLTK data and disables itself when it is missing, so a
    test cannot tell "skipped because disabled" from "built but inert" by looking
    at the validator list.  Counting constructions is the observable behaviour.
    """
    constructed: list[dict] = []

    class SpyNGramValidator:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.is_disabled = False

    original = ngram_module.NGramValidator
    ngram_module.NGramValidator = SpyNGramValidator
    try:
        yield constructed
    finally:
        ngram_module.NGramValidator = original


def _cfg(ngram_section):
    """Minimal config: no slop/regex files, so only the n-gram path runs."""
    return {
        "slop_phrases_file": None,
        "regex_blocklist_file": None,
        "ngram_validator": dict(ngram_section),
    }


def _setup(cfg):
    return main_module._setup_validators(cfg, logging.getLogger("test_ngram_toggle"))


# --------------------------------------------------------------------------- #
#  Behaviour: the toggle                                                     #
# --------------------------------------------------------------------------- #
def test_disabled_ngram_validator_is_not_constructed():
    """``enabled: false`` must skip construction, not just empty the ban list."""
    cfg = _cfg({"enabled": False, "banned_file": BANNED_FILE})
    with _spy_ngram_validator() as constructed:
        validators = _setup(cfg)

    assert constructed == [], f"NGramValidator built despite enabled=false: {constructed}"
    assert not [v for v in validators if type(v).__name__ == "NGramValidator"], validators


def test_disabled_string_form_is_honoured():
    """A YAML value quoted as ``"false"`` must mean false, not "non-empty"."""
    cfg = _cfg({"enabled": "false", "banned_file": BANNED_FILE})
    with _spy_ngram_validator() as constructed:
        _setup(cfg)

    assert constructed == [], constructed


def test_disabled_beats_an_inline_banned_list():
    cfg = _cfg({"enabled": False, "banned_list": ["as an ai language model"]})
    with _spy_ngram_validator() as constructed:
        _setup(cfg)

    assert constructed == [], constructed


def test_enabled_ngram_validator_is_still_constructed():
    """The default path must be unchanged: configured n-grams are still loaded."""
    cfg = _cfg({"enabled": True, "banned_file": BANNED_FILE})
    with _spy_ngram_validator() as constructed:
        _setup(cfg)

    assert len(constructed) == 1, constructed
    assert constructed[0]["banned_ngrams_config"], "ban list was not loaded"


def test_missing_enabled_key_defaults_to_on():
    """Older config files have no ``enabled`` key and must keep working."""
    cfg = _cfg({"banned_file": BANNED_FILE})
    with _spy_ngram_validator() as constructed:
        _setup(cfg)

    assert len(constructed) == 1, constructed


# --------------------------------------------------------------------------- #
#  Config plumbing                                                           #
# --------------------------------------------------------------------------- #
def test_example_config_leaves_ngram_validation_on():
    cfg = load_config("config-example.yaml")
    assert cfg["ngram_validator"].get("enabled", True) is True, cfg["ngram_validator"]


def main() -> int:
    return _runner.run_suite("test_ngram_toggle", globals())


if __name__ == "__main__":
    sys.exit(main())
