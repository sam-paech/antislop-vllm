# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versions are date-based: `YYYY.MM.DD.HHMM` in local time (America/Los_Angeles),
so every entry is ordered and unambiguous without a manual version bump.

---

## [2026.11.02.1315] — back-tracking storm on partial words, cross-request cancellation

Fixes for a production log on a Gemma-class 31B IT model over vLLM in
`request_mode: stream`, with a **custom** slop-phrase banlist that legitimately
contained the short phrase `la`.

Reported symptoms:

* `Back-tracking @tok=2 orig=' la'` repeating at the *same* token index, each
  followed by a burst of `GenerationState initialised` lines and a
  `Connection pool reset`.
* `Stream closed without [DONE] or finish_reason; reporting
  finish_reason='stream_end'` on requests that had done nothing wrong.
* Garbage output such as `LHNDD la`, `ownset-////`, `LH/RH/H1`.
* A generation that never terminated at all (the test suite hung rather than
  failing).

### Fixed — back-tracking storm: validators fired on half-written words

This is the root cause of the repeated `Back-tracking @tok=2 orig=' la'` lines and
the run of `GenerationState initialised` messages that follow them.

`detect_disallowed_sequence()` treated the **end of the text produced so far** as
a word boundary. During decoding that text is only a prefix, so a ban on `la`
fired the moment the model had emitted `" la"` — even though the next token was
about to extend it to `"land"` or `"last"`. The back-tracker then rewrote the
token, the model re-emitted the same prefix, and the sampler re-entered the same
index indefinitely: no token was ever added, so `max_new_tokens` was never
reached and the outer loop had no exit condition.

A match that ends at the end of the text is now **provisional** while generation
is still running, and is deferred until either more text arrives (which decides
it) or the final sweep runs. A match that already ends in a non-word character
needs no lookahead and still fires immediately.

* `detect_disallowed_sequence()` gained `allow_end_of_text_match` (default `True`,
  so existing callers keep the old behaviour).
* `SlopPhraseValidator.check()` and `NGramValidator.check()` accept a `final`
  keyword. While streaming they pass `False`; the post-generation sweep passes
  `True`.
* The N-gram validator additionally defers a match whose last word is the
  still-growing final token, because `isalpha()` filtering makes a partly written
  word look complete.

To keep the guarantee that a banned phrase at the very end of a response is still
caught, both request modes now run a **final validation sweep** after generation
stops, before the tail is flushed. It repeats (bounded) because fixing one
violation can expose another in the rewritten tail. `_perform_backtrack()` takes
the same `final` flag, so a replacement is not accepted merely because the word
looks unfinished.

### Fixed — cross-request stream cancellation

The cancellation flag lived on the `ApiClient`, and in server mode a single client
is shared by every concurrent request. When request A back-tracked and called
`cancel_current_stream()`, the next chunk of request B's unrelated stream saw the
flag, closed its socket, and surfaced as `finish_reason='stream_end'` — which the
sampler then retried. Under load this produced the `Stream closed without [DONE]`
storm, restarted healthy generations, and (with a mid-stream restart) the
fragmented output above.

* New `api_client.StreamCancelToken`: one handle per stream, created by
  `ApiClient.new_stream_token()` and passed to `generate_stream()` as
  `cancel_token=`. `cancel_stream(token)` aborts exactly that stream.
* A cancelled stream returns without emitting a `stream_end` chunk, so the sampler
  does not treat an intentional abort as a backend event to retry.
* `cancel_current_stream()` is retained for batch mode, where one client serves
  one request, but it is no longer what the sampler calls.

### Fixed — `max_retries_per_position` was read but never enforced

`backtracking.max_retries_per_position` was loaded into an attribute that nothing
consulted, so a position whose entire candidate list was banned could be
re-entered without limit. It is now an actual budget: when it is exhausted the
violation is suppressed and generation moves on.

### Fixed — silent text loss after back-tracking

`_generate_streamwise()` tracks a `last_yield` watermark of already-emitted
tokens. Back-tracking truncates the state, so the watermark could point past the
end of the text; every subsequent safe-prefix slice was then empty and all later
text was dropped silently. The watermark is now clamped after every back-track
and every sweep.

### Fixed — noisy, misleading logs

A sampler is constructed **per request** in server mode, so anything logged from
`__init__` was emitted once per request. The log in the report shows all of these.

* `GenerationState initialised. Prompt length (chars) = …` was logged by every
  throw-away state the back-tracker built to test a candidate token — 20 lines per
  back-track, which buried the real events. New
  `GenerationState.hypothetical()` builds a candidate state without logging and
  without the redundant rebuild; `__init__` gained `log_init`.
* The `chunk_size … ~N sequential upstream round-trips` warning fired even in
  stream mode, because `request_mode` was read *after* the warning. It is now
  resolved first and the warning is chunk-mode only.
* `Tiktoken model '…' not found` is logged once per model name, and
  `tail-prefix filter requested but no chat-template tokenizer available` once per
  process.

### Added — test suite can be excluded on small ARM boards

`tests/_platform.py` detects the host board from `/proc/device-tree/model` (and
`/proc/cpuinfo`) and skips the suite on RK3588-class boards such as the Orange Pi
5 Plus. Detection is by **board**, not by architecture: `aarch64` alone also
covers Graviton, Ampere and Apple silicon.

* `ANTISLOP_SKIP_TESTS=1` skips unconditionally.
* `ANTISLOP_RUN_TESTS=1` runs unconditionally, overriding detection.
* Every test now has a `TEST_TIMEOUT_SEC` (30 s) budget implemented with
  `SIGALRM`, so a regression that makes the sampler loop forever is reported as a
  failure instead of wedging the runner. It deliberately does **not** use
  `multiprocessing`: forking a process that already owns threads and a
  `ProcessPoolExecutor` (the regex validator keeps one) can deadlock in the child,
  which looks exactly like the hang the timeout exists to catch.

### Added

* `tests/test_partial_word_bans.py`: 11 regression tests covering the deferred
  end-of-text match, the final sweep, per-stream cancellation, the back-track
  budget and the silent candidate-state logging.

### Changed — behaviour

* A banned phrase that is only a **prefix of a word still being written** no
  longer triggers back-tracking. It is caught by the final sweep if the word
  really does complete as a banned phrase.
* Short banlist entries such as `la` are now usable in `request_mode: stream`
  without a restart loop. They still match only on word boundaries, so `land` and
  `last` are unaffected.

---

## [2026.11.02.1126] — truncated assistant turns and special-token leakage

Fixes for two reported symptoms on a 2000-line dataset generated against a
Gemma-class 31B IT model over vLLM, using `request_mode: stream`:

* ~700 records had the assistant turn ending abruptly, usually 2–6 words before a
  complete sentence.
* Control tokens such as `<turn|>` appeared in the dataset even though
  end-of-turn handling already existed.

Both symptoms had the same underlying cause, plus several independent contributing
bugs. Details below.

### Fixed — truncation: the early-end-of-turn feature was dead code

`ApiAntiSlopSampler._handle_stop()` only treated a backend `stop` as *premature*
when a marker token had been stripped from `logprobs.tokens`. vLLM detokenises the
output, strips the stop token, and drops the matching logprob entry, so a real
premature `<end_of_turn>` arrives as a plain `finish_reason="stop"` with an empty
token list. The marker branch was therefore unreachable and every premature EOS fell
through to "genuinely terminal" — truncating the answer. At temperature 1.0 with a
roughly 20% early-EOS rate this accounts for the observed ~35% of records.

`_handle_stop()` now decides on evidence the backend actually reports:

| Condition | Result |
|---|---|
| no `finish_reason` | not a stop |
| `length` | keep generating (per-request budget only) |
| `error` / `timeout` | transport failure, retried by the caller |
| a configured `stop_sequence` appears in the attempt text | terminal |
| attempt text ends on a natural boundary | terminal |
| stop in the middle of a sentence | **premature, keep generating** |

The strike counter also changed meaning. It previously reset only when a stop was
*accepted*, so a model that wrote hundreds of good tokens between two stray EOS
emissions still burned the 3-strike valve and was cut off on the third. It now counts
*consecutive no-progress* attempts and resets whenever an attempt appends tokens.
The anti-infinite-loop guarantee is preserved; the false truncation is not.

### Fixed — truncation: no recovery from a dropped stream

A mid-stream timeout, connection reset, or malformed event propagated out of
`_generate_streamwise()` and the partial text was written to the dataset. This was a
second, independent truncation source that looks identical to the first to a user.
Stream mode now retries from the text produced so far, and `error` / `timeout` are no
longer treated as a completed generation.

### Fixed — `<turn|>` and other control tokens leaking

Four independent holes, all closed:

1. `_sanitize_chunk_eos()` only stripped markers at the **tail** of a chunk, so
   `["text", "<turn|>", "\n"]` kept the marker. Removal now happens at every
   position, with the per-token logprob map re-indexed so back-tracking still sees
   the correct alternatives.
2. Markers **split across two SSE events** (`"<tur"` + `"n|>"`) never matched. A new
   `SpecialTokenScrubber` holds back the last N characters until they can no longer
   be the start of a marker.
3. The SSE `text` delta was parsed and then **discarded** — only
   `logprobs.tokens` was used, so text present in one but not the other was
   silently lost. The delta is now used as a fallback.
4. Nothing removed special-token-shaped strings from the accumulated text, so any
   marker outside the hard-coded list passed straight through. A configurable regex
   now covers the Gemma, Llama-3, ChatML, Mistral, and Qwen families, so unknown
   markers such as `<start_of_turn>` and `<|im_end|>` are also removed.

### Fixed — SSE stream reader

* `finish_reason` was **overwritten with `None`** on every ordinary chunk, losing
  the reason whenever it arrived on a text-free chunk. Now latched.
* `iter_lines(decode_unicode=True)` mangles a multi-byte UTF-8 character split
  across two TCP packets. The resulting JSON error was handled by *skipping the
  line*, silently dropping tokens mid-response. The reader now consumes raw bytes
  and decodes whole lines.
* `data:` parsing assumed a single space after the colon, ignored `:` comment
  lines, and did not join multi-line `data:` fields. All handled per the SSE spec.
* A malformed event is now a retryable error instead of a silent `continue`.
* The generator could end with **no terminal chunk**, leaving the sampler's outer
  loop with no exit condition. It now always emits exactly one terminal chunk.
* `bad_words` token suppression is sent on restart requests and disabled
  automatically if the backend rejects it, rather than failing the generation.

### Fixed — generation loop could spin forever

* Stream mode had **no progress guard**: a backend that closed the socket without
  `[DONE]` left both `natural_end` and `restart_stream` false while the token
  count never grew. Added a no-progress counter and a `max_stream_attempts` cap.
* Chunk mode ended the generation silently on an empty chunk with no
  `finish_reason`. It is now retried, bounded by the same cap.

### Fixed — API server

* `finish_reason` was **hard-coded to `"stop"`**, so a truncated (`length`) or
  failed (`error`) generation was indistinguishable from a complete one. The server
  now reports the sampler's real reason. This is what makes a truncation-repair
  pass unnecessary on the server path.
* A client-supplied `stop` was **dropped entirely**, so client-side stop
  sequences did nothing. It is now merged with the configured ones.
* No SSE support: an OpenAI SDK client sending `stream: true` received a
  non-stream JSON body. The server now emits proper
  `chat.completion.chunk` events with a real `[DONE]` and `usage`.

### Fixed — configuration

* `--request-mode stream` was **silently ignored**. It was written to a top-level
  key, but the sampler reads `generation_params.request_mode`, so anyone who
  switched to stream mode on the CLI stayed in chunk mode. Fixed, with migration
  for legacy top-level keys.
* `merge_configs()` used `base_cfg.copy()`, a **shallow** copy, so
  `setdefault("generation_params", {})` returned the caller's nested dict and every
  merge mutated the caller's config in place. Now `copy.deepcopy`.

### Fixed — backtracking

* A hard-coded debug write to `/workspace/auto-antislop/fails.txt` raised
  `FileNotFoundError` **inside error handling**, killing the generation on any
  machine without that directory. Now opt-in via `debug_fail_dump` and written
  relative to the working directory.

### Fixed — performance and correctness

* `_probe_stream_logprobs()` ran a **real backend round-trip per sampler**, and a
  sampler is constructed per request in server mode, so every request paid an extra
  probe. The result is now cached per endpoint, switchable via
  `probe_stream_logprobs`, and the probe generator is closed (it previously leaked
  a pooled connection each time).
* `utils/sampler_helpers.py` imported `torch` at module scope although
  `apply_sampling_filters()` — the only function the sampler calls — is pure Python.
  This made the package unimportable without torch. Now imported lazily.
* Removed `print()` calls on the first-request code path (the banned-prefix
  builder) in favour of `logger.info`.

### Added

* `state/generation_state.py`: `SpecialTokenScrubber`, a text-level control-token
  remover with a hold-back buffer, plus `GenerationState.rebuild_decoded_buffer()`.
* `api_client/api_client.py`: `_iter_sse_data()` and `_build_stream_payload()`
  as separately testable units.
* `api_server/server.py`: `_stream_generation()` and `_sse_event()`;
  `_run_generation_for_request()` accepts an `on_text` callback.
* New configuration keys, all documented in `config-example.yaml` and `README.md`:
  `stop_on_clean_end`, `clean_end_chars`, `max_stream_attempts`,
  `restart_suppress_markers`, `special_token_pattern`,
  `probe_stream_logprobs`, `debug_fail_dump`.
* `tests/`: `_stubs.py`, `test_early_eos.py` (15 tests),
  `test_config_and_sse.py` (6 tests). The stubs fake `tiktoken`,
  `transformers`, and `nltk` so the suite runs on a CPU-only box without them.
* `PLAN.md`: design record, including the list of remaining known bugs.

### Changed — behaviour

* `request_mode` now defaults to `stream` in `config-example.yaml`; the
  "stream might be buggy" caveat is removed.
* A `stop` landing on a natural text boundary is now terminal. Previously *any*
  `stop` was terminal. The residual risk of this change is running *past* the end of
  a turn, not truncating it; set `stop_on_clean_end: false` to remove even that.
* `finish_reason` returned by the API server is now truthful. Clients that
  assumed `stop` unconditionally will now also see `length` and `error`.

### Known remaining issues

See the "REMAINING WORK" section of `PLAN.md`. Most significant:

* Batch mode (`--output-jsonl`) does not copy `sampler.finish_reason` into the
  output record, so truncated rows still require post-processing on that path.
* The API server still ignores `max_completion_tokens`, `n`, `seed`,
  `frequency_penalty`, `presence_penalty`, and `stream_options`.
* `/v1/completions` and `/v1/models` are not implemented.
