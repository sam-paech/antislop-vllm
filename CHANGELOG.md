# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versions are date-based: `YYYY.MM.DD.HHMM` in local time (America/Los_Angeles),
so every entry is ordered and unambiguous without a manual version bump.

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
