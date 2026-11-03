# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versions are date-based: `YYYY.MM.DD.HHMM` in local time (America/Los_Angeles),
so every entry is ordered and unambiguous without a manual version bump.

---

## [2026.11.03.0025] — remove the duplicated `skip` block and `_APOSTROPHES` table

Follow-up to `2026.11.02.2357`.  The audit's patch script was re-run after partial
failures, and its re-run guard could not see that two replacements had already been
applied — each replacement's new text contains the text it replaced, so the guard
always matched.  Both duplications were redundant rather than wrong, and neither
changed behaviour: the suite is green with and without this commit (24/24 in
`tests/test_audit_fixes.py`).

### Fixed — duplicated code left behind by the audit patch

* `utils/slop_helpers.py`: the `skip` check in `detect_disallowed_sequence()` was
  pasted **three** times.  Two copies are removed; the one that remains is in the
  position the check needs to be, before the word-boundary tests and the early
  `return`.
* `validators/regex_validator.py`: the module-level `_APOSTROPHES` translation table
  was defined **twice**, back to back.  One definition remains.

### Note — what the duplication did and did not cost

No behaviour, configuration or API change: no match was missed, no match fired twice
at the boundary tests, and nothing in this entry affects how a response is generated or
validated.

It was not entirely free, though.  Because each copy ran before the boundary tests and
the early `return`, the callback was invoked once per copy on every candidate that
reached that point.  Measured directly against the same input (`"the pilgrim walked
on"`, phrase `grim`, a `skip` that suppresses nothing): `12648ff` called `skip()`
**3 times**, this commit calls it **once**.  `SlopPhraseValidator` passes a lambda that
resolves a character offset to a token index and consults `_ignored`, so the cost was
paid on the scanning hot path for every candidate phrase in the window.

This is worth recording anyway, because a duplicated `skip` check is the kind of thing
that reads as intentional and survives review — and because the patch-script re-run is
the mechanism that produced it.

---

## [2026.11.02.2357] — stop policy, back-track splicing and control-token leaks (audit of `aba6cfb`)

An A/B report against `dev @ aba6cfb` (287 conversations) found degenerate repetition
loops, `word.word`-style joins, longer tails than the same client produced against
vLLM directly, and `of course` leaking into user turns.  Every item below was
reproduced from that report before being fixed.  The new suite is
`tests/test_audit_fixes.py` (24 tests): 22 of them fail on `aba6cfb`, and the suite
passes 24/24 on this commit.  The two that already passed there are the negative
cases — `test_a_phrase_the_model_merely_likes_stays_banned` and
`test_the_guard_never_disables_a_regex_rule` — which assert that the forced-phrase
guard does *not* fire, and so hold trivially when the guard does not exist yet.

### Fixed — one refused stop disabled the end-of-turn marker for the whole run

`_generate_streamwise` decided whether to send the markers as `bad_words` with
`attempts > 1`.  That stays true for the rest of the generation after a single
refused stop **or a single dropped connection**, so from then on the model could
not end its turn at all and wrote until `max_tokens`.  This is the repetition loop
and the long tail in the dataset: the reply was finished, the stop was refused once,
and everything after that was padding.

* The ban now applies only to the one attempt that directly follows a refused stop
  (`suppress_next`, set from the stop reason), and that attempt is capped at the new
  `restart_suppress_window` (default 32 tokens).
* `restart_suppress_markers` now defaults to **`false`**.  `bad_words` holds for the
  whole request, so on the banned attempt the model cannot end its turn even when it
  meant to; a model that really wants to stop simply re-emits its marker, which
  counts as a no-progress stop and ends the turn after `max_consecutive_stops`.
* A `length` finish (including the capped attempt itself) also leads to a
  continuation, but the model did not try to end there, so it no longer sets the ban.

### Fixed — roleplay endings were refused as mid-sentence stops

The default `clean_end_chars` only listed prose punctuation, so a reply that ended on
an `*action*`, a trailing `~`, a dash, a colon or an emoji was judged unfinished, its
stop was refused, and the model was asked to keep writing.

* The default now also accepts `*`, `_`, `~`, `—`, `–`, `:`, `;`, `»`, a backtick,
  the typographic quotes `”`/`’`, and a trailing character in Unicode category `So`,
  `Sk` or `Sm` (an emoji or other symbol), skipping a variation selector or ZWJ first.
* In stream mode the clean-end test now looks at the **whole reply**
  (`state.get_generated_text()`) instead of only the current attempt.  Judging the
  attempt alone made an immediate stop on a retry — an empty attempt — look
  mid-sentence, so a finished reply's stop was refused again.

### Fixed — a post-generation back-track spliced two continuations together

`2026.11.02.1930` made the final sweep rewrite a token in place instead of deleting
the tail, which fixed truncation but produced `It was not fear` → `It was. fear` and
turned every n-gram fix into `took deep breath`.  Neither the splice nor the
truncation is acceptable in a dataset.

* End-of-text violations are now resolved **before a stop is accepted**, by the new
  `_resolve_before_stop()`, while a back-track can still be followed by a
  regeneration.  Both generation loops call it and, when it rewrote a token, resume
  generation from there instead of ending the turn.
* `_perform_backtrack(final=True)` no longer rewrites anything.  A violation that is
  still present once generation is over is logged and left as written, because no
  replacement exists that is consistent with the text around it.
* A back-track now takes `emitted_upto`, the last token already sent to the client.
  Text that has been yielded cannot be taken back, so it is never rolled back into.
* This supersedes the "the sweep keeps the tail" behaviour described under
  `2026.11.02.1930`: the sweep still does not truncate, but it is no longer the
  mechanism that repairs a turn — the pre-stop resolution is.

### Fixed — a back-track could pick the end-of-turn token as the replacement

vLLM computes logprobs **before** applying `bad_words` and stop handling, so the
candidate list includes `<turn|>` and other control tokens.  Choosing one put
`<turn|>` inside the assistant text, and the next restart sent it upstream as a real
end of turn — the record then continued as a new turn.

* `_is_end_of_turn_token()` candidates are skipped in both the constrained and the
  unconstrained replacement pass.
* `_new_scrubber()` scrubs the model's own control tokens.  The list is read from the
  chat-template tokenizer (`all_special_tokens` plus every `added_tokens_decoder`
  entry with `special=True`); the built-in pattern named a fixed set of families and
  missed 13 of Gemma 4's 24 (`<|channel>`, `<channel|>`, `<|think|>`, `<|"|>`, …)
  while deleting harmless prose such as a literal `<mask>`.  When a tokenizer is
  known its list replaces the pattern entirely.
* The fallback pattern in `state/generation_state.py` gained the Gemma 4 family
  (`channel`, `think`, `tool`, `image`, `audio`, `video`, `"`) for the no-tokenizer
  case.

### Fixed — a phrase the context forces caused a junk loop

A character called Mara against the one-word entry `mara`, a place name, or a word the
user just used cannot be avoided: each replacement is a junk token and the model
writes the word again straight after it.  A real Gemma 4 run filled the reply with
`**` followed by a run of `\n` junk tokens (bans at tokens 22, 23, 25).  The
per-position budget never trips because every retry lands on a new index.

* `_ban_history` records the token index of every ban per `(validator, rule)`.  Three
  bans of the same phrase at increasing positions, each at most 2 tokens after the
  previous one, is the junk-loop signature: the back-track rolls back to the first of
  the three and allows that phrase for the rest of the request.
* The guard is deliberately narrow.  A model that merely *likes* a word returns to it
  in a new sentence (`flicker` at 87, 93, 96) and stays banned, and regex rules are
  excluded because their key is a whole pattern rather than a phrase.
* It never rolls back into text already sent to the client (`emitted_upto`).
* Whole-request suppression is the `(None, key)` form in `BaseValidator._is_ignored`,
  which all three validators now honour.

### Fixed — back-tracking resampled under different sampling parameters

`_perform_backtrack` draws from cached logprobs using `self.temperature`, `min_p`,
`top_p` and `top_k`, and those kept the **config-file** values.  A client that sends
its own settings (SDG does) therefore got replacement tokens drawn under different
sampling than the rest of the reply — the "B != A with the same client settings"
report.

* The four attributes are updated from the effective per-request values at the top of
  each generation.

### Fixed — suppressions outlived the text they referred to

* `_perform_backtrack` now drops `_ignored` entries for token indices past the
  rewritten one.  Left in place, a regenerated occurrence of the same phrase landing
  on the same index was let through silently.

### Fixed — a suppressed match hid every later one

All three validators returned the **earliest** match unconditionally, so once one
match at that position was suppressed, later violations in the same window were never
reported.

* `detect_disallowed_sequence()` takes a `skip` callback and keeps scanning past a
  match it has already suppressed.
* `SlopPhraseValidator` passes `skip=`, `RegexValidator`'s worker returns up to 32
  matches (`finditer`) instead of the first, and `NGramValidator` builds a candidate
  list, sorts it (earliest first, longer n-gram on a tie) and takes the first
  non-suppressed entry.

### Fixed — `_is_repetition` compared characters, not words

`needle in " ".join(existing_words)` matched `he said` inside `she said`, so a genuine
continuation was rolled back as a repetition.

* Both sides are normalised to `" " + " ".join(text.lower().split()) + " "` before the
  containment test.

### Fixed — the n-gram validator never fired while streaming

The deferral test was `last_end >= len(text.rstrip()) and last_start >= violation_char_pos`,
which is true for practically any text, so every n-gram was deferred and only ever
caught after generation ended.

* The deferral now applies only when the violating n-gram's **last** word is the
  still-growing final word of the text (`raw_idx_of_last_word == last_raw_idx` and
  `last_end >= text_end`).
* `word_str.rstrip(".")` is applied before the `isalpha()` filter.  Treebank only
  splits off a sentence-final period, so a word before a mid-text period arrived as
  `eyes.`, failed `isalpha()`, and vanished — breaking any n-gram that ended a sentence.
* `_ensure_nltk_data()` catches `LookupError` only.  `nltk.downloader.DownloadError`
  does not exist in current nltk, so naming it raised `AttributeError` **while
  handling** the `LookupError` and importing `validators` failed on any machine
  without the data.  `punkt_tab` is fetched alongside `punkt`.

### Fixed — the slop scan window started mid-word

A window that began mid-word treated its first character as a boundary, so single-word
entries fired inside longer words as the window slid (`grim` in `pilgrim`, `aria` in
`Maria`) and each false hit discarded ~50 good tokens.

* The window start is walked back to a word boundary before scanning.

### Fixed — typographic apostrophes never matched

The ban lists are written with ASCII `'`, so `couldn’t help but` (U+2019) never
matched `couldn't help but`.

* `utils/slop_helpers.fold_for_matching()` folds `’`, `‘`, `ʼ`, `“`, `”` to ASCII and
  lower-cases **without changing the length** (`str.lower()` can lengthen a string —
  `İ` becomes two characters — which shifted every later offset and blamed the wrong
  token).  `SlopPhraseValidator` folds its keys the same way at construction.
* `RegexValidator` folds the apostrophes in the text it matches, one character for one
  character, so `it'?s` also matches `it’s`.

### Fixed — configuration keys read from the wrong level

* `--force-backtrack` was written to a top-level key nobody read, so the flag did
  nothing.  `merge_configs()` now writes `backtracking.force_backtrack`.
* `match_requires_word_boundaries` is documented under `generation_params` but was
  read only at the top level.  It is now read from there first, with the top level as
  a fallback.
* `top_k: -1` and `top_k: 0` mean *disabled* in vLLM.  `apply_sampling_filters()`
  sliced with them, dropping the least likely candidate or returning nothing; the
  slice now applies only when `top_k > 0`.

### Fixed — the API server

* **The chat template is required.** Without one the server used to fall back to the
  last user message alone, silently dropping the system prompt and the whole
  conversation history.  It now refuses the request with HTTP 500 and an explanatory
  body.
* **A backend failure is a 502.**  A generation that ended with `finish_reason='error'`
  was returned with HTTP 200, so a client that does not read `finish_reason` stored an
  empty or cut-off turn as a real one.  It now returns 502 with
  `partial_text_chars`.
* **Sampler construction is wrapped.**  The stream-logprobs probe runs in the
  constructor; raised outside a handler it surfaced as an unhandled ASGI exception
  instead of an error body the client can read and retry on.  It is now a 502.
* **`chat_template_kwargs`** from the request are passed through to
  `build_base_prompt_from_messages(..., **template_kwargs)`, so per-request template
  switches (`enable_thinking`, …) are honoured the way vLLM's own chat endpoint
  honours them.  They used to be dropped.
* **`repetition_penalty`, `presence_penalty` and `frequency_penalty`** are forwarded to
  the backend via `sampler.extra_request_params`, which is merged into every stream
  and chunk request.  A client's `repetition_penalty` never reached vLLM through this
  proxy, while it did when the client called vLLM directly.  `seed` is deliberately
  **not** forwarded: every restart re-sends the request, and a fixed seed would replay
  the same continuation.
* **Usage is counted with the model's tokenizer.**  tiktoken's `cl100k` is a different
  vocabulary and misstates usage for any non-OpenAI model; it remains the fallback.
* `generation_params.threads` is now read **before** the top-level `threads`.  The CLI
  always writes a top-level `threads` (default `1`), which used to win and cap the
  server at one concurrent generation.
* `finish_reason` is reported as `error` rather than `length` when the attempt budget
  ran out because the backend kept failing.

### Added

* `_resolve_before_stop()` and `_record_event()` in `core/sampler.py`.
* `self.control_tokens` and `self.extra_request_params` on `ApiAntiSlopSampler`.
* `fold_for_matching()` and the `skip` parameter of `detect_disallowed_sequence()` in
  `utils/slop_helpers.py`.
* `chat_template_kwargs` and the three penalty fields as accepted API-server request
  fields.
* New configuration keys, documented in `config-example.yaml`:
  `generation_params.restart_suppress_window` (default 32), and the expanded
  `generation_params.clean_end_chars` default.
* `tests/test_audit_fixes.py` (24 tests), registered in `tests/run_all.SUITES`.
* `tests/.runlock` is now ignored by git — the run lock is created in the checkout and
  was easy to commit by accident.

### Changed — behaviour

* `restart_suppress_markers` defaults to **`false`** (was `true`).  See above: banning
  the marker for the whole request is what padded finished replies.
* The default `clean_end_chars` accepts roleplay endings and trailing emoji.
* A violation found after generation has ended is **left as written** and logged,
  rather than spliced or truncated.
* A phrase the context forces is allowed for the rest of the request instead of being
  retried into a junk loop.
* `generation_params.threads` now takes precedence over the top-level `threads` in the
  API server.

### Known remaining issues

* `_generate_chunkwise` accepts `emitted_upto` and calls `_resolve_before_stop()`, but
  chunk mode was not exercised by the A/B report; treat stream mode as the tested path.
* `tests/test_audit_fixes.py` imports `api_server.server` in three tests, which needs
  `fastapi` installed.  Without it those three fail on `ModuleNotFoundError`, not on
  an assertion.

---

## [2026.11.02.1930] — the final validation sweep no longer deletes the tail

Reported from a real dataset: the truncation is *less* frequent than before, but a
turn still ends mid-word (`... Her blue expires`).  This one has a different cause
from the two below, and it looks different in the logs: generation finished
normally (`finish_reason='stop'`), the response is complete, and the text simply
stops early.  No attempt budget is involved.

### Fixed — a post-generation repair discarded every token after the repaired one

`_perform_backtrack` commits a replacement by calling `state.truncate(idx + 1)` and
then `replace_token_string(idx, choice)`.  Mid-generation that is correct: the
rewrite invalidates everything after the repaired token, and the caller immediately
asks the backend to regenerate from that point, so nothing is lost.

The **final validation sweep** (`final=True`) runs after generation has stopped and
never calls the backend again.  Truncating there deleted the remainder of the turn
with nothing to replace it, so the record ended at the last repaired word.

This fires constantly against the shipped 3 500-entry ban list.  The sweep exists
precisely for matches that only become definitive once the text stops growing
(``detect_disallowed_sequence(..., allow_end_of_text_match=True)``), so the last
few tokens of a slop-heavy turn are routinely repaired.  The tail that went missing
is whatever the model wrote after the last banned phrase.

* The sweep now rewrites the single token and **keeps** the rest of the state; only
  a live back-track truncates.  A rewritten token can expose a new violation
  further along, which the sweep's next pass already handles.
* `test_final_sweep_keeps_the_text_after_a_repaired_violation` and
  `test_final_sweep_keeps_the_tail_when_several_violations_are_repaired` reproduce
  the truncation (they failed with ``' She felt happy'`` before the fix).
* `test_live_backtrack_still_discards_the_invalidated_tail` is the counterpart:
  removing `truncate` outright would also pass the two tests above, but keeps the
  stale tail *and* appends the regenerated text (``... happy and and cold ...``).

### Note — why the existing sweep coverage did not catch this

`test_truncation_and_repetition` already exercised the sweep, but with
`regex_max_span_tokens: 0`, i.e. `tail_keep_tokens == 0`.  With nothing held back,
all text is emitted before the sweep runs, so a sweep repair cannot be observed in
the returned text at all.  The shipped config uses `regex_max_span_tokens: 999999`
(`tail_keep_tokens` is every token) and *does* hold the tail, which is exactly the
configuration in which the bug is visible.  The new tests set it explicitly.

### Changed — the test harness closes the regex process pool

`RegexValidator` keeps a lazily created `ProcessPoolExecutor` for the life of the
process.  Any suite that exercises the regex path therefore ended with
``WARN  2 thread(s) still alive at exit: QueueFeederThread, Thread-1``, which is
not a bug in the code under test.  `tests/_runner.shutdown_regex_pool()` tears the
pool down before the leak check, so that warning stays meaningful.

---

## [2026.11.02.1800] — stop truncating long turns, stop emitting repetition loops

Reported from a real dataset: assistant turns ended mid-word
(`...the terror`, `...she whim`), and one turn contained `of course.` 327 times.

### Fixed — a back-track consumed the retry budget and truncated the turn

`_generate_streamwise` used one counter, `max_stream_attempts`, for two unrelated
things: a plain retry (rejected stop, dropped connection, empty chunk) and a restart
entered to resume after a back-track.

A back-track is progress by definition: a token was rewritten and generation
resumed. Against the shipped slop list a response routinely needs more than a dozen of
them, so the twelfth one tripped the budget, generation ended, and `finish_reason`
was reported as `length` — even though the backend had more to say. Every truncated
record in the report carries `finish_reason='length'`.

* Restarts after a back-track are now charged to a separate budget,
  `max_backtrack_restarts` (default 200), and refund the retry counter.
* `max_stream_attempts` keeps its original meaning and still bounds a backend that
  never terminates.
* `test_restart_budget_still_bounds_a_backend_that_never_ends` pins that down.

### Fixed — a refused stop that only repeats itself was emitted

After a backend `stop` the sampler rejects the stop (it may be premature) and asks for
a continuation. When the model has nothing left to say it answers with a stock phrase,
and because the rejected stop was reported as `produced_tokens=True` the no-progress
counter was reset on every attempt. The loop therefore ran until the attempt budget was
gone and emitted `of course.` 327 times.

The same reset meant the `max_consecutive_stops` safety valve documented in
`README.md` could never fire against a backend that always emits *something*.

* A continuation that adds nothing new is now recognised by `_is_repetition`: the
  added words either already appear verbatim in the accepted text, or they repeat a
  fragment produced by an earlier refused stop.
* Such a continuation is rolled back and counted as a no-progress attempt, so the
  valve fires after `max_consecutive_stops` repetitions and the turn ends with the
  backend's own `finish_reason`.
* Continuation attempts hold their text back until their stop is seen, because text
  already yielded cannot be taken back. A genuine continuation is flushed when its
  stop is accepted; `test_genuine_continuation_is_still_emitted` guards that path.
* Applied to **both** `_generate_streamwise` and `_generate_chunkwise`. Chunk mode
  re-sends the prompt per chunk, so the loop there is more expensive, not less.
* `_generate_chunkwise` also reset the no-progress counter on every non-empty chunk,
  which made the guard unreachable in that mode; the reset now happens only for a
  chunk that is not a repetition.

### Fixed — a leaked `<tool_call|>` reached the dataset

One turn ends `...She has a distinctive dyed-<tool_call|>`. The default scrubber
pattern in `state/generation_state.py` covered the Gemma / Llama-3 / ChatML / Mistral /
Qwen control tokens but not the tool-call family, so the marker was emitted as prose.
`tool_call`, `tool_response`, `tool_result` and `function_call` are now in the default
pattern.

### Added — `tests/test_truncation_and_repetition.py`

Eight tests, each failing on the old behaviour:

* `test_backtracks_do_not_exhaust_the_restart_budget` — 30 back-tracks, every
  attempt making progress; asserts the turn is not ended as `length` and that the last
  thing the backend said reaches the output.
* `test_restart_budget_still_bounds_a_backend_that_never_ends` — the relaxation
  does not remove the termination guarantee.
* `test_stop_on_clean_end_is_not_a_progress_stop` and
  `test_rejected_stop_that_only_repeats_itself_ends_the_turn` — the `of course.`
  loop, bounded to one emitted repeat.
* `test_repeated_continuation_is_dropped_from_the_output` — a verbatim repeat of an
  existing sentence is dropped entirely.
* `test_chunk_mode_repeated_continuation_ends_the_turn` — the same guard in chunk
  mode.
* `test_genuine_continuation_is_still_emitted` — a real continuation survives.
* `test_tool_call_marker_is_scrubbed_from_the_text` — the leaked marker.

Registered in `run_all.SUITES`.

### Changed — configuration

* `generation_params.max_backtrack_restarts` (default 200) is new; documented in
  `config-example.yaml` and `README.md`.
* `max_stream_attempts` is documented as bounding retries only.
* `max_consecutive_stops` is documented as also ending a repetition loop.

---

## [2026.11.02.1645] — fix back-track ping-pong emitting banned words and repeated text

### Fixed — a banned word was accepted as its own replacement

`_check_hypothetical_state` validated each candidate token with `final=False`.
The validators defer any match whose right edge is the end of the text while
generation is still running, because the next token may extend the word
(`" la"` → `" land"`). A back-track candidate is not that situation: the next
validation pass sees exactly the same text again, so a candidate that already
ends on a banned phrase is the same violation one token later, not a fix.

The effect with a custom banlist containing both forms of a word was a
ping-pong: `gripped` was resampled to `gripping` (also banned), re-detected at
the same index, resampled back to `gripped`, and so on until
`max_stream_attempts` was exhausted. The same path produced the
`spark` → `surge` → `spark` cycle and, once the budget ran out, emitted the
banned word anyway — the repeated `own own own` text and the duplicated
`Lorena felt a spark` prefixes in the report.

* Candidates are now checked with `final=True`, so a banned candidate is
  rejected rather than deferred.
* A position is therefore fixed in one pass instead of being re-entered until
  the attempt budget is gone. `finish_reason` is no longer reported as
  `length` for a run that actually finished.
* The streaming deferral itself is unchanged: `" la"` still grows into
  `" land"` without a ban, and the final sweep still catches a banned phrase at
  the very end of a response.
* `final` is still accepted by `_check_hypothetical_state` for call-site
  compatibility, but no longer changes the verdict.

### Added — regression tests

`tests/test_partial_word_bans.py` gained two tests that fail on the old
behaviour:

* `test_backtrack_never_selects_another_banned_word` — `gripped`/`gripping`
  both banned and each the other's top alternative; asserts neither appears in
  the output, one violation event, and no `length` finish.
* `test_backtrack_does_not_repeat_the_same_banned_word` — the
  `spark`/`surge` pair from the log.

`README.md` documents that replacement candidates are validated definitively.

---

## [2026.11.02.1548] — add `ngram_validator.enabled` to switch n-gram banning off

### Added — `ngram_validator.enabled` config switch

N-gram banning had no off switch. The only way to stop it was to empty
`banned_file` / `banned_list`, which is not a stable "off": the ban file is re-read
on every run, so the validator silently came back as soon as the section was
configured again.

* `ngram_validator.enabled: false` in `config.yaml` now skips the validator
  completely. The ban file is not read and no `NGramValidator` is constructed, even
  when `banned_file` / `banned_list` are present.
* Default is `true`, so existing configs without the key behave exactly as before.
* The value is coerced with the existing `_coerce_bool` helper, so a quoted YAML
  `"false"` is treated as false rather than as a non-empty string.
* Implemented in `main._setup_validators`, which the single-prompt, batch and API
  server paths all share.
* Documented in `config-example.yaml` and `README.md`.

### Added — `tests/test_ngram_toggle.py`

Covers `enabled: false` (dict and quoted-string form), `enabled: false` overriding a
configured `banned_list`, the default-on path, and the missing-key default. Registered
in `run_all.SUITES`. `tests/_stubs.py` gained a `tqdm` stub so `main.py` can be
imported by a suite.

---

## [2026.11.02.1530] — tests: fix the excluded sampler-filter suite, add a run lock

### Fixed — `tests/test_sampler_filters.py` was broken and excluded from the suite

The suite failed with ``AttributeError: 'ApiAntiSlopSampler' object has no attribute
'max_retries_per_position'``. The cause was the test, not the sampler: it built the
sampler with ``object.__new__(ApiAntiSlopSampler)`` and a hand-written list of
attributes, so it bypassed ``__init__``. Every attribute the sampler later gained was
missing from that list, and the test failed on an ``AttributeError`` that said nothing
about the behaviour it was supposed to check.

* The test now builds the sampler through the real constructor, so a new required
attribute is picked up for free and a real misconfiguration surfaces as itself.
* Its candidate-check stub now accepts the ``final`` keyword, which the sampler passes
  through to distinguish a mid-stream candidate from a final-sweep one.
* Added ``test_backtrack_stops_at_the_configured_budget``: a position whose candidates
  are all rejected must be ended by ``max_retries_per_position``, not re-entered
  without limit. This is the second half of the back-tracking storm.
* Registered in ``run_all.SUITES``; the suite is no longer excluded.

### Added — an exclusive run lock for shared checkouts

Several agents may share this checkout, and nothing here is isolated per agent. Two
concurrent runs interleaved their output and competed for the same four cores, which is
exactly what the single-threaded, short-timer design exists to prevent.

* ``tests/run_all.py`` takes an exclusive ``flock`` on ``tests/.runlock`` for the
  duration of the run.
* A second concurrent run refuses to start and exits ``2``, with a message naming the
  lock, rather than queueing silently.
* ``flock`` rather than a plain lockfile is deliberate: the kernel drops it when the
  holder dies, including on ``SIGKILL``, so a killed run cannot wedge every later run
  in the checkout.

### Added

* ``AGENTS.md``: the test-first (red, green, refactor) workflow, the single-thread /
  timer / abort rules, the concurrency rules for agents sharing a checkout, and the
  definition of done.

## [2026.11.02.1521] — tests: single-threaded, short timers, abortable, no board exclusion

The previous entry made the suite skip itself on RK3588-class boards because a
non-terminating test pegged a core with no way to stop it. That is a workaround, not a
fix: the board could not run its own tests. This replaces it with containment, and the
Orange Pi 5 Plus runs the suite again.

### Changed — the suite is now single-threaded

* `tests/run_all.py` runs the suites **one at a time**, each in its own process group,
  and tests run one after another in the main thread. There is no parallelism to
  oversubscribe a 4-core board with.
* `tests/_runner.py` pins `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and
  `TOKENIZERS_PARALLELISM=false` at import, before `core.sampler` is imported.
* `RegexValidator`'s `ProcessPoolExecutor` is now created **lazily** on first `check()`
  instead of forking 16 processes at import, and its default worker count is capped at
  the CPU count (`ANTISLOP_REGEX_MAX_WORKERS` overrides; the harness pins it to 1).
  Importing `validators` no longer starts a single process.
* A suite reports any thread still alive when it finishes.

### Changed — short timers, and a hard kill behind them

* Per-test budget is now `ANTISLOP_TEST_TIMEOUT`, default **10 s** (was 30), enforced
  in-process with `SIGALRM`.
* Per-suite budget is `ANTISLOP_SUITE_TIMEOUT`, default **60 s**, enforced by
  `run_all.py`, which `SIGTERM`s and then `SIGKILL`s the suite's whole **process
  group**. `SIGALRM` cannot interrupt a blocking C call or a catastrophic regex, so a
  kill is the only thing that reliably reclaims the machine; killing the group also
  takes out the validator's pool workers, which would otherwise be orphaned and keep
  burning cores.
* The full suite runs in about 2 s on an Orange Pi 5 Plus, so a test that reaches its
  budget is a loop that does not terminate, not a slow machine.

### Added — a run can be aborted at any time

* `touch tests/.abort` (or `ANTISLOP_TEST_ABORT_FILE=<path>`) stops the run; the
  sentinel is polled every 50 ms *inside* each test, not just between them, so a hung
  test does not have to reach its budget first.
* `Ctrl-C`, `SIGTERM` and `SIGHUP` to the runner do the same, and take the child
  suite down with them.
* An aborted run prints `ABORTED` with the reason and exits `130`; `run_all` removes a
  stale abort file on the next start.

### Removed — the Orange Pi / RK3588 exclusion

* `tests/_platform.py` is gone, along with `ANTISLOP_SKIP_TESTS` and
  `ANTISLOP_RUN_TESTS`: there is no board detection and nothing is skipped by
  hardware. The suite no longer needs an override to run on the board it was written for.

### Added

* `tests/test_harness.py`: 9 tests for the harness itself — the timeout fires on a
  non-terminating loop, the abort file is noticed mid-test, the thread pools are pinned,
  the regex pool stays lazy and single-worker, and importing the sampler starts no
  threads.

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

> Superseded by [2026.11.02.1521], which replaced the exclusion with a
> single-threaded, timed, abortable runner. `tests/_platform.py` no longer exists.

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
