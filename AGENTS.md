# AGENTS.md

Instructions for AI agents and humans working in this repository. Read this before
changing code.

`antislop-vllm` is a sampling library that sits in front of an OpenAI-compatible
endpoint. It generates text, validates it against ban lists, and back-tracks when a
validator fires. **Several code paths here have historically failed to terminate**, so
tests are the primary safety mechanism in this repo, not an afterthought.

---

## 1. The workflow: red, green, refactor

Every behavioural change follows this loop. Do not skip a step, and do not write the
fix before the failing test.

### 1.1 Red — write the failing test first

1. Write a test that encodes the behaviour you want and **fails for the right reason**.
2. Run it and **read the failure**. It must fail on your assertion, not on an
   `ImportError`, a typo, or a missing attribute you forgot to set up. A test that
   fails to *import* tells you nothing about the behaviour under test.
3. Paste or summarise the actual failure in your report. "It fails" is not enough;
   say what it said.

### 1.2 Green — make it pass, minimally

1. Change production code, not the test, to make it pass. If you must change the
   test, say why in the commit message; changing an assertion to match buggy
   output is not a fix.
2. Re-run **the single suite** while iterating:
   `python -m tests.<suite>`
3. Re-run **the whole suite** before declaring done: `python -m tests.run_all`.
   A green single suite next to a red full run is not done.

### 1.3 Refactor — clean up with the tests green

1. Refactor with the suite green, re-running it after each step.
2. Keep the test suite green at every commit.

### 1.4 Never leave a test red and unreported

If you cannot fix a test, leave it **excluded from `run_all.SUITES`** and record it
under [Known broken tests](#6-known-broken-tests). A test that is red, in the suite,
and unexplained is worse than no test: it trains everyone to ignore the suite.

---

## 2. Running the tests

```bash
python -m tests.run_all                  # every suite, one at a time (use this)
python -m tests.run_all test_early_eos     # a subset, by name
python -m tests.run_all --list             # what would run
python -m tests.test_early_eos             # one suite, while iterating
```

Prefer `run_all`. It is the only entry point that enforces the hard kill
(Section 3.3) and the run lock (Section 4).

Exit codes: `0` all passed, `1` failures or a kill, `2` another run holds the
lock, `130` aborted.

---

## 3. Non-negotiable: the tests must not lock up the machine

This repo is developed on an **Orange Pi 5 Plus** (RK3588, 4× Cortex-A76 + 4×
Cortex-A55, 4 GB–16 GB RAM). Tests have previously hung this board. The harness in
`tests/_runner.py` exists to make that impossible; do not weaken it.

### 3.1 Single-threaded

* Suites run **one at a time**; tests run **one at a time**, in the main thread.
  There is no parallelism in the harness, and adding any is a regression.
* `tests/_runner.py` pins `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` to `1` and
  `TOKENIZERS_PARALLELISM=false` at import. It must be imported **before**
  `core.sampler`.
* `RegexValidator`'s `ProcessPoolExecutor` is **lazy** and capped at the CPU count
  (`ANTISLOP_REGEX_MAX_WORKERS`); the harness pins it to 1. Never create a process
  pool at import time again — importing `validators` used to fork 16 processes.
* A suite prints a `WARN` if a thread is still alive when it finishes. Treat a new
  warning as a leak you introduced. `run_suite` first calls
  `shutdown_regex_pool()`, which closes `RegexValidator`'s lazily created
  `ProcessPoolExecutor`; without it, any suite that exercises the regex path ends
  with a spurious `QueueFeederThread` warning.

### 3.2 Short timers, and a hard kill behind them

* Per-test: `ANTISLOP_TEST_TIMEOUT`, default **10 s**, in-process `SIGALRM`.
* Per-suite: `ANTISLOP_SUITE_TIMEOUT`, default **60 s**, enforced by `run_all.py`,
  which `SIGTERM`s then `SIGKILL`s the suite's **whole process group**.
* `SIGALRM` cannot interrupt a blocking C call or a catastrophic regex. The kill is
  not redundant with the timer; it is the only thing that reliably reclaims the board.
  Killing the **group** is what reaps the validator's pool workers, which would
  otherwise survive as orphans and keep burning cores.
* The whole suite runs in about 2 s. A test that reaches its budget is a loop that
  does not terminate, not a slow machine. Do not raise a budget to make a test pass.

### 3.3 Abortable

* `touch tests/.abort` stops the run; the sentinel is polled every 50 ms **inside**
  each test, so a hung test does not have to reach its budget first.
* `Ctrl-C`, `SIGTERM` and `SIGHUP` to the runner do the same.
* An aborted run exits `130`. A stale abort file is removed on the next start.

### 3.4 Writing a test

* `test_*` functions in a `tests/test_*.py` module, plus a `main()` returning
  `_runner.run_suite("<name>", globals())`. No pytest, no network, no GPU: stub the
  backend (`tests/_stubs.py`) and script a fake client.
* Assert on **observable behaviour** (returned text, call count, exit condition), not
  on log strings, unless the logging *is* the behaviour under test.
* If a test exercises a path that used to hang, say so in the docstring. That is why
  the budgets exist.
* Do not let a test mutate shared state that the harness itself polls — in particular,
  do not write `tests/.abort`. Use `_runner.set_abort_file(<tmp>)` and restore it, or
  the watchdog will (correctly) kill your run.

---

## 4. Sub-agents and concurrent work

**Assume other agents are working in this repository at the same time.** Several
sessions may share this checkout. Nothing here is isolated per agent.

* **Take the run lock before running the suite.** `run_all` holds
  `tests/.runlock` (an `flock`) for the duration. A second concurrent run refuses to
  start instead of interleaving output and competing for the same four cores.
  Do not delete `tests/.runlock` while a run is in progress; `flock` releases it on
  exit, including on `SIGKILL`.
* **Never `git checkout`, `git reset`, `git stash` or `git clean`** to undo someone
  else's work, and never revert a file you did not edit. If the tree is dirty with
  changes you do not recognise, leave them alone and say so.
* **Do not rewrite a whole file** to change part of it. Use a targeted replacement so
  a concurrent edit elsewhere in the file is not silently discarded.
* **Re-read before you edit.** Another agent may have changed the file since you read
  it. If a replacement target no longer matches, re-read rather than force it.
* **Do not create a second test harness, runner or `.abort`/`.runlock` path.** Extend
  `tests/_runner.py` and `tests/run_all.py`.
* **Do not commit someone else's uncommitted changes.** Stage only the files you
  edited.
* `git status` before and after your work; report anything you did not touch.

---

## 5. Definition of done

A change is not finished until **all** of these are true:

1. The new behaviour has a test that failed first and passes now.
2. `python -m tests.run_all` is green, with no `WARN` about leaked threads.
3. The suite is registered in `tests/run_all.SUITES`.
4. `README.md` reflects any change to how tests are run, configured or aborted.
5. `CHANGELOG.md` has an entry under the current dated heading, saying what
   was added, changed and removed.
6. `AGENTS.md` (this file) is updated if you changed a rule, a budget, an entry
   point or a documented convention.
7. No stray files: no scratch test modules, no `.abort`, no `__pycache__`, no
   `timings_antislop_*.csv` in the tree.

---

## 6. Known broken tests

Excluded from `run_all.SUITES`; fix these, do not delete them.

* **None.** `tests/test_sampler_filters.py` was excluded because it built the
  sampler with `object.__new__` and predated `max_retries_per_position`; it now uses
  the real constructor and is in the suite, with two tests: the min-p floor is
  measured before the banned top token is removed, and the per-position budget ends
  an unfixable position.

`tests/test_truncation_and_repetition.py` covers the three ways a record used to be cut
off or padded. All were reproduced from a real dataset before being fixed, so keep the
tests narrow: a "repetition" that is really a continuation must survive
(`test_genuine_continuation_is_still_emitted`), the attempt budget must still
terminate a backend that never stops
(`test_restart_budget_still_bounds_a_backend_that_never_ends`), and a *live*
back-track must still discard the tail it invalidated
(`test_live_backtrack_still_discards_the_invalidated_tail`). Loosening
`_is_repetition`, the budgets, or the `truncate` in `_perform_backtrack` to make one
of these pass is a regression in another.

When a test needs a violation to be caught by the **final sweep**, set
`regex_max_span_tokens` high as well as `regex_validation_interval`. With the default
`tail_keep_tokens` of 0 the whole response is emitted before the sweep runs, so a
sweep repair cannot be observed in the returned text.

---

## 7. Layout

| Path | Purpose |
|---|---|
| `core/sampler.py` | `ApiAntiSlopSampler`: generation, validation, back-tracking; `_is_repetition` guards against a refused stop whose continuation only repeats text |
| `core/models.py` | API result and violation dataclasses |
| `state/generation_state.py` | The text being generated; `hypothetical()` builds candidate states |
| `validators/` | `SlopPhraseValidator`, `RegexValidator`, `NGramValidator` (the last is switchable with `ngram_validator.enabled: false`) |
| `utils/` | Config, string, chat-template, refusal, slop helpers |
| `api_client/` | `ApiClient` for OpenAI-compatible endpoints; SSE parsing |
| `api_server/` | FastAPI wrapper |
| `tests/_runner.py` | Single-thread setup, per-test timer, abort check |
| `tests/run_all.py` | Sequential runner, run lock, hard-kill watchdog |
| `tests/_stubs.py` | Import-time stubs for torch/transformers/tiktoken/nltk |
| `tests/test_*.py` | Regression suites |
| `AGENTS.md` | This file: workflow, test rules, concurrency rules |
