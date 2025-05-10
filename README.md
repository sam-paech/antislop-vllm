# AntiSlop-API

This project provides a Python-based tool for generating text using an OpenAI-compatible API while actively filtering out undesirable "slop" content in real-time. It features chunk-wise generation, multiple validation strategies (slop phrases, regex blocklists), and local back-tracking using API-provided logprobs to attempt to fix violations without additional API calls.

The primary goal is to produce cleaner, higher-quality text generations by mitigating common LLM pitfalls like repetitive phrases, unwanted disclaimers, or off-topic content.

## Features

*   **OpenAI-Compatible API Client:** Works with any `/v1/completions` endpoint that supports logprobs (llama.cpp, vLLM). Note: **`chat/completions` endpoints will not work**.
*   **Chunk-wise Generation:** Processes text in manageable chunks for efficient validation.
*   **Real-time Validation:**
    *   **Slop Phrase Blocking:** Ban specific phrases from appearing in the output.
    *   **Regex Blocklisting:** Define regular expression patterns to block more complex unwanted content.
*   **Local Back-tracking:** If a validator flags an issue, the sampler attempts to replace the offending token(s) by resampling from the cached logprobs provided by the API. This avoids costly re-prompts for minor infractions.
*   **Configurable Parameters:** Extensive configuration options for generation (temperature, top-p, etc.), validation, and back-tracking.
*   **Batch Data Generation Mode:**
    *   Process multiple prompts from a JSON file or a Hugging Face dataset (ShareGPT format).
    *   Parallel processing using threads.
    *   Resume capability for long-running jobs.
    *   Customizable output logging with `tqdm` progress and token/second metrics.
*   **Single Prompt Mode:** Generate and stream output for a single prompt directly to the console.
*   **Stress Testing:** Includes a script to benchmark the pipeline under concurrent load.

## Project Structure

```
ai/antislop-api/
├── .gitignore
├── config.yaml             # Default configuration file
├── main.py                 # Main script for single & batch generation
├── stress_test.py          # Script for stress testing the API sampler
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── api_client/             # API interaction logic
│   ├── __init__.py
│   ├── api_client.py       # OpenAI-compatible API client
│   └── base_client.py      # Abstract base client
├── core/                   # Core sampling and model logic
│   ├── __init__.py
│   ├── models.py           # Dataclasses for API results, violations
│   └── sampler.py          # ApiAntiSlopSampler - main generation/validation orchestrator
├── state/                  # State management for generation
│   ├── __init__.py
│   └── generation_state.py # Manages token strings and text views
├── utils/                  # Helper utilities
│   ├── __init__.py
│   ├── helpers.py          # Config loading, CLI argument parsing
│   ├── regex_helpers.py    # Loading regex patterns
│   └── slop_helpers.py     # Loading slop phrases, detection logic
└── validators/             # Validation logic
    ├── __init__.py
    ├── base_validator.py   # Abstract base validator
    ├── regex_validator.py  # Regex-based validator
    └── slop_phrase_validator.py # Slop phrase list validator

# Example data files (create these yourself)
# ├── slop_phrases.json
# └── regex_blocklist.json
```

## Setup

1.  **Clone the repository (if applicable).**
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure `config.yaml`:**
    *   Copy `config.yaml` (if a template is provided) or create one.
    *   **Crucially, set your `api_key`**.
    *   Adjust `model_name`, `api_base_url`, and other generation parameters as needed.
    *   Specify paths to `slop_phrases_file` and `regex_blocklist_file` if you intend to use them. These files should be JSON lists.
        *   `slop_phrases.json` example: `[["a very bad phrase", 0.0], ["another one", 0.0]]`
        *   `regex_blocklist.json` example: `["bad pattern\\d+", "another regex.*here"]`
        If not using, you can point to empty files (`[]`) or leave the config entries commented/empty and the validators won't load.

## Usage (`main.py`)

The `main.py` script is the primary interface and supports two main modes:

### 1. Single Prompt Generation

Generates text for a single prompt and streams the output to the console.

```bash
python ai/antislop-api/main.py --prompt "Tell me a story about a brave knight." \
    --config ai/antislop-api/config.yaml \
    --model-name "google/gemma-2-9b-it" \
    --max-new-tokens 500
```

**Common Arguments for Single Mode:**

*   `--prompt "YOUR PROMPT"`: (Required) The input prompt.
*   `--config FILE_PATH`: Path to the YAML configuration file (defaults to `config.yaml`).
*   Other arguments to override `config.yaml` values (e.g., `--api-key`, `--model-name`, `--temperature`, `--max-new-tokens`, `--slop-phrases-file`, `--logging-level`). Run `python ai/antislop-api/main.py --help` for a full list.

### 2. Batch Data Generation

Generates text for multiple prompts from a source file/dataset and saves results to a JSONL file. Supports resuming and parallel processing.

```bash
python ai/antislop-api/main.py \
    --config ai/antislop-api/config.yaml \
    --output-jsonl path/to/output_generations.jsonl \
    --input-hf-dataset "Nitral-AI/Creative_Writing-ShareGPT" \
    --hf-dataset-split "train" \
    --threads 4 \
    --max-prompts 100 \
    --model-name "anthropic/claude-3-opus-20240229" \
    --logging-level INFO
```

**Key Arguments for Batch Mode:**

*   `--output-jsonl FILE_PATH`: **Enables batch mode.** Path to the output JSONL file where results (one JSON object per prompt) will be saved.
*   **Input Source (choose one):**
    *   `--input-json FILE_PATH`: Path to a JSON file containing a list of prompt strings.
    *   `--input-hf-dataset DATASET_ID`: Hugging Face dataset ID (e.g., "OpenAssistant/oasst_top1_2023-08-25"). Assumes ShareGPT format (extracts "human" turns).
*   `--hf-dataset-split NAME`: Split to use for the HF dataset (default: "train").
*   `--hf-dataset-config-name NAME`: Optional config name for the HF dataset.
*   `--threads N`: Number of parallel generation threads (default: 1).
*   `--max-prompts N`: Maximum number of *new* prompts to process in this run (useful for partial runs or testing).
*   `--logging-level LEVEL`: Controls console output verbosity:
    *   `DEBUG`: Full verbose logging from all modules.
    *   `INFO`: `tqdm` progress bar + printed ban events. Most other logs suppressed.
    *   `WARNING`: `tqdm` progress bar only. Most other logs suppressed.
    *   `ERROR`/`CRITICAL`: Only critical errors logged. No progress bar or ban events printed.

**Output JSONL Format (Batch Mode):**

Each line in the output file is a JSON object with fields like:

```json
{
    "prompt_id": 0, // Original index from input source
    "prompt": "The input prompt text.",
    "generation": "The generated text...", // null if generation failed
    "status": "success" or "failed",
    "error": "Error message if status is failed, else null.",
    "events": [ // List of ban/back-track events during generation
        {
            "type": "slop_phrase" or "regex",
            "index": 123, // Token index of violation
            "details": {"phrase": "...", "context": "..."} or {"pattern": "...", "match": "..."},
            "original_token_string": "offending_token",
            "fixed": true or false // Whether back-tracking successfully replaced it
        }
    ],
    "duration_sec": 10.5, // Time taken for this prompt
    "tokens_generated_prompt": 450 // Number of tokens in the final 'generation' field (tiktoken counted)
}
```

### `--help`

For a full list of all available command-line arguments and their descriptions:
```bash
python ai/antislop-api/main.py --help
```

## Stress Testing (`stress_test.py`)

A separate script is provided to benchmark the pipeline with increasing parallelism. It measures aggregate throughput in tokens per second.

```bash
python ai/antislop-api/stress_test.py \
    --config ai/antislop-api/config.yaml \
    --prompt "Write a detailed analysis of the current economic climate." \
    --max-threads 10
```

## How It Works

1.  **Initialization:** Loads configuration, slop phrases, and regex patterns. Initializes the API client and validators.
2.  **Prompt Input:** Takes a prompt (or a batch of prompts).
3.  **Chunked Generation:**
    *   The `ApiAntiSlopSampler` requests text generation from the API in chunks (e.g., 50-200 tokens at a time), asking for `top_logprobs`.
    *   The API returns the generated token strings and their alternative log probabilities.
4.  **State Update:** The `GenerationState` object appends the new token strings and caches their logprobs.
5.  **Validation:**
    *   After each new set of tokens is added, all registered validators (`SlopPhraseValidator`, `RegexValidator`) inspect the current generated text.
    *   If a validator detects a violation (e.g., a banned phrase or a regex match), it returns `ViolationInfo`.
6.  **Back-tracking (if violation detected):**
    *   The `ApiAntiSlopSampler` attempts to fix the violation *locally*.
    *   It identifies the offending token(s) based on `ViolationInfo.violation_index`.
    *   It uses the cached `logprobs` for the offending token position to select an alternative token. This selection process mimics standard decoding (applying temperature, top-p, top-k, min-p to the alternatives).
    *   The chosen alternative token replaces the original offending token in the `GenerationState`.
    *   The generation continues from this corrected point.
    *   If no suitable alternative is found after `max_retries_per_position`, the violation is "suppressed" (the validator will ignore this specific instance), and generation continues with the original offending token.
7.  **Output:**
    *   Accepted text (either originally valid or corrected via back-tracking) is yielded.
    *   In batch mode, the final generated text and event logs are written to the JSONL output.
    *   In single mode, text is streamed to the console.

## Customization

*   **Validators:** Implement new validators by subclassing `validators.base_validator.BaseValidator` and adding them to the `validators` list in `main.py`.
*   **API Client:** While the current client is for OpenAI-compatible APIs, you could adapt `api_client.base_client.BaseApiClient` for different API structures if they provide token-level logprobs.
*   **Configuration:** Modify `config.yaml` or use CLI arguments to tune nearly every aspect of the generation and validation process.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.