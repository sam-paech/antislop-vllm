# utils/helpers.py
import yaml
import logging
import argparse
import copy
from typing import Dict, Any, List, Tuple
from pathlib import Path # Added Path

logging.basicConfig(
    level=logging.INFO, # Default, will be overridden by main script
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__) # Logger for this module


# --------------------------------------------------------------------- #
#  Built-in fallback                                                   #
# --------------------------------------------------------------------- #
_BUILTIN_DEFAULT: Dict[str, Any] = {
    "logging_level": "INFO",
    "generation_params": {
        "chunk_size":        50,
        "top_logprobs_count": 20,
        "max_new_tokens":   1200,
        "temperature":      0.7,
        "top_p":            1.0,
        "top_k":              50,
        "min_p":            0.05,
        "timeout":           120,
        "stop_sequences":   [],
    },
    "backtracking": {
        "max_retries_per_position": 20,
    },
    "ngram_validator": {
        # no banned list/file by default
        "remove_stopwords": True,
        "language":         "english",
    },
}

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """b ← a  (a wins only where b lacks the key).  Non-dict leaves are copied."""
    out: Dict[str, Any] = copy.deepcopy(b)
    for k, v in a.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(v, out[k])
        else:
            out[k] = copy.deepcopy(v)
    return out

def _str2bool(s: str) -> bool:
    if isinstance(s, bool):
        return s
    if s.lower() in {"true", "1", "yes", "y"}:
        return True
    if s.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false")

# --------------------------------------------------------------------- #
#  Public loader                                                        #
# --------------------------------------------------------------------- #
def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    user_cfg: Dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        logger.debug(f"Configuration loaded from {path}")
    except FileNotFoundError:
        logger.warning(f"Config file '{path}' not found – using built-in defaults.")
    except yaml.YAMLError as e:
        logger.error(f"YAML parse error in '{path}': {e} – using built-in defaults.")
    except Exception as e:
        logger.error(f"Unexpected error loading '{path}': {e} – using built-in defaults.")

    # Merge (built-in ← user) so user values override defaults.
    return _deep_merge(_BUILTIN_DEFAULT, user_cfg)


def add_common_generation_cli_args(parser: argparse.ArgumentParser, base_cfg: Dict[str, Any]):
    """Adds common CLI arguments related to API, model, and generation parameters to an existing parser."""
    g_default = base_cfg.get("generation_params", {})
    b_default = base_cfg.get("backtracking", {})
    ngram_default = base_cfg.get("ngram_validator", {}) # For defaults

    common_group = parser.add_argument_group('Common Generation Options')
    common_group.add_argument("--api-key", type=str, help="API key. Overrides config.yaml.")
    common_group.add_argument("--api-base-url", type=str, help="API base URL. Overrides config.yaml.")
    common_group.add_argument("--model-name", type=str, help="Model name. Overrides config.yaml.")
    common_group.add_argument("--slop-phrases-file", type=str, help="Path to slop phrases JSON file. Overrides config.yaml.")
    common_group.add_argument("--top-n-slop-phrases", type=int, help="Use top N slop phrases. Overrides config.yaml.")
    common_group.add_argument("--regex-blocklist-file", type=str, help="Path to regex blocklist JSON file. Overrides config.yaml.")
    common_group.add_argument("--logging-level",
                              choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                              help="Logging level for the script's operations.")
    common_group.add_argument(
        "--chat-template-model-id",
        type=str,
        help="HF model ID whose chat template should be prepended before "
             "calling the /v1/completions endpoint."
    )
    common_group.add_argument(
        "--request-mode",
        choices=["chunk", "stream"],
        help="How we contact the backend: classic chunk polling or true streaming."
    )
    common_group.add_argument("--force-backtrack", type=_str2bool,
        metavar="true/false",
        default=None,
        help="If true, progressively relax decoding filters when "
            "back-tracking runs out of candidates."
    )
    common_group.add_argument("--prompt-template",
                              type=str,
                              help="Python‐format string applied to --prompt "
                                   "(uses {prompt} placeholder).")
    common_group.add_argument("--system-prompt",
                              type=str,
                              help="System message to prepend when using a "
                                   "chat template.")


    gen_param_group = parser.add_argument_group('Generation Parameters (override config.yaml)')
    gen_param_group.add_argument("--chunk-size", type=int, default=g_default.get("chunk_size"), help="Chunk size for API requests.")
    gen_param_group.add_argument("--top-logprobs-count", type=int, default=g_default.get("top_logprobs_count"), help="Number of top logprobs to request.")
    gen_param_group.add_argument("--max-new-tokens", type=int, default=g_default.get("max_new_tokens"), help="Maximum new tokens to generate.")
    gen_param_group.add_argument("--temperature", type=float, default=g_default.get("temperature"), help="Sampling temperature.")
    gen_param_group.add_argument("--top-p", type=float, default=g_default.get("top_p"), help="Nucleus sampling (top-p).")
    gen_param_group.add_argument("--top-k", type=int, default=g_default.get("top_k"), help="Top-k filtering.")
    gen_param_group.add_argument("--min-p", type=float, default=g_default.get("min_p"), help="Min-p sampling.")
    gen_param_group.add_argument("--timeout", type=int, default=g_default.get("timeout"), help="API request timeout in seconds.")
    gen_param_group.add_argument("--stop-sequences", type=str, help="Comma-separated list of stop sequences.")

    backtrack_group = parser.add_argument_group('Backtracking Parameters (override config.yaml)')
    backtrack_group.add_argument("--max-retries-per-position",
                                 type=int,
                                 default=b_default.get("max_retries_per_position"),
                                 help="Max retries for backtracking at a single token position.")
    
    gen_param_group.add_argument(
        "--invert-probs",
        type=_str2bool,
        default=g_default.get("invert_probs"),
        metavar="true/false",
        help="Invert probability mass before sampling (overrides config.yaml)."
    )

    ngram_group = parser.add_argument_group('N-Gram Validator Options (override config.yaml)')
    ngram_group.add_argument("--ngram-banned-list", type=str, help="Comma-separated list of n-grams to ban (e.g., \"this is one,another one\"). Each n-gram string will be tokenized. Overrides file and config list.")
    ngram_group.add_argument("--ngram-banned-file", type=str, default=ngram_default.get("banned_file"), help="Path to JSON file with banned n-grams (list of strings or list of lists of strings).")
    ngram_group.add_argument("--ngram-remove-stopwords", type=lambda x: (str(x).lower() == 'true'), default=ngram_default.get("remove_stopwords", True), choices=[True, False], help="Remove stopwords before n-gram checking (true/false).")
    ngram_group.add_argument("--ngram-language", type=str, default=ngram_default.get("language", "english"), help="Language for n-gram stopwords.")


def merge_configs(base_cfg: Dict[str, Any], cli_args: argparse.Namespace) -> Dict[str, Any]:
    cfg = base_cfg.copy()
    args_dict = vars(cli_args)

    # Direct overrides for top-level simple settings
    scalar_keys = [
        "prompt", "api_key", "api_base_url", "model_name",
        "slop_phrases_file", "top_n_slop_phrases",
        "regex_blocklist_file", "logging_level",
        "chat_template_model_id", "request_mode",
        "force_backtrack", "prompt_template",
        "system_prompt",
    ]

    for key in scalar_keys:
        if args_dict.get(key) is not None:
            cfg[key] = args_dict[key]
    
    # Batch mode specific args (these don't typically exist in base_cfg but are used by the script)
    batch_keys = [
        "input_json", "input_hf_dataset", "hf_dataset_split", "hf_dataset_config_name",
        "output_jsonl", "threads", "max_prompts"
    ]
    for key in batch_keys:
        if args_dict.get(key) is not None:
            cfg[key] = args_dict[key]

    # Generation parameters
    gen_params_cfg = cfg.setdefault("generation_params", {})
    gen_param_keys = [
        "chunk_size", "top_logprobs_count", "max_new_tokens",
        "temperature", "top_p", "top_k", "min_p", "timeout",
        "invert_probs", 
    ]
    for key in gen_param_keys:
        if hasattr(cli_args, key) and getattr(cli_args, key) is not None:
            # Check if the value is different from the argparse default for that key
            # This ensures that if a user specifies the default value explicitly, it's still considered an override.
            # However, argparse defaults are usually set such that if getattr(cli_args, key) is not None,
            # it means it was either specified or it's a non-None default.
            # For simplicity, if it's not None and present, we take it.
            # This relies on argparse setting defaults correctly.
            arg_val = getattr(cli_args, key)
            # A more robust check would be:
            # if cli_args.__getattribute__(key) != parser.get_default(key):
            # But this requires passing the parser. The current check is usually fine.
            gen_params_cfg[key] = arg_val


    if hasattr(cli_args, "stop_sequences") and getattr(cli_args, "stop_sequences") is not None:
        raw_ss = getattr(cli_args, "stop_sequences")
        if raw_ss == "": 
            gen_params_cfg["stop_sequences"] = []
        else:
            gen_params_cfg["stop_sequences"] = [s.strip() for s in raw_ss.split(",") if s.strip()]

    # Backtracking parameters
    backtracking_cfg = cfg.setdefault("backtracking", {})
    if hasattr(cli_args, "max_retries_per_position") and getattr(cli_args, "max_retries_per_position") is not None:
        backtracking_cfg["max_retries_per_position"] = getattr(cli_args, "max_retries_per_position")

    # N-gram validator settings
    ngram_validator_cfg = cfg.setdefault("ngram_validator", {})
    if hasattr(cli_args, "ngram_banned_file") and getattr(cli_args, "ngram_banned_file") is not None:
        ngram_validator_cfg["banned_file"] = str(getattr(cli_args, "ngram_banned_file"))
    
    if hasattr(cli_args, "ngram_banned_list") and getattr(cli_args, "ngram_banned_list") is not None:
        cli_ngram_strings = [s.strip() for s in getattr(cli_args, "ngram_banned_list").split(',') if s.strip()]
        # This CLI arg takes precedence. Store it in a way that _setup_validators can prioritize it.
        ngram_validator_cfg["banned_list_from_cli"] = cli_ngram_strings
        # If CLI provides this, it might implicitly override 'banned_file' or 'banned_list' from config.yaml
        # depending on loading logic in _setup_validators.

    if hasattr(cli_args, "ngram_remove_stopwords") and getattr(cli_args, "ngram_remove_stopwords") is not None:
        ngram_validator_cfg["remove_stopwords"] = getattr(cli_args, "ngram_remove_stopwords")
    if hasattr(cli_args, "ngram_language") and getattr(cli_args, "ngram_language") is not None:
        ngram_validator_cfg["language"] = getattr(cli_args, "ngram_language")

    # Note: The effective logging level for the *application* (print vs log)
    # is determined in main.py after this merge.
    # The 'logging_level' in cfg here is for configuring the `logging` module itself.
    if args_dict.get("logging_level") is not None: # If CLI specified logging_level
        cfg["logging_level"] = args_dict["logging_level"]
    # else, it keeps the value from base_cfg or defaults to INFO if not in base_cfg.

    return cfg

# Keep setup_cli_args for stress_test.py or other simple scripts if needed.
def setup_cli_args(base_cfg: Dict[str, Any], prompt_required: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="API AntiSlop Sampler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prompt", type=str, required=prompt_required, help="Prompt text.")
    # This function is simpler and doesn't include batch/ngram args by default.
    # It's for scripts that only need the core generation args.
    # For full functionality, main.py builds its parser more comprehensively.
    # We can add a simplified set of common args here if needed for stress_test.
    
    # Simplified common args for this basic parser:
    g_default = base_cfg.get("generation_params", {})
    p.add_argument("--api-key", type=str, help="API key.")
    p.add_argument("--model-name", type=str, help="Model name.")
    p.add_argument("--max-new-tokens", type=int, default=g_default.get("max_new_tokens"), help="Max new tokens.")
    p.add_argument("--temperature", type=float, default=g_default.get("temperature"), help="Temperature.")
    # Add more if stress_test needs them directly.
    return p