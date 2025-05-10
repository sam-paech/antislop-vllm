# ai/antislop-api/main.py
import argparse
import copy
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from tqdm import tqdm

from utils.helpers import load_config, merge_configs, add_common_generation_cli_args
from utils.slop_helpers import load_slop_phrases
from utils.regex_helpers import load_regex_patterns
from api_client.api_client import ApiClient
from validators.slop_phrase_validator import SlopPhraseValidator
from validators.regex_validator import RegexValidator
# NGramValidator is imported conditionally below
from core.sampler import ApiAntiSlopSampler

# Base logging config - this is just an initial setup, will be fully configured in main_cli
# Set to a high level initially to minimize output before full config
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s [%(levelname)-5.5s] [%(name)-20.20s]: %(message)s",
)
# We don't need a separate script_debug_logger anymore; the root logger will be controlled.


progress_lock = Lock()
overall_prompts_processed_count = 0
overall_tokens_generated_count = 0
overall_processing_start_time = 0.0


def _setup_validators(cfg: Dict[str, Any], main_logger: logging.Logger) -> List[Any]:
    validators = []

    # Slop Phrase Validator
    slop_cfg_path_str = cfg.get("slop_phrases_file")
    slop_top_n = cfg.get("top_n_slop_phrases")
    
    if slop_cfg_path_str and str(slop_cfg_path_str).strip():
        slop_dict = load_slop_phrases(str(slop_cfg_path_str), top_n=slop_top_n)
        if slop_dict:
            validators.append(SlopPhraseValidator(slop_dict, app_config=cfg))
        else:
            main_logger.debug(f"Slop phrases file '{slop_cfg_path_str}' configured but no phrases loaded (file might be empty or unreadable).")
    elif "slop_phrases_file" in cfg:
        main_logger.debug("Slop phrases file key ('slop_phrases_file') exists in config but is not set to a valid path or is empty.")

    # Regex Validator
    regex_cfg_path_str = cfg.get("regex_blocklist_file")
    if regex_cfg_path_str and str(regex_cfg_path_str).strip():
        regex_list = load_regex_patterns(str(regex_cfg_path_str))
        if regex_list:
            validators.append(RegexValidator(regex_list))
        else:
            main_logger.debug(f"Regex blocklist file '{regex_cfg_path_str}' configured but no patterns loaded.")
    elif "regex_blocklist_file" in cfg:
        main_logger.debug("Regex blocklist file key ('regex_blocklist_file') exists in config but is not set or path is empty.")

    # N-Gram Validator
    try:
        from validators.ngram_validator import NGramValidator
        
        ngram_validator_config = cfg.get("ngram_validator", {})
        banned_ngrams_for_validator: List[Union[str, List[str]]] = []
        source_of_ngrams = "nothing"

        if ngram_validator_config.get("banned_list_from_cli"):
            banned_ngrams_for_validator = ngram_validator_config["banned_list_from_cli"]
            source_of_ngrams = "CLI --ngram-banned-list"
        elif ngram_validator_config.get("banned_file"):
            ngram_file_path_str = ngram_validator_config.get("banned_file")
            if ngram_file_path_str and str(ngram_file_path_str).strip():
                ngram_file_path = Path(ngram_file_path_str)
                if ngram_file_path.exists():
                    try:
                        with ngram_file_path.open("r", encoding="utf-8") as f:
                            loaded_ngrams = json.load(f)
                        if isinstance(loaded_ngrams, list):
                            banned_ngrams_for_validator = loaded_ngrams
                            source_of_ngrams = f"file '{ngram_file_path}'"
                        else:
                            main_logger.error(f"N-gram file {ngram_file_path} must contain a JSON list. NGramValidator might be ineffective.")
                    except json.JSONDecodeError:
                        main_logger.error(f"JSON decode error in n-gram file {ngram_file_path}. NGramValidator might be ineffective.")
                    except Exception as e:
                        main_logger.error(f"Error loading n-gram file {ngram_file_path}: {e}. NGramValidator might be ineffective.")
                else:
                    main_logger.warning(f"N-gram banned file specified ('{ngram_file_path}') but not found.")
            elif "banned_file" in ngram_validator_config:
                 main_logger.debug("ngram_validator.banned_file path is empty or None in config.")
        elif ngram_validator_config.get("banned_list"): 
            config_list = ngram_validator_config.get("banned_list")
            if isinstance(config_list, list) and config_list:
                banned_ngrams_for_validator = config_list
                source_of_ngrams = "config.yaml 'ngram_validator.banned_list'"
            elif isinstance(config_list, list) and not config_list:
                main_logger.debug("ngram_validator.banned_list in config.yaml is an empty list.")
            elif "banned_list" in ngram_validator_config:
                main_logger.warning("ngram_validator.banned_list in config.yaml is present but not a valid list or is empty.")


        if banned_ngrams_for_validator:
            main_logger.debug(f"Initializing NGramValidator with {len(banned_ngrams_for_validator)} n-gram entries from {source_of_ngrams}.")
            remove_sw = ngram_validator_config.get("remove_stopwords", True)
            lang = ngram_validator_config.get("language", "english")
            
            ngram_validator_instance = NGramValidator(
                banned_ngrams_config=banned_ngrams_for_validator,
                remove_stopwords_flag=remove_sw,
                language=lang
            )
            if not ngram_validator_instance.is_disabled:
                validators.append(ngram_validator_instance)
            else:
                main_logger.warning("NGramValidator was disabled due to NLTK setup issues (check logs from NGramValidator initialization).")
        elif cfg.get("ngram_validator"):
            main_logger.debug("NGramValidator section is configured but no banned n-grams were loaded from any source (CLI, file, or config list).")

    except ImportError:
        main_logger.warning("Could not import NGramValidator. NLTK might not be installed or its data (punkt, stopwords) might be missing. Skipping n-gram validation.")
    except Exception as e_ngram_init:
        main_logger.error(f"Unexpected error during NGramValidator setup: {e_ngram_init}", exc_info=True)

    if not validators:
        main_logger.info("No validators were configured or loaded.")
    else:
        main_logger.info(f"Initialized {len(validators)} validators: {[v.__class__.__name__ for v in validators]}")

    return validators


def _get_api_client(cfg: Dict[str, Any], main_logger: logging.Logger) -> Optional[ApiClient]:
    api_key = cfg.get("api_key")
    if not api_key and "openai.com" in cfg.get("api_base_url", "openai.com"):
        main_logger.error("API key missing for OpenAI-like service. Set it in config.yaml or with --api-key.")
        return None
    if api_key == "YOUR_API_KEY_HERE":
        main_logger.error("API key is a placeholder. Please set a real API key.")
        return None

    model_name = cfg.get("model_name")
    if not model_name:
        main_logger.error("model_name missing in configuration.")
        return None

    # pool size = thread count + 8 (buffer); default thread count = 1
    pool_size = int(cfg.get("threads", 1)) + 8

    return ApiClient(
        api_key=api_key or "",
        model_name=model_name,
        base_url=cfg.get("api_base_url"),
        pool_size=pool_size,
    )



def handle_single_generation(cfg: Dict[str, Any], args: Any, script_effective_log_level: int, main_logger: logging.Logger):
    # In single mode, we still use the logger for its events for now,
    # as the print-based ban events are more for batch mode's cleaner TQDM interface.
    # If truly needed, this could also be adapted.
    main_logger.debug("Running in single prompt generation mode.")

    prompt = args.prompt or cfg.get("prompt")
    if not prompt:
        main_logger.error("Prompt missing for single generation mode.")
        return

    validators = _setup_validators(cfg, main_logger)
    api_client = _get_api_client(cfg, main_logger)
    if not api_client:
        return

    model_name = cfg.get("model_name")

    sampler = ApiAntiSlopSampler(
        api_client=api_client,
        validators=validators,
        config=cfg,
        tiktoken_model_name_for_counting=model_name
    )

    # User-facing start message for single mode
    if script_effective_log_level <= logging.INFO:
        print("--- Generation start ---")
    main_logger.debug("--- Generation start (debug log) ---")
    
    print("\n--- Generated Output (stream) ---")
    full_response = ""
    try:
        for chunk in sampler.generate(prompt):
            print(chunk, end="", flush=True)
            full_response += chunk
    except Exception as e:
        main_logger.error(f"Generation failed: {e}", exc_info=(script_effective_log_level == logging.DEBUG))
    finally:
        print("\n------------------------")
        print("\n--- Full Generated Text ---")
        print(full_response)
        print("----------------------------\n")
        
        # User-facing completion message for single mode
        if script_effective_log_level <= logging.INFO:
            print("--- Generation complete ---")
            print(f"Chars produced: {len(full_response)}")
        main_logger.debug(f"--- Generation complete (debug log) --- Chars: {len(full_response)}")

        if sampler.events:
            # User-facing ban events summary for single mode
            if script_effective_log_level <= logging.INFO:
                print("Ban/back-track events:")
                for ev in sampler.events:
                    # Simplified print for INFO
                    details_str = str(ev.get('details', {}))
                    print(f"  - Type: {ev.get('type')}, Index: {ev.get('index')}, Fixed: {ev.get('fixed')}, Details: {details_str[:100]}{'...' if len(details_str) > 100 else ''}")
            # Full debug log of events
            main_logger.debug("Full Ban/back-track events (debug log):")
            for ev_debug in sampler.events:
                main_logger.debug(ev_debug)


def generate_for_prompt_worker(
    prompt_idx: int,
    prompt_text: str,
    config: dict,
    pbar_global: tqdm,
    script_effective_log_level: int,
    main_logger: logging.Logger
):
    current_prompt_tokens_generated = 0

    def _on_ban_event_callback(event: dict):
        # This callback decides to print (via tqdm.write) or log based on script_effective_log_level
        details = event.get("details", {})
        phrase = details.get("phrase", "")
        match = details.get("match", "")
        pattern = details.get("pattern", "")
        ngram_str = details.get("ngram_string", "")

        ban_message_base = f"BANNED (prompt_idx={prompt_idx}, type={event['type']}"
        ban_detail = ""
        if phrase: ban_detail = f", phrase='{phrase}')"
        elif match and pattern: ban_detail = f", regex='{pattern}', match='{match}')"
        elif ngram_str: ban_detail = f", ngram='{ngram_str}')"
        else: ban_detail = f", details='{str(details)[:50]}...')"

        full_ban_message_for_print = f"{ban_message_base}{ban_detail}"
        
        if script_effective_log_level == logging.DEBUG:
            # Construct more detailed message for debug log
            context = details.get("context", "")
            original_token_debug = event.get("original_token_string", "")
            debug_message = f"{ban_message_base}"
            if phrase: debug_message += f", phrase='{phrase}'"
            elif match and pattern: debug_message += f", regex='{pattern}', match='{match}'"
            elif ngram_str: debug_message += f", ngram='{ngram_str}' (stopwords_removed={details.get('remove_stopwords_active', 'N/A')})"
            else: debug_message += f", details='{str(details)[:100]}'"
            if original_token_debug: debug_message += f" @ orig_token='{original_token_debug}'"
            if context:
                context_snip = str(context).strip().replace('\n', ' ')
                debug_message += f" ...'{context_snip[:100]}'..."
            debug_message += ")"
            main_logger.debug(debug_message) # Use main_logger for debug

        elif script_effective_log_level == logging.INFO:
            # Print simplified ban message using tqdm.write to avoid interfering with the bar
            with progress_lock:
                 tqdm.write(full_ban_message_for_print, file=sys.stdout)

        # No output for WARNING, ERROR, CRITICAL levels for ban events


    def _on_chunk_yielded_callback(text_chunk: str, num_tokens: int):
        nonlocal current_prompt_tokens_generated
        global overall_tokens_generated_count, overall_processing_start_time

        current_prompt_tokens_generated += num_tokens
        with progress_lock:
            overall_tokens_generated_count += num_tokens
            if overall_processing_start_time > 0:
                elapsed_time = time.perf_counter() - overall_processing_start_time
                tok_per_sec = overall_tokens_generated_count / elapsed_time if elapsed_time > 0 else 0.0
                if not pbar_global.disable:
                    pbar_global.set_postfix_str(f"{tok_per_sec:.1f} tok/s", refresh=True)

    thread_cfg = copy.deepcopy(config)
    validators = _setup_validators(thread_cfg, main_logger)
    api_client = _get_api_client(thread_cfg, main_logger)
    if not api_client:
        return {
            "prompt_id": prompt_idx, "prompt": prompt_text, "generation": None,
            "status": "failed", "error": "API client setup failed (key/model missing).",
            "events": [], "duration_sec": 0, "tokens_generated_prompt": 0,
        }
    
    model_name = thread_cfg.get("model_name")

    sampler = ApiAntiSlopSampler(
        api_client=api_client,
        validators=validators,
        config=thread_cfg,
        on_ban_event_callback=_on_ban_event_callback,
        on_chunk_yielded_callback=_on_chunk_yielded_callback,
        tiktoken_model_name_for_counting=model_name
    )

    full_response_parts = []
    generation_successful = True
    error_message = None
    start_time_prompt = time.perf_counter()

    try:
        for chunk_text in sampler.generate(prompt_text):
            full_response_parts.append(chunk_text)
    except Exception as e:
        # Log actual errors using the main_logger, which is configured to show ERROR/CRITICAL
        main_logger.error(f"Error generating for prompt_idx {prompt_idx} ('{prompt_text[:50]}...'): {e}", exc_info=(script_effective_log_level == logging.DEBUG))
        generation_successful = False
        error_message = str(e)

    end_time_prompt = time.perf_counter()
    duration_prompt = end_time_prompt - start_time_prompt
    final_generated_text = "".join(full_response_parts)

    return {
        "prompt_id": prompt_idx, "prompt": prompt_text,
        "generation": final_generated_text if generation_successful else None,
        "status": "success" if generation_successful else "failed",
        "error": error_message, "events": sampler.events,
        "duration_sec": duration_prompt,
        "tokens_generated_prompt": current_prompt_tokens_generated,
    }


def handle_batch_generation(cfg: Dict[str, Any], args: argparse.Namespace, script_effective_log_level: int, main_logger: logging.Logger):
    global overall_prompts_processed_count, overall_tokens_generated_count, overall_processing_start_time
    main_logger.debug("Running in batch data generation mode.")

    all_input_prompts_with_ids: List[Tuple[int, str]] = []
    source_prompts: List[str] = []

    if args.input_json:
        main_logger.debug(f"Loading prompts from JSON file: {args.input_json}")
        try:
            with args.input_json.open("r", encoding="utf-8") as f:
                loaded_prompts = json.load(f)
            if isinstance(loaded_prompts, list) and all(isinstance(p, str) for p in loaded_prompts):
                source_prompts = loaded_prompts
            else:
                main_logger.error(f"Invalid format in JSON: {args.input_json}. Expected list of strings.")
                return
        except Exception as e:
            main_logger.error(f"Failed to load prompts from {args.input_json}: {e}")
            return
    elif args.input_hf_dataset:
        if not DATASETS_AVAILABLE:
            main_logger.error("The 'datasets' library is required. `pip install datasets`.")
            return
        main_logger.debug(f"Loading prompts from HF dataset: {args.input_hf_dataset}")
        try:
            # Suppress datasets library verbose logging unless script is in DEBUG
            if script_effective_log_level > logging.DEBUG:
                logging.getLogger("datasets").setLevel(logging.ERROR)
                logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

            dataset = load_dataset(args.input_hf_dataset, name=args.hf_dataset_config_name, split=args.hf_dataset_split, trust_remote_code=True)
            disable_hf_tqdm = script_effective_log_level > logging.INFO
            for item in tqdm(dataset, desc="Extracting HF prompts", unit="prompt", disable=disable_hf_tqdm, file=sys.stdout):
                prompt_extracted = None
                if "conversations" in item and isinstance(item["conversations"], list):
                    for turn in item["conversations"]:
                        if isinstance(turn, dict) and turn.get("from", "").lower() == "human" and "value" in turn:
                            prompt_extracted = str(turn["value"]); break
                elif "prompt" in item and isinstance(item["prompt"], str): prompt_extracted = item["prompt"]
                elif "text" in item and isinstance(item["text"], str):
                    if not any(kw in item["text"].lower() for kw in ["assistant:", "bot:", "gpt:", "\n\n"]):
                         prompt_extracted = item["text"]
                
                if prompt_extracted:
                    prompt_extracted = f"Writing prompt: {prompt_extracted}\n\nWrite 1000 words to this prompt. Your response:\n"
                    source_prompts.append(prompt_extracted)
                if args.max_prompts is not None and len(source_prompts) >= args.max_prompts: # Check against source_prompts length
                    main_logger.debug(f"Reached max_prompts ({args.max_prompts}) from source during loading.")
                    break
            if not source_prompts: main_logger.warning(f"No prompts extracted from HF dataset '{args.input_hf_dataset}'.")
        except Exception as e:
            main_logger.error(f"Failed to load/process HF dataset {args.input_hf_dataset}: {e}", exc_info=(script_effective_log_level == logging.DEBUG)); return
    
    if not source_prompts: main_logger.error("No prompts to process from any source."); return
    main_logger.debug(f"Total prompts loaded from source: {len(source_prompts)}.")
    
    for original_idx, p_text in enumerate(source_prompts):
        all_input_prompts_with_ids.append((original_idx, p_text))

    processed_prompt_texts = set()
    if args.output_jsonl.exists():
        main_logger.debug(f"Output file {args.output_jsonl} exists. Attempting to resume.")
        try:
            with args.output_jsonl.open("r", encoding="utf-8") as f_in:
                for line in f_in:
                    try:
                        existing_result = json.loads(line)
                        if "prompt" in existing_result: processed_prompt_texts.add(existing_result["prompt"])
                    except json.JSONDecodeError: main_logger.warning(f"Skipping malformed line: {line.strip()}")
            main_logger.debug(f"Resuming. Found {len(processed_prompt_texts)} unique processed prompt texts.")
        except Exception as e:
            main_logger.error(f"Error reading existing output {args.output_jsonl}: {e}.")

    prompts_to_process_this_run = []
    for original_idx, p_text in all_input_prompts_with_ids:
        if p_text not in processed_prompt_texts:
            prompts_to_process_this_run.append((original_idx, p_text))

    if args.max_prompts is not None and len(prompts_to_process_this_run) > args.max_prompts:
        main_logger.debug(f"Limiting new prompts for this run to {args.max_prompts}.")
        prompts_to_process_this_run = prompts_to_process_this_run[:args.max_prompts]

    if not prompts_to_process_this_run:
        if script_effective_log_level <= logging.INFO:
             print("All loaded prompts already processed or max_prompts limit met. Nothing new to do.")
        return
    
    if script_effective_log_level <= logging.INFO:
        print(f"Preparing to process {len(prompts_to_process_this_run)} new prompts in this run.")

    overall_processing_start_time = time.perf_counter()
    overall_prompts_processed_count = 0
    overall_tokens_generated_count = 0

    try:
        disable_main_tqdm = script_effective_log_level > logging.INFO 
        with args.output_jsonl.open("a", encoding="utf-8") as outfile:
            with tqdm(total=len(prompts_to_process_this_run), desc="Batch Generating", unit="prompt", disable=disable_main_tqdm, file=sys.stdout) as pbar_global:
                with ThreadPoolExecutor(max_workers=args.threads, thread_name_prefix="Generator") as executor:
                    future_to_prompt_id = {
                        executor.submit(generate_for_prompt_worker, prompt_idx, prompt_text, cfg, pbar_global, script_effective_log_level, main_logger): prompt_idx
                        for prompt_idx, prompt_text in prompts_to_process_this_run
                    }
                    for future in as_completed(future_to_prompt_id):
                        try:
                            result_data = future.result()
                            json.dump(result_data, outfile)
                            outfile.write("\n")
                            outfile.flush()
                        except Exception as e_fut:
                            main_logger.error(f"Critical error processing a prompt future: {e_fut}", exc_info=(script_effective_log_level == logging.DEBUG))
                        finally:
                            with progress_lock:
                                overall_prompts_processed_count +=1
                                if not disable_main_tqdm:
                                    pbar_global.update(1)
    except IOError as e:
        main_logger.error(f"Could not write to output file {args.output_jsonl}: {e}"); return
    except Exception as e:
        main_logger.error(f"Unexpected error during batch execution: {e}", exc_info=(script_effective_log_level == logging.DEBUG)); return

    total_time_taken_run = time.perf_counter() - overall_processing_start_time
    
    if script_effective_log_level <= logging.INFO:
        print(f"Finished processing {overall_prompts_processed_count} prompts in this run in {total_time_taken_run:.2f}s.")
        if total_time_taken_run > 0 and overall_tokens_generated_count > 0:
            avg_tok_per_sec_run = overall_tokens_generated_count / total_time_taken_run
            print(f"Overall average throughput for this run: {avg_tok_per_sec_run:.2f} tok/s.")
        print(f"Results appended to {args.output_jsonl}")
    elif overall_prompts_processed_count > 0 :
        main_logger.info(f"Batch run completed. Processed {overall_prompts_processed_count} prompts. Results in {args.output_jsonl}")


def main_cli():
    parser = argparse.ArgumentParser(
        description="AntiSlop API Sampler. Single prompt or batch data generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to base YAML config.")
    
    single_mode_group = parser.add_argument_group('Single Prompt Mode (default if no batch args)')
    single_mode_group.add_argument("--prompt", type=str, help="Prompt text for single generation.")

    batch_mode_group = parser.add_argument_group('Batch Data Generation Mode')
    batch_mode_group.add_argument("--output-jsonl", type=Path, help="Enable batch mode: Path to output JSONL file.")
    input_source_group = batch_mode_group.add_mutually_exclusive_group()
    input_source_group.add_argument("--input-json", type=Path, help="Batch mode: Path to JSON file (list of prompts).")
    input_source_group.add_argument("--input-hf-dataset", type=str, help="Batch mode: Hugging Face dataset ID (ShareGPT format).")
    batch_mode_group.add_argument("--hf-dataset-split", type=str, default="train", help="Batch mode: Split for HF dataset.")
    batch_mode_group.add_argument("--hf-dataset-config-name", type=str, default=None, help="Batch mode: Config name for HF dataset.")
    batch_mode_group.add_argument("--threads", type=int, default=1, help="Batch mode: Number of parallel generation threads.")
    batch_mode_group.add_argument("--max-prompts", type=int, default=None, help="Batch mode: Max new prompts to process from source per run.")

    temp_args_for_config, _ = parser.parse_known_args()
    base_cfg_path = temp_args_for_config.config if hasattr(temp_args_for_config, 'config') and temp_args_for_config.config else "config.yaml"
    base_cfg_for_cli = load_config(str(base_cfg_path)) 
    add_common_generation_cli_args(parser, base_cfg_for_cli)
    args = parser.parse_args()

    final_base_cfg = load_config(str(args.config)) 
    cfg = merge_configs(final_base_cfg, args)

    # --- Configure Logging Based on Effective Level ---
    # Determine the user's desired overall script output level
    user_desired_logging_level_str = cfg.get("logging_level", "INFO").upper()
    script_effective_log_level = getattr(logging, user_desired_logging_level_str, logging.INFO)

    # Get the root logger, which all other loggers inherit from
    app_logger = logging.getLogger()
    
    # Clear existing handlers from the root logger to prevent duplicate outputs
    for handler in app_logger.handlers[:]:
        app_logger.removeHandler(handler)
    
    # Add a new stream handler for console output
    stream_handler = logging.StreamHandler(sys.stderr) 
    formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] [%(name)-20.20s]: %(message)s")
    stream_handler.setFormatter(formatter)
    
    # Set the handler's level to DEBUG. This handler will process *any* message
    # that gets past the logger's level filter. The logger's level is the primary control.
    stream_handler.setLevel(logging.DEBUG) 

    app_logger.addHandler(stream_handler)
    
    # Set the root logger's level based on the desired script output level.
    # This is the primary filter for all standard logging messages.
    if script_effective_log_level == logging.DEBUG:
        app_logger.setLevel(logging.DEBUG) # Allow all standard logs
    elif script_effective_log_level in [logging.INFO, logging.WARNING]:
        app_logger.setLevel(logging.ERROR) # Suppress standard logs below ERROR
    else: # ERROR, CRITICAL
        app_logger.setLevel(script_effective_log_level) # Only show errors or critical via standard logging

    # Control verbosity of specific library loggers
    # This is done *after* setting the root logger's level.
    # If the root level is ERROR, setting a library logger to WARNING is fine,
    # but its WARNING/INFO/DEBUG messages won't show because the root logger filters them out.
    # This is the desired behavior for INFO/WARNING script levels.
    # For DEBUG script level, these settings allow finer control over library noise.
    if script_effective_log_level <= logging.DEBUG:
        # In DEBUG mode, allow more verbose library logs, but maybe quiet down known noisy ones slightly
        logging.getLogger("nltk").setLevel(logging.INFO) # NLTK can be noisy
        # Other libraries will inherit DEBUG from root or use their own defaults if higher.
    else: # INFO, WARNING, ERROR, CRITICAL for the script
        # Quiet down common noisy libraries significantly.
        for lib_logger_name in ["openai", "httpx", "httpcore", "requests", "urllib3", "huggingface_hub", "datasets", "nltk"]:
            logging.getLogger(lib_logger_name).setLevel(logging.WARNING)


    # User-facing messages about the logging mode (using print for INFO/WARNING as per requirements)
    # These messages are *not* filtered by the logging system levels we just set.
    if script_effective_log_level == logging.DEBUG:
        # For DEBUG, use the logger itself so it's formatted and part of the debug stream.
        app_logger.info(f"Full DEBUG logging enabled. Effective script level: {user_desired_logging_level_str}")
    elif script_effective_log_level == logging.INFO:
        print(f"INFO mode: Progress bar and ban events will be printed. Most logs suppressed. Effective script level: {user_desired_logging_level_str}")
    elif script_effective_log_level == logging.WARNING:
        print(f"WARNING mode: Progress bar will be printed. Most logs suppressed. Effective script level: {user_desired_logging_level_str}")
    # For ERROR/CRITICAL, actual error messages will be logged by app_logger.error/critical if they occur.

    # Pass app_logger (the configured root logger) to functions that need to log standard messages.
    # The `script_effective_log_level` is passed to control print vs log behavior inside those functions
    # (specifically for the custom BANNED messages and tqdm disable).

    is_batch_mode = args.output_jsonl is not None
    if is_batch_mode:
        if not args.input_json and not args.input_hf_dataset:
            app_logger.critical("For batch mode (--output-jsonl), either --input-json or --input-hf-dataset must be specified.")
            parser.exit(2) 
        handle_batch_generation(cfg, args, script_effective_log_level, app_logger)
    else:
        if not args.prompt:
            app_logger.critical("--prompt is required for single generation mode (when --output-jsonl is not used).")
            parser.exit(2)
        handle_single_generation(cfg, args, script_effective_log_level, app_logger)

if __name__ == "__main__":
    main_cli()