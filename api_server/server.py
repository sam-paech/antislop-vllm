import time
import uuid
import json
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, Response, status
from fastapi.concurrency import run_in_threadpool

from api_client.base_client import BaseApiClient
from core.sampler import ApiAntiSlopSampler
from validators.base_validator import BaseValidator
from utils.chat_template_helper import ChatTemplateFormatter

# --- Globals for shared resources ---
# These will be populated by main.py at startup.
SHARED_RESOURCES: Dict[str, Any] = {}

app = FastAPI()

def setup_shared_resources(
    config: Dict[str, Any],
    validators: List[BaseValidator],
    api_client: BaseApiClient,
    chat_formatter: Optional[ChatTemplateFormatter],
    main_logger: logging.Logger,
):
    """
    Populates the global SHARED_RESOURCES dictionary.
    This is called once at server startup from main.py.
    """
    SHARED_RESOURCES["config"] = config
    SHARED_RESOURCES["validators"] = validators
    SHARED_RESOURCES["api_client"] = api_client
    SHARED_RESOURCES["chat_formatter"] = chat_formatter
    SHARED_RESOURCES["logger"] = main_logger
    SHARED_RESOURCES["tiktoken_model_name"] = config.get("model_name")


def _run_generation_for_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    The actual worker function that runs in a thread pool.
    It creates a sampler and generates the response.
    """
    # Retrieve shared resources
    config = SHARED_RESOURCES["config"]
    validators = SHARED_RESOURCES["validators"]
    api_client = SHARED_RESOURCES["api_client"]
    chat_formatter = SHARED_RESOURCES["chat_formatter"]
    logger = SHARED_RESOURCES["logger"]
    tiktoken_model_name = SHARED_RESOURCES["tiktoken_model_name"]

    # --- 1. Format the prompt ---
    messages = request_data.get("messages", [])
    if not messages:
        return {"error": "Request must contain 'messages'"}

    prompt_text = ""
    if chat_formatter and hasattr(chat_formatter, 'tokenizer'):
        # Use the tokenizer's chat template application for robust formatting
        try:
            prompt_text = chat_formatter.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True # Ensures the prompt ends correctly for the model to continue
            )
        except Exception as e:
            logger.error(f"Error applying chat template: {e}", exc_info=True)
            return {"error": f"Failed to apply chat template: {e}"}
    else:
        # Fallback: simple concatenation if no chat template is configured.
        # We'll just take the last user message as the prompt.
        last_user_message = next((m['content'] for m in reversed(messages) if m.get('role') == 'user'), None)
        if not last_user_message:
             return {"error": "No user message found in 'messages'"}
        prompt_text = last_user_message

    # --- 2. Instantiate Sampler ---
    # Sampler is created per-request to ensure thread safety of its internal state
    sampler = ApiAntiSlopSampler(
        api_client=api_client,
        validators=validators,
        config=config,
        tiktoken_model_name_for_counting=tiktoken_model_name,
        chat_template_formatter=chat_formatter
    )

    # --- 3. Extract per-request generation parameters ---
    max_tokens_req = request_data.get("max_tokens")
    temperature_req = request_data.get("temperature")
    min_p_req = request_data.get("min_p")

    # --- 4. Generate Response ---
    try:
        full_response_parts = list(sampler.generate(
            prompt=prompt_text,
            max_new_tokens=max_tokens_req,
            temperature=temperature_req,
            min_p=min_p_req,
        ))
        full_response = "".join(full_response_parts)
    except Exception as e:
        logger.error(f"Error during generation for a request: {e}", exc_info=True)
        return {"error": f"Generation failed: {e}"}

    # --- 5. Token Counting for Usage ---
    prompt_tokens = 0
    completion_tokens = 0
    if sampler.tiktoken_encoding:
        try:
            prompt_tokens = len(sampler.tiktoken_encoding.encode(prompt_text))
            completion_tokens = len(sampler.tiktoken_encoding.encode(full_response))
        except Exception as e:
            logger.warning(f"Could not count tokens for usage stats: {e}")

    # --- 6. Construct OpenAI-compatible response ---
    response_id = f"chatcmpl-{uuid.uuid4()}"
    created_timestamp = int(time.time())
    model_name = request_data.get("model", config.get("model_name", "antislop-model"))

    # The finish reason can be inferred from the last chunk in a more complex setup,
    # but for now, 'stop' is a reasonable default for a completed generation.
    finish_reason = "stop"

    response_payload = {
        "id": response_id,
        "object": "chat.completion",
        "created": created_timestamp,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_response,
                },
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return response_payload


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    """
    OpenAI-compatible endpoint for chat completions.
    This is non-streaming.
    """
    try:
        request_data = await request.json()
    except Exception:
        return Response(
            content=json.dumps({"error": "Invalid JSON in request body"}),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json"
        )

    # Run the blocking generation code in a thread pool
    result = await run_in_threadpool(_run_generation_for_request, request_data)

    if "error" in result:
        return Response(
            content=json.dumps(result),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json"
        )

    return Response(content=json.dumps(result), media_type="application/json")