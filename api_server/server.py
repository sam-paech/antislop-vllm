import time
import uuid
import json
import logging
from typing import List, Dict, Any, Optional
import requests
import copy

from fastapi import FastAPI, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import anyio.to_thread

from api_client.base_client import BaseApiClient
from core.sampler import ApiAntiSlopSampler
from validators.base_validator import BaseValidator
from utils.chat_template_helper import ChatTemplateFormatter

# The proxy's own Prometheus metrics, plus the upstream /metrics passthrough.  Importing this
# module must never issue a request, so the scrape only happens inside the GET /metrics route.
from api_server.metrics import (
    BAN_EVENTS_TOTAL,
    GENERATIONS_TOTAL,
    GENERATION_TOKENS_TOTAL,
    PROMPT_TOKENS_TOTAL,
    REQUESTS_IN_FLIGHT,
    REQUESTS_TOTAL,
    REQUEST_DURATION,
    STREAMING_TOTAL,
    UPSTREAM_ERRORS_TOTAL,
    render_metrics_body,
)

# --- Globals for shared resources ---
# These will be populated by main.py at startup.
SHARED_RESOURCES: Dict[str, Any] = {}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """
    Apply the configured concurrency limit once the event loop exists.

    anyio.to_thread.current_default_thread_limiter() raises NoEventLoopError
    when called from synchronous code, so this cannot live in
    setup_shared_resources() — it must run inside the ASGI lifespan.
    """
    max_concurrent = SHARED_RESOURCES.get("max_concurrent_requests")
    if max_concurrent:
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = int(max_concurrent)
        logger = SHARED_RESOURCES.get("logger")
        if logger:
            logger.info(
                f"anyio thread limiter set to {limiter.total_tokens} "
                f"concurrent generations (default was 40)"
            )
    yield


app = FastAPI(lifespan=_lifespan)

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

    # --- Concurrency ---
    # FastAPI's run_in_threadpool() is backed by anyio's *default* thread
    # limiter, which caps at 40 tokens regardless of how many requests arrive.
    # Beyond 40 in-flight requests, callers silently queue on the limiter
    # (no error, no 503) and latency inflates with no throughput gain.
    # Size it from the configured `max_concurrent_requests` (falling back to
    # `threads` + 8, matching the upstream connection-pool sizing).
    # The value is recorded here and applied in _lifespan(), because the
    # limiter can only be reached from inside a running event loop.
    #
    # NOTE: these keys live under `generation_params` in the YAML, so read that
    # first and only fall back to the top level for pre-flattened dicts.
    gen_params = config.get("generation_params", {}) or {}
    # generation_params.threads comes before the top-level `threads`: the CLI
    # always writes a top-level threads (its default is 1), which used to win
    # and cap the server at one concurrent generation.
    max_concurrent = (
        config.get("max_concurrent_requests")
        or gen_params.get("max_concurrent_requests")
        or gen_params.get("threads")
        or config.get("threads")
        or 1
    )
    max_concurrent = max(1, int(max_concurrent))
    SHARED_RESOURCES["max_concurrent_requests"] = max_concurrent
    main_logger.info(
        f"Concurrency limit configured: {max_concurrent} concurrent generations "
        f"(anyio default would be 40)"
    )


def _run_generation_for_request(
    request_data: Dict[str, Any],
    on_text: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one generation and account for it in the proxy's metrics.

    A thin wrapper around :func:`_run_generation_for_request_impl` so the in-flight gauge and the
    request timer are recorded on **every** exit path.  The impl returns plain dicts rather than
    raising (an error body, an upstream passthrough), so a ``try/finally`` around the call is the
    only way to guarantee the gauge comes back down: a leaked in-flight count would make the
    generator's 📊 Metrics tab report a permanently busy proxy.
    """
    REQUESTS_IN_FLIGHT.inc()
    started_at = time.time()
    try:
        result = _run_generation_for_request_impl(request_data, on_text=on_text)
        if isinstance(result, dict) and "error" in result:
            UPSTREAM_ERRORS_TOTAL.inc()
        return result
    finally:
        REQUEST_DURATION.observe(time.time() - started_at)
        REQUESTS_IN_FLIGHT.dec()


def _run_generation_for_request_impl(
    request_data: Dict[str, Any],
    on_text: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    The actual worker function that runs in a thread pool.
    It creates a sampler and generates the response.

    *on_text*, when given, is called with each accepted text fragment as it is
    produced, so the SSE endpoint can forward tokens while the generation is
    still running.  The full text is still returned in the result.
    """
    # Retrieve shared resources
    config = SHARED_RESOURCES["config"]
    # Clone shared validators so suppression (_ignored) is per‑request only.
    validators_template = SHARED_RESOURCES["validators"]
    validators = [copy.copy(v) for v in validators_template]
    for v in validators:
        # fresh suppression cache for this generation
        v._ignored = set()
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
        # Use the tokenizer's chat template application for robust formatting.
        # Per-request template switches (enable_thinking, ...) are honoured the
        # way vLLM's own chat endpoint honours them; they used to be dropped.
        template_kwargs = request_data.get("chat_template_kwargs") or {}
        if not isinstance(template_kwargs, dict):
            template_kwargs = {}
        try:
            prompt_text = chat_formatter.build_base_prompt_from_messages(
                messages, **template_kwargs
            )
        except Exception as e:
            logger.error(f"Error applying chat template: {e}", exc_info=True)
            return {"error": f"Failed to apply chat template: {e}"}
    else:
        # This used to fall back to the last user message alone, silently
        # dropping the system prompt and the whole conversation history.
        return {
            "error": "no chat template configured",
            "http_status": 500,
            "error_body": json.dumps({"error": (
                "antislop server has no chat template (set chat_template_model_id); "
                "refusing to drop the system prompt and conversation history"
            )}),
            "content_type": "application/json",
        }

    # --- 2. Instantiate Sampler ---
    # Sampler is created per-request to ensure thread safety of its internal state
    try:
        sampler = ApiAntiSlopSampler(
            api_client=api_client,
            validators=validators,
            config=config,
            tiktoken_model_name_for_counting=tiktoken_model_name,
            chat_template_formatter=chat_formatter
        )
    except Exception as e:
        # The stream-logprobs probe runs here.  Raised outside a handler it
        # surfaced as an unhandled ASGI exception instead of an error body the
        # client can read and retry on.
        logger.error(f"Could not start a generation: {e}")
        return {
            "error": f"Could not start a generation: {e}",
            "http_status": 502,
            "error_body": json.dumps({"error": str(e)}),
            "content_type": "application/json",
        }

    # --- 3. Extract per-request generation parameters ---
    max_tokens_req = request_data.get("max_tokens")
    temperature_req = request_data.get("temperature")
    min_p_req = request_data.get("min_p")
    top_p_req = request_data.get("top_p")
    top_k_req = request_data.get("top_k")

    # A client-supplied `stop` used to be dropped entirely, so client-side stop
    # sequences silently did nothing.  Apply them on top of the configured ones.
    stop_req = request_data.get("stop")
    if stop_req is not None:
        if isinstance(stop_req, str):
            stop_req = [stop_req]
        configured = list(sampler.stop_sequences or [])
        sampler.stop_sequences = configured + [
            str(s) for s in stop_req if str(s)
        ] or None

    # Sampling fields the sampler does not implement itself are forwarded to the
    # backend.  They used to be dropped, so a client's repetition_penalty never
    # reached vLLM through this proxy (while it did when the client called vLLM
    # directly).  `seed` is deliberately not forwarded: every restart re-sends
    # the request, and a fixed seed would replay the same continuation.
    for key in ("repetition_penalty", "presence_penalty", "frequency_penalty"):
        if request_data.get(key) is not None:
            sampler.extra_request_params[key] = request_data[key]

    # Override ban_strength if provided in request
    ban_strength_req = request_data.get("ban_strength")
    if ban_strength_req is not None:
        sampler.ban_strength = max(0.0, min(1.0, float(ban_strength_req)))

    # --- 4. Generate Response ---
    try:
        full_response_parts = []
        for piece in sampler.generate(
            prompt=prompt_text,
            max_new_tokens=max_tokens_req,
            temperature=temperature_req,
            min_p=min_p_req,
            top_p=top_p_req,
            top_k=top_k_req,
        ):
            full_response_parts.append(piece)
            if on_text is not None and piece:
                try:
                    on_text(piece)
                except Exception as e_cb:
                    logger.warning(f"on_text callback failed: {e_cb}")
        full_response = "".join(full_response_parts)
    except requests.HTTPError as e:
        # Upstream provided an HTTP response — propagate its status/body
        resp = getattr(e, "response", None)
        status_code = int(getattr(resp, "status_code", 502) or 502)
        body_text = None
        content_type = "application/json"
        if resp is not None:
            try:
                body_text = json.dumps(resp.json())
            except Exception:
                try:
                    body_text = resp.text
                    # crude content-type inference
                    if not body_text.lstrip().startswith(("{", "[")):
                        content_type = "text/plain; charset=utf-8"
                except Exception:
                    body_text = json.dumps({"error": str(e)})
        else:
            body_text = json.dumps({"error": str(e)})
        return {
            "error": "upstream_error",
            "http_status": status_code,
            "error_body": body_text,
            "content_type": content_type,
        }
    except Exception as e:
        logger.error(f"Error during generation for a request: {e}", exc_info=True)
        return {
            "error": f"Generation failed: {e}",
            "http_status": 500,
            "error_body": json.dumps({"error": str(e)}),
            "content_type": "application/json",
        }


    # A generation that ended because the backend failed is not an answer.
    # Returning it with HTTP 200 (finish_reason "error") let a client that does
    # not read finish_reason store an empty or cut-off turn as a real one.
    if sampler.finish_reason == "error":
        return {
            "error": "backend failed during generation",
            "http_status": 502,
            "error_body": json.dumps({"error": "backend failed during generation",
                                      "partial_text_chars": len(full_response)}),
            "content_type": "application/json",
        }

    # --- 5. Token Counting for Usage ---
    prompt_tokens = 0
    completion_tokens = 0
    model_tok = getattr(chat_formatter, "tokenizer", None)
    if model_tok is not None and callable(model_tok):
        # Count with the model's tokenizer; tiktoken's cl100k is a different
        # vocabulary and misstates usage for any non-OpenAI model.
        try:
            prompt_tokens = len(model_tok(prompt_text, add_special_tokens=False)["input_ids"])
            completion_tokens = len(model_tok(full_response, add_special_tokens=False)["input_ids"])
        except Exception as e:
            logger.warning(f"Could not count tokens for usage stats: {e}")
    elif sampler.tiktoken_encoding:
        try:
            prompt_tokens = len(sampler.tiktoken_encoding.encode(prompt_text))
            completion_tokens = len(sampler.tiktoken_encoding.encode(full_response))
        except Exception as e:
            logger.warning(f"Could not count tokens for usage stats: {e}")

    # --- 6. Construct OpenAI-compatible response ---
    response_id = f"chatcmpl-{uuid.uuid4()}"
    created_timestamp = int(time.time())
    model_name = request_data.get("model", config.get("model_name", "antislop-model"))

    # Report the reason the sampler actually ended for.  This used to be hard-coded
    # to "stop", which meant a truncated (`length`) or failed (`error`) generation
    # was indistinguishable from a complete one for any client building a dataset.
    finish_reason = sampler.finish_reason or "stop"

    # --- 6b. Account for the generation in the proxy's metrics ---
    # The proxy's whole purpose is the ban/back-track loop, so the loop is what the
    # metrics expose: how many violations fired, per validator, and whether the
    # back-track actually fixed them.  `sampler.events` is the sampler's own record
    # (see `_record_event`), so nothing is counted twice or invented here.
    GENERATIONS_TOTAL.inc({"finish_reason": finish_reason})
    PROMPT_TOKENS_TOTAL.inc(amount=prompt_tokens)
    GENERATION_TOKENS_TOTAL.inc(amount=completion_tokens)
    for event in getattr(sampler, "events", None) or []:
        try:
            BAN_EVENTS_TOTAL.inc({
                "validator": str(event.get("type") or "unknown"),
                "fixed": "true" if event.get("fixed") else "false",
            })
        except Exception as e_metric:
            # Metrics must never fail a request that already succeeded.
            logger.warning(f"Could not record a ban event in the metrics: {e_metric}")

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


def _sse_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_generation(request_data: Dict[str, Any]):
    """
    Stream an OpenAI-compatible ``chat.completion.chunk`` sequence.

    Previously the server always returned a single non-streaming JSON body, even when
    the client asked for ``stream: true``.  An OpenAI SDK client in streaming mode
    then had to parse a non-SSE body, which is one of the ways a caller ends up
    with a truncated assistant turn: the first ``[DONE]``-less payload is treated
    as the whole answer.

    Generation itself runs in a worker thread; chunks are handed to the event loop
    through a queue so the blocking sampler never stalls the server.
    """
    import asyncio
    import uuid as _uuid

    queue: "asyncio.Queue[Any]" = asyncio.Queue()
    loop = asyncio.get_running_loop()

    response_id = f"chatcmpl-{_uuid.uuid4()}"
    created = int(time.time())
    model_name = request_data.get("model", SHARED_RESOURCES["config"].get("model_name", "antislop-model"))

    def worker() -> None:
        try:
            result = _run_generation_for_request(request_data, on_text=lambda t: loop.call_soon_threadsafe(queue.put_nowait, ("text", t)))
        except Exception as e:  # pragma: no cover - defensive
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
            return
        loop.call_soon_threadsafe(queue.put_nowait, ("done", result))

    loop.run_in_executor(None, worker)

    # opening chunk with the assistant role
    yield _sse_event({
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    })

    while True:
        kind, payload = await queue.get()
        if kind == "text":
            if not payload:
                continue
            yield _sse_event({
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": payload}, "finish_reason": None}],
            })
        elif kind == "error":
            yield _sse_event({"error": {"message": payload, "type": "upstream_error"}})
            yield "data: [DONE]\n\n"
            return
        else:
            result = payload
            if "error" in result:
                body = result.get("error_body") or json.dumps(result)
                yield _sse_event({"error": {"message": body, "type": "upstream_error"}})
                yield "data: [DONE]\n\n"
                return
            finish_reason = result["choices"][0].get("finish_reason") or "stop"
            yield _sse_event({
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": result.get("usage"),
            })
            yield "data: [DONE]\n\n"
            return


@app.get("/metrics")
async def metrics():
    """Prometheus metrics for this proxy, plus the upstream vLLM's own ``/metrics``.

    The dataset generator's 📊 Metrics tab polls ``<host>:<port>/metrics`` for every API slot and
    treats a non-200 as "this endpoint is unreachable".  This route did not exist, so every slot
    pointed at this proxy showed no figures at all even though the proxy forwards to a vLLM that
    serves them.

    The upstream body is passed through **verbatim** (so the generator's existing ``vllm:`` parsing
    keeps working unchanged, and a metric from a newer vLLM this code has never heard of still
    arrives), and this server's own ``antislop:`` metrics are appended to it.

    A dead or absent upstream still returns **200** with the ``antislop:`` figures and
    ``antislop:upstream_metrics_scrape_total{result="error"}``: the difference between "the proxy is
    unreachable" and "the proxy is up but its backend is not" is exactly what the tab needs, and a
    5xx here would erase that distinction.

    The scrape is blocking, so it runs in the threadpool rather than on the event loop.
    """
    config = SHARED_RESOURCES.get("config") or {}
    api_base_url = config.get("api_base_url")
    body = await run_in_threadpool(render_metrics_body, api_base_url)
    return Response(
        content=body,
        media_type="text/plain; version=0.0.4; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    """
    OpenAI-compatible endpoint for chat completions.

    Supports both non-streaming (single JSON body) and ``stream: true`` (SSE),
    which the dataset generator relies on.
    """
    try:
        request_data = await request.json()
    except Exception:
        REQUESTS_TOTAL.inc({"endpoint": "/v1/chat/completions", "status": "400"})
        return Response(
            content=json.dumps({"error": "Invalid JSON in request body"}),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json"
        )

    STREAMING_TOTAL.inc({"stream": "true" if request_data.get("stream") else "false"})

    if request_data.get("stream"):
        from fastapi.responses import StreamingResponse
        # The status is known as soon as the stream is handed back; a failure *inside* the
        # stream surfaces as an SSE error event, which the generator already handles.
        REQUESTS_TOTAL.inc({"endpoint": "/v1/chat/completions", "status": "200"})
        return StreamingResponse(
            _stream_generation(request_data),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Run the blocking generation code in a thread pool
    result = await run_in_threadpool(_run_generation_for_request, request_data)

    if "error" in result:
        # If upstream status/body were provided, pass them through verbatim
        if "http_status" in result and "error_body" in result:
            code = int(result.get("http_status", 500))
            REQUESTS_TOTAL.inc({"endpoint": "/v1/chat/completions", "status": str(code)})
            return Response(
                content=result["error_body"],
                status_code=code,
                media_type=result.get("content_type", "application/json"),
            )
        # Fallback: generic 500 with our error envelope
        REQUESTS_TOTAL.inc({"endpoint": "/v1/chat/completions", "status": "500"})
        return Response(
            content=json.dumps(result),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

    REQUESTS_TOTAL.inc({"endpoint": "/v1/chat/completions", "status": "200"})


    return Response(content=json.dumps(result), media_type="application/json")