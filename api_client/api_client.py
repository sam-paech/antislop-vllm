import json
import logging
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base_client import BaseApiClient
from core.models import ApiChunkResult

logger = logging.getLogger(__name__)

_SHARED_SESSIONS: Dict[str, requests.Session] = {}

class ApiClient(BaseApiClient):
    """
    OpenAI-compatible completions client that keeps a per-instance connection
    pool.  Pool size = <worker threads> + 8, passed in by the caller.
    """

    OPENAI_V1 = "https://api.openai.com/v1"

    # ------------------------------------------------------------------ #
    #  Init                                                              #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        pool_size: int,
    ):
        self._cancel_stream = False      # sampler sets this to abort

        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.completion_endpoint = f"{self.base_url}/completions"

        # -------- reuse or create session -------------
        try:
            self._session = _SHARED_SESSIONS[self.base_url]      # ← reuse
        except KeyError:
            self._session = self._build_session(pool_size)       # ← create once
            _SHARED_SESSIONS[self.base_url] = self._session

        logger.info(
            f"ApiClient ready (endpoint={self.completion_endpoint}, "
            f"shared_pool={pool_size})"
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_session(pool_size: int) -> requests.Session:
        """
        Create a requests.Session backed by a urllib3 connection pool large
        enough for high-thread workloads.
        """
        s = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=Retry(
                total=3,
                backoff_factor=0.3,
                status_forcelist=(502, 503, 504),
                allowed_methods=frozenset(["POST"]),
            ),
        )
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        return s

    def cancel_current_stream(self):
        self._cancel_stream = True

    # ------------------------------------------------------------------ #
    #  NEW: streaming generator                                          #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    #  NEW: streaming generator                                          #
    # ------------------------------------------------------------------ #
    def generate_stream(
        self,
        prompt_text: str,
        max_tokens: int,
        top_logprobs: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        min_p: Optional[float],
        timeout: int,
        stop_sequences: Optional[List[str]],
        **kwargs,
    ):
        """
        Stream /v1/completions and yield one ApiChunkResult **per token**.

        * Supports vLLM, OpenAI, and OpenRouter.
        * Keeps the same provider-specific payload tweaks used in
        generate_chunk().
        * Early-abort when self._cancel_stream is set by the sampler.
        """

        # ---------- build request payload ----------------------------------
        payload: Dict[str, object] = {
            "model":       self.model_name,
            "prompt":      prompt_text,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "top_p":       top_p,
            "top_k":       top_k,
            "stream":      True,
            **kwargs,
        }
        if stop_sequences:
            payload["stop"] = stop_sequences

        # -- provider switches (mirror of generate_chunk) -------------------
        if self.base_url == self.OPENAI_V1:
            payload["logprobs"] = top_logprobs                 # OpenAI wants int
        elif self.base_url == "https://openrouter.ai/api/v1":
            payload.update({
                "min_p":        min_p,
                "logprobs":     True,
                "top_logprobs": top_logprobs,
                "provider": {
                    "order": ["Fireworks"],
                    "allow_fallbacks": False,
                },
            })
        else:                                                  # vLLM family
            payload.update({
                "min_p":        min_p,
                "logprobs":     top_logprobs,                  # ← vLLM wants int
                "top_logprobs": top_logprobs,
            })

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logger.debug(f"STREAM POST {self.completion_endpoint}")
        logger.debug(f"Payload: {json.dumps(payload)[:500]}")

        # ---------- open HTTP stream ---------------------------------------
        with self._session.post(
            self.completion_endpoint,
            headers=headers,
            json=payload,
            stream=True,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()

            for raw in resp.iter_lines(decode_unicode=True):

                # ---- sampler asked to abort? ------------------------------
                if self._cancel_stream:
                    resp.close()
                    self._cancel_stream = False
                    break

                if not raw or raw == "\n":
                    continue

                if raw.startswith("data: "):
                    raw = raw[6:]

                if raw.strip() == "[DONE]":
                    # signal natural end-of-stream
                    yield ApiChunkResult(
                        generated_text="",
                        finish_reason="stream_end",
                    )
                    break

                # ---------- parse SSE JSON ---------------------------------
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Malformed SSE line: %s", raw[:160])
                    continue

                if not obj.get("choices"):
                    continue
                choice = obj["choices"][0]

                # -- incremental text fragment ------------------------------
                generated_text = choice.get("text", "")
                # (some providers use .delta.content / .message.content – those
                # branches are left in generate_chunk; for vLLM we only need .text)

                # -- extract logprobs --------------------------------------
                token_strings: List[str] = []
                rel_logprobs: Dict[int, List[Tuple[str, float]]] = {}

                lp_obj = choice.get("logprobs")
                if lp_obj and isinstance(lp_obj, dict):
                    token_strings = lp_obj.get("tokens", [])
                    # vLLM: top_logprobs is list[dict] aligned with tokens
                    for idx, alt_dict in enumerate(lp_obj.get("top_logprobs", [])):
                        if isinstance(alt_dict, dict):
                            rel_logprobs[idx] = list(alt_dict.items())

                yield ApiChunkResult(
                    generated_text=generated_text,
                    token_strings=token_strings,
                    logprobs=rel_logprobs,
                    finish_reason=choice.get("finish_reason"),
                )



    # ------------------------------------------------------------------ #
    #  Core call                                                         #
    # ------------------------------------------------------------------ #
    def generate_chunk(
        self,
        prompt_text: str,
        max_tokens: int,
        top_logprobs: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        min_p: Optional[float],
        timeout: int,
        stop_sequences: Optional[List[str]],
        **kwargs,
    ) -> ApiChunkResult:

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, object] = {
            "model":       self.model_name,
            "prompt":      prompt_text,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "top_p":       top_p,
            "top_k":       top_k,
            "stream":      False,
            **kwargs,
        }
        if stop_sequences:
            payload["stop"] = stop_sequences

        # API-flavour switches
        if self.base_url == self.OPENAI_V1:
            payload["logprobs"] = top_logprobs
        elif self.base_url == "https://openrouter.ai/api/v1":
            payload.update({"min_p": min_p, "logprobs": True, "top_logprobs": top_logprobs})
            # few providers support top logprobs or completions properly
            payload["provider"] = {
                "order": [
                    #"DeepInfra", # llama-3.1-8b, mistral-small-3, qwen-72b
                    #"Mistral" # mistral-small-3
                    #"Lambda", # llama-3.1-8b
                    #"NovitaAI",  # qwen-72b, llama-3.1-8b
                    #"Nebius AI Studio", # qwen-72b
                    #"Hyperbolic", # qwen-72b
                    #"inference.net", # llama-3.1-8b
                    #"Groq", # llama 3.1 8b
                    #"inference.net",
                    "Fireworks"
                ],
                "allow_fallbacks": False
            }
        else:  # vLLM-style
            payload.update({"min_p": min_p, "logprobs": top_logprobs})

        logger.debug(f"POST {self.completion_endpoint}")
        logger.debug(f"Payload: {json.dumps(payload)[:500]}")

        try:
            resp = self._session.post(
                self.completion_endpoint,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            logger.error("API request timed out.")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            try:
                logger.error(f"Error detail: {resp.json()}")
            except Exception:
                pass
            raise

        data = resp.json()
        logger.debug(f"Raw response: {json.dumps(data)[:2000]}")

        if not data.get("choices"):
            logger.warning("API response contained no choices.")
            return ApiChunkResult(generated_text="", finish_reason="no_choice")

        choice = data["choices"][0]

        # -------- generated text --------
        # supports both /completions (legacy) and /chat/completions payloads
        if "text" in choice:                              # legacy completions
            generated_text = choice["text"]
        elif "message" in choice and isinstance(choice["message"], dict):  # chat completions
            generated_text = choice["message"].get("content", "")
        else:
            generated_text = ""

        finish_reason = choice.get("finish_reason")


        # ------------ extract logprobs ------------
        token_strings: List[str] = []
        relative_logprobs: Dict[int, List[Tuple[str, float]]] = {}

        lp_obj = choice.get("logprobs")
        if lp_obj:
            if "tokens" in lp_obj:  # legacy format
                token_strings = lp_obj["tokens"]
                for idx, alt_dict in enumerate(lp_obj.get("top_logprobs", [])):
                    if isinstance(alt_dict, dict):
                        relative_logprobs[idx] = list(alt_dict.items())
            elif "content" in lp_obj and isinstance(lp_obj["content"], list):  # chat format
                for idx, item in enumerate(lp_obj["content"]):
                    tok = item.get("token")
                    if tok is None:
                        continue
                    token_strings.append(tok)
                    alt_list = item.get("top_logprobs")
                    if isinstance(alt_list, list):
                        relative_logprobs[idx] = [
                            (alt.get("token"), alt.get("logprob"))
                            for alt in alt_list
                            if alt.get("token") is not None
                        ]

        if not token_strings:
            logger.warning(
                "Logprobs present but no token strings parsed; "
                "back-tracking disabled for this chunk."
            )

        logger.info(
            f"Received {len(token_strings)} token strings "
            f"(finish_reason={finish_reason or 'None'})."
        )

        return ApiChunkResult(
            generated_text=generated_text,
            token_strings=token_strings,
            logprobs=relative_logprobs,
            finish_reason=finish_reason,
        )
