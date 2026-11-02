import json
import logging
import threading
from contextlib import ExitStack, contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base_client import BaseApiClient
from core.models import ApiChunkResult

logger = logging.getLogger(__name__)


class _SessionPool:
    """
    One shared ``requests.Session`` for a ``base_url``, plus a count of how many
    requests are currently using it.

    Reference counting is what makes ``reset_connection_pool()`` safe: a session
    with ``active > 0`` is *retired* rather than closed, so its users keep a
    working pool and the last one to finish closes it.  This also keeps every
    client on the *same* session, so the configured pool size still bounds total
    upstream connections.
    """

    __slots__ = ("session", "pool_size", "active", "retired")

    def __init__(self, session: requests.Session, pool_size: int) -> None:
        self.session = session
        self.pool_size = pool_size
        self.active = 0
        self.retired = False


# Sessions are shared per base_url so that N ApiClient instances (one per batch
# worker, or per request in server mode) do not each open their own pool.
_SESSIONS_LOCK = threading.RLock()
_SHARED_POOLS: Dict[str, _SessionPool] = {}


class StreamCancelToken:
    """Cancellation handle for **one** upstream stream.

    A single ``ApiClient`` instance is shared by every concurrent request in server
    mode (``SHARED_RESOURCES["api_client"]``).  The old ``_cancel_stream``
    boolean therefore belonged to the *client*, not to a stream: when request A
    backtracked and called ``cancel_current_stream()``, the next chunk of request
    B's unrelated stream saw the flag, closed its socket, and surfaced as
    ``finish_reason='stream_end'``.  Under load that produced a storm of
    "Stream closed without [DONE] or finish_reason" warnings and truncated or
    restarted generations that were perfectly healthy.

    A token is created per stream and passed into :meth:`ApiClient.generate_stream`,
    so cancelling one stream can never affect another.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()


@contextmanager
def _lease_session(base_url: str, pool_size: int) -> Iterator[requests.Session]:
    """
    Lease the shared session for *base_url* for the duration of one request.

    The session cannot be closed by a concurrent ``reset_connection_pool()`` while
    the lease is held: the reset only marks it retired, and the close is deferred
    to whichever request releases it last.  This is what makes a mid-stream reset
    safe.
    """
    pool = None
    try:
        with _SESSIONS_LOCK:
            pool = _SHARED_POOLS.get(base_url)
            if pool is None:
                pool = _SessionPool(ApiClient._build_session(pool_size), pool_size)
                _SHARED_POOLS[base_url] = pool
            pool.active += 1
            session = pool.session
        yield session
    finally:
        if pool is not None:
            _release_pool(pool)


def _release_pool(pool: _SessionPool) -> None:
    """Drop one reference; close the session if it was retired and now idle."""
    with _SESSIONS_LOCK:
        pool.active -= 1
        should_close = pool.retired and pool.active <= 0
    if should_close:
        try:
            pool.session.close()
            logger.info("Closed retired API session (last user finished).")
        except Exception as e:
            logger.warning(f"Error closing retired session: {e}")


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
        # Set once the backend rejects `bad_words`; stops us from paying a
        # failed round-trip on every subsequent request.
        self.suppress_tokens_unsupported = False
        # Legacy, client-wide cancellation.  Kept only so older callers do not
        # crash; it is NOT used to cancel an individual stream any more (see
        # StreamCancelToken for why).
        self._cancel_stream = False
        # Tokens of the streams currently open on this client.
        self._active_tokens: set = set()
        self._tokens_lock = threading.Lock()

        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.completion_endpoint = f"{self.base_url}/completions"

        # -------- adopt the shared pool -------------
        # No session is opened here, and no reference to one is retained: every
        # request leases the *current* session for its base_url (see
        # _lease_session).  Holding a session (or pool) on the instance would go
        # stale as soon as another client reset the pool.
        with _SESSIONS_LOCK:
            pool = _SHARED_POOLS.get(self.base_url)
            if pool is None:
                pool = _SessionPool(self._build_session(pool_size), pool_size)
                _SHARED_POOLS[self.base_url] = pool
            self.pool_size = pool.pool_size

        logger.info(
            f"ApiClient ready (endpoint={self.completion_endpoint}, "
            f"shared_pool={self.pool_size})"
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

    def new_stream_token(self) -> StreamCancelToken:
        """Create a cancellation handle for one stream.

        Pass the returned token to :meth:`generate_stream` as
        ``cancel_token=...`` and call :meth:`cancel_stream` on it to abort that
        stream only.
        """
        return StreamCancelToken()

    def register_stream_token(self, token: Optional[StreamCancelToken]) -> None:
        if token is None:
            return
        with self._tokens_lock:
            self._active_tokens.add(token)

    def unregister_stream_token(self, token: Optional[StreamCancelToken]) -> None:
        if token is None:
            return
        with self._tokens_lock:
            self._active_tokens.discard(token)

    def cancel_stream(self, token: Optional[StreamCancelToken]) -> None:
        """Cancel exactly one stream.  No-op when *token* is None."""
        if token is not None:
            token.cancel()

    def cancel_current_stream(self):
        """Cancel every stream open on this client.

        Retained for backwards compatibility, but this is only safe when the
        client serves a single request (batch mode).  Server mode shares one client
        across all in-flight requests and must use a per-stream token instead.
        """
        self._cancel_stream = True
        with self._tokens_lock:
            tokens = list(self._active_tokens)
        for tok in tokens:
            tok.cancel()

    def reset_connection_pool(self):
        """
        Replace the shared connection pool for this ``base_url`` with a fresh one.

        Useful when the server appears stuck or unresponsive and stale keep-alive
        connections need to be discarded.

        This is safe to call while other requests are in flight.  The old session
        is **retired, not closed**: any request still using it keeps a working
        pool, and the session is closed by whichever request releases it last.
        New requests immediately get the replacement session.

        Previously this closed the shared session and dropped it from the cache
        outright, while every other ApiClient kept its own reference to that now
        closed object.  ``Session.close()`` does not break an in-flight response,
        so nothing crashed -- but the clients silently diverged onto separate
        pools, and the configured ``pool_maxsize`` cap stopped applying to the
        whole process.  Each reset could therefore multiply the number of upstream
        connections, which is precisely what the pool sizing is meant to bound.
        """
        pool_size = max(1, int(getattr(self, "pool_size", 10)))

        with _SESSIONS_LOCK:
            old = _SHARED_POOLS.get(self.base_url)
            new_pool = _SessionPool(self._build_session(pool_size), pool_size)
            _SHARED_POOLS[self.base_url] = new_pool

            old_idle = old is None or old.active <= 0
            if old is not None and not old_idle:
                old.retired = True

        if old is None:
            logger.info(f"Connection pool reset (pool_size={pool_size}).")
            return

        if old_idle:
            try:
                old.session.close()
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            logger.info(
                f"Connection pool reset (pool_size={pool_size}, old session closed)."
            )
        else:
            logger.info(
                f"Connection pool reset (pool_size={pool_size}); old session "
                f"retired with {old.active} request(s) still in flight and will "
                f"close when they finish."
            )

    # ------------------------------------------------------------------ #
    #  SSE parsing helpers                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _iter_sse_data(resp) -> Iterator[str]:
        """
        Yield the payload of every SSE ``data:`` event from *resp*.

        ``requests``' ``iter_lines(decode_unicode=True)`` decodes each line in
        isolation, so a multi-byte UTF-8 character split across two TCP packets
        is mangled into U+FFFD.  That corrupts the JSON on that line, and the
        old code responded by *silently skipping the line* — i.e. dropping
        generated tokens in the middle of a response.  We therefore read raw
        bytes, buffer them, and decode whole lines at once.

        Multi-line ``data:`` events are joined with "\n" as the SSE spec
        requires; ``event:``/``id:``/``:`` comment lines are ignored.
        """
        buf = bytearray()
        data_lines: List[str] = []

        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            buf.extend(chunk)
            while True:
                nl = buf.find(b"\n")
                if nl == -1:
                    break
                raw_line = bytes(buf[:nl])
                del buf[: nl + 1]
                if raw_line.endswith(b"\r"):
                    raw_line = raw_line[:-1]

                line = raw_line.decode("utf-8", errors="replace")
                if not line:
                    # Blank line terminates the event.
                    if data_lines:
                        yield "\n".join(data_lines)
                        data_lines = []
                    continue
                if line.startswith(":"):
                    continue                      # comment / keep-alive
                if line.startswith("data:"):
                    # The spec allows both "data: x" and "data:x".
                    payload = line[5:]
                    if payload.startswith(" "):
                        payload = payload[1:]
                    data_lines.append(payload)
                # other fields (event:, id:, retry:) are not used

        if data_lines:
            yield "\n".join(data_lines)

    def _build_stream_payload(
        self,
        prompt_text: str,
        max_tokens: int,
        top_logprobs: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        min_p: Optional[float],
        stop_sequences: Optional[List[str]],
        kwargs: Dict[str, object],
    ) -> Dict[str, object]:
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
        return payload

    # ------------------------------------------------------------------ #
    #  Streaming generator                                               #
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
        cancel_token: Optional[StreamCancelToken] = None,
        **kwargs,
    ):
        """
        Stream /v1/completions and yield one ApiChunkResult **per token**.

        Guarantees, relied upon by the sampler:

        * Exactly one chunk with a non-``None`` ``finish_reason`` is yielded at
          the end, whatever happens (``stop``/``length`` from the backend, or
          ``stream_end`` for a clean ``[DONE]``, or ``error``/``timeout``).  A
          generator that just *stops* without a terminal chunk used to leave the
          sampler's outer loop with no exit condition.
        * ``finish_reason`` is **latched**: vLLM sends it on a final chunk that
          often carries no text, and the old code overwrote the latched value
          with ``None`` on every ordinary chunk.
        * The SSE ``text`` delta is preserved even when ``logprobs`` is absent,
          so text can never be lost between backend and caller.

        ``suppress_tokens`` (list of literal strings) is forwarded as ``bad_words``
        when the backend supports it; see :meth:`_post_stream`.

        ``cancel_token`` scopes cancellation to *this* stream.  Without it, one
        request backtracking used to abort every other in-flight stream on the
        shared client (see :class:`StreamCancelToken`).
        """

        suppress_tokens = kwargs.pop("suppress_tokens", None)
        self.register_stream_token(cancel_token)

        payload = self._build_stream_payload(
            prompt_text=prompt_text,
            max_tokens=max_tokens,
            top_logprobs=top_logprobs,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            stop_sequences=stop_sequences,
            kwargs=kwargs,
        )
        if suppress_tokens:
            payload["bad_words"] = list(suppress_tokens)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logger.debug(f"STREAM POST {self.completion_endpoint}")
        logger.debug(f"Payload: {json.dumps(payload)[:500]}")

        # ---------- open HTTP stream ---------------------------------------
        last_finish_reason = None
        try:
            # Lease a session for the whole life of this stream, and open the
            # response through it.  Both are released when this generator is
            # exhausted or closed, so a concurrent reset_connection_pool() retires
            # the session instead of closing it and cannot tear this stream down
            # mid-flight.
            with ExitStack() as stack:
                session = stack.enter_context(
                    _lease_session(self.base_url, self.pool_size)
                )
                resp = stack.enter_context(session.post(
                    self.completion_endpoint,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=timeout,
                ))
                if resp.status_code == 400 and "bad_words" in payload:
                    # Not every OpenAI-compatible backend accepts bad_words.
                    # Degrade to a normal request rather than failing the
                    # generation: the sampler has its own scrubber as a fallback.
                    resp.close()
                    logger.warning(
                        "Backend rejected 'bad_words'; disabling token "
                        "suppression for this stream and retrying without it."
                    )
                    self.suppress_tokens_unsupported = True
                    payload.pop("bad_words", None)
                    resp = stack.enter_context(session.post(
                        self.completion_endpoint,
                        headers=headers,
                        json=payload,
                        stream=True,
                        timeout=timeout,
                    ))
                resp.raise_for_status()

                saw_done = False
                cancelled = False
                for raw in self._iter_sse_data(resp):

                    # ---- sampler asked to abort THIS stream? ------------------
                    if (cancel_token is not None and cancel_token.cancelled) or self._cancel_stream:
                        cancelled = True
                        resp.close()
                        break

                    if not raw or not raw.strip():
                        continue

                    if raw.strip() == "[DONE]":
                        saw_done = True
                        break

                    # ---------- parse SSE JSON ---------------------------------
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        # A malformed event means we do not know what text it
                        # carried.  Skipping it would silently drop tokens, so
                        # surface it as an error the sampler can retry instead.
                        logger.warning("Malformed SSE event: %s", raw[:160])
                        raise requests.HTTPError(
                            f"Malformed SSE event from backend: {raw[:160]!r}",
                            response=resp,
                        )

                    # Pass-through OpenAI-style streamed error packets
                    if isinstance(obj, dict) and obj.get("error"):
                        err = obj["error"]
                        msg = (err.get("message") if isinstance(err, dict) else str(err)) or "Upstream stream error"
                        raise requests.HTTPError(msg, response=resp)

                    if not obj.get("choices"):
                        continue
                    choice = obj["choices"][0]

                    # Latch the finish reason: it is usually only present on the
                    # last chunk, and that chunk frequently has no text.  The old
                    # code overwrote the latched value with None on every
                    # ordinary chunk, so the reason was lost entirely.
                    if choice.get("finish_reason"):
                        last_finish_reason = choice["finish_reason"]

                    # -- incremental text fragment ------------------------------
                    generated_text = choice.get("text", "")
                    if not generated_text:
                        # Some providers only fill .delta.content / .message.content
                        delta = choice.get("delta")
                        if isinstance(delta, dict):
                            generated_text = delta.get("content") or ""
                        if not generated_text:
                            message = choice.get("message")
                            if isinstance(message, dict):
                                generated_text = message.get("content") or ""

                    # -- extract logprobs --------------------------------------
                    token_strings: List[str] = []
                    rel_logprobs: Dict[int, List[Tuple[str, float]]] = {}

                    lp_obj = choice.get("logprobs")
                    if lp_obj and isinstance(lp_obj, dict):
                        token_strings = lp_obj.get("tokens", []) or []
                        # vLLM: top_logprobs is list[dict] aligned with tokens
                        for idx, alt_dict in enumerate(lp_obj.get("top_logprobs", []) or []):
                            if isinstance(alt_dict, dict):
                                rel_logprobs[idx] = list(alt_dict.items())

                    # No logprobs but there is text: hand the text over as a
                    # single opaque token so it still reaches the generation
                    # state (and the scrubber) instead of being dropped.
                    if not token_strings and generated_text:
                        token_strings = [generated_text]

                    if not token_strings and not last_finish_reason:
                        continue

                    yield ApiChunkResult(
                        generated_text=generated_text,
                        token_strings=token_strings,
                        logprobs=rel_logprobs,
                        finish_reason=last_finish_reason,
                    )

                # Always emit exactly one terminal chunk.  Without this the
                # sampler's outer loop has no exit condition when the backend
                # closes the socket without sending [DONE] or a finish_reason.
                if cancelled or self._cancel_stream:
                    # A cancelled stream is not a backend event: report it as such
                    # instead of 'stream_end', which the sampler would retry.
                    return

                if saw_done and not last_finish_reason:
                    last_finish_reason = "stream_end"
                elif not last_finish_reason:
                    logger.warning(
                        "Stream closed without [DONE] or finish_reason; "
                        "reporting finish_reason='stream_end' so the sampler can "
                        "decide whether to retry."
                    )
                    last_finish_reason = "stream_end"

                yield ApiChunkResult(
                    generated_text="",
                    token_strings=[],
                    logprobs={},
                    finish_reason=last_finish_reason,
                )

        except requests.exceptions.Timeout:
            logger.warning("Stream timed out. Yielding final chunk with finish_reason='timeout'.")
            yield ApiChunkResult(
                generated_text="",
                token_strings=[],
                logprobs={},
                finish_reason="timeout",
            )
            raise
        except Exception as e:
            if cancel_token is not None and cancel_token.cancelled:
                # Closing the socket from the reader side surfaces as a transport
                # error.  That is an intentional abort, not a backend failure, so
                # do not report it as one.
                logger.debug("Stream cancelled by the sampler: %s", e)
                return
            logger.warning(f"Stream ended unexpectedly: {e}")
            # Always give the caller a terminal chunk so it can distinguish
            # "the backend finished" from "the connection broke".
            yield ApiChunkResult(
                generated_text="",
                token_strings=[],
                logprobs={},
                finish_reason=last_finish_reason or "error",
            )
            raise
        finally:
            self.unregister_stream_token(cancel_token)



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
            # Lease the session for the request and the body read only; the
            # response is fully buffered by resp.json(), so the lease can be
            # released before the (possibly slow) payload parsing below.  A
            # concurrent reset_connection_pool() retires the session rather than
            # closing it while this request is using it.
            with _lease_session(self.base_url, self.pool_size) as session:
                resp = session.post(
                    self.completion_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
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

        logger.debug(f"Raw response: {json.dumps(data)[:2000]}")

        # Pass-through OpenAI-style error embedded in a 200 OK
        if "error" in data and data["error"]:
            err = data["error"]
            msg = (err.get("message") if isinstance(err, dict) else str(err)) or "Upstream error"
            # Attach original response so callers can propagate status/body
            raise requests.HTTPError(msg, response=resp)

        # Treat “no choices” as an upstream failure
        if not data.get("choices"):
            raise requests.HTTPError("Upstream returned no choices", response=resp)


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
