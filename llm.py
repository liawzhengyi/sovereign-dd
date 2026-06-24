"""LLM wrapper â€" Gemma 4 31B, key rotation, auto model-ID detection, grounding, JSON extraction."""

import asyncio
import json
import os
import random
import re
import threading
import time
from itertools import cycle

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_raw_keys = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
_keys = [k.strip() for k in _raw_keys.split(",") if k.strip()]
if not _keys:
    raise RuntimeError("No GEMINI_API_KEYS found in .env")

_key_cycle = cycle(_keys)
_key_lock  = threading.Lock()       # thread-safe rotation for concurrent async calls
_clients: dict[str, genai.Client] = {}
_model_ids: dict[str, str] = {}     # key -> verified model ID string

# Hard timeouts. Without them a stalled HTTP connection blocks generate_content
# FOREVER — on 2026-06-10 a moderator call hung 5.5h until GitHub killed the job
# at the 6-hour limit (and the queued next run was auto-cancelled). The SDK
# timeout is per-request (milliseconds); the asyncio guard is a belt-and-braces
# outer bound on the whole thread in case the SDK timeout fails to fire.
_HTTP_TIMEOUT_MS    = int(os.getenv("LLM_HTTP_TIMEOUT_MS", "240000"))   # 4 min
_CALL_TIMEOUT_SECS  = int(os.getenv("LLM_CALL_TIMEOUT_SECS", "300"))    # 5 min

_key_cooldowns: dict[str, float] = {}   # key -> unix timestamp when available again
_key_daily_exhausted: set[str] = set() # keys with daily RPD quota consumed
_cooldown_lock = threading.Lock()


def _is_daily_exhausted(err_str: str) -> bool:
    """True when the error is daily-quota (RPD) exhaustion, not an RPM spike.

    Google uses the SAME quota_metric name for both RPM and RPD limits:
      quota_metric: generate_content_free_tier_requests
    The only reliable differentiator is the quota_limit field:
      RPM → ...per_minute_per_project_per_base_model  (contains "per_minute")
      RPD → ...per_day_per_project_per_base_model     (contains "per_day")

    So we MUST check for "per_minute" first and short-circuit to False —
    any check on "free_tier_requests" alone will mis-fire on RPM errors.
    """
    if "per_minute" in err_str or "per-minute" in err_str:
        return False   # RPM throttle — temporary, key recovers within 60s
    return (
        "per_day"             in err_str   # quota_limit field for RPD limit
        or "generaterequestsperday" in err_str  # GenerateRequestsPerDayPerProjectPerModel
        or "daily quota"      in err_str   # human-readable fallback
    )


def _new_client(key: str) -> genai.Client:
    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=_HTTP_TIMEOUT_MS),
    )


def _client() -> tuple[genai.Client, str]:
    """Return (client, api_key) for the next key in rotation. Thread-safe."""
    with _key_lock:
        key = next(_key_cycle)
    if key not in _clients:
        _clients[key] = _new_client(key)
    return _clients[key], key


def _client_for(key: str) -> genai.Client:
    """Return (or create) a client for a specific key."""
    if key not in _clients:
        _clients[key] = _new_client(key)
    return _clients[key]


def _pick_key() -> tuple[str | None, float]:
    """Return (key, wait_secs) for the soonest-available non-daily-exhausted key.

    Returns (None, 0) when every key has hit its daily RPD quota — callers
    should raise immediately rather than burning retries on dead keys.
    """
    now = time.time()
    with _cooldown_lock:
        live = [k for k in _keys if k not in _key_daily_exhausted]
        if not live:
            return None, 0.0
        best = min(live, key=lambda k: _key_cooldowns.get(k, 0.0))
        wait = max(0.0, _key_cooldowns.get(best, 0.0) - now)
        return best, wait


def _cool_key(key: str, duration: float) -> None:
    """Mark a key as RPM-throttled for duration seconds."""
    with _cooldown_lock:
        _key_cooldowns[key] = time.time() + duration
    print(f"  [llm] key {_key_label(key)} cooling for {duration:.0f}s")


def _exhaust_key(key: str) -> None:
    """Mark a key as daily-quota-exhausted — offline until midnight UTC + 5 min buffer."""
    from datetime import datetime, timezone, timedelta
    now_utc = datetime.now(timezone.utc)
    midnight = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    secs = (midnight - now_utc).total_seconds() + 300  # +5 min buffer
    with _cooldown_lock:
        _key_cooldowns[key] = time.time() + secs
        _key_daily_exhausted.add(key)
    live = len(_keys) - len(_key_daily_exhausted)
    print(f"  [llm] key {_key_label(key)} DAILY QUOTA EXHAUSTED "
          f"({live}/{len(_keys)} keys remaining today)")


def _resolve_model(client: genai.Client, key: str, model: str) -> str:
    """Return the working model ID string for this key (cached after first call)."""
    if key in _model_ids:
        return _model_ids[key]
    for candidate in [model, f"models/{model}"]:
        try:
            client.models.generate_content(
                model=candidate,
                contents="hi",
                config=types.GenerateContentConfig(max_output_tokens=1),
            )
            _model_ids[key] = candidate
            return candidate
        except Exception:
            continue
    # Both probes failed (likely transient 429/auth). Return a default WITHOUT caching
    # so a transient failure doesn't poison this key's model ID for the whole process.
    return model


def _jittered(base: float) -> float:
    """Add +/-25% jitter to a backoff delay to desynchronise concurrent retries."""
    return base * (0.75 + random.random() * 0.5)


def _key_label(key: str) -> str:
    """Stable, non-secret label for a key (its 1-based position in the pool)."""
    try:
        return f"#{_keys.index(key) + 1}"
    except ValueError:
        return "#?"


def _hit_max_tokens(response) -> bool:
    """True if the model stopped because it hit the output token cap (truncated)."""
    try:
        fr = response.candidates[0].finish_reason
        return fr is not None and "MAX_TOKENS" in str(fr).upper()
    except (AttributeError, IndexError, TypeError):
        return False


def call_gemini(
    system: str,
    user: str,
    model: str = "gemma-4-31b-it",
    temperature: float = 0.3,
    max_retries: int = 12,
    grounding: bool = False,
    api_key: str | None = None,
    max_output_tokens: int = 32768,
    thinking_level: str | None = "high",
) -> str:
    """Call the model and return raw text. Retries on 429/500/503. Thread-safe key rotation.

    Distinguishes RPM throttles (temporary, retry with backoff) from RPD exhaustion
    (daily quota dead — marks key offline until midnight UTC, stops retrying it).
    """
    last_err = None
    for attempt in range(max_retries):
        # When called from call_gemini_async a specific key is passed; otherwise
        # rotate through keys while skipping any that are daily-exhausted.
        if api_key is not None:
            cur_key = api_key
        else:
            cur_key, wait = _pick_key()
            if cur_key is None:
                raise RuntimeError(
                    "All API keys have exhausted their daily quota. "
                    "Quota resets at midnight UTC."
                )
            if wait > 0:
                time.sleep(wait)

        try:
            client = _client_for(cur_key)
            resolved = _resolve_model(client, cur_key, model)

            def _generate(thinking: str | None):
                cfg: dict = dict(
                    system_instruction=system,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                # thinking is optional. None omits thinking_config entirely (no thinking,
                # full output budget). NOTE: this model accepts thinking_level="high" but
                # REJECTS "low"/"medium" with a hard 400 ("Thinking level is not supported
                # for this model"), so passing an unsupported level silently killed every
                # R2/R3 debate round. Prefer None over a low level to free output budget.
                if thinking is not None:
                    cfg["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
                if grounding:
                    cfg["tools"] = [{"google_search": {}}]
                try:
                    return client.models.generate_content(
                        model=resolved,
                        contents=user,
                        config=types.GenerateContentConfig(**cfg),
                    )
                except Exception as e:
                    # Self-heal: if the model rejects the requested thinking level,
                    # retry once WITHOUT thinking so the agent still produces an answer
                    # rather than hard-failing (which silently drops it from consensus).
                    es = str(e).lower()
                    if thinking is not None and "thinking" in es and "400" in es:
                        cfg.pop("thinking_config", None)
                        return client.models.generate_content(
                            model=resolved,
                            contents=user,
                            config=types.GenerateContentConfig(**cfg),
                        )
                    raise

            response = _generate(thinking_level)
            text = response.text or ""
            # If the model truncated (finish_reason MAX_TOKENS) the JSON is cut off and
            # downstream extract_json fails -> the agent is silently scored 5.0. Retry
            # once WITHOUT thinking to free the full output budget for the answer.
            if not text or _hit_max_tokens(response):
                response = _generate(None)
                text = response.text or text
            return text
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "503" in err_str:
                if _is_daily_exhausted(err_str):
                    _exhaust_key(cur_key)
                    # If a specific key was passed in and it's daily-exhausted, nothing
                    # we can do here — the async wrapper picks the next key on re-entry.
                    if api_key is not None:
                        raise
                    # Otherwise loop immediately and _pick_key() will skip this key.
                    continue
                wait = _jittered(min(2 ** (attempt + 1), 30))
                print(f"  [llm] rate limit on attempt {attempt + 1}, retrying in {wait:.0f}s...")
                _cool_key(cur_key, wait)
                if api_key is not None:
                    time.sleep(wait)  # sync path: sleep here; async path handles sleep outside
            elif "500" in err_str or "502" in err_str:
                if attempt < max_retries - 1:
                    wait = _jittered(min(2 ** (attempt + 1), 30))
                    print(f"  [llm] server error on attempt {attempt + 1}, retrying in {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    raise
            else:
                raise
    raise RuntimeError(f"LLM failed after {max_retries} attempts: {last_err}")


_api_semaphore: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    """Lazy-init semaphore sized to number of API keys (max concurrent calls = num keys)."""
    global _api_semaphore
    if _api_semaphore is None:
        _api_semaphore = asyncio.Semaphore(len(_keys))
    return _api_semaphore


async def call_gemini_async(
    system: str,
    user: str,
    model: str = "gemma-4-31b-it",
    temperature: float = 0.3,
    max_retries: int = 12,
    grounding: bool = False,
    max_output_tokens: int = 32768,
    thinking_level: str | None = "high",
) -> str:
    """Async wrapper with per-key cooldown tracking to prevent thundering herd.

    On each attempt:
      1. _pick_key() selects the soonest-available non-daily-exhausted key.
         Returns None if every key has hit its daily RPD quota — raise immediately.
      2. If all live keys are RPM-cooling, sleep until the best one is ready.
      3. On 429 + daily quota signal: _exhaust_key() marks it offline until midnight
         UTC and loops to the next key without sleeping.
      4. On 429 + RPM spike: _cool_key() for short backoff, then try next key.
      5. On 5xx: brief sleep without cooling the key.
    """
    last_err = None
    for attempt in range(max_retries):
        key, wait = _pick_key()
        if key is None:
            raise RuntimeError(
                "All API keys have exhausted their daily quota. "
                "Quota resets at midnight UTC."
            )
        if wait > 0:
            wait += random.uniform(0, 2)  # jitter to spread concurrent waiters
            print(f"  [llm] all keys cooling, waiting {wait:.0f}s (attempt {attempt + 1})...")
            await asyncio.sleep(wait)

        server_err_wait = 0.0
        async with _semaphore():
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        call_gemini, system, user, model, temperature, 1, grounding, key,
                        max_output_tokens, thinking_level
                    ),
                    timeout=_CALL_TIMEOUT_SECS,
                )
            except asyncio.TimeoutError as e:
                # The worker thread is abandoned (the SDK's own HTTP timeout will
                # reap it); treat like a transient server error and try another key.
                last_err = e
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"LLM call timed out after {_CALL_TIMEOUT_SECS}s on final attempt"
                    ) from e
                print(f"  [llm] call timed out after {_CALL_TIMEOUT_SECS}s "
                      f"(attempt {attempt + 1}), retrying on another key...")
                _cool_key(key, _jittered(30))
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                # thinking-400: _generate's self-heal normally absorbs this; if one
                # still escapes (seen 2026-06-10 when Google transiently rejected
                # thinking on requests that didn't even send it), it's a server-side
                # incident — retry on another key rather than failing the agent.
                is_retryable = (
                    "429" in err_str or "quota" in err_str
                    or "503" in err_str or "500" in err_str or "502" in err_str
                    or "504" in err_str or "deadline" in err_str
                    or ("thinking" in err_str and "400" in err_str)
                )
                if not is_retryable or attempt == max_retries - 1:
                    raise
                if "429" in err_str or "quota" in err_str:
                    if _is_daily_exhausted(err_str):
                        # Daily quota dead — no point sleeping; loop immediately,
                        # _pick_key() will skip this key for the rest of the day.
                        _exhaust_key(key)
                    else:
                        # RPM spike — cool for a full 60s window + jitter so the
                        # key is guaranteed clear before we retry it.  Other keys
                        # are tried immediately via _pick_key() rotation.
                        _cool_key(key, _jittered(65))
                else:
                    server_err_wait = _jittered(min(2 ** (attempt + 1), 10))

        if server_err_wait > 0:
            print(f"  [llm] server error on attempt {attempt + 1}, retrying in {server_err_wait:.0f}s...")
            await asyncio.sleep(server_err_wait)

    raise RuntimeError(f"LLM failed after {max_retries} attempts: {last_err}")


async def call_gemini_with_tools_async(
    system: str,
    user: str,
    tools: "types.Tool",
    tool_executor,          # callable: (fn_name, fn_args, **kwargs) -> Any
    tool_executor_kwargs: dict | None = None,
    model: str = "gemma-4-31b-it",
    temperature: float = 0.2,
    max_tool_turns: int = 6,
    max_retries: int = 8,
) -> str:
    """Multi-turn function-calling loop for Gemini models.

    Sends system+user prompt with tools declared. On each model turn:
      - If the model returns function_call parts, execute them via tool_executor,
        send results back as tool role messages, and continue.
      - If the model returns a text response (no more function calls), return it.
      - Stop after max_tool_turns to prevent infinite loops.

    Args:
        system: System prompt string.
        user: User message string.
        tools: google.genai types.Tool with function declarations.
        tool_executor: Callable(fn_name: str, fn_args: dict, **tool_executor_kwargs) -> Any.
        tool_executor_kwargs: Extra kwargs forwarded to tool_executor on every call.
        model: Model string (default gemma-4-31b-it).
        temperature: Sampling temperature.
        max_tool_turns: Max number of tool call rounds before forcing a final text answer.
        max_retries: Max API retry attempts (reuses existing key rotation logic).

    Returns:
        Final text response from the model as a string.
    """
    from google.genai import types as _types

    tool_executor_kwargs = tool_executor_kwargs or {}

    # Build initial contents list
    contents: list[_types.Content] = [
        _types.Content(role="user", parts=[_types.Part(text=user)])
    ]

    def _build_config(with_thinking: bool) -> "_types.GenerateContentConfig":
        cfg: dict = dict(
            system_instruction=system,
            tools=[tools],
            temperature=temperature,
            max_output_tokens=32768,
        )
        if with_thinking:
            cfg["thinking_config"] = _types.ThinkingConfig(thinking_level="high")
        return _types.GenerateContentConfig(**cfg)

    config = _build_config(with_thinking=True)

    for turn in range(max_tool_turns + 1):
        # Make API call with retries (reuse existing key rotation)
        last_err = None
        response = None
        for attempt in range(max_retries):
            key, wait = _pick_key()
            if key is None:
                raise RuntimeError(
                    "All API keys have exhausted their daily quota. "
                    "Quota resets at midnight UTC."
                )
            if wait > 0:
                wait += random.uniform(0, 2)
                await asyncio.sleep(wait)

            server_err_wait = 0.0
            async with _semaphore():
                try:
                    client = _client_for(key)
                    resolved_model = _resolve_model(client, key, model)
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            client.models.generate_content,
                            model=resolved_model,
                            contents=contents,
                            config=config,
                        ),
                        timeout=_CALL_TIMEOUT_SECS,
                    )
                    break  # success
                except asyncio.TimeoutError as e:
                    last_err = e
                    if attempt == max_retries - 1:
                        raise RuntimeError(
                            f"LLM tool call timed out after {_CALL_TIMEOUT_SECS}s on final attempt"
                        ) from e
                    print(f"  [llm] tool call timed out after {_CALL_TIMEOUT_SECS}s "
                          f"(attempt {attempt + 1}), retrying on another key...")
                    _cool_key(key, _jittered(30))
                except Exception as e:
                    last_err = e
                    err_str = str(e).lower()
                    # Self-heal: model rejected the thinking level (same failure mode
                    # call_gemini's _generate absorbs) — drop thinking and retry.
                    if "thinking" in err_str and "400" in err_str:
                        if attempt == max_retries - 1:
                            raise
                        config = _build_config(with_thinking=False)
                        continue
                    is_retryable = (
                        "429" in err_str or "quota" in err_str
                        or "503" in err_str or "500" in err_str or "502" in err_str
                        or "504" in err_str or "deadline" in err_str
                    )
                    if not is_retryable or attempt == max_retries - 1:
                        raise
                    if "429" in err_str or "quota" in err_str:
                        if _is_daily_exhausted(err_str):
                            _exhaust_key(key)
                        else:
                            _cool_key(key, _jittered(min(2 ** (attempt + 1), 30)))
                    else:
                        # Set the backoff but DON'T sleep while holding the semaphore —
                        # that serializes all retries under load. Sleep after release.
                        server_err_wait = _jittered(min(2 ** (attempt + 1), 10))
            if server_err_wait > 0:
                print(f"  [llm] tool-call server error attempt {attempt + 1}, retrying in {server_err_wait:.0f}s...")
                await asyncio.sleep(server_err_wait)
        else:
            raise RuntimeError(f"LLM tool call failed after {max_retries} attempts: {last_err}")

        assert response is not None  # for/else guarantees this

        # Check for function call parts
        fn_call_parts = [
            part for part in (response.candidates[0].content.parts or [])
            if hasattr(part, "function_call") and part.function_call is not None
        ]

        if not fn_call_parts:
            # Normal exit — model gave text response
            text_parts = [
                part.text for part in (response.candidates[0].content.parts or [])
                if hasattr(part, "text") and part.text
            ]
            return "\n".join(text_parts).strip() if text_parts else ""

        if turn == max_tool_turns:
            # Forced exit — model still issuing tool calls; re-prompt for final prose answer
            final_user = "You have used the available tools. Now provide your final analysis and recommendations in plain text."
            accumulated = "\n".join(
                part.text
                for c in contents
                for part in (c.parts or [])
                if hasattr(part, "text") and part.text
            )
            final_prompt = f"{accumulated}\n\n{final_user}" if accumulated else final_user
            return await call_gemini_async(system, final_prompt, model=model, temperature=temperature)

        # Add model's function call response to contents
        contents.append(response.candidates[0].content)

        # Execute each function call and add results
        tool_response_parts = []
        for part in fn_call_parts:
            fc = part.function_call
            fn_args = dict(fc.args) if fc.args else {}
            result = tool_executor(fc.name, fn_args, **tool_executor_kwargs)
            tool_response_parts.append(
                _types.Part(
                    function_response=_types.FunctionResponse(
                        name=fc.name,
                        response={"result": result},
                    )
                )
            )

        contents.append(
            _types.Content(role="tool", parts=tool_response_parts)
        )

    # Fallback — should not reach here
    return ""


def extract_json(text: str) -> dict | list:
    """Extract the first JSON object or array from a text response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    for start_char, end_char in [('{', '}'), ('[', ']')]:
        idx = text.find(start_char)
        if idx == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(text[idx:], start=idx):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[idx:i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"No valid JSON found in response:\n{text[:400]}")
