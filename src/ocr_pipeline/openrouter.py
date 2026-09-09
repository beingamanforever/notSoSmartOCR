"""Minimal OpenRouter calls for evidence-preserving OCR repair."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
QWEN_FLASH_MODEL = "qwen/qwen3.8-flash"
QWEN_37_FLASH_MODEL = "qwen/qwen3.7-flash"
QWEN_25_VL_72B_MODEL = "qwen/qwen2.5-vl-72b-instruct"
GLM_53_FLASH_MODEL = "z-ai/glm-5.3-flash"
MUSE_GLIMMER_MODEL = "meta/muse-glimmer-30b"
GEMMA_MODEL = "google/gemma-4-31b-it"
GEMINI_37_FLASH_MODEL = "google/gemini-3.7-flash"
GEMINI_37_FLASH_BATCH_MODEL = "google/gemini-3.7-flash:batch"
GPT_LUNA_MODEL = "openai/gpt-5.6-luna"
GPT_SOL_MODEL = "openai/gpt-5.6-sol"
CLAUDE_OPUS_MODEL = "anthropic/claude-opus-5"
DEEPSEEK_VISION_MODEL = "deepseek/deepseek-v4-flash-vision-exp"
IMAGE_REPAIR_MODELS = frozenset({MUSE_GLIMMER_MODEL, GEMMA_MODEL})
PUBLIC_BENCHMARK_IMAGE_MODELS = frozenset(
    {
        *IMAGE_REPAIR_MODELS,
        QWEN_FLASH_MODEL,
        QWEN_37_FLASH_MODEL,
        QWEN_25_VL_72B_MODEL,
        GLM_53_FLASH_MODEL,
        DEEPSEEK_VISION_MODEL,
        GEMINI_37_FLASH_MODEL,
        GPT_LUNA_MODEL,
        GPT_SOL_MODEL,
        CLAUDE_OPUS_MODEL,
    }
)
PUBLIC_BATCH_IMAGE_MODELS = frozenset({GEMINI_37_FLASH_BATCH_MODEL})
MAX_RETRY_DELAY_SECONDS = 10.0
DEFAULT_MAX_TOKENS = 2048
MAX_TOKENS = 16384

Transport = Callable[[Request, float], tuple[int, Mapping[str, str], bytes]]
Sleeper = Callable[[float], None]


class OpenRouterError(RuntimeError):
    """An invalid request, response, or exhausted OpenRouter call."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "openrouter_error",
        status_code: int | None = None,
        attempts: int | None = None,
        latency_ms: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.attempts = attempts
        self.latency_ms = latency_ms


@dataclass(frozen=True)
class OpenRouterResult:
    content: Any
    model: str
    provider: str | None
    usage: dict[str, Any]
    cost: float | None
    latency_ms: float
    attempts: int
    finish_reason: str = "stop"
    native_finish_reason: str | None = None


def repair_image(
    image_path: str | Path,
    prompt: str,
    response_schema: Mapping[str, Any],
    *,
    model: str = MUSE_GLIMMER_MODEL,
    schema_name: str = "image_repair",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    provider_slug: str | None = None,
    timeout_seconds: float = 120,
    max_attempts: int = 3,
    public_benchmark: bool = False,
    transport: Transport | None = None,
    sleeper: Sleeper = time.sleep,
) -> OpenRouterResult:
    """Repair OCR from a local image using an approved multimodal model."""
    approved_models = (
        PUBLIC_BENCHMARK_IMAGE_MODELS if public_benchmark else IMAGE_REPAIR_MODELS
    )
    if model not in approved_models:
        raise OpenRouterError(f"Model {model!r} is not approved for image repair")

    image = Path(image_path)
    mime_type, _ = mimetypes.guess_type(image.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise OpenRouterError(f"Unsupported image type: {image.suffix or '<none>'}")
    try:
        image_bytes = image.read_bytes()
    except OSError as error:
        raise OpenRouterError(f"Cannot read image: {error}") from error
    if not image_bytes:
        raise OpenRouterError("Cannot send an empty image")

    image_url = (
        f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]
    return _call_openrouter(
        model,
        messages,
        response_schema,
        schema_name,
        max_tokens,
        provider_slug,
        timeout_seconds,
        max_attempts,
        model != DEEPSEEK_VISION_MODEL,
        transport or _default_transport,
        sleeper,
    )


def _call_openrouter(
    model: str,
    messages: list[dict[str, Any]],
    response_schema: Mapping[str, Any],
    schema_name: str,
    max_tokens: int,
    provider_slug: str | None,
    timeout_seconds: float,
    max_attempts: int,
    strict_schema: bool,
    transport: Transport,
    sleeper: Sleeper,
    *,
    zero_data_retention: bool = True,
    reasoning_enabled: bool | None = None,
    session_id: str | None = None,
) -> OpenRouterResult:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY is not set")
    if not _prompt_is_valid(messages):
        raise OpenRouterError("Prompt must not be empty")
    if not schema_name or not isinstance(response_schema, Mapping):
        raise OpenRouterError("A schema name and response schema are required")
    if isinstance(max_tokens, bool) or not 1 <= max_tokens <= MAX_TOKENS:
        raise OpenRouterError(f"Max tokens must be between 1 and {MAX_TOKENS}")
    if provider_slug is not None and (
        not isinstance(provider_slug, str)
        or not provider_slug
        or provider_slug != provider_slug.strip()
    ):
        raise OpenRouterError("Provider slug must be a non-empty trimmed string")
    if timeout_seconds <= 0 or max_attempts < 1:
        raise OpenRouterError("Timeout and max attempts must be positive")
    if not isinstance(zero_data_retention, bool):
        raise OpenRouterError("zero_data_retention must be a boolean")
    if session_id is not None and (
        not isinstance(session_id, str)
        or not session_id.strip()
        or session_id != session_id.strip()
        or len(session_id) > 256
    ):
        raise OpenRouterError(
            "session_id must be a trimmed string of 1 to 256 characters"
        )

    provider: dict[str, Any] = {
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": zero_data_retention,
    }
    if provider_slug:
        provider["order"] = [provider_slug]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": dict(response_schema),
                },
            }
            if strict_schema
            else {"type": "json_object"}
        ),
        "provider": provider,
    }
    if reasoning_enabled is not None:
        if not isinstance(reasoning_enabled, bool):
            raise OpenRouterError("reasoning_enabled must be a boolean")
        payload["reasoning"] = {"enabled": reasoning_enabled}
    if session_id is not None:
        payload["session_id"] = session_id
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    started = time.perf_counter()

    for attempt in range(1, max_attempts + 1):
        request = Request(
            OPENROUTER_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Metadata": "enabled",
            },
            method="POST",
        )
        try:
            status, headers, response_body = transport(request, timeout_seconds)
        except TimeoutError as error:
            if attempt == max_attempts:
                raise OpenRouterError(
                    f"OpenRouter timed out after {attempt} attempts",
                    code="timeout_exhausted",
                    attempts=attempt,
                    latency_ms=_elapsed_ms(started),
                ) from error
            sleeper(_retry_delay({}, attempt))
            continue
        except OSError as error:
            if attempt == max_attempts:
                raise OpenRouterError(
                    f"OpenRouter transport exhausted {attempt} attempts: {error}",
                    code="transport_exhausted",
                    attempts=attempt,
                    latency_ms=_elapsed_ms(started),
                ) from error
            sleeper(_retry_delay({}, attempt))
            continue

        if status == 200:
            return _parse_result(
                response_body,
                response_schema,
                model,
                attempt,
                started,
            )

        retryable = status == 429 or 500 <= status <= 599
        error_message = _error_message(response_body)
        if not retryable:
            raise OpenRouterError(
                f"OpenRouter request failed with HTTP {status}: {error_message}",
                code=f"http_{status}",
                status_code=status,
                attempts=attempt,
                latency_ms=_elapsed_ms(started),
            )
        if attempt == max_attempts:
            raise OpenRouterError(
                f"OpenRouter request exhausted {attempt} attempts with HTTP "
                f"{status}: {error_message}",
                code=f"http_{status}_exhausted",
                status_code=status,
                attempts=attempt,
                latency_ms=_elapsed_ms(started),
            )
        sleeper(_retry_delay(headers, attempt))

    raise AssertionError("unreachable")


def _prompt_is_valid(messages: list[dict[str, Any]]) -> bool:
    first_content = messages[0].get("content") if messages else None
    if isinstance(first_content, str):
        return bool(first_content.strip())
    if isinstance(first_content, list) and first_content:
        text = first_content[0]
        return isinstance(text, Mapping) and bool(str(text.get("text", "")).strip())
    return False


def _parse_result(
    response_body: bytes,
    response_schema: Mapping[str, Any],
    requested_model: str,
    attempts: int,
    started: float,
) -> OpenRouterResult:
    try:
        response = json.loads(response_body)
        choice = response["choices"][0]
        if not isinstance(choice, Mapping):
            raise TypeError("choice is not an object")
        content_text = choice["message"]["content"]
        if not isinstance(content_text, str):
            raise TypeError("message content is not a string")
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise OpenRouterError(
            f"Invalid OpenRouter JSON response: {error}",
            code="invalid_response",
            attempts=attempts,
            latency_ms=_elapsed_ms(started),
        ) from error

    finish_reason = choice.get("finish_reason")
    native_finish_reason = choice.get("native_finish_reason")
    response_model = response.get("model")
    if response_model != requested_model:
        raise OpenRouterError(
            f"OpenRouter returned model {response_model!r}, expected {requested_model!r}",
            code="unexpected_model",
            attempts=attempts,
            latency_ms=_elapsed_ms(started),
        )
    if finish_reason != "stop":
        code = "generation_truncated" if finish_reason == "length" else "finish_reason"
        raise OpenRouterError(
            f"OpenRouter generation ended with finish reason {finish_reason!r}",
            code=code,
            attempts=attempts,
            latency_ms=_elapsed_ms(started),
        )
    try:
        content = json.loads(content_text)
    except json.JSONDecodeError as error:
        raise OpenRouterError(
            f"Invalid OpenRouter JSON response: {error}",
            code="invalid_response",
            attempts=attempts,
            latency_ms=_elapsed_ms(started),
        ) from error

    shape_error = _shape_error(content, response_schema)
    if shape_error:
        raise OpenRouterError(
            f"OpenRouter content does not match schema: {shape_error}",
            code="invalid_content",
            attempts=attempts,
            latency_ms=_elapsed_ms(started),
        )

    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise OpenRouterError(
            "Invalid OpenRouter response: missing usage object",
            code="invalid_usage",
            attempts=attempts,
            latency_ms=_elapsed_ms(started),
        )
    cost_value = usage.get("cost")
    if cost_value is not None and (
        isinstance(cost_value, bool) or not isinstance(cost_value, (int, float))
    ):
        raise OpenRouterError(
            "Invalid OpenRouter response: usage cost is not numeric",
            code="invalid_usage",
            attempts=attempts,
            latency_ms=_elapsed_ms(started),
        )

    return OpenRouterResult(
        content=content,
        model=requested_model,
        provider=_selected_provider(response),
        usage=usage,
        cost=float(cost_value) if cost_value is not None else None,
        latency_ms=_elapsed_ms(started),
        attempts=attempts,
        finish_reason=finish_reason,
        native_finish_reason=(
            str(native_finish_reason) if native_finish_reason is not None else None
        ),
    )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _shape_error(value: Any, schema: Mapping[str, Any], path: str = "$") -> str | None:
    expected_type = schema.get("type")
    type_matches = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if isinstance(expected_type, str) and not type_matches.get(expected_type, False):
        return f"{path} must be {expected_type}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} is not an allowed value"

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            return f"{path} is missing {', '.join(missing)}"
        if schema.get("additionalProperties") is False:
            extra = [name for name in value if name not in properties]
            if extra:
                return f"{path} has unexpected {', '.join(extra)}"
        for name, child_schema in properties.items():
            if name in value and isinstance(child_schema, Mapping):
                error = _shape_error(value[name], child_schema, f"{path}.{name}")
                if error:
                    return error

    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            error = _shape_error(item, schema["items"], f"{path}[{index}]")
            if error:
                return error
    return None


def _selected_provider(response: Mapping[str, Any]) -> str | None:
    provider = response.get("provider")
    if isinstance(provider, str):
        return provider
    metadata = response.get("openrouter_metadata")
    if not isinstance(metadata, Mapping):
        return None
    endpoints = metadata.get("endpoints")
    if not isinstance(endpoints, Mapping):
        return None
    available = endpoints.get("available")
    if not isinstance(available, list):
        return None
    for endpoint in available:
        if isinstance(endpoint, Mapping) and endpoint.get("selected") is True:
            selected = endpoint.get("provider")
            return selected if isinstance(selected, str) else None
    return None


def _retry_delay(headers: Mapping[str, str], attempt: int) -> float:
    retry_after = next(
        (value for name, value in headers.items() if name.lower() == "retry-after"),
        None,
    )
    try:
        requested_delay = (
            float(retry_after) if retry_after is not None else 2 ** (attempt - 1)
        )
    except ValueError:
        requested_delay = 2 ** (attempt - 1)
    return max(0.0, min(requested_delay, MAX_RETRY_DELAY_SECONDS))


def _error_message(response_body: bytes) -> str:
    try:
        response = json.loads(response_body)
        error = response.get("error", {})
        message = error.get("message") if isinstance(error, Mapping) else None
        return str(message or "no error message")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "invalid error response"


def _default_transport(
    request: Request, timeout_seconds: float
) -> tuple[int, Mapping[str, str], bytes]:
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.status, dict(response.headers.items()), response.read()
    except HTTPError as error:
        headers = dict(error.headers.items()) if error.headers else {}
        return error.code, headers, error.read()
