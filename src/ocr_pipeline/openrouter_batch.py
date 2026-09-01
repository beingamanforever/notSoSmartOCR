"""OpenRouter batch calls for public OCR benchmark images."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ocr_pipeline.openrouter import (
    DEFAULT_MAX_TOKENS,
    GEMINI_37_FLASH_BATCH_MODEL,
    GEMINI_37_FLASH_MODEL,
    MAX_TOKENS,
    OpenRouterError,
    _selected_provider,
    _shape_error,
)

BATCHES_URL = "https://openrouter.ai/api/beta/batches"
BATCH_ENDPOINT = "/v1/chat/completions"
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "expired"})
DEFAULT_POLL_INTERVAL_SECONDS = 10.0
DEFAULT_POLL_TIMEOUT_SECONDS = 24 * 60 * 60

Transport = Callable[[Request, float], tuple[int, Mapping[str, str], bytes]]
Sleeper = Callable[[float], None]
Timer = Callable[[], float]


@dataclass(frozen=True)
class BatchItemResult:
    custom_id: str
    content: Any | None = None
    model: str | None = None
    provider: str | None = None
    usage: dict[str, Any] | None = None
    cost: float | None = None
    latency_ms: float | None = None
    finish_reason: str | None = None
    native_finish_reason: str | None = None
    error: OpenRouterError | None = None


@dataclass(frozen=True)
class OpenRouterBatchResult:
    batch_id: str
    status: str
    items: tuple[BatchItemResult, ...]
    latency_ms: float
    polls: int


def repair_images_batch(
    images: Sequence[tuple[str, str | Path]],
    prompt: str,
    response_schema: Mapping[str, Any],
    *,
    model: str = GEMINI_37_FLASH_BATCH_MODEL,
    schema_name: str = "ocr_transcription",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    provider_slug: str,
    batch_id: str | None = None,
    request_timeout_seconds: float = 120,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    poll_timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS,
    transport: Transport | None = None,
    sleeper: Sleeper = time.sleep,
    timer: Timer = time.monotonic,
) -> OpenRouterBatchResult:
    """Submit public images as one batch and return results in input order."""
    _validate_options(
        images,
        prompt,
        response_schema,
        model,
        schema_name,
        max_tokens,
        provider_slug,
        batch_id,
        request_timeout_seconds,
        poll_interval_seconds,
        poll_timeout_seconds,
    )
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY is not set", code="missing_api_key")

    custom_ids = [custom_id for custom_id, _ in images]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Metadata": "enabled",
    }
    call_transport = transport or _default_transport
    started = timer()
    polls = 0
    if batch_id is None:
        requests = [
            {
                "custom_id": custom_id,
                "body": _request_body(
                    image_path,
                    prompt,
                    response_schema,
                    schema_name,
                    max_tokens,
                    provider_slug,
                ),
            }
            for custom_id, image_path in images
        ]
        created = _request_json(
            Request(
                BATCHES_URL,
                data=json.dumps(
                    {
                        "endpoint": BATCH_ENDPOINT,
                        "model": GEMINI_37_FLASH_MODEL,
                        "requests": requests,
                    },
                    separators=(",", ":"),
                ).encode("utf-8"),
                headers=headers,
                method="POST",
            ),
            request_timeout_seconds,
            call_transport,
            "batch_submit",
        )
        batch_id, status = _batch_identity(created, "batch_submit")
        batch = created
        sleep_before_poll = True
    else:
        status = "resuming"
        batch = {"id": batch_id, "status": status}
        sleep_before_poll = False
    while status not in TERMINAL_STATUSES:
        if timer() - started >= poll_timeout_seconds:
            raise OpenRouterError(
                f"OpenRouter batch {batch_id!r} did not finish within "
                f"{poll_timeout_seconds:g} seconds",
                code="batch_poll_timeout",
                latency_ms=round((timer() - started) * 1000, 3),
            )
        if sleep_before_poll:
            sleeper(
                min(
                    poll_interval_seconds,
                    max(0.0, poll_timeout_seconds - (timer() - started)),
                )
            )
        batch = _request_json(
            Request(f"{BATCHES_URL}/{batch_id}", headers=headers, method="GET"),
            request_timeout_seconds,
            call_transport,
            "batch_poll",
        )
        polls += 1
        returned_id, status = _batch_identity(batch, "batch_poll")
        if returned_id != batch_id:
            raise OpenRouterError(
                f"OpenRouter returned batch {returned_id!r}, expected {batch_id!r}",
                code="unexpected_batch_id",
            )
        sleep_before_poll = True

    latency_ms = round((timer() - started) * 1000, 3)
    items = _parse_items(
        batch,
        custom_ids,
        response_schema,
        status,
    )
    return OpenRouterBatchResult(batch_id, status, tuple(items), latency_ms, polls)


def _validate_options(
    images: Sequence[tuple[str, str | Path]],
    prompt: str,
    response_schema: Mapping[str, Any],
    model: str,
    schema_name: str,
    max_tokens: int,
    provider_slug: str,
    batch_id: str | None,
    request_timeout_seconds: float,
    poll_interval_seconds: float,
    poll_timeout_seconds: float,
) -> None:
    if model != GEMINI_37_FLASH_BATCH_MODEL:
        raise OpenRouterError(
            f"Model {model!r} is not approved for public batch image benchmarks",
            code="unsupported_batch_model",
        )
    if not images:
        raise OpenRouterError(
            "At least one batch image is required", code="empty_batch"
        )
    custom_ids = [item[0] for item in images]
    if any(not isinstance(item, str) or not item.strip() for item in custom_ids):
        raise OpenRouterError(
            "Batch custom IDs must be non-empty strings", code="invalid_custom_id"
        )
    if len(set(custom_ids)) != len(custom_ids):
        raise OpenRouterError(
            "Batch custom IDs must be unique", code="duplicate_custom_id"
        )
    if not isinstance(prompt, str) or not prompt.strip():
        raise OpenRouterError("Prompt must not be empty", code="invalid_prompt")
    if not schema_name or not isinstance(response_schema, Mapping):
        raise OpenRouterError(
            "A schema name and response schema are required", code="invalid_schema"
        )
    if isinstance(max_tokens, bool) or not 1 <= max_tokens <= MAX_TOKENS:
        raise OpenRouterError(
            f"Max tokens must be between 1 and {MAX_TOKENS}", code="invalid_max_tokens"
        )
    if (
        not isinstance(provider_slug, str)
        or not provider_slug
        or provider_slug != provider_slug.strip()
    ):
        raise OpenRouterError(
            "Provider slug must be a non-empty trimmed string",
            code="invalid_provider",
        )
    if batch_id is not None and (
        not isinstance(batch_id, str) or not batch_id or batch_id != batch_id.strip()
    ):
        raise OpenRouterError(
            "Batch ID must be a non-empty trimmed string", code="invalid_batch_id"
        )
    if request_timeout_seconds <= 0 or poll_timeout_seconds <= 0:
        raise OpenRouterError("Timeouts must be positive", code="invalid_timeout")
    if poll_interval_seconds < 0:
        raise OpenRouterError(
            "Poll interval must not be negative", code="invalid_poll_interval"
        )


def _request_body(
    image_path: str | Path,
    prompt: str,
    response_schema: Mapping[str, Any],
    schema_name: str,
    max_tokens: int,
    provider_slug: str,
) -> dict[str, Any]:
    image = Path(image_path)
    mime_type, _ = mimetypes.guess_type(image.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise OpenRouterError(
            f"Unsupported image type: {image.suffix or '<none>'}",
            code="unsupported_image_type",
        )
    try:
        image_bytes = image.read_bytes()
    except OSError as error:
        raise OpenRouterError(
            f"Cannot read image: {error}", code="image_read_error"
        ) from error
    if not image_bytes:
        raise OpenRouterError("Cannot send an empty image", code="empty_image")
    image_url = (
        f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    )
    return {
        "model": GEMINI_37_FLASH_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": dict(response_schema),
            },
        },
        "provider": {
            "order": [provider_slug],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
        },
    }


def _request_json(
    request: Request,
    timeout_seconds: float,
    transport: Transport,
    operation: str,
) -> Mapping[str, Any]:
    try:
        status, _, response_body = transport(request, timeout_seconds)
    except TimeoutError as error:
        raise OpenRouterError(
            f"OpenRouter {operation.replace('_', ' ')} timed out",
            code=f"{operation}_timeout",
        ) from error
    except OSError as error:
        raise OpenRouterError(
            f"OpenRouter {operation.replace('_', ' ')} transport failed: {error}",
            code=f"{operation}_transport",
        ) from error
    if status != 200:
        raise OpenRouterError(
            f"OpenRouter {operation.replace('_', ' ')} failed with HTTP {status}: "
            f"{_error_message(response_body)}",
            code=f"{operation}_http_{status}",
            status_code=status,
        )
    try:
        response = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenRouterError(
            f"Invalid OpenRouter {operation.replace('_', ' ')} JSON response",
            code=f"invalid_{operation}_response",
        ) from error
    if not isinstance(response, Mapping):
        raise OpenRouterError(
            f"Invalid OpenRouter {operation.replace('_', ' ')} response",
            code=f"invalid_{operation}_response",
        )
    return response


def _batch_identity(batch: Mapping[str, Any], operation: str) -> tuple[str, str]:
    batch_id = batch.get("id")
    status = batch.get("status")
    if not isinstance(batch_id, str) or not batch_id:
        raise OpenRouterError(
            "OpenRouter batch response is missing a valid ID",
            code=f"invalid_{operation}_response",
        )
    if not isinstance(status, str) or not status:
        raise OpenRouterError(
            "OpenRouter batch response is missing a valid status",
            code=f"invalid_{operation}_response",
        )
    return batch_id, status


def _parse_items(
    batch: Mapping[str, Any],
    custom_ids: list[str],
    response_schema: Mapping[str, Any],
    status: str,
) -> list[BatchItemResult]:
    results = batch.get("results")
    if status != "completed" and results is None:
        error = _batch_terminal_error(batch, status)
        return [BatchItemResult(custom_id, error=error) for custom_id in custom_ids]
    if not isinstance(results, list):
        raise OpenRouterError(
            "Completed OpenRouter batch is missing an inline results list",
            code="invalid_batch_results",
        )

    by_id: dict[str, BatchItemResult] = {}
    expected = set(custom_ids)
    for raw_item in results:
        item = _parse_item(raw_item, response_schema)
        if item.custom_id not in expected:
            raise OpenRouterError(
                f"OpenRouter batch returned unknown custom ID {item.custom_id!r}",
                code="unknown_batch_custom_id",
            )
        if item.custom_id in by_id:
            raise OpenRouterError(
                f"OpenRouter batch returned duplicate custom ID {item.custom_id!r}",
                code="duplicate_batch_result",
            )
        by_id[item.custom_id] = item

    return [
        by_id.get(custom_id)
        or BatchItemResult(
            custom_id,
            error=OpenRouterError(
                "OpenRouter batch returned no result for this request",
                code="missing_batch_result",
            ),
        )
        for custom_id in custom_ids
    ]


def _parse_item(
    raw_item: Any,
    response_schema: Mapping[str, Any],
) -> BatchItemResult:
    if not isinstance(raw_item, Mapping):
        raise OpenRouterError(
            "OpenRouter batch result item is not an object",
            code="invalid_batch_result",
        )
    custom_id = raw_item.get("custom_id")
    if not isinstance(custom_id, str) or not custom_id:
        raise OpenRouterError(
            "OpenRouter batch result is missing a valid custom ID",
            code="invalid_batch_result",
        )
    response_value = raw_item.get("response")
    error_value = raw_item.get("error")
    if (response_value is None) == (error_value is None):
        raise OpenRouterError(
            f"OpenRouter batch result {custom_id!r} must contain one response or error",
            code="invalid_batch_result",
        )
    if error_value is not None:
        return BatchItemResult(custom_id, error=_item_error(error_value))

    response = response_value
    if not isinstance(response, Mapping):
        raise OpenRouterError(
            f"OpenRouter batch response {custom_id!r} is not an object",
            code="invalid_batch_result",
        )
    if "status_code" in response or "body" in response:
        if not isinstance(response.get("status_code"), int) or not isinstance(
            response.get("body"), Mapping
        ):
            raise OpenRouterError(
                f"OpenRouter batch response wrapper {custom_id!r} is malformed",
                code="invalid_batch_result",
            )
        if response["status_code"] != 200:
            return BatchItemResult(
                custom_id,
                error=OpenRouterError(
                    f"OpenRouter batch item failed with HTTP {response['status_code']}: "
                    f"{_mapping_error_message(response['body'])}",
                    code=f"batch_item_http_{response['status_code']}",
                    status_code=response["status_code"],
                ),
            )
        response = response["body"]

    return _parse_success(custom_id, response, response_schema)


def _parse_success(
    custom_id: str,
    response: Mapping[str, Any],
    response_schema: Mapping[str, Any],
) -> BatchItemResult:
    response_model = response.get("model")
    if not isinstance(response_model, str) or response_model != GEMINI_37_FLASH_MODEL:
        return _failed_item(
            custom_id,
            "unexpected_model",
            f"OpenRouter returned model {response_model!r}, expected "
            f"{GEMINI_37_FLASH_MODEL!r}",
        )

    try:
        choices = response["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise TypeError("choices must contain exactly one item")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise TypeError("choice is not an object")
        message = choice["message"]
        if not isinstance(message, Mapping):
            raise TypeError("message is not an object")
        content_text = message["content"]
        if not isinstance(content_text, str):
            raise TypeError("message content is not a string")
        content = json.loads(content_text)
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        return BatchItemResult(
            custom_id,
            error=OpenRouterError(
                f"Invalid OpenRouter batch item response: {error}",
                code="invalid_batch_item_response",
            ),
        )

    selected_provider = _selected_provider(response)
    finish_reason = choice.get("finish_reason")
    if finish_reason != "stop":
        code = "generation_truncated" if finish_reason == "length" else "finish_reason"
        return _failed_item(
            custom_id,
            code,
            f"OpenRouter generation ended with finish reason {finish_reason!r}",
        )
    shape_error = _shape_error(content, response_schema)
    if shape_error:
        return _failed_item(
            custom_id,
            "invalid_content",
            f"OpenRouter content does not match schema: {shape_error}",
        )

    usage_value = response.get("usage")
    if usage_value is not None and not isinstance(usage_value, dict):
        return _failed_item(
            custom_id,
            "invalid_usage",
            "OpenRouter batch item usage is not an object",
        )
    usage = usage_value or {}
    cost_value = usage.get("cost")
    if cost_value is not None and (
        isinstance(cost_value, bool) or not isinstance(cost_value, (int, float))
    ):
        return _failed_item(
            custom_id,
            "invalid_usage",
            "OpenRouter batch item cost is not numeric",
        )
    latency_value = response.get("latency_ms")
    if latency_value is not None and (
        isinstance(latency_value, bool)
        or not isinstance(latency_value, (int, float))
        or latency_value < 0
    ):
        return _failed_item(
            custom_id,
            "invalid_latency",
            "OpenRouter batch item latency is not a non-negative number",
        )
    return BatchItemResult(
        custom_id=custom_id,
        content=content,
        model=response_model,
        provider=selected_provider,
        usage=usage,
        cost=float(cost_value) if cost_value is not None else None,
        latency_ms=float(latency_value) if latency_value is not None else None,
        finish_reason=finish_reason,
        native_finish_reason=(
            str(choice["native_finish_reason"])
            if choice.get("native_finish_reason") is not None
            else None
        ),
    )


def _failed_item(custom_id: str, code: str, message: str) -> BatchItemResult:
    return BatchItemResult(custom_id, error=OpenRouterError(message, code=code))


def _item_error(value: Any) -> OpenRouterError:
    if not isinstance(value, Mapping):
        return OpenRouterError(
            "OpenRouter batch item error is malformed", code="invalid_batch_item_error"
        )
    code = value.get("code")
    message = value.get("message")
    if (
        isinstance(code, bool)
        or not isinstance(code, (str, int))
        or not str(code)
        or not isinstance(message, str)
        or not message
    ):
        return OpenRouterError(
            "OpenRouter batch item error is malformed", code="invalid_batch_item_error"
        )
    status_code = value.get("status_code")
    if status_code is not None and not isinstance(status_code, int):
        return OpenRouterError(
            "OpenRouter batch item error is malformed", code="invalid_batch_item_error"
        )
    return OpenRouterError(message, code=str(code), status_code=status_code)


def _batch_terminal_error(batch: Mapping[str, Any], status: str) -> OpenRouterError:
    error = batch.get("error")
    message = _mapping_error_message(error) if isinstance(error, Mapping) else status
    return OpenRouterError(
        f"OpenRouter batch ended with status {status!r}: {message}",
        code=f"batch_{status}",
    )


def _mapping_error_message(value: Mapping[str, Any]) -> str:
    message = value.get("message")
    if isinstance(message, str) and message:
        return message
    nested = value.get("error")
    if isinstance(nested, Mapping):
        nested_message = nested.get("message")
        if isinstance(nested_message, str) and nested_message:
            return nested_message
    return "no error message"


def _error_message(response_body: bytes) -> str:
    try:
        response = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "invalid error response"
    return (
        _mapping_error_message(response)
        if isinstance(response, Mapping)
        else "invalid error response"
    )


def _default_transport(
    request: Request, timeout_seconds: float
) -> tuple[int, Mapping[str, str], bytes]:
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.status, dict(response.headers.items()), response.read()
    except HTTPError as error:
        headers = dict(error.headers.items()) if error.headers else {}
        return error.code, headers, error.read()
