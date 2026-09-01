from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.request import Request

import pytest

from ocr_pipeline.openrouter import (
    CLAUDE_OPUS_MODEL,
    DEEPSEEK_VISION_MODEL,
    GEMMA_MODEL,
    GEMINI_37_FLASH_BATCH_MODEL,
    GEMINI_37_FLASH_MODEL,
    GLM_53_FLASH_MODEL,
    MAX_TOKENS,
    MAX_RETRY_DELAY_SECONDS,
    MUSE_GLIMMER_MODEL,
    QWEN_37_FLASH_MODEL,
    QWEN_25_VL_72B_MODEL,
    QWEN_FLASH_MODEL,
    OpenRouterError,
    repair_image,
)

TEXT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def test_repair_image_sends_strict_private_request_and_returns_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    image = tmp_path / "crop.png"
    image.write_bytes(b"local-image-bytes")
    captured: dict[str, object] = {}

    def transport(request: Request, timeout: float):
        captured["request"] = request
        captured["timeout"] = timeout
        return 200, {}, _success_response('{"text":"repaired"}')

    result = repair_image(
        image,
        "Transcribe exactly",
        TEXT_SCHEMA,
        model=MUSE_GLIMMER_MODEL,
        max_tokens=512,
        provider_slug="test-provider",
        transport=transport,
    )

    request = captured["request"]
    assert isinstance(request, Request)
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.method == "POST"
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert request.get_header("X-openrouter-metadata") == "enabled"
    payload = json.loads(request.data)
    assert payload["model"] == MUSE_GLIMMER_MODEL
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 512
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "image_repair",
            "strict": True,
            "schema": TEXT_SCHEMA,
        },
    }
    assert payload["provider"] == {
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
        "order": ["test-provider"],
    }
    image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url == (
        "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
    )
    assert result.content == {"text": "repaired"}
    assert result.provider == "Test Provider"
    assert result.cost == 0.00001
    assert result.attempts == 1
    assert "test-secret" not in repr(result)


def test_image_model_roles_are_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    image = tmp_path / "crop.png"
    image.write_bytes(b"image")

    with pytest.raises(OpenRouterError, match="not approved for image repair"):
        repair_image(image, "Read", TEXT_SCHEMA, model="qwen/qwen3-32b")

    with pytest.raises(OpenRouterError, match="not approved for image repair"):
        repair_image(image, "Read", TEXT_SCHEMA, model=QWEN_FLASH_MODEL)

    with pytest.raises(OpenRouterError, match="not approved for image repair"):
        repair_image(
            image,
            "Read",
            TEXT_SCHEMA,
            model="qwen/qwen3-32b",
            public_benchmark=True,
        )


@pytest.mark.parametrize(
    "model",
    [
        QWEN_37_FLASH_MODEL,
        QWEN_25_VL_72B_MODEL,
        GLM_53_FLASH_MODEL,
        GEMINI_37_FLASH_MODEL,
        CLAUDE_OPUS_MODEL,
    ],
)
def test_current_frontier_models_are_public_benchmark_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    image = tmp_path / "crop.png"
    image.write_bytes(b"image")

    with pytest.raises(OpenRouterError, match="not approved for image repair"):
        repair_image(image, "Read", TEXT_SCHEMA, model=model)

    result = repair_image(
        image,
        "Read",
        TEXT_SCHEMA,
        model=model,
        public_benchmark=True,
        transport=lambda request, timeout: (
            200,
            {},
            _success_response('{"text":"public"}', model=model),
        ),
    )
    assert result.content == {"text": "public"}


def test_batch_model_is_not_sent_to_synchronous_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    image = tmp_path / "crop.png"
    image.write_bytes(b"image")

    with pytest.raises(OpenRouterError, match="not approved for image repair"):
        repair_image(
            image,
            "Read",
            TEXT_SCHEMA,
            model=GEMINI_37_FLASH_BATCH_MODEL,
            public_benchmark=True,
        )


def test_public_benchmark_models_require_explicit_public_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    image = tmp_path / "crop.png"
    image.write_bytes(b"image")
    captured = {}

    def transport(request: Request, timeout: float):
        captured["payload"] = json.loads(request.data)
        return (
            200,
            {},
            _success_response(
                '{"text":"ok"}',
                model=DEEPSEEK_VISION_MODEL,
            ),
        )

    result = repair_image(
        image,
        "Read",
        TEXT_SCHEMA,
        model=DEEPSEEK_VISION_MODEL,
        public_benchmark=True,
        transport=transport,
    )

    assert result.content == {"text": "ok"}
    assert captured["payload"]["response_format"] == {"type": "json_object"}

    gemma = repair_image(
        image,
        "Read",
        TEXT_SCHEMA,
        model=GEMMA_MODEL,
        transport=lambda request, timeout: (
            200,
            {},
            _success_response('{"text":"eligible"}', model=GEMMA_MODEL),
        ),
    )
    assert gemma.content == {"text": "eligible"}


def test_key_is_read_at_call_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "crop.png"
    image.write_bytes(b"image")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(OpenRouterError, match="OPENROUTER_API_KEY is not set"):
        repair_image(image, "Read", TEXT_SCHEMA, transport=_unused_transport)

    monkeypatch.setenv("OPENROUTER_API_KEY", "later-key")
    result = repair_image(
        image,
        "Read",
        TEXT_SCHEMA,
        transport=lambda request, timeout: (
            200,
            {},
            _success_response('{"text":"ok"}'),
        ),
    )
    assert result.content == {"text": "ok"}


def test_unexpected_response_model_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    image = tmp_path / "crop.png"
    image.write_bytes(b"image")

    with pytest.raises(OpenRouterError) as raised:
        repair_image(
            image,
            "Read",
            TEXT_SCHEMA,
            model=MUSE_GLIMMER_MODEL,
            transport=lambda request, timeout: (
                200,
                {},
                _success_response('{"text":"wrong model"}', model=QWEN_FLASH_MODEL),
            ),
        )

    assert raised.value.code == "unexpected_model"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_tokens": 0}, "Max tokens must be between"),
        ({"max_tokens": MAX_TOKENS + 1}, "Max tokens must be between"),
        ({"provider_slug": ""}, "Provider slug must be"),
        ({"provider_slug": " provider "}, "Provider slug must be"),
    ],
)
def test_request_controls_are_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    image = tmp_path / "crop.png"
    image.write_bytes(b"image")

    with pytest.raises(OpenRouterError, match=message):
        repair_image(
            image,
            "Read",
            TEXT_SCHEMA,
            transport=_unused_transport,
            **options,
        )


def test_retry_after_is_capped_and_5xx_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    image = tmp_path / "crop.png"
    image.write_bytes(b"image")
    responses = iter(
        [
            (429, {"Retry-After": "999"}, _error_response("busy")),
            (503, {}, _error_response("unavailable")),
            (200, {}, _success_response('{"text":"ok"}')),
        ]
    )
    sleeps: list[float] = []

    result = repair_image(
        image,
        "Read",
        TEXT_SCHEMA,
        transport=lambda request, timeout: next(responses),
        sleeper=sleeps.append,
    )

    assert sleeps == [MAX_RETRY_DELAY_SECONDS, 2]
    assert result.attempts == 3


def test_exhausted_retry_fails_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    image = tmp_path / "crop.png"
    image.write_bytes(b"image")

    with pytest.raises(
        OpenRouterError, match="exhausted 2 attempts with HTTP 429"
    ) as raised:
        repair_image(
            image,
            "Read",
            TEXT_SCHEMA,
            max_attempts=2,
            transport=lambda request, timeout: (
                429,
                {"Retry-After": "0"},
                _error_response("busy"),
            ),
            sleeper=lambda delay: None,
        )
    assert raised.value.code == "http_429_exhausted"
    assert raised.value.status_code == 429
    assert raised.value.attempts == 2


def test_http_timeout_and_truncation_have_distinct_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    image = tmp_path / "crop.png"
    image.write_bytes(b"image")

    with pytest.raises(OpenRouterError) as forbidden:
        repair_image(
            image,
            "Read",
            TEXT_SCHEMA,
            transport=lambda request, timeout: (403, {}, _error_response("denied")),
        )
    assert forbidden.value.code == "http_403"
    assert forbidden.value.status_code == 403

    def timeout(request: Request, timeout_seconds: float):
        raise TimeoutError("timed out")

    with pytest.raises(OpenRouterError) as timed_out:
        repair_image(
            image,
            "Read",
            TEXT_SCHEMA,
            max_attempts=1,
            transport=timeout,
        )
    assert timed_out.value.code == "timeout_exhausted"

    with pytest.raises(OpenRouterError) as truncated:
        repair_image(
            image,
            "Read",
            TEXT_SCHEMA,
            transport=lambda request, timeout: (
                200,
                {},
                _success_response('{"text":"partial"}', finish_reason="length"),
            ),
        )
    assert truncated.value.code == "generation_truncated"


def test_transport_errors_use_bounded_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    image = tmp_path / "crop.png"
    image.write_bytes(b"image")
    calls = 0
    sleeps: list[float] = []

    def transport(request: Request, timeout: float):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise OSError("temporary network failure")
        return 200, {}, _success_response('{"text":"ok"}')

    result = repair_image(
        image,
        "Read",
        TEXT_SCHEMA,
        max_attempts=3,
        transport=transport,
        sleeper=sleeps.append,
    )

    assert calls == 3
    assert sleeps == [1, 2]
    assert result.attempts == 3


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not JSON", "Invalid OpenRouter JSON response"),
        ('{"wrong":"field"}', "does not match schema"),
    ],
)
def test_invalid_structured_content_fails_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: str,
    message: str,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    image = tmp_path / "crop.png"
    image.write_bytes(b"image")

    with pytest.raises(OpenRouterError, match=message):
        repair_image(
            image,
            "Read",
            TEXT_SCHEMA,
            transport=lambda request, timeout: (
                200,
                {},
                _success_response(content),
            ),
        )


def _success_response(
    content: str,
    model: str = MUSE_GLIMMER_MODEL,
    *,
    finish_reason: str = "stop",
) -> bytes:
    return json.dumps(
        {
            "id": "generation-1",
            "model": model,
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": finish_reason,
                    "native_finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "cost": 0.00001,
            },
            "openrouter_metadata": {
                "endpoints": {
                    "available": [{"provider": "Test Provider", "selected": True}]
                }
            },
        }
    ).encode()


def _error_response(message: str) -> bytes:
    return json.dumps({"error": {"message": message}}).encode()


def _unused_transport(request: Request, timeout: float):
    raise AssertionError("transport must not be called")
