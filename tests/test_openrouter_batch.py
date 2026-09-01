from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.request import Request

import pytest

from ocr_pipeline.openrouter import (
    GEMINI_37_FLASH_BATCH_MODEL,
    GEMINI_37_FLASH_MODEL,
    OpenRouterError,
)
from ocr_pipeline.openrouter_batch import repair_images_batch

TEXT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def test_batch_submits_base_model_and_maps_unordered_partial_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    images = []
    for custom_id in ("case-a", "case-b", "case-c"):
        image = tmp_path / f"{custom_id}.png"
        image.write_bytes(custom_id.encode())
        images.append((custom_id, image))
    calls: list[Request] = []

    def transport(request: Request, timeout: float):
        calls.append(request)
        assert timeout == 7
        if request.method == "POST":
            return 200, {}, _json({"id": "batch-1", "status": "validating"})
        return (
            200,
            {},
            _json(
                {
                    "id": "batch-1",
                    "status": "completed",
                    "results": [
                        {
                            "custom_id": "case-b",
                            "response": None,
                            "error": {
                                "code": "provider_rejected",
                                "message": "provider rejected request",
                                "status_code": 422,
                            },
                        },
                        {
                            "custom_id": "case-a",
                            "response": {
                                "status_code": 200,
                                "request_id": "request-1",
                                "body": _success_response("A text"),
                            },
                            "error": None,
                        },
                    ],
                }
            ),
        )

    result = repair_images_batch(
        images,
        "Transcribe exactly",
        TEXT_SCHEMA,
        provider_slug="google-vertex",
        request_timeout_seconds=7,
        poll_interval_seconds=0,
        transport=transport,
        sleeper=lambda seconds: None,
    )

    assert [item.custom_id for item in result.items] == ["case-a", "case-b", "case-c"]
    assert result.items[0].content == {"text": "A text"}
    assert result.items[0].provider == "Google Vertex"
    assert result.items[0].usage == {"prompt_tokens": 10, "cost": 0.002}
    assert result.items[0].cost == 0.002
    assert result.items[0].latency_ms == 123.5
    assert result.items[1].error is not None
    assert result.items[1].error.code == "provider_rejected"
    assert result.items[2].error is not None
    assert result.items[2].error.code == "missing_batch_result"

    post = calls[0]
    assert post.full_url == "https://openrouter.ai/api/beta/batches"
    assert post.method == "POST"
    assert post.get_header("Authorization") == "Bearer test-secret"
    payload = json.loads(post.data)
    assert payload["endpoint"] == "/v1/chat/completions"
    assert payload["model"] == GEMINI_37_FLASH_MODEL
    assert [request["custom_id"] for request in payload["requests"]] == [
        "case-a",
        "case-b",
        "case-c",
    ]
    for request, (_, image_path) in zip(payload["requests"], images, strict=True):
        body = request["body"]
        assert body["model"] == GEMINI_37_FLASH_MODEL
        assert "temperature" not in body
        assert body["provider"] == {
            "order": ["google-vertex"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
        }
        assert body["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "ocr_transcription",
                "strict": True,
                "schema": TEXT_SCHEMA,
            },
        }
        assert body["messages"][0]["content"][1]["image_url"]["url"] == (
            "data:image/png;base64,"
            + base64.b64encode(image_path.read_bytes()).decode()
        )
    assert calls[1].full_url == "https://openrouter.ai/api/beta/batches/batch-1"
    assert "test-secret" not in repr(result)
    assert all("data:image" not in repr(item) for item in result.items)


def test_batch_rejects_unexpected_model_but_records_provider_display_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    image = tmp_path / "page.png"
    image.write_bytes(b"image")

    response = _success_response("text", model="other/model")
    result = repair_images_batch(
        [("case", image)],
        "Read",
        TEXT_SCHEMA,
        provider_slug="google-vertex",
        poll_interval_seconds=0,
        transport=lambda request, timeout: (
            200,
            {},
            _json(
                {
                    "id": "batch-1",
                    "status": "completed",
                    "results": [{"custom_id": "case", "response": response}],
                }
            ),
        ),
        sleeper=lambda seconds: None,
    )
    assert result.items[0].error is not None
    assert result.items[0].error.code == "unexpected_model"

    response = _success_response("text", provider="Google Vertex")
    accepted = repair_images_batch(
        [("case", image)],
        "Read",
        TEXT_SCHEMA,
        provider_slug="google-vertex",
        transport=lambda request, timeout: (
            200,
            {},
            _json(
                {
                    "id": "batch-2",
                    "status": "completed",
                    "results": [{"custom_id": "case", "response": response}],
                }
            ),
        ),
    )
    assert accepted.items[0].error is None
    assert accepted.items[0].provider == "Google Vertex"


@pytest.mark.parametrize(
    ("include_model", "model"),
    [(False, None), (True, None), (True, 37)],
    ids=("missing", "null", "non-string"),
)
def test_batch_rejects_missing_null_or_non_string_response_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_model: bool,
    model: object,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    image = tmp_path / "page.png"
    image.write_bytes(b"image")
    response = _success_response("text")
    if include_model:
        response["model"] = model
    else:
        response.pop("model")

    result = repair_images_batch(
        [("case", image)],
        "Read",
        TEXT_SCHEMA,
        provider_slug="google-vertex",
        transport=lambda request, timeout: (
            200,
            {},
            _json(
                {
                    "id": "batch-1",
                    "status": "completed",
                    "results": [{"custom_id": "case", "response": response}],
                }
            ),
        ),
    )

    assert result.items[0].error is not None
    assert result.items[0].error.code == "unexpected_model"
    assert result.items[0].model is None


def test_resume_polls_existing_batch_without_submitting_or_reading_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    calls: list[Request] = []
    sleeps = []

    def transport(request: Request, timeout: float):
        calls.append(request)
        return (
            200,
            {},
            _json(
                {
                    "id": "batch-existing",
                    "status": "completed",
                    "results": [
                        {
                            "custom_id": "case",
                            "response": _success_response(
                                "resumed", provider="Google Vertex"
                            ),
                        }
                    ],
                }
            ),
        )

    result = repair_images_batch(
        [("case", tmp_path / "does-not-need-to-exist.png")],
        "Read",
        TEXT_SCHEMA,
        provider_slug="google-vertex",
        batch_id="batch-existing",
        transport=transport,
        sleeper=sleeps.append,
    )

    assert len(calls) == 1
    assert calls[0].method == "GET"
    assert calls[0].data is None
    assert calls[0].full_url.endswith("/api/beta/batches/batch-existing")
    assert sleeps == []
    assert result.batch_id == "batch-existing"
    assert result.polls == 1
    assert result.items[0].content == {"text": "resumed"}
    assert result.items[0].provider == "Google Vertex"


@pytest.mark.parametrize(
    "result_item",
    [
        "not-an-object",
        {"custom_id": "case"},
        {"custom_id": "case", "response": {}, "error": {}},
        {"custom_id": "case", "response": None, "error": None},
        {"custom_id": "case", "response": {"status_code": 200}},
    ],
)
def test_malformed_inline_results_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result_item: object,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    image = tmp_path / "page.png"
    image.write_bytes(b"image")

    with pytest.raises(OpenRouterError) as raised:
        repair_images_batch(
            [("case", image)],
            "Read",
            TEXT_SCHEMA,
            provider_slug="google-vertex",
            transport=lambda request, timeout: (
                200,
                {},
                _json(
                    {
                        "id": "batch-1",
                        "status": "completed",
                        "results": [result_item],
                    }
                ),
            ),
        )
    assert raised.value.code == "invalid_batch_result"


def test_poll_timeout_and_terminal_failure_have_explicit_item_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    image = tmp_path / "page.png"
    image.write_bytes(b"image")
    clock = iter((0.0, 2.0, 2.0))

    with pytest.raises(OpenRouterError) as raised:
        repair_images_batch(
            [("case", image)],
            "Read",
            TEXT_SCHEMA,
            provider_slug="google-vertex",
            poll_timeout_seconds=1,
            transport=lambda request, timeout: (
                200,
                {},
                _json({"id": "batch-1", "status": "validating"}),
            ),
            timer=lambda: next(clock),
        )
    assert raised.value.code == "batch_poll_timeout"

    failed = repair_images_batch(
        [("case", image)],
        "Read",
        TEXT_SCHEMA,
        provider_slug="google-vertex",
        transport=lambda request, timeout: (
            200,
            {},
            _json(
                {
                    "id": "batch-2",
                    "status": "failed",
                    "error": {"message": "upstream batch failed"},
                }
            ),
        ),
    )
    assert failed.items[0].error is not None
    assert failed.items[0].error.code == "batch_failed"


def test_only_gemini_batch_alias_is_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    image = tmp_path / "page.png"
    image.write_bytes(b"image")

    with pytest.raises(OpenRouterError) as raised:
        repair_images_batch(
            [("case", image)],
            "Read",
            TEXT_SCHEMA,
            model=GEMINI_37_FLASH_MODEL,
            provider_slug="google-vertex",
        )
    assert raised.value.code == "unsupported_batch_model"
    assert GEMINI_37_FLASH_BATCH_MODEL.endswith(":batch")


def _success_response(
    text: str,
    *,
    model: str = GEMINI_37_FLASH_MODEL,
    provider: str = "Google Vertex",
) -> dict[str, object]:
    return {
        "model": model,
        "openrouter_metadata": {
            "endpoints": {"available": [{"provider": provider, "selected": True}]}
        },
        "choices": [
            {
                "message": {"content": json.dumps({"text": text})},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "cost": 0.002},
        "latency_ms": 123.5,
    }


def _json(value: object) -> bytes:
    return json.dumps(value).encode()
