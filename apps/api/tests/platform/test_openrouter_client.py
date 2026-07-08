import json

import httpx
import pytest

from api.common.usage import InMemoryUsageTracker
from api.config.settings import CoreSettings
from api.platform.openrouter import ImageInput, OpenRouterClient, OpenRouterRequest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _success_response(content: str = '{"name":"Ada"}') -> dict[str, object]:
    return {
        "id": "completion-id",
        "model": "openai/gpt-5.2",
        "provider": "openai",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": "0.00012",
        },
    }


def _openrouter_request() -> OpenRouterRequest:
    return OpenRouterRequest(
        model="openai/gpt-5.2",
        system_prompt="Extract facts.",
        user_content="Document markdown",
        schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        capability="document_parsing",
        consumer_id="internal",
        request_id="request-1",
    )


@pytest.mark.anyio
async def test_openrouter_client_builds_native_structured_request(
    core_settings: CoreSettings,
) -> None:
    captured_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content))
        return httpx.Response(200, json=_success_response())

    tracker = InMemoryUsageTracker()
    client = OpenRouterClient(
        core_settings,
        tracker,
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    result = await client.complete(_openrouter_request())

    payload = captured_payloads[0]
    assert payload["provider"] == {"require_parameters": True}
    assert payload["response_format"]["type"] == "json_schema"  # type: ignore[index]
    assert result.content == '{"name":"Ada"}'
    assert tracker.records[0].provider == "openrouter"
    assert tracker.records[0].total_tokens == 15


@pytest.mark.anyio
async def test_openrouter_client_falls_back_when_native_schema_is_rejected(
    core_settings: CoreSettings,
) -> None:
    captured_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured_payloads.append(payload)
        if len(captured_payloads) == 1:
            return httpx.Response(
                400,
                json={"error": {"message": "response_format json_schema unsupported"}},
            )
        return httpx.Response(200, json=_success_response())

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    result = await client.complete(_openrouter_request())

    assert result.content == '{"name":"Ada"}'
    assert "response_format" in captured_payloads[0]
    assert "response_format" not in captured_payloads[1]
    fallback_messages = captured_payloads[1]["messages"]
    assert "JSON Schema" in fallback_messages[-1]["content"]  # type: ignore[index]


@pytest.mark.anyio
async def test_openrouter_client_retries_retryable_provider_errors(
    core_settings: CoreSettings,
) -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, json={"error": {"message": "provider unavailable"}})
        return httpx.Response(200, json=_success_response())

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    result = await client.complete(_openrouter_request())

    assert result.content == '{"name":"Ada"}'
    assert attempts == 2


@pytest.mark.anyio
async def test_openrouter_client_maps_choice_errors_to_upstream_error(
    core_settings: CoreSettings,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "error",
                        "error": {"message": "provider failed"},
                    }
                ]
            },
        )

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    with pytest.raises(Exception, match="OpenRouter provider returned an error"):
        await client.complete(_openrouter_request())


@pytest.mark.anyio
async def test_openrouter_client_maps_transport_errors_to_upstream_error(
    core_settings: CoreSettings,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    with pytest.raises(Exception, match="OpenRouter request failed"):
        await client.complete(_openrouter_request())


@pytest.mark.anyio
async def test_openrouter_client_maps_malformed_success_json_to_upstream_error(
    core_settings: CoreSettings,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    with pytest.raises(Exception, match="OpenRouter response was not valid JSON"):
        await client.complete(_openrouter_request())


@pytest.mark.anyio
async def test_openrouter_client_maps_malformed_choice_to_upstream_error(
    core_settings: CoreSettings,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": ["bad"]})

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    with pytest.raises(Exception, match="OpenRouter response choice was malformed"):
        await client.complete(_openrouter_request())


@pytest.mark.anyio
async def test_openrouter_client_maps_malformed_usage_cost_to_upstream_error(
    core_settings: CoreSettings,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        response = _success_response()
        response["usage"]["cost"] = "not-a-decimal"  # type: ignore[index]
        return httpx.Response(200, json=response)

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    with pytest.raises(Exception, match="OpenRouter response usage cost was malformed"):
        await client.complete(_openrouter_request())


@pytest.mark.anyio
async def test_openrouter_client_maps_non_finite_usage_cost_to_upstream_error(
    core_settings: CoreSettings,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        response = _success_response()
        response["usage"]["cost"] = "NaN"  # type: ignore[index]
        return httpx.Response(200, json=response)

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    with pytest.raises(Exception, match="OpenRouter response usage cost was malformed"):
        await client.complete(_openrouter_request())


@pytest.mark.anyio
async def test_openrouter_client_supports_image_content_parts(
    core_settings: CoreSettings,
) -> None:
    captured_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content))
        return httpx.Response(200, json=_success_response())

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )
    request = OpenRouterRequest(
        **{
            **_openrouter_request().__dict__,
            "images": [ImageInput(mime_type="image/png", base64_data="abc123")],
        }
    )

    await client.complete(request)

    user_content = captured_payloads[0]["messages"][1]["content"]  # type: ignore[index]
    assert user_content[0] == {"type": "text", "text": "Document markdown"}
    assert user_content[1]["image_url"]["url"] == "data:image/png;base64,abc123"
