import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from api.common.logging import configure_logging
from api.common.usage import InMemoryUsageTracker
from api.config.settings import CoreSettings
from api.platform.openrouter import ImageInput, OpenRouterClient, OpenRouterRequest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _sse_body(
    pieces: list[str],
    *,
    model: str = "openai/gpt-5.2",
    provider: str = "openai",
    cost: str = "0.00012",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
    extra_lines: list[str] | None = None,
) -> bytes:
    lines: list[str] = []
    for index, piece in enumerate(pieces):
        chunk: dict[str, object] = {
            "choices": [{"delta": {"content": piece}}],
        }
        if index == 0:
            chunk["model"] = model
            chunk["provider"] = provider
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    if extra_lines:
        lines.extend(extra_lines)
    lines.append(
        "data: "
        + json.dumps(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost,
                },
            }
        )
        + "\n\n"
    )
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _success_response(content: str = '{"name":"Ada"}') -> httpx.Response:
    return httpx.Response(
        200,
        content=_sse_body([content]),
        headers={"content-type": "text/event-stream"},
    )


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
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="INFO")
    captured_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content))
        return _success_response()

    tracker = InMemoryUsageTracker()
    client = OpenRouterClient(
        core_settings,
        tracker,
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    result = await client.complete(_openrouter_request())

    payload = captured_payloads[0]
    assert payload["provider"] == {"require_parameters": True, "sort": "throughput"}
    assert payload["response_format"]["type"] == "json_schema"  # type: ignore[index]
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert result.content == '{"name":"Ada"}'
    assert tracker.records[0].provider == "openrouter"
    assert tracker.records[0].total_tokens == 15
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    by_name = {event["event"]: event for event in events}
    assert by_name["model.request"] == {
        "attempt": 1,
        "event": "model.request",
        "model": "openai/gpt-5.2",
        "requestId": "request-1",
        "structuredMode": "native",
    }
    assert by_name["model.response"]["provider"] == "openai"
    assert by_name["model.response"]["promptTokens"] == 10
    assert by_name["model.response"]["completionTokens"] == 5
    assert by_name["model.response"]["totalTokens"] == 15
    assert by_name["model.response"]["costUsd"] == "0.00012"
    assert by_name["model.response"]["latencyMs"] >= 0
    assert by_name["usage.recorded"]["capability"] == "document_parsing"
    assert by_name["usage.recorded"]["consumerId"] == "internal"
    assert by_name["usage.recorded"]["costUsd"] == "0.00012"


@pytest.mark.anyio
async def test_openrouter_client_falls_back_when_native_schema_is_rejected(
    core_settings: CoreSettings,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="INFO")
    captured_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured_payloads.append(payload)
        if len(captured_payloads) == 1:
            return httpx.Response(
                400,
                json={"error": {"message": "response_format json_schema unsupported"}},
            )
        return _success_response()

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
    assert captured_payloads[1]["provider"] == {"sort": "throughput"}
    fallback_messages = captured_payloads[1]["messages"]
    assert "JSON Schema" in fallback_messages[-1]["content"]  # type: ignore[index]
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    requests = [event for event in events if event["event"] == "model.request"]
    retry = next(event for event in events if event["event"] == "model.retry")
    assert [(event["structuredMode"], event["attempt"]) for event in requests] == [
        ("native", 1),
        ("fallback", 2),
    ]
    assert retry["reason"] == "native_schema_rejection"
    assert retry["attempt"] == 2


@pytest.mark.anyio
async def test_openrouter_client_sends_models_array_without_model_key(
    core_settings: CoreSettings,
) -> None:
    captured_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content))
        return _success_response()

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    request = replace(_openrouter_request(), models=["a", "b"])
    await client.complete(request)

    payload = captured_payloads[0]
    assert payload["models"] == ["a", "b"]
    assert "model" not in payload


@pytest.mark.anyio
async def test_openrouter_client_sends_single_model_when_no_models_array(
    core_settings: CoreSettings,
) -> None:
    captured_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content))
        return _success_response()

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    request = replace(_openrouter_request(), model="x")
    await client.complete(request)

    payload = captured_payloads[0]
    assert payload["model"] == "x"
    assert "models" not in payload


@pytest.mark.anyio
async def test_openrouter_client_attributes_usage_to_response_model(
    core_settings: CoreSettings,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse_body(['{"name":"Ada"}'], model="openai/gpt-5-mini"),
            headers={"content-type": "text/event-stream"},
        )

    tracker = InMemoryUsageTracker()
    client = OpenRouterClient(
        core_settings,
        tracker,
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    request = replace(
        _openrouter_request(),
        model="openai/gpt-5.2",
        models=["openai/gpt-5.2", "openai/gpt-5-mini"],
    )
    result = await client.complete(request)

    assert result.model == "openai/gpt-5-mini"
    assert tracker.records[0].model == "openai/gpt-5-mini"


@pytest.mark.anyio
async def test_openrouter_attempts_remain_monotonic_across_http_retry_and_fallback(
    core_settings: CoreSettings,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="INFO")
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, json={"error": {"message": "provider unavailable"}})
        if attempts == 2:
            return httpx.Response(
                400,
                json={"error": {"message": "response_format json_schema unsupported"}},
            )
        return _success_response()

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    await client.complete(_openrouter_request())

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    requests = [event for event in events if event["event"] == "model.request"]
    retries = [event for event in events if event["event"] == "model.retry"]
    assert [event["attempt"] for event in requests] == [1, 2, 3]
    assert [event["attempt"] for event in retries] == [2, 3]


@pytest.mark.anyio
async def test_openrouter_client_retries_retryable_provider_errors(
    core_settings: CoreSettings,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="INFO")
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, json={"error": {"message": "provider unavailable"}})
        return _success_response()

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    result = await client.complete(_openrouter_request())

    assert result.content == '{"name":"Ada"}'
    assert attempts == 2
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    requests = [event for event in events if event["event"] == "model.request"]
    retry = next(event for event in events if event["event"] == "model.retry")
    responses = [event for event in events if event["event"] == "model.response"]
    assert [event["attempt"] for event in requests] == [1, 2]
    assert retry["reason"] == "retryable_status"
    assert retry["attempt"] == 2
    assert responses[0]["outcome"] == "failed"
    assert responses[0]["errorCode"] == "upstream_error"
    assert responses[0]["statusCode"] == 503
    assert responses[0]["durationMs"] >= 0


@pytest.mark.anyio
async def test_openrouter_client_logs_terminal_model_failure(
    core_settings: CoreSettings,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="INFO")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "invalid request"}})

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    with pytest.raises(Exception, match="invalid request"):
        await client.complete(_openrouter_request())

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    response = next(event for event in events if event["event"] == "model.response")
    assert response["outcome"] == "failed"
    assert response["errorCode"] == "upstream_error"
    assert response["statusCode"] == 400
    assert response["durationMs"] >= 0


@pytest.mark.anyio
async def test_openrouter_client_maps_choice_errors_to_upstream_error(
    core_settings: CoreSettings,
) -> None:
    body = (
        b'data: {"choices":[{"error":{"message":"provider failed"},'
        b'"finish_reason":"error"}]}\n\n'
        b"data: [DONE]\n\n"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

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
async def test_openrouter_client_retries_network_errors_then_succeeds(
    core_settings: CoreSettings,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="INFO")
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("boom", request=request)
        return _success_response()

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    result = await client.complete(_openrouter_request())

    assert result.content == '{"name":"Ada"}'
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    requests = [event for event in events if event["event"] == "model.request"]
    retry = next(event for event in events if event["event"] == "model.retry")
    assert [event["attempt"] for event in requests] == [1, 2]
    assert retry["reason"] == "network_error"
    assert retry["attempt"] == 2


@pytest.mark.anyio
async def test_openrouter_client_raises_after_persistent_network_errors(
    core_settings: CoreSettings,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="INFO")

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    with pytest.raises(Exception, match="OpenRouter request failed"):
        await client.complete(_openrouter_request())

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    requests = [event for event in events if event["event"] == "model.request"]
    assert [event["attempt"] for event in requests] == [1, 2, 3]


@pytest.mark.anyio
async def test_openrouter_client_composes_network_retry_with_native_schema_fallback(
    core_settings: CoreSettings,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="INFO")
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("boom", request=request)
        if attempts == 2:
            return httpx.Response(
                400,
                json={"error": {"message": "response_format json_schema unsupported"}},
            )
        return _success_response()

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    result = await client.complete(_openrouter_request())

    assert result.content == '{"name":"Ada"}'
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    requests = [event for event in events if event["event"] == "model.request"]
    assert [event["attempt"] for event in requests] == [1, 2, 3]


@pytest.mark.anyio
async def test_openrouter_client_ignores_keepalive_comments_in_stream(
    core_settings: CoreSettings,
) -> None:
    body = _sse_body(
        ["Hel", "lo"],
        extra_lines=[": OPENROUTER PROCESSING\n\n"],
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    result = await client.complete(_openrouter_request())

    assert result.content == "Hello"


@pytest.mark.anyio
async def test_openrouter_client_cuts_off_hung_attempt_at_per_attempt_cap(
    core_settings: CoreSettings,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="INFO")
    settings = core_settings.model_copy(
        update={
            "request_attempt_timeout_seconds": 0.05,
            "request_total_timeout_seconds": 5.0,
        }
    )

    async def slow_stream():  # type: ignore[no-untyped-def]
        yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        await asyncio.sleep(1.0)
        yield b"data: [DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=slow_stream(),
            headers={"content-type": "text/event-stream"},
        )

    client = OpenRouterClient(
        settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
        max_retries=0,
    )

    with pytest.raises(Exception, match="time budget"):
        await client.complete(_openrouter_request())

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    response = next(event for event in events if event["event"] == "model.response")
    assert response["outcome"] == "failed"


@pytest.mark.anyio
async def test_openrouter_client_stops_retrying_once_total_budget_exhausted(
    core_settings: CoreSettings,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="INFO")
    settings = core_settings.model_copy(update={"request_total_timeout_seconds": 0.2})
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, json={"error": {"message": "provider unavailable"}})

    client = OpenRouterClient(
        settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=5.0,
        max_retries=5,
    )

    with pytest.raises(Exception, match="provider unavailable"):
        await client.complete(_openrouter_request())

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    requests = [event for event in events if event["event"] == "model.request"]
    assert attempts == len(requests)
    assert attempts < 5


@pytest.mark.anyio
async def test_openrouter_client_retries_streamed_http_error_status(
    core_settings: CoreSettings,
) -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, json={"error": {"message": "provider unavailable"}})
        return _success_response()

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
async def test_openrouter_client_records_usage_exactly_once_on_success(
    core_settings: CoreSettings,
) -> None:
    tracker = InMemoryUsageTracker()

    async def handler(request: httpx.Request) -> httpx.Response:
        return _success_response()

    client = OpenRouterClient(
        core_settings,
        tracker,
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    await client.complete(_openrouter_request())

    assert len(tracker.records) == 1


@pytest.mark.anyio
async def test_openrouter_client_records_no_usage_when_cancelled(
    core_settings: CoreSettings,
) -> None:
    tracker = InMemoryUsageTracker()
    stall_event = asyncio.Event()

    async def slow_stream():  # type: ignore[no-untyped-def]
        yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        await stall_event.wait()
        yield b"data: [DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=slow_stream(),
            headers={"content-type": "text/event-stream"},
        )

    client = OpenRouterClient(
        core_settings,
        tracker,
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    task = asyncio.ensure_future(client.complete(_openrouter_request()))
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert tracker.records == []


@pytest.mark.anyio
async def test_openrouter_client_maps_malformed_usage_cost_to_upstream_error(
    core_settings: CoreSettings,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse_body(['{"name":"Ada"}'], cost="not-a-decimal"),
            headers={"content-type": "text/event-stream"},
        )

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
        return httpx.Response(
            200,
            content=_sse_body(['{"name":"Ada"}'], cost="NaN"),
            headers={"content-type": "text/event-stream"},
        )

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    with pytest.raises(Exception, match="OpenRouter response usage cost was malformed"):
        await client.complete(_openrouter_request())


@pytest.mark.anyio
async def test_openrouter_client_includes_model_params_when_set(
    core_settings: CoreSettings,
) -> None:
    captured_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content))
        return _success_response()

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )
    request = OpenRouterRequest(
        **{
            **_openrouter_request().__dict__,
            "reasoning_effort": "high",
            "temperature": 0.4,
            "max_output_tokens": 512,
        }
    )

    await client.complete(request)

    payload = captured_payloads[0]
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["temperature"] == 0.4
    assert payload["max_tokens"] == 512


@pytest.mark.anyio
async def test_openrouter_client_omits_model_params_when_unset(
    core_settings: CoreSettings,
) -> None:
    captured_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content))
        return _success_response()

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    await client.complete(_openrouter_request())

    payload = captured_payloads[0]
    assert "reasoning" not in payload
    assert "temperature" not in payload
    assert "max_tokens" not in payload


@pytest.mark.anyio
async def test_openrouter_client_renders_cache_breakpoint_when_enabled(
    core_settings: CoreSettings,
) -> None:
    captured_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content))
        return _success_response()

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )
    request = replace(_openrouter_request(), cache_control=True)

    await client.complete(request)

    payload = captured_payloads[0]
    assert payload["messages"][0]["content"] == [  # type: ignore[index]
        {
            "type": "text",
            "text": "Extract facts.",
            "cache_control": {"type": "ephemeral"},
        }
    ]


@pytest.mark.anyio
async def test_openrouter_client_keeps_plain_system_content_when_cache_control_disabled(
    core_settings: CoreSettings,
) -> None:
    captured_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content))
        return _success_response()

    client = OpenRouterClient(
        core_settings,
        InMemoryUsageTracker(),
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
    )

    await client.complete(_openrouter_request())

    payload = captured_payloads[0]
    assert payload["messages"][0]["content"] == "Extract facts."  # type: ignore[index]


@pytest.mark.anyio
async def test_openrouter_client_supports_image_content_parts(
    core_settings: CoreSettings,
) -> None:
    captured_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content))
        return _success_response()

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
