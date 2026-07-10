import asyncio
import json
import logging
import time
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from api.common.errors import UpstreamError
from api.common.logging import log_event
from api.common.usage import UsageRecord, UsageTracker
from api.config.settings import CoreSettings
from api.platform.schema.normalize import normalize_for_model

from .models import ImageInput, OpenRouterRequest, OpenRouterResult
from .openai_compat import OPENROUTER_BASE_URL

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenRouterClient:
    def __init__(
        self,
        settings: CoreSettings,
        usage_tracker: UsageTracker,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        max_retries: int = 2,
        backoff_seconds: float = 0.25,
    ) -> None:
        self._settings = settings
        self._usage_tracker = usage_tracker
        self._transport = transport
        self._base_url = base_url
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds

    async def complete(self, request: OpenRouterRequest) -> OpenRouterResult:
        start_attempt = max(1, request.attempt)
        native_payload = self._build_payload(request, native_structured=True)
        try:
            return await self._post_payload(
                native_payload,
                request,
                structured_mode="native",
                start_attempt=start_attempt,
            )
        except UpstreamError as exc:
            if not _is_native_schema_rejection(exc):
                raise
            fallback_attempt = (exc.attempt or start_attempt) + 1
            log_event(
                "model.retry",
                level=logging.WARNING,
                reason="native_schema_rejection",
                attempt=fallback_attempt,
                model=request.model,
                requestId=request.request_id,
            )

        fallback_payload = self._build_payload(request, native_structured=False)
        return await self._post_payload(
            fallback_payload,
            request,
            structured_mode="fallback",
            start_attempt=fallback_attempt,
        )

    def _build_payload(
        self,
        request: OpenRouterRequest,
        *,
        native_structured: bool,
    ) -> dict[str, Any]:
        user_content = _build_user_content(request.user_content, request.images)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": user_content},
        ]
        if request.validation_retry_prompt is not None:
            messages.append({"role": "user", "content": request.validation_retry_prompt})

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
        }

        if request.reasoning_effort is not None:
            payload["reasoning"] = {"effort": request.reasoning_effort}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens

        if native_structured:
            payload["provider"] = {"require_parameters": True}
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_result",
                    "strict": True,
                    "schema": normalize_for_model(request.schema),
                },
            }
        else:
            payload["messages"].append(
                {
                    "role": "user",
                    "content": (
                        "Return only JSON that matches this JSON Schema. "
                        "Do not include markdown fences or explanatory text.\n"
                        f"{json.dumps(request.schema, sort_keys=True)}"
                    ),
                }
            )

        return payload

    async def _post_payload(
        self,
        payload: dict[str, Any],
        request: OpenRouterRequest,
        *,
        structured_mode: str,
        start_attempt: int,
    ) -> OpenRouterResult:
        last_error: UpstreamError | None = None
        for retry_offset in range(self._max_retries + 1):
            attempt_number = start_attempt + retry_offset
            log_event(
                "model.request",
                model=request.model,
                structuredMode=structured_mode,
                attempt=attempt_number,
                requestId=request.request_id,
            )
            started_at = time.perf_counter()
            try:
                async with httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=self._settings.timeout_seconds,
                    transport=self._transport,
                ) as client:
                    response = await client.post(
                        "/chat/completions",
                        headers={
                            "Authorization": (
                                f"Bearer {self._settings.openrouter_api_key.get_secret_value()}"
                            ),
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
            except httpx.RequestError as exc:
                error = UpstreamError("OpenRouter request failed", attempt=attempt_number)
                self._log_failed_response(
                    request,
                    attempt=attempt_number,
                    started_at=started_at,
                    structured_mode=structured_mode,
                    error=error,
                )
                if retry_offset >= self._max_retries:
                    raise error from exc
                next_attempt = attempt_number + 1
                log_event(
                    "model.retry",
                    level=logging.WARNING,
                    reason="network_error",
                    attempt=next_attempt,
                    model=request.model,
                    requestId=request.request_id,
                )
                last_error = error
                await asyncio.sleep(self._backoff_seconds * (2**retry_offset))
                continue

            if response.status_code < 400:
                try:
                    result = self._parse_success_response(
                        _json_response(response),
                        request,
                        attempt=attempt_number,
                    )
                except UpstreamError as exc:
                    exc.attempt = attempt_number
                    self._log_failed_response(
                        request,
                        attempt=attempt_number,
                        started_at=started_at,
                        structured_mode=structured_mode,
                        error=exc,
                        status_code=response.status_code,
                    )
                    raise
                self._record_usage(result, request)
                duration_ms = _duration_ms(started_at)
                log_event(
                    "model.response",
                    outcome="succeeded",
                    attempt=attempt_number,
                    structuredMode=structured_mode,
                    model=result.model,
                    provider=result.provider,
                    promptTokens=result.prompt_tokens,
                    completionTokens=result.completion_tokens,
                    totalTokens=result.total_tokens,
                    costUsd=result.cost_usd,
                    latencyMs=duration_ms,
                    durationMs=duration_ms,
                    requestId=request.request_id,
                )
                return result

            last_error = _map_error_response(response)
            last_error.attempt = attempt_number
            self._log_failed_response(
                request,
                attempt=attempt_number,
                started_at=started_at,
                structured_mode=structured_mode,
                error=last_error,
                status_code=response.status_code,
            )
            if (
                response.status_code not in RETRYABLE_STATUS_CODES
                or retry_offset >= self._max_retries
            ):
                raise last_error
            next_attempt = attempt_number + 1
            log_event(
                "model.retry",
                level=logging.WARNING,
                reason="retryable_status",
                attempt=next_attempt,
                statusCode=response.status_code,
                model=request.model,
                requestId=request.request_id,
            )
            await asyncio.sleep(self._backoff_seconds * (2**retry_offset))

        raise last_error or UpstreamError()

    def _parse_success_response(
        self,
        data: dict[str, Any],
        request: OpenRouterRequest,
        *,
        attempt: int,
    ) -> OpenRouterResult:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise UpstreamError("OpenRouter response did not include choices")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise UpstreamError("OpenRouter response choice was malformed")
        if choice.get("error"):
            raise UpstreamError("OpenRouter provider returned an error")

        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise UpstreamError("OpenRouter response did not include text content")

        usage_data = data.get("usage")
        usage = usage_data if isinstance(usage_data, dict) else {}
        result = OpenRouterResult(
            content=content,
            model=str(data.get("model") or request.model),
            provider=str(data["provider"]) if data.get("provider") is not None else None,
            prompt_tokens=_int_usage(usage.get("prompt_tokens")),
            completion_tokens=_int_usage(usage.get("completion_tokens")),
            total_tokens=_int_usage(usage.get("total_tokens")),
            cost_usd=_decimal_usage_cost(usage.get("cost")),
            attempt=attempt,
        )
        return result

    def _log_failed_response(
        self,
        request: OpenRouterRequest,
        *,
        attempt: int,
        started_at: float,
        structured_mode: str,
        error: UpstreamError,
        status_code: int | None = None,
    ) -> None:
        duration_ms = _duration_ms(started_at)
        log_event(
            "model.response",
            level=logging.WARNING,
            outcome="failed",
            errorCode=error.code,
            attempt=attempt,
            structuredMode=structured_mode,
            model=request.model,
            statusCode=status_code,
            latencyMs=duration_ms,
            durationMs=duration_ms,
            requestId=request.request_id,
        )

    def _record_usage(self, result: OpenRouterResult, request: OpenRouterRequest) -> None:
        usage = UsageRecord(
            provider="openrouter",
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_usd=result.cost_usd,
            capability=request.capability,
            consumer_id=request.consumer_id,
            request_id=request.request_id,
        )
        self._usage_tracker.record(usage)
        log_event(
            "usage.recorded",
            provider=usage.provider,
            model=usage.model,
            promptTokens=usage.prompt_tokens,
            completionTokens=usage.completion_tokens,
            totalTokens=usage.total_tokens,
            costUsd=usage.cost_usd,
            capability=usage.capability,
            consumerId=usage.consumer_id,
            requestId=usage.request_id,
        )


def _build_user_content(text: str, images: list[ImageInput]) -> str | list[dict[str, Any]]:
    if not images:
        return text

    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image.data_url, "detail": image.detail},
            }
        )
    return content


def _map_error_response(response: httpx.Response) -> UpstreamError:
    try:
        data = response.json()
    except ValueError:
        data = {}

    message = "OpenRouter request failed"
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        message = error["message"]
    return UpstreamError(message)


def _json_response(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise UpstreamError("OpenRouter response was not valid JSON") from exc
    if not isinstance(data, dict):
        raise UpstreamError("OpenRouter response JSON was malformed")
    return data


def _is_native_schema_rejection(error: UpstreamError) -> bool:
    message = error.message.lower()
    markers = ("schema", "response_format", "structured", "require_parameters", "unsupported")
    return any(marker in message for marker in markers)


def _int_usage(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _decimal_usage_cost(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        cost = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise UpstreamError("OpenRouter response usage cost was malformed") from exc
    if not cost.is_finite():
        raise UpstreamError("OpenRouter response usage cost was malformed")
    return cost


def _duration_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)
