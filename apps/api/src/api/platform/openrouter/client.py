import asyncio
import json
import logging
import random
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
from .streaming import consume_completion_stream

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
        deadline = time.monotonic() + self._settings.request_total_timeout_seconds
        start_attempt = max(1, request.attempt)
        native_payload = self._build_payload(request, native_structured=True)
        try:
            return await self._post_payload(
                native_payload,
                request,
                structured_mode="native",
                start_attempt=start_attempt,
                deadline=deadline,
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
            deadline=deadline,
        )

    def _build_payload(
        self,
        request: OpenRouterRequest,
        *,
        native_structured: bool,
    ) -> dict[str, Any]:
        user_content = _build_user_content(request.user_content, request.images)
        system_content: str | list[dict[str, Any]] = request.system_prompt
        if request.cache_control:
            system_content = [
                {
                    "type": "text",
                    "text": request.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        if request.validation_retry_prompt is not None:
            messages.append({"role": "user", "content": request.validation_retry_prompt})

        payload: dict[str, Any] = {
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.models:
            payload["models"] = request.models
        else:
            payload["model"] = request.model

        if request.reasoning_effort is not None:
            payload["reasoning"] = {"effort": request.reasoning_effort}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens

        provider_sort = self._settings.openrouter_provider_sort

        if native_structured:
            provider: dict[str, Any] = {"require_parameters": True}
            if provider_sort is not None:
                provider["sort"] = provider_sort
            payload["provider"] = provider
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_result",
                    "strict": True,
                    "schema": normalize_for_model(request.schema),
                },
            }
        else:
            if provider_sort is not None:
                payload["provider"] = {"sort": provider_sort}
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
        deadline: float,
    ) -> OpenRouterResult:
        last_error: UpstreamError | None = None
        retry_offset = 0
        while True:
            attempt_number = start_attempt + retry_offset
            remaining = remaining_budget(deadline)
            if remaining <= 0:
                raise last_error or UpstreamError(
                    "OpenRouter request exceeded its total time budget",
                    attempt=attempt_number,
                )
            attempt_timeout = effective_attempt_timeout(
                remaining, self._settings.request_attempt_timeout_seconds
            )

            log_event(
                "model.request",
                model=request.model,
                structuredMode=structured_mode,
                attempt=attempt_number,
                requestId=request.request_id,
            )
            started_at = time.perf_counter()
            try:
                result = await self._attempt(
                    payload,
                    request,
                    structured_mode=structured_mode,
                    attempt_number=attempt_number,
                    attempt_timeout=attempt_timeout,
                    started_at=started_at,
                )
            except _RetryableAttemptError as retryable:
                error = retryable.error
                last_error = error
                remaining = remaining_budget(deadline)
                backoff = None
                if retry_offset < self._max_retries:
                    backoff = backoff_with_jitter(self._backoff_seconds, retry_offset, remaining)
                if backoff is None:
                    raise error from retryable.__cause__
                next_attempt = attempt_number + 1
                log_event(
                    "model.retry",
                    level=logging.WARNING,
                    reason=retryable.reason,
                    attempt=next_attempt,
                    statusCode=retryable.status_code,
                    model=request.model,
                    requestId=request.request_id,
                )
                await asyncio.sleep(backoff)
                retry_offset += 1
                continue

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

    async def _attempt(
        self,
        payload: dict[str, Any],
        request: OpenRouterRequest,
        *,
        structured_mode: str,
        attempt_number: int,
        attempt_timeout: float,
        started_at: float,
    ) -> OpenRouterResult:
        try:
            return await asyncio.wait_for(
                self._run_attempt(payload, request, attempt_number=attempt_number),
                timeout=attempt_timeout,
            )
        except TimeoutError as exc:
            error = UpstreamError(
                "OpenRouter attempt exceeded its time budget", attempt=attempt_number
            )
            self._log_failed_response(
                request,
                attempt=attempt_number,
                started_at=started_at,
                structured_mode=structured_mode,
                error=error,
            )
            raise _RetryableAttemptError(error, reason="deadline_exceeded") from exc
        except httpx.RequestError as exc:
            error = UpstreamError("OpenRouter request failed", attempt=attempt_number)
            self._log_failed_response(
                request,
                attempt=attempt_number,
                started_at=started_at,
                structured_mode=structured_mode,
                error=error,
            )
            raise _RetryableAttemptError(error, reason="network_error") from exc
        except _MappedStatusError as exc:
            error = exc.error
            error.attempt = attempt_number
            self._log_failed_response(
                request,
                attempt=attempt_number,
                started_at=started_at,
                structured_mode=structured_mode,
                error=error,
                status_code=exc.status_code,
            )
            if exc.status_code not in RETRYABLE_STATUS_CODES:
                raise error from exc
            raise _RetryableAttemptError(
                error, reason="retryable_status", status_code=exc.status_code
            ) from exc
        except UpstreamError as exc:
            exc.attempt = attempt_number
            self._log_failed_response(
                request,
                attempt=attempt_number,
                started_at=started_at,
                structured_mode=structured_mode,
                error=exc,
            )
            raise

    async def _run_attempt(
        self,
        payload: dict[str, Any],
        request: OpenRouterRequest,
        *,
        attempt_number: int,
    ) -> OpenRouterResult:
        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                connect=self._settings.timeout_seconds,
                read=self._settings.stream_stall_timeout_seconds,
                write=self._settings.timeout_seconds,
                pool=self._settings.timeout_seconds,
            ),
            transport=self._transport,
        ) as client:
            async with client.stream(
                "POST",
                "/chat/completions",
                headers={
                    "Authorization": (
                        f"Bearer {self._settings.openrouter_api_key.get_secret_value()}"
                    ),
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise _MappedStatusError(
                        _map_error_response(response), status_code=response.status_code
                    )
                assembly = await consume_completion_stream(response)

        usage = assembly.usage or {}
        return OpenRouterResult(
            content=assembly.content,
            model=str(assembly.model or request.model),
            provider=assembly.provider,
            prompt_tokens=_int_usage(usage.get("prompt_tokens")),
            completion_tokens=_int_usage(usage.get("completion_tokens")),
            total_tokens=_int_usage(usage.get("total_tokens")),
            cost_usd=_decimal_usage_cost(usage.get("cost")),
            attempt=attempt_number,
        )

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


class _RetryableAttemptError(Exception):
    def __init__(
        self, error: UpstreamError, *, reason: str, status_code: int | None = None
    ) -> None:
        self.error = error
        self.reason = reason
        self.status_code = status_code
        super().__init__(error.message)


class _MappedStatusError(Exception):
    def __init__(self, error: UpstreamError, *, status_code: int) -> None:
        self.error = error
        self.status_code = status_code
        super().__init__(error.message)


def remaining_budget(deadline: float) -> float:
    return deadline - time.monotonic()


def effective_attempt_timeout(remaining: float, cap: float) -> float:
    return min(cap, remaining)


def backoff_with_jitter(base: float, offset: int, remaining: float) -> float | None:
    nominal = base * (2**offset)
    if nominal >= remaining:
        return None
    return random.uniform(0, nominal)


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
