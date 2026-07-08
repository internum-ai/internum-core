import asyncio
import json
from decimal import Decimal
from typing import Any

import httpx

from api.common.errors import UpstreamError
from api.common.usage import UsageRecord, UsageTracker
from api.config.settings import CoreSettings
from api.platform.schema.normalize import normalize_for_model

from .models import ImageInput, OpenRouterRequest, OpenRouterResult

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenRouterClient:
    def __init__(
        self,
        settings: CoreSettings,
        usage_tracker: UsageTracker,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
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
        native_payload = self._build_payload(request, native_structured=True)
        try:
            return await self._post_payload(native_payload, request)
        except UpstreamError as exc:
            if not _is_native_schema_rejection(exc):
                raise

        fallback_payload = self._build_payload(request, native_structured=False)
        return await self._post_payload(fallback_payload, request)

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
    ) -> OpenRouterResult:
        last_error: UpstreamError | None = None
        for attempt in range(self._max_retries + 1):
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

            if response.status_code < 400:
                result = self._parse_success_response(response.json(), request)
                self._record_usage(result, request)
                return result

            last_error = _map_error_response(response)
            if response.status_code not in RETRYABLE_STATUS_CODES or attempt >= self._max_retries:
                raise last_error
            await asyncio.sleep(self._backoff_seconds * (2**attempt))

        raise last_error or UpstreamError()

    def _parse_success_response(
        self,
        data: dict[str, Any],
        request: OpenRouterRequest,
    ) -> OpenRouterResult:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise UpstreamError("OpenRouter response did not include choices")

        choice = choices[0]
        if isinstance(choice, dict) and choice.get("error"):
            raise UpstreamError("OpenRouter provider returned an error")

        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise UpstreamError("OpenRouter response did not include text content")

        usage_data = data.get("usage")
        usage = usage_data if isinstance(usage_data, dict) else {}
        cost = usage.get("cost")
        result = OpenRouterResult(
            content=content,
            model=str(data.get("model") or request.model),
            provider=str(data["provider"]) if data.get("provider") is not None else None,
            prompt_tokens=_int_usage(usage.get("prompt_tokens")),
            completion_tokens=_int_usage(usage.get("completion_tokens")),
            total_tokens=_int_usage(usage.get("total_tokens")),
            cost_usd=Decimal(str(cost)) if cost is not None else None,
        )
        return result

    def _record_usage(self, result: OpenRouterResult, request: OpenRouterRequest) -> None:
        self._usage_tracker.record(
            UsageRecord(
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


def _is_native_schema_rejection(error: UpstreamError) -> bool:
    message = error.message.lower()
    markers = ("schema", "response_format", "structured", "require_parameters", "unsupported")
    return any(marker in message for marker in markers)


def _int_usage(value: Any) -> int:
    return value if isinstance(value, int) else 0
