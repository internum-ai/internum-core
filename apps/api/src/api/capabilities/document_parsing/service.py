import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any, Protocol

from api.capabilities.document_parsing.intake import stored_upload
from api.capabilities.document_parsing.models import (
    ExtractedDocument,
    ParsedDocument,
    ParseMetadata,
    ParseMultipartRequest,
    PostCheckResult,
    UsageSummary,
)
from api.common.errors import SchemaError
from api.common.logging import log_event
from api.config.overrides import resolve_request_overrides
from api.config.settings import CoreSettings
from api.platform.openrouter import OpenRouterRequest, OpenRouterResult
from api.platform.schema import (
    evaluate_post_checks,
    format_validation_retry,
    repair_json_output,
    validate_against_original,
)

DISCONNECT_POLL_INTERVAL_SECONDS = 0.05


class OpenRouterCompleter(Protocol):
    async def complete(self, request: OpenRouterRequest) -> OpenRouterResult: ...


class DocumentExtractor(Protocol):
    async def extract(self, upload: Any) -> ExtractedDocument | str: ...


class DocumentParsingService:
    def __init__(
        self,
        settings: CoreSettings,
        extractor: DocumentExtractor,
        openrouter_client: OpenRouterCompleter,
    ) -> None:
        self._settings = settings
        self._extractor = extractor
        self._openrouter_client = openrouter_client

    async def parse(
        self,
        request: ParseMultipartRequest,
        *,
        consumer_id: str | None,
        request_id: str | None,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> ParsedDocument:
        if is_disconnected is None:
            return await self._run_parse(
                request,
                consumer_id=consumer_id,
                request_id=request_id,
            )
        return await self._run_with_disconnect_supervisor(
            request,
            consumer_id=consumer_id,
            request_id=request_id,
            is_disconnected=is_disconnected,
        )

    async def _run_with_disconnect_supervisor(
        self,
        request: ParseMultipartRequest,
        *,
        consumer_id: str | None,
        request_id: str | None,
        is_disconnected: Callable[[], Awaitable[bool]],
    ) -> ParsedDocument:
        task = asyncio.create_task(
            self._run_parse(request, consumer_id=consumer_id, request_id=request_id)
        )
        poller = asyncio.create_task(_poll_disconnect(task, is_disconnected))
        try:
            return await task
        finally:
            poller.cancel()
            try:
                await poller
            except (asyncio.CancelledError, Exception):
                pass

    async def _run_parse(
        self,
        request: ParseMultipartRequest,
        *,
        consumer_id: str | None,
        request_id: str | None,
    ) -> ParsedDocument:
        resolved = resolve_request_overrides(self._settings, request.overrides)
        async with stored_upload(
            request.upload,
            max_upload_bytes=self._settings.max_upload_bytes,
            max_image_pixels=self._settings.max_image_pixels,
            max_ooxml_zip_entries=self._settings.max_ooxml_zip_entries,
            max_ooxml_uncompressed_bytes=self._settings.max_ooxml_uncompressed_bytes,
            max_ooxml_compression_ratio=self._settings.max_ooxml_compression_ratio,
        ) as upload:
            extracted = await self._extractor.extract(upload)
            if isinstance(extracted, ExtractedDocument):
                markdown = extracted.markdown
                metadata = extracted.metadata
            else:
                markdown = extracted
                metadata = ParseMetadata(
                    document_type=upload.document_type,
                    extraction_mode=None,
                    page_count=None,
                    ocr_page_count=None,
                    converter=None,
                )
        openrouter_request = OpenRouterRequest(
            model=resolved.model,
            system_prompt=resolved.system_prompt,
            user_content=_build_user_content(markdown, request.additional_context),
            schema=request.schema,
            capability="document_parsing",
            consumer_id=consumer_id,
            request_id=request_id,
            reasoning_effort=resolved.reasoning_effort,
            temperature=resolved.temperature,
            max_output_tokens=resolved.max_output_tokens,
        )
        result = await self._openrouter_client.complete(openrouter_request)
        try:
            data = _repair_and_validate(
                result.content,
                request.schema,
                failure_triggers_retry=True,
            )
        except SchemaError as error:
            retry_attempt = result.attempt + 1
            log_event(
                "model.retry",
                level=logging.WARNING,
                reason="schema_rejection",
                attempt=retry_attempt,
            )
            retry_result = await self._openrouter_client.complete(
                replace(
                    openrouter_request,
                    validation_retry_prompt=format_validation_retry(error),
                    attempt=retry_attempt,
                )
            )
            data = _repair_and_validate(
                retry_result.content,
                request.schema,
                validation_retry_triggered=True,
            )
            result = retry_result

        usage = UsageSummary(
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_usd=result.cost_usd,
        )
        checks = [
            PostCheckResult(op=outcome.op, passed=outcome.passed, detail=outcome.detail)
            for outcome in evaluate_post_checks(data, request.checks)
        ]
        for check in checks:
            log_event(
                "schema.postcheck",
                level=logging.INFO if check.passed else logging.WARNING,
                op=check.op,
                passed=check.passed,
                detail=check.detail,
                requestId=request_id,
            )
        metadata = replace(metadata, usage=usage, checks=checks)

        return ParsedDocument(data=data, metadata=metadata)


async def _poll_disconnect(
    task: asyncio.Task[ParsedDocument],
    is_disconnected: Callable[[], Awaitable[bool]],
) -> None:
    while not task.done():
        if await is_disconnected() is True:
            task.cancel()
            return
        await asyncio.sleep(DISCONNECT_POLL_INTERVAL_SECONDS)


def _build_user_content(markdown: str, additional_context: str | None) -> str:
    sections = [
        "Extract structured JSON from the Markdown document below.",
        "Return null for unresolved fields instead of guessing.",
    ]
    if additional_context is not None:
        sections.append(f"Additional context:\n{additional_context}")
    sections.append(f"Document Markdown:\n{markdown}")
    return "\n\n".join(sections)


def _repair_and_validate(
    raw_output: str,
    schema: dict[str, Any],
    *,
    validation_retry_triggered: bool = False,
    failure_triggers_retry: bool = False,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    repair_applied = _requires_json_repair(raw_output)
    try:
        repaired = repair_json_output(raw_output)
        validated = validate_against_original(repaired, schema)
        if not isinstance(validated, dict):
            raise SchemaError("Model output must be a JSON object")
    except SchemaError:
        log_event(
            "schema.validation",
            level=logging.WARNING,
            passed=False,
            repairApplied=repair_applied,
            validationRetryTriggered=failure_triggers_retry or validation_retry_triggered,
            durationMs=_duration_ms(started_at),
        )
        raise

    log_event(
        "schema.validation",
        passed=True,
        repairApplied=repair_applied,
        validationRetryTriggered=validation_retry_triggered,
        durationMs=_duration_ms(started_at),
    )
    log_event("schema.values", level=logging.DEBUG, values=validated)
    return validated


def _requires_json_repair(raw_output: str) -> bool:
    try:
        json.loads(raw_output)
    except json.JSONDecodeError:
        return True
    return False


def _duration_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)
