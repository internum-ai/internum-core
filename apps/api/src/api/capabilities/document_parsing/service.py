import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from decimal import Decimal
from typing import Any, Protocol

from api.capabilities.document_parsing.chunking import (
    ChunkingConfig,
    build_chunk_plan,
    merge_chunk_rows,
)
from api.capabilities.document_parsing.chunking import (
    assemble_result as assemble_chunked_result,
)
from api.capabilities.document_parsing.intake import stored_upload
from api.capabilities.document_parsing.models import (
    ChunkingSummary,
    ChunkOutcome,
    ChunkPlan,
    ExtractedDocument,
    ParsedDocument,
    ParseMetadata,
    ParseMultipartRequest,
    PostCheckResult,
    UsageSummary,
)
from api.common.errors import SchemaError
from api.common.logging import log_event
from api.config.overrides import ResolvedModelConfig, resolve_request_overrides
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

        chunk_config = ChunkingConfig(
            row_threshold=self._settings.chunk_row_threshold,
            rows_per_chunk=self._settings.chunk_rows_per_chunk,
        )
        plan = build_chunk_plan(markdown, request.schema, chunk_config)
        if plan is not None:
            return await self._run_chunked_parse(
                request,
                plan,
                resolved,
                metadata,
                consumer_id=consumer_id,
                request_id=request_id,
            )

        openrouter_request = OpenRouterRequest(
            model=resolved.models[0],
            models=resolved.models,
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
        data, result = await self._complete_and_validate(openrouter_request, request.schema)

        usage = UsageSummary(
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_usd=result.cost_usd,
        )
        checks = _run_post_checks(data, request.checks, request_id=request_id)
        metadata = replace(metadata, usage=usage, checks=checks)

        return ParsedDocument(data=data, metadata=metadata)

    async def _complete_and_validate(
        self,
        request: OpenRouterRequest,
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any], OpenRouterResult]:
        result = await self._openrouter_client.complete(request)
        try:
            data = _repair_and_validate(
                result.content,
                schema,
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
                    request,
                    validation_retry_prompt=format_validation_retry(error),
                    attempt=retry_attempt,
                )
            )
            data = _repair_and_validate(
                retry_result.content,
                schema,
                validation_retry_triggered=True,
            )
            result = retry_result

        return data, result

    async def _run_chunked_parse(
        self,
        request: ParseMultipartRequest,
        plan: ChunkPlan,
        resolved: ResolvedModelConfig,
        metadata: ParseMetadata,
        *,
        consumer_id: str | None,
        request_id: str | None,
    ) -> ParsedDocument:
        chunk_model = self._settings.chunk_model or resolved.models[0]
        chunk_models = [chunk_model] if self._settings.chunk_model else resolved.models

        log_event(
            "chunk.plan",
            totalRows=plan.total_rows,
            chunkCount=len(plan.chunks),
            rowsPerChunk=self._settings.chunk_rows_per_chunk,
            requestId=request_id,
        )

        semaphore = asyncio.Semaphore(self._settings.chunk_max_concurrency)
        chunk_results = await asyncio.gather(
            *(
                self._run_chunk(
                    chunk,
                    plan,
                    request,
                    resolved,
                    chunk_model,
                    chunk_models,
                    semaphore,
                    consumer_id=consumer_id,
                    request_id=request_id,
                )
                for chunk in plan.chunks
            ),
            return_exceptions=True,
        )

        outcomes: list[ChunkOutcome] = []
        results: list[OpenRouterResult] = []
        for chunk, outcome_result in zip(plan.chunks, chunk_results, strict=True):
            if isinstance(outcome_result, BaseException):
                outcomes.append(
                    ChunkOutcome(index=chunk.index, rows=None, error=str(outcome_result))
                )
                continue
            outcome, result = outcome_result
            outcomes.append(outcome)
            results.append(result)

        summary_data: dict[str, Any] | None = None
        if plan.summary_schema is not None:
            summary_request = OpenRouterRequest(
                model=chunk_model,
                models=chunk_models,
                system_prompt=resolved.system_prompt,
                user_content=_build_user_content(plan.summary_markdown, request.additional_context),
                schema=plan.summary_schema,
                capability="document_parsing",
                consumer_id=consumer_id,
                request_id=request_id,
                reasoning_effort=resolved.reasoning_effort,
                temperature=resolved.temperature,
                max_output_tokens=resolved.max_output_tokens,
                cache_control=self._settings.chunk_prompt_cache,
            )
            summary_data, summary_result = await self._complete_and_validate(
                summary_request, plan.summary_schema
            )
            results.append(summary_result)

        merged_rows, failed = merge_chunk_rows(outcomes)
        assembled = assemble_chunked_result(plan, merged_rows, summary_data)

        if failed:
            log_event(
                "chunk.partial",
                level=logging.WARNING,
                failedChunks=failed,
                requestId=request_id,
            )
            if not self._settings.chunk_allow_partial:
                failed_ranges = [
                    [chunk.start_row, chunk.end_row]
                    for chunk in plan.chunks
                    if chunk.index in failed
                ]
                raise SchemaError(
                    "Chunked extraction failed for one or more row ranges",
                    details={"failedChunks": failed, "failedRanges": failed_ranges},
                )

        validated = validate_against_original(assembled, request.schema)
        if not isinstance(validated, dict):
            raise SchemaError("Model output must be a JSON object")

        usage = _aggregate_usage(results, chunk_model)
        checks = _run_post_checks(validated, request.checks, request_id=request_id)
        chunking_summary = ChunkingSummary(
            chunked=True,
            total_rows=plan.total_rows,
            chunk_count=len(plan.chunks),
            failed_chunks=failed,
            partial=bool(failed),
            model=chunk_model,
        )
        metadata = replace(metadata, usage=usage, checks=checks, chunking=chunking_summary)

        return ParsedDocument(data=validated, metadata=metadata)

    async def _run_chunk(
        self,
        chunk: Any,
        plan: ChunkPlan,
        request: ParseMultipartRequest,
        resolved: ResolvedModelConfig,
        chunk_model: str,
        chunk_models: list[str],
        semaphore: asyncio.Semaphore,
        *,
        consumer_id: str | None,
        request_id: str | None,
    ) -> tuple[ChunkOutcome, OpenRouterResult]:
        async with semaphore:
            chunk_request = OpenRouterRequest(
                model=chunk_model,
                models=chunk_models,
                system_prompt=resolved.system_prompt,
                user_content=_build_user_content(chunk.markdown, request.additional_context),
                schema=plan.chunk_schema,
                capability="document_parsing",
                consumer_id=consumer_id,
                request_id=request_id,
                reasoning_effort=resolved.reasoning_effort,
                temperature=resolved.temperature,
                max_output_tokens=resolved.max_output_tokens,
                cache_control=self._settings.chunk_prompt_cache,
            )
            log_event(
                "chunk.request",
                index=chunk.index,
                requestId=request_id,
            )
            property_name = plan.array_location.property_name
            last_error: Exception | None = None
            for _attempt in range(2):
                try:
                    data, result = await self._complete_and_validate(
                        chunk_request, plan.chunk_schema
                    )
                    rows = data[property_name]
                    log_event(
                        "chunk.response",
                        index=chunk.index,
                        outcome="succeeded",
                        rowCount=len(rows),
                        requestId=request_id,
                    )
                    return ChunkOutcome(index=chunk.index, rows=rows, error=None), result
                except Exception as error:  # noqa: BLE001 - captured per-chunk, never bubbles
                    last_error = error
                    continue

            log_event(
                "chunk.response",
                level=logging.WARNING,
                index=chunk.index,
                outcome="failed",
                requestId=request_id,
            )
            raise last_error if last_error is not None else RuntimeError("chunk failed")


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


def _run_post_checks(
    data: dict[str, Any],
    checks: list[dict[str, Any]],
    *,
    request_id: str | None,
) -> list[PostCheckResult]:
    results = [
        PostCheckResult(op=outcome.op, passed=outcome.passed, detail=outcome.detail)
        for outcome in evaluate_post_checks(data, checks)
    ]
    for check in results:
        log_event(
            "schema.postcheck",
            level=logging.INFO if check.passed else logging.WARNING,
            op=check.op,
            passed=check.passed,
            detail=check.detail,
            requestId=request_id,
        )
    return results


def _aggregate_usage(results: list[OpenRouterResult], model: str) -> UsageSummary:
    prompt_tokens = sum(result.prompt_tokens for result in results)
    completion_tokens = sum(result.completion_tokens for result in results)
    total_tokens = sum(result.total_tokens for result in results)
    costs = [result.cost_usd for result in results]
    cost_usd: Decimal | None
    if all(cost is None for cost in costs):
        cost_usd = None
    else:
        cost_usd = sum((cost for cost in costs if cost is not None), Decimal("0"))
    return UsageSummary(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )


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
