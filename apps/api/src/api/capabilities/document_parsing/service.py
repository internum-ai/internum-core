from dataclasses import replace
from typing import Any, Protocol

from api.capabilities.document_parsing.intake import stored_upload
from api.capabilities.document_parsing.models import ParseMultipartRequest
from api.common.errors import SchemaError
from api.config.overrides import resolve_request_overrides
from api.config.settings import CoreSettings
from api.platform.openrouter import OpenRouterRequest, OpenRouterResult
from api.platform.schema import (
    format_validation_retry,
    repair_json_output,
    validate_against_original,
)


class OpenRouterCompleter(Protocol):
    async def complete(self, request: OpenRouterRequest) -> OpenRouterResult: ...


class DocumentExtractor(Protocol):
    async def extract(self, upload: Any) -> str: ...


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
    ) -> dict[str, Any]:
        resolved = resolve_request_overrides(self._settings, request.overrides)
        async with stored_upload(
            request.upload,
            max_upload_bytes=self._settings.max_upload_bytes,
        ) as upload:
            markdown = await self._extractor.extract(upload)

        openrouter_request = OpenRouterRequest(
            model=resolved.model,
            system_prompt=resolved.system_prompt,
            user_content=_build_user_content(markdown, request.additional_context),
            schema=request.schema,
            capability="document_parsing",
            consumer_id=consumer_id,
            request_id=request_id,
        )
        result = await self._openrouter_client.complete(openrouter_request)
        try:
            return _repair_and_validate(result.content, request.schema)
        except SchemaError as error:
            retry_result = await self._openrouter_client.complete(
                replace(
                    openrouter_request,
                    validation_retry_prompt=format_validation_retry(error),
                )
            )
            return _repair_and_validate(retry_result.content, request.schema)


def _build_user_content(markdown: str, additional_context: str | None) -> str:
    sections = [
        "Extract structured JSON from the Markdown document below.",
        "Return null for unresolved fields instead of guessing.",
    ]
    if additional_context is not None:
        sections.append(f"Additional context:\n{additional_context}")
    sections.append(f"Document Markdown:\n{markdown}")
    return "\n\n".join(sections)


def _repair_and_validate(raw_output: str, schema: dict[str, Any]) -> dict[str, Any]:
    repaired = repair_json_output(raw_output)
    validated = validate_against_original(repaired, schema)
    if not isinstance(validated, dict):
        raise SchemaError("Model output must be a JSON object")
    return validated
