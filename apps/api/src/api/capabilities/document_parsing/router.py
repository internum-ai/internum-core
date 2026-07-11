from typing import Any

from fastapi import APIRouter, Depends, Request

from api.capabilities.document_parsing.dependencies import get_document_parsing_service
from api.capabilities.document_parsing.intake import parse_multipart_request
from api.capabilities.document_parsing.schemas import ParseResponseSchema
from api.common.auth import ConsumerIdentity, get_settings, require_consumer
from api.common.logging import log_event
from api.common.schemas import ErrorResponseSchema

router = APIRouter(prefix="/v1", dependencies=[Depends(require_consumer)])

PARSE_REQUEST_BODY = {
    "content": {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "format": "binary",
                        "description": (
                            "The document to parse. Supported types: PDF, DOCX, DOC, HTML, "
                            "XLSX, XLS, JPG, PNG."
                        ),
                    },
                    "schema": {
                        "type": "string",
                        "description": (
                            "JSON Schema (encoded as a JSON string) describing the shape of "
                            "the structured data to extract from the document."
                        ),
                    },
                    "additionalContext": {
                        "type": "string",
                        "description": (
                            "Optional free-text context to guide extraction, e.g. domain "
                            "hints or instructions for ambiguous fields."
                        ),
                    },
                    "checks": {
                        "type": "string",
                        "description": (
                            "Optional JSON array (encoded as a JSON string) of post-extraction "
                            "checks to evaluate against the extracted data."
                        ),
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional override of the model used for extraction.",
                    },
                    "models": {
                        "type": "string",
                        "description": (
                            "Optional JSON array (encoded as a JSON string) of model names to "
                            "try in order, overriding the default fallback list."
                        ),
                    },
                    "systemPrompt": {
                        "type": "string",
                        "description": "Optional override of the system prompt for extraction.",
                    },
                },
                "required": ["file", "schema"],
            }
        }
    },
    "required": True,
}

PARSE_DESCRIPTION = """
Parses an uploaded document and extracts structured data from it according to a caller-supplied
JSON Schema.

The request body is `multipart/form-data` with the following fields:

- **file** (required): the document to parse. Supported types are PDF, DOCX, DOC, HTML, XLSX,
  XLS, JPG, and PNG.
- **schema** (required): a JSON Schema, encoded as a JSON string, describing the structured data
  to extract.
- **additionalContext** (optional): free-text context to guide extraction.
- **checks** (optional): a JSON array, encoded as a JSON string, of post-extraction checks to run
  against the extracted data.
- **model** (optional): override the model used for extraction.
- **models** (optional): a JSON array, encoded as a JSON string, of models to try in order.
- **systemPrompt** (optional): override the system prompt used for extraction.

Authentication is via the `X-API-Key` header.

On success, the response body contains a `data` object shaped according to the provided schema,
and a `meta` object describing how the document was parsed (document type, extraction mode, page
counts, converter, token usage, post-check results, and chunking summary).
"""


@router.post(
    "/parse",
    summary="Parse a document into structured data",
    description=PARSE_DESCRIPTION,
    tags=["Document Parsing"],
    responses={
        200: {
            "model": ParseResponseSchema,
            "description": "The document was parsed and structured data extracted successfully.",
        },
        400: {
            "model": ErrorResponseSchema,
            "description": (
                "The request could not be processed: malformed multipart form, "
                "unsupported file type (`unsupported_file_type`), or an oversized image "
                "(`image_too_large`)."
            ),
        },
        401: {
            "model": ErrorResponseSchema,
            "description": ("Missing (`missing_api_key`) or invalid (`invalid_api_key`) API key."),
        },
        403: {
            "model": ErrorResponseSchema,
            "description": "The API key has been revoked (`revoked_api_key`).",
        },
        422: {
            "model": ErrorResponseSchema,
            "description": (
                "The supplied schema was invalid (`schema_error`) or the request failed "
                "validation (`validation_error`)."
            ),
        },
        502: {
            "model": ErrorResponseSchema,
            "description": "The upstream extraction provider failed (`upstream_error`).",
        },
    },
    openapi_extra={"requestBody": PARSE_REQUEST_BODY},
)
async def parse_document(request: Request) -> dict[str, Any]:
    settings = get_settings(request)
    consumer = getattr(request.state, "consumer", None)
    log_event(
        "request.received",
        method=request.method,
        path=request.url.path,
        consumerId=consumer.id if isinstance(consumer, ConsumerIdentity) else None,
        contentLength=_content_length(request),
    )
    parse_request = await parse_multipart_request(
        request,
        max_upload_bytes=settings.max_upload_bytes,
    )
    service = get_document_parsing_service(request, settings)
    parsed = await service.parse(
        parse_request,
        consumer_id=consumer.id if isinstance(consumer, ConsumerIdentity) else None,
        request_id=getattr(request.state, "request_id", None),
        is_disconnected=request.is_disconnected,
    )
    return {"data": parsed.data, "meta": parsed.metadata.to_api()}


def _content_length(request: Request) -> int | None:
    value = request.headers.get("content-length")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
