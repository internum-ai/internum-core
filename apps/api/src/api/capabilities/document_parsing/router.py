from typing import Any

from fastapi import APIRouter, Depends, Request

from api.capabilities.document_parsing.dependencies import get_document_parsing_service
from api.capabilities.document_parsing.intake import parse_multipart_request
from api.common.auth import ConsumerIdentity, get_settings, require_consumer
from api.common.logging import log_event

router = APIRouter(prefix="/v1", dependencies=[Depends(require_consumer)])


@router.post("/parse")
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
