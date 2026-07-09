from typing import Any

from fastapi import APIRouter, Depends, Request

from api.capabilities.document_parsing.dependencies import get_document_parsing_service
from api.capabilities.document_parsing.intake import parse_multipart_request
from api.common.auth import ConsumerIdentity, get_settings, require_consumer

router = APIRouter(prefix="/v1", dependencies=[Depends(require_consumer)])


@router.post("/parse")
async def parse_document(request: Request) -> dict[str, Any]:
    settings = get_settings(request)
    parse_request = await parse_multipart_request(
        request,
        max_upload_bytes=settings.max_upload_bytes,
    )
    consumer = getattr(request.state, "consumer", None)
    service = get_document_parsing_service(request, settings)
    parsed = await service.parse(
        parse_request,
        consumer_id=consumer.id if isinstance(consumer, ConsumerIdentity) else None,
        request_id=getattr(request.state, "request_id", None),
    )
    return {"data": parsed.data, "meta": parsed.metadata.to_api()}
