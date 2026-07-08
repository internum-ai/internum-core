from typing import Any

from fastapi import APIRouter, Depends, Request

from api.capabilities.document_parsing.intake import parse_multipart_request
from api.capabilities.document_parsing.service import DocumentParsingService, OpenRouterCompleter
from api.common.auth import ConsumerIdentity, get_settings, require_consumer
from api.common.usage import UsageTracker
from api.config.settings import CoreSettings
from api.platform.openrouter import OpenRouterClient

router = APIRouter(prefix="/v1", dependencies=[Depends(require_consumer)])


@router.post("/parse")
async def parse_document(request: Request) -> dict[str, Any]:
    parse_request = await parse_multipart_request(request)
    settings = get_settings(request)
    consumer = getattr(request.state, "consumer", None)
    service = _build_service(request, settings)
    data = await service.parse(
        parse_request,
        consumer_id=consumer.id if isinstance(consumer, ConsumerIdentity) else None,
        request_id=getattr(request.state, "request_id", None),
    )
    return {"data": data}


def _build_service(request: Request, settings: CoreSettings) -> DocumentParsingService:
    configured_service = getattr(request.app.state, "document_parsing_service", None)
    if isinstance(configured_service, DocumentParsingService):
        return configured_service

    extractor = getattr(request.app.state, "document_extractor", None)
    openrouter_client = _get_openrouter_client(request, settings)
    return DocumentParsingService(settings, extractor, openrouter_client)


def _get_openrouter_client(request: Request, settings: CoreSettings) -> OpenRouterCompleter:
    configured_client = getattr(request.app.state, "openrouter_client", None)
    if configured_client is not None:
        return configured_client

    tracker = getattr(request.app.state, "usage_tracker", None)
    if not isinstance(tracker, UsageTracker):
        raise TypeError("usage_tracker must implement UsageTracker")
    return OpenRouterClient(settings, tracker)
