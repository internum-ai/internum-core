from fastapi import Request

from api.capabilities.document_parsing.extraction import build_extractor
from api.capabilities.document_parsing.models import ExtractedDocument, StoredUpload
from api.capabilities.document_parsing.service import (
    DocumentExtractor,
    DocumentParsingService,
    OpenRouterCompleter,
)
from api.common.usage import UsageTracker
from api.config.settings import CoreSettings
from api.platform.openrouter import OpenRouterClient


def get_document_parsing_service(
    request: Request,
    settings: CoreSettings,
) -> DocumentParsingService:
    configured_service = getattr(request.app.state, "document_parsing_service", None)
    if isinstance(configured_service, DocumentParsingService):
        return configured_service

    extractor = getattr(request.app.state, "document_extractor", None) or LazyDocumentExtractor(
        settings
    )
    openrouter_client = get_openrouter_client(request, settings)
    return DocumentParsingService(settings, extractor, openrouter_client)


def get_openrouter_client(request: Request, settings: CoreSettings) -> OpenRouterCompleter:
    configured_client = getattr(request.app.state, "openrouter_client", None)
    if configured_client is not None:
        return configured_client

    tracker = getattr(request.app.state, "usage_tracker", None)
    if not isinstance(tracker, UsageTracker):
        raise TypeError("usage_tracker must implement UsageTracker")
    return OpenRouterClient(settings, tracker)


class LazyDocumentExtractor:
    def __init__(self, settings: CoreSettings) -> None:
        self._settings = settings
        self._extractor: DocumentExtractor | None = None

    async def extract(self, upload: StoredUpload) -> ExtractedDocument | str:
        if self._extractor is None:
            self._extractor = build_extractor(self._settings)
        return await self._extractor.extract(upload)
