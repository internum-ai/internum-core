from pathlib import Path
from typing import Any, Protocol

from starlette.concurrency import run_in_threadpool

from api.capabilities.document_parsing.models import StoredUpload
from api.common.errors import IntakeError
from api.config.settings import CoreSettings
from api.platform.openrouter import build_openai_compatible_client


class MarkItDownConverter(Protocol):
    def convert_local(
        self,
        path: str | Path,
        **kwargs: Any,
    ) -> Any: ...


class MarkItDownExtractor:
    def __init__(self, converter: MarkItDownConverter) -> None:
        self._converter = converter

    async def extract(self, upload: StoredUpload) -> str:
        try:
            result = await run_in_threadpool(
                self._converter.convert_local,
                upload.path,
                file_extension=upload.document_type.extension,
            )
        except Exception as exc:
            raise IntakeError("Document could not be converted to Markdown") from exc

        text_content = getattr(result, "text_content", None)
        if not isinstance(text_content, str) or not text_content.strip():
            raise IntakeError("Document conversion produced no Markdown content")
        return text_content


def build_markitdown_converter(
    settings: CoreSettings,
    *,
    markitdown_cls: type[Any] | None = None,
    openai_cls: type[Any] | None = None,
) -> MarkItDownConverter:
    if markitdown_cls is None:
        from markitdown import MarkItDown

        markitdown_cls = MarkItDown
    llm_client = build_openai_compatible_client(settings, openai_cls=openai_cls)
    return markitdown_cls(
        enable_plugins=True,
        llm_client=llm_client,
        llm_model=settings.default_model,
    )


def build_extractor(settings: CoreSettings) -> MarkItDownExtractor:
    return MarkItDownExtractor(build_markitdown_converter(settings))
