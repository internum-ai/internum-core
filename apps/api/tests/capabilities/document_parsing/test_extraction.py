from dataclasses import dataclass
from pathlib import Path

import pytest

from api.capabilities.document_parsing.extraction import (
    OPENROUTER_BASE_URL,
    MarkItDownExtractor,
    build_markitdown_converter,
)
from api.capabilities.document_parsing.models import StoredUpload, SupportedDocumentType
from api.config.settings import CoreSettings


def test_build_markitdown_converter_wires_openrouter_vision_client(
    core_settings: CoreSettings,
) -> None:
    calls: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            calls["openai"] = kwargs

    class FakeMarkItDown:
        def __init__(self, **kwargs: object) -> None:
            calls["markitdown"] = kwargs

        def convert_local(self, path: str | Path, **kwargs: object) -> object:
            return object()

    build_markitdown_converter(
        core_settings,
        markitdown_cls=FakeMarkItDown,
        openai_cls=FakeOpenAI,
    )

    assert calls["openai"] == {
        "api_key": "openrouter-test-key",
        "base_url": OPENROUTER_BASE_URL,
    }
    markitdown_args = calls["markitdown"]
    assert markitdown_args["enable_plugins"] is True  # type: ignore[index]
    assert markitdown_args["llm_model"] == core_settings.default_model  # type: ignore[index]
    assert "llm_client" in markitdown_args  # type: ignore[operator]


@pytest.mark.anyio
async def test_extractor_uses_convert_local_with_detected_extension(tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    @dataclass
    class Result:
        text_content: str

    class Converter:
        def convert_local(self, path: str | Path, **kwargs: object) -> Result:
            calls["path"] = path
            calls["kwargs"] = kwargs
            return Result(text_content="# Markdown")

    upload_path = tmp_path / "sample.pdf"
    upload_path.write_bytes(b"%PDF-1.4\n")
    upload = StoredUpload(
        path=upload_path,
        document_type=SupportedDocumentType.PDF,
        size_bytes=9,
        original_filename="sample.pdf",
        detected_mime="application/pdf",
    )

    markdown = await MarkItDownExtractor(Converter()).extract(upload)

    assert markdown == "# Markdown"
    assert calls["path"] == upload_path
    assert calls["kwargs"] == {"file_extension": ".pdf"}
