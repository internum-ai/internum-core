import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from api.capabilities.document_parsing.extraction import (
    MarkItDownExtractor,
    build_markitdown_converter,
)
from api.capabilities.document_parsing.models import StoredUpload, SupportedDocumentType
from api.common.errors import IntakeError
from api.config.settings import CoreSettings
from api.platform.openrouter import OPENROUTER_BASE_URL


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
@pytest.mark.parametrize(
    ("document_type", "expected_extension"),
    [
        (SupportedDocumentType.HTML, ".html"),
        (SupportedDocumentType.XLS, ".xls"),
    ],
)
async def test_extractor_uses_convert_local_with_detected_extension(
    core_settings: CoreSettings,
    tmp_path: Path,
    document_type: SupportedDocumentType,
    expected_extension: str,
) -> None:
    calls: dict[str, object] = {}

    @dataclass
    class Result:
        text_content: str

    class Converter:
        def convert_local(self, path: str | Path, **kwargs: object) -> Result:
            calls["path"] = path
            calls["kwargs"] = kwargs
            return Result(text_content="# Markdown")

    upload_path = tmp_path / f"sample{expected_extension}"
    upload_path.write_bytes(b"content")
    upload = StoredUpload(
        path=upload_path,
        document_type=document_type,
        size_bytes=9,
        original_filename=f"sample{expected_extension}",
        detected_mime=None,
    )

    extracted = await MarkItDownExtractor(Converter(), core_settings).extract(upload)

    assert extracted.markdown == "# Markdown"
    assert extracted.metadata.document_type is document_type
    assert calls["path"] == upload_path
    assert calls["kwargs"] == {"file_extension": expected_extension}


@pytest.mark.anyio
async def test_pdf_preflight_reports_native_metadata(
    core_settings: CoreSettings,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    @dataclass
    class Result:
        text_content: str

    class Converter:
        def convert_local(self, path: str | Path, **kwargs: object) -> Result:
            calls["kwargs"] = kwargs
            return Result(text_content="# Native")

    upload_path = _native_pdf(tmp_path / "native.pdf")
    extracted = await MarkItDownExtractor(Converter(), core_settings).extract(
        _upload(upload_path, SupportedDocumentType.PDF)
    )

    assert extracted.metadata.to_api() == {
        "documentType": "pdf",
        "extractionMode": "native",
        "pageCount": 1,
        "ocrPageCount": 0,
        "converter": "markitdown",
    }
    assert calls["kwargs"] == {"file_extension": ".pdf"}


@pytest.mark.anyio
async def test_pdf_preflight_reports_scan_metadata(
    core_settings: CoreSettings,
    tmp_path: Path,
) -> None:
    @dataclass
    class Result:
        text_content: str

    class Converter:
        def convert_local(self, path: str | Path, **kwargs: object) -> Result:
            return Result(text_content="# OCR")

    upload_path = _scan_pdf(tmp_path / "scan.pdf")
    extracted = await MarkItDownExtractor(Converter(), core_settings).extract(
        _upload(upload_path, SupportedDocumentType.PDF)
    )

    assert extracted.metadata.to_api()["extractionMode"] == "scan"
    assert extracted.metadata.to_api()["ocrPageCount"] == 1


@pytest.mark.anyio
async def test_pdf_preflight_reports_mixed_metadata(
    core_settings: CoreSettings,
    tmp_path: Path,
) -> None:
    @dataclass
    class Result:
        text_content: str

    class Converter:
        def convert_local(self, path: str | Path, **kwargs: object) -> Result:
            return Result(text_content="# Mixed")

    upload_path = _mixed_pdf(tmp_path / "mixed.pdf")
    extracted = await MarkItDownExtractor(Converter(), core_settings).extract(
        _upload(upload_path, SupportedDocumentType.PDF)
    )

    assert extracted.metadata.to_api()["extractionMode"] == "mixed"
    assert extracted.metadata.to_api()["pageCount"] == 2
    assert extracted.metadata.to_api()["ocrPageCount"] == 1


@pytest.mark.anyio
async def test_pdf_over_page_limit_returns_stable_error(
    core_settings: CoreSettings,
    tmp_path: Path,
) -> None:
    settings = core_settings.model_copy(update={"max_pdf_pages": 1})
    upload_path = _two_page_pdf(tmp_path / "large.pdf")

    with pytest.raises(IntakeError) as exc_info:
        await MarkItDownExtractor(_NeverCalledConverter(), settings).extract(
            _upload(upload_path, SupportedDocumentType.PDF)
        )

    assert exc_info.value.code == "pdf_page_limit_exceeded"


@pytest.mark.anyio
async def test_encrypted_pdf_returns_stable_error(
    core_settings: CoreSettings,
    tmp_path: Path,
) -> None:
    upload_path = _encrypted_pdf(tmp_path / "encrypted.pdf")

    with pytest.raises(IntakeError) as exc_info:
        await MarkItDownExtractor(_NeverCalledConverter(), core_settings).extract(
            _upload(upload_path, SupportedDocumentType.PDF)
        )

    assert exc_info.value.code == "encrypted_pdf"


@pytest.mark.anyio
async def test_pdf_over_ocr_page_limit_returns_stable_error(
    core_settings: CoreSettings,
    tmp_path: Path,
) -> None:
    settings = core_settings.model_copy(update={"max_ocr_pages": 0})
    upload_path = _scan_pdf(tmp_path / "scan.pdf")

    with pytest.raises(IntakeError) as exc_info:
        await MarkItDownExtractor(_NeverCalledConverter(), settings).extract(
            _upload(upload_path, SupportedDocumentType.PDF)
        )

    assert exc_info.value.code == "ocr_page_limit_exceeded"


@pytest.mark.anyio
async def test_pdf_over_ocr_pixel_limit_returns_stable_error(
    core_settings: CoreSettings,
    tmp_path: Path,
) -> None:
    settings = core_settings.model_copy(update={"max_ocr_rendered_pixels": 1})
    upload_path = _scan_pdf(tmp_path / "scan.pdf")

    with pytest.raises(IntakeError) as exc_info:
        await MarkItDownExtractor(_NeverCalledConverter(), settings).extract(
            _upload(upload_path, SupportedDocumentType.PDF)
        )

    assert exc_info.value.code == "ocr_image_too_large"


@pytest.mark.anyio
async def test_doc_upload_converts_to_docx_before_markitdown(
    core_settings: CoreSettings,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    binary = _fake_soffice(tmp_path)
    settings = core_settings.model_copy(update={"libreoffice_binary": str(binary)})

    @dataclass
    class Result:
        text_content: str

    class Converter:
        def convert_local(self, path: str | Path, **kwargs: object) -> Result:
            calls["path"] = Path(path)
            calls["kwargs"] = kwargs
            assert Path(path).exists()
            return Result(text_content="# DOC")

    upload_path = tmp_path / "sample.doc"
    upload_path.write_bytes(b"legacy-doc")

    extracted = await MarkItDownExtractor(Converter(), settings).extract(
        _upload(upload_path, SupportedDocumentType.DOC)
    )

    assert extracted.markdown == "# DOC"
    assert calls["path"].suffix == ".docx"  # type: ignore[union-attr]
    assert calls["kwargs"] == {"file_extension": ".docx"}
    assert not calls["path"].exists()  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_missing_doc_converter_returns_stable_error(
    core_settings: CoreSettings,
    tmp_path: Path,
) -> None:
    settings = core_settings.model_copy(update={"libreoffice_binary": str(tmp_path / "missing")})
    upload_path = tmp_path / "sample.doc"
    upload_path.write_bytes(b"legacy-doc")

    with pytest.raises(IntakeError) as exc_info:
        await MarkItDownExtractor(_NeverCalledConverter(), settings).extract(
            _upload(upload_path, SupportedDocumentType.DOC)
        )

    assert exc_info.value.code == "doc_converter_unavailable"


@pytest.mark.anyio
async def test_invalid_doc_converter_cleans_temp_files_and_returns_stable_error(
    core_settings: CoreSettings,
    tmp_path: Path,
) -> None:
    binary = tmp_path / "soffice"
    binary.write_text("not an executable format")
    binary.chmod(0o755)
    settings = core_settings.model_copy(update={"libreoffice_binary": str(binary)})
    upload_path = tmp_path / "sample.doc"
    upload_path.write_bytes(b"legacy-doc")
    before = _conversion_temp_dirs()

    with pytest.raises(IntakeError) as exc_info:
        await MarkItDownExtractor(_NeverCalledConverter(), settings).extract(
            _upload(upload_path, SupportedDocumentType.DOC)
        )

    assert exc_info.value.code == "doc_converter_unavailable"
    assert _conversion_temp_dirs() == before


@pytest.mark.anyio
async def test_generic_conversion_failure_returns_stable_error(
    core_settings: CoreSettings,
    tmp_path: Path,
) -> None:
    class Converter:
        def convert_local(self, path: str | Path, **kwargs: object) -> object:
            raise RuntimeError("boom")

    upload_path = tmp_path / "sample.html"
    upload_path.write_text("<html></html>")

    with pytest.raises(IntakeError) as exc_info:
        await MarkItDownExtractor(Converter(), core_settings).extract(
            _upload(upload_path, SupportedDocumentType.HTML)
        )

    assert exc_info.value.code == "document_conversion_failed"


class _NeverCalledConverter:
    def convert_local(self, path: str | Path, **kwargs: object) -> object:
        raise AssertionError("converter should not be called")


def _upload(path: Path, document_type: SupportedDocumentType) -> StoredUpload:
    return StoredUpload(
        path=path,
        document_type=document_type,
        size_bytes=path.stat().st_size,
        original_filename=path.name,
        detected_mime=None,
    )


def _native_pdf(path: Path) -> Path:
    import fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "This native PDF has enough text to classify as native.")
    document.save(path)
    document.close()
    return path


def _two_page_pdf(path: Path) -> Path:
    import fitz

    document = fitz.open()
    document.new_page().insert_text((72, 72), "Page one has native text.")
    document.new_page().insert_text((72, 72), "Page two has native text.")
    document.save(path)
    document.close()
    return path


def _encrypted_pdf(path: Path) -> Path:
    import fitz

    document = fitz.open()
    document.new_page().insert_text((72, 72), "Secret PDF")
    document.save(
        path,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    document.close()
    return path


def _scan_pdf(path: Path) -> Path:
    import fitz

    image_path = path.with_suffix(".png")
    _scan_image(image_path)
    document = fitz.open()
    page = document.new_page(width=300, height=180)
    page.insert_image(page.rect, filename=str(image_path))
    document.save(path)
    document.close()
    return path


def _mixed_pdf(path: Path) -> Path:
    import fitz

    image_path = path.with_suffix(".png")
    _scan_image(image_path)
    document = fitz.open()
    native_page = document.new_page(width=300, height=180)
    native_page.insert_text((24, 48), "This page contains enough native text for classification.")
    scan_page = document.new_page(width=300, height=180)
    scan_page.insert_image(scan_page.rect, filename=str(image_path))
    document.save(path)
    document.close()
    return path


def _scan_image(path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (300, 180), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 80), "Scanned title", fill="black")
    image.save(path, format="PNG")


def _fake_soffice(tmp_path: Path) -> Path:
    binary = tmp_path / "soffice"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "outdir = pathlib.Path(sys.argv[sys.argv.index('--outdir') + 1])\n"
        "outdir.mkdir(parents=True, exist_ok=True)\n"
        "(outdir / 'input.docx').write_bytes(b'docx')\n"
    )
    binary.chmod(0o755)
    return binary


def _conversion_temp_dirs() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob("internum-doc-convert-*"))
