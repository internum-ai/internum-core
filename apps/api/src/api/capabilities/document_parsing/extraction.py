import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

import fitz  # type: ignore[import-untyped]
from starlette.concurrency import run_in_threadpool

from api.capabilities.document_parsing.models import (
    ExtractedDocument,
    ExtractionMode,
    ParseMetadata,
    PdfPreflightResult,
    StoredUpload,
    SupportedDocumentType,
)
from api.common.errors import ConfigurationError, IntakeError
from api.common.logging import log_event
from api.config.settings import CoreSettings
from api.platform.openrouter import build_openai_compatible_client


class MarkItDownConverter(Protocol):
    def convert_local(
        self,
        path: str | Path,
        **kwargs: Any,
    ) -> Any: ...


class MarkItDownExtractor:
    def __init__(self, converter: MarkItDownConverter, settings: CoreSettings) -> None:
        self._converter = converter
        self._settings = settings

    async def extract(self, upload: StoredUpload) -> ExtractedDocument:
        preflight = None
        source_path = upload.path
        extension = upload.document_type.extension
        document_type = upload.document_type

        if upload.document_type is SupportedDocumentType.PDF:
            started_at = time.perf_counter()
            try:
                preflight = await run_in_threadpool(_preflight_pdf, upload.path, self._settings)
            except IntakeError as exc:
                details = exc.details if isinstance(exc.details, dict) else {}
                log_event(
                    "pdf.preflight",
                    level=logging.WARNING,
                    outcome="rejected",
                    errorCode=exc.code,
                    pageCount=details.get("pageCount"),
                    durationMs=_duration_ms(started_at),
                )
                raise
            log_event(
                "pdf.preflight",
                outcome="passed",
                pageCount=preflight.page_count,
                nativeTextPages=preflight.native_text_pages,
                scanLikePages=preflight.scan_like_pages,
                extractionMode=preflight.extraction_mode.value,
                durationMs=_duration_ms(started_at),
            )
        elif upload.document_type is SupportedDocumentType.DOC:
            source_path = await run_in_threadpool(_convert_doc_to_docx, upload.path, self._settings)
            extension = SupportedDocumentType.DOCX.extension

        started_at = time.perf_counter()
        try:
            result = await run_in_threadpool(
                self._converter.convert_local,
                source_path,
                file_extension=extension,
            )
        except Exception as exc:
            log_event(
                "markitdown.convert",
                level=logging.WARNING,
                converter="markitdown",
                extension=extension,
                outcome="failed",
                errorCode="document_conversion_failed",
                durationMs=_duration_ms(started_at),
            )
            raise IntakeError(
                "Document could not be converted to Markdown",
                code="document_conversion_failed",
                details={"documentType": document_type.value},
            ) from exc
        finally:
            if source_path != upload.path:
                _cleanup_conversion_path(source_path)

        text_content = getattr(result, "text_content", None)
        if not isinstance(text_content, str) or not text_content.strip():
            log_event(
                "markitdown.convert",
                level=logging.WARNING,
                converter="markitdown",
                extension=extension,
                outcome="empty",
                errorCode="document_conversion_failed",
                durationMs=_duration_ms(started_at),
                markdownLength=len(text_content) if isinstance(text_content, str) else 0,
            )
            raise IntakeError(
                "Document conversion produced no Markdown content",
                code="document_conversion_failed",
                details={"documentType": document_type.value},
            )
        log_event(
            "markitdown.convert",
            converter="markitdown",
            extension=extension,
            outcome="succeeded",
            durationMs=_duration_ms(started_at),
            markdownLength=len(text_content),
        )
        log_event("markitdown.output", level=logging.DEBUG, markdown=text_content)
        return ExtractedDocument(
            markdown=text_content,
            metadata=_metadata_for(upload.document_type, preflight),
        )


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
    return MarkItDownExtractor(build_markitdown_converter(settings), settings)


def validate_document_runtime(settings: CoreSettings) -> None:
    binary = _resolve_executable(settings.libreoffice_binary)
    if binary is None or not _is_runnable_libreoffice(binary):
        raise ConfigurationError(
            "LibreOffice binary is required for DOC parsing but was not found",
            code="doc_converter_unavailable",
        )


def _metadata_for(
    document_type: SupportedDocumentType,
    preflight: PdfPreflightResult | None,
) -> ParseMetadata:
    if preflight is None:
        return ParseMetadata(
            document_type=document_type,
            extraction_mode=None,
            page_count=None,
            ocr_page_count=None,
            converter="markitdown",
        )
    return ParseMetadata(
        document_type=document_type,
        extraction_mode=preflight.extraction_mode,
        page_count=preflight.page_count,
        ocr_page_count=preflight.scan_like_pages,
        converter="markitdown",
    )


def _preflight_pdf(path: Path, settings: CoreSettings) -> PdfPreflightResult:
    try:
        document = fitz.open(path)
    except Exception as exc:
        raise IntakeError("Unsupported file type", code="unsupported_file_type") from exc
    try:
        if document.is_encrypted or getattr(document, "needs_pass", False):
            raise IntakeError("Encrypted PDFs are not supported", code="encrypted_pdf")
        page_count = document.page_count
        if page_count > settings.max_pdf_pages:
            raise IntakeError(
                "PDF exceeds the configured page limit",
                code="pdf_page_limit_exceeded",
                details={"pageCount": page_count, "maxPdfPages": settings.max_pdf_pages},
            )

        native_text_pages = 0
        scan_like_pages = 0
        empty_pages = 0
        for page in document:
            text = page.get_text("text").strip()
            has_native_text = len(text) >= 20
            has_image = bool(page.get_images(full=True))
            if has_native_text:
                native_text_pages += 1
            elif has_image:
                _enforce_ocr_pixel_budget(page.rect, settings)
                scan_like_pages += 1
            else:
                empty_pages += 1

        if scan_like_pages > settings.max_ocr_pages:
            raise IntakeError(
                "PDF exceeds the configured OCR page limit",
                code="ocr_page_limit_exceeded",
                details={"ocrPageCount": scan_like_pages, "maxOcrPages": settings.max_ocr_pages},
            )

        if native_text_pages and scan_like_pages:
            mode = ExtractionMode.MIXED
        elif scan_like_pages:
            mode = ExtractionMode.SCAN
        elif native_text_pages:
            mode = ExtractionMode.NATIVE
        elif empty_pages == page_count:
            mode = ExtractionMode.EMPTY
        else:
            mode = ExtractionMode.NATIVE

        return PdfPreflightResult(
            page_count=page_count,
            native_text_pages=native_text_pages,
            scan_like_pages=scan_like_pages,
            extraction_mode=mode,
        )
    finally:
        document.close()


def _enforce_ocr_pixel_budget(rect: Any, settings: CoreSettings) -> None:
    width_pixels = int((float(rect.width) / 72) * settings.ocr_render_dpi)
    height_pixels = int((float(rect.height) / 72) * settings.ocr_render_dpi)
    pixels = width_pixels * height_pixels
    if pixels > settings.max_ocr_rendered_pixels:
        raise IntakeError(
            "OCR page exceeds the configured pixel limit",
            code="ocr_image_too_large",
            details={
                "width": width_pixels,
                "height": height_pixels,
                "maxPixels": settings.max_ocr_rendered_pixels,
            },
        )


def _convert_doc_to_docx(path: Path, settings: CoreSettings) -> Path:
    started_at = time.perf_counter()
    binary = _resolve_executable(settings.libreoffice_binary)
    if binary is None:
        _log_doc_conversion(started_at, return_code=None, timed_out=False, failed=True)
        raise IntakeError(
            "DOC converter is unavailable",
            code="doc_converter_unavailable",
            details={"binary": settings.libreoffice_binary},
        )

    temp_root = Path(tempfile.mkdtemp(prefix="internum-doc-convert-"))
    input_path = temp_root / "input.doc"
    output_dir = temp_root / "out"
    profile_dir = temp_root / "profile"
    output_dir.mkdir()
    profile_dir.mkdir()
    input_path.write_bytes(path.read_bytes())

    command = [
        binary,
        "--headless",
        "--nologo",
        "--nolockcheck",
        "--nodefault",
        "--nofirststartwizard",
        f"-env:UserInstallation={profile_dir.resolve().as_uri()}",
        "--convert-to",
        "docx",
        "--outdir",
        str(output_dir),
        str(input_path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=settings.doc_conversion_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(temp_root, ignore_errors=True)
        _log_doc_conversion(started_at, return_code=None, timed_out=True, failed=True)
        raise IntakeError(
            "DOC conversion timed out",
            code="doc_conversion_timeout",
            details={"timeoutSeconds": settings.doc_conversion_timeout_seconds},
        ) from exc
    except FileNotFoundError as exc:
        shutil.rmtree(temp_root, ignore_errors=True)
        _log_doc_conversion(started_at, return_code=None, timed_out=False, failed=True)
        raise IntakeError("DOC converter is unavailable", code="doc_converter_unavailable") from exc
    except OSError as exc:
        shutil.rmtree(temp_root, ignore_errors=True)
        _log_doc_conversion(started_at, return_code=None, timed_out=False, failed=True)
        raise IntakeError("DOC converter is unavailable", code="doc_converter_unavailable") from exc

    converted_path = output_dir / "input.docx"
    if completed.returncode != 0 or not converted_path.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
        _log_doc_conversion(
            started_at,
            return_code=completed.returncode,
            timed_out=False,
            failed=True,
        )
        raise IntakeError(
            "DOC conversion failed",
            code="doc_conversion_failed",
            details={"returnCode": completed.returncode},
        )

    _log_doc_conversion(
        started_at,
        return_code=completed.returncode,
        timed_out=False,
        failed=False,
    )
    return converted_path


def _log_doc_conversion(
    started_at: float,
    *,
    return_code: int | None,
    timed_out: bool,
    failed: bool,
) -> None:
    log_event(
        "doc.convert",
        level=logging.WARNING if failed else logging.INFO,
        converter="libreoffice",
        via="libreoffice",
        durationMs=_duration_ms(started_at),
        returnCode=return_code,
        timedOut=timed_out,
    )


def _cleanup_conversion_path(path: Path) -> None:
    for parent in path.parents:
        if parent.name.startswith("internum-doc-convert-"):
            shutil.rmtree(parent, ignore_errors=True)
            return
    path.unlink(missing_ok=True)


def _resolve_executable(binary: str) -> str | None:
    candidate = Path(binary)
    if candidate.is_absolute() or "/" in binary:
        return (
            str(candidate)
            if candidate.exists() and candidate.is_file() and os.access(candidate, os.X_OK)
            else None
        )
    return shutil.which(binary)


def _is_runnable_libreoffice(binary: str) -> bool:
    try:
        completed = subprocess.run(
            [binary, "--headless", "--version"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _duration_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)
