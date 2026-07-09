import json
import tempfile
import zipfile
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError
from starlette.datastructures import FormData, Headers, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import Request

from api.capabilities.document_parsing.models import (
    ParseMultipartRequest,
    StoredUpload,
    SupportedDocumentType,
)
from api.common.errors import IntakeError
from api.config.overrides import SafeRequestOverrides

try:
    import magic
except ImportError:  # pragma: no cover - dependency is declared, fallback is defensive.
    magic = None  # type: ignore[assignment]


CHUNK_SIZE = 1024 * 1024
HEADER_BYTES = 4096
FORM_FIELD_LIMIT_BYTES = 64 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024
OLE_CFB_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


async def parse_multipart_request(
    request: Request,
    *,
    max_upload_bytes: int,
) -> ParseMultipartRequest:
    _reject_oversized_body(request, max_upload_bytes=max_upload_bytes)
    form: FormData | None = None
    try:
        form = await _parse_limited_multipart_form(
            request,
            max_upload_bytes=max_upload_bytes,
        )
        upload = _extract_single_file(form)
        schema = _extract_schema(form)
        overrides = _extract_overrides(form)
        additional_context = _optional_text(form, "additionalContext")
        return ParseMultipartRequest(
            upload=upload,
            schema=schema,
            additional_context=additional_context,
            overrides=overrides,
        )
    except MultiPartException as exc:
        if form is not None:
            await form.close()
        raise IntakeError(_multipart_error_message(exc.message)) from exc
    except Exception:
        if form is not None:
            await form.close()
        raise


@asynccontextmanager
async def stored_upload(
    upload: UploadFile,
    *,
    max_upload_bytes: int,
    max_image_pixels: int,
    max_ooxml_zip_entries: int,
    max_ooxml_uncompressed_bytes: int,
    max_ooxml_compression_ratio: float,
) -> AsyncIterator[StoredUpload]:
    temp_path: Path | None = None
    try:
        temp_path, size_bytes = await _stream_to_temp(upload, max_upload_bytes=max_upload_bytes)
        document_type, detected_mime = detect_document_type(
            temp_path,
            max_ooxml_zip_entries=max_ooxml_zip_entries,
            max_ooxml_uncompressed_bytes=max_ooxml_uncompressed_bytes,
            max_ooxml_compression_ratio=max_ooxml_compression_ratio,
        )
        if document_type in {SupportedDocumentType.JPEG, SupportedDocumentType.PNG}:
            _validate_image_dimensions(temp_path, max_image_pixels=max_image_pixels)
        yield StoredUpload(
            path=temp_path,
            document_type=document_type,
            size_bytes=size_bytes,
            original_filename=upload.filename,
            detected_mime=detected_mime,
        )
    finally:
        await upload.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def detect_document_type(
    path: Path,
    *,
    max_ooxml_zip_entries: int = 1_000,
    max_ooxml_uncompressed_bytes: int = 100 * 1024 * 1024,
    max_ooxml_compression_ratio: float = 100.0,
) -> tuple[SupportedDocumentType, str | None]:
    header = path.read_bytes()[:HEADER_BYTES]
    detected_mime = _detect_mime(header)
    suffix = path.suffix.lower()

    if header.startswith(b"%PDF-"):
        return SupportedDocumentType.PDF, detected_mime
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return SupportedDocumentType.PNG, detected_mime
    if header.startswith(b"\xff\xd8\xff"):
        return SupportedDocumentType.JPEG, detected_mime
    if zipfile.is_zipfile(path):
        return (
            _detect_office_zip(
                path,
                max_entries=max_ooxml_zip_entries,
                max_uncompressed_bytes=max_ooxml_uncompressed_bytes,
                max_compression_ratio=max_ooxml_compression_ratio,
            ),
            detected_mime,
        )
    if _looks_like_html(header) and suffix in {".html", ".htm"}:
        return SupportedDocumentType.HTML, detected_mime
    if header.startswith(OLE_CFB_SIGNATURE):
        return _detect_legacy_office_type(suffix, detected_mime), detected_mime

    raise IntakeError("Unsupported file type", code="unsupported_file_type")


async def _stream_to_temp(upload: UploadFile, *, max_upload_bytes: int) -> tuple[Path, int]:
    suffix = _safe_suffix(upload.filename)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="internum-upload-", suffix=suffix, delete=False
        ) as temp:
            temp_path = Path(temp.name)
            size_bytes = 0
            while True:
                chunk = await upload.read(CHUNK_SIZE)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > max_upload_bytes:
                    raise IntakeError(
                        "Uploaded file exceeds the configured size limit",
                        details={"maxUploadBytes": max_upload_bytes},
                    )
                temp.write(chunk)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise

    if size_bytes == 0:
        temp_path.unlink(missing_ok=True)
        raise IntakeError("Uploaded file is empty")
    return temp_path, size_bytes


def _extract_single_file(form: FormData) -> UploadFile:
    files = [
        value
        for key, value in form.multi_items()
        if key == "file" and isinstance(value, UploadFile)
    ]
    if len(files) != 1:
        raise IntakeError("Exactly one file must be provided")
    return files[0]


def _reject_oversized_body(request: Request, *, max_upload_bytes: int) -> None:
    content_length = request.headers.get("content-length")
    if content_length is None:
        return
    try:
        body_bytes = int(content_length)
    except ValueError as exc:
        raise IntakeError("Invalid Content-Length header") from exc
    max_body_bytes = _max_body_bytes(max_upload_bytes)
    if body_bytes > max_body_bytes:
        raise IntakeError(
            "Uploaded file exceeds the configured size limit",
            details={"maxUploadBytes": max_upload_bytes},
        )


async def _parse_limited_multipart_form(
    request: Request,
    *,
    max_upload_bytes: int,
) -> FormData:
    parser = MultiPartParser(
        Headers(raw=request.headers.raw),
        _limited_body_stream(request, max_body_bytes=_max_body_bytes(max_upload_bytes)),
        max_files=1,
        max_fields=4,
        max_part_size=FORM_FIELD_LIMIT_BYTES,
    )
    return await parser.parse()


async def _limited_body_stream(
    request: Request,
    *,
    max_body_bytes: int,
) -> AsyncGenerator[bytes, None]:
    total_bytes = 0
    async for chunk in request.stream():
        total_bytes += len(chunk)
        if total_bytes > max_body_bytes:
            raise MultiPartException("Uploaded file exceeds the configured size limit")
        yield chunk


def _max_body_bytes(max_upload_bytes: int) -> int:
    return max_upload_bytes + MULTIPART_OVERHEAD_BYTES


def _multipart_error_message(detail: str) -> str:
    if detail.startswith("Too many files."):
        return "Exactly one file must be provided"
    return detail


def _extract_schema(form: FormData) -> dict[str, Any]:
    raw_schema = _required_text(form, "schema")
    try:
        schema = json.loads(raw_schema)
    except json.JSONDecodeError as exc:
        raise IntakeError("Schema field must contain valid JSON") from exc

    if not isinstance(schema, dict):
        raise IntakeError("Schema field must contain a JSON object")
    return schema


def _extract_overrides(form: FormData) -> SafeRequestOverrides:
    try:
        return SafeRequestOverrides.model_validate(
            {
                key: value
                for key in ("model", "systemPrompt")
                if (value := _optional_text(form, key)) is not None
            }
        )
    except ValidationError as exc:
        raise IntakeError("Request overrides are invalid") from exc


def _required_text(form: FormData, key: str) -> str:
    value = _optional_text(form, key)
    if value is None:
        raise IntakeError(f"Missing required form field: {key}")
    return value


def _optional_text(form: FormData, key: str) -> str | None:
    value = form.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise IntakeError(f"Form field must be text: {key}")
    stripped = value.strip()
    return stripped or None


def _safe_suffix(filename: str | None) -> str:
    if not filename:
        return ""
    suffix = Path(filename).suffix.lower()
    return (
        suffix
        if suffix
        in {
            ".pdf",
            ".docx",
            ".doc",
            ".html",
            ".htm",
            ".xlsx",
            ".xls",
            ".jpg",
            ".jpeg",
            ".png",
        }
        else ""
    )


def _detect_mime(header: bytes) -> str | None:
    if magic is None:
        return None
    try:
        detected = magic.from_buffer(header, mime=True)
    except Exception:
        return None
    return detected if isinstance(detected, str) else None


def _detect_office_zip(
    path: Path,
    *,
    max_entries: int,
    max_uncompressed_bytes: int,
    max_compression_ratio: float,
) -> SupportedDocumentType:
    try:
        with zipfile.ZipFile(path) as archive:
            _validate_ooxml_archive(
                archive,
                max_entries=max_entries,
                max_uncompressed_bytes=max_uncompressed_bytes,
                max_compression_ratio=max_compression_ratio,
            )
            names = set(archive.namelist())
    except zipfile.BadZipFile as exc:
        raise IntakeError("Unsupported file type", code="unsupported_file_type") from exc

    if "[Content_Types].xml" not in names:
        raise IntakeError("Unsupported file type", code="unsupported_file_type")
    if any(name.startswith("word/") for name in names):
        return SupportedDocumentType.DOCX
    if any(name.startswith("xl/") for name in names):
        return SupportedDocumentType.XLSX
    raise IntakeError("Unsupported file type", code="unsupported_file_type")


def _looks_like_html(header: bytes) -> bool:
    sample = header.lstrip().lower()
    return (
        sample.startswith((b"<!doctype html", b"<html", b"<head", b"<body")) or b"<html" in sample
    )


def _detect_legacy_office_type(suffix: str, detected_mime: str | None) -> SupportedDocumentType:
    mime = (detected_mime or "").lower()
    if suffix == ".doc" or mime in {"application/msword"}:
        return SupportedDocumentType.DOC
    if suffix == ".xls" or mime == "application/vnd.ms-excel":
        return SupportedDocumentType.XLS
    raise IntakeError("Unsupported file type", code="unsupported_file_type")


def _validate_ooxml_archive(
    archive: zipfile.ZipFile,
    *,
    max_entries: int,
    max_uncompressed_bytes: int,
    max_compression_ratio: float,
) -> None:
    entries = archive.infolist()
    if len(entries) > max_entries:
        raise IntakeError(
            "OOXML archive contains too many entries",
            code="unsafe_archive",
            details={"maxEntries": max_entries},
        )

    total_uncompressed = 0
    for entry in entries:
        if _is_unsafe_zip_path(entry.filename):
            raise IntakeError(
                "OOXML archive contains an unsafe path",
                code="unsafe_archive",
            )
        total_uncompressed += entry.file_size
        if total_uncompressed > max_uncompressed_bytes:
            raise IntakeError(
                "OOXML archive exceeds the configured uncompressed size limit",
                code="unsafe_archive",
                details={"maxUncompressedBytes": max_uncompressed_bytes},
            )
        if entry.compress_size == 0:
            if entry.file_size > 0:
                raise IntakeError(
                    "OOXML archive has an unsafe compression ratio",
                    code="unsafe_archive",
                )
            continue
        ratio = entry.file_size / entry.compress_size
        if ratio > max_compression_ratio:
            raise IntakeError(
                "OOXML archive has an unsafe compression ratio",
                code="unsafe_archive",
                details={"maxCompressionRatio": max_ooxml_ratio_detail(max_compression_ratio)},
            )


def _is_unsafe_zip_path(name: str) -> bool:
    if not name or "\\" in name or name.startswith("/"):
        return True
    path = PurePosixPath(name)
    return path.is_absolute() or ".." in path.parts


def max_ooxml_ratio_detail(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _validate_image_dimensions(path: Path, *, max_image_pixels: int) -> None:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - dependency is declared.
        return

    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:
        raise IntakeError("Unsupported file type", code="unsupported_file_type") from exc

    pixels = width * height
    if pixels > max_image_pixels:
        raise IntakeError(
            "Image exceeds the configured pixel limit",
            code="image_too_large",
            details={
                "width": width,
                "height": height,
                "maxPixels": max_image_pixels,
            },
        )
