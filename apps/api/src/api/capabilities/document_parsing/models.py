from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from starlette.datastructures import UploadFile

from api.config.overrides import SafeRequestOverrides


class SupportedDocumentType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    HTML = "html"
    XLSX = "xlsx"
    XLS = "xls"
    JPEG = "jpg"
    PNG = "png"

    @property
    def extension(self) -> str:
        return f".{self.value}"


class ExtractionMode(StrEnum):
    NATIVE = "native"
    SCAN = "scan"
    MIXED = "mixed"
    EMPTY = "empty"


@dataclass(frozen=True)
class ParseMultipartRequest:
    upload: UploadFile
    schema: dict[str, Any]
    additional_context: str | None
    overrides: SafeRequestOverrides
    checks: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class StoredUpload:
    path: Path
    document_type: SupportedDocumentType
    size_bytes: int
    original_filename: str | None
    detected_mime: str | None


@dataclass(frozen=True)
class PdfPreflightResult:
    page_count: int
    native_text_pages: int
    scan_like_pages: int
    extraction_mode: ExtractionMode
    native_text: str | None = None


@dataclass(frozen=True)
class UsageSummary:
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: Decimal | None

    def to_api(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "promptTokens": self.prompt_tokens,
            "completionTokens": self.completion_tokens,
            "totalTokens": self.total_tokens,
            "costUsd": str(self.cost_usd) if self.cost_usd is not None else None,
        }


@dataclass(frozen=True)
class PostCheckResult:
    op: str
    passed: bool
    detail: dict[str, Any]

    def to_api(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ParseMetadata:
    document_type: SupportedDocumentType
    extraction_mode: ExtractionMode | None
    page_count: int | None
    ocr_page_count: int | None
    converter: str | None
    usage: UsageSummary | None = None
    checks: list[PostCheckResult] = field(default_factory=list)

    def to_api(self) -> dict[str, Any]:
        return {
            "documentType": self.document_type.value,
            "extractionMode": self.extraction_mode.value if self.extraction_mode else None,
            "pageCount": self.page_count,
            "ocrPageCount": self.ocr_page_count,
            "converter": self.converter,
            "usage": self.usage.to_api() if self.usage else None,
            "checks": [check.to_api() for check in self.checks],
        }


@dataclass(frozen=True)
class ExtractedDocument:
    markdown: str
    metadata: ParseMetadata


@dataclass(frozen=True)
class ParsedDocument:
    data: dict[str, Any]
    metadata: ParseMetadata
