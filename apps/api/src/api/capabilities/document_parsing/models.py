from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from starlette.datastructures import UploadFile

from api.config.overrides import SafeRequestOverrides


class SupportedDocumentType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    JPEG = "jpg"
    PNG = "png"

    @property
    def extension(self) -> str:
        return f".{self.value}"


@dataclass(frozen=True)
class ParseMultipartRequest:
    upload: UploadFile
    schema: dict[str, Any]
    additional_context: str | None
    overrides: SafeRequestOverrides


@dataclass(frozen=True)
class StoredUpload:
    path: Path
    document_type: SupportedDocumentType
    size_bytes: int
    original_filename: str | None
    detected_mime: str | None
