from typing import Any

from pydantic import BaseModel, Field

from api.capabilities.document_parsing.models import ExtractionMode, SupportedDocumentType


class UsageSummarySchema(BaseModel):
    model: str = Field(description="Name of the model used to extract structured data.")
    prompt_tokens: int = Field(
        alias="promptTokens", description="Number of prompt tokens consumed."
    )
    completion_tokens: int = Field(
        alias="completionTokens", description="Number of completion tokens consumed."
    )
    total_tokens: int = Field(
        alias="totalTokens", description="Total tokens consumed for the request."
    )
    cost_usd: str | None = Field(
        default=None,
        alias="costUsd",
        description="Estimated cost in USD as a decimal string, if available.",
    )


class PostCheckResultSchema(BaseModel):
    op: str = Field(description="Identifier of the post-check operation that was evaluated.")
    passed: bool = Field(description="Whether the check passed.")
    detail: dict[str, Any] = Field(description="Structured detail describing the check outcome.")


class ChunkingSummarySchema(BaseModel):
    chunked: bool = Field(description="Whether the document's row array was split into chunks.")
    total_rows: int = Field(
        alias="totalRows", description="Total number of rows extracted across all chunks."
    )
    chunk_count: int = Field(
        alias="chunkCount", description="Number of chunks the document was split into."
    )
    failed_chunks: list[int] = Field(
        alias="failedChunks", description="Indexes of chunks that failed to extract, if any."
    )
    partial: bool = Field(description="Whether the result is partial due to a failed chunk.")
    model: str = Field(description="Name of the model used to extract chunked rows.")


class ParseMetaSchema(BaseModel):
    document_type: SupportedDocumentType = Field(
        alias="documentType", description="Detected document type of the uploaded file."
    )
    extraction_mode: ExtractionMode | None = Field(
        default=None,
        alias="extractionMode",
        description="How text was extracted (native, scan, mixed, or empty).",
    )
    page_count: int | None = Field(
        default=None, alias="pageCount", description="Total number of pages, if applicable."
    )
    ocr_page_count: int | None = Field(
        default=None,
        alias="ocrPageCount",
        description="Number of pages that required OCR, if applicable.",
    )
    converter: str | None = Field(
        default=None, description="Name of the converter used to extract the document's content."
    )
    usage: UsageSummarySchema | None = Field(
        default=None, description="Token usage and cost summary, if applicable."
    )
    checks: list[PostCheckResultSchema] = Field(
        default_factory=list,
        description="Results of post-extraction checks, if any were requested.",
    )
    chunking: ChunkingSummarySchema | None = Field(
        default=None, description="Row chunking summary, if the document was chunked."
    )


class ParseResponseSchema(BaseModel):
    data: dict[str, Any] = Field(
        description="Extracted document data, shaped according to the provided schema."
    )
    meta: ParseMetaSchema = Field(description="Metadata describing how the document was parsed.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "data": {"title": "Invoice #1234", "total": 199.99},
                "meta": {
                    "documentType": "pdf",
                    "extractionMode": "native",
                    "pageCount": 3,
                    "ocrPageCount": 0,
                    "converter": "pymupdf",
                    "usage": {
                        "model": "gpt-4.1",
                        "promptTokens": 1200,
                        "completionTokens": 340,
                        "totalTokens": 1540,
                        "costUsd": "0.0123",
                    },
                    "checks": [{"op": "non_empty", "passed": True, "detail": {}}],
                    "chunking": None,
                },
            }
        }
    }
