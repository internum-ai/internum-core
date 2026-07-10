import asyncio
import json
from decimal import Decimal
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from api.capabilities.document_parsing.models import (
    ExtractedDocument,
    ExtractionMode,
    ParseMetadata,
    ParseMultipartRequest,
    SupportedDocumentType,
)
from api.capabilities.document_parsing.service import DocumentParsingService
from api.common.errors import SchemaError
from api.common.logging import configure_logging
from api.config.overrides import SafeRequestOverrides
from api.config.settings import CoreSettings
from api.main import create_app
from api.platform.openrouter import OpenRouterResult


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _schema() -> str:
    return json.dumps(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "missing": {"type": ["string", "null"]},
            },
            "required": ["name", "missing"],
            "additionalProperties": False,
        }
    )


def _totals_schema() -> str:
    return json.dumps(
        {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "total": {"type": "number"},
            },
            "required": ["a", "b", "total"],
            "additionalProperties": False,
        }
    )


class Extractor:
    async def extract(self, upload):  # type: ignore[no-untyped-def]
        return ExtractedDocument(
            markdown="Name: Ada",
            metadata=ParseMetadata(
                document_type=upload.document_type,
                extraction_mode=ExtractionMode.NATIVE
                if upload.document_type is SupportedDocumentType.PDF
                else None,
                page_count=1 if upload.document_type is SupportedDocumentType.PDF else None,
                ocr_page_count=0 if upload.document_type is SupportedDocumentType.PDF else None,
                converter="markitdown",
            ),
        )


class QueueingClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.requests = []

    async def complete(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        return OpenRouterResult(
            content=self.contents.pop(0),
            model=request.model,
            provider="openai",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            cost_usd=Decimal("0"),
        )


def test_parse_endpoint_returns_schema_validated_json_with_nulls(
    core_settings: CoreSettings,
) -> None:
    client = QueueingClient(['{"name":"Ada","missing":null}'])
    app = _app(core_settings, client)

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={
            "schema": _schema(),
            "additionalContext": "Use the document title.",
            "model": "anthropic/claude-sonnet-4.5",
            "systemPrompt": "Return JSON only.",
        },
        files={"file": ("sample.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "data": {"name": "Ada", "missing": None},
        "meta": {
            "documentType": "pdf",
            "extractionMode": "native",
            "pageCount": 1,
            "ocrPageCount": 0,
            "converter": "markitdown",
            "usage": {
                "model": "anthropic/claude-sonnet-4.5",
                "promptTokens": 1,
                "completionTokens": 1,
                "totalTokens": 2,
                "costUsd": "0",
            },
            "checks": [],
            "chunking": None,
        },
    }
    assert client.requests[0].model == "anthropic/claude-sonnet-4.5"
    assert client.requests[0].system_prompt == "Return JSON only."
    assert "Use the document title." in client.requests[0].user_content


def test_parse_endpoint_normalizes_hinted_field_in_response(
    core_settings: CoreSettings,
) -> None:
    client = QueueingClient(['{"issuedAt":"5.7.2026.","missing":null}'])
    app = _app(core_settings, client)
    schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "issuedAt": {"type": "string", "format": "date"},
                "missing": {"type": ["string", "null"]},
            },
            "required": ["issuedAt", "missing"],
            "additionalProperties": False,
        }
    )

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={"schema": schema},
        files={"file": ("sample.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["data"] == {"issuedAt": "2026-07-05", "missing": None}


def test_parse_endpoint_forwards_models_field_to_client_request(
    core_settings: CoreSettings,
) -> None:
    client = QueueingClient(['{"name":"Ada","missing":null}'])
    app = _app(core_settings, client)

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={
            "schema": _schema(),
            "models": json.dumps(["openai/gpt-5.2", "openai/gpt-5-mini"]),
        },
        files={"file": ("sample.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
    )

    assert response.status_code == 200
    assert client.requests[0].models == ["openai/gpt-5.2", "openai/gpt-5-mini"]
    assert client.requests[0].model == "openai/gpt-5.2"


def test_parse_endpoint_retries_once_after_schema_validation_failure(
    core_settings: CoreSettings,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="INFO")
    client = QueueingClient(['{"name":null,"missing":null}', '{"name":"Ada","missing":null}'])
    app = _app(core_settings, client)

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={"schema": _schema()},
        files={"file": ("sample.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "data": {"name": "Ada", "missing": None},
        "meta": {
            "documentType": "pdf",
            "extractionMode": "native",
            "pageCount": 1,
            "ocrPageCount": 0,
            "converter": "markitdown",
            "usage": {
                "model": "openai/gpt-5.2",
                "promptTokens": 1,
                "completionTokens": 1,
                "totalTokens": 2,
                "costUsd": "0",
            },
            "checks": [],
            "chunking": None,
        },
    }
    assert len(client.requests) == 2
    assert [request.attempt for request in client.requests] == [1, 2]
    assert client.requests[1].validation_retry_prompt is not None
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    validations = [event for event in events if event["event"] == "schema.validation"]
    assert [event["passed"] for event in validations] == [False, True]
    assert all(event["validationRetryTriggered"] is True for event in validations)
    retry = next(event for event in events if event["event"] == "model.retry")
    assert retry["reason"] == "schema_rejection"
    assert retry["attempt"] == 2


def test_parse_endpoint_logs_unsupported_intake_rejection_once(
    core_settings: CoreSettings,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="INFO")
    app = _app(core_settings, QueueingClient([]))

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key", "X-Request-ID": "request-rejected"},
        data={"schema": _schema()},
        files={"file": ("unknown.bin", b"not-supported", "application/octet-stream")},
    )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    rejected = [event for event in events if event["event"] == "intake.rejected"]
    assert response.status_code == 400
    assert len(rejected) == 1
    assert rejected[0]["code"] == "unsupported_file_type"
    assert rejected[0]["documentType"] == "unknown"
    assert rejected[0]["durationMs"] >= 0


def test_parse_endpoint_logs_oversized_intake_rejection_once(
    core_settings: CoreSettings,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="INFO")
    settings = core_settings.model_copy(update={"max_upload_bytes": 1})
    app = _app(settings, QueueingClient([]))

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key", "X-Request-ID": "request-oversized"},
        data={"schema": _schema()},
        files={"file": ("sample.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
    )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    rejected = [event for event in events if event["event"] == "intake.rejected"]
    assert response.status_code == 400
    assert len(rejected) == 1
    assert rejected[0]["code"] == "intake_error"
    assert rejected[0]["documentType"] == "unknown"
    assert rejected[0]["durationMs"] >= 0


def test_parse_endpoint_returns_common_error_after_retry_failure(
    core_settings: CoreSettings,
) -> None:
    client = QueueingClient(['{"name":null,"missing":null}', '{"name":null,"missing":null}'])
    app = _app(core_settings, client)

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={"schema": _schema()},
        files={"file": ("sample.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "schema_error"


def test_parse_endpoint_requires_auth(core_settings: CoreSettings) -> None:
    app = _app(core_settings, QueueingClient(['{"name":"Ada","missing":null}']))

    response = TestClient(app).post(
        "/v1/parse",
        data={"schema": _schema()},
        files={"file": ("sample.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "missing_api_key"


def test_parse_endpoint_logs_metadata_only_pipeline_events_at_info(
    core_settings: CoreSettings,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="INFO")
    app = _app(core_settings, QueueingClient(['{"name":"Ada","missing":null}']))

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key", "X-Request-ID": "request-logged"},
        data={"schema": _schema()},
        files={"file": ("sample.pdf", b"%PDF-1.4\nName: Ada", "application/pdf")},
    )

    output = capsys.readouterr().out
    events = [json.loads(line) for line in output.splitlines()]
    by_name = {event["event"]: event for event in events}
    assert response.status_code == 200
    assert by_name["request.received"]["consumerId"] == "internal"
    assert by_name["request.received"]["contentLength"] > 0
    assert by_name["intake.stored"]["documentType"] == "pdf"
    assert by_name["intake.stored"]["sizeBytes"] > 0
    assert by_name["schema.validation"] == {
        "durationMs": by_name["schema.validation"]["durationMs"],
        "event": "schema.validation",
        "passed": True,
        "repairApplied": False,
        "requestId": "request-logged",
        "validationRetryTriggered": False,
    }
    assert "Name: Ada" not in output
    assert '"name":"Ada"' not in output


def test_parse_endpoint_logs_repaired_extracted_values_only_at_debug(
    core_settings: CoreSettings,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="DEBUG")
    app = _app(core_settings, QueueingClient(["{'name':'Ada','missing':null}"]))

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key", "X-Request-ID": "request-debug"},
        data={"schema": _schema()},
        files={"file": ("sample.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
    )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    validation = next(event for event in events if event["event"] == "schema.validation")
    values = next(event for event in events if event["event"] == "schema.values")
    assert response.status_code == 200
    assert validation["repairApplied"] is True
    assert values["values"] == {"name": "Ada", "missing": None}


def test_parse_endpoint_reports_failed_post_check_in_meta_and_logs_it(
    core_settings: CoreSettings,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="INFO")
    client = QueueingClient(['{"a":1,"b":2,"total":5}'])
    app = _app(core_settings, client)

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={
            "schema": _totals_schema(),
            "checks": json.dumps(
                [
                    {
                        "op": "sum_equals",
                        "addends": ["/a", "/b"],
                        "total": "/total",
                        "tolerance": 0,
                    }
                ]
            ),
        },
        files={"file": ("sample.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
    )

    assert response.status_code == 200
    checks = response.json()["meta"]["checks"]
    assert len(checks) == 1
    assert checks[0]["op"] == "sum_equals"
    assert checks[0]["passed"] is False

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    postcheck_events = [event for event in events if event["event"] == "schema.postcheck"]
    assert len(postcheck_events) == 1
    assert postcheck_events[0]["passed"] is False


def test_parse_endpoint_returns_empty_checks_when_none_supplied(
    core_settings: CoreSettings,
) -> None:
    client = QueueingClient(['{"a":1,"b":2,"total":3}'])
    app = _app(core_settings, client)

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={"schema": _totals_schema()},
        files={"file": ("sample.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["meta"]["checks"] == []


def _app(core_settings: CoreSettings, client: QueueingClient):
    app = create_app(settings=core_settings)
    app.state.document_extractor = Extractor()
    app.state.openrouter_client = client
    return app


def _table_markdown(row_count: int) -> str:
    header = "| id | amount |"
    delimiter = "| --- | --- |"
    rows = [f"| {index} | {index * 10} |" for index in range(row_count)]
    return "\n".join([header, delimiter, *rows])


def _row_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "amount": {"type": "number"},
                    },
                    "required": ["id", "amount"],
                    "additionalProperties": False,
                },
            },
            "grandTotal": {"type": "number"},
        },
        "required": ["rows", "grandTotal"],
        "additionalProperties": False,
    }


class TableExtractor:
    def __init__(self, row_count: int) -> None:
        self._row_count = row_count

    async def extract(self, upload):  # type: ignore[no-untyped-def]
        return ExtractedDocument(
            markdown=_table_markdown(self._row_count),
            metadata=ParseMetadata(
                document_type=upload.document_type,
                extraction_mode=None,
                page_count=None,
                ocr_page_count=None,
                converter="markitdown",
            ),
        )


def _parse_pipe_row(line: str) -> tuple[int, int]:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return int(cells[0]), int(cells[1])


class DerivingChunkClient:
    """Fake OpenRouter completer that derives its response from each request's
    schema/markdown so it works regardless of concurrency ordering."""

    def __init__(self, *, fail_row_id: int | None = None) -> None:
        self.requests = []
        self._fail_row_id = fail_row_id
        self._fail_attempts: dict[int, int] = {}
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            properties = request.schema.get("properties", {})
            if "rows" in properties:
                lines = [line for line in request.user_content.splitlines() if line.startswith("|")]
                data_lines = lines[2:]
                rows = []
                for line in data_lines:
                    row_id, amount = _parse_pipe_row(line)
                    if self._fail_row_id is not None and row_id == self._fail_row_id:
                        attempts = self._fail_attempts.get(row_id, 0) + 1
                        self._fail_attempts[row_id] = attempts
                        raise SchemaError("simulated persistent chunk failure")
                    rows.append({"id": row_id, "amount": amount})
                content = json.dumps({"rows": rows})
            else:
                content = json.dumps({"grandTotal": 0})
            return OpenRouterResult(
                content=content,
                model=request.model,
                provider="openai",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                cost_usd=Decimal("0"),
            )
        finally:
            self.in_flight -= 1


def test_parse_endpoint_below_threshold_document_stays_single_pass(
    core_settings: CoreSettings,
) -> None:
    client = QueueingClient(['{"name":"Ada","missing":null}'])
    app = _app(core_settings, client)

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={"schema": _schema()},
        files={"file": ("sample.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["meta"]["chunking"] is None
    assert len(client.requests) == 1


def test_parse_endpoint_chunks_large_tabular_document(
    core_settings: CoreSettings,
) -> None:
    settings = core_settings.model_copy(
        update={"chunk_row_threshold": 10, "chunk_rows_per_chunk": 20}
    )
    client = DerivingChunkClient()
    app = create_app(settings=settings)
    app.state.document_extractor = TableExtractor(60)
    app.state.openrouter_client = client

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={"schema": json.dumps(_row_schema())},
        files={"file": ("sample.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    ids = [row["id"] for row in body["data"]["rows"]]
    assert ids == list(range(60))

    chunk_requests = [
        request for request in client.requests if "rows" in request.schema["properties"]
    ]
    for request in chunk_requests:
        assert set(request.schema["properties"]) == {"rows"}

    summary_requests = [
        request for request in client.requests if "grandTotal" in request.schema["properties"]
    ]
    assert len(summary_requests) == 1
    assert set(summary_requests[0].schema["properties"]) == {"grandTotal"}

    chunking = body["meta"]["chunking"]
    assert chunking["chunked"] is True
    assert chunking["totalRows"] == 60
    assert chunking["chunkCount"] == 3
    assert chunking["partial"] is False
    assert chunking["failedChunks"] == []


def test_parse_endpoint_runs_post_check_over_merged_chunk_rows(
    core_settings: CoreSettings,
) -> None:
    settings = core_settings.model_copy(
        update={"chunk_row_threshold": 10, "chunk_rows_per_chunk": 20}
    )
    client = DerivingChunkClient()
    app = create_app(settings=settings)
    app.state.document_extractor = TableExtractor(30)
    app.state.openrouter_client = client

    expected_total = sum(index * 10 for index in range(30))
    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={
            "schema": json.dumps(_row_schema()),
            "checks": json.dumps(
                [
                    {
                        "op": "sum_equals",
                        "addends": [f"/rows/{index}/amount" for index in range(30)],
                        "total": "/grandTotal",
                        "tolerance": 0,
                    }
                ]
            ),
        },
        files={"file": ("sample.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
    )

    assert response.status_code == 200
    checks = response.json()["meta"]["checks"]
    assert len(checks) == 1
    assert checks[0]["op"] == "sum_equals"
    # The fake summary pass always returns grandTotal=0, so the sum-check fails,
    # signalling the gap between the merged rows and the reported total.
    assert checks[0]["passed"] is (expected_total == 0)


def test_parse_endpoint_bounds_chunk_concurrency(
    core_settings: CoreSettings,
) -> None:
    settings = core_settings.model_copy(
        update={
            "chunk_row_threshold": 10,
            "chunk_rows_per_chunk": 5,
            "chunk_max_concurrency": 2,
        }
    )
    client = DerivingChunkClient()
    app = create_app(settings=settings)
    app.state.document_extractor = TableExtractor(50)
    app.state.openrouter_client = client

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={"schema": json.dumps(_row_schema())},
        files={"file": ("sample.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
    )

    assert response.status_code == 200
    assert client.max_in_flight <= 2


def test_parse_endpoint_reports_partial_chunking_on_persistent_chunk_failure(
    core_settings: CoreSettings,
) -> None:
    settings = core_settings.model_copy(
        update={"chunk_row_threshold": 10, "chunk_rows_per_chunk": 20}
    )
    client = DerivingChunkClient(fail_row_id=25)
    app = create_app(settings=settings)
    app.state.document_extractor = TableExtractor(60)
    app.state.openrouter_client = client

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={"schema": json.dumps(_row_schema())},
        files={"file": ("sample.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    chunking = body["meta"]["chunking"]
    assert chunking["partial"] is True
    assert chunking["failedChunks"] == [1]
    ids = [row["id"] for row in body["data"]["rows"]]
    assert 25 not in ids
    assert 0 in ids
    assert 59 in ids


def test_parse_endpoint_fails_whole_document_when_partial_not_allowed(
    core_settings: CoreSettings,
) -> None:
    settings = core_settings.model_copy(
        update={
            "chunk_row_threshold": 10,
            "chunk_rows_per_chunk": 20,
            "chunk_allow_partial": False,
        }
    )
    client = DerivingChunkClient(fail_row_id=25)
    app = create_app(settings=settings)
    app.state.document_extractor = TableExtractor(60)
    app.state.openrouter_client = client

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={"schema": json.dumps(_row_schema())},
        files={"file": ("sample.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "schema_error"
    assert response.json()["error"]["details"]["failedChunks"] == [1]


class BlockingClient:
    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.finished = False

    async def complete(self, request):  # type: ignore[no-untyped-def]
        await self.event.wait()
        self.finished = True
        return OpenRouterResult(
            content='{"name":"Ada","missing":null}',
            model=request.model,
            provider="openai",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            cost_usd=Decimal("0"),
        )


def _parse_request() -> ParseMultipartRequest:
    return ParseMultipartRequest(
        upload=UploadFile(
            file=BytesIO(b"%PDF-1.4\ncontent"),
            filename="sample.pdf",
        ),
        schema=json.loads(_schema()),
        additional_context=None,
        overrides=SafeRequestOverrides(),
    )


@pytest.mark.anyio
async def test_parse_cancels_in_flight_completion_on_client_disconnect(
    core_settings: CoreSettings,
) -> None:
    client = BlockingClient()
    service = DocumentParsingService(core_settings, Extractor(), client)

    async def is_disconnected() -> bool:
        return True

    with pytest.raises(asyncio.CancelledError):
        await service.parse(
            _parse_request(),
            consumer_id=None,
            request_id=None,
            is_disconnected=is_disconnected,
        )

    assert client.finished is False
