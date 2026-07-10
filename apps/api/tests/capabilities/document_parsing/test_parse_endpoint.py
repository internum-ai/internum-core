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
        },
    }
    assert client.requests[0].model == "anthropic/claude-sonnet-4.5"
    assert client.requests[0].system_prompt == "Return JSON only."
    assert "Use the document title." in client.requests[0].user_content


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
