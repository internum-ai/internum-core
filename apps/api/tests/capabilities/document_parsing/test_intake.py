import json
import zipfile
from io import BytesIO

from fastapi.testclient import TestClient

from api.config.settings import CoreSettings
from api.main import create_app


def _schema() -> str:
    return json.dumps(
        {
            "type": "object",
            "properties": {"name": {"type": ["string", "null"]}},
            "required": ["name"],
            "additionalProperties": False,
        }
    )


def _docx_bytes() -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", "<document />")
    return buffer.getvalue()


def test_valid_docx_intake_reaches_parser_and_cleans_temp_file(
    core_settings: CoreSettings,
) -> None:
    app = create_app(settings=core_settings)
    captured_paths: list[str] = []

    class Extractor:
        async def extract(self, upload):  # type: ignore[no-untyped-def]
            captured_paths.append(str(upload.path))
            assert upload.path.exists()
            assert upload.document_type.value == "docx"
            return "Name: Ada"

    class Client:
        async def complete(self, request):  # type: ignore[no-untyped-def]
            return _result('{"name":"Ada"}')

    app.state.document_extractor = Extractor()
    app.state.openrouter_client = Client()

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={"schema": _schema()},
        files={"file": ("sample.docx", _docx_bytes(), "application/octet-stream")},
    )

    assert response.status_code == 200
    assert response.json() == {"data": {"name": "Ada"}}
    assert captured_paths
    assert not any_path_exists(captured_paths)


def test_unsupported_file_type_is_rejected(core_settings: CoreSettings) -> None:
    app = create_app(settings=core_settings)

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={"schema": _schema()},
        files={"file": ("sample.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "intake_error"


def test_oversized_file_is_rejected(core_settings: CoreSettings) -> None:
    settings = core_settings.model_copy(update={"max_upload_bytes": 4})
    app = create_app(settings=settings)

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={"schema": _schema()},
        files={"file": ("sample.pdf", b"%PDF-1.4\nlarge", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "intake_error"


def test_multiple_files_are_rejected(core_settings: CoreSettings) -> None:
    app = create_app(settings=core_settings)

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={"schema": _schema()},
        files=[
            ("file", ("one.pdf", b"%PDF-1.4\none", "application/pdf")),
            ("file", ("two.pdf", b"%PDF-1.4\ntwo", "application/pdf")),
        ],
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Exactly one file must be provided"


def any_path_exists(paths: list[str]) -> bool:
    from pathlib import Path

    return any(Path(path).exists() for path in paths)


def _result(content: str):
    from decimal import Decimal

    from api.platform.openrouter import OpenRouterResult

    return OpenRouterResult(
        content=content,
        model="openai/gpt-5.2",
        provider="openai",
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        cost_usd=Decimal("0"),
    )
