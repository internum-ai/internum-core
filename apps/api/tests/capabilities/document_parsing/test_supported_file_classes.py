import json
import zipfile
from decimal import Decimal
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from api.config.settings import CoreSettings
from api.main import create_app
from api.platform.openrouter import OpenRouterResult


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("sample.pdf", b"%PDF-1.4\nTitle: PDF"),
        ("sample.docx", lambda: _office_zip("word/document.xml")),
        ("sample.xlsx", lambda: _office_zip("xl/workbook.xml")),
        ("sample.png", b"\x89PNG\r\n\x1a\nimage"),
        ("sample.jpg", b"\xff\xd8\xff\xe0image"),
    ],
)
def test_parse_endpoint_accepts_supported_file_classes_and_preserves_nulls(
    core_settings: CoreSettings,
    filename: str,
    content,
) -> None:
    app = create_app(settings=core_settings)
    app.state.document_extractor = Extractor()
    app.state.openrouter_client = Client()
    payload = content() if callable(content) else content

    response = TestClient(app).post(
        "/v1/parse",
        headers={"X-API-Key": "valid-key"},
        data={"schema": _schema()},
        files={"file": (filename, payload, "application/octet-stream")},
    )

    assert response.status_code == 200
    assert response.json() == {"data": {"title": filename, "unresolved": None}}


class Extractor:
    async def extract(self, upload):  # type: ignore[no-untyped-def]
        return f"Title: {upload.original_filename}"


class Client:
    async def complete(self, request):  # type: ignore[no-untyped-def]
        title = request.user_content.split("Title: ", maxsplit=1)[1].split("\n", maxsplit=1)[0]
        return OpenRouterResult(
            content=json.dumps({"title": title, "unresolved": None}),
            model=request.model,
            provider="openai",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            cost_usd=Decimal("0"),
        )


def _schema() -> str:
    return json.dumps(
        {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "unresolved": {"type": ["string", "null"]},
            },
            "required": ["title", "unresolved"],
            "additionalProperties": False,
        }
    )


def _office_zip(member: str) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr(member, "<document />")
    return buffer.getvalue()
