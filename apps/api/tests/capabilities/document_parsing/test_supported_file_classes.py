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
        ("sample.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1doc"),
        ("sample.html", b"<!doctype html><html><body>Title</body></html>"),
        ("sample.htm", b"<html><body>Title</body></html>"),
        ("sample.xlsx", lambda: _office_zip("xl/workbook.xml")),
        ("sample.xls", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1xls"),
        ("sample.png", lambda: _image_bytes("PNG")),
        ("sample.jpg", lambda: _image_bytes("JPEG")),
        ("sample.jpeg", lambda: _image_bytes("JPEG")),
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
    assert response.json()["data"] == {"title": filename, "unresolved": None}
    assert response.json()["meta"]["documentType"] in {
        "pdf",
        "docx",
        "doc",
        "html",
        "xlsx",
        "xls",
        "png",
        "jpg",
    }


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


def _image_bytes(format_name: str) -> bytes:
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (8, 8), color="white").save(buffer, format=format_name)
    return buffer.getvalue()
