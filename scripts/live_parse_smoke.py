import io
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from api.main import create_app
from fastapi.testclient import TestClient


def main() -> int:
    if os.getenv("RUN_LIVE_OPENROUTER_SMOKE") != "true":
        print("Live OpenRouter smoke skipped: set RUN_LIVE_OPENROUTER_SMOKE=true to enable.")
        return 0

    api_key = os.getenv("CORE_LIVE_SMOKE_API_KEY")
    if not api_key:
        print("Live OpenRouter smoke skipped: CORE_LIVE_SMOKE_API_KEY is required.")
        return 1

    schema = {
        "type": "object",
        "properties": {
            "title": {"type": ["string", "null"]},
            "unresolvedField": {"type": ["string", "null"]},
        },
        "required": ["title", "unresolvedField"],
        "additionalProperties": False,
    }

    with tempfile.TemporaryDirectory(prefix="internum-smoke-") as temp_dir:
        samples = _create_samples(Path(temp_dir))
        with TestClient(create_app()) as client:
            for sample in samples:
                response = _post_sample(client, api_key, sample, schema)
                if response.status_code != 200:
                    print(f"Smoke failed for {sample.name}: {response.status_code} {response.text}")
                    return 1
                data = response.json()["data"]
                if "unresolvedField" not in data or data["unresolvedField"] is not None:
                    print(f"Smoke failed for {sample.name}: unresolvedField was not null: {data}")
                    return 1
                print(f"Smoke passed for {sample.name}: {json.dumps(data, sort_keys=True)}")

    return 0


def _post_sample(
    client: TestClient,
    api_key: str,
    sample: Path,
    schema: dict[str, Any],
):
    with sample.open("rb") as file:
        return client.post(
            "/v1/parse",
            headers={"X-API-Key": api_key},
            data={
                "schema": json.dumps(schema),
                "additionalContext": "The unresolvedField is intentionally absent.",
            },
            files={"file": (sample.name, file, "application/octet-stream")},
        )


def _create_samples(temp_dir: Path) -> Iterable[Path]:
    return [
        _create_pdf(temp_dir / "sample.pdf"),
        _create_docx(temp_dir / "sample.docx"),
        _create_xlsx(temp_dir / "sample.xlsx"),
        _create_png(temp_dir / "sample.png"),
    ]


def _create_pdf(path: Path) -> Path:
    import fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Title: Smoke PDF")
    document.save(path)
    document.close()
    return path


def _create_docx(path: Path) -> Path:
    from docx import Document

    document = Document()
    document.add_paragraph("Title: Smoke DOCX")
    document.save(path)
    return path


def _create_xlsx(path: Path) -> Path:
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet["A1"] = "Title"
    sheet["B1"] = "Smoke XLSX"
    workbook.save(path)
    return path


def _create_png(path: Path) -> Path:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (500, 140), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 50), "Title: Smoke PNG", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    path.write_bytes(buffer.getvalue())
    return path


if __name__ == "__main__":
    raise SystemExit(main())
