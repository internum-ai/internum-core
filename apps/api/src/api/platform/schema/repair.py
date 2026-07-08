from typing import Any

from json_repair import repair_json

from api.common.errors import SchemaError


def repair_json_output(raw: str) -> Any:
    try:
        return repair_json(raw, return_objects=True)
    except Exception as exc:  # pragma: no cover - json-repair has broad exception behavior.
        raise SchemaError("Model output could not be parsed as JSON") from exc
