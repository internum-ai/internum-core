import logging
import re
from typing import Any

from api.common.logging import log_event

_CROATIAN_DATE_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})\.?$")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_values(data: Any, schema: dict[str, Any]) -> Any:
    """Apply opt-in value normalization to validated output, guided by schema hints.

    Only fields carrying a `"format": "date"` or `"x-normalize"` hint are touched.
    Any per-field failure leaves that field's original value intact.
    """
    if not isinstance(schema, dict):
        return data
    return _normalize_node(data, schema)


def _normalize_node(value: Any, schema: dict[str, Any]) -> Any:
    if not isinstance(schema, dict) or value is None:
        return value

    if isinstance(value, dict):
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return value
        result = dict(value)
        for key, subschema in properties.items():
            if key in result:
                result[key] = _normalize_node(result[key], subschema)
        return result

    if isinstance(value, list):
        items = schema.get("items")
        if not isinstance(items, dict):
            return value
        return [_normalize_node(item, items) for item in value]

    return _normalize_leaf(value, schema)


def _normalize_leaf(value: Any, schema: dict[str, Any]) -> Any:
    if not isinstance(value, str):
        return value

    try:
        result = value
        if schema.get("format") == "date":
            result = _normalize_date(result)

        hints = schema.get("x-normalize")
        if isinstance(hints, str):
            hints = [hints]
        if isinstance(hints, list):
            for hint in hints:
                result = _apply_hint(result, hint)
        return result
    except Exception:  # noqa: BLE001 - a single field's failure must never break output
        log_event("schema.normalize_failed", level=logging.DEBUG, schema=schema)
        return value


def _normalize_date(value: str) -> str:
    match = _CROATIAN_DATE_RE.match(value.strip())
    if not match:
        return value
    day, month, year = match.groups()
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _apply_hint(value: str, hint: Any) -> str:
    if hint == "digits":
        return re.sub(r"\D", "", value)
    if hint == "collapse-whitespace":
        return _WHITESPACE_RE.sub(" ", value).strip()
    return value
