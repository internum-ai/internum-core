from copy import deepcopy
from typing import Any

JsonSchema = dict[str, Any]


def normalize_for_model(schema: JsonSchema) -> JsonSchema:
    normalized = deepcopy(schema)
    _normalize_node(normalized, make_nullable=False)
    return normalized


def _normalize_node(node: Any, *, make_nullable: bool) -> None:
    if not isinstance(node, dict):
        return

    schema_type = node.get("type")
    if schema_type == "object" or "properties" in node:
        properties = node.get("properties")
        if isinstance(properties, dict):
            node["type"] = _with_null("object") if make_nullable else "object"
            node["additionalProperties"] = False
            node["required"] = sorted(properties.keys())
            for child in properties.values():
                _normalize_node(child, make_nullable=True)
        return

    if schema_type == "array":
        if make_nullable:
            node["type"] = _with_null("array")
        _normalize_node(node.get("items"), make_nullable=False)
        return

    for keyword in ("anyOf", "oneOf", "allOf"):
        options = node.get(keyword)
        if isinstance(options, list):
            for option in options:
                _normalize_node(option, make_nullable=False)

    if make_nullable and "type" in node:
        node["type"] = _with_null(node["type"])


def _with_null(schema_type: Any) -> Any:
    if isinstance(schema_type, list):
        return schema_type if "null" in schema_type else [*schema_type, "null"]
    if schema_type == "null":
        return schema_type
    return [schema_type, "null"]
