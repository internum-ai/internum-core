import pytest

from api.common.errors import SchemaError
from api.platform.schema import (
    evaluate_post_checks,
    format_validation_retry,
    normalize_for_model,
    repair_json_output,
    validate_against_original,
)


def test_normalize_for_model_copies_closes_requires_and_nulls_nested_schema() -> None:
    original = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "address": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
            "tags": {"type": "array", "items": {"type": "string"}},
        },
    }

    normalized = normalize_for_model(original)

    assert original["properties"]["name"]["type"] == "string"
    assert normalized["additionalProperties"] is False
    assert normalized["required"] == ["address", "name", "tags"]
    assert normalized["properties"]["name"]["type"] == ["string", "null"]
    assert normalized["properties"]["address"]["type"] == ["object", "null"]
    assert normalized["properties"]["address"]["additionalProperties"] is False
    assert normalized["properties"]["address"]["required"] == ["city"]
    assert normalized["properties"]["tags"]["type"] == ["array", "null"]


def test_repair_json_output_repairs_trivial_syntax() -> None:
    assert repair_json_output('{"name": "Ada",}') == {"name": "Ada"}


def test_validate_against_original_accepts_valid_output() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }

    output = validate_against_original({"name": "Ada"}, schema)

    assert output == {"name": "Ada"}


def test_validate_against_original_rejects_invalid_output() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }

    with pytest.raises(SchemaError) as error:
        validate_against_original({"name": None}, schema)

    assert error.value.code == "schema_error"
    assert error.value.details is not None


def test_validate_against_original_rejects_invalid_schema() -> None:
    with pytest.raises(SchemaError) as error:
        validate_against_original({"name": "Ada"}, {"type": "not-a-json-schema-type"})

    assert error.value.message == "Caller schema is not a valid JSON Schema"


def test_format_validation_retry_includes_path_and_message() -> None:
    error = SchemaError(
        "Model output did not match the requested schema",
        details={"path": ["name"], "message": "None is not of type 'string'"},
    )

    retry = format_validation_retry(error)

    assert "name" in retry
    assert "None is not of type" in retry


def test_evaluate_post_checks_returns_empty_list_for_no_checks() -> None:
    assert evaluate_post_checks({"total": 10}, []) == []


def test_evaluate_post_checks_passes_when_sum_matches_total() -> None:
    data = {"lineItems": {"a": 4, "b": 6}, "total": 10}
    checks = [
        {
            "op": "sum_equals",
            "addends": ["/lineItems/a", "/lineItems/b"],
            "total": "/total",
            "tolerance": 0,
        }
    ]

    results = evaluate_post_checks(data, checks)

    assert len(results) == 1
    assert results[0].op == "sum_equals"
    assert results[0].passed is True


def test_evaluate_post_checks_fails_when_sum_exceeds_tolerance() -> None:
    data = {"lineItems": {"a": 4, "b": 6}, "total": 11}
    checks = [
        {
            "op": "sum_equals",
            "addends": ["/lineItems/a", "/lineItems/b"],
            "total": "/total",
            "tolerance": 0.01,
        }
    ]

    results = evaluate_post_checks(data, checks)

    assert len(results) == 1
    assert results[0].passed is False


def test_evaluate_post_checks_missing_pointer_fails_with_reason() -> None:
    data = {"total": 10}
    checks = [
        {
            "op": "sum_equals",
            "addends": ["/lineItems/a", "/lineItems/b"],
            "total": "/total",
            "tolerance": 0,
        }
    ]

    results = evaluate_post_checks(data, checks)

    assert len(results) == 1
    assert results[0].passed is False
    assert "reason" in results[0].detail


def test_evaluate_post_checks_fans_out_over_scope_array() -> None:
    data = {
        "invoices": [
            {"a": 1, "b": 2, "total": 3},
            {"a": 1, "b": 2, "total": 5},
        ]
    }
    checks = [
        {
            "op": "sum_equals",
            "addends": ["/a", "/b"],
            "total": "/total",
            "tolerance": 0,
            "scope": "/invoices",
        }
    ]

    results = evaluate_post_checks(data, checks)

    assert len(results) == 2
    assert results[0].passed is True
    assert results[1].passed is False
    assert results[1].detail.get("index") == 1
