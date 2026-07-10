import pytest

from api.common.errors import SchemaError
from api.platform.schema import (
    evaluate_post_checks,
    format_validation_retry,
    normalize_for_model,
    normalize_values,
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


def test_normalize_values_converts_croatian_date_when_hinted() -> None:
    schema = {
        "type": "object",
        "properties": {"issuedAt": {"type": "string", "format": "date"}},
    }

    result = normalize_values({"issuedAt": "5.7.2026."}, schema)

    assert result == {"issuedAt": "2026-07-05"}


def test_normalize_values_leaves_non_matching_date_unchanged() -> None:
    schema = {
        "type": "object",
        "properties": {"issuedAt": {"type": "string", "format": "date"}},
    }

    result = normalize_values({"issuedAt": "not a date"}, schema)

    assert result == {"issuedAt": "not a date"}


def test_normalize_values_strips_non_digits_when_hinted() -> None:
    schema = {
        "type": "object",
        "properties": {"ean": {"type": "string", "x-normalize": "digits"}},
    }

    result = normalize_values({"ean": "978-953-358-763-9"}, schema)

    assert result == {"ean": "9789533587639"}


def test_normalize_values_collapses_whitespace_when_hinted() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string", "x-normalize": "collapse-whitespace"}},
    }

    result = normalize_values({"name": "  Ada   Lovelace \n"}, schema)

    assert result == {"name": "Ada Lovelace"}


def test_normalize_values_applies_list_of_hints_in_order() -> None:
    schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "x-normalize": ["collapse-whitespace", "digits"],
            }
        },
    }

    result = normalize_values({"code": " 978 953  358 "}, schema)

    assert result == {"code": "978953358"}


def test_normalize_values_recurses_into_nested_objects() -> None:
    schema = {
        "type": "object",
        "properties": {
            "supplier": {
                "type": "object",
                "properties": {"ean": {"type": "string", "x-normalize": "digits"}},
            }
        },
    }

    result = normalize_values({"supplier": {"ean": "978-1"}}, schema)

    assert result == {"supplier": {"ean": "9781"}}


def test_normalize_values_recurses_into_arrays_of_objects() -> None:
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"date": {"type": ["string", "null"], "format": "date"}},
                },
            }
        },
    }

    result = normalize_values({"rows": [{"date": "1.1.2026"}, {"date": "2.2.2026."}]}, schema)

    assert result == {"rows": [{"date": "2026-01-01"}, {"date": "2026-02-02"}]}


def test_normalize_values_passes_through_null_values() -> None:
    schema = {
        "type": "object",
        "properties": {"issuedAt": {"type": ["string", "null"], "format": "date"}},
    }

    result = normalize_values({"issuedAt": None}, schema)

    assert result == {"issuedAt": None}


def test_normalize_values_passes_through_fields_without_hints() -> None:
    schema = {
        "type": "object",
        "properties": {"note": {"type": "string"}},
    }

    result = normalize_values({"note": "  1.1.2026.  "}, schema)

    assert result == {"note": "  1.1.2026.  "}
