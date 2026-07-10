from typing import Any, NamedTuple

from jsonschema import Draft202012Validator, ValidationError
from jsonschema.exceptions import SchemaError as JsonSchemaDefinitionError

from api.common.errors import SchemaError


class PostCheckOutcome(NamedTuple):
    op: str
    passed: bool
    detail: dict[str, Any]


def validate_against_original(output: Any, schema: dict[str, Any]) -> Any:
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(output)
    except JsonSchemaDefinitionError as exc:
        raise SchemaError(
            "Caller schema is not a valid JSON Schema",
            details={"message": exc.message},
        ) from exc
    except ValidationError as exc:
        raise SchemaError(
            "Model output did not match the requested schema",
            details={"path": list(exc.path), "message": exc.message},
        ) from exc
    return output


def evaluate_post_checks(data: Any, checks: list[dict[str, Any]]) -> list[PostCheckOutcome]:
    results: list[PostCheckOutcome] = []
    for check in checks:
        results.extend(PostCheckOutcome(*outcome) for outcome in _evaluate_check(data, check))
    return results


def _evaluate_check(data: Any, check: dict[str, Any]) -> list[tuple[str, bool, dict[str, Any]]]:
    if not isinstance(check, dict):
        return [("unknown", False, {"reason": "check must be an object"})]

    op = check.get("op")
    if op != "sum_equals":
        unknown_op = str(op) if op is not None else "unknown"
        return [(unknown_op, False, {"reason": f"unsupported op: {op!r}"})]

    scope = check.get("scope")
    if scope is None:
        return [_evaluate_sum_equals(data, check)]

    try:
        elements = _resolve_pointer(data, scope)
    except _PointerError as exc:
        return [(op, False, {"reason": str(exc)})]

    if not isinstance(elements, list):
        return [(op, False, {"reason": "scope must resolve to an array"})]

    return [
        _evaluate_sum_equals(element, check, index=index) for index, element in enumerate(elements)
    ]


def _evaluate_sum_equals(
    data: Any,
    check: dict[str, Any],
    *,
    index: int | None = None,
) -> tuple[str, bool, dict[str, Any]]:
    op = "sum_equals"
    detail: dict[str, Any] = {} if index is None else {"index": index}
    try:
        addends = check.get("addends")
        total_pointer = check.get("total")
        tolerance = check.get("tolerance", 0)
        if not isinstance(addends, list) or not addends:
            raise _PointerError("addends must be a non-empty list of pointers")
        if not isinstance(total_pointer, str):
            raise _PointerError("total must be a JSON pointer string")
        if not isinstance(tolerance, int | float):
            raise _PointerError("tolerance must be a number")

        addend_values = [_numeric(_resolve_pointer(data, pointer)) for pointer in addends]
        total_value = _numeric(_resolve_pointer(data, total_pointer))
        actual_sum = sum(addend_values)
        passed = abs(actual_sum - total_value) <= tolerance
        detail = {
            **detail,
            "addendValues": addend_values,
            "sum": actual_sum,
            "total": total_value,
            "tolerance": tolerance,
        }
        return (op, passed, detail)
    except _PointerError as exc:
        return (op, False, {**detail, "reason": str(exc)})


class _PointerError(Exception):
    pass


def _numeric(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _PointerError(f"expected a numeric value, got {value!r}")
    return float(value)


def _resolve_pointer(data: Any, pointer: str) -> Any:
    if not isinstance(pointer, str):
        raise _PointerError(f"pointer must be a string, got {pointer!r}")
    if pointer == "":
        return data
    if not pointer.startswith("/"):
        raise _PointerError(f"invalid JSON pointer: {pointer!r}")

    current = data
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                raise _PointerError(f"pointer {pointer!r} does not resolve: missing key {token!r}")
            current = current[token]
        elif isinstance(current, list):
            try:
                position = int(token)
            except ValueError as exc:
                raise _PointerError(
                    f"pointer {pointer!r} does not resolve: invalid index {token!r}"
                ) from exc
            if position < 0 or position >= len(current):
                raise _PointerError(f"pointer {pointer!r} does not resolve: index out of range")
            current = current[position]
        else:
            raise _PointerError(f"pointer {pointer!r} does not resolve: not indexable")
    return current


def format_validation_retry(error: SchemaError) -> str:
    details = error.details if isinstance(error.details, dict) else {}
    path = details.get("path", [])
    message = details.get("message", error.message)
    path_text = ".".join(str(part) for part in path) if path else "<root>"
    return (
        "Your previous JSON output failed validation against the caller schema. "
        f"Fix the JSON and return only valid JSON. Validation path: {path_text}. "
        f"Validation error: {message}."
    )
