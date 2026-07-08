from typing import Any

from jsonschema import Draft202012Validator, ValidationError
from jsonschema.exceptions import SchemaError as JsonSchemaDefinitionError

from api.common.errors import SchemaError


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
