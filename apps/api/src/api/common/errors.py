from collections.abc import Sequence
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = HTTPStatus.BAD_REQUEST,
        details: dict[str, Any] | list[Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class ConfigurationError(ApiError):
    def __init__(
        self,
        message: str = "Invalid server configuration",
        *,
        code: str = "configuration_error",
    ) -> None:
        super().__init__(code, message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)


class AuthenticationError(ApiError):
    def __init__(
        self,
        code: str = "authentication_failed",
        message: str = "Authentication failed",
        *,
        status_code: int = HTTPStatus.UNAUTHORIZED,
    ) -> None:
        super().__init__(code, message, status_code=status_code)


class IntakeError(ApiError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "intake_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, message, status_code=HTTPStatus.BAD_REQUEST, details=details)


class SchemaError(ApiError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            "schema_error", message, status_code=HTTPStatus.UNPROCESSABLE_ENTITY, details=details
        )


class UpstreamError(ApiError):
    def __init__(self, message: str = "Upstream provider failed") -> None:
        super().__init__("upstream_error", message, status_code=HTTPStatus.BAD_GATEWAY)


def build_error_envelope(
    code: str,
    message: str,
    *,
    request_id: str | None = None,
    details: dict[str, Any] | list[Any] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": code,
        "message": message,
    }
    if details is not None:
        error["details"] = details
    if request_id is not None:
        error["requestId"] = request_id
    return {"error": error}


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_envelope(
                exc.code,
                exc.message,
                request_id=getattr(request.state, "request_id", None),
                details=exc.details,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            content=build_error_envelope(
                "validation_error",
                "Request validation failed",
                request_id=getattr(request.state, "request_id", None),
                details=_sanitize_validation_errors(exc.errors()),
            ),
        )


def _sanitize_validation_errors(errors: Sequence[Any]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for error in errors:
        if not isinstance(error, dict):
            continue
        sanitized_error = {
            key: value for key, value in error.items() if key not in {"input", "ctx"}
        }
        sanitized.append(sanitized_error)
    return sanitized
