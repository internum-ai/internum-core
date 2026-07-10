import logging
import time
from collections.abc import Sequence
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.common.logging import log_event


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
        self.intake_event_logged = False
        super().__init__(code, message, status_code=HTTPStatus.BAD_REQUEST, details=details)


class SchemaError(ApiError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            "schema_error", message, status_code=HTTPStatus.UNPROCESSABLE_ENTITY, details=details
        )


class UpstreamError(ApiError):
    def __init__(
        self,
        message: str = "Upstream provider failed",
        *,
        attempt: int | None = None,
    ) -> None:
        self.attempt = attempt
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
        if isinstance(exc, IntakeError):
            details = exc.details if isinstance(exc.details, dict) else {}
            if not exc.intake_event_logged:
                log_event(
                    "intake.rejected",
                    level=logging.WARNING,
                    code=exc.code,
                    documentType=details.get("documentType", "unknown"),
                    durationMs=_request_duration_ms(request),
                    requestId=getattr(request.state, "request_id", None),
                )
        log_event(
            "request.failed",
            level=logging.ERROR,
            code=exc.code,
            statusCode=exc.status_code,
            exceptionType=type(exc).__name__,
            requestId=getattr(request.state, "request_id", None),
        )
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
        log_event(
            "request.failed",
            level=logging.ERROR,
            code="validation_error",
            statusCode=HTTPStatus.UNPROCESSABLE_ENTITY,
            exceptionType=type(exc).__name__,
            requestId=getattr(request.state, "request_id", None),
        )
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


def _request_duration_ms(request: Request) -> float:
    started_at = getattr(request.state, "request_started_at", time.perf_counter())
    return round((time.perf_counter() - started_at) * 1000, 3)
