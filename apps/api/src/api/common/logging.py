import json
import logging as py_logging
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

logger = py_logging.getLogger("internum.api")
_request_id: ContextVar[str | None] = ContextVar("internum_request_id", default=None)


class _JsonFormatter(py_logging.Formatter):
    def format(self, record: py_logging.LogRecord) -> str:
        payload = _payload_for(record)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class _PrettyFormatter(py_logging.Formatter):
    def format(self, record: py_logging.LogRecord) -> str:
        payload = _payload_for(record)
        event = str(payload.pop("event", record.getMessage()))
        fields = " ".join(
            f"{key}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
            for key, value in sorted(payload.items())
        )
        line = f"{record.levelname} {event}"
        if fields:
            line = f"{line} {fields}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def configure_logging(*, environment: str, log_level: str | None) -> None:
    """Configure the application logger with exactly one stdout handler."""
    level_name = (log_level or ("INFO" if environment == "production" else "DEBUG")).upper()
    level = py_logging.getLevelNamesMapping().get(level_name)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported log level: {level_name}")

    for existing_handler in list(logger.handlers):
        logger.removeHandler(existing_handler)
        existing_handler.close()

    handler = py_logging.StreamHandler(sys.stdout)
    handler._internum_stdout_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(_JsonFormatter() if environment == "production" else _PrettyFormatter())
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        request.state.request_started_at = started_at = time.perf_counter()
        token = _request_id.set(request_id)

        try:
            try:
                response = await call_next(request)
            except Exception as exc:
                log_event(
                    "request.failed",
                    level=py_logging.ERROR,
                    exc_info=True,
                    code="internal_error",
                    statusCode=500,
                    exceptionType=type(exc).__name__,
                    durationMs=_duration_ms(started_at),
                )
                response = JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "code": "internal_error",
                            "message": "Internal server error",
                            "requestId": request_id,
                        }
                    },
                )
            duration_ms = _duration_ms(started_at)
            response.headers["X-Request-ID"] = request_id

            log_event(
                "request.completed",
                method=request.method,
                path=request.url.path,
                statusCode=response.status_code,
                durationMs=duration_ms,
                consumerId=getattr(getattr(request.state, "consumer", None), "id", None),
            )
            return response
        finally:
            _request_id.reset(token)


def log_event(
    event: str,
    *,
    level: int = py_logging.INFO,
    exc_info: bool = False,
    **fields: object,
) -> None:
    """Emit a structured application event through the sole application log path."""
    request_id = _request_id.get()
    payload = {
        "event": event,
        **({"requestId": request_id} if request_id is not None else {}),
        **{key: value for key, value in fields.items() if value is not None},
    }
    logger.log(level, event, extra={"event_payload": payload}, exc_info=exc_info)


def _payload_for(record: py_logging.LogRecord) -> dict[str, Any]:
    payload = getattr(record, "event_payload", {"event": record.getMessage()})
    if not isinstance(payload, Mapping):
        payload = {"event": record.getMessage()}
    return {str(key): _json_safe(value) for key, value in payload.items()}


def _json_safe(value: object) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _duration_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)
