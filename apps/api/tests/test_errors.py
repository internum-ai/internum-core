import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.common.errors import ApiError, IntakeError, install_exception_handlers
from api.common.logging import RequestContextMiddleware, configure_logging


def test_api_errors_use_common_envelope() -> None:
    app = FastAPI()
    install_exception_handlers(app)
    app.add_middleware(RequestContextMiddleware)

    @app.get("/boom")
    async def boom() -> None:
        raise ApiError("example_error", "Example failed", details={"field": "value"})

    response = TestClient(app).get("/boom")

    assert response.status_code == 400
    assert response.json()["error"] == {
        "code": "example_error",
        "message": "Example failed",
        "details": {"field": "value"},
        "requestId": response.headers["X-Request-ID"],
    }


def test_request_validation_errors_use_common_envelope() -> None:
    app = FastAPI()
    install_exception_handlers(app)
    app.add_middleware(RequestContextMiddleware)

    @app.get("/items")
    async def items(limit: int) -> dict[str, int]:
        return {"limit": limit}

    response = TestClient(app).get("/items", params={"limit": "not-an-int"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert "requestId" in response.json()["error"]


def test_api_errors_log_stable_code_and_intake_rejection(
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="DEBUG")
    app = FastAPI()
    install_exception_handlers(app)
    app.add_middleware(RequestContextMiddleware)

    @app.get("/rejected")
    async def rejected() -> None:
        raise IntakeError(
            "Unsupported file type",
            code="unsupported_file_type",
            details={"documentType": "unknown"},
        )

    response = TestClient(app).get("/rejected", headers={"X-Request-ID": "request-1"})

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert response.status_code == 400
    assert any(
        event["event"] == "intake.rejected"
        and event["code"] == "unsupported_file_type"
        and event["documentType"] == "unknown"
        for event in events
    )
    assert any(
        event["event"] == "request.failed"
        and event["code"] == "unsupported_file_type"
        and event["requestId"] == "request-1"
        for event in events
    )


def test_unexpected_exception_is_logged_once_with_stack_trace(
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level="DEBUG")
    app = FastAPI()
    install_exception_handlers(app)
    app.add_middleware(RequestContextMiddleware)

    @app.get("/unexpected")
    async def unexpected() -> None:
        raise RuntimeError("unexpected secret")

    response = TestClient(app).get("/unexpected", headers={"X-Request-ID": "request-2"})

    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.out.splitlines()]
    failures = [event for event in events if event["event"] == "request.failed"]
    completed = [event for event in events if event["event"] == "request.completed"]
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert response.headers["X-Request-ID"] == "request-2"
    assert len(failures) == 1
    assert len(completed) == 1
    assert completed[0]["statusCode"] == 500
    assert failures[0]["requestId"] == "request-2"
    assert failures[0]["exceptionType"] == "RuntimeError"
    assert "RuntimeError: unexpected secret" in failures[0]["exception"]
    assert captured.err == ""
