import json
import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.common.logging import RequestContextMiddleware, configure_logging, log_event


def test_production_logging_emits_compact_json_with_request_context(
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="production", log_level=None)
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/logged")
    async def logged() -> dict[str, bool]:
        log_event("example.logged", value=1)
        return {"ok": True}

    response = TestClient(app).get("/logged", headers={"X-Request-ID": "request-123"})

    lines = capsys.readouterr().out.splitlines()
    events = [json.loads(line) for line in lines]
    assert response.headers["X-Request-ID"] == "request-123"
    assert events[0] == {
        "event": "example.logged",
        "requestId": "request-123",
        "value": 1,
    }
    assert events[1]["event"] == "request.completed"
    assert events[1]["requestId"] == "request-123"


def test_development_logging_defaults_to_debug_pretty_stdout_and_is_idempotent(
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    configure_logging(environment="development", log_level=None)
    configure_logging(environment="development", log_level=None)

    log_event("debug.detail", level=logging.DEBUG, markdown="# Private")

    output = capsys.readouterr().out
    assert "DEBUG debug.detail" in output
    assert 'markdown="# Private"' in output
    assert (
        len(
            [
                handler
                for handler in logging.getLogger("internum.api").handlers
                if getattr(handler, "_internum_stdout_handler", False)
            ]
        )
        == 1
    )
