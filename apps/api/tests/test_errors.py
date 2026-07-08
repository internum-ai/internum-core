from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.common.errors import ApiError, install_exception_handlers
from api.common.logging import RequestContextMiddleware


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
