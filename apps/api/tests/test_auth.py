from fastapi import APIRouter, Depends, Request
from fastapi.testclient import TestClient

from api.common.auth import require_consumer
from api.config.settings import CoreSettings
from api.main import create_app


def _client_with_protected_route(settings: CoreSettings) -> TestClient:
    app = create_app(settings=settings)
    router = APIRouter(prefix="/v1", dependencies=[Depends(require_consumer)])

    @router.get("/protected")
    async def protected(request: Request) -> dict[str, str]:
        return {"consumerId": request.state.consumer.id}

    app.include_router(router)
    return TestClient(app)


def test_missing_api_key_returns_common_error(core_settings: CoreSettings) -> None:
    client = _client_with_protected_route(core_settings)

    response = client.get("/v1/protected")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "missing_api_key"
    assert "requestId" in response.json()["error"]


def test_invalid_api_key_returns_common_error(core_settings: CoreSettings) -> None:
    client = _client_with_protected_route(core_settings)

    response = client.get("/v1/protected", headers={"X-API-Key": "wrong"})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_revoked_api_key_is_rejected(core_settings: CoreSettings) -> None:
    client = _client_with_protected_route(core_settings)

    response = client.get("/v1/protected", headers={"X-API-Key": "revoked-key"})

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "revoked_api_key"


def test_valid_api_key_sets_consumer_identity(core_settings: CoreSettings) -> None:
    client = _client_with_protected_route(core_settings)

    response = client.get("/v1/protected", headers={"X-API-Key": "valid-key"})

    assert response.status_code == 200
    assert response.json() == {"consumerId": "internal"}
