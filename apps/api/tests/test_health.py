from fastapi.testclient import TestClient

from api.config.settings import CoreSettings
from api.main import create_app


def test_health_route_returns_ok(core_settings: CoreSettings) -> None:
    client = TestClient(create_app(settings=core_settings))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
