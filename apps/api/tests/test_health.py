import pytest
from fastapi.testclient import TestClient

from api.common.errors import ConfigurationError
from api.config.settings import CoreSettings
from api.main import create_app


def test_health_route_returns_ok(core_settings: CoreSettings) -> None:
    client = TestClient(create_app(settings=core_settings))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_startup_fails_when_libreoffice_is_missing(
    core_settings: CoreSettings,
    tmp_path,
) -> None:
    settings = core_settings.model_copy(update={"libreoffice_binary": str(tmp_path / "missing")})

    with pytest.raises(ConfigurationError) as exc_info:
        with TestClient(create_app(settings=settings)):
            pass

    assert exc_info.value.code == "doc_converter_unavailable"


def test_startup_fails_when_libreoffice_binary_is_not_runnable(
    core_settings: CoreSettings,
    tmp_path,
) -> None:
    binary = tmp_path / "soffice"
    binary.write_text("not an executable format")
    binary.chmod(0o755)
    settings = core_settings.model_copy(update={"libreoffice_binary": str(binary)})

    with pytest.raises(ConfigurationError) as exc_info:
        with TestClient(create_app(settings=settings)):
            pass

    assert exc_info.value.code == "doc_converter_unavailable"
