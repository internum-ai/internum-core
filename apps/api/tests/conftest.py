import pytest

from api.config.settings import ApiConsumerSettings, CoreSettings


@pytest.fixture
def core_settings() -> CoreSettings:
    # `_env_file=None` keeps the fixture hermetic: without it, any field not
    # passed here (e.g. `default_models`) is silently populated from the
    # developer's local `apps/api/.env`, making tests depend on that file.
    return CoreSettings(
        _env_file=None,
        openrouter_api_key="openrouter-test-key",
        default_model="openai/gpt-5.2",
        default_models=None,
        default_system_prompt="Return factual JSON only.",
        timeout_seconds=30,
        max_upload_bytes=1024 * 1024,
        api_consumers=[
            ApiConsumerSettings(id="internal", api_key="valid-key", revoked=False),
            ApiConsumerSettings(id="revoked", api_key="revoked-key", revoked=True),
        ],
    )
