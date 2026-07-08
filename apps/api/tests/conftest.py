import pytest

from api.config.settings import ApiConsumerSettings, CoreSettings


@pytest.fixture
def core_settings() -> CoreSettings:
    return CoreSettings(
        openrouter_api_key="openrouter-test-key",
        default_model="openai/gpt-5.2",
        default_system_prompt="Return factual JSON only.",
        timeout_seconds=30,
        max_upload_bytes=1024 * 1024,
        api_consumers=[
            ApiConsumerSettings(id="internal", api_key="valid-key", revoked=False),
            ApiConsumerSettings(id="revoked", api_key="revoked-key", revoked=True),
        ],
    )
