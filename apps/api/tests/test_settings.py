import pytest
from pydantic import ValidationError

from api.config.overrides import SafeRequestOverrides, resolve_request_overrides
from api.config.settings import ApiConsumerSettings, CoreSettings


def test_settings_parse_consumers_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("CORE_DEFAULT_MODEL", "openai/gpt-5.2")
    monkeypatch.setenv("CORE_DEFAULT_SYSTEM_PROMPT", "Extract facts.")
    monkeypatch.setenv("CORE_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("CORE_MAX_UPLOAD_BYTES", "4096")
    monkeypatch.setenv(
        "CORE_API_CONSUMERS",
        '[{"id":"internal","api_key":"consumer-key","revoked":false}]',
    )

    settings = CoreSettings.from_env(env_file=None)

    assert settings.default_model == "openai/gpt-5.2"
    assert settings.api_consumers[0].id == "internal"


def test_missing_required_settings_fail_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "CORE_OPENROUTER_API_KEY",
        "CORE_DEFAULT_MODEL",
        "CORE_DEFAULT_SYSTEM_PROMPT",
        "CORE_TIMEOUT_SECONDS",
        "CORE_MAX_UPLOAD_BYTES",
        "CORE_API_CONSUMERS",
    ):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValidationError):
        CoreSettings.from_env(env_file=None)


def test_settings_parse_consumers_from_local_env_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    env_file = tmp_path / "local.env"
    env_file.write_text(
        "\n".join(
            [
                'CORE_OPENROUTER_API_KEY="openrouter-key"',
                'CORE_DEFAULT_MODEL="openai/gpt-5.2"',
                'CORE_DEFAULT_SYSTEM_PROMPT="Extract facts."',
                'CORE_TIMEOUT_SECONDS="20"',
                'CORE_MAX_UPLOAD_BYTES="4096"',
                'CORE_API_CONSUMERS=\'[{"id":"internal","api_key":"consumer-key","revoked":false}]\'',
            ]
        )
    )

    settings = CoreSettings.from_env(env_file=env_file)

    assert settings.openrouter_api_key.get_secret_value() == "openrouter-key"
    assert settings.api_consumers[0].api_key.get_secret_value() == "consumer-key"


def test_safe_overrides_resolve_defaults_and_allowed_fields(core_settings: CoreSettings) -> None:
    overrides = SafeRequestOverrides.model_validate(
        {
            "model": "anthropic/claude-sonnet-4.5",
            "systemPrompt": "Use null for unknowns.",
        }
    )

    resolved = resolve_request_overrides(core_settings, overrides)

    assert resolved.model == "anthropic/claude-sonnet-4.5"
    assert resolved.system_prompt == "Use null for unknowns."


def test_safe_overrides_reject_secret_fields() -> None:
    with pytest.raises(ValidationError):
        SafeRequestOverrides.model_validate(
            {
                "model": "openai/gpt-5.2",
                "openrouterApiKey": "must-not-be-accepted",
            }
        )


def test_duplicate_consumer_ids_fail_validation() -> None:
    with pytest.raises(ValidationError):
        CoreSettings(
            openrouter_api_key="openrouter-key",
            default_model="openai/gpt-5.2",
            default_system_prompt="Extract facts.",
            timeout_seconds=30,
            max_upload_bytes=4096,
            api_consumers=[
                ApiConsumerSettings(id="internal", api_key="one"),
                ApiConsumerSettings(id="internal", api_key="two"),
            ],
        )
