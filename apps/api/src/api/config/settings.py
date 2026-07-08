from functools import lru_cache
from pathlib import Path
from typing import Self

from internum_config import InternumBaseSettings
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import SettingsConfigDict

LOCAL_ENV_FILE = "apps/api/.env"


class ApiConsumerSettings(InternumBaseSettings):
    id: str = Field(min_length=1)
    api_key: SecretStr = Field(min_length=1)
    revoked: bool = False

    @field_validator("id")
    @classmethod
    def strip_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("consumer id must not be blank")
        return stripped


class CoreSettings(InternumBaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CORE_",
        env_file=LOCAL_ENV_FILE,
        extra="forbid",
        case_sensitive=False,
    )

    openrouter_api_key: SecretStr = Field(min_length=1)
    default_model: str = Field(min_length=1)
    default_system_prompt: str = Field(min_length=1)
    timeout_seconds: float = Field(gt=0)
    max_upload_bytes: int = Field(gt=0)
    api_consumers: list[ApiConsumerSettings] = Field(min_length=1)

    @field_validator("default_model", "default_system_prompt")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @field_validator("api_consumers")
    @classmethod
    def require_unique_consumer_ids(
        cls, value: list[ApiConsumerSettings]
    ) -> list[ApiConsumerSettings]:
        seen: set[str] = set()
        for consumer in value:
            if consumer.id in seen:
                raise ValueError(f"duplicate consumer id: {consumer.id}")
            seen.add(consumer.id)
        return value

    def find_consumer_by_key(self, api_key: str) -> ApiConsumerSettings | None:
        from hmac import compare_digest

        for consumer in self.api_consumers:
            if compare_digest(consumer.api_key.get_secret_value(), api_key):
                return consumer
        return None

    @classmethod
    def from_env(cls, *, env_file: str | Path | None = LOCAL_ENV_FILE) -> Self:
        return cls(_env_file=env_file)  # type: ignore[call-arg]


@lru_cache
def get_cached_settings() -> CoreSettings:
    return CoreSettings.from_env()
