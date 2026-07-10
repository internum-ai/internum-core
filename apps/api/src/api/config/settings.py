from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

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
    environment: Literal["development", "production"] = "development"
    log_level: str | None = None
    default_model: str = Field(min_length=1)
    default_system_prompt: str = Field(min_length=1)
    timeout_seconds: float = Field(gt=0)
    max_upload_bytes: int = Field(gt=0)
    max_image_pixels: int = Field(default=25_000_000, gt=0)
    max_pdf_pages: int = Field(default=200, gt=0)
    max_ocr_pages: int = Field(default=25, ge=0)
    max_ocr_rendered_pixels: int = Field(default=20_000_000, gt=0)
    ocr_render_dpi: int = Field(default=200, gt=0)
    max_ooxml_zip_entries: int = Field(default=1_000, gt=0)
    max_ooxml_uncompressed_bytes: int = Field(default=100 * 1024 * 1024, gt=0)
    max_ooxml_compression_ratio: float = Field(default=100.0, gt=0)
    libreoffice_binary: str = Field(default="soffice", min_length=1)
    doc_conversion_timeout_seconds: float = Field(default=30.0, gt=0)
    api_consumers: list[ApiConsumerSettings] = Field(min_length=1)

    @field_validator("default_model", "default_system_prompt", "libreoffice_binary")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("log level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        return normalized

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
