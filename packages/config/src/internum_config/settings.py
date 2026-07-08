from pydantic_settings import BaseSettings, SettingsConfigDict


class InternumBaseSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", case_sensitive=False)
