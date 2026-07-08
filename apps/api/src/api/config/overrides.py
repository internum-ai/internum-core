from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from api.config.settings import CoreSettings


class SafeRequestOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    model: str | None = Field(default=None, min_length=1)
    system_prompt: str | None = Field(default=None, alias="systemPrompt", min_length=1)


@dataclass(frozen=True)
class ResolvedModelConfig:
    model: str
    system_prompt: str


def resolve_request_overrides(
    settings: CoreSettings,
    overrides: SafeRequestOverrides | None,
) -> ResolvedModelConfig:
    if overrides is None:
        return ResolvedModelConfig(
            model=settings.default_model,
            system_prompt=settings.default_system_prompt,
        )

    return ResolvedModelConfig(
        model=overrides.model or settings.default_model,
        system_prompt=overrides.system_prompt or settings.default_system_prompt,
    )
