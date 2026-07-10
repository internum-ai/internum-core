from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from api.config.settings import CoreSettings


class SafeRequestOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    model: str | None = Field(default=None, min_length=1)
    models: list[str] | None = Field(default=None)
    system_prompt: str | None = Field(default=None, alias="systemPrompt", min_length=1)


MAX_MODELS_CHAIN_LENGTH = 3


@dataclass(frozen=True)
class ResolvedModelConfig:
    model: str
    models: list[str]
    system_prompt: str
    reasoning_effort: str | None
    temperature: float | None
    max_output_tokens: int | None


def resolve_request_overrides(
    settings: CoreSettings,
    overrides: SafeRequestOverrides | None,
) -> ResolvedModelConfig:
    system_prompt = settings.default_system_prompt
    if overrides is not None and overrides.system_prompt:
        system_prompt = overrides.system_prompt

    chain = _resolve_model_chain(settings, overrides)

    return ResolvedModelConfig(
        model=chain[0],
        models=chain,
        system_prompt=system_prompt,
        reasoning_effort=settings.default_reasoning_effort,
        temperature=settings.default_temperature,
        max_output_tokens=settings.default_max_output_tokens,
    )


def _resolve_model_chain(
    settings: CoreSettings,
    overrides: SafeRequestOverrides | None,
) -> list[str]:
    if overrides is not None and overrides.models:
        chain = list(overrides.models)
    elif overrides is not None and overrides.model:
        chain = [overrides.model]
    elif settings.default_models:
        chain = list(settings.default_models)
    else:
        chain = [settings.default_model]

    deduped: list[str] = []
    for model in chain:
        if model and model not in deduped:
            deduped.append(model)
    deduped = deduped[:MAX_MODELS_CHAIN_LENGTH]

    return deduped or [settings.default_model]
