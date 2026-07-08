from typing import Any

from api.config.settings import CoreSettings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_openai_compatible_client(
    settings: CoreSettings,
    *,
    openai_cls: type[Any] | None = None,
) -> Any:
    if openai_cls is None:
        from openai import OpenAI

        openai_cls = OpenAI

    return openai_cls(
        api_key=settings.openrouter_api_key.get_secret_value(),
        base_url=OPENROUTER_BASE_URL,
    )
