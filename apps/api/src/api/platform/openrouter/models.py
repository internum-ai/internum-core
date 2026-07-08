from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class ImageInput:
    mime_type: str
    base64_data: str
    detail: str = "auto"

    @property
    def data_url(self) -> str:
        return f"data:{self.mime_type};base64,{self.base64_data}"


@dataclass(frozen=True)
class OpenRouterRequest:
    model: str
    system_prompt: str
    user_content: str
    schema: dict[str, Any]
    capability: str
    consumer_id: str | None = None
    request_id: str | None = None
    images: list[ImageInput] = field(default_factory=list)
    validation_retry_prompt: str | None = None


@dataclass(frozen=True)
class OpenRouterResult:
    content: str
    model: str
    provider: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: Decimal | None
