from dataclasses import dataclass
from http import HTTPStatus
from typing import Annotated

from fastapi import Header, Request

from api.common.errors import AuthenticationError, ConfigurationError
from api.config.settings import CoreSettings


@dataclass(frozen=True)
class ConsumerIdentity:
    id: str


def get_settings(request: Request) -> CoreSettings:
    settings = getattr(request.app.state, "settings", None)
    if not isinstance(settings, CoreSettings):
        raise ConfigurationError()
    return settings


async def require_consumer(
    request: Request,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> ConsumerIdentity:
    if api_key is None or not api_key:
        raise AuthenticationError("missing_api_key", "Missing API key")

    settings = get_settings(request)
    consumer = settings.find_consumer_by_key(api_key)
    if consumer is None:
        raise AuthenticationError("invalid_api_key", "Invalid API key")
    if consumer.revoked:
        raise AuthenticationError(
            "revoked_api_key",
            "API key has been revoked",
            status_code=HTTPStatus.FORBIDDEN,
        )

    identity = ConsumerIdentity(id=consumer.id)
    request.state.consumer = identity
    return identity
