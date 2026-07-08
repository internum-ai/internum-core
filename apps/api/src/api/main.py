from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.common.errors import install_exception_handlers
from api.common.logging import RequestContextMiddleware
from api.common.usage import InMemoryUsageTracker
from api.config.settings import CoreSettings


def create_app(settings: CoreSettings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings is None:
            app.state.settings = CoreSettings.from_env()
        yield

    app = FastAPI(title="Internum Core API", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.usage_tracker = InMemoryUsageTracker()
    install_exception_handlers(app)
    app.add_middleware(RequestContextMiddleware)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
