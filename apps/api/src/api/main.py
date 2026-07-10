from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.capabilities.document_parsing import router as document_parsing_router
from api.capabilities.document_parsing.extraction import validate_document_runtime
from api.common.errors import install_exception_handlers
from api.common.logging import RequestContextMiddleware, configure_logging
from api.common.usage import InMemoryUsageTracker
from api.config.settings import CoreSettings


def create_app(settings: CoreSettings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_settings = settings or CoreSettings.from_env()
        app.state.settings = active_settings
        configure_logging(
            environment=active_settings.environment,
            log_level=active_settings.log_level,
        )
        validate_document_runtime(active_settings)
        yield

    app = FastAPI(title="Internum Core API", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.usage_tracker = InMemoryUsageTracker()
    install_exception_handlers(app)
    app.add_middleware(RequestContextMiddleware)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(document_parsing_router)

    return app


app = create_app()
