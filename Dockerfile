# syntax=docker/dockerfile:1

# =============================================================================
# Stage 1: builder
#   Resolves and installs the uv workspace into a self-contained venv using
#   the base image's system Python interpreter (not a uv-managed download).
#   Dependency resolution is cached separately from source changes by copying
#   the lockfile + all workspace pyproject.toml files first, syncing deps,
#   and only then copying source and installing the workspace packages.
# =============================================================================
FROM python:3.12-slim AS builder

# Pull the uv/uvx binaries from the official distroless uv image instead of
# installing via pip/curl, keeping the builder stage minimal and reproducible.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Pin uv to the system interpreter so the venv it builds is portable to the
# runtime stage's own python:3.12-slim base (same interpreter, no downloads).
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# --- Layer 1: dependency resolution only -----------------------------------
# Copy just the lockfile and the pyproject.toml of every workspace member.
# This lets Docker cache the (slow) dependency sync as long as none of these
# manifest files change, independent of application source edits.
COPY uv.lock pyproject.toml ./
COPY apps/api/pyproject.toml apps/api/pyproject.toml
COPY packages/config/pyproject.toml packages/config/pyproject.toml
COPY packages/shared/pyproject.toml packages/shared/pyproject.toml

# Install only the dependencies internum-api needs (transitively, via the
# workspace), without installing the workspace packages themselves yet.
RUN uv sync --frozen --no-install-workspace --no-dev --package internum-api

# --- Layer 2: install the workspace packages --------------------------------
# Now bring in the actual source trees and install them non-editably so that
# `api`, `internum_config`, and `internum_shared` are real packages inside
# the venv's site-packages (no editable/path-based imports at runtime).
COPY apps/ apps/
COPY packages/ packages/

RUN uv sync --frozen --no-editable --no-dev --package internum-api

# =============================================================================
# Stage 2: runtime
#   Minimal image containing only the built venv and the system libraries
#   required at runtime (LibreOffice for document conversion, libmagic for
#   MIME sniffing). No app source, no build tooling, no uv.
# =============================================================================
FROM python:3.12-slim AS runtime

# Only the runtime system dependencies markitdown/document parsing needs:
# LibreOffice (writer + impress components) for document conversion, and
# libmagic for python-magic. No Tesseract, no curl.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libreoffice-writer \
        libreoffice-impress \
        libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Dedicated non-root user. LibreOffice writes a per-request user profile
# under $HOME/.config and Python writes tempfiles under /tmp, so both must
# be writable by this user.
RUN groupadd --system appuser \
    && useradd --system --gid appuser --home-dir /home/appuser --create-home appuser \
    && mkdir -p /tmp \
    && chown -R appuser:appuser /home/appuser /tmp

# Bring in only the built venv from the builder stage — no app source, no
# uv, no build-only dependencies.
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV=/app/.venv \
    HOME=/home/appuser \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    WEB_CONCURRENCY=2

EXPOSE 8000

USER appuser
WORKDIR /home/appuser

# Generous start-period to absorb LibreOffice's slow first-invocation warmup
# (soffice spins up a headless UNO process on first conversion request).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health',timeout=5).status==200 else 1)"

# Shell form so $PORT/$WEB_CONCURRENCY are expanded, combined with `exec` so
# uvicorn replaces the shell as PID 1 and receives SIGTERM directly.
CMD exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT" --workers "$WEB_CONCURRENCY"
