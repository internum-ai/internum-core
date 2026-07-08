set shell := ["zsh", "-cu"]

default:
    @just --list

install:
    uv sync --all-packages --all-groups
    if command -v lefthook >/dev/null 2>&1; then lefthook install; fi

dev:
    uv run uvicorn api.main:app --app-dir apps/api/src --reload

check: lint type test build

fix:
    uv run ruff format .
    uv run ruff check . --fix

lint:
    uv run ruff format --check .
    uv run ruff check .

type:
    uv run mypy --config-file packages/config/mypy.ini apps/api/src packages/config/src packages/shared/src

test:
    uv run pytest

build:
    uv build --all-packages

clean:
    rm -rf .mypy_cache .pytest_cache .ruff_cache dist build
    find . -type d -name '__pycache__' -prune -exec rm -rf {} +

