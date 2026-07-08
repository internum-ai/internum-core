set shell := ["zsh", "-cu"]

default:
    @just --list

install:
    uv sync --all-packages --all-groups
    if command -v lefthook >/dev/null 2>&1; then lefthook install; fi

dev:
    uv run uvicorn api.main:app --app-dir apps/api/src --reload

check: lint type test doctor build

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

doctor: dead-code dependencies

dead-code:
    uv run vulture apps/api/src packages/config/src packages/shared/src apps/api/tests packages/config/tests packages/shared/tests --min-confidence 90 --ignore-names app

dependencies:
    uv run deptry apps/api/src --config apps/api/pyproject.toml --known-first-party api --package-module-name-map python-magic=magic,json-repair=json_repair,markitdown-ocr=markitdown_ocr,internum-config=internum_config --per-rule-ignores 'DEP002=markitdown-ocr|python-multipart|uvicorn'
    uv run deptry packages/config/src --config packages/config/pyproject.toml --known-first-party internum_config --package-module-name-map pydantic-settings=pydantic_settings
    uv run deptry packages/shared/src --config packages/shared/pyproject.toml --known-first-party internum_shared

smoke:
    uv run python scripts/live_parse_smoke.py

build:
    uv build --all-packages

clean:
    rm -rf .mypy_cache .pytest_cache .ruff_cache dist build
    find . -type d -name '__pycache__' -prune -exec rm -rf {} +
