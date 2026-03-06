.PHONY: install test lint typecheck check fmt run check-telegram set-webhook run-flow clean test-llm-api test-llm-agent test-llm

install:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

typecheck:
	uv run mypy src/elephant/

check: lint typecheck test

fmt:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

run:
	PYTHONPATH=src uv run python -m elephant.main --config config.yaml

check-telegram:
	PYTHONPATH=src uv run python -m elephant.check_telegram --config config.yaml

set-webhook:
	PYTHONPATH=src uv run python -m elephant.check_telegram --config config.yaml --set-webhook

run-flow:
	PYTHONPATH=src uv run python -m elephant.run_flow $(FLOW)

test-llm-api:
	uv run pytest -m llm_api

test-llm-agent:
	uv run pytest -m llm_agent

test-llm:
	uv run pytest -m "llm_api or llm_agent"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf .ruff_cache
