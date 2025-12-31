.PHONY: help install dev fmt lint check test clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

dev: ## Install with dev dependencies
	uv sync --all-extras

fmt: ## Format code with ruff
	uv run ruff format src/ tests/

lint: ## Lint code with ruff
	uv run ruff check src/ tests/

fix: ## Fix auto-fixable lint errors
	uv run ruff check --fix src/ tests/

check: lint ## Run all checks (lint + type check)
	uv run python -c "import immagent"

test: ## Run tests
	uv run pytest tests/ -v

test-cov: ## Run tests with coverage
	uv run pytest tests/ -v --cov=immagent --cov-report=term-missing

clean: ## Remove build artifacts and caches
	rm -rf .venv/ .pytest_cache/ .ruff_cache/ dist/ build/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +

db-init: ## Initialize database schema (requires DATABASE_URL)
	uv run python -c "import asyncio; from immagent import Database; asyncio.run(Database.connect('$${DATABASE_URL}').init_schema())"
