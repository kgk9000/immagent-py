.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install dependencies
	uv sync

.PHONY: dev
dev: ## Install with dev dependencies
	uv sync --all-extras

.PHONY: fmt
fmt: ## Format code with ruff
	uv run ruff format src/ tests/

.PHONY: lint
lint: ## Lint code with ruff
	uv run ruff check src/ tests/

.PHONY: fix
fix: ## Fix auto-fixable lint errors
	uv run ruff check --fix src/ tests/

.PHONY: typecheck
typecheck: ## Run type checker (pyright)
	uv run pyright src/

.PHONY: check
check: lint typecheck ## Run all checks (lint + type check)

.PHONY: test
test: ## Run tests
	@bash -c 'test -f .env && source .env; uv run pytest tests/ -v'

.PHONY: test-cov
test-cov: ## Run tests with coverage
	@bash -c 'test -f .env && source .env; uv run pytest tests/ -v --cov=immagent --cov-report=term-missing'

.PHONY: clean
clean: ## Remove build artifacts and caches
	rm -rf .venv/ .pytest_cache/ .ruff_cache/ dist/ build/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +

.PHONY: db-init
db-init: ## Initialize database schema (requires DATABASE_URL)
	uv run python -c "import asyncio; from immagent import Database; asyncio.run(Database.connect('$${DATABASE_URL}').init_schema())"
