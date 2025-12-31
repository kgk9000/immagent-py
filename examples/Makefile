.PHONY: help postgres stop-postgres simple chat

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

postgres: ## Start PostgreSQL in Docker
	docker run -d --name immagent-postgres \
		-e POSTGRES_PASSWORD=postgres \
		-p 5432:5432 \
		postgres:16
	@echo "Waiting for PostgreSQL to start..."
	@sleep 2
	@echo "PostgreSQL ready at localhost:5432"

stop-postgres: ## Stop and remove PostgreSQL container
	docker stop immagent-postgres && docker rm immagent-postgres

simple: ## Run simple.py example
	uv run python simple.py

chat: ## Run basic_chat.py interactive example
	uv run python basic_chat.py
