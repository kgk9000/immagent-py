.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: start-postgres
start-postgres: ## Start PostgreSQL in Docker
	@if docker ps -q -f name=immagent-postgres | grep -q .; then \
		echo "PostgreSQL already running"; \
	elif docker ps -aq -f name=immagent-postgres | grep -q .; then \
		echo "Starting existing container..."; \
		docker start immagent-postgres; \
	else \
		echo "Creating new PostgreSQL container..."; \
		docker run -d --name immagent-postgres \
			-e POSTGRES_PASSWORD=postgres \
			-p 5432:5432 \
			postgres:16; \
		echo "Waiting for PostgreSQL to start..."; \
		sleep 2; \
	fi
	@echo "PostgreSQL ready at localhost:5432"

.PHONY: stop-postgres
stop-postgres: ## Stop and remove PostgreSQL container
	@if docker ps -q -f name=immagent-postgres | grep -q .; then \
		docker stop immagent-postgres; \
	fi
	@if docker ps -aq -f name=immagent-postgres | grep -q .; then \
		docker rm immagent-postgres; \
		echo "PostgreSQL container removed"; \
	else \
		echo "No PostgreSQL container to remove"; \
	fi

.PHONY: simple
simple: ## Run simple.py example
	uv run python simple.py

.PHONY: chat
chat: ## Run basic_chat.py interactive example
	uv run python basic_chat.py

.PHONY: mcp-tools
mcp-tools: ## Run mcp_tools.py example
	uv run python mcp_tools.py
