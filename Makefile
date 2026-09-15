# ConsciousAI Journal V2 — Makefile
# ============================================================
# NOTE: On Windows without Make, use the equivalent commands
# shown in the README or run them directly in PowerShell.
# ============================================================

.PHONY: install dev test lint format check run clean help

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@echo "  make install  — Install project dependencies"
	@echo "  make dev      — Install with dev dependencies"
	@echo "  make test     — Run test suite"
	@echo "  make lint     — Run linter (ruff check)"
	@echo "  make format   — Format code (ruff format)"
	@echo "  make check    — Run lint + format check + tests"
	@echo "  make run      — Start the development server"
	@echo "  make clean    — Remove caches and build artifacts"

install: ## Install production dependencies
	pip install -e .

dev: ## Install with development dependencies
	pip install -e ".[dev]"

test: ## Run the test suite
	python -m pytest tests/ -v

lint: ## Run ruff linter
	python -m ruff check app/ tests/

format: ## Format code with ruff
	python -m ruff format app/ tests/

check: ## Run all checks (lint + format check + tests)
	python -m ruff check app/ tests/
	python -m ruff format --check app/ tests/
	python -m pytest tests/ -v

run: ## Start the development server
	python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/
