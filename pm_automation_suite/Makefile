# PM Automation Suite Makefile
# Common development commands

.PHONY: help install install-dev test test-coverage lint format clean run-api run-scheduler docs build docker-build docker-run setup-pre-commit

# Default target - show help
help:
	@echo "PM Automation Suite - Development Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install all dependencies (including dev)"
	@echo "  make setup-pre-commit Set up pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  make test            Run all tests"
	@echo "  make test-coverage   Run tests with coverage report"
	@echo "  make lint            Run linting checks"
	@echo "  make format          Format code with black and isort"
	@echo "  make type-check      Run mypy type checking"
	@echo ""
	@echo "Running Services:"
	@echo "  make run-api         Run the FastAPI server"
	@echo "  make run-scheduler   Run the task scheduler"
	@echo "  make run-worker      Run Celery worker (if using)"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs            Build documentation"
	@echo "  make docs-serve      Serve documentation locally"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-run      Run Docker container"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean           Clean up generated files"
	@echo "  make update-deps     Update dependencies to latest versions"
	@echo "  make security-check  Run security vulnerability scan"

# Installation targets
install:
	pip install -r requirements.txt

install-dev: install
	pip install -e .
	@echo "✅ Development environment ready!"

# Setup pre-commit hooks
setup-pre-commit:
	pre-commit install
	@echo "✅ Pre-commit hooks installed!"

# Testing targets
test:
	pytest tests/ -v --tb=short

test-coverage:
	pytest tests/ --cov=pm_automation_suite --cov-report=html --cov-report=term-missing
	@echo "📊 Coverage report generated in htmlcov/index.html"

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# Code quality targets
lint:
	@echo "🔍 Running flake8..."
	flake8 pm_automation_suite tests
	@echo "🔍 Running pylint..."
	pylint pm_automation_suite
	@echo "✅ Linting complete!"

format:
	@echo "🎨 Formatting with black..."
	black pm_automation_suite tests
	@echo "🎨 Sorting imports with isort..."
	isort pm_automation_suite tests
	@echo "✅ Formatting complete!"

type-check:
	@echo "🔍 Running mypy type checking..."
	mypy pm_automation_suite
	@echo "✅ Type checking complete!"

# Running services
run-api:
	uvicorn pm_automation_suite.api.main:app --reload --host $(API_HOST) --port $(API_PORT)

run-scheduler:
	python -m pm_automation_suite.orchestration.scheduler

run-worker:
	celery -A pm_automation_suite.orchestration.celery_app worker --loglevel=info

run-flower:
	celery -A pm_automation_suite.orchestration.celery_app flower

# Documentation
docs:
	mkdocs build

docs-serve:
	mkdocs serve --dev-addr=127.0.0.1:8001

# Docker targets
docker-build:
	docker build -t pm-automation-suite:latest .

docker-run:
	docker run -p 8000:8000 --env-file .env pm-automation-suite:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# Database management
db-init:
	python -m pm_automation_suite.database.init_db

db-migrate:
	alembic upgrade head

db-rollback:
	alembic downgrade -1

# Maintenance targets
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage
	@echo "✅ Cleanup complete!"

update-deps:
	pip install --upgrade pip setuptools wheel
	pip install --upgrade -r requirements.txt
	@echo "✅ Dependencies updated!"

security-check:
	@echo "🔒 Running security checks..."
	pip install safety
	safety check --json
	@echo "✅ Security check complete!"

# Development workflow shortcuts
dev: install-dev setup-pre-commit
	@echo "🚀 Development environment is ready!"

check: lint type-check test
	@echo "✅ All checks passed!"

# Environment setup
env-setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "📝 Created .env file from template. Please update with your values."; \
	else \
		echo "⚠️  .env file already exists."; \
	fi

# Quick commands for common tasks
quick-fix: format lint
	@echo "✨ Code formatted and linted!"

# Performance profiling
profile:
	python -m cProfile -o profile.stats -m pm_automation_suite.main
	python -m pstats profile.stats

# Generate requirements from setup.py
freeze:
	pip freeze > requirements-frozen.txt
	@echo "📦 Frozen requirements saved to requirements-frozen.txt"

# Variables (can be overridden)
API_HOST ?= 0.0.0.0
API_PORT ?= 8000

# Color output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

# Validation target
validate:
	@echo "$(YELLOW)Validating environment...$(NC)"
	@python -c "import pm_automation_suite; print('✅ Package imports successfully')"
	@python -c "from dotenv import load_dotenv; load_dotenv(); print('✅ Environment variables loaded')"
	@echo "$(GREEN)Environment validation complete!$(NC)"