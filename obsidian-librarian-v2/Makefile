.PHONY: help build test clean install dev format lint all

help:
	@echo "Available commands:"
	@echo "  make build       - Build both Rust and Python components"
	@echo "  make test        - Run all tests"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make install     - Install the package"
	@echo "  make dev         - Install in development mode"
	@echo "  make format      - Format all code"
	@echo "  make lint        - Run linters"
	@echo "  make all         - Build, test, and install"

# Build targets
build: build-rust build-python

build-rust:
	@echo "Building Rust components..."
	cd rust-core && cargo build --release

build-python:
	@echo "Building Python package with maturin..."
	cd python && maturin build --release

# Test targets
test: test-rust test-python

test-rust:
	@echo "Running Rust tests..."
	cd rust-core && cargo test

test-python:
	@echo "Running Python tests..."
	cd python && pytest

test-integration:
	@echo "Running integration tests..."
	cd python && pytest -m integration

# Development
dev:
	@echo "Installing in development mode..."
	pip install maturin
	cd python && maturin develop --release
	cd python && pip install -e .[dev]

# Code quality
format: format-rust format-python

format-rust:
	@echo "Formatting Rust code..."
	cd rust-core && cargo fmt

format-python:
	@echo "Formatting Python code..."
	cd python && black obsidian_librarian/
	cd python && isort obsidian_librarian/

lint: lint-rust lint-python

lint-rust:
	@echo "Linting Rust code..."
	cd rust-core && cargo clippy -- -D warnings

lint-python:
	@echo "Linting Python code..."
	cd python && flake8 obsidian_librarian/
	cd python && mypy obsidian_librarian/

# Benchmarks
bench:
	@echo "Running Rust benchmarks..."
	cd rust-core && cargo bench

# Clean
clean:
	@echo "Cleaning build artifacts..."
	cd rust-core && cargo clean
	cd python && rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Install
install: build
	@echo "Installing package..."
	cd python && pip install .

# Full pipeline
all: clean format lint build test install
	@echo "Build complete!"