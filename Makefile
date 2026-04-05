# Makefile for Python project management using uv

# Variables
VENV_DIR = .venv
PYTHON = python3
REQUIREMENTS = requirements.txt

.PHONY: env venv clean clean-caches test test-abstractbeam-gpu sync update help

help:
	@echo "Available commands:"
	@echo "  make env      - Create and activate virtual environment with dependencies"
	@echo "  make venv     - Create virtual environment only"
	@echo "  make sync     - Sync dependencies from requirements.txt"
	@echo "  make test     - Run tests"
	@echo "  make test-abstractbeam-gpu - Pytest AbstractBeam (incl. smoke CUDA si dispo)"
	@echo "  make clean-caches - Purge caches pip/uv (libère de la place, sans toucher .venv)"
	@echo "  make update   - Update all dependencies"
	@echo "  make clean    - Remove virtual environment and cache files"

env: venv sync
	@echo "$(GREEN)Virtual environment created and dependencies installed$(NC)"

venv:
	@echo "Creating virtual environment..."
	@uv venv $(VENV_DIR)
	@echo "$(GREEN)Virtual environment created at $(VENV_DIR)$(NC)"

sync:
	@echo "Installing dependencies..."
	@uv sync --group dev
	@echo "$(GREEN)Dependencies installed$(NC)"

test:
	@echo "Running tests..."
	@uv run pytest tests/
	@echo "$(GREEN)Tests completed$(NC)"

test-abstractbeam-gpu:
	@cd src/AbstractBeam && \
	if [ -x .venv/bin/python ]; then \
	  PYTHONPATH=. .venv/bin/python -m pytest crossbeam/tests/test_gpu_optional.py -q --tb=short; \
	else \
	  PYTHONPATH=. uv run --project ../.. --group dev --extra abstractbeam python -m pytest crossbeam/tests/test_gpu_optional.py -q --tb=short; \
	fi

clean-caches:
	@bash scripts/reclaim_disk.sh

update:
	@echo "Updating dependencies..."
	@uv pip install --upgrade -r $(REQUIREMENTS)
	@echo "$(GREEN)Dependencies updated$(NC)"

clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV_DIR)
	@rm -rf .pytest_cache
	@rm -rf __pycache__
	@rm -rf .ruff_cache
	@echo "$(GREEN)Cleanup complete$(NC)"
