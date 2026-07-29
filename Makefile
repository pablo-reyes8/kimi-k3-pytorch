PYTHON ?= python3
PROFILE ?= config/kimi_full_pipeline/cpu_smoke

.DEFAULT_GOAL := help

.PHONY: help install install-dev validate check test test-config test-inference \
	test-training docker-build docker-validate docker-tests docker-shell

help: ## Show the available developer commands.
	@awk 'BEGIN {FS = ":.*## "; printf "Kimi-K3 Mini targets:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the project in editable mode.
	$(PYTHON) -m pip install -e .

install-dev: ## Install tests and optional dataset dependencies.
	$(PYTHON) -m pip install -e ".[dev,data]"

validate: ## Validate PROFILE without allocating a model or training.
	$(PYTHON) -m scripts.train_kimi --profile $(PROFILE) --validate-only

check: ## Run the fast public-contract checks used during iteration.
	$(PYTHON) -m compileall -q configuration data inference scripts src training
	$(PYTHON) -m pytest tests/configuration tests/inference

test: ## Run the complete CPU-safe test suite.
	$(PYTHON) -m pytest

test-config: ## Validate every complete YAML profile and CLI contract.
	$(PYTHON) -m pytest tests/configuration

test-inference: ## Test sampling, checkpoint loading and native cache decode.
	$(PYTHON) -m pytest tests/inference

test-training: ## Run the modular training-engine tests.
	$(PYTHON) -m pytest tests/training

docker-build: ## Build the non-root CPU runtime image.
	docker build --target runtime -t kimi-k3-mini:local .

docker-validate: ## Validate the CPU profile through Docker Compose.
	docker compose run --rm validate

docker-tests: ## Run focused configuration/inference tests in Compose.
	docker compose --profile test run --rm tests

docker-shell: ## Open a development shell with the repository mounted.
	docker compose --profile dev run --rm shell
