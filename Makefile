# =============================================================================
# Reality Intelligence Platform - Makefile
# =============================================================================
# Usage: make <target>
# Run `make help` to see all available targets.

.PHONY: help build up down restart logs shell test lint format migrate seed \
        clean build-gpu up-gpu install-nvidia-toolkit docs k8s-apply \
        k8s-delete train-detection train-segmentation run-reconstruction \
        export-models generate-certs backup db-setup

# Default target
.DEFAULT_GOAL := help

# Colors for output
CYAN  := \033[0;36m
GREEN := \033[0;32m
RED   := \033[0;31m
RESET := \033[0m

# Environment
ENV ?= development
COMPOSE_FILE := docker-compose.yml
GPU_COMPOSE_FILE := docker-compose.gpu.yml

# Project name
PROJECT_NAME := rip

# =============================================================================
# HELP
# =============================================================================
help: ## Show this help message
	@echo ""
	@echo "$(CYAN)Reality Intelligence Platform - Available Commands$(RESET)"
	@echo "================================================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-30s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# DOCKER COMMANDS
# =============================================================================
build: ## Build all Docker images (CPU)
	@echo "$(CYAN)Building Docker images...$(RESET)"
	docker compose -f $(COMPOSE_FILE) build --parallel

build-gpu: ## Build GPU-enabled Docker images
	@echo "$(CYAN)Building GPU Docker images...$(RESET)"
	docker compose -f $(COMPOSE_FILE) -f $(GPU_COMPOSE_FILE) build --parallel

up: ## Start all services (CPU)
	@echo "$(CYAN)Starting services...$(RESET)"
	docker compose -f $(COMPOSE_FILE) up -d
	@echo "$(GREEN)Services started! Dashboard: http://localhost:3000 | API: http://localhost:8000/docs$(RESET)"

up-gpu: ## Start all services with GPU support
	@echo "$(CYAN)Starting GPU services...$(RESET)"
	docker compose -f $(COMPOSE_FILE) -f $(GPU_COMPOSE_FILE) up -d

down: ## Stop all services
	@echo "$(CYAN)Stopping services...$(RESET)"
	docker compose -f $(COMPOSE_FILE) down

restart: ## Restart all services
	$(MAKE) down
	$(MAKE) up

logs: ## View logs from all services
	docker compose -f $(COMPOSE_FILE) logs -f

logs-backend: ## View backend logs
	docker compose logs -f backend

logs-ml: ## View ML worker logs
	docker compose logs -f ml-worker

logs-celery: ## View Celery worker logs
	docker compose logs -f celery-worker

shell: ## Open shell in backend container
	docker compose exec backend bash

shell-ml: ## Open shell in ML worker container
	docker compose exec ml-worker bash

ps: ## Show running containers
	docker compose ps

# =============================================================================
# DEVELOPMENT
# =============================================================================
install: ## Install development dependencies
	@echo "$(CYAN)Installing dependencies...$(RESET)"
	pip install -r backend/requirements.txt -r backend/requirements-dev.txt
	cd frontend && npm install
	pre-commit install
	@echo "$(GREEN)Dependencies installed!$(RESET)"

install-nvidia-toolkit: ## Install NVIDIA Container Toolkit
	@echo "$(CYAN)Installing NVIDIA Container Toolkit...$(RESET)"
	distribution=$$(. /etc/os-release;echo $$ID$$VERSION_ID) \
		&& curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
		&& curl -s -L https://nvidia.github.io/libnvidia-container/$$distribution/libnvidia-container.list | \
			sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
			sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
	sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
	sudo nvidia-ctk runtime configure --runtime=docker
	sudo systemctl restart docker

# =============================================================================
# DATABASE
# =============================================================================
db-setup: ## Setup PostgreSQL database
	@echo "$(CYAN)Setting up database...$(RESET)"
	docker compose exec postgres psql -U postgres -c "CREATE DATABASE rip_db;" 2>/dev/null || true
	docker compose exec postgres psql -U postgres -c "CREATE DATABASE rip_test_db;" 2>/dev/null || true

migrate: ## Run database migrations
	@echo "$(CYAN)Running migrations...$(RESET)"
	docker compose exec backend alembic upgrade head

migrate-create: ## Create new migration (usage: make migrate-create MSG="description")
	docker compose exec backend alembic revision --autogenerate -m "$(MSG)"

migrate-down: ## Rollback last migration
	docker compose exec backend alembic downgrade -1

seed: ## Seed database with sample data
	@echo "$(CYAN)Seeding database...$(RESET)"
	docker compose exec backend python scripts/seed_data.py

# =============================================================================
# TESTING
# =============================================================================
test: ## Run all tests
	@echo "$(CYAN)Running tests...$(RESET)"
	docker compose exec backend pytest tests/ -v --tb=short

test-unit: ## Run unit tests
	pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests
	pytest tests/integration/ -v --tb=short

test-ml: ## Run ML pipeline tests
	pytest tests/ml_pipeline/ -v --tb=short

test-api: ## Run API tests
	pytest tests/integration/api/ -v --tb=short

test-coverage: ## Run tests with coverage
	pytest --cov=app --cov-report=html --cov-report=term-missing tests/
	open htmlcov/index.html

test-load: ## Run load tests with locust
	locust -f tests/e2e/locustfile.py --host=http://localhost:8000

# =============================================================================
# CODE QUALITY
# =============================================================================
lint: ## Run all linters
	@echo "$(CYAN)Running linters...$(RESET)"
	ruff check backend/app/
	mypy backend/app/ --ignore-missing-imports
	cd frontend && npm run lint

format: ## Format code
	@echo "$(CYAN)Formatting code...$(RESET)"
	black backend/app/ tests/
	ruff check --fix backend/app/
	cd frontend && npm run format

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

type-check: ## Run type checking
	mypy backend/app/ --ignore-missing-imports --strict

# =============================================================================
# ML TRAINING
# =============================================================================
train-detection: ## Train object detection model
	@echo "$(CYAN)Training YOLOv8 detection model...$(RESET)"
	docker compose exec ml-worker python ml/detection/training/train_yolov8.py \
		--config configs/model/yolov8_construction.yaml \
		--device cuda:0

train-segmentation: ## Train segmentation model
	@echo "$(CYAN)Training segmentation model...$(RESET)"
	docker compose exec ml-worker python ml/segmentation/training/train_deeplabv3.py \
		--config configs/model/deeplabv3_construction.yaml

train-analytics: ## Train delay prediction model
	@echo "$(CYAN)Training delay prediction model...$(RESET)"
	docker compose exec ml-worker python ml/analytics/delay_prediction/train.py

export-models: ## Export models to ONNX and TensorRT
	@echo "$(CYAN)Exporting models...$(RESET)"
	docker compose exec ml-worker python ml/deployment/onnx/export.py
	docker compose exec ml-worker python ml/deployment/tensorrt/optimize.py

# =============================================================================
# 3D RECONSTRUCTION
# =============================================================================
run-reconstruction: ## Run 3D reconstruction pipeline (usage: make run-reconstruction SITE=site_001)
	@echo "$(CYAN)Running reconstruction for site: $(SITE)$(RESET)"
	docker compose exec ml-worker python ml/reconstruction/sfm/run_colmap_sfm.py \
		--image-dir data/sites/$(SITE)/frames \
		--output-dir outputs/$(SITE)/sparse

run-mvs: ## Run Multi-View Stereo (usage: make run-mvs SITE=site_001)
	@echo "$(CYAN)Running MVS for site: $(SITE)$(RESET)"
	docker compose exec ml-worker python ml/reconstruction/mvs/run_colmap_mvs.py \
		--sparse-dir outputs/$(SITE)/sparse \
		--output-dir outputs/$(SITE)/dense

# =============================================================================
# KUBERNETES
# =============================================================================
k8s-apply: ## Apply Kubernetes manifests (production)
	@echo "$(CYAN)Deploying to Kubernetes...$(RESET)"
	kubectl apply -k kubernetes/overlays/production
	kubectl rollout status deployment/backend -n rip-production
	kubectl rollout status deployment/frontend -n rip-production
	kubectl rollout status deployment/ml-worker -n rip-production

k8s-delete: ## Delete Kubernetes resources
	kubectl delete -k kubernetes/overlays/production

k8s-status: ## Check Kubernetes deployment status
	kubectl get pods,services,ingress -n rip-production

k8s-logs: ## View Kubernetes logs (usage: make k8s-logs POD=backend)
	kubectl logs -f deployment/$(POD) -n rip-production

k8s-scale: ## Scale deployment (usage: make k8s-scale DEPLOY=ml-worker REPLICAS=5)
	kubectl scale deployment/$(DEPLOY) --replicas=$(REPLICAS) -n rip-production

helm-install: ## Install via Helm
	helm install rip ./kubernetes/helm/rip \
		--namespace rip-production \
		--create-namespace \
		--values kubernetes/helm/rip/values.production.yaml

helm-upgrade: ## Upgrade Helm release
	helm upgrade rip ./kubernetes/helm/rip \
		--namespace rip-production \
		--values kubernetes/helm/rip/values.production.yaml

helm-uninstall: ## Uninstall Helm release
	helm uninstall rip -n rip-production

# =============================================================================
# MONITORING
# =============================================================================
monitoring-up: ## Start monitoring stack
	docker compose -f docker-compose.monitoring.yml up -d
	@echo "$(GREEN)Grafana: http://localhost:3001 | Prometheus: http://localhost:9090$(RESET)"

mlflow-ui: ## Open MLflow UI
	mlflow ui --host 0.0.0.0 --port 5000 &
	open http://localhost:5000

airflow-up: ## Start Airflow
	docker compose -f docker-compose.airflow.yml up -d
	@echo "$(GREEN)Airflow: http://localhost:8080 (admin/admin)$(RESET)"

# =============================================================================
# STORAGE
# =============================================================================
minio-setup: ## Configure MinIO buckets
	docker compose exec minio mc alias set local http://localhost:9000 minioadmin minioadmin
	docker compose exec minio mc mb local/rip-media --ignore-existing
	docker compose exec minio mc mb local/rip-models --ignore-existing
	docker compose exec minio mc mb local/rip-outputs --ignore-existing

# =============================================================================
# SECURITY
# =============================================================================
generate-certs: ## Generate SSL certificates for development
	@echo "$(CYAN)Generating SSL certificates...$(RESET)"
	mkdir -p docker/nginx/certs
	openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
		-keyout docker/nginx/certs/server.key \
		-out docker/nginx/certs/server.crt \
		-subj "/C=US/ST=State/L=City/O=RIP/CN=localhost"

security-scan: ## Run security scan
	bandit -r backend/app/ -f json -o reports/security.json
	safety check -r backend/requirements.txt

# =============================================================================
# DOCUMENTATION
# =============================================================================
docs: ## Generate API documentation
	@echo "$(CYAN)Generating documentation...$(RESET)"
	docker compose exec backend python -c "import app.main; import json; print(json.dumps(app.main.app.openapi()))" > docs/api/openapi.json
	redoc-cli build docs/api/openapi.json -o docs/api/index.html

# =============================================================================
# UTILITIES
# =============================================================================
clean: ## Remove build artifacts and caches
	@echo "$(CYAN)Cleaning up...$(RESET)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	cd frontend && rm -rf node_modules/.cache build 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete!$(RESET)"

backup: ## Backup database
	@echo "$(CYAN)Creating database backup...$(RESET)"
	docker compose exec postgres pg_dump -U postgres rip_db | gzip > backups/rip_db_$(shell date +%Y%m%d_%H%M%S).sql.gz
	@echo "$(GREEN)Backup created!$(RESET)"

env-check: ## Validate environment configuration
	python scripts/validate_env.py

version: ## Show component versions
	@echo "$(CYAN)Component Versions:$(RESET)"
	@docker compose exec backend python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "Backend not running"
	@docker compose exec backend python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')" 2>/dev/null || echo "Backend not running"
	@node --version
	@python --version

.PHONY: all
