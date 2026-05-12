#!/usr/bin/env bash
# =============================================================================
# Reality Intelligence Platform – Development Setup Script
# =============================================================================
# Usage: bash scripts/setup_dev.sh
# =============================================================================

set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[OK]${RESET}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
error()   { echo -e "${RED}[ERR]${RESET}  $*"; exit 1; }

# ── Check prerequisites ───────────────────────────────────────────────────────
info "Checking prerequisites..."

command -v docker   >/dev/null 2>&1 || error "Docker not found. Install from https://docs.docker.com/get-docker/"
command -v python3  >/dev/null 2>&1 || error "Python 3 not found. Install Python 3.11+"
command -v node     >/dev/null 2>&1 || error "Node.js not found. Install Node 20+"
command -v git      >/dev/null 2>&1 || error "Git not found."

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
NODE_VERSION=$(node --version | cut -c2- | cut -d'.' -f1)

[[ "$(echo "$PYTHON_VERSION >= 3.11" | bc -l)" == "1" ]] || \
  warn "Python $PYTHON_VERSION detected. Python 3.11+ recommended."
[[ "$NODE_VERSION" -ge 18 ]] || \
  warn "Node $NODE_VERSION detected. Node 20+ recommended."

success "Prerequisites OK"

# ── Environment file ──────────────────────────────────────────────────────────
info "Setting up environment..."
if [[ ! -f .env ]]; then
    cp .env.example .env
    # Generate a random secret key
    SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    sed -i.bak "s/change-me-to-a-random-64-char-hex-string-in-production/$SECRET/" .env
    rm -f .env.bak
    success ".env created with random SECRET_KEY"
else
    warn ".env already exists – skipping"
fi

# ── Python virtual environment ────────────────────────────────────────────────
info "Setting up Python virtual environment..."
if [[ ! -d backend/venv ]]; then
    python3 -m venv backend/venv
    source backend/venv/bin/activate
    pip install --upgrade pip wheel setuptools --quiet
    pip install -r backend/requirements.txt --quiet
    success "Python venv created and deps installed"
else
    warn "backend/venv already exists – skipping"
fi

# ── Node dependencies ─────────────────────────────────────────────────────────
info "Installing frontend dependencies..."
cd frontend && npm install --prefer-offline --silent && cd ..
success "Frontend deps installed"

# ── Pre-commit hooks ──────────────────────────────────────────────────────────
info "Installing pre-commit hooks..."
source backend/venv/bin/activate
pip install pre-commit --quiet
pre-commit install
success "Pre-commit hooks installed"

# ── Docker services ───────────────────────────────────────────────────────────
info "Starting infrastructure services (postgres, redis, minio)..."
docker compose up -d postgres redis minio minio-init
sleep 5

# ── Database migrations ───────────────────────────────────────────────────────
info "Running database migrations..."
source backend/venv/bin/activate
cd backend
PYTHONPATH=. alembic upgrade head
cd ..
success "Migrations applied"

# ── Seed data ─────────────────────────────────────────────────────────────────
info "Seeding development data..."
source backend/venv/bin/activate
cd backend && PYTHONPATH=. python ../scripts/seed_data.py && cd ..
success "Database seeded"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}║         Reality Intelligence Platform – Ready! 🚀           ║${RESET}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo "  Start all services:   make up"
echo "  Backend only:         cd backend && uvicorn app.main:app --reload"
echo "  Frontend only:        cd frontend && npm run dev"
echo ""
echo "  Dashboard:            http://localhost:3000"
echo "  API docs:             http://localhost:8000/docs"
echo "  MinIO console:        http://localhost:9001"
echo "  Flower (Celery):      http://localhost:5555"
echo ""
echo "  Demo login:"
echo "    Email:    demo@reality-intelligence.io"
echo "    Password: Demo2024!"
echo ""
