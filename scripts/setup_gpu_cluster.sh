#!/usr/bin/env bash
# =============================================================================
# Reality Intelligence Platform – GPU Cluster Setup Script
# =============================================================================
# Installs NVIDIA drivers, Container Toolkit, and downloads ML model weights.
# Run on each GPU worker node.
# =============================================================================

set -euo pipefail

info()    { echo -e "\033[0;36m[INFO]\033[0m $*"; }
success() { echo -e "\033[0;32m[OK]\033[0m   $*"; }
warn()    { echo -e "\033[1;33m[WARN]\033[0m $*"; }

# ── Detect NVIDIA GPU ─────────────────────────────────────────────────────────
info "Detecting GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    warn "nvidia-smi not found. Installing NVIDIA drivers..."

    # Ubuntu 22.04
    sudo apt-get update
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    sudo reboot
fi

nvidia-smi
success "GPU detected"

# ── NVIDIA Container Toolkit ──────────────────────────────────────────────────
info "Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release && echo "$ID$VERSION_ID")
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -sL "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
success "NVIDIA Container Toolkit installed"

# ── COLMAP ────────────────────────────────────────────────────────────────────
info "Installing COLMAP..."
sudo apt-get install -y colmap
colmap --version
success "COLMAP installed"

# ── Model weights directory ───────────────────────────────────────────────────
MODEL_DIR="${MODEL_DIR:-/models}"
info "Setting up model weights directory: $MODEL_DIR"
sudo mkdir -p "$MODEL_DIR"
sudo chown "$USER:$USER" "$MODEL_DIR"

# ── YOLOv8 weights ────────────────────────────────────────────────────────────
info "Downloading YOLOv8-L base weights..."
if [[ ! -f "$MODEL_DIR/yolov8l.pt" ]]; then
    pip install ultralytics --quiet
    python3 -c "from ultralytics import YOLO; YOLO('yolov8l.pt')" 2>/dev/null || true
    cp ~/.config/Ultralytics/yolov8l.pt "$MODEL_DIR/" 2>/dev/null \
      || wget -q -O "$MODEL_DIR/yolov8l.pt" \
         "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt"
    success "YOLOv8-L weights downloaded"
else
    warn "YOLOv8-L weights already exist"
fi

# ── SAM weights ───────────────────────────────────────────────────────────────
info "Downloading SAM ViT-H weights (2.5GB)..."
SAM_PATH="$MODEL_DIR/sam_vit_h_4b8939.pth"
if [[ ! -f "$SAM_PATH" ]]; then
    wget -q --show-progress \
         -O "$SAM_PATH" \
         "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    success "SAM ViT-H weights downloaded"
else
    warn "SAM weights already exist"
fi

# ── COLMAP Vocabulary Tree ────────────────────────────────────────────────────
info "Downloading COLMAP vocabulary tree (300MB)..."
VOCAB_PATH="$MODEL_DIR/vocab_tree_flickr100K_words32K.bin"
if [[ ! -f "$VOCAB_PATH" ]]; then
    wget -q --show-progress \
         -O "$VOCAB_PATH" \
         "https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin"
    export COLMAP_VOCAB_TREE="$VOCAB_PATH"
    success "Vocabulary tree downloaded"
else
    warn "Vocabulary tree already exists"
fi

# ── Test GPU Docker ───────────────────────────────────────────────────────────
info "Testing GPU Docker integration..."
docker run --rm --gpus all nvcr.io/nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
success "GPU Docker working"

# ── Print summary ─────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           GPU Cluster Setup Complete! ✅                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Models at:    $MODEL_DIR"
echo "  GPU status:   $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""
echo "  Set env vars:"
echo "    export MODEL_DIR=$MODEL_DIR"
echo "    export COLMAP_VOCAB_TREE=$VOCAB_PATH"
echo ""
