#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh — sync repo to VM, install Docker, build & run the study hub
#
# Usage:  ./deploy.sh
# Requires: ssh access via AI_EXAM_PREP host alias in ~/.ssh/config
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HOST="AI_EXAM_PREP"
REMOTE_DIR="/home/ubuntu/aau-ce2-ai"
IMAGE="studyhub"
CONTAINER="studyhub"
PORT=6767

echo "──────────────────────────────────────────"
echo " AAU CE2 AI Study Hub — Deploy"
echo " Target: ${HOST}  →  port ${PORT}"
echo "──────────────────────────────────────────"

# ── 1. Sync only the files Docker needs (skip .git, .venv, code/, tools/, etc.)
echo ""
echo "[1/3] Syncing files to VM..."
rsync -avz --progress \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='code/' \
  --exclude='tools/' \
  --exclude='webapp/dist/' \
  --exclude='webapp/node_modules/' \
  --exclude='exam_prep/slides/' \
  --exclude='exam_prep/exercises/*.pdf' \
  --exclude='exam_prep/exercises/*.ipynb' \
  --exclude='.DS_Store' \
  --exclude='*.pyc' \
  -e "ssh" \
  ./ "${HOST}:${REMOTE_DIR}/"

# ── 2. Install Docker on the VM (idempotent — skips if already installed)
echo ""
echo "[2/3] Ensuring Docker is installed on VM..."
ssh "${HOST}" bash <<'ENDSSH'
set -e
if command -v docker &>/dev/null; then
  echo "  Docker already installed: $(docker --version)"
  exit 0
fi
echo "  Installing Docker..."
sudo apt-get update -qq
sudo apt-get install -y -qq ca-certificates curl gnupg lsb-release
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -qq
sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker ubuntu
echo "  Docker installed: $(docker --version)"
ENDSSH

# ── 3. Build image and (re)start container
echo ""
echo "[3/3] Building image and starting container..."
ssh "${HOST}" bash <<ENDSSH
set -e
cd ${REMOTE_DIR}

# Stop + remove old container if running
if sudo docker ps -q --filter name=${CONTAINER} | grep -q .; then
  echo "  Stopping existing container..."
  sudo docker stop ${CONTAINER}
fi
if sudo docker ps -aq --filter name=${CONTAINER} | grep -q .; then
  sudo docker rm ${CONTAINER}
fi

# Build
echo "  Building Docker image '${IMAGE}'..."
sudo docker build -f webapp/Dockerfile -t ${IMAGE} .

# Run
echo "  Starting container on port ${PORT}..."
sudo docker run -d \
  --name ${CONTAINER} \
  -p ${PORT}:${PORT} \
  --restart unless-stopped \
  ${IMAGE}

echo ""
echo "  Container status:"
sudo docker ps --filter name=${CONTAINER} --format "  {{.Names}}  {{.Status}}  {{.Ports}}"
ENDSSH

echo ""
echo "──────────────────────────────────────────"
echo " Done!  http://10.92.1.193:${PORT}"
echo "──────────────────────────────────────────"
