#!/usr/bin/env bash
# Provision a fresh Vultr Cloud GPU instance (Ubuntu 24.04) for Mul AI.
#
# Prereqs: pick a Vultr GPU plan with stock in your region —
#   L40S 48GB  → conductor up to ~14B bf16 (or ~32B AWQ/GPTQ quantised)
#   A100 80GB  → conductor up to ~32B bf16
# Vultr's "NVIDIA GPU" marketplace image ships drivers + CUDA preinstalled;
# on a plain Ubuntu image uncomment the driver install below.
#
# Usage (as root on the fresh box, from a clone of this repo):
#   CONDUCTOR_MODEL="Qwen/Qwen2.5-32B-Instruct" bash deploy/vultr/provision.sh
set -euo pipefail

CONDUCTOR_MODEL="${CONDUCTOR_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.92}"
APP_DIR="${APP_DIR:-/opt/mul-ai}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "==> apt basics"
apt-get update -y
apt-get install -y python3.12-venv python3-pip git

# echo "==> NVIDIA drivers + CUDA (skip if using Vultr's GPU-enabled image)"
# apt-get install -y ubuntu-drivers-common && ubuntu-drivers autoinstall && reboot

nvidia-smi >/dev/null || { echo "ERROR: no NVIDIA GPU visible (nvidia-smi failed)"; exit 1; }

echo "==> app install to ${APP_DIR}"
mkdir -p "$APP_DIR"
cp -r "$REPO_DIR"/. "$APP_DIR"/
cd "$APP_DIR"
python3.12 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install .
./.venv/bin/pip install vllm  # pulls torch/CUDA wheels — takes a while

echo "==> switch conductor to vllm serving"
sed -i 's/^  serving: mock/  serving: vllm/' mul.config.yaml

echo "==> systemd units"
cat > /etc/systemd/system/mul-vllm.service <<EOF
[Unit]
Description=Mul AI conductor (vLLM OpenAI-compatible server)
After=network-online.target

[Service]
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/.venv/bin/python -m vllm.entrypoints.openai.api_server \\
  --model ${CONDUCTOR_MODEL} \\
  --max-model-len ${MAX_MODEL_LEN} \\
  --gpu-memory-utilization ${GPU_UTIL} \\
  --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

cp "${REPO_DIR}/deploy/vultr/mul.service" /etc/systemd/system/mul.service
sed -i "s|__APP_DIR__|${APP_DIR}|g" /etc/systemd/system/mul.service

systemctl daemon-reload
systemctl enable --now mul-vllm   # first start downloads the model weights
systemctl enable --now mul

echo "==> done. Gateway on :8080 once vLLM finishes loading the model."
echo "    Watch:  journalctl -fu mul-vllm"
