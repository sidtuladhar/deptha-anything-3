#!/usr/bin/env bash
set -e

echo "🚀 Depth Anything V3 Streaming Server"
echo "======================================"

# Optional noninteractive HF login
if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "🔑 Logging into Hugging Face..."
  huggingface-cli login --token "${HUGGING_FACE_HUB_TOKEN}" >/dev/null 2>&1 || true
fi

# Change to app directory
cd /opt/depth-streaming/app

# Check CUDA availability
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__}'); print(f'🖥️  CUDA available: {torch.cuda.is_available()}'); print(f'📊 GPU count: {torch.cuda.device_count()}'); print(f'🎯 Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}'); print(f'💾 GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Default model and device from environment variables
MODEL=${MODEL:-base}
DEVICE=${DEVICE:-cuda}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Device: $DEVICE"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo ""

# Run the streaming server
exec python3 stream_server.py \
  --model "$MODEL" \
  --device "$DEVICE" \
  --host "$HOST" \
  --port "$PORT"
