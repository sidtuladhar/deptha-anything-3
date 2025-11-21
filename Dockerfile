# CUDA 12.1 runtime (smaller than devel, no build tools)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps (added libavformat, libavcodec for aiortc/av)
RUN apt-get update && apt-get install -y \
  git git-lfs curl wget ca-certificates openssh-client \
  python3 python3-pip python3-venv python3-dev \
  ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev \
  libgomp1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
  libopus-dev libvpx-dev pkg-config \
  && rm -rf /var/lib/apt/lists/* \
  && git lfs install \
  && ln -s /usr/bin/python3 /usr/bin/python

# Python + Torch pinned for CUDA 12.1
RUN python3 -m pip install --upgrade pip
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# Enable fast transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV MKL_THREADING_LAYER=GNU

# Cache dirs
ENV HF_HOME=/opt/depth-streaming/cache
ENV TORCH_HOME=/opt/depth-streaming/cache
RUN mkdir -p /opt/depth-streaming/cache

# Create app directory
RUN mkdir -p /opt/depth-streaming/app
WORKDIR /opt/depth-streaming/app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt || \
  (grep -v "xformers" requirements.txt > requirements_slim.txt && \
  pip3 install --no-cache-dir -r requirements_slim.txt)

# Copy project files
COPY src/ ./src/
COPY stream_server.py .
COPY viewer.html .
COPY viewer-multicam.html .

# Add src to Python path so imports work
ENV PYTHONPATH=/opt/depth-streaming/app/src:$PYTHONPATH

# Copy entrypoint script
COPY entrypoint.sh /opt/depth-streaming/entrypoint.sh

# --- FIX WINDOWS CRLF ISSUE ---
RUN sed -i 's/\r$//' /opt/depth-streaming/*.sh && chmod +x /opt/depth-streaming/*.sh
# --- END FIX ---

# Default folders for RunPod mapping
RUN mkdir -p /input /output

# Expose HTTP + WebSocket port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import torch; import websockets" || exit 1

ENTRYPOINT ["/opt/depth-streaming/entrypoint.sh"]
