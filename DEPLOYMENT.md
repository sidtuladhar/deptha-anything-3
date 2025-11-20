# Deployment Guide for Runpod

This guide explains how to deploy the Depth Anything v3 cloud streaming service to Runpod.

## Architecture

- **Server** (`stream_server.py`): Runs on Runpod GPU, receives webcam frames from users, processes depth, sends back point clouds
- **Client** (`viewer_cloud.html`): Web page users visit, captures their webcam, sends frames to server, displays 3D point cloud

## Quick Start

### 1. Build Docker Image

```bash
docker build -t depth-streaming:latest .
```

### 2. Test Locally (Optional)

```bash
# Run with GPU (if available)
docker run --gpus all -p 8765:8765 depth-streaming:latest

# Or with CPU
docker run -p 8765:8765 depth-streaming:latest python stream_server.py --device cpu
```

Then open `viewer_cloud.html` in your browser and click "Start Webcam".

### 3. Deploy to Runpod

#### Option A: Using Runpod Docker Template

1. Push your image to Docker Hub:
```bash
docker tag depth-streaming:latest your-dockerhub-username/depth-streaming:latest
docker push your-dockerhub-username/depth-streaming:latest
```

2. In Runpod:
   - Go to "Templates" → "New Template"
   - Docker Image: `your-dockerhub-username/depth-streaming:latest`
   - Expose HTTP Ports: `8765`
   - GPU: Select any (RTX 3090, A40, etc.)

3. Deploy a pod with this template

#### Option B: Using Runpod Dockerfile

1. Upload your code to GitHub
2. In Runpod, create a new pod with:
   - Container Image: `nvidia/cuda:12.1.0-runtime-ubuntu22.04`
   - Container Start Command: Clone your repo and run the server
   - Expose port 8765

### 4. Access Your Service

1. Get your pod's public URL from Runpod (e.g., `https://abc123-8765.proxy.runpod.net`)

2. Update `viewer_cloud.html` line 25 with your Runpod URL:
```javascript
const WS_URL = 'wss://abc123-8765.proxy.runpod.net';
```

3. Host `viewer_cloud.html` somewhere (GitHub Pages, Netlify, Vercel, etc.) or open it locally

4. Users visit the page, grant webcam access, and see their depth point cloud!

## Configuration

### Server Options

```bash
python stream_server.py --help

Options:
  --model {small,base}    Model size (default: small)
  --device {cuda,mps,cpu} Device (default: cuda)
  --host HOST             Bind address (default: 0.0.0.0)
  --port PORT             Port (default: 8765)
```

### Performance Tuning

**For faster processing:**
- Use `--model small` (default, fastest)
- Ensure GPU is being used (check logs for "Using device: cuda")
- Lower webcam resolution in `viewer_cloud.html` (line 284)

**For higher quality:**
- Use `--model base` (slower but better quality)
- Increase resolution in viewer (but will be slower)

## Troubleshooting

### WebSocket Connection Fails

1. Check Runpod exposes port 8765
2. Update viewer with correct WSS URL (note: `wss://` not `ws://` for HTTPS)
3. Check firewall settings

### Slow Performance

1. Verify GPU is being used: Check docker logs for "Using device: cuda"
2. Use smaller model: `--model small`
3. Reduce frame sending rate in viewer (line 387, increase timeout)

### Out of Memory

- Use smaller model: `--model small`
- Increase downsampling in `stream_server.py` line 86
- Choose Runpod instance with more VRAM

## Cost Optimization

- Use **on-demand** pods for testing
- Use **spot instances** for production (cheaper, may be interrupted)
- Stop pods when not in use
- Use smaller GPU (RTX 3090 Ti is cheaper than A100)

## Monitoring

Check docker logs:
```bash
docker logs <container-id>
```

You should see:
- "✅ Model loaded successfully!"
- "🌐 Starting WebSocket server on 0.0.0.0:8765"
- "👤 Client XXX connected" when users connect
- "📡 Processed frame..." for each frame

## Multiple Users

The server supports multiple concurrent users. Each user:
- Has their own WebSocket connection
- Sends their own webcam frames
- Receives their own point clouds

GPU will process frames from all users. Monitor GPU utilization and scale up if needed.
