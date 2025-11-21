"""
WebRTC-based depth streaming server for Runpod
Uses WebRTC for efficient video streaming instead of base64 JPEG frames
"""

import json
import asyncio
import numpy as np
import torch
import msgpack
from io import BytesIO
from PIL import Image
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn
from depth_anything_3.api import DepthAnything3

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole
from av import VideoFrame


class DepthProcessor(VideoStreamTrack):
    """
    Custom video track that processes incoming frames with depth estimation
    """
    def __init__(self, track, depth_server, websocket):
        super().__init__()
        self.track = track
        self.depth_server = depth_server
        self.websocket = websocket
        self.frame_count = 0
        self.is_processing = False
        self.last_frame_time = datetime.now()

    async def recv(self):
        """Receive frame from client - just pass through, don't process here"""
        frame = await self.track.recv()
        return frame

    async def process_frame_async(self, img):
        """Process a single frame asynchronously"""
        try:
            # Process with depth estimation (blocking, but in separate task)
            process_start = datetime.now()
            points, colors, intrinsics = await asyncio.to_thread(
                self.depth_server.process_frame_to_pointcloud, img
            )
            process_time = (datetime.now() - process_start).total_seconds() * 1000

            # Pack and send point cloud via WebSocket
            pack_start = datetime.now()
            response_data = {
                "type": "pointcloud",
                "timestamp": datetime.now().isoformat(),
                "num_points": len(points),
                "points": points.astype(np.float32).tobytes(),
                "colors": colors.astype(np.uint8).tobytes(),
            }
            binary_response = msgpack.packb(response_data, use_bin_type=True)
            pack_time = (datetime.now() - pack_start).total_seconds() * 1000

            send_start = datetime.now()
            await self.websocket.send_bytes(binary_response)
            send_time = (datetime.now() - send_start).total_seconds() * 1000

            # Calculate actual FPS
            now = datetime.now()
            actual_fps = 1.0 / (now - self.last_frame_time).total_seconds() if self.frame_count > 1 else 0
            self.last_frame_time = now

            print(
                f"📊 Frame {self.frame_count} ({actual_fps:.1f} FPS): {len(points)} pts ({len(binary_response)/1024:.1f}KB) | "
                f"Process: {process_time:.1f}ms | Pack: {pack_time:.1f}ms | Send: {send_time:.1f}ms"
            )

        finally:
            self.is_processing = False

    async def process_frames(self):
        """Actively consume and process frames with skipping"""
        print(f"🎬 Starting process_frames loop")
        skipped = 0
        frames_received = 0

        while True:
            try:
                # Always consume frames to prevent queue buildup
                # Add timeout to prevent indefinite hanging
                if frames_received == 0:
                    print(f"⏳ Waiting for first frame from track...")

                frame = await asyncio.wait_for(self.track.recv(), timeout=30.0)

                if frames_received == 0:
                    print(f"✅ First frame received! Stream is flowing.")

                frames_received += 1

                # Skip if still processing
                if self.is_processing:
                    skipped += 1
                    if skipped % 30 == 0:
                        print(f"⏭️  Skipped {skipped} frames (total received: {frames_received})")
                    continue

                skipped = 0
                self.is_processing = True
                self.frame_count += 1

                # Convert av.VideoFrame to numpy array
                img = frame.to_ndarray(format="rgb24")

                # Process in background (doesn't block frame consumption)
                asyncio.create_task(self.process_frame_async(img))

            except asyncio.TimeoutError:
                print(f"⏱️  Timeout waiting for frame (waited 30s, received {frames_received} frames total)")
                print(f"💡 This usually means ICE connection failed to establish media path")
                break
            except Exception as e:
                print(f"❌ Error in frame loop: {e}")
                import traceback
                traceback.print_exc()
                break


class CloudDepthServer:
    def __init__(self, model_size="small", device="cuda"):
        """Initialize the cloud depth estimation server"""
        print(f"🚀 Initializing DepthAnything v3 model ({model_size})...")

        # Model mapping
        model_map = {
            "small": "depth-anything/da3-small",
            "base": "depth-anything/da3-base",
        }

        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"🖥️  Using device: {self.device}")

        self.model = DepthAnything3.from_pretrained(model_map.get(model_size, model_map["small"]))
        self.model = self.model.to(device=self.device)

        print("✅ Model loaded successfully!")

        self.clients = {}  # client_id -> RTCPeerConnection
        self.pcs = set()  # All peer connections

    def process_frame_to_pointcloud(self, rgb_image):
        """Process a single frame to generate point cloud"""
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(rgb_image)

        # Run DepthAnything v3 inference
        prediction = self.model.inference(
            [pil_image],
            export_dir=None,
            export_format=[],
        )

        # Get depth map and intrinsics
        depth_map = prediction.depth[0]
        if torch.is_tensor(depth_map):
            depth_map = depth_map.cpu().numpy()

        confidence = None
        if hasattr(prediction, "conf") and prediction.conf is not None:
            conf = prediction.conf[0]
            confidence = conf.cpu().numpy() if torch.is_tensor(conf) else conf

        intrinsics = prediction.intrinsics[0]
        if torch.is_tensor(intrinsics):
            intrinsics = intrinsics.cpu().numpy()

        # Extract camera parameters
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Generate point cloud
        h, w = depth_map.shape

        # Resize RGB image to match depth map if needed
        if rgb_image.shape[0] != h or rgb_image.shape[1] != w:
            rgb_image = np.array(pil_image.resize((w, h), Image.BILINEAR))

        downsample = 3  # Adjust for performance (higher = fewer points, faster)

        # Create mesh grid
        xx, yy = np.meshgrid(np.arange(0, w, downsample), np.arange(0, h, downsample))

        # Get depth values
        z = depth_map[yy, xx]

        # Filter by confidence if available
        if confidence is not None:
            conf_threshold = 0.5
            conf_values = confidence[yy, xx]
            valid_mask = conf_values > conf_threshold
            xx = xx[valid_mask]
            yy = yy[valid_mask]
            z = z[valid_mask]

        # Calculate 3D coordinates
        x = (xx - cx) * z / fx
        y = -((yy - cy) * z / fy)  # Negate Y to flip vertically

        # Get colors
        yy_int = yy.astype(int)
        xx_int = xx.astype(int)

        if xx_int.ndim == 1:
            colors = rgb_image[yy_int, xx_int]
        else:
            colors = rgb_image[yy_int.flatten(), xx_int.flatten()]

        # Stack into point cloud
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
        colors = colors.reshape(-1, 3)

        return points, colors, intrinsics


# Create FastAPI app
app = FastAPI(title="Depth Anything v3 WebRTC Streaming Server")

# Global server instance
depth_server = None


@app.get("/")
async def serve_viewer():
    """Serve the viewer.html file"""
    return FileResponse("viewer.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for WebRTC signaling and point cloud data"""
    await websocket.accept()
    client_id = id(websocket)
    print(f"👤 Client {client_id} connected")

    # Configure STUN servers for NAT traversal
    pc = RTCPeerConnection(configuration={
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"},
        ]
    })
    depth_server.pcs.add(pc)

    # Monitor ICE connection state
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"🔌 ICE connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
            print(f"❌ ICE connection failed - check firewall/NAT settings")
        elif pc.connectionState == "connected":
            print(f"✅ Media connection established!")

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print(f"🧊 ICE state: {pc.iceConnectionState}")

    @pc.on("track")
    async def on_track(track):
        print(f"📹 Track received: {track.kind}")
        if track.kind == "video":
            # Create depth processor that wraps the video track
            depth_processor = DepthProcessor(track, depth_server, websocket)

            # Start processing frames (with built-in skipping)
            asyncio.create_task(depth_processor.process_frames())

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data["type"] == "offer":
                # Receive WebRTC offer from client
                offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                await pc.setRemoteDescription(offer)

                # Create answer
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

                # Send answer back to client
                await websocket.send_text(json.dumps({
                    "type": pc.localDescription.type,
                    "sdp": pc.localDescription.sdp
                }))
                print(f"✅ WebRTC connection established with client {client_id}")

            elif data["type"] == "ice":
                # Handle ICE candidates
                candidate_dict = data["candidate"]
                if candidate_dict:
                    candidate = RTCIceCandidate(
                        component=candidate_dict.get("component", 1),
                        foundation=candidate_dict.get("foundation", ""),
                        ip=candidate_dict.get("address", candidate_dict.get("ip", "")),
                        port=candidate_dict.get("port", 0),
                        priority=candidate_dict.get("priority", 0),
                        protocol=candidate_dict.get("protocol", "udp"),
                        type=candidate_dict.get("type", "host"),
                        sdpMid=candidate_dict.get("sdpMid"),
                        sdpMLineIndex=candidate_dict.get("sdpMLineIndex")
                    )
                    await pc.addIceCandidate(candidate)

    except WebSocketDisconnect:
        print(f"👤 Client {client_id} disconnected")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await pc.close()
        depth_server.pcs.discard(pc)
        print(f"👤 Total connections: {len(depth_server.pcs)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WebRTC depth streaming server")
    parser.add_argument(
        "--model",
        choices=["small", "base"],
        default="small",
        help="DepthAnything model size",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args()

    # Initialize depth server
    depth_server = CloudDepthServer(model_size=args.model, device=args.device)

    print(f"🌐 Starting WebRTC server on {args.host}:{args.port}")
    print(f"📄 Viewer available at: http://{args.host}:{args.port}/")

    # Start FastAPI server
    uvicorn.run(app, host=args.host, port=args.port)
