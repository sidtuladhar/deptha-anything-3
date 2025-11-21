"""
WebSocket-based depth streaming server for Runpod
Uses WebSocket for video streaming (JPEG frames) - works on TCP-only Runpod network
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


class DepthProcessor:
    """
    Processes incoming JPEG frames with depth estimation
    """

    def __init__(self, depth_server, websocket):
        self.depth_server = depth_server
        self.websocket = websocket
        self.frame_count = 0
        self.is_processing = False
        self.last_frame_time = datetime.now()

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
            actual_fps = (
                1.0 / (now - self.last_frame_time).total_seconds() if self.frame_count > 1 else 0
            )
            self.last_frame_time = now

            print(
                f"📊 Frame {self.frame_count} ({actual_fps:.1f} FPS): {len(points)} pts ({len(binary_response)/1024:.1f}KB) | "
                f"Process: {process_time:.1f}ms | Pack: {pack_time:.1f}ms | Send: {send_time:.1f}ms"
            )

        finally:
            self.is_processing = False

    async def process_jpeg_frame(self, jpeg_bytes):
        """Process a JPEG frame received via WebSocket"""
        # Skip if still processing previous frame
        if self.is_processing:
            return  # Silently skip - client will send more frames

        try:
            self.is_processing = True
            self.frame_count += 1

            # Decode JPEG to numpy array
            pil_image = Image.open(BytesIO(jpeg_bytes))
            img = np.array(pil_image)

            # Ensure RGB format
            if img.shape[2] == 4:  # RGBA
                img = img[:, :, :3]

            # Process in background (doesn't block WebSocket)
            asyncio.create_task(self.process_frame_async(img))

        except Exception as e:
            print(f"❌ Error processing JPEG frame: {e}")
            import traceback

            traceback.print_exc()
            self.is_processing = False


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

        self.active_processors = {}  # client_id -> DepthProcessor

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

        downsample = 3  # Optimized for 30 FPS (fewer points, much faster)

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
app = FastAPI(title="Depth Anything v3 WebSocket Streaming Server")

# Global server instance
depth_server = None


@app.get("/")
async def serve_viewer():
    """Serve the viewer.html file"""
    return FileResponse("viewer.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for JPEG video frames and point cloud data"""
    await websocket.accept()
    client_id = id(websocket)
    print(f"👤 Client {client_id} connected via WebSocket")

    # Create depth processor for this client
    depth_processor = DepthProcessor(depth_server, websocket)
    depth_server.active_processors[client_id] = depth_processor

    try:
        print(f"🎬 Ready to receive JPEG frames from client {client_id}")
        frame_count = 0

        while True:
            # Receive message - could be binary (JPEG frame) or text (control)
            try:
                # Try to receive as bytes first (JPEG frames)
                jpeg_bytes = await websocket.receive_bytes()
                frame_count += 1

                if frame_count == 1:
                    print(f"✅ First frame received from client {client_id}!")

                # Process the JPEG frame
                await depth_processor.process_jpeg_frame(jpeg_bytes)

            except:
                # If not bytes, try text (control messages)
                try:
                    message = await websocket.receive_text()
                    data = json.loads(message)

                    # Handle control messages if needed
                    if data.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))

                except:
                    # Connection closed
                    break

    except WebSocketDisconnect:
        print(f"👤 Client {client_id} disconnected")
    except Exception as e:
        print(f"❌ Error with client {client_id}: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up
        if client_id in depth_server.active_processors:
            del depth_server.active_processors[client_id]
        print(f"👤 Total active clients: {len(depth_server.active_processors)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WebSocket depth streaming server")
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
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args()

    # Initialize depth server
    depth_server = CloudDepthServer(model_size=args.model, device=args.device)

    print(f"🌐 Starting WebSocket server on {args.host}:{args.port}")
    print(f"📄 Viewer available at: http://{args.host}:{args.port}/")

    # Start FastAPI server
    uvicorn.run(app, host=args.host, port=args.port)
