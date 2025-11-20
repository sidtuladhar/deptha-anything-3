#!/usr/bin/env python3
"""
Cloud-based depth streaming server for Runpod
Receives webcam frames from browser clients and sends back point clouds
"""

import asyncio
import websockets
import json
import base64
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from depth_anything_3.api import DepthAnything3
from datetime import datetime


class CloudDepthServer:
    def __init__(self, model_size="small", device="cuda"):
        """Initialize the cloud depth estimation server"""
        print(f"🚀 Initializing DepthAnything v3 model ({model_size})...")

        # Model mapping
        model_map = {
            "small": "depth-anything/da3-small",
            "base": "depth-anything/da3-base",
        }

        # Initialize device (prefer CUDA for Runpod, fallback to CPU)
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"🖥️  Using device: {self.device}")

        # Initialize DepthAnything v3
        self.model = DepthAnything3.from_pretrained(
            model_map.get(model_size, model_map["small"])
        )
        self.model = self.model.to(device=self.device)

        print("✅ Model loaded successfully!")

        # Track connected clients
        self.clients = set()

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

        downsample = 2  # Adjust for performance

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

    async def handle_client(self, websocket):
        """Handle WebSocket client connection"""
        client_id = id(websocket)
        self.clients.add(websocket)
        print(f"👤 Client {client_id} connected. Total clients: {len(self.clients)}")

        try:
            async for message in websocket:
                try:
                    # Parse incoming message
                    data = json.loads(message)

                    if data.get("type") == "frame":
                        # Decode base64 image
                        img_data = base64.b64decode(data["image"])
                        img = Image.open(BytesIO(img_data))
                        rgb_image = np.array(img)

                        # Process frame
                        points, colors, intrinsics = self.process_frame_to_pointcloud(
                            rgb_image
                        )

                        # Send point cloud back to client
                        response = json.dumps(
                            {
                                "type": "pointcloud",
                                "timestamp": datetime.now().isoformat(),
                                "num_points": len(points),
                                "points": points.tolist(),
                                "colors": colors.tolist(),
                            }
                        )

                        await websocket.send(response)

                        print(
                            f"📡 Processed frame for client {client_id}: {len(points)} points"
                        )

                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON from client {client_id}")
                except Exception as e:
                    print(f"❌ Error processing frame from client {client_id}: {e}")
                    import traceback

                    traceback.print_exc()

        except websockets.exceptions.ConnectionClosed:
            print(f"👤 Client {client_id} disconnected")
        finally:
            self.clients.remove(websocket)
            print(f"👤 Total clients: {len(self.clients)}")

    async def start_server(self, host="0.0.0.0", port=8765):
        """Start the WebSocket server"""
        print(f"🌐 Starting WebSocket server on {host}:{port}")
        async with websockets.serve(self.handle_client, host, port):
            print("✅ Server ready! Waiting for connections...")
            await asyncio.Future()  # Run forever


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cloud depth streaming server for Runpod"
    )
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
        "--host", default="0.0.0.0", help="Host to bind to (0.0.0.0 for all interfaces)"
    )
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")

    args = parser.parse_args()

    server = CloudDepthServer(model_size=args.model, device=args.device)
    asyncio.run(server.start_server(host=args.host, port=args.port))
