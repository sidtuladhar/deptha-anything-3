#!/usr/bin/env python3
"""
Real-time point cloud streaming from webcam using DepthAnything v3
"""

import cv2
import numpy as np
import torch
import asyncio
import websockets
import json
from datetime import datetime
import threading
import queue
import tempfile
import os
from PIL import Image
from depth_anything_3.api import DepthAnything3


class DepthPointCloudStreamer:
    def __init__(self, model_size="small", device="cuda"):
        """Initialize the depth estimation and streaming pipeline"""
        print(f"🚀 Initializing DepthAnything v3 model ({model_size})...")

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

        # Initialize DepthAnything v3
        self.model = DepthAnything3.from_pretrained(model_map.get(model_size, model_map["small"]))
        self.model = self.model.to(device=self.device)

        # Streaming settings
        self.frame_queue = queue.Queue(maxsize=2)
        self.pointcloud_queue = queue.Queue(maxsize=2)
        self.clients = set()
        self.streaming = False

        # Create temp directory for exports
        self.temp_dir = tempfile.mkdtemp(prefix="depth_stream_")

        print("✅ Model loaded successfully!")
        print(f"📁 Temp directory: {self.temp_dir}")

    def process_frame_to_pointcloud(self, rgb_image):
        """Process a single frame to generate point cloud using DepthAnything v3"""
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(rgb_image)

        # Run DepthAnything v3 inference
        prediction = self.model.inference(
            [pil_image],
            export_dir=None,  # We'll process in memory
            export_format=[],  # We'll use raw outputs (empty list instead of None)
        )

        # Get depth map and confidence (handle both tensor and numpy array)
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

        # Extract camera parameters from intrinsics
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Generate point cloud with proper intrinsics
        h, w = depth_map.shape

        # Resize RGB image to match depth map dimensions if needed
        if rgb_image.shape[0] != h or rgb_image.shape[1] != w:
            from PIL import Image as PILImage

            rgb_image = np.array(PILImage.fromarray(rgb_image).resize((w, h), PILImage.BILINEAR))

        downsample = 2  # Adjust for performance (lower = more points)

        # Create mesh grid
        xx, yy = np.meshgrid(np.arange(0, w, downsample), np.arange(0, h, downsample))

        # Get depth values
        z = depth_map[yy, xx]

        # Filter by confidence if available
        if confidence is not None:
            conf_threshold = 0.5  # Adjust threshold as needed
            conf_values = confidence[yy, xx]
            valid_mask = conf_values > conf_threshold
            xx = xx[valid_mask]
            yy = yy[valid_mask]
            z = z[valid_mask]

        # Calculate 3D coordinates using proper intrinsics
        x = (xx - cx) * z / fx
        y = -((yy - cy) * z / fy)  # Negate Y to flip vertically

        # Get colors - sample from aligned RGB image
        # Ensure indices are integers
        yy_int = yy.astype(int)
        xx_int = xx.astype(int)

        # xx, yy are either 2D meshgrids or 1D arrays (after masking)
        if xx_int.ndim == 1:
            # After masking, xx and yy are 1D arrays
            colors = rgb_image[yy_int, xx_int]
        else:
            # 2D meshgrids - need to flatten
            colors = rgb_image[yy_int.flatten(), xx_int.flatten()]

        # Stack into point cloud
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
        colors = colors.reshape(-1, 3)

        return points, colors, intrinsics

    def capture_frames(self):
        """Capture frames from webcam"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        print("📸 Starting webcam capture...")

        frame_count = 0
        while self.streaming:
            ret, frame = cap.read()
            if ret:
                # Skip every other frame for better performance
                frame_count += 1
                if frame_count % 2 != 0:
                    continue

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Add to queue (drop old frames if full)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                self.frame_queue.put(frame_rgb)

        cap.release()

    def process_depth(self):
        """Process frames to generate depth maps and point clouds"""
        print("🧠 Starting depth processing...")

        while self.streaming:
            try:
                frame = self.frame_queue.get(timeout=1)

                # Process frame to point cloud
                points, colors, intrinsics = self.process_frame_to_pointcloud(frame)

                # Create message with camera intrinsics
                pointcloud_data = {
                    "timestamp": datetime.now().isoformat(),
                    "num_points": len(points),
                    "points": points.tolist(),
                    "colors": colors.tolist(),
                    "intrinsics": intrinsics.tolist(),  # Include for client-side processing
                }

                # Add to queue
                if self.pointcloud_queue.full():
                    try:
                        self.pointcloud_queue.get_nowait()
                    except:
                        pass
                self.pointcloud_queue.put(pointcloud_data)

            except queue.Empty:
                continue
            except Exception as e:
                import traceback

                print(f"❌ Error in depth processing: {e}")
                traceback.print_exc()

    async def handle_client(self, websocket):
        """Handle WebSocket client connections"""
        self.clients.add(websocket)
        print(f"👤 Client connected. Total clients: {len(self.clients)}")

        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            print(f"👤 Client disconnected. Total clients: {len(self.clients)}")

    async def broadcast_pointclouds(self):
        """Broadcast point clouds to all connected clients"""
        while self.streaming:
            try:
                # Get latest point cloud
                pointcloud_data = self.pointcloud_queue.get(timeout=1)

                if self.clients:
                    # Send full point cloud data
                    message = json.dumps(
                        {
                            "type": "pointcloud",
                            "timestamp": pointcloud_data["timestamp"],
                            "num_points": len(pointcloud_data["points"]),
                            "points": pointcloud_data["points"],
                            "colors": pointcloud_data["colors"],
                        }
                    )

                    # Send to all clients
                    disconnected = set()
                    for client in self.clients:
                        try:
                            await client.send(message)
                        except:
                            disconnected.add(client)

                    # Remove disconnected clients
                    self.clients -= disconnected

                    print(
                        f"📡 Sent {pointcloud_data['num_points']//10} points to {len(self.clients)} clients"
                    )

            except queue.Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"❌ Broadcast error: {e}")
                await asyncio.sleep(0.1)

    async def start_websocket_server(self):
        """Start the WebSocket server"""
        print("🌐 Starting WebSocket server on ws://localhost:8765")
        async with websockets.serve(self.handle_client, "localhost", 8765):
            await self.broadcast_pointclouds()

    def start(self):
        """Start the streaming pipeline"""
        self.streaming = True

        # Start capture thread
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.start()

        # Start processing thread
        process_thread = threading.Thread(target=self.process_depth)
        process_thread.start()

        # Start WebSocket server
        try:
            asyncio.run(self.start_websocket_server())
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            self.streaming = False
            capture_thread.join()
            process_thread.join()

            # Cleanup temp directory
            import shutil

            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"🗑️ Cleaned up temp directory")

            print("✅ Shutdown complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stream point clouds from webcam using DepthAnything"
    )
    parser.add_argument(
        "--model",
        choices=["small", "base", "large"],
        default="small",
        help="DepthAnything model size",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default="mps",
        help="Device to run on (mps=Mac GPU, cuda=NVIDIA GPU, cpu=CPU)",
    )

    args = parser.parse_args()

    streamer = DepthPointCloudStreamer(model_size=args.model, device=args.device)
    streamer.start()
