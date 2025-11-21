"""
WebSocket-based depth streaming server for Runpod
Uses WebSocket for video streaming (JPEG frames) - works on TCP-only Runpod network
Supports multi-camera mode with timestamp synchronization
"""

import json
import asyncio
import numpy as np
import torch
import msgpack
import logging
from io import BytesIO
from PIL import Image
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional
from dataclasses import dataclass
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn
from depth_anything_3.api import DepthAnything3

# Suppress all INFO logs from dependencies
logging.basicConfig(level=logging.WARNING)


# =========================
# Multi-view Fusion Helper Functions (from export_to_glb)
# =========================


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """
    Accept (4,4) or (3,4) extrinsic parameters, return (4,4) homogeneous matrix.
    """
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")


def _depths_to_world_points_with_colors(
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    images_u8: np.ndarray,
    conf: np.ndarray | None,
    conf_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each frame, transform (u,v,1) through K^{-1} to get rays,
    multiply by depth to camera frame, then use (w2c)^{-1} to transform to world frame.
    Simultaneously extract colors.

    This is the CORE multi-view fusion algorithm - transforms all camera point clouds
    to world coordinates and merges them.
    """
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W,3)

    pts_all, col_all = [], []

    for i in range(N):
        d = depth[i]  # (H,W)
        valid = np.isfinite(d) & (d > 0)
        if conf is not None:
            valid &= conf[i] >= conf_thr
        if not np.any(valid):
            continue

        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        K_inv = np.linalg.inv(K[i])  # (3,3)
        c2w = np.linalg.inv(_as_homogeneous44(ext_w2c[i]))  # (4,4)

        rays = K_inv @ pix[vidx].T  # (3,M)
        Xc = rays * d_flat[vidx][None, :]  # (3,M)

        # Flip Y coordinate to match image coordinate system (consistent with single-camera mode)
        Xc[1, :] = -Xc[1, :]

        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M,3)

        cols = images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8)  # (M,3)

        pts_all.append(Xw)
        col_all.append(cols)

    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    return np.concatenate(pts_all, 0), np.concatenate(col_all, 0)


def _filter_and_downsample(points: np.ndarray, colors: np.ndarray, num_max: int):
    """Remove invalid points and downsample to max count for streaming performance"""
    if points.shape[0] == 0:
        return points, colors
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if points.shape[0] > num_max:
        idx = np.random.choice(points.shape[0], num_max, replace=False)
        points, colors = points[idx], colors[idx]
    return points, colors


def get_conf_thresh(
    conf: np.ndarray,
    base_thresh: float,
    conf_thresh_percentile: float = 40.0,
    ensure_thresh_percentile: float = 90.0,
) -> float:
    """Calculate adaptive confidence threshold based on percentiles"""
    lower = np.percentile(conf, conf_thresh_percentile)
    upper = np.percentile(conf, ensure_thresh_percentile)
    conf_thresh = min(max(base_thresh, lower), upper)
    return conf_thresh


# =========================
# Frame and Connection Classes
# =========================


@dataclass
class TimestampedFrame:
    """Frame with high-precision timestamp for synchronization"""

    timestamp: float  # UTC milliseconds from performance API
    jpeg_bytes: bytes
    client_id: int


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
        self.last_log_time = datetime.now()

    async def process_frame_async(self, img):
        """Process a single frame asynchronously"""
        try:
            process_start = datetime.now()

            # Process with point cloud generation
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

            # Log every 5 seconds
            time_since_last_log = (now - self.last_log_time).total_seconds()
            if time_since_last_log >= 5.0:
                device_label = "GPU" if self.depth_server.device.type in ["cuda", "mps"] else "CPU"
                print(
                    f"📊 Frame {self.frame_count} ({actual_fps:.1f} FPS) [{device_label}]: "
                    f"{len(points)} pts ({len(binary_response)/1024:.1f}KB) | "
                    f"Process: {process_time:.1f}ms | Pack: {pack_time:.1f}ms | Send: {send_time:.1f}ms"
                )
                self.last_log_time = now

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


class MultiCameraManager:
    """
    Manages multiple camera streams with timestamp-based synchronization
    Buffers frames from each camera and aligns them temporally
    """

    def __init__(self, time_window_ms: float = 500.0, max_buffer_size: int = 30):
        """
        Args:
            time_window_ms: Maximum time difference (ms) between frames to consider aligned
            max_buffer_size: Maximum frames to buffer per camera before dropping old ones
        """
        self.camera_buffers: Dict[int, deque] = {}  # client_id -> deque of TimestampedFrame
        self.camera_roles: Dict[int, str] = {}  # client_id -> "viewer" or "camera"
        self.viewer_clients: List[int] = []
        self.camera_clients: List[int] = []
        self.time_window_ms = time_window_ms
        self.max_buffer_size = max_buffer_size
        self.last_process_time = datetime.now()

    def register_client(self, client_id: int, role: str):
        """Register a new client with their role"""
        self.camera_roles[client_id] = role

        if role == "viewer":
            self.viewer_clients.append(client_id)
            print(f"👁️  Client {client_id} registered as VIEWER")
        elif role == "camera":
            self.camera_clients.append(client_id)
            self.camera_buffers[client_id] = deque(maxlen=self.max_buffer_size)
            print(f"📹 Client {client_id} registered as CAMERA")

        print(f"📊 Total: {len(self.viewer_clients)} viewers, {len(self.camera_clients)} cameras")

    def unregister_client(self, client_id: int):
        """Unregister a client and clean up their data"""
        role = self.camera_roles.get(client_id)

        if role == "viewer" and client_id in self.viewer_clients:
            self.viewer_clients.remove(client_id)
        elif role == "camera":
            if client_id in self.camera_clients:
                self.camera_clients.remove(client_id)
            if client_id in self.camera_buffers:
                del self.camera_buffers[client_id]

        if client_id in self.camera_roles:
            del self.camera_roles[client_id]

        print(f"👋 Client {client_id} ({role}) unregistered")
        print(f"📊 Total: {len(self.viewer_clients)} viewers, {len(self.camera_clients)} cameras")

    def add_frame(self, client_id: int, timestamp: float, jpeg_bytes: bytes):
        """Add a timestamped frame from a camera client"""
        if client_id not in self.camera_buffers:
            print(f"⚠️  Received frame from unregistered camera {client_id}")
            return

        frame = TimestampedFrame(timestamp=timestamp, jpeg_bytes=jpeg_bytes, client_id=client_id)
        self.camera_buffers[client_id].append(frame)

    def get_aligned_frames(self) -> Optional[List[TimestampedFrame]]:
        """
        Get temporally aligned frames from all cameras
        Works with 1 camera (single-view) or multiple cameras (multi-view)
        Returns None if no cameras or frames can't be aligned
        """
        # Need at least 1 camera
        if len(self.camera_clients) < 1:
            return None

        # Check if all cameras have frames
        for cam_id in self.camera_clients:
            if not self.camera_buffers.get(cam_id):
                return None

        # Single camera - just return latest frame
        if len(self.camera_clients) == 1:
            cam_id = self.camera_clients[0]
            buffer = self.camera_buffers[cam_id]
            if buffer:
                frame = buffer.popleft()
                return [frame]
            return None

        # Multi-camera - find aligned frames
        # Find the most recent timestamp among the oldest frames in each buffer
        min_timestamps = [
            self.camera_buffers[cam_id][0].timestamp for cam_id in self.camera_clients
        ]
        target_timestamp = max(min_timestamps)

        # Try to find frames within time window of target
        aligned_frames = []
        for cam_id in self.camera_clients:
            buffer = self.camera_buffers[cam_id]

            # Find closest frame to target timestamp
            closest_frame = None
            closest_diff = float("inf")
            closest_idx = -1

            for idx, frame in enumerate(buffer):
                diff = abs(frame.timestamp - target_timestamp)
                if diff < closest_diff:
                    closest_diff = diff
                    closest_frame = frame
                    closest_idx = idx

            # Check if within time window
            if closest_frame and closest_diff <= self.time_window_ms:
                aligned_frames.append(closest_frame)
                # Remove all frames up to and including the selected one
                for _ in range(closest_idx + 1):
                    if buffer:
                        buffer.popleft()
            else:
                # Can't align - frames too far apart
                return None

        # Only return if we have frames from all cameras
        if len(aligned_frames) == len(self.camera_clients):
            return aligned_frames

        return None

    def has_enough_cameras(self) -> bool:
        """Check if we have at least 2 cameras for multi-view"""
        return len(self.camera_clients) >= 2

    def get_camera_list(self) -> List[Dict]:
        """Get list of active cameras for status updates"""
        return [
            {"client_id": cam_id, "buffer_size": len(self.camera_buffers.get(cam_id, []))}
            for cam_id in self.camera_clients
        ]


class CloudDepthServer:
    def __init__(self, model_size="small", device="cuda"):
        """Initialize the cloud depth estimation server"""
        print(f"🚀 Initializing DepthAnything v3 model ({model_size})...")

        # Model mapping
        model_map = {
            "small": "depth-anything/da3-small",
            "base": "depth-anything/da3-base",
            "giant": "depth-anything/da3-giant",
        }

        # Store model size
        self.model_size = model_size

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

        # Show GPU optimization status
        if self.device.type == "cuda":
            print(f"⚡ GPU-accelerated point cloud generation enabled (CUDA)")
        elif self.device.type == "mps":
            print(f"⚡ GPU-accelerated point cloud generation enabled (MPS)")
        else:
            print(f"⚠️  Using CPU for point cloud generation")

        self.active_processors = {}  # client_id -> DepthProcessor
        self.multi_camera = MultiCameraManager(
            time_window_ms=500.0, max_buffer_size=3
        )  # Small buffer for multi-camera alignment

    def process_frame_to_pointcloud_gpu(self, rgb_image, prediction):
        """GPU-accelerated point cloud generation - keeps tensors on GPU"""
        # Get depth map and intrinsics (keep as tensors on GPU!)
        depth_map = prediction.depth[0]
        # Ensure it's a tensor (not numpy)
        if not torch.is_tensor(depth_map):
            depth_map = torch.from_numpy(depth_map).to(self.device)

        confidence = None
        if hasattr(prediction, "conf") and prediction.conf is not None:
            confidence = prediction.conf[0]
            if not torch.is_tensor(confidence):
                confidence = torch.from_numpy(confidence).to(self.device)

        intrinsics = prediction.intrinsics[0]
        # Ensure it's a tensor
        if not torch.is_tensor(intrinsics):
            intrinsics = torch.from_numpy(intrinsics).to(self.device)

        # Extract camera parameters (move scalars to CPU)
        fx = intrinsics[0, 0].item()
        fy = intrinsics[1, 1].item()
        cx = intrinsics[0, 2].item()
        cy = intrinsics[1, 2].item()

        # Get dimensions
        h, w = depth_map.shape

        # Convert RGB image to torch tensor on GPU
        if rgb_image.shape[0] != h or rgb_image.shape[1] != w:
            # Resize using PIL then convert to tensor
            pil_image = Image.fromarray(rgb_image)
            rgb_resized = np.array(pil_image.resize((w, h), Image.BILINEAR))
            rgb_tensor = torch.from_numpy(rgb_resized).to(self.device)
        else:
            rgb_tensor = torch.from_numpy(rgb_image).to(self.device)

        downsample = 2

        # Create mesh grid on GPU
        x_coords = torch.arange(0, w, downsample, device=self.device)
        y_coords = torch.arange(0, h, downsample, device=self.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Flatten indices for compatible indexing (works across PyTorch versions)
        yy_flat = yy.flatten()
        xx_flat = xx.flatten()

        # Convert to linear indices for 1D indexing (more compatible)
        linear_indices = yy_flat * w + xx_flat

        # Use torch.index_select to avoid NumPy conversion (MPS-safe)
        depth_flat = depth_map.flatten()
        z = torch.index_select(depth_flat, 0, linear_indices.long()).reshape(yy.shape)

        # Filter by confidence if available (all on GPU!)
        if confidence is not None:
            conf_threshold = 0.5
            conf_flat = confidence.flatten()
            conf_sampled = torch.index_select(conf_flat, 0, linear_indices.long()).reshape(
                yy.shape
            )
            valid_mask = conf_sampled > conf_threshold

            # Apply mask
            xx = xx[valid_mask]
            yy = yy[valid_mask]
            z = z[valid_mask]

        # Calculate 3D coordinates (GPU operations)
        x = (xx.float() - cx) * z / fx
        y = -((yy.float() - cy) * z / fy)  # Negate Y to flip vertically

        # Get colors using torch.index_select (MPS-safe)
        yy_int = yy.long()
        xx_int = xx.long()
        # Convert to linear indices for color extraction
        color_linear_idx = yy_int * w + xx_int
        # Flatten RGB tensor to (H*W, 3) and use index_select
        rgb_flat = rgb_tensor.reshape(-1, 3)
        colors = torch.index_select(rgb_flat, 0, color_linear_idx)  # Shape: (N, 3)

        # Stack into point cloud (still on GPU)
        points = torch.stack([x, y, z], dim=-1)  # Shape: (N, 3)

        # Move to CPU and convert to numpy only at the end
        points_np = points.cpu().numpy().astype(np.float32)
        colors_np = colors.cpu().numpy().astype(np.uint8)
        intrinsics_np = intrinsics.cpu().numpy()

        return points_np, colors_np, intrinsics_np

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

        # Use GPU-accelerated version for CUDA and MPS
        if self.device.type in ["cuda", "mps"]:
            return self.process_frame_to_pointcloud_gpu(rgb_image, prediction)

        # Fallback to CPU version
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

    def process_multiview_to_pointcloud(self, rgb_images: List[np.ndarray]):
        """
        Process multiple synchronized frames for multi-view depth reconstruction using proper fusion

        Uses the official fusion algorithm from export_to_glb to:
        1. Transform each camera's depth map to world coordinates using auto-estimated extrinsics
        2. Merge all cameras' point clouds in world space
        3. Apply confidence filtering to remove low-quality points
        4. Downsample for streaming performance

        Args:
            rgb_images: List of RGB images from different cameras (as numpy arrays)

        Returns:
            Merged point cloud (points, colors, intrinsics) from all cameras
        """
        if len(rgb_images) < 2:
            raise ValueError(f"Multi-view requires at least 2 images, got {len(rgb_images)}")

        # Convert all images to PIL
        pil_images = [Image.fromarray(img) for img in rgb_images]

        # Run multi-view inference - DepthAnything v3 auto-estimates extrinsics
        print(f"🎥 Processing {len(pil_images)} views for multi-view fusion...")
        prediction = self.model.inference(
            pil_images,  # Pass list of images for multi-view
            export_dir=None,
            export_format=[],
        )

        # Validate that we have extrinsics (auto-estimated by model's camera decoder)
        if prediction.extrinsics is None:
            raise ValueError(
                "Model did not estimate camera extrinsics. "
                "Make sure you're using a model with camera decoder (da3-base/large/giant)."
            )

        print(f"✅ Model auto-estimated extrinsics for {len(pil_images)} cameras")

        # Convert all tensors to numpy arrays
        depth = prediction.depth
        if torch.is_tensor(depth):
            depth = depth.cpu().numpy()

        intrinsics = prediction.intrinsics
        if torch.is_tensor(intrinsics):
            intrinsics = intrinsics.cpu().numpy()

        extrinsics = prediction.extrinsics
        if torch.is_tensor(extrinsics):
            extrinsics = extrinsics.cpu().numpy()

        conf = None
        if hasattr(prediction, "conf") and prediction.conf is not None:
            conf = prediction.conf
            if torch.is_tensor(conf):
                conf = conf.cpu().numpy()

        # Prepare processed images (resize to match depth map size if needed)
        N, H, W = depth.shape
        images_u8 = []
        for i, rgb_img in enumerate(rgb_images):
            if rgb_img.shape[0] != H or rgb_img.shape[1] != W:
                rgb_img = np.array(Image.fromarray(rgb_img).resize((W, H), Image.BILINEAR))
            images_u8.append(rgb_img)
        images_u8 = np.stack(images_u8, axis=0).astype(np.uint8)  # (N, H, W, 3)

        # Calculate adaptive confidence threshold (match official export_to_glb defaults)
        base_conf_threshold = 0.9  # Higher threshold = only high-confidence points
        if conf is not None:
            conf_threshold = get_conf_thresh(
                conf.reshape(-1),  # Flatten all confidence values
                base_conf_threshold,
                conf_thresh_percentile=40.0,
                ensure_thresh_percentile=90.0,
            )
            print(f"📊 Using adaptive confidence threshold: {conf_threshold:.3f}")
        else:
            conf_threshold = base_conf_threshold

        # CORE FUSION: Transform all cameras to world space and merge
        print(f"🔄 Fusing {len(pil_images)} camera point clouds in world space...")
        points, colors = _depths_to_world_points_with_colors(
            depth=depth,  # (N, H, W) - ALL cameras
            K=intrinsics,  # (N, 3, 3) - ALL camera intrinsics
            ext_w2c=extrinsics,  # (N, 4, 4) or (N, 3, 4) - Auto-estimated poses
            images_u8=images_u8,  # (N, H, W, 3) - ALL camera images
            conf=conf,  # (N, H, W) - Confidence maps
            conf_thr=conf_threshold,
        )

        # Downsample for streaming (250k points = smooth + high quality)
        num_max_points = 250_000
        points, colors = _filter_and_downsample(points, colors, num_max_points)

        # Use primary camera's intrinsics for reference
        primary_intrinsics = intrinsics[0]

        print(
            f"✅ Multi-view fusion complete: {len(points):,} points from {len(pil_images)} cameras"
        )

        return points, colors, primary_intrinsics


# Create FastAPI app
app = FastAPI(title="Depth Anything v3 WebSocket Streaming Server")

# Global server instance
depth_server = None


@app.get("/")
async def serve_viewer():
    """Serve the viewer.html file"""
    return FileResponse("viewer.html")


@app.get("/viewer.html")
async def serve_point_cloud_viewer():
    """Serve the point cloud viewer (base model)"""
    return FileResponse("viewer.html")


@app.get("/viewer-multicam.html")
async def serve_multicam_viewer():
    """Serve the multi-camera viewer"""
    return FileResponse("viewer-multicam.html")


async def keepalive_ping_task(websocket: WebSocket, client_id: int, interval: float = 15.0):
    """Background task to send periodic pings to keep WebSocket connection alive"""
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await websocket.send_text(json.dumps({"type": "ping"}))
            except Exception as e:
                # Connection closed, exit task
                break
    except asyncio.CancelledError:
        # Task was cancelled (client disconnected)
        pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for multi-camera depth streaming"""
    await websocket.accept()
    client_id = id(websocket)
    client_role = None
    keepalive_task = None
    print(f"👤 Client {client_id} connected via WebSocket")

    # Create depth processor for legacy single-camera mode
    depth_processor = DepthProcessor(depth_server, websocket)
    depth_server.active_processors[client_id] = depth_processor

    try:
        frame_count = 0

        while True:
            # Receive message - could be binary or text
            message = await websocket.receive()

            # Handle text messages (control/registration)
            if "text" in message:
                try:
                    data = json.loads(message["text"])

                    if data.get("type") == "register_role":
                        # Client is registering their role
                        client_role = data.get("role")
                        client_timestamp = data.get("timestamp")

                        depth_server.multi_camera.register_client(client_id, client_role)

                        print(
                            f"📝 Client {client_id} registered as {client_role} (timestamp: {client_timestamp})"
                        )

                        # Start keepalive ping for camera clients to prevent 30s timeout
                        if client_role == "camera" and keepalive_task is None:
                            keepalive_task = asyncio.create_task(
                                keepalive_ping_task(websocket, client_id, interval=15.0)
                            )
                            print(f"🔄 Started keepalive pings for camera {client_id}")

                        # Broadcast updated camera list to ALL viewers
                        await broadcast_camera_list_to_viewers(depth_server)

                    elif data.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))

                except json.JSONDecodeError:
                    print(f"⚠️  Invalid JSON from client {client_id}")
                    continue

            # Handle binary messages (frames)
            elif "bytes" in message:
                try:
                    binary_data = message["bytes"]

                    # Check if this is a registered camera sending frames
                    if client_role == "camera":
                        # Multi-camera mode - raw JPEG with server-side timestamp
                        timestamp = datetime.now().timestamp() * 1000  # UTC milliseconds

                        # Add to multi-camera buffer
                        depth_server.multi_camera.add_frame(client_id, timestamp, binary_data)

                        frame_count += 1
                        if frame_count == 1:
                            print(f"✅ First frame received from camera {client_id}!")

                        # Debug: Log buffer status every 30 frames
                        if frame_count % 30 == 0:
                            buffer_sizes = {
                                cid: len(depth_server.multi_camera.camera_buffers.get(cid, []))
                                for cid in depth_server.multi_camera.camera_clients
                            }
                            print(
                                f"📊 Buffer status: {buffer_sizes}, Viewers: {len(depth_server.multi_camera.viewer_clients)}"
                            )

                        # Try to process aligned frames
                        # Works with 1 camera (single-view) or 2+ cameras (multi-view)
                        aligned_frames = depth_server.multi_camera.get_aligned_frames()

                        if aligned_frames:
                            # Process either single-view or multi-view
                            await process_multiview_frames(aligned_frames, depth_server)
                        elif frame_count % 30 == 0:
                            print(
                                f"⏸️  No aligned frames available (cameras: {len(depth_server.multi_camera.camera_clients)})"
                            )

                    else:
                        # Legacy single-camera mode (no role registration)
                        frame_count += 1

                        if frame_count == 1:
                            print(
                                f"✅ First frame received from client {client_id} (legacy mode)!"
                            )

                        # Process single frame directly
                        await depth_processor.process_jpeg_frame(binary_data)

                except Exception as e:
                    print(f"❌ Error processing binary message from {client_id}: {e}")
                    import traceback

                    traceback.print_exc()

    except WebSocketDisconnect:
        print(f"👤 Client {client_id} disconnected")
    except RuntimeError as e:
        # Handle "Cannot call receive once a disconnect message has been received"
        if "disconnect" in str(e).lower():
            print(f"👤 Client {client_id} disconnected (clean)")
        else:
            print(f"❌ RuntimeError with client {client_id}: {e}")
            import traceback

            traceback.print_exc()
    except Exception as e:
        print(f"❌ Error with client {client_id}: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cancel keepalive task if it exists
        if keepalive_task is not None:
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass

        # Clean up
        if client_role:
            depth_server.multi_camera.unregister_client(client_id)
            # Broadcast updated camera list after unregistration
            try:
                await broadcast_camera_list_to_viewers(depth_server)
            except:
                pass  # Don't fail cleanup if broadcast fails

        if client_id in depth_server.active_processors:
            del depth_server.active_processors[client_id]

        print(f"👤 Total active clients: {len(depth_server.active_processors)}")


async def broadcast_camera_list_to_viewers(depth_server: CloudDepthServer):
    """Send updated camera list to all connected viewers"""
    try:
        camera_list = depth_server.multi_camera.get_camera_list()
        message = json.dumps(
            {"type": "camera_list", "cameras": camera_list, "count": len(camera_list)}
        )

        # Send to all viewer clients
        viewer_clients = depth_server.multi_camera.viewer_clients
        for viewer_id in viewer_clients:
            if viewer_id in depth_server.active_processors:
                processor = depth_server.active_processors[viewer_id]
                try:
                    await processor.websocket.send_text(message)
                except Exception as e:
                    print(f"⚠️  Failed to send camera list to viewer {viewer_id}: {e}")

        print(
            f"📡 Broadcast camera list to {len(viewer_clients)} viewers: {len(camera_list)} cameras"
        )

    except Exception as e:
        print(f"❌ Error broadcasting camera list: {e}")


async def process_multiview_frames(
    aligned_frames: List[TimestampedFrame], depth_server: CloudDepthServer
):
    """Process aligned frames (single-view or multi-view) and send to all viewers"""
    try:
        # Convert JPEG bytes to RGB numpy arrays
        rgb_images = []
        for frame in aligned_frames:
            img = Image.open(BytesIO(frame.jpeg_bytes))
            rgb_array = np.array(img)
            rgb_images.append(rgb_array)

        # Get timestamp range for logging
        timestamps = [f.timestamp for f in aligned_frames]
        time_spread = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0

        # Choose processing mode based on number of cameras
        if len(rgb_images) == 1:
            # Single-view mode
            print(f"📹 Processing single-view frame...")
            points, colors, intrinsics = await asyncio.to_thread(
                depth_server.process_frame_to_pointcloud, rgb_images[0]
            )
        else:
            # Multi-view mode
            print(
                f"🎥 Processing {len(rgb_images)} aligned views (time spread: {time_spread:.1f}ms)..."
            )
            points, colors, intrinsics = await asyncio.to_thread(
                depth_server.process_multiview_to_pointcloud, rgb_images
            )

        # Pack point cloud data
        response_data = {
            "type": "pointcloud",
            "timestamp": datetime.now().isoformat(),
            "num_points": len(points),
            "points": points.astype(np.float32).tobytes(),
            "colors": colors.astype(np.uint8).tobytes(),
            "num_views": len(rgb_images),
        }
        binary_response = msgpack.packb(response_data, use_bin_type=True)

        # Send to all viewer clients
        viewer_clients = depth_server.multi_camera.viewer_clients
        for viewer_id in viewer_clients:
            if viewer_id in depth_server.active_processors:
                processor = depth_server.active_processors[viewer_id]
                try:
                    await processor.websocket.send_bytes(binary_response)
                except Exception as e:
                    print(f"⚠️  Failed to send to viewer {viewer_id}: {e}")

        mode_str = "single-view" if len(rgb_images) == 1 else f"{len(rgb_images)}-view"
        print(f"📤 Sent {len(points):,} points ({mode_str}) to {len(viewer_clients)} viewers")

    except Exception as e:
        print(f"❌ Error processing frames: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WebSocket depth streaming server")
    parser.add_argument(
        "--model",
        choices=["small", "base", "giant"],
        default="base",
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

    print(f"\n🌐 Starting WebSocket server on {args.host}:{args.port}")
    print(f"📄 Single-camera viewer: http://{args.host}:{args.port}/")
    print(f"🎥 Multi-camera viewer:  http://{args.host}:{args.port}/viewer-multicam.html")
    print()

    # Start FastAPI server
    uvicorn.run(app, host=args.host, port=args.port)
