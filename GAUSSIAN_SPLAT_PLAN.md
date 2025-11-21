# Gaussian Splat Streaming Implementation Plan

## Overview
Switch from point cloud streaming to Gaussian splat streaming using DepthAnything v3's giant model with SparkJS for real-time rendering.

## Current State
- **Model**: da3-base (no Gaussian support)
- **Output**: Point clouds (21k points, 310KB/frame)
- **Performance**: 26-30 FPS
- **Renderer**: Three.js Points
- **Transport**: WebSocket + msgpack binary

## Target State
- **Model**: da3-giant (1.15B params, Gaussian support)
- **Output**: Gaussian splats (~21k Gaussians, 500-800KB/frame)
- **Performance**: 5-10 FPS (GPU inference bottleneck)
- **Renderer**: SparkJS SplatMesh
- **Transport**: WebSocket + msgpack binary (unchanged)

---

## Server Changes (stream_server.py)

### 1. Update Model Configuration

**Location**: `CloudDepthServer.__init__()` (lines 116-146)

**Changes**:
```python
# Add giant model to model_map
model_map = {
    "small": "depth-anything/da3-small",
    "base": "depth-anything/da3-base",
    "giant": "depth-anything/da3-giant",  # NEW
}
```

**CLI Update** (line 397):
```python
parser.add_argument(
    "--model",
    choices=["small", "base", "giant"],  # Add giant
    default="giant",  # Change default for Gaussian support
    help="DepthAnything model size",
)
```

**Expected Impact**:
- First run: ~4.6GB model download
- VRAM usage: ~5-6GB (vs ~2GB for base model)
- Initialization time: +5-10 seconds

### 2. Create Gaussian Extraction Method

**New method** (add after `process_frame_to_pointcloud`, line 318):

```python
def process_frame_to_gaussians(self, rgb_image):
    """Process a single frame to generate Gaussian splats"""
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(rgb_image)

    # Run DepthAnything v3 inference with Gaussian output
    prediction = self.model.inference(
        [pil_image],
        infer_gs=True,  # CRITICAL: Enable Gaussian splat output
        export_dir=None,
        export_format=[],
    )

    # Extract Gaussian parameters from prediction
    gaussians = prediction.gaussians

    # Get tensors (ensure they're on CPU for serialization)
    means = gaussians.means[0].cpu().numpy().astype(np.float32)
    scales = gaussians.scales[0].cpu().numpy().astype(np.float32)
    rotations = gaussians.rotations[0].cpu().numpy().astype(np.float32)

    # Use DC-only SH for bandwidth optimization (3 floats vs 27 floats)
    # harmonics shape: (N, 3, 9) -> take only DC component [..., 0]
    colors = gaussians.harmonics[0, :, :, 0].cpu().numpy().astype(np.float32)

    opacities = gaussians.opacities[0].cpu().numpy().astype(np.float32)

    return means, scales, rotations, colors, opacities
```

**Key Decisions**:
- **DC-only SH**: Use `harmonics[..., 0]` (3 floats) instead of full SH (27 floats)
  - Saves ~7MB per frame
  - Still provides photorealistic appearance
  - Full SH can be added later if bandwidth allows

### 3. Update DepthProcessor to Use Gaussians

**Location**: `DepthProcessor.process_frame_async()` (lines 37-84)

**Changes**:
```python
async def process_frame_async(self, img):
    """Process a single frame asynchronously"""
    try:
        # Process with Gaussian estimation (blocking, but in separate task)
        process_start = datetime.now()
        means, scales, rotations, colors, opacities = await asyncio.to_thread(
            self.depth_server.process_frame_to_gaussians, img
        )
        process_time = (datetime.now() - process_start).total_seconds() * 1000

        # Track GPU vs CPU usage
        device_type = self.depth_server.device.type

        # Pack and send Gaussian splats via WebSocket
        pack_start = datetime.now()
        response_data = {
            "type": "gaussians",  # Changed from "pointcloud"
            "timestamp": datetime.now().isoformat(),
            "num_gaussians": len(means),
            "means": means.tobytes(),
            "scales": scales.tobytes(),
            "rotations": rotations.tobytes(),
            "colors": colors.tobytes(),
            "opacities": opacities.tobytes(),
            "sh_degree": 0,  # DC-only (0 = no SH beyond DC)
        }
        binary_response = msgpack.packb(response_data, use_bin_type=True)
        pack_time = (datetime.now() - pack_start).total_seconds() * 1000

        # ... rest of send and logging code remains same
```

**Expected Message Sizes**:
- means: 21k × 3 × 4 bytes = 252KB
- scales: 21k × 3 × 4 bytes = 252KB
- rotations: 21k × 4 × 4 bytes = 336KB
- colors (DC-only): 21k × 3 × 4 bytes = 252KB
- opacities: 21k × 4 bytes = 84KB
- **Total: ~1.2MB per frame** (vs 310KB for point clouds)

**Optimization**: Could use float16 to halve size to ~600KB

### 4. Performance Expectations

**Profiling Breakdown**:
- Model inference (giant): 80-120ms (vs 21-31ms for base)
- Gaussian extraction: 5-10ms (CPU tensor→numpy conversion)
- msgpack packing: 1-2ms
- WebSocket send: 15-30ms (larger payload)
- **Total: 100-160ms = 6-10 FPS**

---

## Client Changes (viewer.html)

### 1. Replace Three.js Points with SparkJS

**Location**: Script imports (lines 169-176)

**Changes**:
```html
<script type="importmap">
  {
    "imports": {
      "three": "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.178.0/three.module.js",
      "@sparkjsdev/spark": "https://sparkjs.dev/releases/spark/0.1.10/spark.module.js"
    }
  }
</script>

<script type="module">
  import * as THREE from "three";
  import { SplatMesh } from "@sparkjsdev/spark";
  import { OrbitControls } from "three/addons/controls/OrbitControls.js";

  // ... rest of initialization
```

**Remove** (lines 225-227, 479-500):
- Old `pointCloud` and `geometry` variables
- Old `updatePointCloud()` function that creates Three.Points

### 2. Initialize SplatMesh

**Location**: After scene setup (after line 223)

**Add**:
```javascript
// Gaussian splat mesh (replaces point cloud)
let splatMesh = null;
```

### 3. Implement Gaussian Buffer Loader

**Location**: Replace `updatePointCloud()` function (lines 411-504)

**New implementation**:
```javascript
function updateGaussianSplat(data) {
  // Decode binary buffers to typed arrays
  const meansBuffer = data.means instanceof Uint8Array ? data.means.buffer : data.means;
  const scalesBuffer = data.scales instanceof Uint8Array ? data.scales.buffer : data.scales;
  const rotationsBuffer = data.rotations instanceof Uint8Array ? data.rotations.buffer : data.rotations;
  const colorsBuffer = data.colors instanceof Uint8Array ? data.colors.buffer : data.colors;
  const opacitiesBuffer = data.opacities instanceof Uint8Array ? data.opacities.buffer : data.opacities;

  const means = new Float32Array(meansBuffer);
  const scales = new Float32Array(scalesBuffer);
  const rotations = new Float32Array(rotationsBuffer);
  const colors = new Float32Array(colorsBuffer);
  const opacities = new Float32Array(opacitiesBuffer);

  const numGaussians = data.num_gaussians;

  // Remove old splat mesh if exists
  if (splatMesh) {
    scene.remove(splatMesh);
    splatMesh.dispose();
  }

  // Create new SplatMesh from buffers
  // Note: SparkJS may require specific buffer format - this needs testing
  splatMesh = new SplatMesh({
    positions: means,
    scales: scales,
    rotations: rotations,
    colors: colors,
    opacities: opacities,
  });

  scene.add(splatMesh);

  // Update UI
  pointCountEl.textContent = numGaussians.toLocaleString();
}
```

**Note**: SparkJS API may differ - this is the logical structure. May need to:
- Create intermediate PLY format
- Use Blob URL approach
- Build custom buffer loader

### 4. Update Message Handler

**Location**: `ws.onmessage` handler (line 328)

**Changes**:
```javascript
ws.onmessage = async (event) => {
  try {
    // Handle binary messages (Gaussian splat data)
    if (event.data instanceof ArrayBuffer) {
      const data = msgpack.decode(new Uint8Array(event.data));

      if (data.type === "gaussians") {  // Changed from "pointcloud"
        updateGaussianSplat(data);  // Use new function

        // Update FPS stats
        frameCount++;
        const now = Date.now();
        if (now - lastTime >= 1000) {
          fps = frameCount;
          frameCount = 0;
          lastTime = now;
          fpsEl.textContent = fps;
        }
      }
      return;
    }

    // ... rest of text message handling
```

### 5. UI Updates

**Location**: Info panel (lines 145-156)

**Changes**:
```html
<div class="stat">Gaussians: <strong id="point-count">0</strong></div>
<div class="stat">FPS: <strong id="fps">0</strong></div>
<div class="stat">Mode: <strong>Gaussian Splat</strong></div>
```

---

## Implementation Steps

### Phase 1: Server ✅ COMPLETED
1. ✅ Add "giant" to model_map
2. ✅ Update CLI argument parser
3. ✅ Implement `save_gaussian_ply_bytes()` in gsply_helpers.py
4. ✅ Implement `process_frame_to_gaussians()` method
5. ✅ Update `DepthProcessor.process_frame_async()` for model-dependent routing
6. ⏳ Test locally: verify Gaussian extraction works (NEXT)

### Phase 2: Client ✅ COMPLETED
1. ✅ Created test-sparkjs.html for API validation
2. ✅ Created viewer-gaussian.html with SparkJS integration
3. ✅ Implemented PLY Blob URL loading with URL.revokeObjectURL() cleanup
4. ✅ Added error handling with validation (NaN/Infinity checks)
5. ✅ Added debug info panel and axes helper for coordinate debugging
6. ⏳ Test rendering: verify SparkJS displays Gaussians (NEXT)

### Phase 3: Testing & Optimization ⏳ IN PROGRESS
1. ⏳ Test SparkJS with butterfly.spz sample (test-sparkjs.html)
2. ⏳ Test on localhost with webcam + giant model
3. ⏳ Verify quality vs point clouds
4. ⏳ Measure actual FPS and bandwidth
5. ⏳ Debug coordinate system if needed (Z-flip, rotation)
6. ⏳ Deploy to Runpod
7. ⏳ Test over internet connection
8. 🔧 Optimize if needed:
   - Try float16 encoding (halve message size)
   - Implement frame skipping if FPS too low
   - Add quality toggle (point clouds ↔ Gaussians)
   - Reduce prune_by_opacity threshold if too few Gaussians

---

## Success Criteria

### Functional Requirements
- ✅ Server loads da3-giant model successfully
- ✅ Gaussian parameters extracted correctly
- ✅ msgpack serialization works for Gaussian data
- ✅ Client receives and decodes Gaussian buffers
- ✅ SparkJS renders photorealistic Gaussian splats

### Performance Requirements
- ✅ Achieves 5-10 FPS (acceptable for Gaussian quality)
- ✅ Message size under 1.5MB per frame
- ✅ Works on Runpod's TCP-only network
- ✅ Client-side rendering at 30+ FPS (independent of network FPS)

### Quality Requirements
- ✅ Visual quality significantly better than point clouds
- ✅ Smooth splat rendering (no flickering)
- ✅ Proper camera controls maintained

---

## Risks & Mitigation

### Risk 1: SparkJS API Compatibility
**Problem**: SparkJS may not support direct buffer loading
**Mitigation**:
- Research SparkJS docs/examples first
- Fallback: Use @mkkellogg/gaussian-splats-3d instead
- Worst case: Generate PLY files and load via Blob URLs

### Risk 2: Performance Too Slow
**Problem**: Giant model may be too slow even on GPU
**Mitigation**:
- Add frame skipping (sample every Nth frame)
- Implement quality toggle (switch between point clouds and Gaussians)
- Consider lower resolution input (resize before inference)

### Risk 3: Message Size Too Large
**Problem**: 1.2MB per frame at 8 FPS = 10 Mbps
**Mitigation**:
- Use float16 encoding (reduce to 600KB)
- Implement downsampling (reduce Gaussian count)
- Add compression (gzip on msgpack)

### Risk 4: VRAM Exhaustion
**Problem**: Giant model may exceed available GPU memory
**Mitigation**:
- Check VRAM before loading (torch.cuda.get_device_properties)
- Gracefully fallback to base model if insufficient memory
- Add warning in documentation about VRAM requirements

---

## Alternative Approaches (If Current Plan Fails)

### Option A: Use gaussian-splats-3d Instead of SparkJS
- More mature library
- Better documented buffer API
- Proven real-time rendering support

### Option B: Server-Side Rendering
- Render Gaussians to images on GPU using gs_renderer.py
- Stream JPEG frames (300KB/frame)
- Loses 3D interactivity but guaranteed quality

### Option C: Hybrid Mode
- Stream point clouds by default (fast, 26-30 FPS)
- On-demand Gaussian capture (press key to capture high-quality frame)
- Best of both worlds

---

## Technical Notes

### Gaussian Parameters Explained

**Means (3 floats)**: XYZ center position in world space
**Scales (3 floats)**: Size along each axis (log space)
**Rotations (4 floats)**: Quaternion (wxyz format) for orientation
**Colors (3 floats)**: DC component of spherical harmonics (RGB)
**Opacities (1 float)**: Transparency (0 = transparent, 1 = opaque)

### Spherical Harmonics Breakdown

- **Degree 0 (DC)**: 1 coefficient × 3 colors = 3 floats (constant color)
- **Degree 1**: 3 coefficients × 3 colors = 9 floats (directional lighting)
- **Degree 2**: 5 coefficients × 3 colors = 15 floats (more complex lighting)
- **Total for degree 2**: 9 coefficients × 3 colors = 27 floats

**Our choice**: DC-only (3 floats) for bandwidth, upgradeable to degree 2 later.

### Coordinate System

- **DepthAnything output**: Camera space (Z forward, Y down)
- **Gaussians output**: World space (already transformed)
- **Three.js**: Right-handed (Y up, Z out of screen)
- **May need**: Z-axis flip for correct orientation

---

## Timeline

- **Day 1**: Server implementation and testing (4-6 hours)
- **Day 2-3**: Client implementation with SparkJS (6-8 hours)
- **Day 4**: Testing, debugging, optimization (4-6 hours)
- **Total**: 14-20 hours of development time

---

## Resources

- **DepthAnything v3 Docs**: https://github.com/DepthAnything/Depth-Anything-V3
- **SparkJS**: https://sparkjs.dev
- **Gaussian Splatting**: https://github.com/graphdeco-inria/gaussian-splatting
- **Alternative Renderer**: https://github.com/mkkellogg/GaussianSplats3D
