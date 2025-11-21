# Performance Optimization Findings

## Summary

Migrated from base64 JPEG WebSocket streaming to WebRTC with binary msgpack encoding. Achieved significant bandwidth reduction but GPU processing remains the bottleneck.

## Key Mistakes & Lessons Learned

### 1. **Assumed Network Was the Bottleneck (WRONG)**

**Mistake:** Initially believed 5 FPS was due to network transfer of large messages.

**Reality:** GPU processes each frame in **28ms (~35 FPS potential)**, but end-to-end was 200ms (5 FPS).

**Lesson:** Always measure each pipeline stage separately before optimizing.

### 2. **Used .tolist() for msgpack Encoding (658KB messages)**

**Mistake:** Converted numpy arrays to Python lists for msgpack serialization.

```python
# BAD: Creates huge messages (658KB for 21k points)
"points": points.astype(np.float32).flatten().tolist()
```

**Fix:** Use raw bytes instead:

```python
# GOOD: Compact binary (310KB for 21k points)
"points": points.astype(np.float32).tobytes()
```

**Impact:** 53% size reduction (658KB → 310KB)

### 3. **Didn't Implement Frame Skipping in WebRTC**

**Mistake:** WebRTC streams at 30 FPS but processing takes 120-150ms (max 8 FPS). Without skipping, frames queued up causing "slow motion" effect.

**Fix:** Added processing flag to skip frames while busy:

```python
async def recv(self):
    frame = await self.track.recv()
    if self.is_processing:
        return frame  # Skip if busy
    self.is_processing = True
    # ... process ...
    self.is_processing = False
```

**Impact:** Eliminated queueing lag, live updates restored

### 4. **ICE Candidate Type Mismatch**

**Mistake:** Sent JavaScript ICE candidate objects as JSON dicts, but aiortc expects `RTCIceCandidate` Python objects.

**Fix:** Manual reconstruction:

```python
candidate = RTCIceCandidate(
    ip=candidate_dict.get("address", ""),
    port=candidate_dict.get("port", 0),
    sdpMid=candidate_dict.get("sdpMid"),
    # ... etc
)
```

## Current Performance Baseline (Localhost, CUDA)

### Network Transfer (Optimized ✅)

- **Upload (WebRTC)**: Negligible (hardware H.264 encoding)
- **Download (msgpack binary)**: 310KB per frame
- **Pack time**: 0.1ms (excellent)
- **Send time**: 8-21ms (acceptable)

### GPU Processing (Bottleneck ⚠️)

- **Model inference**: 78-102ms
- **Image preprocessing**: 5-9ms
- **Prediction conversion**: 30-36ms
- **Total processing**: 120-150ms per frame
- **Actual FPS**: ~7 FPS (limited by GPU, not network)

## Confirmed Bottleneck

**The depth model processing time (120-150ms) is the limiting factor**, not network transfer.

To reach 15-20 FPS on a 4090:

- Would need to reduce processing to 50-66ms per frame
- Potential optimizations:
  - Lower input resolution (currently 378x504)
  - Increase downsample parameter (reduce point cloud density)
  - Model quantization (FP16/INT8)
  - Batch processing (if multiple clients)

## Architecture Decisions

### Why WebRTC Over Base64 JPEG?

- **Upload bandwidth**: ~90% reduction (no base64 overhead, video compression)
- **Latency**: Lower (UDP vs TCP)
- **Browser efficiency**: Hardware video encoding
- **Trade-off**: More complex signaling setup

### Why msgpack Binary Over JSON?

- **Size**: 53% smaller than tolist() arrays
- **Speed**: 0.1ms pack time vs 2ms with tolist()
- **Browser support**: msgpack-lite library handles Uint8Array natively

---

## Migration: WebRTC → Pure WebSocket (Dec 2024)

### Problem: Runpod Blocks UDP

**Discovery:** Runpod only exposes TCP ports. WebRTC media (RTP) requires UDP for optimal performance.

**Attempted Solutions:**
1. ❌ **STUN servers** - Revealed public IP but couldn't establish media path (UDP blocked)
2. ❌ **Free TURN relay** - Unreliable, rate-limited, added 50ms latency
3. ❌ **Self-hosted TURN** - Would work but adds infrastructure complexity + $10-30/month

**Final Solution:** ✅ **Switch to pure WebSocket with JPEG frames**

### Why WebSocket Won

| Metric | WebRTC + TURN | WebSocket JPEG |
|--------|---------------|----------------|
| Network latency | ~100ms (relay hop) | ~80ms (direct) |
| Infrastructure | TURN server required | None |
| Reliability | Depends on TURN uptime | Direct connection |
| Bandwidth | 2-3 Mbps (H.264) | 10-15 Mbps (JPEG) |
| Complexity | High (ICE/SDP/TURN) | Low (just WebSocket) |
| Cost | $10-30/month | $0 |

**Key Insight:** Since GPU depth inference (30-40ms) dominated latency anyway, the network transport choice (WebSocket vs WebRTC) barely mattered. WebSocket's simplicity won.

### Implementation Changes

**Client (viewer.html):**
- Removed all RTCPeerConnection code
- Added canvas-based frame capture: `canvas.drawImage(video)` → `canvas.toBlob('image/jpeg', 0.7)`
- Send JPEG blobs via WebSocket at 30 FPS
- Total removed: ~150 lines

**Server (stream_server.py):**
- Removed aiortc dependency
- Simplified WebSocket handler (no SDP offer/answer, no ICE candidates)
- Decode JPEG → depth estimation → send point cloud
- Total removed: ~140 lines

**Result:** 60% less code, works on TCP-only Runpod, easier to debug.

---

## Point Cloud Generation Bottleneck (Dec 2024)

### Discovery: CPU-bound NumPy Operations

**Profiling revealed:**
```
GPU inference:        21-31ms (fast, stable)
Point cloud gen:      6-23ms  (CPU, highly variable!) ← BOTTLENECK
Pack/send:           10ms
──────────────────────────────
Total:               50-70ms = 14-18 FPS
```

**Why variable?** NumPy operations on CPU subject to:
- Python GIL contention
- CPU scheduler interruptions
- Memory allocation delays
- Cache misses

**The CPU bottleneck:**
```python
# All these run on CPU (slow, variable):
xx, yy = np.meshgrid(...)           # 2-5ms
z = depth_map[yy, xx]               # 1-3ms
valid_mask = conf_values > threshold # 3-8ms
x = (xx - cx) * z / fx              # 2-5ms
colors = rgb_image[yy_int, xx_int]  # 2-8ms
points = np.stack([x, y, z])        # 1-3ms
```

### Solution: GPU-Accelerated Point Cloud Generation

**Strategy:** Keep all tensors on GPU, use PyTorch operations instead of NumPy.

### Failed Attempts

#### Attempt 1: Direct Tensor Indexing (FAILED)
```python
z = depth_map[yy, xx]  # ❌ Triggers NumPy conversion
```
**Error:** `TypeError: can't convert cuda:0 device type tensor to numpy`

**Why it failed:** PyTorch's advanced indexing with 2D tensor indices triggers internal NumPy conversion on some backends.

#### Attempt 2: Linear Indexing with Bracket Notation (FAILED on MPS)
```python
linear_idx = yy * w + xx
z = depth_map.flatten()[linear_idx]  # ❌ Still triggers NumPy on MPS
```
**Error:** `TypeError: can't convert mps:0 device type tensor to numpy`

**Why it failed:** MPS backend has stricter rules about bracket indexing, even with 1D tensors.

#### Attempt 3: CUDA-only Optimization (WORKAROUND)
```python
if self.device.type == 'cuda':  # Works on CUDA, skip MPS
    return gpu_version()
```
**Why suboptimal:** Leaves MPS users (Mac developers) without acceleration.

### Final Solution: torch.index_select() ✅

**The fix:** Use PyTorch's explicit indexing function that never touches NumPy.

```python
# Convert 2D coords to 1D linear indices
linear_indices = yy_flat * w + xx_flat

# Use torch.index_select - pure PyTorch, no NumPy conversion
depth_flat = depth_map.flatten()
z = torch.index_select(depth_flat, 0, linear_indices.long()).reshape(yy.shape)

# Same for colors
rgb_flat = rgb_tensor.reshape(-1, 3)
colors = torch.index_select(rgb_flat, 0, color_linear_idx)
```

**Critical detail:** Must also ensure inputs are tensors, not NumPy:
```python
if not torch.is_tensor(depth_map):
    depth_map = torch.from_numpy(depth_map).to(self.device)
```

### Why torch.index_select Works

- Pure PyTorch operation (no NumPy code path)
- Works on CUDA, MPS, and CPU
- Actually often faster than bracket indexing
- Explicit API - no hidden conversions

### Performance Impact

**Before (CPU NumPy):**
```
Point cloud gen: 6-23ms (3.4x variance)
Total:          50-70ms
FPS:            14-18 (±40% variance)
```

**After (GPU PyTorch):**
```
Point cloud gen: 2-4ms (1.2x variance)
Total:          30-38ms
FPS:            26-30 (±10% variance)
```

**Improvements:**
- **3-6x faster** point cloud generation
- **3x less variance** (much smoother FPS)
- **70% FPS increase** (14 → 28 FPS)
- Works on both CUDA and MPS

### Key Lessons

1. **Check tensor types:** Model outputs can be NumPy or tensors - always validate
2. **Use explicit PyTorch ops:** `torch.index_select()` > bracket indexing for GPU
3. **Linear indexing is universal:** `idx = row * width + col` works everywhere
4. **Profile each stage:** Point cloud gen looked fast (~10ms avg) but variance was the killer
5. **Test on target platforms:** CUDA and MPS have different indexing behaviors

### Current Performance (GPU-Optimized)

**Mac (MPS):**
- Process: 32-40ms
- FPS: 20-25

**Runpod (CUDA):**
- Process: 28-35ms
- FPS: 26-30

**Remaining bottleneck:** GPU depth inference (21-31ms) - can't optimize further without model changes.
