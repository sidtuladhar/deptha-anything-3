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
