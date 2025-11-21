# Gaussian Splat Streaming - Testing Guide

## Implementation Status: ✅ CODE COMPLETE

All server and client code has been implemented. Ready for testing.

---

## Test Sequence

### Test 1: Verify SparkJS Renderer (5 min)

**Purpose**: Confirm SparkJS library loads and renders correctly before testing our implementation.

**Steps**:
1. Start a local web server:
   ```bash
   python -m http.server 8080
   ```

2. Open browser to: `http://localhost:8080/test-sparkjs.html`

3. **Expected Result**:
   - ✅ "SparkJS loaded successfully!" status message
   - ✅ Rotating butterfly Gaussian splat visible
   - ✅ Smooth 60 FPS rendering
   - ✅ Mouse controls work (drag to rotate, scroll to zoom)

4. **If it fails**:
   - Check browser console for errors
   - Verify internet connection (SparkJS loads from CDN)
   - Try different browser (Chrome/Firefox recommended)

**STOP HERE if test fails** - SparkJS must work before proceeding.

---

### Test 2: Start Server with Giant Model (10-15 min first run)

**Purpose**: Load the giant model and verify Gaussian extraction works.

**Steps**:
1. Start server with giant model:
   ```bash
   python stream_server.py --model=giant --device=cuda --host=0.0.0.0 --port=8000
   ```

2. **First run only**: Model will download (~4.6GB). Watch for:
   ```
   🚀 Initializing DepthAnything v3 model (giant)...
   🖥️  Using device: cuda
   Downloading model... [progress bar]
   ✅ Model loaded successfully!
   ✨ Gaussian splat mode ENABLED (giant model)
   🌐 Starting WebSocket server on 0.0.0.0:8000
   ```

3. **Subsequent runs** (model cached):
   ```
   🚀 Initializing DepthAnything v3 model (giant)...
   🖥️  Using device: cuda
   ✅ Model loaded successfully!
   ✨ Gaussian splat mode ENABLED (giant model)
   🌐 Starting WebSocket server on 0.0.0.0:8000
   ```

**Expected Output**:
- ✅ No errors during model loading
- ✅ "Gaussian splat mode ENABLED" message
- ✅ Server listening on port 8000

**Common Issues**:
- **Out of VRAM**: Giant model needs ~5-6GB GPU memory
  - Solution: Close other GPU applications or use smaller model for testing
- **Download fails**: Slow/flaky connection
  - Solution: Wait and retry, or pre-download model manually
- **ImportError**: Missing dependencies
  - Solution: `pip install einops plyfile`

---

### Test 3: Test Gaussian Viewer (10 min)

**Purpose**: Verify end-to-end Gaussian streaming works.

**Steps**:
1. With server running, open browser to: `http://localhost:8000/viewer-gaussian.html`

2. Click "Start Webcam" button

3. Allow webcam access when prompted

4. **Expected Behavior**:
   - ✅ Status dot turns green
   - ✅ "Connected" status message
   - ✅ Webcam preview appears in top-right
   - ✅ Server logs show: `👤 Client {id} connected via WebSocket`
   - ✅ Server logs show: `✅ First frame received from client {id}!`
   - ✅ After ~1-2 seconds, server logs:
     ```
     ✨ Frame 1 (0.0 FPS) [GPU]: Gaussian PLY (XXX.XKB) | Process: XXXms | Pack: Xms | Send: XXms
     ```

5. **Wait for Gaussian splat to appear** (~2-5 seconds for first frame)

6. **Verify rendering**:
   - ✅ Gaussian splat mesh appears in viewport
   - ✅ Colors match webcam feed
   - ✅ 3D structure looks correct (not flat)
   - ✅ FPS counter updates (should show 5-10 FPS)
   - ✅ Debug info shows: "✅ Loaded XX.XKB PLY"

7. **Test controls**:
   - ✅ Drag to rotate camera around Gaussian splat
   - ✅ Scroll to zoom in/out
   - ✅ Splat updates in real-time as you move in front of webcam

**What Good Output Looks Like**:
```
Server logs (every 5 seconds):
✨ Frame 15 (7.2 FPS) [GPU]: Gaussian PLY (2145.3KB) | Process: 142ms | Pack: 1ms | Send: 18ms
✨ Frame 20 (7.8 FPS) [GPU]: Gaussian PLY (2089.7KB) | Process: 136ms | Pack: 1ms | Send: 15ms
```

Browser debug info:
```
✅ Loaded 2145.3KB PLY
```

---

### Test 4: Coordinate System Check

**Purpose**: Verify Gaussians are oriented correctly.

**Visual Checks**:
1. **Axes helper** (colored lines):
   - Red = X axis (right)
   - Green = Y axis (up)
   - Blue = Z axis (forward)

2. **Expected orientation**:
   - ✅ Gaussian splat faces camera (not sideways)
   - ✅ Up/down matches webcam (raise hand → hand goes up in viewer)
   - ✅ Left/right matches webcam (wave left → splat moves right, mirrored)

3. **If orientation is wrong**:
   - Check server logs for Gaussian extraction errors
   - May need to add Z-flip or rotation in viewer-gaussian.html
   - See "Coordinate System Fixes" section below

---

### Test 5: Performance Benchmarking

**Purpose**: Measure actual FPS and bandwidth.

**Metrics to Record**:
1. **Server FPS**: From server logs (every 5 seconds)
   - Target: 5-10 FPS
   - Good: 6-8 FPS
   - Excellent: 10+ FPS

2. **Process Time**: From server logs
   - Giant model inference: 80-120ms expected
   - PLY generation: 10-30ms expected
   - Total: 100-150ms expected

3. **Message Size**: From server logs
   - Expected: 1.5-3MB per frame
   - With pruning: Could be lower (~1-2MB)
   - Too small (<500KB): May need to reduce pruning

4. **Client FPS**: From browser FPS counter
   - Should match server FPS (5-10)
   - If lower: Network bottleneck or client rendering issue

**Record Results**:
```
Server FPS: ___
Process Time: ___ ms
Message Size: ___ KB
Client FPS: ___
Quality: Photorealistic / Good / Poor / Broken
```

---

## Troubleshooting

### Issue: SparkJS test fails

**Symptoms**: test-sparkjs.html shows error

**Diagnosis**:
1. Check browser console (F12)
2. Look for network errors (SparkJS CDN unreachable)
3. Look for CORS errors

**Solutions**:
- Try different browser
- Check internet connection
- Verify CDN URL is accessible: https://sparkjs.dev/releases/spark/0.1.10/spark.module.js

---

### Issue: Server fails to load giant model

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Check GPU memory: `nvidia-smi` (CUDA) or Activity Monitor (Mac)
2. Close other GPU applications
3. Try with MPS (Mac): `--device=mps`
4. Fallback to CPU (slow): `--device=cpu`
5. Test with base model first: `--model=base` (point clouds only)

---

### Issue: "Gaussian splat mode requires giant model" error

**Symptoms**: Client connects but server crashes

**Cause**: Server started with wrong model (base/small instead of giant)

**Solution**: Restart server with `--model=giant`

---

### Issue: Webcam preview works but no Gaussian splat appears

**Symptoms**:
- ✅ Server logs show frames being processed
- ✅ Client connected
- ❌ No splat mesh visible in viewer
- ❌ Error in browser console

**Diagnosis**:
1. Open browser console (F12)
2. Look for errors in console
3. Check debug info panel for error messages

**Common Causes**:
- **"Invalid PLY data"**: Server generated corrupt PLY
  - Check server logs for Python errors during Gaussian extraction
  - Verify model loaded correctly
- **SparkJS load error**: PLY format incompatible with SparkJS
  - Check PLY file structure (may need different format)
  - Try saving one PLY to disk and inspecting with text editor
- **Blob URL error**: Browser security blocking blob URLs
  - Check browser settings
  - Try different browser

---

### Issue: Gaussian splat appears but orientation is wrong

**Symptoms**:
- ✅ Splat visible
- ❌ Upside down / sideways / backwards

**Solutions**:

**Option 1: Quick test - manual rotation**
In viewer-gaussian.html, find this line:
```javascript
splatMesh.rotation.set(0, 0, 0);
```

Try different rotations:
```javascript
// Flip upside down
splatMesh.rotation.set(Math.PI, 0, 0);

// Rotate 90 degrees
splatMesh.rotation.set(0, Math.PI/2, 0);

// Flip Z axis
splatMesh.position.set(0, 0, 0);
splatMesh.scale.set(1, 1, -1); // Flip Z
```

**Option 2: Fix in server**
In stream_server.py, `process_frame_to_gaussians()`, adjust `shift_and_scale` or add transformation.

---

### Issue: Performance too slow (<5 FPS)

**Symptoms**: FPS counter shows 2-4 FPS

**Diagnosis**:
1. Check server logs - which step is slow?
   - `Process: 200+ms` → GPU too slow / not using GPU
   - `Send: 50+ms` → Network too slow
   - `Pack: 10+ms` → PLY generation slow

**Solutions**:

**If GPU inference is slow (Process > 150ms)**:
- Verify GPU is being used: Look for `[GPU]` in logs
- Check GPU utilization: `nvidia-smi` or `watch -n 1 nvidia-smi`
- Reduce input resolution (edit captureCanvas.width/height in viewer)

**If PLY generation is slow (Pack > 5ms)**:
- Increase pruning in stream_server.py:
  ```python
  prune_by_opacity=0.2,  # Was 0.1, now more aggressive
  ```

**If network is slow (Send > 30ms)**:
- Messages are too large (>3MB)
- Consider compression (gzip)
- Reduce Gaussian count with more pruning

---

### Issue: PLY files are too large (>5MB)

**Symptoms**: Bandwidth usage very high, laggy streaming

**Solutions**:

1. **Increase opacity pruning** (stream_server.py):
   ```python
   prune_by_opacity=0.15,  # Was 0.1
   ```

2. **Enable border trimming** (already enabled):
   ```python
   prune_border_gs=True,  # Already set
   ```

3. **Reduce depth range** (stream_server.py):
   ```python
   prune_by_depth_percent=0.8,  # Was 1.0, now trim 20% of far points
   ```

---

## Coordinate System Fixes

### Understanding the Coordinate Systems

**DepthAnything Output** (camera space):
- X: Right
- Y: Down
- Z: Forward (into scene)

**Gaussians Output** (world space):
- X: Right
- Y: ?
- Z: ?

**Three.js** (WebGL):
- X: Right
- Y: Up
- Z: Out of screen (towards camera)

**SparkJS**: Follows Three.js conventions

### If Gaussian splat is upside down:

Add Y-flip in viewer-gaussian.html:
```javascript
splatMesh.scale.set(1, -1, 1);  // Flip Y axis
```

### If Gaussian splat is backwards:

Add Z-flip:
```javascript
splatMesh.scale.set(1, 1, -1);  // Flip Z axis
```

### If Gaussian splat is rotated 90 degrees:

```javascript
splatMesh.rotation.y = Math.PI / 2;  // Rotate 90 degrees around Y
```

---

## Performance Expectations

### Giant Model on RTX 4090 (Runpod):
- Inference: 80-100ms
- PLY generation: 5-15ms
- Pack/Send: 10-20ms
- **Total: 100-135ms = 7-10 FPS**

### Giant Model on M1/M2 Mac (MPS):
- Inference: 120-180ms
- PLY generation: 10-20ms
- Pack/Send: 10-20ms
- **Total: 140-220ms = 4-7 FPS**

### Giant Model on CPU (not recommended):
- Inference: 500-1000ms
- **Total: >1 second = <1 FPS**

---

## Next Steps After Testing

### If everything works ✅:
1. Deploy to Runpod
2. Test over internet (not localhost)
3. Measure bandwidth usage
4. Optimize if needed (see GAUSSIAN_SPLAT_PLAN.md)
5. Consider adding quality toggle (point clouds ↔ Gaussians)

### If SparkJS doesn't work with our PLY ❌:
**Fallback options**:
1. Try @mkkellogg/gaussian-splats-3d library instead
2. Debug PLY format (save to file, inspect structure)
3. Consider server-side rendering (render Gaussians to images on GPU)

### If performance is too slow ❌:
1. Add frame skipping (process every 2nd or 3rd frame)
2. Implement adaptive quality (reduce Gaussian count dynamically)
3. Add quality toggle (let user choose point clouds for speed)

---

## Files Modified/Created

**Created**:
- `test-sparkjs.html` - SparkJS renderer test page
- `viewer-gaussian.html` - Main Gaussian splat viewer
- `TESTING_GUIDE.md` - This file

**Modified**:
- `stream_server.py` - Added giant model support, Gaussian extraction, model-dependent routing
- `src/depth_anything_3/utils/gsply_helpers.py` - Added `save_gaussian_ply_bytes()` function
- `GAUSSIAN_SPLAT_PLAN.md` - Updated implementation status

**Unchanged**:
- `viewer.html` - Original point cloud viewer (still works with base model)
