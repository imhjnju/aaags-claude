# AAA-Gaussians Feature Status

## Enabled Features

| Feature | Stage | Description |
|---------|-------|-------------|
| eval_3D / gauss2screen matrix | Preprocess | Full 3D Gaussian evaluation via 4x4 transform matrix |
| Scale dilation + 3D mip filter | Preprocess | Frequency-aware adaptive scale with filter_3D |
| Camera-in-ellipsoid detection | Preprocess | Skip Gaussian when camera is inside ellipsoid |
| Global 3D frustum culling | Preprocess | Full-screen frustum culling via maxContribGaussianFrustum3D |
| Tight opacity bounding | Preprocess | Dynamic cutoff based on opacity: min(11.11, 2*opt) |
| Opacity dilation factor | Preprocess | Opacity correction after scale dilation |
| Screen-space AABB (Hahlbohm) | Preprocess | Screen-space bounding box for tile coverage |
| **View-space AABB fallback** | **Preprocess** | **Fallback to view-space AABB when screen-space fails (near-plane overflow)** |
| Per-pixel 3D evaluation (plane-based) | Rasterize | Per-pixel Mahalanobis distance via cross-product of two planes |
| 4-bit Radix Sort (subgroup shuffle_up) | Sort | 16-pass 64-bit key-value radix sort replacing bitonic sort |
| cov3D_inv / mean_offset computation | Preprocess | Infrastructure computed but not consumed (reserved for future use) |
| depthAlongRay / invertMatrix4x4 | Math | Math utilities implemented but not consumed (reserved for future use) |

> **Bold** = fallback feature, only triggered when primary path fails.

## Disabled Features (cause tile-boundary block artifacts)

| Feature | Reason |
|---------|--------|
| Per-tile 3D frustum culling | Low-opacity Gaussians culled in some tiles but retained in neighbors, causing tile-boundary discontinuities |
| Per-tile depth key (depthAlongRay) | Different tiles compute different depth keys for the same Gaussian, causing inconsistent sort orders at boundaries |
| Per-pixel kBuffer sorting (K=4) | K=4 window too small; pop timing differs across pixels, amplifying tile-boundary artifacts |

These features require **hierarchical per-pixel sorting (StopThePop)** to work correctly. StopThePop relies on CUDA warp-level primitives (`__shfl_sync`, `__ballot_sync`, cooperative groups) that have no direct OpenCL equivalent on the Maleoon 920 GPU.

## Not Implemented (training / CUDA-specific)

| Feature | Reason |
|---------|--------|
| Backward pass (gradient computation) | Training-only; HarmonyOS deployment is inference |
| Hierarchical 3-level sorting (StopThePop) | Deeply coupled to CUDA warp semantics |
| Load balancing (cooperative groups) | CUDA cooperative groups not available in OpenCL |
| Rectangle bounding (vs circle) | Redundant with per-tile culling disabled; marginal benefit |
| Debug visualization system | Development utility, not production feature |

## Performance Summary (basketball.ply, 400K Gaussians, 720x960, Maleoon 920)

| Version | Total | Sort | Rasterize |
|---------|-------|------|-----------|
| Original (bitonic sort) | 5,040 ms | 4,755 ms | 127 ms |
| + Radix Sort | 791 ms | 193 ms | 505 ms |
| + All AAA features (current) | 777 ms | 178 ms | 492 ms |
| **Speedup** | **6.5x** | **26.7x** | — |
