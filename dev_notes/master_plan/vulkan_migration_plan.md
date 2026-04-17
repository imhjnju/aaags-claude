# Vulkan Migration Plan: AAA-Gaussians for HarmonyOS

## 1. Project Context

**Goal**: Port AAA-Gaussians (Python+CUDA) to C++17 + Vulkan Compute for HarmonyOS NEXT.
Train 2000 steps on basketball dataset, pixel-level consistent with Python reference.

**Pivotal decision**: Replace OpenCL backend entirely with Vulkan. No OpenCL residue.

### Why Vulkan over OpenCL

| Aspect | OpenCL (old) | Vulkan (new) |
|--------|-------------|--------------|
| Subgroup shuffle | Undeclared, works by luck on Maleoon | Mandatory in Vulkan 1.3 Roadmap 2022 |
| Subgroup ballot | Unavailable on Maleoon | Mandatory in Vulkan 1.3 Roadmap 2022 |
| Memory model | Informal | Formal (`vulkanMemoryModel` mandatory in 1.3) |
| SPIR-V | N/A (source compilation) | Native, offline glslc |
| Driver quality | Extension reporting unreliable | 1.3 = all features mandatory, no guessing |
| Future | Deprecated on HarmonyOS | Primary GPU API on HarmonyOS NEXT |

### Development Environment

| Spec | Dev Machine (Tegra Thor) | Target (Maleoon 920) |
|------|--------------------------|----------------------|
| Vulkan | 1.4 | 1.3 |
| GPU | NVIDIA Tegra Thor | Huawei Maleoon 920 |
| Subgroup size | 32 | ~16 or 32 (TBD) |
| Shared memory | 48 KB | 16 KB min (likely 32+ KB) |
| Workgroup max | 1024 | 128 min (likely 256+) |
| Push constants | 256 B | 128 B min |
| Storage buffers | 1M/stage | 4 min (likely 30+) |
| All subgroup ops | Yes | Yes (Vulkan 1.3 Roadmap 2022) |

**Constraint**: Design for Maleoon minimums. Test on Tegra Thor. Validate on device.

### Dataset

- **Point cloud**: `/home/robota/Downloads/basketball/sparse/0/points3D.ply` (78 KB)
- **Images**: `/home/robota/Downloads/basketball/images/` (77 images)
- **Training**: 2000 steps, all hyperparameters identical to Python reference

---

## 2. Architecture

### Code Structure

```
harmonyos_3dgs/
├── include/
│   ├── (existing headers — unchanged)
│   ├── preprocessor_vk.h
│   ├── rasterizer_vk.h
│   ├── rasterizer_backward_vk.h
│   ├── preprocessor_backward_vk.h
│   ├── tile_binner_vk.h
│   ├── sorter_vk.h
│   └── trainer_vk.h
├── src/
│   ├── cpu/          (keep as TDD reference)
│   ├── gpu/          (OpenCL — will be removed after Vulkan works)
│   └── vulkan/       ← NEW
│       ├── vk_context.h/cpp       # Instance, device, queue, command pool
│       ├── vk_buffer.h/cpp        # SSBO + staging buffer abstraction
│       ├── vk_pipeline.h/cpp      # Compute pipeline + descriptor set management
│       ├── vk_shader.h/cpp        # SPIR-V loading (embedded or file)
│       ├── preprocessor_vk.cpp
│       ├── tile_binner_vk.cpp
│       ├── sorter_vk.cpp
│       ├── rasterizer_vk.cpp
│       ├── rasterizer_backward_vk.cpp
│       ├── preprocessor_backward_vk.cpp
│       ├── trainer_vk.cpp
│       └── shaders/
│           ├── common.glsl         # Shared types, constants, SH basis
│           ├── preprocess.comp     # Preprocessing kernel
│           ├── scatter.comp        # Tile binning (Gaussian → tile pairs)
│           ├── prefix_sum.comp     # Blelloch parallel scan
│           ├── radix_sort.comp     # 4-bit radix sort on uint64 keys
│           ├── tile_range.comp     # Identify per-tile ranges
│           ├── rasterize.comp      # Per-tile alpha blending
│           ├── rasterize_backward.comp
│           ├── preprocess_backward.comp
│           └── adam_step.comp      # Adam optimizer (optional — may run on CPU)
├── tests/
│   ├── test_vk_context.cpp        # Vulkan init, device query
│   ├── test_vk_compute.cpp        # Hello-world compute shader
│   ├── test_preprocessor_vk.cpp   # GPU vs CPU preprocessing
│   ├── test_sorter_vk.cpp         # GPU vs CPU sort
│   ├── test_rasterizer_vk.cpp     # GPU vs CPU rendering
│   ├── test_rasterizer_backward_vk.cpp
│   ├── test_preprocessor_backward_vk.cpp
│   ├── test_trainer_vk.cpp        # GPU training loss decrease
│   └── test_e2e_basketball.cpp    # Full 2000-step validation
```

### Interface Pattern

All Vulkan implementations follow existing abstract interfaces:
```cpp
class PreprocessorVK : public Preprocessor { ... };
class RasterizerVK : public Rasterizer { ... };
```

This allows CPU-vs-GPU comparison tests using the same API.

### Shader Compilation

1. GLSL `.comp` files in `src/vulkan/shaders/`
2. CMake custom command: `glslc -O --target-env=vulkan1.1 shader.comp -o shader.spv`
   - Target vulkan1.1 (not 1.3) for maximum compatibility
   - Subgroup ops available via SPIR-V capability, not API version
3. Embed `.spv` as C arrays via `xxd -i` or CMake CONFIGURE_FILE
4. Load at pipeline creation time (no runtime compilation)

### Memory Strategy

- **Device-local buffers** for all GPU data (positions, SH, scales, etc.)
- **Staging buffers** (host-visible) for upload/download
- **Upload once** at pipeline start; **download** only rendered image + loss
- During training: keep all data on GPU. Only copy back for loss computation.
- Push constants for per-frame uniforms (camera, config) — fits in 128 bytes

### Synchronization

All compute dispatches on same queue → use pipeline barriers:
```
preprocess dispatch → BARRIER → scatter dispatch → BARRIER → sort dispatch →
BARRIER → tile_range dispatch → BARRIER → rasterize dispatch
```
Barrier: `VK_ACCESS_SHADER_WRITE_BIT → VK_ACCESS_SHADER_READ_BIT`

---

## 3. Critical Python-C++ Gaps

These must be closed BEFORE Vulkan migration (or during Phase 6):

### 3.1 MCMC Densification (replaces clone/split/prune)

Python's AAA-Gaussians does NOT use traditional 3DGS densification.
Instead: **relocate dead Gaussians + grow by 5%**.

```
Every 100 iterations (from iter 500 to 25000):
  1. dead_mask = (opacity <= 0.005)
  2. relocate_gs(dead_mask):
     - Sample alive Gaussians proportional to opacity
     - Clone to dead positions
     - Adjust: new_opacity = old^(1/N), new_scale = old * (new_op/old_op)^(1/3)
  3. add_new_gs(cap_max):
     - Grow by 5% per step
     - Sample + clone with opacity/scale adjustment
```

**Impact**: Existing `DensityController` (clone/split/prune) must be rewritten.

### 3.2 DSSIM Loss (20% weight)

```
loss = 0.8 * L1 + 0.2 * (1 - SSIM) + 0.01 * mean(|opacity|) + 0.01 * mean(|scale|)
```

Need: fused SSIM implementation (11x11 window, C1=0.0001, C2=0.0009).
Current C++: L1 only.

### 3.3 Position Noise Injection

```python
noise = randn_like(xyz) * sigmoid(1 - opacity) * noise_lr * xyz_lr
noise = bmm(covariance_matrix, noise.unsqueeze(-1))
xyz += noise
```

Not in current C++. Must add after optimizer step.

### 3.4 Learning Rate Schedule

Python uses exponential with sin warmup:
```python
delay_rate = 0.01 + 0.99 * sin(0.5 * pi * clamp(step/delay_steps, 0, 1))
log_lerp = exp(log(lr_init) * (1-t) + log(lr_final) * t)
lr = delay_rate * log_lerp
```

Current C++: simpler exponential without delay. Must match exactly.

### 3.5 Adam Optimizer Per-Parameter LRs

| Parameter | LR |
|-----------|----|
| xyz | 1.6e-4 * spatial_lr_scale → 1.6e-6 (exponential decay) |
| features_dc | 2.5e-3 |
| features_rest | 2.5e-3 / 20 = 1.25e-4 |
| opacity | 0.05 |
| scaling | 0.005 |
| rotation | 0.001 |

Adam: beta1=0.9, beta2=0.999, eps=1e-15

### 3.6 SH Feature Layout

Python: `features_dc [N,1,3]` + `features_rest [N,15,3]` (degree 3 → 16 basis)
C++: `sh_coeffs [N*max_coeffs*3]` (interleaved: basis_k * 3 + channel)

The layouts differ. Must ensure consistent addressing in GLSL shaders.

### 3.7 SH Degree Schedule

Every 1000 iterations: increment `active_sh_degree` (0 → 1 → 2 → 3).
At 2000 steps: degree will be 2.

### 3.8 3D Mip Filter Initialization

```python
def compute_3D_filter(xyz, cameras):
    for each Gaussian, for each camera:
        project to screen; if valid: track min distance
    filter_3D = min_distance / max_focal * sqrt(0.3)
```

Must be computed once from all training cameras at init.

---

## 4. Phased Implementation Plan

### Phase 0: Environment Setup (this session)

1. Install `glslc` (Vulkan shader compiler)
2. Write migration plan (this document)
3. Create Vulkan test plan
4. Update CMakeLists.txt with Vulkan build option
5. Validate: compile trivial GLSL → SPIR-V

### Phase 1: Vulkan Infrastructure

**Goal**: Minimal Vulkan abstraction that can create device, load shader, dispatch compute, read results.

| Component | Description |
|-----------|-------------|
| `vk_context` | Instance + physical device + logical device + compute queue + command pool |
| `vk_buffer` | Create/destroy SSBO, staging buffer upload/download, map/unmap |
| `vk_pipeline` | Create compute pipeline from SPIR-V, manage descriptor sets |
| `vk_shader` | Load embedded SPIR-V, create shader module |

**TDD Gate**: `test_vk_compute.cpp` — dispatch `add_one.comp` (each element += 1), verify output.

### Phase 2: Forward - Preprocessing

**Shader**: `preprocess.comp` — 1 invocation per Gaussian

Computes:
- View-space position (multiply by view matrix)
- Screen-space position (project via viewproj)
- 3D covariance → 2D covariance (Jacobian chain)
- Conic (inverse 2D covariance)
- SH evaluation to RGB
- Screen radius + tiles_touched
- 3D mip filter dilation (if eval_3D)
- Frustum culling
- gauss2screen matrix (if eval_3D)

**Push constants**: Camera struct (viewmatrix, viewprojmatrix, campos, tanfov, resolution)
**SSBOs**: positions, scales, rotations, sh_coeffs, opacities, filter_3D → means2D, depths, conics, rgb, radii, tiles_touched, etc.

**TDD Gate**: `test_preprocessor_vk.cpp`
- Single Gaussian: exact match with CPU preprocessor
- 100 Gaussians: per-Gaussian max error < 1e-5
- Frustum culling: culled Gaussians have radii=0

### Phase 3: Forward - Sorting

**Shaders**: `scatter.comp` + `prefix_sum.comp` + `radix_sort.comp` + `tile_range.comp`

Pipeline:
1. **Prefix sum on tiles_touched** → compute offsets for scatter
2. **Scatter**: emit (key=tile_y|tile_x|depth, value=gauss_idx) pairs
3. **Radix sort**: 4-bit radix, 16 passes on 64-bit keys, subgroup shuffle_up
4. **Tile range**: scan sorted keys, mark per-tile start/end

**TDD Gate**: `test_sorter_vk.cpp`
- 1000 random key-value pairs: sorted output matches std::sort
- Tile ranges: match CPU tile binner output
- Edge cases: empty tiles, single Gaussian per tile, max Gaussians per tile

### Phase 4: Forward - Rasterization

**Shader**: `rasterize.comp` — 1 workgroup per tile (16x16 = 256 threads)

Per-pixel alpha blending:
- Load sorted Gaussians for this tile from tile_ranges
- Shared memory batch loading (collaborative)
- Evaluate Gaussian contribution: `alpha = opacity * exp(-0.5 * mahalanobis)`
- Alpha compositing: `color += alpha * T * gaussian_rgb; T *= (1-alpha)`
- Store: rendered_image, T_final, n_contrib (for backward)

For eval_3D: plane intersection evaluation using gauss2screen matrix.

**TDD Gate**: `test_rasterizer_vk.cpp`
- Single Gaussian at image center: exact pixel match with CPU
- 10 overlapping Gaussians: per-pixel error < 1e-4
- Full forward pipeline: basketball frame 0, PSNR > 80 dB vs CPU

### Phase 5: Backward Pipeline

**Shaders**: `rasterize_backward.comp` + `preprocess_backward.comp`

Rasterizer backward:
- Replay forward per pixel, traverse back-to-front
- Accumulate d_means2D, d_conics, d_rgb, d_opacities_2d
- Atomic float add via compareExchange loop (Vulkan doesn't have native atomicAdd for float)

Preprocessor backward (4 chains):
- Chain A: d_conics → d_cov2D → d_cov3D → d_scales, d_rotations
- Chain B: d_rgb → d_sh_coeffs
- Chain C: d_opacities_2d → d_raw_opacities
- Chain D: d_means2D → d_positions

**TDD Gate**: `test_*_backward_vk.cpp`
- Finite difference gradient verification: |analytical - numerical| / max(|analytical|, 1e-7) < 1e-3
- GPU vs CPU backward: per-element max error < 1e-4

### Phase 6: Training Pipeline

Close all Python-C++ gaps:
1. MCMC densification (relocate + grow)
2. DSSIM loss (fused SSIM)
3. Noise injection
4. Regularization (opacity + scale)
5. LR schedule with warmup delay
6. Adam per-parameter LRs
7. SH degree scheduling
8. 3D mip filter initialization

**TDD Gate**: `test_trainer_vk.cpp`
- 100-step training: loss monotonically decreasing (after step 10)
- Densification: Gaussian count changes after step 500
- LR values at step 0, 500, 1000, 2000: match Python reference

### Phase 7: End-to-End Validation

1. Run Python reference: train 2000 steps on basketball, save per-step loss + final rendered images
2. Run C++ Vulkan: train 2000 steps on same data
3. Compare:
   - Per-step loss curve: max relative error < 5% after step 100
   - Final rendered images: PSNR > 25 dB (visual match)
   - SSIM > 0.90
4. If not matching: bisect which step diverges, compare intermediate values

---

## 5. Risk Matrix

| # | Risk | Probability | Impact | Mitigation |
|---|------|-------------|--------|------------|
| 1 | Vulkan atomicAdd(float) not available on Maleoon | Medium | High | Use CAS loop: `atomicCompSwap` with float reinterpret. Standard pattern. |
| 2 | Shared memory 16 KB not enough for 16x16 tile rasterizer | Medium | High | Design for 16 KB: batch size = 16KB / (per-Gaussian data). Reduce batch if needed. |
| 3 | DSSIM backward complex to implement | Low | Medium | Implement on CPU first. SSIM gradient is analytically tractable. Move to GPU later if bottleneck. |
| 4 | Radix sort subgroup shuffle not working on Maleoon | Low | Medium | Vulkan 1.3 mandates it. Fallback: shared memory based sort. |
| 5 | Python MCMC densification uses CUDA-specific relocation kernel | Medium | Medium | Rewrite in C++. Math is simple: opacity^(1/N), scale adjustment. |
| 6 | Numerical divergence between NVIDIA and Maleoon GPUs | High | High | All math uses `precise` qualifier. No fast-math. Test on both. |
| 7 | Push constant 128B limit on Maleoon vs 256B on Tegra | High | Low | Design camera struct ≤ 128B. Move overflow to UBO. |
| 8 | glslc not available — cannot compile shaders | Low | Blocking | Install via Vulkan SDK or build from source. |
| 9 | Flutter Vulkan driver crash pattern affects compute | Low | Critical | Compute pipeline uses different driver path. Validate early with minimal test. |
| 10 | SSIM window kernel too slow for 2000 training steps | Medium | Medium | Implement as separable 1D convolutions. Consider CPU-only SSIM. |

---

## 6. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-16 | OpenCL → Vulkan migration | Vulkan 1.3 mandates subgroup ops; OpenCL extensions unreliable on Maleoon |
| 2026-04-16 | Offline GLSL → SPIR-V (glslc) | No runtime shader compilation. Faster startup, more portable. |
| 2026-04-16 | Keep CPU reference as TDD oracle | CPU implementations are proven correct. Vulkan tested against CPU. |
| 2026-04-16 | MCMC densification (match Python) | Python AAA-Gaussians uses relocate+grow, NOT clone/split. Must match for pixel consistency. |
| 2026-04-16 | Target Vulkan 1.1 SPIR-V | Maximum compatibility. Subgroup ops via SPIR-V capability, not API version. |
| 2026-04-16 | Push constants ≤ 128B | Maleoon 920 minimum. Camera struct must fit. |
