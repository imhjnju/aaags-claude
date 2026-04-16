# HarmonyOS 3DGS GPU (OpenCL) Design Spec

## Overview

Port the CPU 3DGS renderer to GPU using OpenCL on HarmonyOS (Maleoon GPU 920).
Approach: incremental module replacement with CPU ground truth validation.

## Target Environment

- Device: HarmonyOS phone with Maleoon GPU 920
- API: OpenCL (dynamically loaded via dlopen)
- NDK: `/home/randy/harmonyos/cmdline-tools/sdk/default/openharmony/native/`
- Compiler: `aarch64-unknown-linux-ohos-clang++`
- Deploy: `hdc file send` + `hdc shell`

## Project Structure

```
harmonyos_3dgs/
├── src/gpu/
│   ├── opencl_loader.h/.cpp        # dlopen + function pointer table (isolated)
│   ├── opencl_context.h/.cpp       # Device/queue/program + buffer pool + profiling
│   ├── preprocessor_gpu.h/.cpp     # Preprocess kernel dispatch
│   ├── tile_binner_gpu.h/.cpp      # Prefix sum + scatter kernel dispatch
│   ├── sorter_gpu.h/.cpp           # Radix sort + tile range identification
│   ├── rasterizer_gpu.h/.cpp       # Tile-based alpha blending kernel dispatch
│   └── kernels/
│       ├── preprocess.cl
│       ├── prefix_sum.cl
│       ├── scatter.cl              # Key/value generation from prefix sum
│       ├── radix_sort.cl
│       └── rasterize.cl
├── third_party/
│   └── OpenCL-Headers/             # Khronos v2023.12.14 (vendored)
└── ohos_build.sh
```

GPU headers in `src/gpu/` (not `include/gpu/`) to match CPU pattern.

## Key Design Decisions

### 1. OpenCL Dynamic Loading (opencl_loader)

Separate from context management. Tries multiple paths:
```cpp
const char* paths[] = {
    "libOpenCL.so",
    "/system/lib64/libOpenCL.so",
    "/vendor/lib64/libOpenCL.so",
    "/system/vendor/lib64/libOpenCL.so",
    nullptr
};
```
- All `clXxx` function pointers null-checked at load time
- `std::call_once` for thread-safe init
- Graceful fallback: if load fails, `main.cpp` uses CPU backend

### 2. GPU Buffer Management

**Problem:** `PreprocessOutput`/`BinningOutput` use raw `float*` (CPU memory).
GPU stages need `cl_mem` device buffers for zero-copy pipeline flow.

**Solution:** Add opaque device handle to output structs:
```cpp
struct PreprocessOutput {
    // ... existing float* fields (null for GPU path) ...
    void* device_data = nullptr;  // GPU: points to PreprocessDeviceBuffers
};

struct PreprocessDeviceBuffers {
    cl_mem means2D, depths, conics, rgb, opacities_2d, radii, tiles_touched;
    int num_valid;
};
```

GPU stages check `device_data`; CPU stages ignore it. `FrameAllocator` unused by GPU.

### 3. Buffer Pool

Pre-allocate `cl_mem` buffers sized by max Gaussian count. Reuse across frames.
```cpp
class GPUBufferPool {
    cl_mem getBuffer(size_t size);  // Returns existing or creates new
    void reset();                    // Mark all as available (no dealloc)
};
```
Avoids per-frame `clCreateBuffer` overhead on mobile drivers.

### 4. Kernel Embedding

`.cl` files embedded via CMake `xxd -i` at compile time.
Debug mode: `LOAD_KERNELS_FROM_DISK` option loads from filesystem for kernel iteration.

### 5. Error Handling

```cpp
#define CL_CHECK(err) do { if ((err) != CL_SUCCESS) \
    throw std::runtime_error("OpenCL error " + std::to_string(err) + \
    " at " __FILE__ ":" + std::to_string(__LINE__)); } while(0)
```

### 6. Profiling

`cl_event`-based per-kernel timing via `clGetEventProfilingInfo`.
Enabled with `CL_QUEUE_PROFILING_ENABLE` flag on command queue.

## Kernel Design

### preprocess.cl (1 work-item per Gaussian)
- Frustum test, NDC projection
- computeCov3D, computeCov2D (matching CPU math_utils exactly)
- SH evaluation (degree 0-3) with clamp [0,1]
- Output: means2D, depths, conics, rgb, opacities_2d, radii, tiles_touched

### prefix_sum.cl (parallel scan)
- Work-efficient Blelloch scan on `tiles_touched`
- Output: `point_offsets[N]`, `total_pairs`

### scatter.cl (1 work-item per Gaussian)
- For each valid Gaussian, write `(tile_id << 32 | depth_bits, gaussian_idx)` pairs
  at `point_offsets[i]` positions

### radix_sort.cl (multi-pass)
- Sort uint64 keys with associated uint32 values
- 4-bit radix, 16 passes for 64-bit keys
- Output: sorted keys/values, tile_ranges (scan for tile boundaries)

### rasterize.cl (1 work-group per tile, 256 threads = 16x16 pixels)
- Local memory for batch-loading Gaussians (`BLOCK_SIZE` at a time)
- Per-pixel alpha blending loop
- Early termination when T < 0.0001
- Work-group size adapted to `CL_DEVICE_MAX_WORK_GROUP_SIZE`

## Build System

```cmake
option(ENABLE_OPENCL "Build with OpenCL GPU backend" OFF)

if(ENABLE_OPENCL)
    target_compile_definitions(gs3d_core PUBLIC ENABLE_OPENCL=1)
    target_sources(gs3d_core PRIVATE
        src/gpu/opencl_loader.cpp src/gpu/opencl_context.cpp
        src/gpu/preprocessor_gpu.cpp src/gpu/tile_binner_gpu.cpp
        src/gpu/sorter_gpu.cpp src/gpu/rasterizer_gpu.cpp)
    target_include_directories(gs3d_core PRIVATE third_party/OpenCL-Headers)
    target_link_libraries(gs3d_core PRIVATE dl)
endif()
```

Cross-compile: `BUILD_TESTS=OFF` when `CMAKE_CROSSCOMPILING`.

## Implementation Order (TDD, each step verified on device)

1. Cross-compile CPU version to ARM64, run on phone → baseline
2. OpenCL loader + context init → print Maleoon GPU info
3. Preprocessor GPU kernel → compare output with CPU
4. Tile Binner GPU (prefix sum + scatter) → compare with CPU
5. Sorter GPU (radix sort) → compare with CPU
6. Rasterizer GPU kernel → pixel-level comparison with CPU render
7. End-to-end GPU render basketball.ply → compare with CPU reference

## Validation

Each step: render basketball.ply camera 0, compare with CPU output.
- PSNR > 30 dB (float precision differences expected)
- Visual diff: no structural artifacts
- Performance: measure GPU kernel time vs CPU total time
