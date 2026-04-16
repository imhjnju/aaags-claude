# GPU OpenCL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the CPU 3DGS renderer to OpenCL GPU on HarmonyOS (Maleoon GPU 920), validated against CPU reference renders of basketball.ply.

**Architecture:** Incremental module replacement — each GPU module inherits the same abstract interface as CPU, dispatches OpenCL kernels, and passes data via `cl_mem` handles stored in output structs. OpenCL is dynamically loaded via `dlopen` at runtime.

**Tech Stack:** C++17, OpenCL 1.2+, HarmonyOS NDK (aarch64-unknown-linux-ohos-clang++), CMake, hdc

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `CMakeLists.txt` | Modify | Add `ENABLE_OPENCL` option, GPU sources, cross-compile guard |
| `include/types.h` | Modify | Add `void* device_data` to `PreprocessOutput` and `BinningOutput` |
| `src/gpu/opencl_loader.h` | Create | `dlopen`/`dlsym` function pointer table |
| `src/gpu/opencl_loader.cpp` | Create | Multi-path `dlopen`, null-check all pointers |
| `src/gpu/opencl_context.h` | Create | Device/queue/program management, buffer pool, profiling |
| `src/gpu/opencl_context.cpp` | Create | Init, kernel build, buffer ops, CL_CHECK macro |
| `src/gpu/preprocessor_gpu.h` | Create | GPU preprocessor interface |
| `src/gpu/preprocessor_gpu.cpp` | Create | Buffer upload, kernel dispatch, readback |
| `src/gpu/tile_binner_gpu.h` | Create | GPU tile binner interface |
| `src/gpu/tile_binner_gpu.cpp` | Create | Prefix sum + scatter kernel dispatch |
| `src/gpu/sorter_gpu.h` | Create | GPU sorter interface |
| `src/gpu/sorter_gpu.cpp` | Create | Radix sort kernel dispatch |
| `src/gpu/rasterizer_gpu.h` | Create | GPU rasterizer interface |
| `src/gpu/rasterizer_gpu.cpp` | Create | Tile-based rasterize kernel dispatch |
| `src/gpu/kernels/preprocess.cl` | Create | Per-Gaussian projection + SH + cov2D kernel |
| `src/gpu/kernels/prefix_sum.cl` | Create | Work-efficient parallel scan |
| `src/gpu/kernels/scatter.cl` | Create | Key/value pair generation |
| `src/gpu/kernels/radix_sort.cl` | Create | 64-bit radix sort |
| `src/gpu/kernels/rasterize.cl` | Create | Per-tile alpha blending with shared memory |
| `src/main.cpp` | Modify | Add `--gpu` flag, construct GPU backends |
| `ohos_build.sh` | Create | Cross-compile script for HarmonyOS ARM64 |
| `third_party/OpenCL-Headers/` | Create | Vendored Khronos headers (v2023.12.14) |

---

### Task 1: Cross-Compile CPU Version to ARM64 + Run on Phone

**Goal:** Establish baseline — verify CPU renderer works on HarmonyOS device.

**Files:**
- Create: `ohos_build.sh`
- Modify: `CMakeLists.txt` (cross-compile guard for tests)

- [ ] **Step 1: Create ohos_build.sh**

```bash
#!/bin/bash
set -e
NDK=/home/randy/harmonyos/cmdline-tools/sdk/default/openharmony/native
HOS_HOS_SDK=$NDK
cmake -B build-ohos \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/ohos.toolchain.cmake \
    -DOHOS_ARCH=arm64-v8a \
    -DBUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-ohos -j$(nproc)
echo "Binary: build-ohos/gs3d_render"
file build-ohos/gs3d_render
```

- [ ] **Step 2: Guard BUILD_TESTS for cross-compilation in CMakeLists.txt**

Add after `option(BUILD_TESTS ...)`:
```cmake
if(CMAKE_CROSSCOMPILING)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
endif()
```

- [ ] **Step 3: Build for ARM64**

Run: `chmod +x ohos_build.sh && ./ohos_build.sh`
Expected: `build-ohos/gs3d_render` is an aarch64 ELF binary

- [ ] **Step 4: Push to phone and run**

```bash
HDC=/home/randy/harmonyos/cmdline-tools/sdk/default/openharmony/toolchains/hdc
$HDC file send build-ohos/gs3d_render /data/local/tmp/
$HDC file send ../basketball.ply /data/local/tmp/
$HDC file send ../cameras.json /data/local/tmp/
$HDC shell "cd /data/local/tmp && chmod +x gs3d_render && ./gs3d_render basketball.ply cameras.json 0"
```
Expected: Renders successfully, outputs `output.ppm`

- [ ] **Step 5: Retrieve and verify result**

```bash
$HDC file recv /data/local/tmp/output.ppm ./ohos_cpu_output.ppm
# Convert to PNG and compare with PC CPU render
```
Expected: Identical (or near-identical) to PC CPU render

- [ ] **Step 6: Commit**

```bash
git add ohos_build.sh CMakeLists.txt
git commit -m "feat: add OHOS ARM64 cross-compilation support"
```

---

### Task 2: Vendor OpenCL Headers + OpenCL Loader

**Goal:** Dynamic OpenCL loading with multi-path fallback.

**Files:**
- Create: `third_party/OpenCL-Headers/CL/cl.h` (and related headers)
- Create: `src/gpu/opencl_loader.h`
- Create: `src/gpu/opencl_loader.cpp`

- [ ] **Step 1: Download and vendor Khronos OpenCL headers**

```bash
cd harmonyos_3dgs
mkdir -p third_party/OpenCL-Headers
# Download CL/cl.h, CL/cl_platform.h, CL/cl_version.h from Khronos
# Pin to v2023.12.14
```

- [ ] **Step 2: Write opencl_loader.h**

```cpp
#pragma once
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

// Function pointer typedefs
typedef cl_int (*pfn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*pfn_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
// ... all ~25 needed CL functions ...

struct OpenCLFunctions {
    pfn_clGetPlatformIDs clGetPlatformIDs;
    pfn_clGetDeviceIDs clGetDeviceIDs;
    pfn_clCreateContext clCreateContext;
    pfn_clCreateCommandQueue clCreateCommandQueue;
    pfn_clCreateBuffer clCreateBuffer;
    pfn_clReleaseMemObject clReleaseMemObject;
    pfn_clCreateProgramWithSource clCreateProgramWithSource;
    pfn_clBuildProgram clBuildProgram;
    pfn_clCreateKernel clCreateKernel;
    pfn_clSetKernelArg clSetKernelArg;
    pfn_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
    pfn_clEnqueueReadBuffer clEnqueueReadBuffer;
    pfn_clEnqueueWriteBuffer clEnqueueWriteBuffer;
    pfn_clFinish clFinish;
    pfn_clReleaseKernel clReleaseKernel;
    pfn_clReleaseProgram clReleaseProgram;
    pfn_clReleaseCommandQueue clReleaseCommandQueue;
    pfn_clReleaseContext clReleaseContext;
    pfn_clGetDeviceInfo clGetDeviceInfo;
    pfn_clGetProgramBuildInfo clGetProgramBuildInfo;
    pfn_clGetEventProfilingInfo clGetEventProfilingInfo;
    pfn_clWaitForEvents clWaitForEvents;
    pfn_clReleaseEvent clReleaseEvent;
    // add more as needed
};

bool loadOpenCL(OpenCLFunctions& funcs);
void unloadOpenCL();
```

- [ ] **Step 3: Write opencl_loader.cpp**

Multi-path `dlopen`, null-check every pointer, `std::call_once` for thread safety.

- [ ] **Step 4: Test on phone**

Push a small test binary that calls `loadOpenCL()` and prints device name.

```bash
$HDC shell "/data/local/tmp/gs3d_render --cl-info"
```
Expected: `OpenCL device: Maleoon GPU 920` (or similar)

- [ ] **Step 5: Commit**

```bash
git add third_party/OpenCL-Headers src/gpu/opencl_loader.*
git commit -m "feat: add OpenCL dynamic loader with multi-path dlopen"
```

---

### Task 3: OpenCL Context + Buffer Pool + Error Handling

**Goal:** Device management, buffer allocation, kernel compilation, profiling.

**Files:**
- Create: `src/gpu/opencl_context.h`
- Create: `src/gpu/opencl_context.cpp`

- [ ] **Step 1: Write opencl_context.h**

```cpp
#pragma once
#include "opencl_loader.h"
#include <string>
#include <vector>
#include <unordered_map>

#define CL_CHECK(fn_call) do { \
    cl_int _err = (fn_call); \
    if (_err != CL_SUCCESS) \
        throw std::runtime_error("OpenCL error " + std::to_string(_err) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
} while(0)

class OpenCLContext {
public:
    bool init();
    void release();

    cl_kernel buildKernel(const char* source, size_t len, const char* name,
                          const char* build_opts = "");
    cl_mem createBuffer(size_t size, cl_mem_flags flags, void* host = nullptr);
    void readBuffer(cl_mem buf, void* dst, size_t size);
    void writeBuffer(cl_mem buf, const void* src, size_t size);
    void enqueueKernel(cl_kernel k, int dim, const size_t* global, const size_t* local);
    void finish();

    // Device capabilities
    size_t maxWorkGroupSize() const { return max_wg_size_; }
    size_t maxLocalMemSize() const { return max_local_mem_; }
    cl_uint maxComputeUnits() const { return max_cu_; }
    size_t maxAllocSize() const { return max_alloc_; }

    // Profiling
    double getEventTimeMs(cl_event ev);

    const OpenCLFunctions& cl() const { return cl_; }

private:
    OpenCLFunctions cl_{};
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    size_t max_wg_size_ = 0;
    size_t max_local_mem_ = 0;
    cl_uint max_cu_ = 0;
    size_t max_alloc_ = 0;
};
```

- [ ] **Step 2: Write opencl_context.cpp**

Init: load OpenCL, find GPU device, create context + profiling-enabled queue, query device caps.
buildKernel: `clCreateProgramWithSource` + `clBuildProgram` + error log on failure.

- [ ] **Step 3: Test on phone — print device capabilities**

- [ ] **Step 4: Commit**

```bash
git add src/gpu/opencl_context.*
git commit -m "feat: add OpenCL context with device caps and error handling"
```

---

### Task 4: Modify types.h — Add device_data to Output Structs

**Goal:** Enable zero-copy GPU pipeline by adding opaque device handles.

**Files:**
- Modify: `include/types.h`

- [ ] **Step 1: Add device_data to PreprocessOutput and BinningOutput**

```cpp
struct PreprocessOutput {
    // ... existing fields unchanged ...
    void* device_data = nullptr;  // GPU: points to device buffer struct
};

struct BinningOutput {
    // ... existing fields unchanged ...
    void* device_data = nullptr;  // GPU: points to device buffer struct
};
```

- [ ] **Step 2: Verify CPU tests still pass**

Run: `cd build && ctest --output-on-failure`
Expected: All 47 tests pass (new field defaults to nullptr, no behavior change)

- [ ] **Step 3: Commit**

```bash
git add include/types.h
git commit -m "feat: add device_data opaque handle to output structs for GPU path"
```

---

### Task 5: Preprocessor GPU Kernel

**Goal:** GPU-accelerated projection, SH evaluation, covariance computation.

**Files:**
- Create: `src/gpu/kernels/preprocess.cl`
- Create: `src/gpu/preprocessor_gpu.h`
- Create: `src/gpu/preprocessor_gpu.cpp`
- Modify: `CMakeLists.txt` (add GPU sources under ENABLE_OPENCL)

- [ ] **Step 1: Write preprocess.cl kernel**

Port `preprocessor_cpu.cpp` logic to OpenCL. One work-item per Gaussian.
Include: `transformPoint4x3`, `transformPoint4x4`, `computeCov3D`, `computeCov2D`,
`computeColorFromSH` (degree 0-3), `ndc2Pix`, `getRect`.

Key differences from CPU:
- All math functions use OpenCL built-ins (`native_exp`, `native_sqrt`, etc.)
- Matrix convention: same column-major as CPU (important — see debugging-report.md)
- SH clamp to [0,1] (matches our CPU fix)

- [ ] **Step 2: Write preprocessor_gpu.h/.cpp**

```cpp
class PreprocessorGPU : public Preprocessor {
public:
    PreprocessorGPU(OpenCLContext& ctx);
    PreprocessOutput process(const GaussianData& g, const Camera& cam,
                             const RenderConfig& cfg, FrameAllocator& alloc) override;
private:
    OpenCLContext& ctx_;
    cl_kernel kernel_ = nullptr;
    // Persistent GPU buffers for model data
    cl_mem d_positions_ = nullptr;
    cl_mem d_sh_coeffs_ = nullptr;
    cl_mem d_scales_ = nullptr;
    cl_mem d_rotations_ = nullptr;
    cl_mem d_opacities_ = nullptr;
    bool model_uploaded_ = false;
};
```

Host side: upload model data once, dispatch kernel, read back `radii`/`tiles_touched` to CPU
(needed by tile binner), keep rest on GPU in `device_data`.

- [ ] **Step 3: Update CMakeLists.txt with ENABLE_OPENCL**

- [ ] **Step 4: Cross-compile, push to phone, compare preprocessor output with CPU**

Test approach: render basketball.ply cam 0 with GPU preprocessor + CPU binner/sorter/rasterizer.
Compare output image with full-CPU render.

- [ ] **Step 5: Commit**

```bash
git add src/gpu/kernels/preprocess.cl src/gpu/preprocessor_gpu.* CMakeLists.txt
git commit -m "feat: add GPU preprocessor kernel (projection + SH + cov2D)"
```

---

### Task 6: Tile Binner GPU (Prefix Sum + Scatter)

**Goal:** GPU parallel prefix sum on tiles_touched + key/value scatter.

**Files:**
- Create: `src/gpu/kernels/prefix_sum.cl`
- Create: `src/gpu/kernels/scatter.cl`
- Create: `src/gpu/tile_binner_gpu.h`
- Create: `src/gpu/tile_binner_gpu.cpp`

- [ ] **Step 1: Write prefix_sum.cl**

Work-efficient Blelloch parallel scan. Handle arrays larger than one work-group
with a two-level scan (scan blocks, scan block sums, add back).

- [ ] **Step 2: Write scatter.cl**

Each work-item writes `(tile_id << 32 | depth_bits, gaussian_idx)` pairs at
the prefix-sum offset for its Gaussian. Matches `tile_binner_cpu.cpp` inner loop.

- [ ] **Step 3: Write tile_binner_gpu.h/.cpp**

Dispatch prefix_sum kernel, read `total_pairs` back to host, allocate key/value
buffers, dispatch scatter kernel.

- [ ] **Step 4: Validate on phone — compare binning output with CPU**

- [ ] **Step 5: Commit**

```bash
git add src/gpu/kernels/prefix_sum.cl src/gpu/kernels/scatter.cl src/gpu/tile_binner_gpu.*
git commit -m "feat: add GPU tile binner (parallel prefix sum + scatter)"
```

---

### Task 7: Sorter GPU (Radix Sort)

**Goal:** GPU radix sort for 64-bit keys with 32-bit values.

**Files:**
- Create: `src/gpu/kernels/radix_sort.cl`
- Create: `src/gpu/sorter_gpu.h`
- Create: `src/gpu/sorter_gpu.cpp`

- [ ] **Step 1: Write radix_sort.cl**

4-bit radix sort (16 passes for 64-bit keys). Each pass:
1. Count histogram per radix digit
2. Prefix sum on histogram
3. Scatter keys and values to sorted positions

After sorting: scan for tile boundaries to fill `tile_ranges`.

- [ ] **Step 2: Write sorter_gpu.h/.cpp**

Dispatch radix sort passes, then tile range identification kernel.

- [ ] **Step 3: Validate on phone — compare sorted output with CPU**

- [ ] **Step 4: Commit**

```bash
git add src/gpu/kernels/radix_sort.cl src/gpu/sorter_gpu.*
git commit -m "feat: add GPU radix sort for tile-based rendering"
```

---

### Task 8: Rasterizer GPU Kernel

**Goal:** GPU tile-based alpha blending with shared memory optimization.

**Files:**
- Create: `src/gpu/kernels/rasterize.cl`
- Create: `src/gpu/rasterizer_gpu.h`
- Create: `src/gpu/rasterizer_gpu.cpp`

- [ ] **Step 1: Write rasterize.cl**

One work-group per tile. 256 threads (16x16 pixels per tile).
Shared memory batch loading of Gaussians (BLOCK_SIZE at a time).
Per-pixel alpha blending with early termination.
Work-group size adapted from `OpenCLContext::maxWorkGroupSize()`.

- [ ] **Step 2: Write rasterizer_gpu.h/.cpp**

Dispatch rasterize kernel, read back output image to host.

- [ ] **Step 3: Validate on phone — pixel-level comparison with CPU render**

- [ ] **Step 4: Commit**

```bash
git add src/gpu/kernels/rasterize.cl src/gpu/rasterizer_gpu.*
git commit -m "feat: add GPU rasterizer with shared memory tile blending"
```

---

### Task 9: Main Integration + --gpu Flag

**Goal:** Wire up GPU backends in main.cpp, add --gpu CLI flag.

**Files:**
- Modify: `src/main.cpp`

- [ ] **Step 1: Add --gpu flag and GPU backend construction**

```cpp
#ifdef ENABLE_OPENCL
#include "gpu/opencl_context.h"
#include "gpu/preprocessor_gpu.h"
#include "gpu/tile_binner_gpu.h"
#include "gpu/sorter_gpu.h"
#include "gpu/rasterizer_gpu.h"
#endif

// In main():
bool use_gpu = false;
for (int i = 1; i < argc; i++)
    if (strcmp(argv[i], "--gpu") == 0) use_gpu = true;

#ifdef ENABLE_OPENCL
if (use_gpu) {
    static OpenCLContext cl_ctx;
    if (!cl_ctx.init()) {
        printf("OpenCL init failed, falling back to CPU\n");
        use_gpu = false;
    }
}
if (use_gpu) {
    renderer = Renderer(
        std::make_unique<PreprocessorGPU>(cl_ctx),
        std::make_unique<TileBinnerGPU>(cl_ctx),
        std::make_unique<SorterGPU>(cl_ctx),
        std::make_unique<RasterizerGPU>(cl_ctx),
        alloc_size);
}
#endif
```

- [ ] **Step 2: Cross-compile, push to phone, run with --gpu**

```bash
$HDC shell "/data/local/tmp/gs3d_render basketball.ply cameras.json 0 --gpu"
```

- [ ] **Step 3: Compare GPU output with CPU reference**

Retrieve output, compute PSNR. Target: > 30 dB.

- [ ] **Step 4: Performance comparison**

Run both CPU and GPU, compare render times.

- [ ] **Step 5: Commit**

```bash
git add src/main.cpp
git commit -m "feat: add --gpu flag for OpenCL GPU rendering on HarmonyOS"
```

---

### Task 10: End-to-End Validation + Performance Report

**Goal:** Final validation with basketball.ply, multiple cameras, performance benchmarks.

- [ ] **Step 1: Render basketball.ply cam 0 and cam 5 with GPU**

- [ ] **Step 2: Pixel-level comparison with CPU reference**

PSNR, mean absolute diff, visual diff overlay.

- [ ] **Step 3: Performance benchmarks**

| Metric | CPU (ARM64) | GPU (Maleoon 920) |
|--------|-------------|-------------------|
| Preprocess time | ? ms | ? ms |
| Sort time | ? ms | ? ms |
| Rasterize time | ? ms | ? ms |
| Total render time | ? ms | ? ms |
| Speedup | 1x | ?x |

- [ ] **Step 4: Commit results and push to Gitee**

```bash
git add harmonyos_3dgs/docs/
git commit -m "docs: add GPU vs CPU validation results and performance report"
git push gitee main
```
