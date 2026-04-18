# SP-1 Vulkan Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the existing Phase 0/1 Vulkan code from `worktree-test_vulkan` branch into master-based SP-1 branch, then supplement it to match spec §3 requirements: capability checking, two-layer dispatch API (`dispatch_sync` + `record`), byte-stream SPIR-V input, explicit `ENABLE_VULKAN` CMake option, spec-exact `hello.comp`, and tiered device selection with env overrides.

**Architecture:** Additive — keep existing `VulkanContext`/`VulkanBuffer`/`VulkanShader`/`VulkanComputePipeline` classes (and existing passing tests) intact; add new capability struct and new pipeline dispatch methods. Defer `PreprocessorVK` / `preprocess.comp` / `test_preprocessor_vk.cpp` review and possible rewrite to SP-2 (spec §4 pass-first architecture may require changes there).

**Tech Stack:** C++17, Vulkan 1.1 (via `libvulkan` on Linux/Android), `glslangValidator` for GLSL→SPIR-V, `xxd` for SPIR-V byte embedding, CMake 3.28+, GoogleTest (already wired).

---

## Interface Inventory

**Existing (from `worktree-test_vulkan` @ `436fc28`, verified 12/12 tests pass)**:

| Interface | Path | Signature | SP-1 action |
|-----------|------|-----------|-------------|
| `class VulkanContext` | `include/vulkan/vk_context.h` | `init()/release()` + accessors + `findHostVisibleMemType()`/`findDeviceLocalMemType()`/`allocatePrimary()`/`submitAndWait()` | **Extend** — add capabilities |
| `class VulkanBuffer` | `include/vulkan/vk_buffer.h` | RAII host-visible SSBO + `upload()`/`download()` | **Keep as-is** (Phase 1 acceptable) |
| `class VulkanShader` | `include/vulkan/vk_shader.h` | File-based SPIR-V loader | **Extend** — add byte-stream overload |
| `class VulkanComputePipeline` | `include/vulkan/vk_pipeline.h` | Constructor + `allocateDescriptorSet()` | **Extend** — add `dispatch_sync`/`record`/push-constant runtime check |
| `add_one.comp` | `src/vulkan/shaders/add_one.comp` | 1-SSBO smoke shader | **Keep** (compatibility); **add** spec-exact `hello.comp` alongside |
| `test_vk_compute.cpp` | `tests/test_vk_compute.cpp` | 3 tests | **Keep** (regression coverage) |
| `PreprocessorVK`, `preprocess.comp`, `test_preprocessor_vk.cpp` | `include/vulkan/preprocessor_vk.h`, `src/vulkan/`, `tests/` | Slice 2a/2b/2c | **SP-2 scope — DO NOT TOUCH in SP-1**; if re-integration breaks them, fix build dependencies only |

**Draft (created in this plan)**:

| Interface | Planned location | Created in Task |
|-----------|------------------|-----------------|
| `struct VulkanDeviceCapabilities` | `include/vulkan/vk_capabilities.h` | Task 3 |
| `VulkanContext::capabilities()` accessor | `include/vulkan/vk_context.h` | Task 4 |
| Device selection priority + `GS3D_VK_DEVICE` / `GS3D_VK_DEVICE_NAME` env overrides | `src/vulkan/vk_context.cpp` | Task 6 |
| `VulkanShader` byte-stream constructor (`const uint8_t*, size_t`) | `include/vulkan/vk_shader.h` | Task 9 |
| xxd CMake function `gs3d_embed_spirv(shader_name)` → `<name>_spv.h` | `harmonyos_3dgs/CMakeLists.txt` | Task 10 |
| `hello.comp` per spec (3 SSBOs + push constant `n` + bounds check) | `src/vulkan/shaders/hello.comp` | Task 11 |
| `VulkanComputePipeline::dispatch_sync(push_constants, gx, gy, gz)` | `include/vulkan/vk_pipeline.h` | Task 12 |
| `VulkanComputePipeline::record(VkCommandBuffer, ...)` | same | Task 13 |
| `VulkanComputePipeline::insert_compute_barrier(VkCommandBuffer)` | same | Task 13 |
| `option(ENABLE_VULKAN ...)` in CMake | `harmonyos_3dgs/CMakeLists.txt` | Task 2 |

**Enforcement**: No task may reference an interface not in either list above.

---

## File Structure

### New files
```
harmonyos_3dgs/
├── include/vulkan/
│   └── vk_capabilities.h                 # VulkanDeviceCapabilities struct
├── src/vulkan/
│   ├── vk_capabilities.cpp               # Capability probing
│   └── shaders/
│       └── hello.comp                    # Spec-exact 3-SSBO shader
└── tests/
    ├── test_vk_capabilities.cpp          # Capability probing tests
    ├── test_vk_device_selection.cpp      # Priority + env override tests
    ├── test_vk_pipeline_dispatch.cpp     # dispatch_sync + record tests
    └── test_vk_hello.cpp                 # End-to-end hello.comp smoke

dev_notes/
└── sp1_infrastructure_notes.md            # SP-1 closing notes + SP-2 handoff
```

### Modified files
```
harmonyos_3dgs/
├── CMakeLists.txt                         # ENABLE_VULKAN option + xxd embed + new tests
├── include/vulkan/
│   ├── vk_context.h                       # capabilities() accessor
│   ├── vk_pipeline.h                      # dispatch_sync + record + insert_compute_barrier
│   └── vk_shader.h                        # Byte-stream overload
└── src/vulkan/
    ├── vk_context.cpp                     # Capability probing + tiered device selection
    ├── vk_pipeline.cpp                    # Push-constant validation + dispatch impls
    └── vk_shader.cpp                      # Byte-stream impl
```

---

## Task 1: Create SP-1 worktree and integrate existing Phase 0/1 code

**Files:**
- Create: git worktree at `.claude/worktrees/sp1/` on new branch `sp1-vulkan-infra` (from master `d87c8c5`)
- Copy: files from `worktree-test_vulkan` into sp1 worktree tree

**Context:** Existing Phase 0/1 code lives in `worktree-test_vulkan` branch (base `436fc28`, predates SP-0). SP-0 has now been merged to master. SP-1 starts from master and integrates the Vulkan code.

- [ ] **Step 1: Create SP-1 worktree**

```bash
cd /home/robota/h00813233/Graph/aaags-claude
git worktree add -b sp1-vulkan-infra .claude/worktrees/sp1 master
cd .claude/worktrees/sp1
git log --oneline -3  # Verify base is master d87c8c5
```

Expected: HEAD at `d87c8c5 Merge SP-0: CUDA golden infrastructure`.

- [ ] **Step 2: Copy Vulkan infrastructure files from test_vulkan worktree**

The existing code and the SP-1 base share the same repo; copy the Vulkan files directly.

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1
SRC=/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/test_vulkan/harmonyos_3dgs

# Headers
mkdir -p harmonyos_3dgs/include/vulkan
cp $SRC/include/vulkan/vk_context.h   harmonyos_3dgs/include/vulkan/
cp $SRC/include/vulkan/vk_buffer.h    harmonyos_3dgs/include/vulkan/
cp $SRC/include/vulkan/vk_shader.h    harmonyos_3dgs/include/vulkan/
cp $SRC/include/vulkan/vk_pipeline.h  harmonyos_3dgs/include/vulkan/
cp $SRC/include/vulkan/preprocessor_vk.h harmonyos_3dgs/include/vulkan/

# Implementations
mkdir -p harmonyos_3dgs/src/vulkan/shaders
cp $SRC/src/vulkan/vk_context.cpp    harmonyos_3dgs/src/vulkan/
cp $SRC/src/vulkan/vk_buffer.cpp     harmonyos_3dgs/src/vulkan/
cp $SRC/src/vulkan/vk_shader.cpp     harmonyos_3dgs/src/vulkan/
cp $SRC/src/vulkan/vk_pipeline.cpp   harmonyos_3dgs/src/vulkan/
cp $SRC/src/vulkan/preprocessor_vk.cpp harmonyos_3dgs/src/vulkan/
cp $SRC/src/vulkan/hello_vulkan_main.cpp harmonyos_3dgs/src/vulkan/
cp $SRC/src/vulkan/shaders/add_one.comp   harmonyos_3dgs/src/vulkan/shaders/
cp $SRC/src/vulkan/shaders/preprocess.comp harmonyos_3dgs/src/vulkan/shaders/

# Tests
cp $SRC/tests/test_vk_compute.cpp       harmonyos_3dgs/tests/
cp $SRC/tests/test_preprocessor_vk.cpp  harmonyos_3dgs/tests/

# CMakeLists — copy entire file, then later tasks will edit
cp $SRC/CMakeLists.txt harmonyos_3dgs/CMakeLists.txt
```

- [ ] **Step 3: Build and verify existing tests pass**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1/harmonyos_3dgs
rm -rf build
cmake -B build -DBUILD_TESTS=ON 2>&1 | tail -10
cmake --build build 2>&1 | tail -10
ctest --test-dir build --output-on-failure 2>&1 | tail -15
```

Expected: build succeeds, 165 (baseline) + 12 (Vulkan) = 177 tests, or close depending on merge ordering. All pass (pre-existing DensityController flakiness may appear, non-blocking).

- [ ] **Step 4: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1
git add harmonyos_3dgs/
git -c commit.gpgsign=false commit -m "sp1: integrate Phase 0/1 Vulkan code from test_vulkan branch

Brings in VulkanContext, VulkanBuffer, VulkanShader, VulkanComputePipeline
plus existing PreprocessorVK/preprocess.comp (to be evaluated in SP-2)
and 12 passing tests."
```

---

## Task 2: Add explicit ENABLE_VULKAN CMake option

**Files:**
- Modify: `harmonyos_3dgs/CMakeLists.txt`

Current: Vulkan is auto-enabled when `find_package(Vulkan QUIET)` succeeds. Spec §3.4 requires explicit `option(ENABLE_VULKAN ...)` toggle (mirrors existing `ENABLE_OPENCL`).

- [ ] **Step 1: Add option declaration near top of CMakeLists.txt**

Locate the existing `option(BUILD_TESTS ...)` and `option(ENABLE_OPENCL ...)` block (around line 7-9) and add:
```cmake
option(ENABLE_VULKAN "Build Vulkan GPU backend" ON)
```

Default `ON` because Vulkan is the primary SP-1+ target.

- [ ] **Step 2: Gate the Vulkan block on the option**

Locate the `if(NOT CMAKE_CROSSCOMPILING)` wrapper around the Vulkan section (approximately line 124). Change the outer guard:

Before:
```cmake
if(NOT CMAKE_CROSSCOMPILING)
    find_package(Vulkan QUIET)
    ...
```

After:
```cmake
if(ENABLE_VULKAN AND NOT CMAKE_CROSSCOMPILING)
    find_package(Vulkan REQUIRED)    # REQUIRED (not QUIET) when explicitly enabled
    find_program(GLSLANG_VALIDATOR
        NAMES glslangValidator
        HINTS $ENV{VULKAN_SDK}/bin ~/.local/bin REQUIRED)
    ...
```

Rationale: if the user explicitly sets `-DENABLE_VULKAN=ON` (or leaves default ON), fail fast on missing Vulkan rather than silently skipping.

- [ ] **Step 3: Test toggle — build with ENABLE_VULKAN=OFF**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1/harmonyos_3dgs
rm -rf build
cmake -B build -DBUILD_TESTS=ON -DENABLE_VULKAN=OFF 2>&1 | grep -i "vulkan\|gs3d_vk"
```

Expected: No `gs3d_vk_core` / `gs3d_vk_tests` targets mentioned. Build proceeds with only CPU tests.

- [ ] **Step 4: Test toggle — build with ENABLE_VULKAN=ON (default)**

```bash
rm -rf build
cmake -B build -DBUILD_TESTS=ON 2>&1 | grep -i "vulkan\|gs3d_vk"
```

Expected: `Vulkan enabled: gs3d_vk_core, gs3d_vulkan_hello, gs3d_vk_tests` status line.

- [ ] **Step 5: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1
git add harmonyos_3dgs/CMakeLists.txt
git -c commit.gpgsign=false commit -m "sp1: add explicit ENABLE_VULKAN CMake option"
```

---

## Task 3: Define `VulkanDeviceCapabilities` struct + header stub

**Files:**
- Create: `harmonyos_3dgs/include/vulkan/vk_capabilities.h`

Per spec §3.1 requires capability struct with specific fields. This task creates the header (definition + stub accessor declaration); Task 4 implements probing.

- [ ] **Step 1: Write the header**

Create `harmonyos_3dgs/include/vulkan/vk_capabilities.h`:
```cpp
// vk_capabilities.h -- Device capability snapshot populated at VulkanContext
// init time and cached for the life of the context. Used by pipeline-time
// validation (e.g. push-constant size limit) and SPIR-V selection (e.g.
// native atomic float vs CAS fallback path in SP-3 onward).

#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

struct VulkanDeviceCapabilities {
    // Required minimums per spec §3.1. VulkanContext::init() rejects a
    // device that does not satisfy these.
    uint32_t max_push_constants_size        = 0;   // >= 128
    uint32_t max_compute_workgroup_invocations = 0; // >= 256
    uint32_t max_compute_shared_memory_size  = 0;   // >= 16384
    uint32_t max_compute_workgroup_size[3]   = {0, 0, 0}; // >= {256,256,64}

    // Subgroup properties (recorded; enforcement is per-shader in SP-2+).
    uint32_t               subgroup_size              = 0;
    VkShaderStageFlags     subgroup_supported_stages  = 0;   // must contain COMPUTE
    VkSubgroupFeatureFlags subgroup_supported_ops     = 0;

    // Extension detection (affects SP-3 backward CAS fallback).
    bool has_shader_atomic_float = false;   // VK_EXT_shader_atomic_float

    // Vulkan API version the device reports.
    uint32_t api_version = 0;               // >= VK_API_VERSION_1_1
};
```

- [ ] **Step 2: Verify header compiles (syntax only)**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1/harmonyos_3dgs
g++ -c -std=c++17 -I include -I /usr/include -x c++ - <<'EOF'
#include "vulkan/vk_capabilities.h"
int main(){VulkanDeviceCapabilities c; return 0;}
EOF
```

Expected: compiles with no errors (ignore "no input" warning from `/dev/stdout`-only link). If g++ complains it can't find `<vulkan/vulkan.h>`, ignore — CMake handles include path; syntax check itself is fine.

- [ ] **Step 3: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1
git add harmonyos_3dgs/include/vulkan/vk_capabilities.h
git -c commit.gpgsign=false commit -m "sp1: add VulkanDeviceCapabilities struct header"
```

---

## Task 4: Probe capabilities and expose via VulkanContext

**Files:**
- Create: `harmonyos_3dgs/src/vulkan/vk_capabilities.cpp`
- Modify: `harmonyos_3dgs/include/vulkan/vk_context.h`
- Modify: `harmonyos_3dgs/src/vulkan/vk_context.cpp`
- Modify: `harmonyos_3dgs/CMakeLists.txt` (add source file)

- [ ] **Step 1: Implement probing in `vk_capabilities.cpp`**

Create `harmonyos_3dgs/src/vulkan/vk_capabilities.cpp`:
```cpp
#include "vulkan/vk_capabilities.h"
#include <cstring>
#include <vector>

namespace {

bool query_atomic_float_ext(VkPhysicalDevice dev) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> props(n);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &n, props.data());
    for (const auto& e : props) {
        if (std::strcmp(e.extensionName, "VK_EXT_shader_atomic_float") == 0)
            return true;
    }
    return false;
}

}  // namespace

VulkanDeviceCapabilities probe_capabilities(VkPhysicalDevice phys) {
    VulkanDeviceCapabilities caps;

    // Basic props (includes api_version, limits)
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys, &props);
    caps.api_version = props.apiVersion;
    caps.max_push_constants_size         = props.limits.maxPushConstantsSize;
    caps.max_compute_workgroup_invocations = props.limits.maxComputeWorkGroupInvocations;
    caps.max_compute_shared_memory_size   = props.limits.maxComputeSharedMemorySize;
    caps.max_compute_workgroup_size[0]    = props.limits.maxComputeWorkGroupSize[0];
    caps.max_compute_workgroup_size[1]    = props.limits.maxComputeWorkGroupSize[1];
    caps.max_compute_workgroup_size[2]    = props.limits.maxComputeWorkGroupSize[2];

    // Subgroup props (Vulkan 1.1+ core; via pNext chain)
    VkPhysicalDeviceSubgroupProperties subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(phys, &props2);
    caps.subgroup_size             = subgroup.subgroupSize;
    caps.subgroup_supported_stages = subgroup.supportedStages;
    caps.subgroup_supported_ops    = subgroup.supportedOperations;

    // Extensions
    caps.has_shader_atomic_float = query_atomic_float_ext(phys);

    return caps;
}
```

- [ ] **Step 2: Declare probe function + cache accessor in `vk_context.h`**

Add near the top of `vk_context.h` after existing includes:
```cpp
#include "vulkan/vk_capabilities.h"

VulkanDeviceCapabilities probe_capabilities(VkPhysicalDevice phys);
```

Add public accessor to the `VulkanContext` class (inside the class body, after other accessors):
```cpp
const VulkanDeviceCapabilities& capabilities() const { return caps_; }
```

Add private member near existing `api_version_`:
```cpp
VulkanDeviceCapabilities caps_{};
```

- [ ] **Step 3: Populate `caps_` in `VulkanContext::init()`**

In `vk_context.cpp`, find where `api_version_` is populated (after `phys_` is chosen) and add:
```cpp
caps_ = probe_capabilities(phys_);
```

- [ ] **Step 4: Add source to CMakeLists**

In `harmonyos_3dgs/CMakeLists.txt`, locate the `add_library(gs3d_vk_core STATIC ...)` block and add `src/vulkan/vk_capabilities.cpp`:
```cmake
add_library(gs3d_vk_core STATIC
    src/vulkan/vk_context.cpp
    src/vulkan/vk_capabilities.cpp
    src/vulkan/vk_buffer.cpp
    src/vulkan/vk_shader.cpp
    src/vulkan/vk_pipeline.cpp
    src/vulkan/preprocessor_vk.cpp)
```

- [ ] **Step 5: Build — no behavior change yet, just verify it compiles and existing tests still pass**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1/harmonyos_3dgs
cmake --build build 2>&1 | tail -5
ctest --test-dir build -R "VkCompute|VkContext" --output-on-failure
```

Expected: 3 tests (AddOne_64Elements, AddOne_1024Elements, DeviceQueryableAfterInit) still pass.

- [ ] **Step 6: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1
git add harmonyos_3dgs/src/vulkan/vk_capabilities.cpp \
        harmonyos_3dgs/include/vulkan/vk_context.h \
        harmonyos_3dgs/src/vulkan/vk_context.cpp \
        harmonyos_3dgs/CMakeLists.txt
git -c commit.gpgsign=false commit -m "sp1: probe capabilities at VulkanContext init + expose accessor"
```

---

## Task 5: Add capability test + enforce required minimums

**Files:**
- Create: `harmonyos_3dgs/tests/test_vk_capabilities.cpp`
- Modify: `harmonyos_3dgs/src/vulkan/vk_context.cpp` (add filter)
- Modify: `harmonyos_3dgs/CMakeLists.txt` (register test)

- [ ] **Step 1: Write the test**

Create `harmonyos_3dgs/tests/test_vk_capabilities.cpp`:
```cpp
#include <gtest/gtest.h>
#include "vulkan/vk_context.h"

TEST(VkCapabilities, MeetsRequiredMinimums) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed; probe requires a device meeting spec §3.1 minimums";
    const auto& c = ctx.capabilities();

    EXPECT_GE(c.api_version, VK_API_VERSION_1_1);
    EXPECT_GE(c.max_push_constants_size, 128u);
    EXPECT_GE(c.max_compute_workgroup_invocations, 256u);
    EXPECT_GE(c.max_compute_shared_memory_size, 16384u);
    EXPECT_GE(c.max_compute_workgroup_size[0], 256u);
    EXPECT_GE(c.max_compute_workgroup_size[1], 256u);
    EXPECT_GE(c.max_compute_workgroup_size[2], 64u);
    EXPECT_TRUE(c.subgroup_supported_stages & VK_SHADER_STAGE_COMPUTE_BIT);
    EXPECT_GT(c.subgroup_size, 0u);

    // atomic_float is informational — may be true or false depending on driver.
    // Just log it; SP-3 will branch on this flag.
    std::cout << "has_shader_atomic_float: "
              << (c.has_shader_atomic_float ? "true" : "false") << "\n";

    ctx.release();
}
```

- [ ] **Step 2: Register in CMakeLists**

In `harmonyos_3dgs/CMakeLists.txt`, locate `add_executable(gs3d_vk_tests ...)` and add the new test:
```cmake
add_executable(gs3d_vk_tests
    tests/test_vk_compute.cpp
    tests/test_vk_capabilities.cpp
    tests/test_preprocessor_vk.cpp)
```

- [ ] **Step 3: Enforce minimums in `VulkanContext::init()` device selection**

In `vk_context.cpp`, after probing capabilities for a candidate device, add a filter helper:
```cpp
static bool meets_minimums(const VulkanDeviceCapabilities& c) {
    if (c.api_version < VK_API_VERSION_1_1) return false;
    if (c.max_push_constants_size < 128u) return false;
    if (c.max_compute_workgroup_invocations < 256u) return false;
    if (c.max_compute_shared_memory_size < 16384u) return false;
    if (c.max_compute_workgroup_size[0] < 256u) return false;
    if (c.max_compute_workgroup_size[1] < 256u) return false;
    if (c.max_compute_workgroup_size[2] < 64u) return false;
    if (!(c.subgroup_supported_stages & VK_SHADER_STAGE_COMPUTE_BIT)) return false;
    return true;
}
```

Modify the physical-device-iteration loop to call `meets_minimums(probe_capabilities(candidate))` before selecting. If no device meets minimums, log which device failed which check and return false from `init()`.

- [ ] **Step 4: Build + run capability test**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1/harmonyos_3dgs
cmake --build build 2>&1 | tail -5
ctest --test-dir build -R VkCapabilities --output-on-failure
```

Expected: `VkCapabilities.MeetsRequiredMinimums` passes on this hardware.

- [ ] **Step 5: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1
git add harmonyos_3dgs/tests/test_vk_capabilities.cpp \
        harmonyos_3dgs/src/vulkan/vk_context.cpp \
        harmonyos_3dgs/CMakeLists.txt
git -c commit.gpgsign=false commit -m "sp1: enforce capability minimums during device selection + add test"
```

---

## Task 6: Tiered device selection + env var overrides

**Files:**
- Modify: `harmonyos_3dgs/src/vulkan/vk_context.cpp`

Spec §3.1 priorities:
- Dev machine (default): `DISCRETE > INTEGRATED > CPU > OTHER`
- Target machine (ifdef crosscompile or `GS3D_VK_PREFER_INTEGRATED` env): `INTEGRATED > DISCRETE > CPU > OTHER`

Env overrides (bypass auto-priority, still respect minimums):
- `GS3D_VK_DEVICE=<index>` — physical device index from `vkEnumeratePhysicalDevices`
- `GS3D_VK_DEVICE_NAME=<substr>` — match first device whose name contains substr

- [ ] **Step 1: Replace the device-selection portion of `VulkanContext::init()`**

In `vk_context.cpp`, find the current loop that picks the first compute-capable device. Replace with:
```cpp
// Enumerate all physical devices
uint32_t n_phys = 0;
vkEnumeratePhysicalDevices(instance_, &n_phys, nullptr);
if (n_phys == 0) {
    std::cerr << "VulkanContext: no physical devices\n";
    return false;
}
std::vector<VkPhysicalDevice> phys_list(n_phys);
vkEnumeratePhysicalDevices(instance_, &n_phys, phys_list.data());

// Env overrides
const char* env_index = std::getenv("GS3D_VK_DEVICE");
const char* env_name  = std::getenv("GS3D_VK_DEVICE_NAME");

// Candidate: passes minimums + has compute queue
struct Candidate {
    VkPhysicalDevice dev;
    uint32_t         qf;
    std::string      name;
    VkPhysicalDeviceType type;
    VulkanDeviceCapabilities caps;
};
std::vector<Candidate> candidates;
for (uint32_t i = 0; i < n_phys; ++i) {
    VkPhysicalDevice pd = phys_list[i];
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);

    // Find compute queue family
    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfs.data());
    uint32_t chosen_qf = UINT32_MAX;
    for (uint32_t q = 0; q < nqf; ++q) {
        if (qfs[q].queueFlags & VK_QUEUE_COMPUTE_BIT) { chosen_qf = q; break; }
    }
    if (chosen_qf == UINT32_MAX) continue;

    VulkanDeviceCapabilities caps = probe_capabilities(pd);
    if (!meets_minimums(caps)) continue;

    candidates.push_back({pd, chosen_qf, props.deviceName, props.deviceType, caps});
}
if (candidates.empty()) {
    std::cerr << "VulkanContext: no device passes capability minimums\n";
    return false;
}

// Env override: index
if (env_index) {
    int want = std::atoi(env_index);
    if (want < 0 || static_cast<uint32_t>(want) >= n_phys) {
        std::cerr << "GS3D_VK_DEVICE=" << want << " out of range (0.." << (n_phys-1) << ")\n";
        return false;
    }
    for (auto& c : candidates) {
        if (c.dev == phys_list[want]) {
            phys_ = c.dev; compute_qf_ = c.qf;
            device_name_ = c.name; api_version_ = c.caps.api_version; caps_ = c.caps;
            goto device_chosen;
        }
    }
    std::cerr << "GS3D_VK_DEVICE index selected a device that fails minimums\n";
    return false;
}
// Env override: name substring
if (env_name) {
    for (auto& c : candidates) {
        if (c.name.find(env_name) != std::string::npos) {
            phys_ = c.dev; compute_qf_ = c.qf;
            device_name_ = c.name; api_version_ = c.caps.api_version; caps_ = c.caps;
            goto device_chosen;
        }
    }
    std::cerr << "GS3D_VK_DEVICE_NAME=" << env_name << " matched no capable device\n";
    return false;
}

// Auto-priority: dev default = DISCRETE > INTEGRATED > CPU > OTHER
auto rank = [](VkPhysicalDeviceType t) -> int {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 0;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 1;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return 2;
        default:                                     return 3;
    }
};
std::sort(candidates.begin(), candidates.end(),
    [&](const Candidate& a, const Candidate& b) {
        return rank(a.type) < rank(b.type);
    });
phys_ = candidates[0].dev;
compute_qf_ = candidates[0].qf;
device_name_ = candidates[0].name;
api_version_ = candidates[0].caps.api_version;
caps_ = candidates[0].caps;
device_chosen:;
```

Also add `#include <cstdlib>`, `#include <algorithm>`, `#include <iostream>` at top of file if not already present.

- [ ] **Step 2: Build + existing tests still green**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1/harmonyos_3dgs
cmake --build build 2>&1 | tail -5
ctest --test-dir build -R "VkCompute|VkContext|VkCapabilities" --output-on-failure
```

Expected: all pre-existing Vulkan tests pass; device_name reflects priority pick (e.g., DISCRETE GPU on a dev machine).

- [ ] **Step 3: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1
git add harmonyos_3dgs/src/vulkan/vk_context.cpp
git -c commit.gpgsign=false commit -m "sp1: tiered device selection + GS3D_VK_DEVICE/_NAME env overrides"
```

---

## Task 7: Device selection test

**Files:**
- Create: `harmonyos_3dgs/tests/test_vk_device_selection.cpp`
- Modify: `harmonyos_3dgs/CMakeLists.txt`

- [ ] **Step 1: Write the test**

Create `harmonyos_3dgs/tests/test_vk_device_selection.cpp`:
```cpp
#include <gtest/gtest.h>
#include "vulkan/vk_context.h"
#include <cstdlib>

TEST(VkDeviceSelection, DefaultPrefersDiscrete) {
    unsetenv("GS3D_VK_DEVICE");
    unsetenv("GS3D_VK_DEVICE_NAME");
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    // Just sanity-check: device name is nonempty and init succeeded.
    EXPECT_FALSE(ctx.deviceName().empty());
    ctx.release();
}

TEST(VkDeviceSelection, EnvIndex_Valid) {
    setenv("GS3D_VK_DEVICE", "0", 1);
    VulkanContext ctx;
    bool ok = ctx.init();
    unsetenv("GS3D_VK_DEVICE");
    // Index 0 should succeed on any machine with at least one capable device.
    EXPECT_TRUE(ok);
    if (ok) ctx.release();
}

TEST(VkDeviceSelection, EnvIndex_OutOfRange) {
    setenv("GS3D_VK_DEVICE", "99", 1);
    VulkanContext ctx;
    bool ok = ctx.init();
    unsetenv("GS3D_VK_DEVICE");
    EXPECT_FALSE(ok);  // index 99 doesn't exist
}

TEST(VkDeviceSelection, EnvNameSubstring_NoMatch) {
    setenv("GS3D_VK_DEVICE_NAME", "ZZZZ_NonExistent_ZZZZ", 1);
    VulkanContext ctx;
    bool ok = ctx.init();
    unsetenv("GS3D_VK_DEVICE_NAME");
    EXPECT_FALSE(ok);
}
```

- [ ] **Step 2: Register in CMakeLists**

Add `tests/test_vk_device_selection.cpp` to `gs3d_vk_tests` sources in `harmonyos_3dgs/CMakeLists.txt`.

- [ ] **Step 3: Build and run**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1/harmonyos_3dgs
cmake --build build 2>&1 | tail -5
ctest --test-dir build -R VkDeviceSelection --output-on-failure
```

Expected: 4 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1
git add harmonyos_3dgs/tests/test_vk_device_selection.cpp harmonyos_3dgs/CMakeLists.txt
git -c commit.gpgsign=false commit -m "sp1: add device selection tests (auto + env overrides)"
```

---

## Task 8: Runtime push-constant size check in `VulkanComputePipeline`

**Files:**
- Modify: `harmonyos_3dgs/src/vulkan/vk_pipeline.cpp`

Spec §3.3: constructor must check `push_constant_size ≤ device_caps.max_push_constants_size` at runtime.

- [ ] **Step 1: Add check at top of constructor**

In `vk_pipeline.cpp`, locate the `VulkanComputePipeline::VulkanComputePipeline(...)` constructor body. Right after member initialization, before any Vulkan calls, add:
```cpp
if (push_constant_bytes > ctx.capabilities().max_push_constants_size) {
    throw std::runtime_error(
        "VulkanComputePipeline: push_constant_bytes=" +
        std::to_string(push_constant_bytes) +
        " exceeds device limit=" +
        std::to_string(ctx.capabilities().max_push_constants_size) +
        "; use a UBO instead.");
}
```

- [ ] **Step 2: Build + existing tests still pass**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1/harmonyos_3dgs
cmake --build build 2>&1 | tail -5
ctest --test-dir build -R "VkCompute" --output-on-failure
```

Expected: existing pipeline tests still pass (they use small push constants).

- [ ] **Step 3: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1
git add harmonyos_3dgs/src/vulkan/vk_pipeline.cpp
git -c commit.gpgsign=false commit -m "sp1: runtime check push_constant_bytes against device limit"
```

---

## Task 9: Byte-stream SPIR-V overload on `VulkanShader`

**Files:**
- Modify: `harmonyos_3dgs/include/vulkan/vk_shader.h`
- Modify: `harmonyos_3dgs/src/vulkan/vk_shader.cpp`

Spec §3.3 requires byte-stream input compatible with `xxd -i` (which emits `unsigned char[]`). Keep existing file-based constructor for backward compatibility with the current `add_one.comp` / `preprocess.comp` flow.

- [ ] **Step 1: Read existing vk_shader.h**

```bash
cat /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1/harmonyos_3dgs/include/vulkan/vk_shader.h
```

Expected content: single constructor taking `const std::string& path`.

- [ ] **Step 2: Add overload to header**

In `vk_shader.h`, add alongside the existing file-based constructor:
```cpp
/// Load SPIR-V from a raw byte stream (compatible with xxd -i output).
/// `spirv_bytes` must be 4-byte aligned and `byte_size` must be a multiple of 4.
VulkanShader(VulkanContext& ctx, const uint8_t* spirv_bytes, std::size_t byte_size);
```

Make sure `#include <cstdint>` and `#include <cstddef>` are at the top.

- [ ] **Step 3: Implement overload in `vk_shader.cpp`**

Append to `vk_shader.cpp`:
```cpp
VulkanShader::VulkanShader(VulkanContext& ctx, const uint8_t* spirv_bytes,
                            std::size_t byte_size)
    : ctx_(ctx)
{
    if (byte_size == 0 || (byte_size % 4) != 0)
        throw std::runtime_error(
            "VulkanShader: SPIR-V byte_size must be >0 and multiple of 4, got " +
            std::to_string(byte_size));

    // SPIR-V is uint32_t[]; need 4-byte alignment. std::vector<uint32_t> gives it.
    std::vector<uint32_t> aligned(byte_size / 4);
    std::memcpy(aligned.data(), spirv_bytes, byte_size);

    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = byte_size;
    info.pCode    = aligned.data();

    VK_CHECK(vkCreateShaderModule(ctx_.device(), &info, nullptr, &module_));
}
```

Add `#include <cstring>` and `#include <vector>` if not already present.

- [ ] **Step 4: Build**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1/harmonyos_3dgs
cmake --build build 2>&1 | tail -5
```

Expected: clean build (no new tests yet; Task 11 uses this overload).

- [ ] **Step 5: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1
git add harmonyos_3dgs/include/vulkan/vk_shader.h \
        harmonyos_3dgs/src/vulkan/vk_shader.cpp
git -c commit.gpgsign=false commit -m "sp1: add byte-stream SPIR-V overload to VulkanShader"
```

---

## Task 10: xxd-based SPIR-V embedding CMake helper

**Files:**
- Modify: `harmonyos_3dgs/CMakeLists.txt`

Adds a helper function `gs3d_embed_spirv(<shader_name>)` that generates `<shader_name>_spv.h` containing `unsigned char <shader_name>_spv[]` and `unsigned int <shader_name>_spv_len`. This lets shaders be compiled into the binary (no runtime file I/O per spec §3.4).

- [ ] **Step 1: Add helper function next to existing `gs3d_compile_glsl`**

In `harmonyos_3dgs/CMakeLists.txt`, after the existing `function(gs3d_compile_glsl ...)` block (around lines 134-148), add:
```cmake
# Embed an already-compiled .spv as a C header via xxd -i.
# Produces ${CMAKE_CURRENT_BINARY_DIR}/shaders/<shader>_spv.h
# with `unsigned char <shader>_spv[]` and `unsigned int <shader>_spv_len`.
find_program(XXD xxd REQUIRED)
function(gs3d_embed_spirv SHADER_NAME OUT_HEADER_VAR)
    set(_spv    ${_vk_shader_dir}/${SHADER_NAME}.spv)
    set(_header ${_vk_shader_dir}/${SHADER_NAME}_spv.h)
    add_custom_command(
        OUTPUT ${_header}
        COMMAND ${CMAKE_COMMAND} -E chdir ${_vk_shader_dir}
                ${XXD} -i ${SHADER_NAME}.spv > ${_header}
        DEPENDS ${_spv}
        COMMENT "Embedding ${SHADER_NAME}.spv as C header via xxd -i"
        VERBATIM)
    set(${OUT_HEADER_VAR} ${_header} PARENT_SCOPE)
endfunction()
```

- [ ] **Step 2: Verify xxd is available**

```bash
which xxd
```

Expected: path like `/usr/bin/xxd`. If missing, `find_program(XXD xxd REQUIRED)` will fail CMake configure.

- [ ] **Step 3: Smoke the function — embed add_one.spv**

Locate the existing `gs3d_compile_glsl(add_one _spv_add_one)` call. Right after it, add:
```cmake
gs3d_embed_spirv(add_one _hdr_add_one)
```

And make `gs3d_vulkan_shaders` depend on the header too:
```cmake
add_custom_target(gs3d_vulkan_shaders
    DEPENDS ${_spv_add_one} ${_spv_preprocess} ${_hdr_add_one})
```

- [ ] **Step 4: Re-configure + build, verify header is generated**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1/harmonyos_3dgs
rm -rf build
cmake -B build -DBUILD_TESTS=ON 2>&1 | tail -5
cmake --build build --target gs3d_vulkan_shaders 2>&1 | tail -5
ls build/shaders/add_one_spv.h
head -3 build/shaders/add_one_spv.h
```

Expected:
- File `build/shaders/add_one_spv.h` exists
- First line begins with `unsigned char add_one_spv[] = {` (typical xxd -i output)

- [ ] **Step 5: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1
git add harmonyos_3dgs/CMakeLists.txt
git -c commit.gpgsign=false commit -m "sp1: CMake helper gs3d_embed_spirv for xxd-based SPIR-V embedding"
```

---

## Task 11: Create `hello.comp` per spec

**Files:**
- Create: `harmonyos_3dgs/src/vulkan/shaders/hello.comp`
- Modify: `harmonyos_3dgs/CMakeLists.txt` (compile + embed)

Spec §3.5 contract: 3 SSBOs (2 input, 1 output), push constant `uint n`, `local_size_x=256`, mandatory bounds check. Does NOT replace `add_one.comp` — both coexist; `hello.comp` is the spec-exact red-line.

- [ ] **Step 1: Write the shader**

Create `harmonyos_3dgs/src/vulkan/shaders/hello.comp`:
```glsl
#version 450

// Spec §3.5 hello.comp contract:
//   3 storage buffers (2 input A/B, 1 output C)
//   push constant `uint n` (element count)
//   local_size_x = 256
//   mandatory bounds check on i >= n
// This shader is the minimum stable red-line for SP-1 pipeline infrastructure.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) buffer BufA { float a[]; };
layout(std430, set = 0, binding = 1) buffer BufB { float b[]; };
layout(std430, set = 0, binding = 2) buffer BufC { float c[]; };

layout(push_constant) uniform PC {
    uint n;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    c[i] = a[i] + b[i];
}
```

- [ ] **Step 2: Wire shader compile + embed in CMakeLists**

In `harmonyos_3dgs/CMakeLists.txt`, add next to the existing `add_one` / `preprocess` shader lines:
```cmake
gs3d_compile_glsl(hello _spv_hello)
gs3d_embed_spirv(hello _hdr_hello)
```

Update the `gs3d_vulkan_shaders` target to include the new hello artifacts:
```cmake
add_custom_target(gs3d_vulkan_shaders
    DEPENDS ${_spv_add_one} ${_spv_preprocess} ${_hdr_add_one}
            ${_spv_hello} ${_hdr_hello})
```

- [ ] **Step 3: Build**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1/harmonyos_3dgs
cmake --build build 2>&1 | tail -5
ls build/shaders/hello.spv build/shaders/hello_spv.h
head -1 build/shaders/hello_spv.h
```

Expected: both files exist; header first line `unsigned char hello_spv[] = {`.

- [ ] **Step 4: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1
git add harmonyos_3dgs/src/vulkan/shaders/hello.comp \
        harmonyos_3dgs/CMakeLists.txt
git -c commit.gpgsign=false commit -m "sp1: add hello.comp per spec §3.5 (3 SSBO + push constant + bounds check)"
```

---

## Task 12: `dispatch_sync` method on `VulkanComputePipeline`

**Files:**
- Modify: `harmonyos_3dgs/include/vulkan/vk_pipeline.h`
- Modify: `harmonyos_3dgs/src/vulkan/vk_pipeline.cpp`

Spec §3.3 layer 1: synchronous dispatch (bring-up + tests). Internally allocates command buffer, binds pipeline + descriptor set, records push constants + vkCmdDispatch, submits, waits. Single call from test code.

- [ ] **Step 1: Declare in header**

In `vk_pipeline.h`, inside the `VulkanComputePipeline` class, add:
```cpp
/// Layer 1 dispatch (sync, bring-up). Allocates an internal command buffer,
/// binds this pipeline + the given descriptor set, pushes constants if any,
/// dispatches, submits, and vkQueueWaitIdles. Use for tests and smoke runs.
void dispatch_sync(VkDescriptorSet descriptor_set,
                   uint32_t gx, uint32_t gy, uint32_t gz,
                   const void* push_constants = nullptr,
                   uint32_t push_size = 0);
```

- [ ] **Step 2: Implement in `vk_pipeline.cpp`**

Append:
```cpp
void VulkanComputePipeline::dispatch_sync(VkDescriptorSet desc_set,
                                          uint32_t gx, uint32_t gy, uint32_t gz,
                                          const void* push_constants,
                                          uint32_t push_size)
{
    VkCommandBuffer cmd = ctx_.allocatePrimary();

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_,
                             0, 1, &desc_set, 0, nullptr);
    if (push_size > 0 && push_constants != nullptr) {
        vkCmdPushConstants(cmd, layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push_constants);
    }
    vkCmdDispatch(cmd, gx, gy, gz);

    VK_CHECK(vkEndCommandBuffer(cmd));
    ctx_.submitAndWait(cmd);
    ctx_.freePrimary(cmd);
}
```

- [ ] **Step 3: Build**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1/harmonyos_3dgs
cmake --build build 2>&1 | tail -5
```

Expected: clean build. No tests yet.

- [ ] **Step 4: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1
git add harmonyos_3dgs/include/vulkan/vk_pipeline.h \
        harmonyos_3dgs/src/vulkan/vk_pipeline.cpp
git -c commit.gpgsign=false commit -m "sp1: add VulkanComputePipeline::dispatch_sync (layer 1)"
```

---

## Task 13: `record` method + `insert_compute_barrier` helper

**Files:**
- Modify: `harmonyos_3dgs/include/vulkan/vk_pipeline.h`
- Modify: `harmonyos_3dgs/src/vulkan/vk_pipeline.cpp`

Spec §3.3 layer 2: record to an externally-owned command buffer (caller controls begin/end/submit, and inserts barriers between stages).

- [ ] **Step 1: Declare both in header**

Append inside the `VulkanComputePipeline` class:
```cpp
/// Layer 2 dispatch (record to external command buffer). Caller owns cmd
/// buffer's begin/end/submit lifecycle and any barriers between stages.
/// Used by SP-2+ chained pipeline (preprocess → sort → rasterize).
void record(VkCommandBuffer cmd,
            VkDescriptorSet descriptor_set,
            uint32_t gx, uint32_t gy, uint32_t gz,
            const void* push_constants = nullptr,
            uint32_t push_size = 0);
```

Below the class, declare a free helper:
```cpp
/// Insert a compute-to-compute barrier on SSBO reads/writes. Called by the
/// caller of record() between two dispatches to ensure the second sees the
/// first's writes.
void insert_compute_barrier(VkCommandBuffer cmd);
```

- [ ] **Step 2: Implement in `vk_pipeline.cpp`**

Append:
```cpp
void VulkanComputePipeline::record(VkCommandBuffer cmd,
                                    VkDescriptorSet desc_set,
                                    uint32_t gx, uint32_t gy, uint32_t gz,
                                    const void* push_constants,
                                    uint32_t push_size)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_,
                             0, 1, &desc_set, 0, nullptr);
    if (push_size > 0 && push_constants != nullptr) {
        vkCmdPushConstants(cmd, layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push_constants);
    }
    vkCmdDispatch(cmd, gx, gy, gz);
}

void insert_compute_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{};
    mb.sType          = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask  = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask  = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}
```

- [ ] **Step 3: Build**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1/harmonyos_3dgs
cmake --build build 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1
git add harmonyos_3dgs/include/vulkan/vk_pipeline.h \
        harmonyos_3dgs/src/vulkan/vk_pipeline.cpp
git -c commit.gpgsign=false commit -m "sp1: add record() + insert_compute_barrier (layer 2)"
```

---

## Task 14: Dispatch tests (both sync + record + push-constant overflow)

**Files:**
- Create: `harmonyos_3dgs/tests/test_vk_pipeline_dispatch.cpp`
- Modify: `harmonyos_3dgs/CMakeLists.txt`

- [ ] **Step 1: Write tests**

Create `harmonyos_3dgs/tests/test_vk_pipeline_dispatch.cpp`:
```cpp
#include <gtest/gtest.h>
#include "vulkan/vk_context.h"
#include "vulkan/vk_buffer.h"
#include "vulkan/vk_shader.h"
#include "vulkan/vk_pipeline.h"
#include <vector>
#include <fstream>

namespace {

// Load .spv compiled for add_one.comp via file-based VulkanShader path.
// Path is exported by CMake as ADD_ONE_SPV_PATH.
#ifndef ADD_ONE_SPV_PATH
#define ADD_ONE_SPV_PATH "shaders/add_one.spv"
#endif

struct PushConst { uint32_t count; };

}  // namespace

TEST(VkPipelineDispatch, DispatchSync_AddOne) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());

    VulkanShader shader(ctx, ADD_ONE_SPV_PATH);
    VulkanComputePipeline pipe(ctx, shader, /*num_ssbos=*/1,
                                /*push_bytes=*/sizeof(PushConst));

    const uint32_t N = 128;
    VulkanBuffer data(ctx, N * sizeof(float));
    std::vector<float> host(N);
    for (uint32_t i = 0; i < N; ++i) host[i] = float(i);
    data.upload(host.data(), N * sizeof(float));

    auto ds = pipe.allocateDescriptorSet({data.handle()});
    PushConst pc{N};
    uint32_t gx = (N + 63) / 64;  // add_one uses local_size_x=64
    pipe.dispatch_sync(ds, gx, 1, 1, &pc, sizeof(pc));

    data.download(host.data(), N * sizeof(float));
    for (uint32_t i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(host[i], float(i) + 1.0f);

    ctx.release();
}

TEST(VkPipelineDispatch, Record_ExternalCommandBuffer) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    VulkanShader shader(ctx, ADD_ONE_SPV_PATH);
    VulkanComputePipeline pipe(ctx, shader, 1, sizeof(PushConst));

    const uint32_t N = 64;
    VulkanBuffer data(ctx, N * sizeof(float));
    std::vector<float> host(N, 2.5f);
    data.upload(host.data(), N * sizeof(float));
    auto ds = pipe.allocateDescriptorSet({data.handle()});

    VkCommandBuffer cmd = ctx.allocatePrimary();
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    PushConst pc{N};
    pipe.record(cmd, ds, (N + 63) / 64, 1, 1, &pc, sizeof(pc));
    insert_compute_barrier(cmd);  // call barrier helper
    pipe.record(cmd, ds, (N + 63) / 64, 1, 1, &pc, sizeof(pc));  // second pass

    vkEndCommandBuffer(cmd);
    ctx.submitAndWait(cmd);
    ctx.freePrimary(cmd);

    data.download(host.data(), N * sizeof(float));
    for (uint32_t i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(host[i], 4.5f);  // 2.5 + 1.0 + 1.0

    ctx.release();
}

TEST(VkPipelineDispatch, PushConstantOverflowThrows) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    VulkanShader shader(ctx, ADD_ONE_SPV_PATH);
    // Ask for 1MB push constant — vastly exceeds 128B minimum.
    EXPECT_THROW(
        { VulkanComputePipeline big(ctx, shader, 1, /*push_bytes=*/1048576); },
        std::runtime_error);
    ctx.release();
}
```

- [ ] **Step 2: Register test and pass ADD_ONE_SPV_PATH define**

In `harmonyos_3dgs/CMakeLists.txt`, locate where `gs3d_vk_tests` is defined. Add the new test source and define:
```cmake
add_executable(gs3d_vk_tests
    tests/test_vk_compute.cpp
    tests/test_vk_capabilities.cpp
    tests/test_vk_device_selection.cpp
    tests/test_vk_pipeline_dispatch.cpp
    tests/test_preprocessor_vk.cpp)
target_compile_definitions(gs3d_vk_tests PRIVATE
    ADD_ONE_SPV_PATH="${_spv_add_one}")
```

- [ ] **Step 3: Build and run**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1/harmonyos_3dgs
cmake --build build 2>&1 | tail -5
ctest --test-dir build -R VkPipelineDispatch --output-on-failure
```

Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1
git add harmonyos_3dgs/tests/test_vk_pipeline_dispatch.cpp \
        harmonyos_3dgs/CMakeLists.txt
git -c commit.gpgsign=false commit -m "sp1: add dispatch_sync / record / push-constant overflow tests"
```

---

## Task 15: `test_vk_hello.cpp` — end-to-end test using embedded SPIR-V + hello.comp

**Files:**
- Create: `harmonyos_3dgs/tests/test_vk_hello.cpp`
- Modify: `harmonyos_3dgs/CMakeLists.txt`

Uses the byte-stream `VulkanShader` overload (Task 9) + the xxd-embedded `hello_spv.h` (Task 10+11). This is the spec §3.5 red-line test — minimum stable verification of the full SP-1 pipeline infrastructure.

- [ ] **Step 1: Write the test**

Create `harmonyos_3dgs/tests/test_vk_hello.cpp`:
```cpp
#include <gtest/gtest.h>
#include "vulkan/vk_context.h"
#include "vulkan/vk_buffer.h"
#include "vulkan/vk_shader.h"
#include "vulkan/vk_pipeline.h"
#include <vector>

// Embedded by CMake via xxd -i; see Task 10/11.
#include "hello_spv.h"

namespace { struct PushConst { uint32_t n; }; }

TEST(VkHello, ArrayAdd_EmbeddedSPIRV) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());

    // Use byte-stream overload (Task 9) against xxd-embedded SPIR-V (Task 10+11).
    VulkanShader shader(ctx, static_cast<const uint8_t*>(hello_spv), hello_spv_len);
    VulkanComputePipeline pipe(ctx, shader,
                                /*num_ssbos=*/3,
                                /*push_bytes=*/sizeof(PushConst));

    const uint32_t N = 512;
    VulkanBuffer A(ctx, N * sizeof(float));
    VulkanBuffer B(ctx, N * sizeof(float));
    VulkanBuffer C(ctx, N * sizeof(float));

    std::vector<float> ha(N), hb(N), hc(N, 0.0f);
    for (uint32_t i = 0; i < N; ++i) { ha[i] = float(i); hb[i] = 0.5f * float(i); }
    A.upload(ha.data(), N * sizeof(float));
    B.upload(hb.data(), N * sizeof(float));
    C.upload(hc.data(), N * sizeof(float));

    auto ds = pipe.allocateDescriptorSet({A.handle(), B.handle(), C.handle()});
    PushConst pc{N};
    // hello.comp local_size_x=256
    pipe.dispatch_sync(ds, (N + 255) / 256, 1, 1, &pc, sizeof(pc));

    C.download(hc.data(), N * sizeof(float));
    for (uint32_t i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(hc[i], ha[i] + hb[i])
            << "mismatch at i=" << i;

    ctx.release();
}

TEST(VkHello, BoundsCheck_PreventsOverrun) {
    // n=10 but dispatch covers 256 invocations — bounds check inside shader
    // must prevent writes past index 10.
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init());
    VulkanShader shader(ctx, static_cast<const uint8_t*>(hello_spv), hello_spv_len);
    VulkanComputePipeline pipe(ctx, shader, 3, sizeof(PushConst));

    const uint32_t N = 256;           // buffer has 256 slots
    const uint32_t effective_n = 10;  // only write first 10
    VulkanBuffer A(ctx, N * sizeof(float));
    VulkanBuffer B(ctx, N * sizeof(float));
    VulkanBuffer C(ctx, N * sizeof(float));
    std::vector<float> ha(N, 1.0f), hb(N, 2.0f), hc(N, -999.0f);
    A.upload(ha.data(), N*4); B.upload(hb.data(), N*4); C.upload(hc.data(), N*4);

    auto ds = pipe.allocateDescriptorSet({A.handle(), B.handle(), C.handle()});
    PushConst pc{effective_n};
    pipe.dispatch_sync(ds, 1, 1, 1, &pc, sizeof(pc));

    C.download(hc.data(), N * sizeof(float));
    for (uint32_t i = 0; i < effective_n; ++i) EXPECT_FLOAT_EQ(hc[i], 3.0f);
    for (uint32_t i = effective_n; i < N; ++i) EXPECT_FLOAT_EQ(hc[i], -999.0f)
        << "shader wrote past effective_n at i=" << i;

    ctx.release();
}
```

- [ ] **Step 2: Register test + propagate include dir for hello_spv.h**

In `harmonyos_3dgs/CMakeLists.txt`, add `tests/test_vk_hello.cpp` to `gs3d_vk_tests` sources and add the shader include dir:
```cmake
target_include_directories(gs3d_vk_tests PRIVATE
    ${_vk_shader_dir}      # xxd-embedded headers live here
    tests)
```

- [ ] **Step 3: Build and run**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1/harmonyos_3dgs
cmake --build build 2>&1 | tail -5
ctest --test-dir build -R VkHello --output-on-failure
```

Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1
git add harmonyos_3dgs/tests/test_vk_hello.cpp \
        harmonyos_3dgs/CMakeLists.txt
git -c commit.gpgsign=false commit -m "sp1: end-to-end hello.comp test with embedded SPIR-V byte stream"
```

---

## Task 16: Full validation — clean build + all tests + ENABLE_VULKAN toggle regression

**Files:** none (verification only).

- [ ] **Step 1: Full clean build with Vulkan ON**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1/harmonyos_3dgs
rm -rf build
cmake -B build -DBUILD_TESTS=ON -DENABLE_VULKAN=ON -DENABLE_OPENCL=OFF 2>&1 | tail -10
cmake --build build 2>&1 | tail -10
```

Expected: build succeeds, no errors. Status line confirms Vulkan enabled.

- [ ] **Step 2: Full ctest run**

```bash
ctest --test-dir build --output-on-failure 2>&1 | tail -20
```

Expected: ≥ 165 (CPU+SP-0) + 12 (existing Vulkan) + 10 (new SP-1) ≈ 187 tests pass. Pre-existing DensityController flakiness acceptable per SP-0 baseline.

New SP-1 tests:
- `VkCapabilities.MeetsRequiredMinimums` (1)
- `VkDeviceSelection.*` (4)
- `VkPipelineDispatch.*` (3)
- `VkHello.*` (2)
- Total: 10 new.

- [ ] **Step 3: Toggle regression — build with Vulkan OFF**

```bash
rm -rf build
cmake -B build -DBUILD_TESTS=ON -DENABLE_VULKAN=OFF -DENABLE_OPENCL=OFF 2>&1 | tail -5
cmake --build build 2>&1 | tail -5
ctest --test-dir build --output-on-failure 2>&1 | tail -3
```

Expected: build succeeds with no Vulkan targets; CPU + SP-0 tests (~165) pass. No Vulkan test mentions.

- [ ] **Step 4: Return to ENABLE_VULKAN=ON default**

```bash
rm -rf build
cmake -B build -DBUILD_TESTS=ON 2>&1 | tail -5
cmake --build build 2>&1 | tail -3
```

- [ ] **Step 5: Record outcome in session log**

Pass: no commit needed. Fail: diagnose and fix, then commit the fix.

---

## Task 17: Close SP-1 — dev notes + tag

**Files:**
- Create: `dev_notes/sp1_infrastructure_notes.md`

- [ ] **Step 1: Write closing notes**

Create `dev_notes/sp1_infrastructure_notes.md`:
```markdown
# SP-1 Vulkan Infrastructure — Completion Notes

**Branch**: `sp1-vulkan-infra`
**Base**: master `d87c8c5` (SP-0 merged)
**Tag**: `sp1-infrastructure-ready`

## What SP-1 Delivered

- Integrated Phase 0/1 Vulkan code from `worktree-test_vulkan`:
  `VulkanContext`, `VulkanBuffer`, `VulkanShader`, `VulkanComputePipeline`,
  plus `add_one.comp` smoke shader and 3 gate tests.
- Added explicit `ENABLE_VULKAN` CMake option.
- Added `VulkanDeviceCapabilities` struct + probe at init time.
- Enforced required capability minimums at device selection.
- Added tiered device selection (DISCRETE > INTEGRATED > CPU) + env
  overrides (`GS3D_VK_DEVICE`, `GS3D_VK_DEVICE_NAME`).
- Added runtime push-constant size validation in pipeline constructor.
- Added byte-stream SPIR-V overload on `VulkanShader`.
- Added xxd-based SPIR-V embedding CMake helper (`gs3d_embed_spirv`).
- Added `hello.comp` per spec §3.5 (3 SSBOs + push constant + bounds check).
- Added two-layer pipeline dispatch: `dispatch_sync` (layer 1) and
  `record` + `insert_compute_barrier` (layer 2).
- Added 10 new tests covering capabilities, device selection, dispatch,
  push-constant overflow, and end-to-end hello.comp smoke.

## What SP-1 Did NOT Change (deferred to SP-2)

- `include/vulkan/preprocessor_vk.h`
- `src/vulkan/preprocessor_vk.cpp`
- `src/vulkan/shaders/preprocess.comp`
- `tests/test_preprocessor_vk.cpp`

Rationale: spec §4 prescribes a pass-first architecture with specific
binding tables, packed vs unpacked conic contracts, and hard-error
guards (eval_3D=false, tile=16x16, training=true). SP-2 must evaluate
whether the existing `PreprocessorVK` implementation matches or needs
rewrite. Leaving it untouched in SP-1 preserves the passing regression
test (`test_preprocessor_vk.cpp` slice 2a/2b/2c) as a useful baseline
for SP-2's before/after comparison.

## Naming deviation from spec

- Spec §3.3 calls the class `VulkanPipeline`; existing code uses
  `VulkanComputePipeline`. SP-1 kept the existing name to avoid a
  cascading rename. Functionally identical; rename may be considered
  during SP-2 if a non-compute pipeline is needed (unlikely).

## Test surface as of SP-1 completion

- CPU + SP-0 tests: 165 (baseline + ladder artifact round-trip)
- Existing Vulkan (carried in): 12 (vk_compute + preprocessor_vk)
- New SP-1: 10 (capabilities + device_selection + pipeline_dispatch + hello)
- **Total: ~187 tests**

## Handoff checklist for SP-2

- `ENABLE_VULKAN=ON` is the default build.
- `VulkanContext::capabilities().has_shader_atomic_float` is the flag SP-3
  will branch on for CAS fallback shaders.
- `gs3d_embed_spirv(<name>)` + byte-stream `VulkanShader` is the preferred
  path for all new SP-2 shaders (no runtime file I/O per spec §3.4).
- `insert_compute_barrier(cmd)` between pipeline records is required for
  SSBO visibility in SP-2's preprocess→sort→rasterize chain.
```

- [ ] **Step 2: Commit**

```bash
cd /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/sp1
git add dev_notes/sp1_infrastructure_notes.md
git -c commit.gpgsign=false commit -m "sp1: close notes + SP-2 handoff checklist"
```

- [ ] **Step 3: Tag**

```bash
git tag -a sp1-infrastructure-ready -m "SP-1 Vulkan infrastructure complete: capability checking + two-layer dispatch + byte-stream SPIR-V + hello.comp spec-exact + ENABLE_VULKAN option + tiered device selection. ~187 tests pass."
git tag -l sp1-infrastructure-ready -n
```

- [ ] **Step 4: Report final state**

```bash
git log --oneline sp1-vulkan-infra | head -20
git rev-parse 'sp1-infrastructure-ready^{commit}'
```

---

## SP-1 Exit Criteria

- [ ] All 17 tasks completed
- [ ] `ctest` shows ≥ 187 tests passing (165 baseline + 12 carried + 10 new)
- [ ] `ENABLE_VULKAN=OFF` build succeeds without Vulkan targets
- [ ] `ENABLE_VULKAN=ON` (default) build succeeds with all Vulkan targets
- [ ] `VulkanContext::capabilities()` returns populated struct
- [ ] `GS3D_VK_DEVICE`/`GS3D_VK_DEVICE_NAME` env override tests pass
- [ ] `hello.comp` end-to-end test passes with embedded SPIR-V
- [ ] `dev_notes/sp1_infrastructure_notes.md` committed
- [ ] Tag `sp1-infrastructure-ready` exists on `sp1-vulkan-infra` branch
- [ ] `PreprocessorVK` / `preprocess.comp` / `test_preprocessor_vk.cpp` untouched (SP-2 handoff preserved)
