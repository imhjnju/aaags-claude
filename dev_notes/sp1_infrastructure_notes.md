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
- **Total: 187 tests**

Hardware probed (NVIDIA Thor, Vulkan 1.4.315):
- `max_push_constants_size`: 256 (meets ≥128 requirement)
- `max_compute_workgroup_invocations`: 1024
- `max_compute_shared_memory_size`: 49152
- `subgroup_size`: 32
- `has_shader_atomic_float`: **true** (SP-3 native path available)

## Handoff checklist for SP-2

- `ENABLE_VULKAN=ON` is the default build.
- `VulkanContext::capabilities().has_shader_atomic_float` is the flag SP-3
  will branch on for CAS fallback shaders.
- `gs3d_embed_spirv(<name>)` + byte-stream `VulkanShader` is the preferred
  path for all new SP-2 shaders (no runtime file I/O per spec §3.4).
- `insert_compute_barrier(cmd)` between pipeline records is required for
  SSBO visibility in SP-2's preprocess→sort→rasterize chain.

## Known issues (pre-existing, non-blocking)

- `DensityController.CloneSmallGaussians`/`SingleGaussianClone`/`CloneAndSplitSameStep`/
  `MultipleDensifyCycles` are intermittently flaky (1-3 failures out of
  ~165 tests per run, non-deterministic). Pre-existing from before SP-0.
  Not SP-1 introduced. Re-runs typically pass. Deferred investigation.
