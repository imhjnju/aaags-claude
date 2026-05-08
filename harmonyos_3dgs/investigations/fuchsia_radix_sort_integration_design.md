# Fuchsia Vulkan Radix Sort Integration Design

## Integration approach

- Vendor the profiling worktree's `third_party/fuchsia_vk_radix_sort` tree and build it as a local static dependency through CMake.
- Add the local RAII wrapper `RadixSortFuchsia` plus `VkAlignedBuffer` for aligned scratch/keyval storage.
- Use the Fuchsia path from `SorterVulkan` only when target detection and required Vulkan features are available; preserve existing sort behavior as the compatibility path.
- Keep all caller-owned synchronization explicit: callers record upload/barriers before sort and consume the wrapper's sorted-buffer result after sort.

## Conflict policy for this merge

- Do not replace `include/splatting_settings.h`, `src/splatting_settings.cpp`, or current `SplattingSettings` validation.
- Do not replace `include/vulkan/preprocess_bindings.h`; specialization IDs remain `EVAL_3D=0`, `TRACE_ENABLED=1`, `SORT_MODE=2`, `EVAL3D_RAW_REPLAY=3`.
- Do not replace `src/vulkan/shaders/rasterize.comp`; current merge keeps the TAIL/MID/HEAD cascade trace path, non-trace HEAD_W=8 fallback, and CUDA-style `n_contrib` replay contract.
- Port profiling performance changes in dependency order: vendor/wrapper, CMake wiring, sorter packed-key path, then optional renderer recorded fast path.

## Validation

- Build with `cmake --build build`.
- Run focused CTest filters for sorter/Fuchsia/rasterizer/cascade/training parity before full CTest.
- Full suite must pass before treating the sync as complete.
