# Vendored: Fuchsia vk_radix_sort

## Upstream
- **Canonical project:** Fuchsia, subtree `src/graphics/lib/compute/radix_sort/`
  - URL: https://fuchsia.googlesource.com/fuchsia
- **Mirror used to vendor:** https://github.com/juliusikkala/fuchsia_radix_sort
- **Mirror commit SHA:** `4165014fb8dd77effeb046a874236e15113dd4de`
  - (captured from `spike/fuchsia_radix_sort/upstream/.git`)

## Copy
- **Date:** 2026-04-25
- **Vendored path:** `harmonyos_3dgs/third_party/fuchsia_vk_radix_sort/`
- **License:** BSD-2-Clause (Copyright 2019 The Fuchsia Authors). See `LICENSE`.

## Trim list
Per integration plan §2 + §8 Decision #3 (vendor-tree scope: targets-only).

**Dropped vendor targets** (source files removed from `platforms/vk/targets/vendors/`):
- `amd/gcn3`
- `arm/bifrost4`
- `intel/gen8`

**Kept vendor targets:**
- `nvidia/sm35` — Tegra Thor today.
- `arm/bifrost8` — Maleoon 920 (HarmonyOS NEXT) defensive default.

The vendored top-level `CMakeLists.txt` `targets` list was edited to reference
only the kept vendors (otherwise CMake configure would fail trying to glob
shader includes from the deleted dirs).

## Local patches
1. **`-Os` removal from glslangValidator invocation** in vendored top-level
   `CMakeLists.txt` (the `shader()` function's `add_custom_command`). Our
   `glslangValidator` is built without SPIRV-Tools optimizer and errors with
   `-Os not available`. SPIR-V output is identical without the size-optimizer.
   See `spike/fuchsia_radix_sort/SPIKE_REPORT.md` §2. Comment marker:
   `# AAAGS LOCAL PATCH: removed -Os, ...`.
2. **`targets` list trim** in vendored top-level `CMakeLists.txt`. Comment
   marker: `# AAAGS LOCAL PATCH: vendor-tree trimmed ...`.
3. **`platforms/vk/target.c` auto-detect trim** — removed `extern` declarations
   and switch cases for the dropped vendors (gcn3/gen8/bifrost4); otherwise
   the static lib would have unresolved external references at link time.
   Comment marker: `// AAAGS LOCAL PATCH: vendor-tree trimmed ...`.

## Library target
The vendored CMakeLists creates `vk-radix-sort` (STATIC), with PUBLIC link to
`Vulkan::Vulkan` and PUBLIC include of `platforms/vk/include/`. Consumers
include `<radix_sort/platforms/vk/radix_sort_vk.h>`.
