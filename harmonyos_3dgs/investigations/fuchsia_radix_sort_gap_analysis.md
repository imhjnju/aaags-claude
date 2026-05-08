# Fuchsia Vulkan Radix Sort Gap Analysis

## Needed by this project

- Sort 64-bit packed key/value entries for 3DGS tile/depth ordering.
- Record sort work into an existing Vulkan command buffer.
- Support NVIDIA development hardware and HarmonyOS/Maleoon target planning without removing the existing CPU/std sort fallback path.
- Preserve current cascade/config specialization IDs and eval_3D training replay behavior.

## Available in Fuchsia library

- Direct command-buffer sort of 64-bit keyvals via `radix_sort_vk_sort`.
- Per-target auto-detect for vendored target blobs.
- Explicit memory-requirement queries for keyval and internal scratch buffers.
- Sorted-output ping-pong buffer reporting through `keyvals_sorted`.

## Gaps and constraints

- Vendored targets in the profiling worktree cover NVIDIA sm35 and ARM bifrost8. HarmonyOS Maleoon availability is not proven by the vendored target set.
- The library requires Vulkan 1.2 features including shader int64, buffer device address, and Vulkan memory model support; current runtime must continue to fail clearly or fall back if unavailable.
- The library sorts packed keyvals, so existing users of separate `keys_sorted` / `values_sorted` need careful adaptation.
- The current merge worktree already has master cascade/config and eval_3D replay specialization constants; profiling shader/pass files must be merged selectively rather than copied wholesale.
