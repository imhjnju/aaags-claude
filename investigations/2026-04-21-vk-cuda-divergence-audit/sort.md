# Sort Stage Audit — VK vs CUDA

**Date:** 2026-04-21
**CUDA anchor:** `rasterizer_impl.cu` CUB SortPairs, `auxiliary.h:264-270` key construction
**VK anchor:** `radix_sort_count.comp`, `radix_sort_scatter.comp`, `sorter_vulkan.cpp`

## Summary Table

| Item | CUDA | VK | Match? | PSNR Impact |
|---|---|---|---|---|
| Key width | 64-bit | 64-bit | YES | — |
| Upper 32 bits | tile_id | tile_id | YES | — |
| Lower 32 bits eval_3D | floatBitsToUint(max_pos_depth + 8.0f) | floatBitsToUint(max_pos_depth + 8.0) | YES | — |
| Lower 32 bits non-eval_3D | floatBitsToUint(depths[idx] + 8.0f) | floatBitsToUint(depths[i]) — NO +8 | **MISMATCH** | N/A (eval_3D=true) |
| sort_mode=3 HIERARCHICAL | Per-pixel HEAD_WINDOW=4 k-buffer | NOT IMPLEMENTED | **MISSING** | High |
| queue_sizes (per_pixel=4, tile_2x2=8, tile_4x4=64) | Used by HIERARCHICAL | Not used | **MISSING** | High |
| Bit range | 32 + ceil(log2(num_tiles)) ≈ 45 bits | All 64 bits (16 passes) | Correctness OK; perf waste | Low |
| Stability | CUB stable | VK thread-0 sequential: stable | Match | — |
| CPU fallback | N/A | std::stable_sort for R > 256 | Functionally correct | — |

## Detailed Findings

### sort_mode=3 HIERARCHICAL — CRITICAL MISSING

aaa.json: sort_mode=3 = SortMode::HIERARCHICAL, queue_sizes: {per_pixel: 4, tile_2x2: 8, tile_4x4: 64}.

CUDA: sortGaussiansRayHierarchicalCUDA_forward (hierarchical_render.cuh 1062–1159).
Each pixel maintains sorted k-buffer of up to 4 Gaussians.
2×2 tile holds 8; 4×4 tile holds 64.
Alpha culling at each level.

VK: simple global sort + linear rasterize traversal. Equivalent to sort_mode=0 (GLOBAL).

### CPU fallback for R > 256

sorter_vulkan.cpp: GPU radix shaders are used only for R ≤ 256. Real scenes (basketball: >2000 pairs) fall back to std::stable_sort on CPU. Sort results are correct; GPU path is proof-of-concept only.

### Negative depth (eval_3D=false path)

VK scatter.comp:277: `floatBitsToUint(depths[i])` — no +8.0 bias.
CUDA: all paths add +8.0 before passing to constructSortKey.
Latent bug; inactive when eval_3D=true (current golden config).
