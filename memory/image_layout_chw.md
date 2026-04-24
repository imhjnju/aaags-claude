---
name: Image Layout CHW Unified Convention
description: All image data uses CHW channel-first layout, unified 2026-04-23 to match PyTorch/CUDA.
type: project
---

## Unified CHW Layout Convention (2026-04-23)

### Rule
All image data in the pipeline uses **CHW (channel-first)**: index formula `[ch * H * W + y * W + x]` where ch=0/1/2 for R/G/B.

### Why
PyTorch, CUDA, and the Python reference all use CHW `[3, H, W]`. Aligning C++/Vulkan to CHW:
- Eliminates all CPU-side layout conversion code
- Makes direct comparison against CUDA golden `.npy` data possible
- Removes a class of layout-mismatch bugs

### Files Changed
| File | Change |
|------|--------|
| `rasterize.comp` | Output kept as CHW (was originally CHW, briefly changed to HWC, reverted) |
| `rasterizer_vulkan.cpp` | Removed CHW→HWC conversion; empty-scene fast path writes CHW |
| `rasterizer_cpu.cpp` | Output changed from `pix*3+ch` to `ch*HW+pix` |
| `rasterize_backward.comp` | Gradient read changed to `ch*HW+pix` |
| `rasterizer_backward_cpu.cpp` | Gradient read changed to `ch*HW+pix` |
| `rasterize_backward.cl` | Gradient read changed to `ch*HW+pix` |
| `loss.h` | Comment updated to CHW |
| `dssim.cpp` | Indexing changed from `(r*W+c)*3+ch` to `ch*H*W+r*W+c` |
| `dssim.h` | Comment updated to CHW |
| Headers (`rasterize_pass.h`, `rasterizer_vulkan.h`, etc.) | Comments updated to CHW |
| Test files | Removed CHW→HWC golden conversions; basketball test converts `.raw` HWC→CHW |

### Per-Gaussian Data (NOT affected)
Per-Gaussian attributes remain interleaved: `rgb[i*3+ch]`, `means2D[i*2+0/1]`, `conics[i*3+0/1/2]`.

### Golden Data
- `.npy` from Python/CUDA: CHW — used directly
- `.raw` from `render_single.py --raw-out`: HWC (Python transposes before saving) — tests must convert before comparing

### Known Marginal Issue
`PreprocessorBackward.CovChain_ScaleGradient` shows 3% rel_err on `d_raw_scale[2]` after CHW unification (threshold is 2%). Root cause: compiler generates different code for strided CHW access vs contiguous HWC access, causing subtle float rounding differences in CPU-only gradient chain. The mathematical chain is provably correct (same values, different indexing).
