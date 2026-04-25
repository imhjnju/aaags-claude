# Gotchas — Bug Prevention Patterns

Patterns discovered during development. Check after implementing changes.
Canonical source for "don't do X because Y happened."

Cross-references to deeper docs use "See also:" lines.

## Known Domain Traps (pre-populated from domain knowledge)

### Floating-Point Determinism
- Build flag `-ffp-contract=off` is MANDATORY on all targets. Without it, FMA instructions produce different results on ARM vs x86, breaking cross-platform numerical matching.
- On cross-compile (HarmonyOS/OHOS), the CMakeLists.txt already forces this flag. Don't remove it.
- See also: `spec/README.md` §Floating-Point Invariants

### Spherical Harmonics Degree
- SH evaluation is degree-dependent. The number of SH coefficients per Gaussian is `(degree+1)^2 * 3`. Mismatched degree between preprocessing and rasterization causes silent output corruption (wrong colors, no crash).

### Tile Binner / Rasterizer Key Ordering
- Tile-based rasterization requires Gaussians to be sorted by (tile_id, depth). Wrong sort order = wrong alpha compositing = incorrect rendering. The sorter must run BEFORE the rasterizer.

### Image Buffer Layout: Unified CHW (2026-04-23)

**Rule**: All image data in this project uses **CHW (channel-first)** layout: `[ch * H * W + y * W + x]`. This matches PyTorch/CUDA/Python conventions.

**Components using CHW**:
- `rasterize.comp` output: `out_image[ch*HW + px]`
- `rasterize_backward.comp` gradient input: `dL_dpixels[ch*HW + pix]`
- CPU rasterizer output: `out_img[ch*HW + pix]`
- CPU backward gradient input: `d_image[ch*HW + pix]`
- Loss function (`loss.h`): operates on flat `[3*H*W]` array (layout-invariant)
- SSIM (`dssim.cpp`): `rendered[ch*H*W + r*W + c]`
- OpenCL backward: `d_image[ch*HW + pix]`
- Python reference: PyTorch tensor `[3, H, W]` — same CHW convention
- CUDA golden data: `.npy` files are CHW `[3, H, W]`
- Empty-scene fast path: `output_image[ch*HW + px] = bg_color[ch]`

**Why CHW**: PyTorch, CUDA, and the Python reference implementation all use CHW. Aligning C++/Vulkan to CHW eliminates all layout conversion code and makes direct comparison against reference data possible.

**Per-Gaussian data is interleaved, NOT CHW**: `rgb[i*3+ch]`, `means2D[i*2+0/1]`, `conics[i*3+0/1/2]` — these are per-element attributes, not image planes. Don't confuse the two.

**Writing new shaders**: For image output, use `out_image[ch * HW + px]`. For per-Gaussian data, use interleaved `data[i * stride + component]`.

### Layout-Invariant Loss Can Hide Layout Bugs
- L1 loss and MSE loss iterate linearly over all elements — layout-invariant. A layout mismatch between rendered and target would NOT be detected by loss alone.
- SSIM IS layout-sensitive — it uses 2D spatial windows per channel.
- **Detection method**: Compare gradient norms against Python autograd reference, or use PSNR against CHW golden data.

### Golden Data Formats
- `.npy` files (from Python/CUDA): CHW `[3, H, W]` float32 — matches our C++ output directly.
- `.raw` files (from `render_single.py --raw-out`): HWC `[H, W, 3]` float32 — Python tool transposes before saving. Tests that read `.raw` must convert HWC→CHW before comparing.
