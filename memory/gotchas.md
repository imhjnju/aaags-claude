# Gotchas — Bug Prevention Patterns

Patterns discovered during development. Check after implementing changes.
Canonical source for "don't do X because Y happened."

Cross-references to deeper docs use "See also:" lines.

(None yet — will be populated as bugs are found and root-caused.)

## Known Domain Traps (pre-populated from domain knowledge)

### Floating-Point Determinism
- Build flag `-ffp-contract=off` is MANDATORY on all targets. Without it, FMA instructions produce different results on ARM vs x86, breaking cross-platform numerical matching.
- On cross-compile (HarmonyOS/OHOS), the CMakeLists.txt already forces this flag. Don't remove it.
- See also: `spec/README.md` §Floating-Point Invariants

### Spherical Harmonics Degree
- SH evaluation is degree-dependent. The number of SH coefficients per Gaussian is `(degree+1)^2 * 3`. Mismatched degree between preprocessing and rasterization causes silent output corruption (wrong colors, no crash).

### Tile Binner / Rasterizer Key Ordering
- Tile-based rasterization requires Gaussians to be sorted by (tile_id, depth). Wrong sort order = wrong alpha compositing = incorrect rendering. The sorter must run BEFORE the rasterizer.

### rasterize.comp Output Is CHW, Not HWC (discovered 2026-04-20)
- **Bug**: `rasterize.comp` writes `out_image[ch * HW + px]` (channel-first, CHW layout). The loss function (`compute_combined_loss_gradient`) and `rasterize_backward.comp` both expect HWC (`dL_dpixels[px*3+ch]`).
- **Fix**: `vulkan_trainer.cpp` converts `image_` CHW→HWC after the forward pass, before loss/backward.
- **Symptom**: Without fix, gradient L2 norms are ~50% below Python autograd reference for all param groups. Loss values are correct (L1 loss on zero target is layout-invariant, so it cannot detect this).
- **Why CpuVkCompare didn't catch it**: That test uses an all-zero target. L1 loss = sum|rendered| is invariant to pixel layout permutation, so loss values matched despite scrambled backward gradients.
- **Long-term fix**: Change `rasterize.comp` to write HWC directly (tracked TODO in vulkan_trainer.cpp line ~283). At 720×960 the current CPU copy is ~8 MB/step.
- **Detection method**: Compare VK gradient norms against Python autograd reference (`test_vk_vs_py_reference.cpp`). Use a non-zero GT image or gradient norm comparison — loss alone is insufficient.
