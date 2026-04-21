# SH Evaluation + RGB Audit — VK vs CUDA

**Date:** 2026-04-21
**CUDA anchor:** `forward_common.h::computeColorFromSH`, `auxiliary.h` SH constants
**VK anchor:** `preprocess.comp::shToRGB` (lines 820–878)

## Summary Table

| Item | CUDA | VK | Match? | PSNR Impact |
|---|---|---|---|---|
| SH degree evaluated | up to D=3 | up to pc.sh_degree=3 | YES | — |
| SH_C0–C3 coefficients | all 16 values | identical values | YES | — |
| Coefficient memory layout | sh[idx*max_coeffs*3 + k*3 + ch] | sh[base + k*3 + ch] where base=i*K*3 | YES | — |
| View direction sign | dir = mean3D - campos, normalize | dir = pos - cam_pos, normalize | YES | — |
| +0.5 DC offset | result += 0.5f | result += vec3(0.5) | YES | — |
| Lower RGB clamp | max(result, 0.0f) — always | max(result, vec3(0.0)) — always | YES | — |
| Upper RGB clamp (training) | No upper clamp in CUDA forward | min(result, 1.0) only if spec_training==0 | DIVERGENCE (training mode: inactive) | None for golden |
| spec_training | N/A (CUDA uses clamped[] array for backward) | spec_training=1u hardcoded | VK matches aaa.json intent | — |

## Conclusion

**All SH math is numerically identical for training-mode evaluation (the golden test scenario).**
- Identical SH constants (bit-for-bit)
- Identical coefficient memory layout
- Identical view direction convention
- Identical polynomial expressions for all 16 bands
- Identical +0.5 offset and lower-clamp to zero

The only divergence (VK inference upper-clamp min(result,1.0)) is INACTIVE for the golden test
case since spec_training=1u is hardcoded. No SH-related bug exists for the current golden.
