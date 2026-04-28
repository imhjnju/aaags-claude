# scan1 VK↔CUDA Parity — S13 n_contrib Replay-Boundary Fix

Date: 2026-04-28
Dataset: `/home/robota/h00813233/Graph/datasets/scan1`
Camera: 0 (`00000000`), 1600×1200
Initial PLY: `/tmp/scan1_init_3dgs.ply` (28,747 Gaussians)
Mode investigated: independent training, `eval_3D=false`, `proper_ewa_scaling=false`, L1-only loss

## Goal

Measure Vulkan vs CUDA training drift at 10, 200, and 1000 iterations on scan1, then locate stable implementation differences by backtracking from observed numerical discrepancies.

## Baseline Observations

Initial VK and CUDA renders were bit-identical and all black. The apparent initial loss mismatch was not a render/training bug: CUDA reports the PyTorch reduction value, while the C++ path logs a sequential `float` accumulation over 5.76M pixels. The latter reproduces the VK log value.

A disabled-DSSIM performance bug made full-resolution scan1 training too slow: `compute_combined_loss_gradient()` computed SSIM even when `lambda_dssim == 0`. Adding an early return after the L1 path restored feasible scan1 runtime.

## Main Symptom

Before the fix, step1 showed SH-gradient drift while position, opacity, scale, and rotation gradients were zero/aligned. Longer runs diverged strongly:

| Steps | VK↔CUDA PSNR before fix | VK↔CUDA PSNR after fix |
|-------|--------------------------|-------------------------|
| 10 | 90.91 dB | 107.136986 dB |
| 200 | 22.42 dB | 49.822707 dB |
| 1000 | 8.24 dB | 27.773822 dB |

After the fix, the 1000-step fixed run reported:

- CUDA-vs-GT PSNR: 19.753679 dB
- VK-vs-GT PSNR: 19.868585 dB
- VK-vs-CUDA PSNR: 27.773822 dB
- Render max_abs: 0.6992977
- Render mean_abs: 0.0225525

## Root Cause

`n_contrib` semantics differed between CUDA and Vulkan/CPU 2D paths.

CUDA stores the 1-based position of the last candidate Gaussian that actually blended during tile traversal. Candidates skipped by power/alpha checks before that position still advance the replay position.

The Vulkan and CPU 2D paths had stored the number of Gaussians that actually blended. Backward replay used that count as a stopping condition, so it replayed the wrong suffix when skipped candidates existed before the last blended candidate.

This primarily perturbed `d_rgb`, which then affected SH gradients and accumulated through Adam.

## Fix

Updated the 2D path to use CUDA-style replay-boundary semantics:

- `harmonyos_3dgs/src/vulkan/shaders/rasterize.comp`
  - Track candidate position in tile traversal.
  - Store the last candidate position that blended as `n_contrib`.

- `harmonyos_3dgs/src/vulkan/shaders/rasterize_backward.comp`
  - Interpret `n_contrib` as last candidate position.
  - In back-to-front replay, ignore candidates behind that position rather than counting blended contributors.

- `harmonyos_3dgs/src/cpu/rasterizer_cpu.cpp`
  - Align CPU 2D cache semantics with CUDA/Vulkan 2D behavior.

- `harmonyos_3dgs/src/cpu/rasterizer_backward_cpu.cpp`
  - Replay only candidates up to `last_contrib`.

- `harmonyos_3dgs/include/types.h` and `tests/test_rasterizer_backward_vulkan.cpp`
  - Update comments/contracts so tests assert the new semantics rather than documenting the old count-based behavior.

## Validation

Build and tests after the fix:

```text
cmake -S harmonyos_3dgs -B harmonyos_3dgs/build -DBUILD_TESTS=ON
cmake --build harmonyos_3dgs/build
ctest --test-dir harmonyos_3dgs/build --output-on-failure
```

Result: 273/273 tests passed in 36.35s.

Focused affected test:

```text
RasterizerBackwardVulkan.MatchesCPU_TinyFixture PASS
```

Stepdump after the fix:

- Step1 render: exact match, all zeros.
- Step1 `grad_pos`, `grad_op`, `grad_sca`, `grad_rot`: exact zero/aligned.
- Step1 SH gradients: remaining atomic-order-scale difference only (`l2_rel≈6.2e-4`, max `≈1.9e-7`).
- Step2 render: still close (`l2_rel≈4.3e-5`, max `≈3.9e-6`).

## Remaining Risk / Next Experiment

The remaining long-run drift appears numerically amplified rather than a newly localized formula bug. Adam uses `eps=1e-15`, so near-zero gradients can become finite parameter deltas once tiny render/atomic-order differences appear.

Do not chase another shader formula change until a controlled same-state experiment proves a deterministic split. The recommended next experiments are:

1. Start both implementations from an identical post-step1 state and compare step2 forward/backward.
2. Use a non-degenerate initialization to avoid near-zero/identity-quaternion sensitivity.
3. Run `eval_3D=true` training/rendering separately and record it as a new mode, not as evidence against the fixed 2D replay-boundary result.
