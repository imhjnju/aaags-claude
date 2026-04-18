# SP-0 External Dependency Inventory

> Source of truth for external packages, hardware, and decisions SP-0
> validation depends on. Updated when dependency versions change.

## fused_ssim

- **Package**: `fused_ssim` (PyPI distribution name: `fused-ssim`)
- **Version**: 1.0.0
- **Install path (aaa-gs env)**: `/home/robota/miniconda3/envs/aaa-gs/lib/python3.10/site-packages/fused_ssim/`
- **Compiled extension**: `/home/robota/miniconda3/envs/aaa-gs/lib/python3.10/site-packages/fused_ssim_cuda.cpython-310-aarch64-linux-gnu.so` (aarch64, prebuilt against the current torch/CUDA stack for Thor)
- **Source availability**:
  - Site-packages dir contains ONLY the Python wrapper — no `.cu` / `.cpp` source.
  - Contents of `site-packages/fused_ssim/`:
    - `__init__.py` (2813 bytes — pure Python wrapper, defines `fused_ssim`, `fused_ssim3d`, `FusedSSIMMap` autograd.Function; imports the native symbols from `fused_ssim_cuda`)
    - `__pycache__/`
  - Native symbols (`fusedssim`, `fusedssim_backward`, `fusedssim3d`, `fusedssim_backward3d`) live in the separate compiled `.so`; no CUDA source is distributed with the wheel.
  - HOWEVER, the wheel's `direct_url.json` records the exact upstream git install source:
    - `url`: `https://github.com/rahul-goel/fused-ssim/`
    - `vcs_info.commit_id`: `a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38`
    - `vcs`: `git`
  - Meaning: CUDA source IS obtainable upstream at a pinned commit; it is just not redistributed into site-packages. Cloning the repo at that commit reproduces the exact kernels behind this installation.
- **Git source (if known)**: `https://github.com/rahul-goel/fused-ssim/` @ `a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38` (recorded in `fused_ssim-1.0.0.dist-info/direct_url.json`).
- **Home-page (METADATA)**: `https://github.com/rahul-goel/fused-ssim`
- **Downstream usage**: `AAA-Gaussians/train.py:103` — `fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))`.
  The training loop uses `fused_ssim(img1, img2, padding='same', train=True)` with `C1 = 0.01**2`, `C2 = 0.03**2`, returning `map.mean()` (scalar).
- **SP-4 impact**: Vulkan SSIM shader output MUST match `fused_ssim` bitwise.
  - Since upstream CUDA source is available at a pinned commit, the port path is: clone `rahul-goel/fused-ssim@a7c48d6`, read the CUDA kernel, re-implement directly in GLSL with the same constants, tile layout, and accumulation order.
  - Probing fallback is NOT required: the operator is not a black box because the pinned upstream source is public.
- **SP-0 verdict**: **Source available (upstream git, pinned commit) — direct port path clear.** Site-packages ships only the compiled `.so`, but `direct_url.json` gives us a reproducible git coordinate for the kernel source. SP-4 can port the CUDA kernel to GLSL directly; no black-box probing phase is required before SSIM golden is trusted.

## CUB DeviceRadixSort Determinism

- **Probe**: `tools/verify_cub_determinism.py`
- **Run date**: 2026-04-17
- **Verdict**: DETERMINISTIC
- **Evidence**: 10 runs on fixed-seed scene (N=500 gaussians) produced identical
  `binningBuffer` SHA256:
  `3f3e5914cbc97d4a9b7bc277e30e48751a5215b350265d2ff5383f74d5ae39db`
- **Scene**: 500 gaussians clustered at world z=-3, camera looking down -z,
  256x256 @ fov=90°, settings: `SortMode.GLOBAL`, `Z_DEPTH`, `eval_3D=False`.
  `num_rendered = 11923` sort entries confirmed meaningful sort work (not a
  trivial/empty sort).
- **Hardware tested**: NVIDIA Thor
- **Toolchain**: PyTorch 2.9.0+cu130, torch.version.cuda = 13.0
- **Implication for SP-0..SP-6**:
  - Sort golden validation uses EXACT byte-match (not the weaker
    "tile-grouping consistent" fallback that would be needed if CUB produced
    run-to-run variation).
  - Spec §1.2 tie-break handling: CUB is deterministic => the Vulkan radix
    sort must replicate its stable ordering (key: tile-id | depth; ties
    broken by original gaussian index), not just "any sort that puts equal
    keys in some order."
  - Any future change of CUB version, GPU architecture, or `SortMode` MUST
    re-run this probe — the DETERMINISTIC verdict is specific to the
    toolchain recorded here.

## Toolchain

- **PyTorch**: 2.9.0+cu130
- **torch.version.cuda**: 13.0
- **CUDA (nvcc)**: release 13.0, V13.0.48 (Build cuda_13.0.r13.0/compiler.36260728_0, built Wed Jul 16 2025)
- **GPU**: NVIDIA Thor
- **Python**: 3.10.20 (main, Mar 11 2026, 17:41:27) [GCC 14.3.0]
- **diff-gaussian-rasterization**: locally patched at
  `AAA-Gaussians/submodules/diff-gaussian-rasterization/` on branch
  `sp0-cuda-golden-infra` (inner submodule commit `3373529467665e00632747e316a9d1502c54bd92`).
  Adds `materialize_dump` and `rasterize_gaussians_backward_dump` bindings
  per SP-0 T8–T11. This is a WORKTREE-local patch; other clones need the
  same local-protocol submodule init trick (file://-style submodule remap)
  or a cherry-pick of the same commit into their own fork of
  diff-gaussian-rasterization.

## Conda env

- **Name**: `aaa-gs`
- **Location**: `/home/robota/miniconda3/envs/aaa-gs`
- **Activation**: `conda run -n aaa-gs <cmd>` (used throughout SP-0 tools —
  all Python probes, dump scripts, and CUB determinism verification run
  under this environment).
