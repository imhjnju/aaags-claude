# CUDA Reference Checkout 3373529

All worktree agents should use the clean CUDA/DGR reference commit `3373529` as the default CUDA-side reference for AAA-Gaussians comparison, parity, and quality harness runs unless the user explicitly asks for another reference.

## Why

Recent Vulkan/CUDA training comparisons rely on a known-good CUDA reference state. Dirty/current CUDA extension checkouts can diverge or produce NaN/failure modes that should be reported as reference failures, not silently substituted into Vulkan quality results.

## How to apply

- Prefer an existing clean checkout such as `harmonyos_3dgs/build/dgr_3373529_clean` when present; otherwise create or verify a clean checkout at commit `3373529` before CUDA reference runs.
- Keep Vulkan and CUDA camera schedules, train/test splits, seeds, and training parameters explicit in reports.
- Do not silently replace a failing CUDA reference run with another checkout or parameter set; report the CUDA reference failure and the exact checkout/config used.
- Historical stability/parity runs may still pass explicit zero-reg/noise/constant-LR overrides, but production AAA-GS runs should keep the README-aligned defaults unless the task says otherwise.

## Related operational reminder

For long-running CUDA/Vulkan training and rendering jobs, start background commands with `stdbuf -oL -eL` or ensure the executable flushes progress logs with `fflush(stdout)`. Once launched without flushing, external force-flush may be blocked by ptrace/Yama and should not be treated as reliable.
