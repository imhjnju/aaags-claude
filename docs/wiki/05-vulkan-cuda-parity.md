# Vulkan/CUDA Parity

## Goal

The Vulkan/C++ implementation should match the Python/CUDA reference behavior before performance-specific divergence is accepted.

## Authority Stack

1. Python/CUDA reference in `AAA-Gaussians/`.
2. Behavioral contracts in `spec/README.md`.
3. Golden fixtures and parity tests in `harmonyos_3dgs/tests/`.
4. Harness results and session records in `dev_notes/`.

## Parity Discipline

Use evidence before theory:

- Read the relevant reference code.
- Compare intermediate states, not only final images.
- Measure errors and locate first divergence.
- Change one variable per experiment.
- Keep diagnostic patches separate from production changes.
- Preserve exact feature/config parity before interpreting metrics.

## Common Parity Dimensions

| Dimension | Examples |
|-----------|----------|
| Camera/config | view/projection matrices, FOV, image dimensions, background. |
| Render flags | `eval_3D`, `proper_ewa`, parity mode, SH degree. |
| Data layout | CHW/HWC image layout, SH DC/rest layout, raw parameter order. |
| Forward state | means2D, conics, opacity, radii, tile bins, sorted IDs. |
| Backward state | image gradients, raster/preprocess gradients, raw param gradients. |
| Optimizer state | Adam m/v groups, LR schedule, eps, terminal step semantics. |
| Densification | clone/split/prune vs MCMC relocate/add, cap, gates, opacity reset. |

## Current Known Themes

For current status, read `dev_notes/session_state.md` and topic memories via `memory/MEMORY_INDEX.md`.

Durable lessons already captured include:

- Terminal training step semantics matter: final forward/backward may skip Adam to match CUDA harness behavior.
- Render-config parity matters: saved PLY standalone render must match training render flags.
- SH group layout must respect DC/rest gather/scatter, not contiguous memcpy assumptions.
- `n_contrib` replay-boundary semantics must match CUDA's last-contributor position contract.
- DSSIM performance work must preserve direct clamp-window loss semantics.

## Parity Experiment Template

Use [Experiment Report Template](templates/experiment-report.md) and include:

- Dataset and camera/view schedule.
- Initial PLY and any preprocessing/filter generation.
- Exact CUDA and Vulkan commands.
- Shared seeds and densification gates.
- Feature flags and render flags.
- Gaussian counts.
- PSNR/SSIM/L1 or other metrics.
- Artifact paths.
- Gate status and failure reason.

## Interpreting Metrics

Do not rely on a single final PSNR number.

- A high saved-vs-train PSNR confirms serialization/render-config consistency.
- A small CUDA/VK GT-quality gap can be acceptable even when parameter trajectories differ.
- Count mismatch under densification usually invalidates visual metric comparison.
- Feature mismatches can create visible artifacts that look like serialization bugs.

## Escalation Rule

After three failed hypotheses or a fix that does not improve the measured target:

1. Stop editing.
2. Save the current evidence.
3. Re-read reference and relevant spec/tests.
4. Ask for or spawn a fresh-eyes review.
