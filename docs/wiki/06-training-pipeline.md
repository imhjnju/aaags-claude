# Training Pipeline

## Scope

This page is a reader-facing map of the training pipeline. For exact behavior, read source, tests, and `spec/README.md`.

## High-Level Flow

```text
load model + cameras + GT
      ↓
preprocess Gaussians
      ↓
tile/bin/sort
      ↓
rasterize forward
      ↓
compute loss and image gradients
      ↓
rasterize/preprocess backward
      ↓
regularization / gradient processing
      ↓
Adam update
      ↓
optional densification / opacity reset
      ↓
save final render / PLY / metrics
```

## Main Components

| Component | Responsibility |
|-----------|----------------|
| Model loader/saver | Raw Gaussian parameters, SH, opacity, scale, rotation, optional `filter_3D`. |
| Camera loader | View/projection setup, dimensions, GT mapping. |
| Preprocessor | 3D to 2D projection, covariance, radii, culling, AAA/eval_3D behavior. |
| Tile binner/sorter | Candidate enumeration and sorted per-tile processing order. |
| Rasterizer | Alpha compositing and forward image output. |
| Loss | L1/DSSIM combined loss and image gradients. |
| Backward | Raster and preprocess gradient propagation. |
| Optimizer | Adam updates for position, SH, opacity, scale, rotation. |
| Densification | Legacy clone/split/prune or MCMC relocate/add growth. |
| Harnesses | End-to-end CUDA/Vulkan comparison and reporting. |

## Training Semantics to Preserve

- Image buffers use the project's established layout; verify before editing loss/backward paths.
- Final-step semantics must match the comparison target: CUDA harness may compute final forward/backward while skipping final Adam.
- SH active degree may differ during warmup; saved PLY rendering must match final-forward active degree when comparing.
- `proper_ewa` and `eval_3D` flags must be consistent across training and standalone render.
- Densification gates must align between CUDA and Vulkan before comparing counts or metrics.

## Densification Modes

| Mode | Trigger | Notes |
|------|---------|-------|
| Legacy clone/split/prune | `cap_max <= 0` or explicit legacy config | Preserves older controller behavior. |
| MCMC relocate/add | `cap_max > 0` | Matches AAA-Gaussians MCMC-style growth and relocation behavior. |

For exact current CLI behavior, read `dev_notes/session_state.md` and `gs3d_vk_train` code.

## Loss and DSSIM Notes

DSSIM should preserve clamp-window SSIM semantics while remaining performant. If changing loss code:

- Compare against direct reference behavior.
- Validate gradients with finite difference or reference tests.
- Benchmark runtime if training loop cost changes.
- Record performance evidence in dev notes.

## Pipeline Change Checklist

Before changing training pipeline code:

- [ ] Read relevant spec section.
- [ ] Read existing tests covering the subsystem.
- [ ] Identify the Python/CUDA reference code.
- [ ] Decide whether the change affects behavior, performance, or only structure.
- [ ] Add/adjust tests before implementation if behavior changes.
- [ ] Run targeted and full tests.
- [ ] Update docs/state if semantics changed.
