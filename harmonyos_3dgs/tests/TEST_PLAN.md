# Test Plan — harmonyos_3dgs

Run: `ctest --test-dir harmonyos_3dgs/build`
Build: `cd harmonyos_3dgs && cmake -B build -DBUILD_TESTS=ON && cmake --build build`

## Test Inventory (24 files)

### CPU / Unit Tests
| File | Subsystem | What It Tests |
|------|-----------|---------------|
| `test_types.cpp` | Types | Gaussian struct layout, field access |
| `test_math_utils.cpp` | Math | Matrix ops, quaternion/scale transforms |
| `test_ply_loader.cpp` | I/O | PLY read/write round-trip, field mapping |
| `test_preprocessor.cpp` | Preprocessor | 3D→2D projection, frustum culling, SH eval |
| `test_sh_eval.cpp` | SH Eval | Spherical harmonics evaluation per degree |
| `test_tile_binner.cpp` | Tile Binner | Tile assignment, depth key generation |
| `test_sorter.cpp` | Sorter | Radix sort correctness on (tile_id, depth) keys |
| `test_rasterizer.cpp` | Rasterizer | Alpha compositing, pixel color accumulation |
| `test_rasterizer_backward.cpp` | Backward/Rasterizer | Gradient flow through rasterizer |
| `test_preprocessor_backward.cpp` | Backward/Preprocessor | Gradient flow through preprocessor |
| `test_forward_cache.cpp` | Forward Cache | Caching of per-Gaussian forward pass data |
| `test_loss.cpp` | Loss | L1 + SSIM loss values and gradients |
| `test_optimizer.cpp` | Optimizer | Adam update step, β₁=0.9, β₂=0.999 |
| `test_density_controller.cpp` | Density Controller | Clone/split/prune logic thresholds |
| `test_train_types.cpp` | Training Types | Trainer data structures |

### GPU Tests (OpenCL)
| File | Subsystem | What It Tests |
|------|-----------|---------------|
| `test_preprocessor_gpu.cpp` | Preprocessor GPU | GPU preprocessing matches CPU reference |
| `test_rasterizer_gpu.cpp` | Rasterizer GPU | GPU rasterization matches CPU reference |
| `test_rasterizer_backward_gpu.cpp` | Backward GPU | GPU backward matches CPU reference |
| `test_preprocessor_backward_gpu.cpp` | Backward GPU | GPU preprocessor backward matches CPU |
| `test_trainer_gpu.cpp` | Trainer GPU | Full GPU training step |

### Integration / End-to-End Tests
| File | Subsystem | What It Tests |
|------|-----------|---------------|
| `test_renderer_e2e.cpp` | E2E Render | Full forward render pipeline on tiny_3gaussians.ply |
| `test_trainer_e2e.cpp` | E2E Train | Full training loop (forward + backward + optimize) |
| `test_train_pipeline.cpp` | Train Pipeline | Pipeline orchestration, state transitions |
| `test_train_loop.cpp` | Train Loop | Loop termination, loss convergence on toy data |

## Acceptance Criteria

- All CPU unit tests: PASS
- GPU tests: PASS on OpenCL device (skip if no device)
- E2E render: RMSE vs Python reference ≤ 1e-5
- E2E train: loss decreasing after 100 iterations
- No test modified to weaken an assertion

## Test Data

- `test_data/tiny_3gaussians.ply` — 3-Gaussian scene for fast smoke tests
- `test_data/l1_verify/` — Pre-computed L1 loss ground truth for CPU validation
