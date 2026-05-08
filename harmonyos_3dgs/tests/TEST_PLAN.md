# Test Plan — harmonyos_3dgs

Run: `ctest --test-dir harmonyos_3dgs/build`
Build: `cd harmonyos_3dgs && cmake -B build -DBUILD_TESTS=ON && cmake --build build`
Current CTest total: **315 tests** (2026-04-30 baseline, 66 .cpp test files)

## Test Inventory (66 files)

### CPU / Unit Tests
| File | Subsystem | What It Tests |
|------|-----------|---------------|
| `test_types.cpp` | Types | Gaussian struct layout, field access |
| `test_math_utils.cpp` | Math | Matrix ops, quaternion/scale transforms |
| `test_ply_loader.cpp` | I/O | PLY read/write round-trip, field mapping |
| `test_npy_reader.cpp` | I/O | NPY fixture loading |
| `test_manifest.cpp` | I/O | Golden manifest parsing |
| `test_golden_roundtrip.cpp` | I/O | Golden fixture serialization round-trip |
| `test_preprocessor.cpp` | Preprocessor | 3D→2D projection, frustum culling, SH eval |
| `test_sh_eval.cpp` | SH Eval | Spherical harmonics evaluation per degree |
| `test_tile_binner.cpp` | Tile Binner | Tile assignment, depth key generation |
| `test_sorter.cpp` | Sorter | Radix sort correctness on `(tile_id, depth)` keys |
| `test_rasterizer.cpp` | Rasterizer | Alpha compositing, pixel color accumulation |
| `test_rasterizer_backward.cpp` | Backward/Rasterizer | Gradient flow through rasterizer |
| `test_preprocessor_backward.cpp` | Backward/Preprocessor | Gradient flow through preprocessor |
| `test_forward_cache.cpp` | Forward Cache | Caching of per-Gaussian forward pass data |
| `test_loss.cpp` | Loss | L1 loss values and gradients |
| `test_dssim.cpp` | Loss | DSSIM loss values and gradients |
| `test_optimizer.cpp` | Optimizer | Adam update step |
| `test_cpu_adam.cpp` | Optimizer | CPU Adam reference behavior |
| `test_density_controller.cpp` | Density Controller | Clone/split/prune logic thresholds |
| `test_densification.cpp` | Density Controller | CPU densification and Vulkan densification smoke |
| `test_relocation.cpp` | MCMC Densification | AAA-Gaussians relocation opacity/scale redistribution |
| `test_mcmc_densification.cpp` | MCMC Densification | Relocate/add/densify behavior and deterministic sampling |
| `test_mcmc_cuda_golden.cpp` | MCMC Densification | Replay against CUDA-generated MCMC golden fixtures |
| `test_train_types.cpp` | Training Types | Trainer data structures and config defaults |
| `test_training_gap_sp6.cpp` | Training Gaps | SP-6 gap closure utilities |
| `test_compare.cpp` | Compare Utilities | Numeric comparison helpers |

### GPU Tests (OpenCL legacy)
| File | Subsystem | What It Tests |
|------|-----------|---------------|
| `test_preprocessor_gpu.cpp` | Preprocessor GPU | GPU preprocessing matches CPU reference |
| `test_rasterizer_gpu.cpp` | Rasterizer GPU | GPU rasterization matches CPU reference |
| `test_rasterizer_backward_gpu.cpp` | Backward GPU | GPU backward matches CPU reference |
| `test_preprocessor_backward_gpu.cpp` | Backward GPU | GPU preprocessor backward matches CPU |
| `test_trainer_gpu.cpp` | Trainer GPU | Full GPU training step |

### Vulkan Infrastructure / Pass Tests
| File | Subsystem | What It Tests |
|------|-----------|---------------|
| `test_vk_capabilities.cpp` | Vulkan | Device capabilities |
| `test_vk_compute.cpp` | Vulkan | Basic compute dispatch |
| `test_vk_device_selection.cpp` | Vulkan | Device selection |
| `test_vk_hello.cpp` | Vulkan | Minimal Vulkan smoke |
| `test_vk_pipeline_dispatch.cpp` | Vulkan | Pipeline dispatch plumbing |
| `test_prefix_scan_pass_vk.cpp` | Vulkan Pass | Prefix scan pass |
| `test_scatter_pass_vk.cpp` | Vulkan Pass | Scatter pass |
| `test_radix_sort_pass_vk.cpp` | Vulkan Pass | Radix sort pass |
| `test_tile_range_pass_vk.cpp` | Vulkan Pass | Tile range construction |
| `test_preprocess_pass_vk.cpp` | Vulkan Pass | Forward preprocess pass, including eval_3D smoke |
| `test_rasterize_pass_vk.cpp` | Vulkan Pass | Forward rasterize pass |
| `test_preprocess_backward_pass_vk.cpp` | Vulkan Pass | Backward preprocess pass |
| `test_rasterize_backward_pass_vk.cpp` | Vulkan Pass | Backward rasterize pass |
| `test_sp7_cb_chain.cpp` | Vulkan Infra | Command-buffer chaining smoke |

### Vulkan Pipeline / Training Tests
| File | Subsystem | What It Tests |
|------|-----------|---------------|
| `test_forward_cache_vk.cpp` | Vulkan Pipeline | Forward cache integration |
| `test_forward_pipeline_vk.cpp` | Vulkan Pipeline | Full forward chain, including eval_3D fixture |
| `test_backward_pipeline_vk.cpp` | Vulkan Pipeline | Full backward chain |
| `test_preprocessor_backward_vulkan.cpp` | Vulkan Backward | Preprocessor backward parity |
| `test_rasterizer_backward_vulkan.cpp` | Vulkan Backward | Rasterizer backward parity |
| `test_tile_binner_vulkan.cpp` | Vulkan Tile Binner | Tile binning and eval_3D parity-mode keying |
| `test_sorter_vulkan.cpp` | Vulkan Sorter | Vulkan sorting |
| `test_vulkan_adam.cpp` | Vulkan Optimizer | GPU Adam update, moment resize, selective zeroing |
| `test_training_step_vk.cpp` | Vulkan Trainer | VulkanTrainer step, parity, eval_3D non-parity replay smoke/gradient parity |
| `test_mcmc_trainer_integration.cpp` | Vulkan Trainer | MCMC densification integration, Adam preservation, opacity reset |
| `test_vk_vs_cpu_render.cpp` | Vulkan Parity | CPU↔VK render parity |
| `test_vk_vs_py_reference.cpp` | Vulkan Parity | VK↔Python reference training parity |
| `test_vk_vs_cuda_basketball.cpp` | Vulkan Parity | VK↔CUDA basketball render/training parity |
| `test_vk_vs_cuda_first_loss.cpp` | Vulkan Parity | VK↔CUDA first-loss gate ladder |
| `test_vk_vs_cuda_10step.cpp` | Vulkan Parity | VK↔CUDA 10-step trajectory parity |
| `test_vk_vs_cuda_100step.cpp` | Vulkan Parity | VK↔CUDA 100-step trajectory parity |

### Integration / End-to-End Tests
| File | Subsystem | What It Tests |
|------|-----------|---------------|
| `test_renderer_e2e.cpp` | E2E Render | Full forward render pipeline on tiny fixture |
| `test_trainer_e2e.cpp` | E2E Train | Full training loop |
| `test_train_pipeline.cpp` | Train Pipeline | Pipeline orchestration, state transitions |
| `test_train_loop.cpp` | Train Loop | Loop termination and loss convergence |
| `test_e2e_basketball.cpp` | E2E Train | Basketball training smoke |

## Acceptance Criteria

- All CPU unit tests: PASS
- Vulkan tests: PASS on a Vulkan compute device; skip only when the fixture/device gate explicitly requires it
- OpenCL legacy GPU tests: PASS on an OpenCL device; skip if no device
- E2E render: RMSE/PSNR within the test-specific reference tolerance
- E2E train: loss decreases or matches the configured parity reference
- No test modified to weaken an assertion

## Test Data

- `test_data/tiny_3gaussians.ply` — 3-Gaussian scene for fast smoke tests
- `test_data/l1_verify/` — Pre-computed L1 loss ground truth for CPU validation
- `tests/golden/tiny/` — tiny-fixture Python/CUDA/Vulkan parity artifacts
- `tests/golden/mcmc_cuda/` — CUDA-generated MCMC densification replay fixtures
