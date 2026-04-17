# SP-0 Interface Inventory

> Living document. Update when interfaces are added or their signature changes.
> Source of truth for what exists vs what is draft.

**Lock this inventory in `dev_notes/interfaces_sp0.md` during Task 1 to prevent drift.** Any implementation that calls a "draft" interface before it is created MUST fail first.

### Existing interfaces (real, already in the codebase / dependencies)

| Interface | Location | Usage |
|-----------|----------|-------|
| `AAA-Gaussians/train.py -m / --model_path` | `AAA-Gaussians/arguments/__init__.py:52-59` | Output directory for baseline runs |
| `AAA-Gaussians/train.py -s` | same | Source dataset path |
| `_C.rasterize_gaussians(...)` → `(int, color, radii, geomBuffer, binningBuffer, imgBuffer)` | `rasterize_points.cu:44-142` | Forward pass; buffers returned as raw bytes |
| `_C.rasterize_gaussians_backward(...)` → 8-tuple | `rasterize_points.cu:144-236` | Backward; d_means2D/colors/opacity/means3D/cov3D/sh/scales/rotations |
| `GaussianRasterizationSettings.debug: bool` | `diff_gaussian_rasterization/__init__.py:254-269` | Snapshot-on-exception flag (DO NOT reuse) |
| `GaussianModel.capture()` / `.restore()` | `scene/gaussian_model.py` | Full model state serialization |
| `torch.Tensor.retain_grad()` | PyTorch built-in | Required to read gradient of non-leaf tensors |
| `CudaRasterizer::GeometryState::fromChunk(chunk, P, requires_cov3D_inv, requires_gauss2screen)` | `rasterizer_impl.cu:175` | Parse geomBuffer byte layout |
| `CudaRasterizer::BinningState::fromChunk(chunk, P)` | `rasterizer_impl.cu:206` | Parse binningBuffer |
| `CudaRasterizer::ImageState::fromChunk(chunk, N)` | `rasterizer_impl.cu:197` | Parse imgBuffer |
| `cub::DeviceRadixSort::SortPairs` | CUDA toolkit | Used for (key, value) sort inside rasterizer |
| conda env `aaa-gs` | `/home/robota/miniconda3/envs/aaa-gs` | Runtime for Python dumper and baseline |
| GoogleTest fetched v1.14.0 | `harmonyos_3dgs/CMakeLists.txt` | C++ unit test framework |

### Draft interfaces (to be created in this plan)

| Interface | Planned location | Created in Task |
|-----------|------------------|-----------------|
| `materialize_dump(geomBuffer, binningBuffer, imgBuffer, P, R, num_tiles, H, W, requires_cov3D_inv, requires_gauss2screen)` → `dict[str, torch.Tensor]` | `rasterize_points.{h,cu}` | Task 10, 11 |
| `_C.materialize_dump(...)` Python binding | `ext.cpp` | Task 12 |
| New optional param `dump_mode: bool` on `RasterizeGaussiansBackwardCUDA` | `rasterize_points.cu:170` | Task 13 |
| `tools/dump_tool.py` CLI (`--fixture`, `--output`, `--ladder`, `--dump_steps`) | `tools/dump_tool.py` | Task 17-22 |
| `tests/golden/npy_reader.h` (header-only, 5 dtypes) | `harmonyos_3dgs/tests/golden/` | Task 4-6 |
| `tests/golden/compare.h` (dual-threshold compare) | same | Task 7 |
| `tests/golden/manifest.h` (manifest.json reader) | same | Task 8 |
| `manifest.json` schema (step/cam/artifacts) | described in Task 16 | Task 16 |
| `dev_notes/dependencies.md` | `dev_notes/dependencies.md` | Task 26 |
| `dev_notes/interfaces_sp0.md` | `dev_notes/interfaces_sp0.md` | Task 1 |

**Enforcement rule:** If any task references an interface not in either list above, plan is broken — fix the plan, do not invent interfaces on the fly.
