# AAA-Gaussians CUDA/Vulkan 跨算子联合优化调研

Date: 2026-05-20

## 目标

本报告基于三个并行 read-only subagent 调研结果，梳理 AAA-Gaussians CUDA/PyTorch 实现与当前 HarmonyOS/Vulkan 实现之间的训练数据流差异，重点寻找：

- 算子之间可联合优化的边界；
- forward 与 backward 之间可复用的数据；
- 可以避免的 host/device 上传、下载和同步；
- CUDA 实现中已经存在、Vulkan 应优先借鉴的优化；
- 当前 Vulkan 达到 CUDA 90% 训练性能前最值得做的优化顺序。

## 当前性能基线

Vulkan 当前基线：

```text
/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/best-3dgs-optimized-20260512/harmonyos_3dgs/reports/regression_10000_cap200000_20260519_recorded_sortpairs/summary.json
```

CUDA 基线：

```text
/home/robota/h00813233/Graph/AAA-Gaussians-0508/basketball_cuda_stage_10000_cap200000_20260519_100943/cuda_stage_summary.json
```

| Scope | CUDA mean / wall | Vulkan mean / wall | Vulkan / CUDA | Vulkan throughput |
|---|---:|---:|---:|---:|
| Wall time | 770.04 s | 1213.23 s | 1.576x | 63.47% |
| `total_ms` | 67.628 ms | 114.559 ms | 1.694x | 59.03% |
| Forward/render scope | 11.722 ms | 28.340 ms | 2.418x | 41.36% |
| Loss | 1.413 ms | 1.503 ms | 1.064x | 94.00% |
| Backward | 54.315 ms | 84.480 ms | 1.555x | 64.29% |
| Optimizer + noise | 7.352 ms | 1.202 ms | 0.164x | 611.58% |
| Densify mean | 0.094 ms | 0.418 ms | 4.425x | 22.60% |

Vulkan forward 内部分解：

| Vulkan forward sub-stage | Mean | Share of `forward_total_ms` | Share of `total_ms` |
|---|---:|---:|---:|
| `preprocess_process_ms` | 7.970 ms | 28.12% | 6.96% |
| `bin_ms` | 2.306 ms | 8.14% | 2.01% |
| `sort_ms` | 2.127 ms | 7.51% | 1.86% |
| `raster_ms` | 13.231 ms | 46.69% | 11.55% |
| `backward_gpu_ms` | 84.480 ms | — | 73.75% |

结论：Vulkan 已经在 loss、optimizer/noise 上接近或超过 CUDA，但 forward 和 backward 仍是主差距；其中 backward 是绝对主瓶颈。

## CUDA 实现中的关键数据流优化

CUDA/PyTorch 训练主循环：

```text
/home/robota/h00813233/Graph/AAA-Gaussians-0508/train.py
symbol: training
```

### 1. 高斯参数长期 GPU resident

相关文件：

```text
/home/robota/h00813233/Graph/AAA-Gaussians-0508/scene/gaussian_model.py
```

关键点：

- `_xyz`
- `_features_dc`
- `_features_rest`
- `_scaling`
- `_rotation`
- `_opacity`

这些字段在 `GaussianModel.create_from_pcd` 中作为 CUDA `nn.Parameter` 创建，并由 PyTorch optimizer 长期管理在 GPU 上。

`GaussianModel.training_setup` 还创建：

- `xyz_gradient_accum`
- `denom`
- Adam optimizer state

这些同样是 CUDA tensor。训练 steady-state 中，参数、梯度、optimizer state 基本不需要来回 CPU/GPU 搬运。

### 2. 相机和训练图片默认 GPU resident

相关文件：

```text
/home/robota/h00813233/Graph/AAA-Gaussians-0508/arguments/__init__.py
/home/robota/h00813233/Graph/AAA-Gaussians-0508/scene/cameras.py
```

`ModelParams.data_device = "cuda"`，`Camera.__init__` 中将：

- `original_image`
- view/projection matrices
- inverse projection
- camera center

放到 CUDA 上。`train.py` 中仍可见 `viewpoint_cam.original_image.cuda()`，但默认配置下这通常已经是 GPU tensor，不构成实际大规模 transfer。

### 3. Forward 保存 GPU 中间状态供 backward 复用

Python autograd bridge：

```text
/home/robota/h00813233/Graph/AAA-Gaussians-0508/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py
symbol: _RasterizeGaussians.forward / backward
```

CUDA extension：

```text
/home/robota/h00813233/Graph/AAA-Gaussians-0508/submodules/diff-gaussian-rasterization/rasterize_points.cu
/home/robota/h00813233/Graph/AAA-Gaussians-0508/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
```

`_RasterizeGaussians.forward` 调用 `_C.rasterize_gaussians` 后保存：

- `colors_precomp`
- `means3D`
- `opacities`
- `scales`
- `rotations`
- `cov3Ds_precomp`
- `radii`
- `sh`
- `color`
- `geomBuffer`
- `binningBuffer`
- `imgBuffer`

其中最重要的是：

| CUDA saved buffer | 作用 |
|---|---|
| `geomBuffer` | forward preprocess 几何状态，用于 backward 还原 GeometryState |
| `binningBuffer` | tile/bin/sort 结果，用于 backward 遍历同一排序和 tile ranges |
| `imgBuffer` | forward raster 图像侧状态，如 alpha/contrib，用于 backward |
| `radii` | visibility / backward mask |
| `color` | rendered output，用于 backward 输入之一 |

CUDA 的核心优势是：这些 forward 内部状态作为 GPU byte tensors 保存给 autograd，backward 直接从 GPU buffer 重建状态，而不是重新计算或下载到 CPU 再上传。

### 4. CUDA forward 算子边界

`CudaRasterizer::Rasterizer::forward` 位于：

```text
/home/robota/h00813233/Graph/AAA-Gaussians-0508/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
```

主要阶段：

1. `FORWARD::preprocess`
   - 3D→2D transform
   - depth
   - covariance / conic opacity
   - radii
   - tile touched count
2. CUB `DeviceScan::InclusiveSum`
   - prefix sum tile touches
3. `cudaMemcpy` 读取最后 offset 得到 `num_rendered`
4. `FORWARD::duplicate`
   - 生成 `(tile/depth key, gaussian_id value)`
5. CUB `DeviceRadixSort::SortPairs`
6. `identifyTileRanges`
7. `FORWARD::render`

CUDA 也有一个同步点：为了知道 `num_rendered`，会把最后一个 prefix sum scalar 从 GPU 拷回 CPU。这是 CUDA 当前实现中的边界，不是 Vulkan 必须照搬的优化。

### 5. CUDA backward 直接复用 forward buffer

`_RasterizeGaussians.backward` 恢复 forward 保存的 tensors，调用 `_C.rasterize_gaussians_backward`，传入：

- 原始 Gaussian 参数
- `radii`
- rendered `color`
- `grad_out_color`
- `geomBuffer`
- `binningBuffer`
- `imgBuffer`
- `num_rendered`

CUDA backward 在：

```text
/home/robota/h00813233/Graph/AAA-Gaussians-0508/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
symbol: CudaRasterizer::Rasterizer::backward
```

过程：

1. 从 `geomBuffer` 重建 `GeometryState`；
2. 从 `binningBuffer` 重建 `BinningState`；
3. 从 `imgBuffer` 重建 `ImageState`；
4. 运行 `BACKWARD::render`；
5. 运行 `BACKWARD::preprocess`。

这就是 CUDA 最值得 Vulkan 借鉴的跨算子复用方式：forward 的 geometry/binning/image 状态全部留在 GPU 上，backward 不重新 materialize CPU cache。

### 6. CUDA host transfer 主要集中在非 steady-state 路径

CUDA steady-state 也不是完全无同步，但大多数 host transfer 发生在：

- `loss.item()` logging；
- TensorBoard image/scalar logging；
- GUI preview `.cpu().numpy()`；
- `GaussianModel.save_ply` 中 `.detach().cpu().numpy()`；
- checkpoint `torch.save`；
- debug exception snapshot；
- `applyDebugVisualization` full image copy。

这些不是每个训练 step 的核心数据流。

## 当前 Vulkan 实现的数据流

Vulkan training 入口：

```text
/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/best-3dgs-optimized-20260512/harmonyos_3dgs/src/vulkan_trainer.cpp
/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/best-3dgs-optimized-20260512/harmonyos_3dgs/include/vulkan_trainer.h
```

`VulkanTrainer::step(...)` 组织：

1. raw activation
2. preprocess
3. tile binning
4. sort
5. raster
6. loss
7. raster backward
8. preprocess backward
9. gradient handling
10. Adam
11. raw materialization policy
12. noise
13. densification / opacity reset

### 1. 参数与梯度 buffer

Vulkan 已经有 raw GPU parameter buffers：

- `raw_param_gpu_bufs_[6]`
  - positions
  - SH DC
  - SH rest
  - opacities
  - scales
  - rotations

对应梯度：

- `grad_positions_gpu_`
- `grad_sh_dc_gpu_`
- `grad_sh_rest_gpu_`
- `grad_opacities_gpu_`
- `grad_scales_gpu_`
- `grad_rotations_gpu_`

`GS3D_TRAIN_GPU_RAW_ACTIVATE=1` 时，`activate_params_gpu()` 通过 `RawActivationPass` 在 GPU 上生成 active params，避免 CPU activation/upload。

### 2. Preprocess 当前仍有较多 CPU materialization

相关文件：

```text
include/vulkan/preprocessor_vulkan.h
src/vulkan/preprocessor_vulkan.cpp
```

`PreprocessorVulkan::process_gpu_inputs(...)` 能消费 GPU active/raw buffers，并产生 GPU preprocess 输出：

- `means2D_gpu`
- `depths_gpu`
- `rgb_gpu`
- `radii_gpu`
- `tiles_touched_gpu`
- `conic_opacity_packed_gpu`
- `radius_f_gpu`
- eval3D buffers：
  - `gauss2screen_gpu`
  - `cov3D_inv_gpu`
  - `mean_offset_gpu`

但当前仍会下载不少 CPU mirror：

- `means2D`
- `depths`
- `rgb`
- `radii`
- `tiles_touched`
- `radius_f`
- `conic_opacity_packed`
- eval3D side buffers

此外，`VulkanTrainer::run_forward_and_loss(...)` 中仍调用：

```cpp
preprocessor_.download_cache(N_, out_cache, alloc_)
```

这会下载：

- `cov3D`
- `p_view`
- `p_hom_w`
- `cov2D`
- `cov2D_det`

即使 fused eval3D GPU backward 已经可能拥有所需 GPU inputs，这些 CPU cache 下载仍可能发生。这是最明确的跨算子冗余之一。

### 3. Tile binner 已有 host mirror gate

相关文件：

```text
include/vulkan/tile_binner_vulkan.h
src/vulkan/tile_binner_vulkan.cpp
```

`TileBinnerVulkan::bin(...)` 已经能使用 GPU preprocess handles，尤其是：

- `tiles_touched_gpu`
- `means2D_gpu`
- `depths_gpu`
- `radii_gpu`
- `radius_f_gpu`
- eval3D GPU handles

已有 persistent buffers：

- `bin_tt_buf_`
- `bin_po_buf_`
- `bin_ws_buf_`
- `bin_keys_buf_`
- `bin_vals_buf_`
- `bin_keyvals_buf_`
- persistent scatter UBO / dummy buffer

host mirror 由：

```cpp
set_host_mirror_enabled(...)
```

控制。GPU-resident training steady-state 下应尽量保持关闭，除非测试、调试或 CPU consumer 需要。

### 4. Sort 已保持 CUDA-compatible SortPairs

相关文件：

```text
include/vulkan/sorter_vulkan.h
src/vulkan/sorter_vulkan.cpp
```

当前正确性路径已经保持：

```text
key   = (tile_id << 32) | depth_bits
value = full uint32 gaussian_index
```

Fuchsia path 使用 3-dword canonical records：

1. canonical `(key,value)` pack；
2. Fuchsia radix sort；
3. extract canonical `keys_sorted`、`values_sorted`、`tile_ranges`。

注意：recent recorded SortPairs API 已存在：

- `prepare_record_fuchsia_sortpairs(...)`
- `record_fuchsia_sortpairs(...)`

但生产训练路径当前主要仍调用 Layer-1 `sort()`，不是完整 recorded forward command path。

### 5. Rasterizer / ForwardCache 是 Vulkan 复用的核心

相关文件：

```text
include/vulkan/rasterizer_vulkan.h
src/vulkan/rasterizer_vulkan.cpp
include/types.h
```

`ForwardCache` 包含 CPU 与 GPU replay/cache state：

CPU side：

- `T_final`
- `n_contrib`
- `replay_order_offsets`
- `replay_order_gids`

GPU side：

- `rendered_image_gpu`
- `T_final_gpu`
- `n_contrib_gpu`
- `dL_dpixels_gpu`
- `gauss2screen_gpu`
- `replay_order_offsets_gpu`
- `replay_order_gids_gpu`

flags：

- `gpu_resident_outputs`
- `retain_gpu_outputs`

最近 forward buffer reuse 已经将：

- image
- `T_final`
- `n_contrib`
- UBO
- eval3D UBO
- dummy buffer

做成 retained/persistent reuse。短跑 benchmark 显示：

```text
forward_total_ms: 16.434 -> 11.565 ms
raster_ms: 6.228 -> 2.371 ms
```

说明 Vulkan 这条路是有效的。

### 6. Loss 已接近 CUDA，但仍可减少 scalar/partial readback

GPU L1/DSSIM loss 可以消费：

- `rendered_image_gpu`

并产生：

- `dL_dpixels_gpu`

这允许 raster backward 直接消费 GPU loss gradient，避免 CPU loss-gradient upload。

当前 DSSIM target precompute/cache 已使 loss 与 CUDA 接近：

```text
CUDA loss:   1.413 ms
Vulkan loss: 1.503 ms
```

但 loss scalar / partial sum 下载仍可能为了 logging 每步发生或过于频繁。下一步可改为：

- 每步保留 GPU loss/gradient；
- 只在 `log_every` 或需要 report 时下载 scalar；
- 使用 async/staging readback，避免阻塞训练 command stream。

### 7. Backward 已有 GPU chaining，但仍是主瓶颈

相关文件：

```text
include/vulkan/rasterizer_backward_vulkan.h
src/vulkan/rasterizer_backward_vulkan.cpp
src/vulkan/preprocessor_backward_vulkan.cpp
src/vulkan/shaders/rasterize_backward_eval3d_replay_fused.comp
```

`RasterizerBackwardVulkan` 可读取 forward retained GPU buffers：

- tile ranges
- sorted values
- `T_final`
- `n_contrib`
- `dL_dpixels`
- `gauss2screen`
- RGB
- conic/opacity

并输出 GPU gradient buffers：

- `dL_dmeans2D_buf()`
- `dL_dconics_buf()`
- `dL_dopacity_buf()`
- `dL_dcolors_buf()`
- `dL_dgauss2screen_buf()`

`backward_record_fused_eval3d_replay_into(...)` 能直接写入 preprocess backward gradient buffers，并使用 GPU replay buffers。这是 Vulkan 当前最接近 CUDA `geomBuffer/binningBuffer/imgBuffer` autograd saved state 的路径。

但 benchmark 显示：

```text
CUDA backward:   54.315 ms
Vulkan backward: 84.480 ms
```

这仍是最大 gap。

### 8. Adam/noise 已优于 CUDA，但 densify 仍是同步岛

相关文件：

```text
src/vulkan/vulkan_adam.cpp
src/vulkan/position_noise_pass.cpp
src/vulkan_trainer.cpp
```

`GS3D_TRAIN_GPU_GRAD_ADAM=1` 时 Adam 直接更新 raw GPU parameter buffers，避免梯度 download/upload。

`PositionNoisePass` 支持 GPU-side position noise。

当前结果：

```text
CUDA optimizer_noise:   7.352 ms
Vulkan adam+noise:      1.202 ms
```

Vulkan 这里已经明显更快。

但 densification / resize 仍会触发 CPU materialization 与 buffer reallocation：

- `materialize_raw_params()`
- `reallocate_for_n(...)`
- Adam moment resize / zero / copy

这是低频同步岛，均值影响不大，但可能造成 tail spike。

## 可联合优化点排序

### P0：消除 fused GPU backward 路径中不必要的 preprocess CPU cache download

优先级：最高
预期影响：中高，可能 2–6 ms/step，并降低 forward/preprocess tail
置信度：高
难度：中
风险：中

证据：

- Vulkan `preprocess_process_ms` 均值 7.970 ms；
- 当前 `VulkanTrainer::run_forward_and_loss(...)` 仍调用 `preprocessor_.download_cache(N_, out_cache, alloc_)`；
- fused eval3D GPU backward 已经有 GPU inputs 和 replay buffers；
- CUDA backward 直接从 GPU `geomBuffer/binningBuffer/imgBuffer` 恢复，不下载 CPU cache。

涉及文件：

```text
src/vulkan_trainer.cpp
src/vulkan/preprocessor_vulkan.cpp
src/vulkan/preprocessor_backward_vulkan.cpp
include/types.h
include/vulkan/preprocessor_vulkan.h
```

建议实现：

1. 在 `ForwardCache` 或 trainer 内增加更明确的 cache materialization policy：
   - `CpuCacheRequired`
   - `GpuBackwardOnly`
   - `DebugCapture`
2. 当满足以下条件时跳过 `download_cache`：
   - eval3D GPU fused backward 可用；
   - loss gradient GPU resident；
   - raster backward / preprocess backward 均消费 GPU handles；
   - 当前 step 不需要 debug/parity CPU capture。
3. 测试确保 densify、opacity reset、非 eval3D fallback 不受影响。

验证：

- `test_training_step_vk.cpp` 增加 fused GPU backward 下 cache 不 materialize 的断言；
- `test_backward_pipeline_vk.cpp` gradient parity；
- 1000-step cap30000 benchmark 看 `preprocess_process_ms` / `forward_total_ms`；
- 10000-step cap200000 验证 final N/loss 和 stage timing。

### P1：训练 forward 使用完整 recorded command orchestration

优先级：高
预期影响：3–8 ms/step
置信度：中高
难度：中高
风险：中高

证据：

- Vulkan forward 28.340 ms vs CUDA render 11.722 ms；
- 已有 record API：preprocess/bin/sort/raster；
- 已新增 canonical Fuchsia recorded SortPairs；
- 但生产训练仍主要走 Layer-1 `sort()` / sync-style orchestration。

涉及文件：

```text
src/vulkan_trainer.cpp
src/vulkan/preprocessor_vulkan.cpp
src/vulkan/tile_binner_vulkan.cpp
src/vulkan/sorter_vulkan.cpp
src/vulkan/rasterizer_vulkan.cpp
```

建议实现：

1. 为训练 steady-state 建立 recorded forward graph：

```text
preprocess.record
→ barrier
→ binner.record
→ barrier
→ sorter.record_fuchsia_sortpairs
→ barrier
→ raster.record
```

2. capacity 稳定时复用 command/descriptor setup；
3. R 动态时用已有 `R_max` / capacity strategy，避免每 step 重新创建 buffer；
4. 保持 canonical SortPairs，不引入 packed-keyval correctness path。

验证：

- `ForwardPipeline.FullChain_TinyFixture`；
- eval3D forward pipeline；
- Vulkan vs CUDA first loss / 10-step / 100-step；
- stage timing 看 `forward_total_ms`、`sort_ms`、`raster_ms` tail。

### P2：backward kernel/replay 继续向 CUDA saved-buffer 模式靠拢

优先级：最高但更重
预期影响：10–25 ms/step
置信度：高确认是瓶颈，中等确认具体收益
难度：高
风险：高

证据：

- Vulkan backward 占 total 73.75%；
- Vulkan backward 84.480 ms，CUDA backward 54.315 ms；
- CUDA backward 直接重建 `GeometryState/BinningState/ImageState`；
- Vulkan 已有 `ForwardCache`，但仍可能存在 replay/order/cache materialization 与 shader效率差距。

涉及文件：

```text
src/vulkan/rasterizer_backward_vulkan.cpp
src/vulkan/preprocessor_backward_vulkan.cpp
src/vulkan/shaders/rasterize_backward_eval3d_replay_fused.comp
src/vulkan/shaders/rasterize_backward_eval3d.comp
src/vulkan/shaders/preprocess_backward_eval3d.comp
src/vulkan_trainer.cpp
```

建议工作：

1. profile `rasterize_backward_eval3d_replay_fused.comp`：
   - atomic hotspots；
   - memory bandwidth；
   - replay order distribution；
   - tile / gaussian imbalance；
2. 对照 CUDA `BACKWARD::render` 和 `BACKWARD::preprocess` 的 saved state 使用方式；
3. 确认当前 `GS3D_EVAL3D_FUSED_REPLAY_BWD=1` 在 `--proper_ewa 1` 下是否实际启用；如果 gate 关闭，先明确原因，不能误判优化无效；
4. 减少 backward 中重复读取/写回，优先把 gradient accumulation、conic/opacities/color gradients 的中间态保持在 device-local buffer。

验证：

- raster backward parity；
- preprocess backward parity；
- eval3D fused replay gradient parity；
- 1000-step fixed schedule；
- Nsight / Vulkan timestamp 分解。

### P3：loss scalar/partial readback 延迟到 logging interval

优先级：中
预期影响：1–3 ms/step 或降低同步 tail
置信度：中
难度：中
风险：低中

证据：

- Vulkan loss 已接近 CUDA：1.503 ms vs 1.413 ms；
- `dL_dpixels_gpu` 已可直接进入 backward；
- 每步 scalar loss 对训练更新不是必须，主要用于 logging/report。

涉及文件：

```text
src/vulkan_trainer.cpp
src/vulkan/l1_loss_pass.cpp
src/vulkan/dssim_loss_pass.cpp
```

建议：

- 每 step 保留 GPU-side loss reduction 和 `dL_dpixels_gpu`；
- scalar loss 只在 `log_every` 或用户请求时 readback；
- 如 API 需要返回 `float loss`，可以允许 stale/last-read loss 或引入 async staging ring。

风险：

- 现有 tests 可能假设每步 `step(...)` 返回实时 loss；
- parity tests 需要保留 synchronous readback 模式。

### P4：GPU-side R / replay count，减少小 scalar download 导致的同步

优先级：中
预期影响：1–3 ms/step、降低 tail
置信度：中
难度：中高
风险：中

证据：

- CUDA 也有 `num_rendered` scalar readback，但 Vulkan 可通过 over-allocation / indirect dispatch 改进；
- 当前 tile binner / replay finalization 仍可能读取 last offset/count；
- scalar readback 本身字节少，但会引入 pipeline sync。

涉及文件：

```text
src/vulkan/tile_binner_vulkan.cpp
src/vulkan/rasterizer_vulkan.cpp
src/vulkan/rasterizer_backward_vulkan.cpp
src/vulkan_trainer.cpp
```

建议：

- 用 GPU count buffer 作为后续 indirect dispatch 参数；
- 或固定按 capacity dispatch，由 shader 内部 guard；
- 对 replay offsets/counts 使用 GPU-resident sideband。

### P5：persistent arena / device-local scratch pool

优先级：中
预期影响：1–4 ms/step，主要降低 p95/p99
置信度：中
难度：中
风险：中

证据：

- forward buffer reuse 已明显提升；
- 仍有部分小 UBO、scratch、zero-fill 和 staging upload；
- CUDA/PyTorch allocator 会 amortize tensor allocation。

涉及文件：

```text
src/vulkan/vk_buffer.cpp
src/vulkan/preprocessor_vulkan.cpp
src/vulkan/tile_binner_vulkan.cpp
src/vulkan/sorter_vulkan.cpp
src/vulkan/rasterizer_vulkan.cpp
src/vulkan/vulkan_adam.cpp
```

建议：

- grow-only device-local scratch pool；
- persistent UBO ring；
- GPU zero-fill/clear kernels 替代 host zero upload；
- 避免 per-step `vkMapMemory/vkUnmapMemory`。

### P6：densify / Adam moment resize GPU 化或预分配 capacity

优先级：中低
预期影响：均值低，tail 可能明显
置信度：中
难度：高
风险：高

证据：

- densify all-step mean 只有 0.418 ms；
- 但 densify max 可达 197 ms；
- CPU materialization、moment resize 和 buffer reallocation 是同步岛。

涉及文件：

```text
src/vulkan_trainer.cpp
src/vulkan/vulkan_adam.cpp
```

建议：

- 预分配到 `cap_max`；
- active count/free-list 管理；
- Adam moments GPU-side extend/shrink/zero；
- densify 后避免全量 reupload。

## 推荐实施顺序

### Slice 1：CPU cache materialization gate

目标：不改变算法，仅减少不必要 CPU cache download。

预期：

- 降低 `preprocess_process_ms` 或 `preprocess_cache_refresh_ms`；
- 降低 `forward_total_ms`；
- 不影响 loss/N/parity。

这是最适合作为下一步的低风险联合优化。

### Slice 2：训练 forward recorded orchestration

目标：把已有 recorded APIs 真正接入生产训练 forward。

预期：

- 降低 `forward_total_ms`；
- 降低 command submit / descriptor update / barrier overhead；
- 让 recently added recorded canonical SortPairs 产生实际训练收益。

### Slice 3：backward focused profiling + shader/dataflow optimization

目标：将 Vulkan backward 从 84.480 ms 拉近 CUDA 54.315 ms。

预期：

- 这是达到 CUDA 90% 的关键；
- 需要更细 profiler，而不是继续盲目 micro-opt。

### Slice 4：loss scalar async/stale-read mode

目标：只在 logging interval 下载 loss scalar。

预期：

- 均值可能小幅改善；
- 降低同步边界；
- 需要保留 strict test/parity 模式。

### Slice 5：densify/moment tail 优化

目标：降低 densify spike，而非优先优化均值。

## 验证计划

每个 slice 都应遵循：

1. 正确性优先：

```bash
cmake --build harmonyos_3dgs/build -j$(nproc)
harmonyos_3dgs/build/gs3d_vk_tests --gtest_filter='ForwardCacheVK.*:ForwardPipeline.*:BackwardPipeline.*:RasterizerBackwardVulkan.*:PreprocessorBackwardVulkan.*:VulkanTrainer.*:SorterVulkan.*:FuchsiaRadixE2E.*'
ctest --test-dir harmonyos_3dgs/build --output-on-failure
```

2. 短跑 benchmark：

```text
1000 steps
cap_max=30000
same view schedule
same full-feature env
```

重点观察：

- `total_ms`
- `forward_total_ms`
- `preprocess_process_ms`
- `preprocess_cache_refresh_ms`
- `bin_ms`
- `sort_ms`
- `raster_ms`
- `loss_ms`
- `backward_gpu_ms`
- wall time
- final N/loss

3. 长跑回归：

```text
10000 steps
cap_max=200000
same full-feature env
```

对比基准：

```text
reports/regression_10000_cap200000_20260519_recorded_sortpairs/summary.json
/home/robota/h00813233/Graph/AAA-Gaussians-0508/basketball_cuda_stage_10000_cap200000_20260519_100943/cuda_stage_summary.json
```

## 最终判断

当前 Vulkan 的关键问题已经不是单个 sort 算子，也不是 loss 或 Adam/noise。下一阶段优化应转向跨算子数据流：

1. 像 CUDA 一样，把 forward 产生的 geometry/binning/image state 尽可能作为 GPU-resident saved state 直接交给 backward；
2. 不为 backward 已经能从 GPU 读取的数据再生成 CPU cache；
3. 将训练 forward 真实改成 recorded command orchestration；
4. 对 backward shader/replay 做针对性 profiling 和结构优化。

如果目标是整体达到 CUDA 90%，仅优化 forward 不够。按当前数据，Vulkan `backward_gpu_ms` 从 84.480 ms 降到接近 CUDA 54.315 ms，才是最大杠杆。