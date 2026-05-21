# Vulkan training 阶段耦合系统级分析

## 结论

当前 Vulkan training 阶段之间的耦合还不是最优。

已经做得比较好的部分：

- backward 主路径已经能把 rasterize backward 和 preprocess backward 录进同一个 command buffer。
- GPU Adam 已经把 6 个参数组串进一个 command buffer，避免了早期 6 次 submit/wait。
- 部分 forward/backward cache 已有 GPU handle sideband，例如 `ForwardCache::*_gpu`、`PreprocessOutput::*_gpu`、`BinningOutput::*_gpu`。

仍然不理想的部分：

- forward prefix 仍然由多个同步阶段拼接：activate → preprocess → bin → sort → raster → loss。
- 多个阶段仍然发布 CPU host arrays 作为兼容接口，即使后续阶段已经有 GPU handle 可用。
- loss 每步上传 GT target，并下载 partial reductions 到 CPU 求和。
- noise 仍然是 CPU 全量遍历 + positions upload。
- densification 仍然需要 CPU-side raw params / gradient side effects，导致训练主循环必须保留 host 可见边界。

## CUDA 参考的阶段耦合方式

参考目录：`/home/robota/h00813233/Graph/AAA-Gaussians`

CUDA 参考并不是把所有训练逻辑融合成一个巨型 kernel，而是采用以下系统形态：

1. **render forward/backward 是主要 fused CUDA extension 边界**
   - `gaussian_renderer/__init__.py::render()` 构造 raster settings 后调用 `GaussianRasterizer`。
   - `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py::_RasterizeGaussians.forward/backward` 调用 `_C.rasterize_gaussians` 和 `_C.rasterize_gaussians_backward`。
   - Python 只负责 orchestration，真正的 render/backward cache 留在 CUDA extension/autograd 边界内。

2. **GT image/camera tensors 常驻 GPU**
   - `scene/cameras.py::Camera.__init__` 把 `original_image` 放在 `data_device`。
   - training loop 中 loss 直接消费 GPU tensor，而不是每步把 GT 从 CPU 上传。

3. **loss 组合是 Python 层，但输入输出仍是 GPU tensor**
   - `train.py` 计算 L1、fused SSIM、opacity/scale regularization，然后 `loss.backward()`。
   - loss scalar 进入 autograd，触发 rasterizer custom backward。

4. **optimizer/densification/noise 是阶段边界，但主要操作 torch CUDA tensor**
   - `optimizer.step()`、`relocate_gs()`、`add_new_gs()`、position noise 位于 Python orchestration 层。
   - 这些不是完全 fused，但状态仍以 GPU tensor 为主，不需要 Vulkan 当前这种频繁显式 host materialization。

对 Vulkan 的启发：不是必须把所有阶段合成一个 kernel，而是应把阶段边界提升为 GPU-resident contract，CPU 只在日志、诊断、densify plan 或最终输出时介入。

## 现有 timing 证据

### 短版真实 VK/CUDA 对比

报告：`harmonyos_3dgs/reports/vk_cuda_perf_gap_20260518_103334/vk_cuda_perf_gap_summary.md`

| steps | CUDA train | VK train | VK/CUDA train | VK/CUDA total |
|---:|---:|---:|---:|---:|
| 100 | 3.90s | 8.71s | 2.23x | 2.39x |
| 500 | 21.91s | 48.51s | 2.21x | 2.32x |
| 1000 | 51.56s | 99.17s | 1.92x | 1.95x |

### AAA 30000-step 长跑

报告：`harmonyos_3dgs/reports/aaa_defaults_30000_cap400000_render_all_20260516_151758/basketball_vk_30000_cap400000_aaa_defaults/summary.md`

全局均值：

| stage | mean ms |
|---|---:|
| total | 249.326 |
| forward_total | 70.400 |
| backward_gpu | 133.394 |
| preprocess_process | 13.705 |
| bin | 4.078 |
| sort | 10.085 |
| raster | 27.799 |
| loss | 11.123 |
| noise | 38.161 |
| adam_gpu | 5.010 |

`step_ge_9000` 均值：

| stage | mean ms |
|---|---:|
| total | 296.890 |
| forward_total | 82.505 |
| backward_gpu | 152.398 |
| noise | 52.031 |
| loss | 11.199 |

这说明阶段耦合优化和 kernel 优化同等重要：即使 backward kernel 是最大单项，forward residency、loss boundary、noise boundary 也分别有十几到几十毫秒的收益空间。

## 可并行优化点

### Track A：Forward GPU-resident stage context

目标：把 preprocess → bin → sort → raster 的 handoff 从 CPU-host arrays 改成 GPU buffer contract。

当前问题：

- `PreprocessOutput` 同时发布 host arrays 和 GPU handles。
- `BinningOutput` 同时发布 host keys/values/ranges 和 GPU handles。
- `SorterVulkan::sort()` 仍然会把 keys/values 从 host 上传到新的 sort buffer，再把 sorted keys/values/ranges 下载回来。

关键位置：

- `harmonyos_3dgs/include/types.h`：`PreprocessOutput`、`BinningOutput`、`ForwardCache`
- `harmonyos_3dgs/src/vulkan/preprocessor_vulkan.cpp:262-287`：下载 means/depth/rgb/radii/tiles/conic 等 CPU outputs
- `harmonyos_3dgs/src/vulkan/tile_binner_vulkan.cpp:169-193`：scan 后计算 `R`
- `harmonyos_3dgs/src/vulkan/tile_binner_vulkan.cpp:350-363`：下载 unsorted keys/values
- `harmonyos_3dgs/src/vulkan/sorter_vulkan.cpp:174-237`：重新分配 sort buffers、上传 unsorted keys/values、下载 sorted outputs/ranges

建议实验：

1. 新增 GPU-only forward context，默认训练路径只传 GPU handles。
2. sort 直接消费 `bin_keyvals_buf_` / `keyvals_unsorted_gpu`，避免 R-sized CPU download/upload。
3. tile ranges 用 GPU zero-fill / persistent buffer，避免 CPU zeros vector upload。
4. CPU arrays 只在 parity/debug/capture 模式 materialize。

并行性：与 backward kernel、loss、noise 基本独立，可单独推进。

### Track B：Forward prefix record-mode chaining

目标：减少 forward 阶段多个 `dispatch_sync()` / `submitAndWait()` 边界。

当前问题：

- backward/Adam 已经有 command-buffer chaining。
- forward prefix 仍然更像 Layer-1 sync pipeline：preprocess sync、bin scan/scatter sync、sort sync、raster sync、loss sync。

关键位置：

- `harmonyos_3dgs/src/vulkan_trainer.cpp:758-988`：`VulkanTrainer::run_forward_and_loss()`
- `harmonyos_3dgs/src/vulkan/preprocess_pass.cpp`：`PreprocessPass::dispatch_sync()`
- `harmonyos_3dgs/src/vulkan/sorter_vulkan.cpp`：sort sync / Fuchsia sort path
- `harmonyos_3dgs/src/vulkan/rasterizer_vulkan.cpp`：forward raster dispatch/download path
- `harmonyos_3dgs/src/vulkan/vk_context.cpp:306-311`：`VulkanContext::submitAndWait()` hard barrier

建议实验：

1. 给 preprocess/bin/sort/raster 提供训练专用 `record()` path。
2. 在 `VulkanTrainer::run_forward_and_loss()` 中录入同一个 command buffer。
3. 只在确实需要 CPU scalar `R`、loss scalar、debug capture 时同步。

并行性：可以与 Track A 协同，但也可先对已有 GPU handle path 做局部 record-mode 实验。

### Track C：GPU-only ForwardCache / Backward cache contract

目标：让 backward/replay 消费 GPU-resident cache，避免每步下载 forward cache 再上传/重建 backward inputs。

当前问题：

- `ForwardCache` 仍保存大量 host arrays：`T_final`、`n_contrib`、`cov2D`、`cov3D`、`p_view`、`p_hom_w` 等。
- 部分路径仍调用 `PreprocessorVulkan::download_cache()`。
- backward bindings 已经能接受一些 GPU cache，但 contract 没完全收敛为 GPU-only。

关键位置：

- `harmonyos_3dgs/include/types.h`：`ForwardCache`
- `harmonyos_3dgs/include/vulkan/preprocessor_vulkan.h`：`download_cache()`、persistent buffer getters
- `harmonyos_3dgs/include/vulkan/preprocessor_backward_vulkan.h`：`PreprocessBackwardVulkan::backward_record_into(...)`
- `harmonyos_3dgs/include/vulkan/backward_bindings.h`：backward binding slots
- `harmonyos_3dgs/src/vulkan/preprocessor_vulkan.cpp:470-488`：cache download
- `harmonyos_3dgs/src/vulkan/preprocessor_backward_vulkan.cpp:363-522`：backward cache consumption

建议实验：

1. 定义训练路径的 `ForwardCacheGpuView`，只含 `VkBuffer` 和 counts。
2. backward/preprocess-backward 优先只接受 GPU view。
3. host `ForwardCache` 改为 diagnostic/capture path。

并行性：和 Track A/B 相关，但可由 backward-cache 专人独立梳理接口。

### Track D：Loss residency + GPU final reduction

目标：消除每步 GT upload 和 partial loss download。

当前问题：

- `run_gpu_l1_loss()` / `run_gpu_dssim_loss()` 每步从 `const float* target` 上传 GT image。
- loss pass 生成 partials 后下载到 CPU，由 CPU 求和。
- CUDA 参考中 GT image 是 GPU tensor，loss scalar 留在 autograd/GPU 体系中。

关键位置：

- `harmonyos_3dgs/src/vulkan_trainer.cpp:581-616`：`VulkanTrainer::run_gpu_l1_loss()`
- `harmonyos_3dgs/src/vulkan_trainer.cpp:618-661`：`VulkanTrainer::run_gpu_dssim_loss()`
- `harmonyos_3dgs/src/vulkan/shaders/l1_loss.comp`
- `harmonyos_3dgs/src/vulkan/shaders/dssim_loss.comp`

建议实验：

1. 训练初始化时为每个 view 预上传 GT image buffer。
2. step 中按 view index 选择 GT buffer，不再上传 target。
3. 增加 GPU final reduction pass，CPU 只在 `log_every` 或 final step 下载 scalar。

并行性：完全独立于 forward/backward math，适合单独优化。

### Track E：GPU position noise pass

目标：消除 AAA 默认下 late-stage `noise_ms` 的 CPU 全量遍历。

当前问题：

- 当前 CPU 遍历所有 Gaussians，根据 activated opacity/scale/rotation 构造 covariance-scaled noise。
- 改动 raw positions 后再上传全部 positions。
- AAA 长跑中 `noise_ms` 全局均值 `38.161ms`，`step_ge_9000` 均值 `52.031ms`。

关键位置：

- `harmonyos_3dgs/src/vulkan_trainer.cpp:1290-1304`：raw materialization / noise stage
- `harmonyos_3dgs/src/vulkan_trainer.cpp:1422-1465`：`VulkanTrainer::inject_position_noise()`
- `harmonyos_3dgs/include/vulkan_trainer.h:198`：noise stage declaration

建议实验：

1. 新增 `PositionNoisePass` compute shader。
2. 输入 raw positions、activated scales/rotations/opacities、step seed、pos_lr。
3. 只更新 `opacity_factor >= 1e-6` 的 Gaussian。
4. 若 CPU RNG bitwise parity 难以保持，先做 gated experimental path，并用 loss/PSNR/N/R 稳定性验证。

并行性：和 forward/backward/loss 解耦，收益空间大。

### Track F：Densification side-effect isolation

目标：把 densification 的 CPU side-effect 从每步主路径中隔离出来。

当前问题：

- MCMC densification 仍消费 CPU `OwnedRawParams` 并重写 raw arrays。
- `grad_means2D_accum` / relocation / add / Adam state surgery 使训练主循环必须保留 host materialization 入口。
- 这不一定每步发生，但它约束了整体接口设计，使前面的 GPU-only contract 不够干净。

关键位置：

- `harmonyos_3dgs/src/vulkan_trainer.cpp:1317-1370`：densify path
- `harmonyos_3dgs/src/densification.cpp`：`densify_and_prune(...)`
- `harmonyos_3dgs/include/train_types.h`：`RawGaussianParams` / `OwnedRawParams`

建议实验：

1. 先不把整个 densify 搬 GPU，而是生成 compact densify plan。
2. GPU 端计算/压缩 dead mask、source indices、add count；CPU 只执行低频 topology surgery。
3. densify 不发生的普通 step 保证完全走 GPU-resident path。

并行性：低频 topology track，可独立于 hot steady-state optimization 推进。

## 总体优先级

1. **Track A + B：Forward GPU-resident + record-mode chaining**
   - 这是当前 Vulkan 与 CUDA 系统形态差异最大的地方。
   - 能减少 preprocess/bin/sort/raster 之间的 CPU contract 和 submit/wait 边界。

2. **Track E：GPU position noise**
   - AAA 默认下明确有 `38–52ms` 级别热点。
   - 和主数学路径解耦，适合并行快速出结果。

3. **Track D：Loss residency + GPU final reduction**
   - 稳定 `~11ms` 收益，风险低。
   - 也更接近 CUDA 的 resident GT image 形态。

4. **Track C：GPU-only cache contract**
   - 是 backward/forward 长期架构清理项。
   - 能减少未来优化时的兼容分支和 host download。

5. **Track F：Densification side-effect isolation**
   - 低频但架构影响大。
   - 推荐在 steady-state GPU-only path 稳定后推进。

## 最小验证矩阵

每条优化都应至少跑：

1. `VkVsCudaFirstLoss` gates，确认 first-step parity 不退化。
2. 100/500/1000-step `vk_cuda_perf_gap` 短版，对比 VK/CUDA ratio。
3. AAA 5000-step basketball smoke，检查 `N/R/final_loss` 没有异常漂移。
4. 对 Track A/B/C，额外检查 sorted IDs/tile ranges/replay-order gate。
5. 对 Track D，额外检查 DSSIM scalar 和 `dL_dpixels` 与当前 GPU DSSIM path 一致。
6. 对 Track E，额外记录 `noise_ms`、`R/N` 和 final PSNR，避免噪声统计变化造成训练轨迹异常。
