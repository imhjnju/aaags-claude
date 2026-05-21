# Vulkan/CUDA 性能差距优化点分析

## 基线证据

最近的短版真实 VK/CUDA 对比位于：

- `harmonyos_3dgs/reports/vk_cuda_perf_gap_20260518_103334/vk_cuda_perf_gap_summary.md`
- 配置：basketball，`training_preset=aaa`，`max_views=8`，`steps=100/500/1000`
- 结论：Vulkan train 约为 CUDA 的 `1.92x–2.23x`，total train+render 约为 `1.95x–2.39x`

长跑 AAA 全特性 30000-step 证据位于：

- `harmonyos_3dgs/reports/aaa_defaults_30000_cap400000_render_all_20260516_151758/basketball_vk_30000_cap400000_aaa_defaults/summary.md`
- `all` 平均：`total_ms=249.326`，`backward_gpu_ms=133.394`，`forward_total_ms=70.400`，`noise_ms=38.161`，`loss_ms=11.123`
- `step_ge_9000` 平均：`backward_gpu_ms=152.398`，`forward_total_ms=82.505`，`noise_ms=52.031`

## 可以并行推进的优化点

| 优化点 | 当前证据 | 主要文件位置 | 并行性 |
|---|---|---|---|
| 1. 优化 eval_3D backward replay kernel | `backward_gpu_ms` 是最大单项热点：全局均值 `133.394ms`，`step_ge_9000` 均值 `152.398ms` | `harmonyos_3dgs/src/vulkan/shaders/rasterize_backward_eval3d_replay.comp:65-183`; `harmonyos_3dgs/src/vulkan/rasterize_backward_eval3d_pass.cpp:63-85`; `harmonyos_3dgs/src/vulkan/rasterizer_backward_vulkan.cpp:277-302` | 独立 shader/dispatch 路径，可由 backward 专人优化 |
| 2. 减少 forward preprocess/bin/sort 的 host-device 往返 | `forward_total_ms` 全局均值 `70.400ms`，其中 preprocess/sort/raster/bin 分别约 `13.705/10.085/27.799/4.078ms`；R 变大时 sort/raster 放大 | `harmonyos_3dgs/src/vulkan/preprocessor_vulkan.cpp:69-160`, `262-287`; `harmonyos_3dgs/src/vulkan/tile_binner_vulkan.cpp:169-193`, `350-363`; `harmonyos_3dgs/src/vulkan/sorter_vulkan.cpp:174-237` | 与 backward kernel 独立，可由 forward/residency 专人优化 |
| 3. 把 CPU position noise 搬到 GPU 或减少 CPU 遍历/上传 | AAA 默认下 `noise_ms` 全局均值 `38.161ms`，`step_ge_9000` 均值 `52.031ms`，是除 backward/forward 外最大热点 | `harmonyos_3dgs/src/vulkan_trainer.cpp:1290-1304`; `harmonyos_3dgs/src/vulkan_trainer.cpp:1422-1465`; `harmonyos_3dgs/include/vulkan_trainer.h:198` | 与 forward/backward shader 独立，可单独实现 GPU noise pass |
| 4. 降低 DSSIM/L1 loss 的 target upload 和 partial download 开销 | `loss_ms` 稳定约 `11.1ms`；当前每步上传 target、dispatch 后下载 partials CPU 求和 | `harmonyos_3dgs/src/vulkan_trainer.cpp:581-616`; `harmonyos_3dgs/src/vulkan_trainer.cpp:618-661`; `harmonyos_3dgs/src/vulkan/shaders/l1_loss.comp`; `harmonyos_3dgs/src/vulkan/shaders/dssim_loss.comp` | 与 noise/backward 独立，可做 resident GT + GPU final reduction |
| 5. 合并/减少 Adam + regularization dispatch barrier | `adam_gpu_ms` 全局均值 `5.010ms`，高步数均值 `6.776ms`；当前 6 个 Adam group 串行 record，中间多次 compute barrier | `harmonyos_3dgs/src/vulkan_trainer.cpp:1209-1268`; `harmonyos_3dgs/src/vulkan/adam_optimizer_vulkan.cpp`; `harmonyos_3dgs/src/vulkan/shaders/adam_update.comp` | 与 loss/noise/forward/backward 都可分离推进 |

## 推荐并行拆分

### Track A — Backward kernel

目标：降低最大热点 `backward_gpu_ms`。

入口：

- `harmonyos_3dgs/src/vulkan/shaders/rasterize_backward_eval3d_replay.comp:65-183`
- `harmonyos_3dgs/src/vulkan/rasterizer_backward_vulkan.cpp:277-302`

初始实验：

- 统计 replay 每像素 contributor 分布与 atomic 热点。
- 尝试减少 `atomicAddG2S`/`atomicAdd` 压力，例如按 tile/workgroup 局部聚合、分离 color/opacity 与 gauss2screen 梯度，或对低贡献 replay 提前退出。
- 验证门槛：`VkVsCudaFirstLoss`、篮球 cam0 PSNR、短版 `vk_cuda_perf_gap` 不能退化。

### Track B — Forward residency / sort-binning

目标：减少 preprocess/bin/sort 中的 host materialization 和 R-sized download/upload。

入口：

- `harmonyos_3dgs/src/vulkan/preprocessor_vulkan.cpp:262-287`
- `harmonyos_3dgs/src/vulkan/tile_binner_vulkan.cpp:360-363`
- `harmonyos_3dgs/src/vulkan/sorter_vulkan.cpp:174-237`

初始实验：

- 保持 `keys/values/tile_ranges` GPU-resident，避免 `bin_keys_buf_`/`bin_vals_buf_` 下载后又在 sorter 重新上传。
- 复用 sorter 临时 buffer，避免每步 `keys_a/vals_a/keys_b/vals_b/hist/ranges` 重新分配。
- 用 GPU zero-fill 或 persistent ranges 替代 CPU `zeros` vector 上传。
- 验证门槛：R、sort/raster 结果不变，`VkVsCudaFirstLoss` sorted ids/ranges gate 不退化。

### Track C — GPU noise

目标：消除 late-stage `noise_ms` 的 CPU 全量遍历和 positions upload。

入口：

- `harmonyos_3dgs/src/vulkan_trainer.cpp:1290-1304`
- `harmonyos_3dgs/src/vulkan_trainer.cpp:1422-1465`

初始实验：

- 新增 GPU noise pass，输入 activated scale/rotation/opacity、raw positions、step seed、pos_lr。
- 只对 `opacity_factor >= 1e-6` 的 Gaussian 更新位置。
- 保持 RNG 语义可测；如果完全复现 CPU RNG 成本过高，先做 parity-gated 实验分支，不直接替换默认。
- 验证门槛：短步 loss 有界、最终 N/R 不异常、篮球短版 PSNR 不退化。

### Track D — Loss residency/reduction

目标：降低稳定的 `loss_ms≈11ms`。

入口：

- `harmonyos_3dgs/src/vulkan_trainer.cpp:647-656`
- `harmonyos_3dgs/src/vulkan/shaders/dssim_loss.comp`

初始实验：

- 预上传/缓存每个训练 view 的 GT image GPU buffer，避免每步 target upload。
- 把 partials 的最终 reduction 留在 GPU，CPU 只在 log/diagnostic 需要时下载 scalar。
- 验证门槛：DSSIM scalar 和 dL/dpixel 与当前 GPU DSSIM 结果一致。

## 优先级建议

1. **Track A backward kernel**：最大单项热点，收益上限最高。
2. **Track C GPU noise**：AAA 默认 late-stage 最大 CPU 热点，和 shader 主路径解耦，适合并行推进。
3. **Track B forward residency/sort-binning**：解决 R-sized host-device 往返，是 sort/raster 高 R 场景的基础优化。
4. **Track D loss residency/reduction**：稳定 11ms，收益明确但上限低于 backward/noise。
5. **Adam/barrier 合并**：收益中等，可作为后续清理项。
