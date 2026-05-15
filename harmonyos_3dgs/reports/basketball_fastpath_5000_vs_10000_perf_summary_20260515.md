# Basketball Vulkan Fastpath 5000 vs 10000 Performance Summary

日期：2026-05-15

## 1. 结论

本轮有效性能结果包含两组 Vulkan 多视角 GT 训练：

- 5000 step / `cap_max=30000`
- 10000 step / `cap_max=200000`

两组均从初始点云 `init_filter3d.ply` 开始训练，均使用 multi-view GT scope，并开启当前 fastpath：

- `GS3D_TRAIN_STAGE_TIMING=1`
- `GS3D_USE_FUCHSIA_SORT=1`
- `GS3D_TRAIN_GPU_GRAD_ADAM=1`
- `GS3D_TRAIN_GPU_RAW_ACTIVATE=1`

10000-step corrected detached benchmark 已完成：PID 不再运行，`exit_code=0`，`stage_timing.csv` 有 10000 行。

总体结论：

- transfer 优化目标已达成：`grad_upload_ms` 和 `backward_download_ms` 已消除，steady-state `raw_download_ms` 为 0。
- 5000-step fastpath 相比之前 Release+Fuchsia baseline 约提升 9–10%。
- 10000-step / `cap_max=200000` fastpath 相比之前 10000-step Release+Fuchsia baseline 约提升 24–34%。
- 当前剩余瓶颈不再是 host/device transfer，而是计算阶段，主要是 `backward_gpu_ms`、`forward_total_ms`，以及后期随 Gaussian 数量增长明显上升的 `preprocess_process_ms`、`raster_ms`、`sort_ms`、`adam_gpu_ms`。

## 2. 报告来源

### 5000-step fastpath

路径：

`harmonyos_3dgs/reports/basketball_vk_5000_cap30000_fastpath_20260515_095745`

关键配置：

- 初始点云：`init_filter3d.ply`
- 训练步数：5000
- `cap_max=30000`
- `densify_until_step=4999`
- multi-view GT：`--cameras` / `--gt_dir` / `--view_schedule` / `--require_all_gt 1`

### 10000-step corrected detached fastpath

路径：

`harmonyos_3dgs/reports/basketball_vk_10000_cap200000_fastpath_from_init_detached_20260515_103518`

关键配置：

- 初始点云：`init_filter3d.ply`
- 训练步数：10000
- `cap_max=200000`
- `densify_until_step=9999`
- multi-view GT：`--cameras` / `--gt_dir` / `--view_schedule` / `--require_all_gt 1`
- detached run，完成后 `exit_code=0`

无效结果说明：

`harmonyos_3dgs/reports/basketball_vk_10000_cap200000_fastpath_from_init_20260515_102650` 只产生 2283 行 timing，`exit_code=143`，是被 Claude 工具 10 分钟 timeout 终止的中间结果，不作为最终性能结论。

## 3. 总体性能对比

| Metric | 5000 step / cap30000 | 10000 step / cap200000 | Ratio |
|---|---:|---:|---:|
| Rows | 5000 | 10000 | 2.00x |
| Exit code | 0 | 0 | - |
| Elapsed | 315.79s | 848.32s | 2.69x |
| Final loss | 0.036112 | 0.025231 | lower |
| Max RSS | 890,912 KB | 1,721,196 KB | 1.93x |
| total_ms mean | 62.236 ms | 78.205 ms | 1.26x |
| total_ms p50 | 61.388 ms | 71.317 ms | 1.16x |
| total_ms p95 | 80.664 ms | 118.646 ms | 1.47x |
| total_ms p99 | 90.949 ms | 127.085 ms | 1.40x |
| total_ms max | 108.162 ms | 315.959 ms | 2.92x |

10000-step run 的 wall time 是 5000-step 的 2.69x，不是单纯 2x。原因是 10000-step 同时把 `cap_max` 从 30000 提高到 200000，后期 Gaussian 数量更大，导致每步计算成本上升。

## 4. 与旧 Release+Fuchsia baseline 对比

| Run | 旧 Release+Fuchsia | 新 fastpath | 提升 |
|---|---:|---:|---:|
| 5000 step / cap30000 | 344.41s elapsed, mean 68.740 ms/step | 315.79s, mean 62.236 ms/step | elapsed 约 1.09x；mean 约 1.10x |
| 10000 step / cap200000 | 1049.79s elapsed, mean 104.900 ms/step | 848.32s, mean 78.205 ms/step | elapsed 约 1.24x；mean 约 1.34x |

fastpath 在 10000-step / high-N 情况下收益更明显，因为原来的 transfer 和 raw activation 路径在高 Gaussian 数量下更容易成为额外负担。

## 5. 全程 stage timing 对比

| Stage | 5000 mean | 5000 p50 | 5000 p95 | 10000 mean | 10000 p50 | 10000 p95 | Mean ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| total_ms | 62.236 | 61.388 | 80.664 | 78.205 | 71.317 | 118.646 | 1.26x |
| forward_total_ms | 20.161 | 19.242 | 28.569 | 28.936 | 24.614 | 50.987 | 1.44x |
| backward_gpu_ms | 41.168 | 40.781 | 53.053 | 47.690 | 45.933 | 64.753 | 1.16x |
| preprocess_process_ms | 5.321 | 5.127 | 7.718 | 7.957 | 6.501 | 15.703 | 1.50x |
| raster_ms | 8.806 | 8.608 | 11.928 | 12.256 | 11.390 | 18.847 | 1.39x |
| sort_ms | 2.317 | 2.031 | 4.258 | 4.458 | 3.519 | 9.492 | 1.92x |
| bin_ms | 1.846 | 1.406 | 3.594 | 2.225 | 1.828 | 3.817 | 1.21x |
| loss_ms | 1.227 | 0.989 | 2.589 | 0.962 | 0.915 | 1.270 | 0.78x |
| adam_gpu_ms | 0.758 | 0.313 | 1.707 | 1.080 | 0.684 | 3.032 | 1.42x |

全程均值看，`backward_gpu_ms` 仍是最大单项；但 10000-step 额外增长主要来自 forward-side：`preprocess_process_ms`、`raster_ms`、`sort_ms` 和 `adam_gpu_ms` 都随高 N 增长。

## 6. 后期 stage timing 对比

5000-step 使用 `step >= 4500` 作为后期窗口；10000-step 使用 `step >= 9900` 作为最终后期窗口。

| Stage | 5000 late mean | 5000 late p50 | 10000 late mean | 10000 late p50 | Mean ratio |
|---|---:|---:|---:|---:|---:|
| total_ms | 74.759 | 73.171 | 120.463 | 117.789 | 1.61x |
| forward_total_ms | 26.121 | 26.042 | 52.246 | 51.201 | 2.00x |
| backward_gpu_ms | 46.887 | 45.195 | 63.637 | 64.145 | 1.36x |
| preprocess_process_ms | 6.521 | 6.382 | 15.664 | 15.842 | 2.40x |
| raster_ms | 10.930 | 10.809 | 18.874 | 18.647 | 1.73x |
| sort_ms | 3.647 | 3.529 | 9.763 | 9.490 | 2.68x |
| bin_ms | 2.583 | 2.484 | 3.981 | 3.651 | 1.54x |
| adam_gpu_ms | 1.480 | 1.660 | 3.056 | 3.039 | 2.06x |

后期窗口显示得更清楚：10000-step 后期每步约 120 ms，5000-step 后期约 75 ms。增长最明显的是：

1. `sort_ms`：2.68x
2. `preprocess_process_ms`：2.40x
3. `adam_gpu_ms`：2.06x
4. `forward_total_ms`：2.00x
5. `raster_ms`：1.73x

这说明高 N 后当前瓶颈已经从 transfer 转向前向处理、排序、raster 和 optimizer 的规模化成本。

## 7. Transfer fastpath 结果

| Transfer metric | 5000 result | 10000 result | 状态 |
|---|---:|---:|---|
| backward_download_ms | mean 0.000, p50 0.000, max 0.000 | mean ~0.0000001, p50 0.000, max 0.001 | 已消除 |
| grad_upload_ms | mean 0.000, p50 0.000, max 0.000 | mean 0.000, p50 0.000, max 0.000 | 已消除 |
| raw_download_ms | mean 0.007, p50 0.000, p95 0.000, max 4.176 | mean 0.061, p50 0.000, p95 0.000, max 43.745 | steady-state 已消除；边界 spike 保留 |

`raw_download_ms` 仍有少数 spike，尤其 10000-step max 为 43.745 ms。这些 spike 来自 densification/materialization 边界，不是 steady-state 每步下载。p50 和 p95 均为 0，说明常规训练步已经不再走 raw download。

## 8. 当前瓶颈判断

当前 fastpath 后，性能瓶颈分类为 compute-bound / scale-bound，而不是 host-transfer-bound。

证据：

- `grad_upload_ms` 全程为 0。
- `backward_download_ms` 全程近似为 0。
- `raw_download_ms` p50/p95 为 0，只在边界 materialization 有 spike。
- 10000-step 后期 `forward_total_ms` 达到 52.246 ms mean，是 5000-step 后期的 2.00x。
- 10000-step 后期 `sort_ms` 达到 9.763 ms mean，是 5000-step 后期的 2.68x。
- 10000-step 后期 `preprocess_process_ms` 达到 15.664 ms mean，是 5000-step 后期的 2.40x。

## 9. 后续优化方向

建议下一轮不要继续盯 `raw_download` / `grad_upload`，因为 steady-state transfer 目标已经完成。更有价值的方向是：

1. sort scaling：分析 high-N 下 Fuchsia sort 的 key/value 数量、tile occupancy 和 sort pass 分布。
2. preprocess scaling：拆分 `preprocess_process_ms`，确认高 N 后主要成本在投影、SH、covariance、filter_3D 还是 buffer lifecycle。
3. raster scaling：针对后期 heavy tiles 做 work concentration / tail 分析，判断是否需要算法级 culling 或 tile-level load balancing。
4. backward_gpu：继续拆分 backward raster 与 preprocess backward，找出当前最大 kernel 或同步点。
5. Adam scaling：`adam_gpu_ms` 后期约 3 ms，已不是最大瓶颈，但随 N 基本线性增长，可作为后续次级优化点。

## 10. 最终结论

本轮 fastpath 达成了原目标：`raw_download` 和 `grad_upload` 的 steady-state transfer 开销已消除，10000-step corrected detached benchmark 也已 clean 完成。当前 10000-step 比 5000-step 慢 2.69x 的主要原因不是 transfer，而是 `cap_max=200000` 带来的后期高 Gaussian 数量，使 preprocess、sort、raster、backward 和 Adam 的每步计算成本显著上升。
