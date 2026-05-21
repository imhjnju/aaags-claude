# Vulkan 3DGS 10000-step 训练性能拆解与继续优化汇报材料

日期：2026-05-20
范围：basketball 76-view multi-view GT，10000 step，`cap_max=200000`；区分 fast/L1-style Release+Fuchsia、fastpath/L1、full AAA 1200s+ 三类配置口径。

## 1. 一句话结论

必须先把三条 10000-step 口径分清：`1049.79s` 是 Release+Fuchsia 的 fast/L1-style 基线；`848.32s` 是在同一 fast/L1-style 口径上打开 GPU-resident fastpath 后的结果；后续完整 AAA 特性（`--training_preset aaa`、GPU DSSIM、regularization、noise、LR/SH schedule 等）打开后，实测是 `1200s+` 量级，代表性结果为 `1213.23s`、`1282.81s`、`1306.95s`。

性能主因已经分三阶段迁移：

1. 早期最大瓶颈是 high-N sort，已通过 Fuchsia SortPairs 路径从秒级降到百毫秒级；
2. fast/L1-style 的 1049s 基线尾部，瓶颈是 `backward_gpu`、`forward/raster/preprocess` 与 host/device 边界传输共同主导；
3. fastpath/L1 消除 steady-state raw/grad transfer 后，848s 结果转向 compute/scale-bound，但这个结论不能直接外推到 full AAA；full AAA 的 1200s+ 结果还叠加了 DSSIM、regularization/noise 和更完整训练语义，当前主差距仍集中在 backward 与 forward/render。

因此下一轮优化不应把 848s 当作完整 AAA 结果，也不应继续盲目调局部 shader 常数，而应按“配置口径隔离 → stage 归因 → kernel/tile 分布 → 与 CUDA saved-buffer 数据流对齐”的路径推进。

## 2. 基线配置与可比性边界

### 2.1 三条必须分开的 10000-step 口径

| 口径 | 代表报告 | 关键配置 | Wall / elapsed | Mean step | 结论用途 |
|---|---|---|---:|---:|---|
| fast/L1-style Release+Fuchsia 基线 | `basketball_vk_10000_cap200000_release_fuchsia_20260514_193728` | `GS3D_USE_FUCHSIA_SORT=1`，未打开完整 AAA DSSIM/reg/noise preset | 1049.79s | 104.900 ms | 分析 Fuchsia 后、transfer fastpath 前的旧瓶颈 |
| fastpath/L1-style 结果 | `basketball_vk_10000_cap200000_fastpath_from_init_detached_20260515_103518` | 在 fast/L1-style 上加 `GS3D_TRAIN_GPU_GRAD_ADAM=1`、`GS3D_TRAIN_GPU_RAW_ACTIVATE=1` | 848.32s | 78.205 ms | 证明 raw/grad steady-state transfer 优化有效；不能当作 full AAA 性能 |
| full AAA feature 结果 | `regression_10000_cap200000_20260519_recorded_sortpairs` 等 | `--training_preset aaa`、`GS3D_TRAIN_GPU_DSSIM_LOSS=1`、GPU-resident fastpaths、regularization/noise/LR/SH schedule | 1213.23s（另有 1282.81s、1306.95s、1454.39s） | 114.559 ms（代表性最好 full-feature run） | 衡量完整 AAA 训练语义下的真实性能和 CUDA 差距 |

关键口径修正：`848.32s` 不是“完整 AAA 特性打开后”的训练时间；它只说明在 fast/L1-style 口径下，GPU Grad Adam / GPU Raw Activation 等 fastpath 能消除 steady-state transfer。完整 AAA 特性打开后的后续实验是 `1200s+`。

### 2.2 1049s fast/L1-style 基线

来源：

```text
harmonyos_3dgs/reports/basketball_vk_10000_cap200000_release_fuchsia_20260514_193728/summary.md
harmonyos_3dgs/reports/basketball_vk_10000_cap200000_release_fuchsia_20260514_193728/stage_timing.csv
```

配置：

| 项 | 值 |
|---|---|
| executable | `harmonyos_3dgs/build_release/gs3d_vk_train` |
| build | Release |
| dataset | basketball multi-view GT |
| cameras | 76 views |
| schedule | 10000-entry deterministic schedule, seed 42 |
| iterations | 10000 |
| final cap | `cap_max=200000` |
| final N | 200000 |
| render mode | `eval_3d=1`, `proper_ewa=1`, `parity_mode=1` |
| densify | MCMC enabled, `densify_from_step=600`, `densify_until_step=9999`, interval 100 |
| env | `GS3D_TRAIN_STAGE_TIMING=1`, `GS3D_USE_FUCHSIA_SORT=1` |
| loss/reg/noise | fast/L1 baseline style；非后续 AAA-default 全特性配置 |

结果：

| 指标 | 值 |
|---|---:|
| exit code | 0 |
| stage rows | 10000 |
| trainer internal total | 1049.00s |
| elapsed | 1049.79s |
| mean step | 104.900 ms/step |
| stage `total_ms` sum | 1047.53s |
| first loss | 0.550562 |
| final loss | 0.024686 |
| final vertex count | 200000 |
| max RSS | 1668.8 MiB |

### 2.3 完整 AAA 1200s+ 实验

代表性 full AAA 报告：

| 报告 | Wall time | Mean step | Final loss | Final N | 关键备注 |
|---|---:|---:|---:|---:|---|
| `regression_10000_cap200000_20260519_recorded_sortpairs` | 1213.23s | 114.559 ms | 0.007439 | 200000 | 当前较好的 full-feature 对 CUDA 对比口径；canonical SortPairs |
| `dssim_nohash_final_10000_cap200000_20260520_152435` | 1282.81s | 121.364 ms | 0.007485 | 200000 | 去掉 DSSIM target hash 后恢复到 1282s，但仍是 full AAA 1200s+ |
| `aaa_full_features_10000_cap200000_20260519_dssim_precompute` | 1306.95s | 125.621 ms | 0.007142 | 200000 | DSSIM target precompute/cache 后 full-feature benchmark |
| `master_reduce_cache_final_10000_cap200000_20260520_140029` | 1454.39s | 140.510 ms | 0.007003 | 200000 | loss stage regression 较重，`loss_ms` mean 23.524ms |

full AAA 代表性配置：

```bash
GS3D_TRAIN_STAGE_TIMING=1
GS3D_USE_FUCHSIA_SORT=1
GS3D_TRAIN_GPU_DSSIM_LOSS=1
GS3D_REUSE_FORWARD_OUTPUTS=1
GS3D_TRAIN_GPU_GRAD_ADAM=1
GS3D_TRAIN_GPU_RAW_ACTIVATE=1
GS3D_EVAL3D_FUSED_REPLAY_BWD=1

--iterations 10000
--training_preset aaa
--eval_3d 1
--parity_mode 1
--proper_ewa 1
--densify 1
--densify_from_step 600
--densify_until_step 9999
--densify_interval 100
--cap_max 200000
--opacity_reset_interval 0
```

注意：`GS3D_EVAL3D_FUSED_REPLAY_BWD=1` 虽然设置了，但在 `--proper_ewa 1` full AAA proper-EWA 路径下可能仍被 gate 关闭；因此 full AAA 的 backward 性能不能简单等同于 fused replay fastpath 已生效。

### 2.4 不能直接比较的结果

| 结果 | 为什么不能直接作为同口径对比 |
|---|---|
| 243.4s / 5000 step | cam0-only，不是 76-view multi-view |
| 5000-step / cap30000 | step 数和最终 N 都更小，尾部负载完全不同 |
| 848.32s fastpath/L1 | 未打开完整 AAA 特性，不能代表 AAA-default/full-feature 性能 |
| 1049.79s Release+Fuchsia | 同样是 fast/L1-style 基线，不能和 full AAA 1200s+ 直接评价“变慢/变快” |

## 3. 主要算子拆解

### 3.1 全程 10000 step 分解

来源：`stage_timing.csv` 聚合，单位 ms。

| Stage / 算子 | p50 | p95 | p99 | max | mean | 占总均值 |
|---|---:|---:|---:|---:|---:|---:|
| total | 88.197 | 203.142 | 224.128 | 3530.452 | 104.753 | 100.0% |
| backward GPU | 48.466 | 71.912 | 76.150 | 3501.663 | 50.909 | 48.6% |
| forward total | 34.800 | 69.533 | 111.797 | 3300.411 | 41.024 | 39.2% |
| raster | 14.067 | 25.334 | 47.791 | 3280.012 | 16.257 | 15.5% |
| preprocess process | 9.726 | 20.613 | 44.308 | 124.804 | 11.982 | 11.4% |
| raw download | 1.887 | 33.263 | 39.339 | 47.012 | 6.047 | 5.8% |
| sort | 4.899 | 10.917 | 15.519 | 33.743 | 5.752 | 5.5% |
| grad upload | 1.000 | 27.478 | 36.617 | 41.624 | 4.316 | 4.1% |
| bin | 2.559 | 4.457 | 12.313 | 34.142 | 2.866 | 2.7% |
| loss | 1.050 | 2.566 | 5.080 | 23.265 | 1.270 | 1.2% |
| densify | 0.000 | 0.000 | 0.001 | 177.726 | 0.397 | 0.4% |

解读：

- 全程均值看，`backward_gpu` 是最大单项，占约 49%。
- `forward_total` 占约 39%，其中主要来自 `raster` 和 `preprocess_process`。
- `raw_download` 与 `grad_upload` 虽不是均值第一，但在 high-N 尾部明显放大，说明当时 1049s 基线仍有 host/device 边界成本。
- `densify` 均值小，但 max 177ms，属于低频同步岛，会拉高 tail，不是 steady-state 主因。

### 3.2 high-N 后半段：`step >= 5000`

| Stage / 算子 | p50 | p95 | mean | 占总均值 |
|---|---:|---:|---:|---:|
| total | 124.634 | 214.646 | 134.027 | 100.0% |
| backward GPU | 59.040 | 74.308 | 59.807 | 44.6% |
| forward total | 50.371 | 71.248 | 50.529 | 37.7% |
| raster | 18.869 | 25.922 | 19.001 | 14.2% |
| preprocess process | 14.427 | 21.080 | 14.576 | 10.9% |
| raw download | 6.956 | 37.589 | 11.364 | 8.5% |
| grad upload | 4.094 | 33.794 | 8.255 | 6.2% |
| sort | 7.552 | 11.583 | 7.620 | 5.7% |

解读：

- 当 N 增长后，forward 与 backward 均明显上升。
- `raw_download + grad_upload` 合计均值约 19.6ms，占 high-N 后半段 14.7%，已经是可见的边界成本。
- sort 在 Fuchsia 之后不再是绝对主瓶颈，但仍随 R/N 增长。

### 3.3 最终尾部：`step >= 9900`, `N=200000`

| Stage / 算子 | p50 | p95 | mean | 占总均值 |
|---|---:|---:|---:|---:|
| total | 211.811 | 230.271 | 211.321 | 100.0% |
| backward GPU | 70.627 | 75.725 | 70.215 | 33.2% |
| forward total | 65.085 | 73.351 | 64.948 | 30.7% |
| raw download | 37.821 | 40.982 | 36.641 | 17.3% |
| grad upload | 34.103 | 37.295 | 32.505 | 15.4% |
| raster | 23.396 | 27.657 | 23.693 | 11.2% |
| preprocess process | 18.731 | 22.074 | 18.964 | 9.0% |
| sort | 9.328 | 12.517 | 9.619 | 4.6% |

解读：

- 1049s 基线尾部已经不是单一 kernel 问题。
- `backward_gpu + forward_total` 合计约 135ms，占 64%。
- `raw_download + grad_upload` 合计约 69ms，占 32.7%，几乎等于一个 `backward_gpu`。
- 所以当时最优先的优化确实是减少 steady-state transfer；这也解释了后续 fastpath 为什么能显著改善 10000-step。

## 4. 已验证的瓶颈迁移

### 4.1 旧瓶颈：high-N sort

早期 high-N 性能瓶颈是排序：

来源：

```text
harmonyos_3dgs/reports/basketball_stage_timing_step5000plus_N10000plus.md
harmonyos_3dgs/reports/basketball_stage_timing_step5000plus_N10000plus_fuchsia.md
```

结果：

| 指标 | 原路径 | Fuchsia SortPairs | 改善 |
|---|---:|---:|---:|
| high-N median total | 1974.840 ms/step | 437.539 ms/step | 4.51x |
| median sort | 1610.384 ms | 166.510 ms | 9.67x |

结论：

- 这证明早期不是单纯 shader 微优化问题，而是算法/实现路径差异。
- Fuchsia sort 把排序从主导瓶颈降为后续基线中的 5% 左右。

### 4.2 1049s 基线瓶颈：GPU 算子 + transfer 边界共同主导

1049s Release+Fuchsia 基线最终尾部：

```text
backward_gpu mean: 70.215 ms
forward_total mean: 64.948 ms
raw_download mean: 36.641 ms
grad_upload mean: 32.505 ms
```

结论：

- 当时继续优化应优先消除每步 raw/grad 方向的 CPU/GPU 数据往返。
- 这不是因为 GPU 算子不重要，而是因为 transfer 在尾部已经和核心算子同量级。

### 4.3 fastpath/L1 后瓶颈：转向 compute-bound / scale-bound

来源：

```text
harmonyos_3dgs/reports/basketball_fastpath_5000_vs_10000_perf_summary_20260515.md
```

后续 fastpath 开启：

```text
GS3D_TRAIN_STAGE_TIMING=1
GS3D_USE_FUCHSIA_SORT=1
GS3D_TRAIN_GPU_GRAD_ADAM=1
GS3D_TRAIN_GPU_RAW_ACTIVATE=1
```

10000-step fast/L1-style 结果：

| Run | elapsed | mean step | 提升 | 口径 |
|---|---:|---:|---:|---|
| 旧 Release+Fuchsia | 1049.79s | 104.900 ms | baseline | fast/L1-style，非完整 AAA |
| fastpath | 848.32s | 78.205 ms | elapsed 1.24x；mean 1.34x | fast/L1-style + GPU Grad Adam/Raw Activation，非完整 AAA |

fastpath 后 steady-state：

| Transfer metric | 状态 |
|---|---|
| `backward_download_ms` | 基本消除 |
| `grad_upload_ms` | 消除 |
| `raw_download_ms` | p50/p95 为 0，仅边界 materialization spike |

后续瓶颈：

| Stage | 10000-step late mean | 说明 |
|---|---:|---|
| backward GPU | 63.637 ms | 最大稳定单项 |
| forward total | 52.246 ms | 随 N 增长明显 |
| raster | 18.874 ms | forward 内主要项 |
| preprocess process | 15.664 ms | 高 N 后增长 2.4x |
| sort | 9.763 ms | 不再主导，但仍随规模增长 |
| adam GPU | 3.056 ms | 可接受，非主瓶颈 |

结论：

- transfer fastpath 方向已被数据验证，但验证对象是 fast/L1-style 口径。
- 848.32s 不能作为 full AAA 性能结论；full AAA 仍要看 1213s/1282s/1306s 等后续报告。
- 当前继续优化应转向 full AAA 口径下的 forward/backward 算子结构、跨算子数据复用、kernel 内部 tail/atomic/memory 行为，并单独跟踪 DSSIM/loss 相关开销。

### 4.4 full AAA 1200s+ 瓶颈：backward 与 forward/render 仍是主差距

以 `regression_10000_cap200000_20260519_recorded_sortpairs` 为代表的 full AAA 口径：

| Metric | CUDA | Vulkan full AAA | Vulkan/CUDA |
|---|---:|---:|---:|
| wall time | 770.04s | 1213.23s | Vulkan throughput 63.47% of CUDA |
| mean `total_ms` | 67.628 ms | 114.559 ms | Vulkan throughput 59.03% of CUDA |
| forward/render | 11.722 ms | 28.340 ms | Vulkan 2.418x CUDA |
| loss | 1.413 ms | 1.503 ms | Vulkan 1.064x CUDA |
| backward | 54.315 ms | 84.480 ms | Vulkan 1.555x CUDA |
| optimizer/noise | 7.352 ms | 1.202 ms | Vulkan faster |

代表性解读：

- full AAA 的 wall time 是 1200s+，不是 848s。
- 当前 full AAA 最好口径中，`backward_gpu_ms` mean 84.480ms，是最大单项；`forward_total_ms` mean 28.340ms，与 CUDA render 11.722ms 相比仍有 2.4x 差距。
- loss 在优化后的 full AAA 口径中已经接近 CUDA，但需要警惕回归：`master_reduce_cache_final_10000_cap200000_20260520_140029` 中 `loss_ms` mean 达到 23.524ms，导致 wall time 1454.39s；`dssim_nohash_final_10000_cap200000_20260520_152435` 把 `loss_ms` 恢复到 1.512ms 后，wall time 回到 1282.81s。

## 5. 定位逻辑：为什么我们这么判断

### Step 1：先确认基线有效

证据：

- 10000 行 stage timing；
- exit code 0；
- final loss 正常下降；
- final vertex count 达到 200000；
- 使用 Release build 和 Fuchsia sort；
- 不是 cam0-only，也不是 CTest。

因此 1049s 是有效训练基线，不是测量假象。

### Step 2：先做 stage 分解，而不是直接调 shader

分解结果显示：

```text
all steps: backward 48.6%, forward 39.2%
final tail: backward 33.2%, forward 30.7%, raw+grad transfer 32.7%
```

这说明尾部不是单点优化可解决，而是三个方向共同影响：

1. backward kernel；
2. forward/raster/preprocess；
3. CPU/GPU 边界数据流。

### Step 3：区分早期瓶颈和当前瓶颈

排序曾经是秒级瓶颈，Fuchsia 已经证明算法路径优化有效。Fuchsia 后 sort 仍增长，但不再解释 1049s 的主时间。

所以不能继续把“排序”作为唯一问题。

### Step 4：用 fastpath/L1 结果验证 transfer 假设，但不外推 full AAA

如果 1049s fast/L1-style 基线尾部 transfer 判断正确，那么在同一 fast/L1-style 口径下去掉 steady-state raw/grad transfer 应该显著改善 10000-step。实际结果：

```text
1049.79s -> 848.32s
mean 104.900 ms -> 78.205 ms
```

假设成立，但结论边界必须写清：这只证明 fast/L1-style 口径下 transfer fastpath 有效。完整 AAA 特性打开后，训练时间回到 `1200s+`，因为训练语义加入 DSSIM、regularization、noise、LR/SH schedule 等成本，且 full AAA proper-EWA 路径下 fused replay backward fastpath 可能被 gate 关闭。因此 full AAA 需要单独用 1213s/1282s/1306s 报告分析。

### Step 5：与 CUDA 对齐判断差距性质

来源：

```text
harmonyos_3dgs/reports/joint_operator_dataflow_optimization_research_20260520.md
```

CUDA/Vulkan 对比：

| Scope | CUDA | Vulkan | Vulkan/CUDA |
|---|---:|---:|---:|
| wall time | 770.04s | 1213.23s | 1.576x |
| total_ms | 67.628 ms | 114.559 ms | 1.694x |
| forward/render | 11.722 ms | 28.340 ms | 2.418x |
| loss | 1.413 ms | 1.503 ms | 1.064x |
| backward | 54.315 ms | 84.480 ms | 1.555x |
| optimizer + noise | 7.352 ms | 1.202 ms | 0.164x |

结论：

- Vulkan loss、optimizer/noise 已经不是核心差距；
- 主差距在 forward/render 和 backward；
- CUDA 的优势来自 autograd saved GPU buffers：`geomBuffer`、`binningBuffer`、`imgBuffer`、radii/color 等中间态长期留在 GPU，backward 直接消费；
- Vulkan 虽已有 `ForwardCache` 和 GPU chaining，但仍存在 CPU mirror/cache materialization、sync-style orchestration、kernel 效率和 replay/atomic 成本。

## 6. 已排除或降级的解释

| 解释 | 结论 | 证据 |
|---|---|---|
| “1049s 是 CTest 或 debug build 假象” | 排除 | Release executable，独立训练报告，exit 0，10000 stage rows |
| “243s 说明当前很慢 4x” | 降级 | 243s 是 cam0-only，不能和 76-view 对比 |
| “sort 仍是唯一主因” | 排除 | Fuchsia 后 sort 在 1049s 基线约 5%，fastpath 后尾部约 10ms |
| “loss/DSSIM 是主要瓶颈” | 需按口径区分 | fast/L1 的 1049s 基线 loss 约 1ms；full AAA 中 GPU DSSIM 正常优化后也接近 CUDA（约 1.5ms），但曾出现 `loss_ms` 23.524ms 的回归，需要单独监控 |
| “optimizer 是主要瓶颈” | 排除 | full AAA 口径下 Vulkan Adam/noise 约 1.2ms，已优于 CUDA，对总时间贡献小 |
| “继续消 raw_download/grad_upload 一定还有大收益” | 对 fast/L1 已部分排除；full AAA 需重测 | fastpath/L1 后 steady-state transfer 已消除；full AAA proper-EWA/fused-gate 状态不同，必须按 full AAA 报告重新验证 |
| “只要调 workgroup/subgroup 就能解决” | 未证实 | 缺少 per-kernel counter、occupancy、atomic 分布；需要先 profile |

## 7. 下一步优化路线

### P0：跳过 fused GPU backward 路径中不必要的 preprocess CPU cache download

优先级：最高
预期收益：2–6 ms/step，降低 forward/preprocess tail
置信度：高
风险：中

原因：

- `PreprocessorVulkan` 已有 GPU preprocess 输出；
- fused eval3D GPU backward 已能消费 GPU replay/handle；
- 但 `VulkanTrainer::run_forward_and_loss()` 仍可能调用 `preprocessor_.download_cache(...)` 下载 CPU cache；
- CUDA backward 直接消费 GPU saved buffers，不走 CPU cache。

需要确认：

- 在 eval3D + fused GPU backward + GPU loss gradient + no debug capture 时，CPU cache 是否确实无人消费；
- densify、opacity reset、parity/debug capture 是否需要保留同步 fallback。

验证门禁：

- gradient parity；
- `VulkanTrainer.Eval3D*`；
- `VkVsCudaFirstLoss`；
- 1000-step fixed schedule；
- 10000-step stage timing 对比 `preprocess_process_ms`、`forward_total_ms`。

### P1：训练 forward 改成完整 recorded command orchestration

优先级：高
预期收益：3–8 ms/step
置信度：中高
风险：中高

原因：

当前已有 preprocess/bin/sort/raster 的 record API，以及 canonical Fuchsia recorded SortPairs，但生产训练路径仍偏 Layer-1/sync-style。CUDA/PyTorch 的优势之一是算子边界更少、状态长期 GPU resident。

目标图：

```text
preprocess.record
→ barrier
→ binner.record
→ barrier
→ sorter.record_fuchsia_sortpairs
→ barrier
→ raster.record
```

重点：

- 保持 canonical SortPairs 正确性，不引入 packed-keyval 非等价路径；
- capacity 稳定时复用 descriptor/command setup；
- 对 R 动态变化使用 capacity guard 或 indirect/fixed dispatch。

### P2：对 backward fused replay 做 kernel-level profiling

优先级：最高但工作量大
预期收益：10–25 ms/step
置信度：瓶颈确认高，具体优化点待 profiling
风险：高

需要拆的 kernel/问题：

- `rasterize_backward_eval3d_replay_fused.comp`；
- tangent/subgroup variants；
- preprocess backward eval3D；
- atomic accumulation 热点；
- replay order distribution；
- tile/pixel work imbalance；
- memory bandwidth vs occupancy；
- 是否有 heavy tile 造成 p95/p99 tail。

成功标准：

- 不破坏 replay-order correctness；
- `VkVsCudaFirstLoss` 和 backward parity 通过；
- kernel timestamp 显示某一项实际下降，而不是 stage timing 被同步/下载掩盖。

### P3：loss scalar/partial readback 延迟到 logging interval

优先级：中
预期收益：1–3 ms/step 或降低 tail
风险：低中

原因：

- `dL_dpixels_gpu` 已可直接进入 backward；
- 每步 scalar loss 主要用于 logging/report，不一定需要阻塞训练 command stream；
- 可在 `log_every` 或用户请求时 readback。

注意：

- parity tests 需要同步 readback 模式；
- API 若要求 `step()` 返回实时 loss，需要增加显式模式或 async staging ring。

### P4：GPU-side R / replay count，减少小 scalar download 同步

优先级：中
预期收益：1–3 ms/step，主要降低 tail
风险：中

方向：

- count buffer 进入 indirect dispatch；
- 或固定按 capacity dispatch，shader 内 guard；
- replay offsets/counts 全 GPU resident。

### P5：persistent arena / device-local scratch pool

优先级：中
预期收益：1–4 ms/step，主要降低 p95/p99
风险：中

原因：

- forward buffer reuse 已证明有效；
- 仍可能存在小 UBO、scratch、staging、zero-fill 的生命周期开销；
- CUDA/PyTorch allocator 会摊销这类成本。

## 8. 下一步需要的 profiling / 求助手段

### 8.1 需要 GPU kernel 级时间线

目前 stage timing 能告诉我们哪个 stage 慢，但不能回答 backward 内部到底慢在：

- replay traversal；
- atomic add；
- memory bandwidth；
- subgroup divergence；
- occupancy/register pressure；
- tile imbalance。

需要支持：

1. Vulkan timestamp query per dispatch；
2. per-kernel CSV 输出：dispatch name、N、R、tiles、duration；
3. 对 backward 分成 raster backward / tangent reduce / preprocess backward / clears / barriers；
4. 对 forward 分成 preprocess / scan / scatter / sort / tile range / raster。

### 8.2 需要硬件 counter / profiler 支持

在 Tegra Thor / 目标 Maleoon 920 上，需要尽量获得：

- GPU occupancy / active warps 或等价指标；
- memory bandwidth；
- L2/cache hit；
- atomic throughput / contention；
- subgroup utilization；
- shader register pressure / spills；
- dispatch occupancy limits。

可求助对象：

- GPU driver / BSP 团队：确认 Vulkan timestamp/counter 可用性；
- 芯片/GPU 团队：提供 Maleoon 920 subgroup、shared memory、atomic 性能建议；
- 工具链团队：确认 `glslangValidator` / `spirv-opt` / driver compiler 对 subgroup shuffle、atomic、barrier 的优化行为；
- 算法参考团队：确认 CUDA eval3D backward 是否使用了额外 culling/saved-state trick，避免 Vulkan 优化错方向。

### 8.3 需要 tile/replay 分布数据

为了判断是 tail-dominated 还是 uniform inner-loop，需要增加诊断输出：

| 数据 | 用途 |
|---|---|
| per-tile splat count histogram | 判断 heavy tiles 是否主导 |
| replay_order_count per pixel/tile | 判断 backward replay 是否 tail-heavy |
| top 1/5/10% tile work share | 判断是否需要 load balancing |
| p50/p95/p99/max tile workload | 判断局部调度还是算法 culling |
| atomic target hotness per Gaussian | 判断是否需要 sharded accumulation |

如果 top 10% tile 占比很高，应优先做 load balancing / sharding / culling；如果 heavy tile 内部非常均匀，则 workgroup 微调收益有限，应做算法级减少工作量。

### 8.4 需要 CUDA 同口径 trace

现有 CUDA/Vulkan 对比已经显示 forward/backward gap，但下一步要拆到 CUDA kernel 级：

- CUDA `FORWARD::preprocess`；
- duplicate / sort / identifyTileRanges；
- CUDA `FORWARD::render`；
- CUDA `BACKWARD::render`；
- CUDA `BACKWARD::preprocess`。

目标是确认 Vulkan 落后是：

1. 算法状态复用不足；
2. shader/kernel 实现效率不足；
3. GPU architecture/compiler 差异；
4. 还是 stage orchestration/sync 边界。

## 9. 向上汇报建议版本

### 9.1 当前状态

我们已经把 10000-step basketball 训练的 fast/L1-style 路径从早期 high-N sort 秒级瓶颈推进到 Release+Fuchsia 1049s clean baseline，并在同一 fast/L1-style 口径下通过 GPU-resident fastpath 降到 848s 级别。这个 848s 不是完整 AAA 特性结果。完整 AAA 特性打开后，当前代表性结果是 1213.23s（另有 1282.81s、1306.95s 等），因此向上汇报必须以 full AAA 1200s+ 作为 feature-complete 性能口径。

### 9.2 关键原因

Vulkan 当前和 CUDA 的差距不是单点 API 开销，而是训练数据流和 saved-buffer 复用程度不同。CUDA/PyTorch extension 在 forward 保存 `geomBuffer/binningBuffer/imgBuffer` 等 GPU buffer，backward 直接消费；Vulkan 虽已建立 `ForwardCache` 和 GPU chaining，但仍有部分 CPU cache materialization、sync-style forward orchestration 和 backward replay/atomic kernel 效率差距。

### 9.3 已完成优化证明

- Fuchsia SortPairs：high-N median total 1974.840ms → 437.539ms，sort 1610.384ms → 166.510ms；
- GPU Grad Adam / Raw Activation fastpath：fast/L1-style 10000-step elapsed 1049.79s → 848.32s；这证明 transfer 优化有效，但不能代表 full AAA；
- full AAA 当前代表性结果：1213.23s wall，mean total 114.559ms，final loss 0.007439，final N=200000；
- full AAA 中 loss 正常优化后已接近 CUDA，optimizer/noise 已优于 CUDA；下一阶段主攻仍是 backward 与 forward/render，但需继续防止 DSSIM/loss 回归。

### 9.4 下一步计划

1. P0：在 full AAA proper-EWA 口径下确认 fused replay backward 是否实际启用；若 gate 关闭，先明确原因和替代优化路径；
2. P1：跳过 GPU-resident backward 路径中不必要的 preprocess CPU cache/materialization；
3. P2：训练 forward 改为完整 recorded command graph；
4. P3：对 full AAA backward / proper-EWA backward 做 kernel-level timestamp/counter profiling；
5. P4/P5：延迟 loss scalar readback、GPU-side R/replay count、persistent scratch/arena，降低同步与 tail。

### 9.5 需要支持

请协助提供 GPU profiler/counter 能力、Maleoon 920 subgroup/atomic/shared-memory 性能建议、Vulkan timestamp query 可用性确认，以及 CUDA 同口径 kernel trace。没有这些数据，继续做 shader 微调容易变成猜测，无法判断是 atomic contention、memory bandwidth、occupancy 还是算法 tail 主导。

## 10. 建议下一轮实验

### 实验 A：full AAA fused/proper-EWA backward gate audit

Hypothesis：full AAA proper-EWA 口径下虽然设置了 `GS3D_EVAL3D_FUSED_REPLAY_BWD=1`，但 fused replay backward 可能因 `--proper_ewa 1` 被 gate 关闭，导致 1200s+ full-feature 结果不能享受 848s fastpath/L1 的同类 backward 优化。
Measurement/change：在 trainer stage timing 中显式记录 backward path：standard eval3D / fused replay / tangent / subgroup / proper-EWA fallback；同时记录 gate 关闭原因。
Expected movement：先不期待性能变化，目标是把 full AAA 的真实执行路径讲清楚。
Validation：`VulkanTrainer.Eval3D*`、`VkVsCudaFirstLoss`、full AAA 1000-step fixed schedule。
Stop/pivot：如果 fused path 已经启用但 `backward_gpu_ms` 仍高，则转 kernel-level profiling；如果 gate 关闭，则先设计 proper-EWA 等价 fastpath。

### 实验 B：full AAA cache/materialization gate

Hypothesis：full AAA GPU-resident backward 路径仍有不必要 CPU preprocess cache/materialization 或 scalar sync，去掉后 `preprocess_process_ms` / `forward_total_ms` 降低。
Measurement/change：新增 cache materialization policy，只在 CPU consumer/debug/parity/densify 需要时下载；full AAA proper-EWA 单独验证。
Expected movement：1000-step full AAA cap30000 降 2–6ms/step；10000-step full AAA high-N tail 降低 p50/p95。
Validation：gradient parity、first loss gates、1000-step fixed schedule、full grouped CTest。
Stop/pivot：如果 `preprocess_process_ms` 不动，说明瓶颈不在 CPU cache download，应转 recorded forward 或 backward kernel profiling。

### 实验 C：backward dispatch timestamp split

Hypothesis：full AAA `backward_gpu_ms` 内部由 proper-EWA raster backward 或 preprocess backward 某一 kernel 主导。
Measurement/change：为 backward command chain 添加 per-dispatch timestamp CSV，并区分 standard/fused/proper-EWA 路径。
Expected movement：明确最大 kernel 和 barrier/clear 占比，不直接期待性能变化。
Validation：timestamp overhead 单独标记，不把诊断 run 当生产 timing。
Stop/pivot：如果 timestamp 显示多个 kernel 均匀且无明显同步，转 tile/replay 分布分析。

### 实验 D：tile/replay concentration histogram

Hypothesis：tail 是 heavy tile / replay order concentration 导致，而不是每 tile 均匀变慢。
Measurement/change：输出 per-tile work histogram、top 10% work share、p50/p95/p99/max。
Expected movement：确认是否 tail-dominated。
Validation：diagnostic-only，不改变训练结果；抽样或单 step 运行避免诊断 overhead 污染。
Stop/pivot：如果 top 10% 不集中，则优先做 kernel inner-loop / memory optimization，而不是 load balancing。
