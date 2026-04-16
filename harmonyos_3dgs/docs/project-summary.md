# HarmonyOS 3DGS 移植项目总结

## 项目目标

将 3D Gaussian Splatting (3DGS) 渲染器从 CUDA/Python 移植到鸿蒙手机，实现 CPU 和 GPU (OpenCL) 两个版本，在 Maleoon GPU 920 上验证正确性和性能。

## 完成状态

### CPU 版本 ✅ 完成

- 纯 C++17 实现，无第三方依赖
- 四阶段管线：Preprocessor → TileBinner → Sorter → Rasterizer
- 策略模式接口，CPU/GPU 可热切换
- 支持 SH degree 0-3，完整 alpha blending
- 与原始 Python 3DGS `eval_sh` 数值完全一致
- PC x86 和 ARM64 手机跨平台一致（PSNR 89-90 dB，`-ffp-contract=off`）

### GPU 版本 ✅ 完成（部分 kernel 回退 CPU）

- OpenCL 1.2，动态加载（dlopen），Khronos headers vendored
- 9 个 kernel 编写完成，3 个在前向渲染中调用
- 与 CPU 版本同设备对比 PSNR 80-90 dB
- basketball.ply + basket0.ply 全 76 相机验证通过

## 渲染管线架构

```
PLY 模型加载 → [Preprocessor] → [TileBinner] → [Sorter] → [Rasterizer] → PPM 输出
                    │                 │              │            │
                GPU kernel        GPU kernel      CPU 回退     GPU kernel
                preprocess         scatter       std::sort     rasterize
                 7.3 ms           14.4 ms        846 ms        314 ms
```

## 性能数据

### basketball.ply (400K Gaussians, 720×960, camera 0)

| 版本 | 总耗时 | 加速比 |
|------|--------|--------|
| CPU (PC x86) | 28,488 ms | 1x |
| CPU (ARM64 手机) | 12,953 ms | 2.2x |
| **GPU (Maleoon 920)** | **1,263 ms** | **22.6x** |
| 目标 (60 FPS) | 16.7 ms | ~1700x |

### basket0.ply (784K Gaussians, 720×960, camera 0)

| 版本 | 总耗时 | 加速比 |
|------|--------|--------|
| CPU (ARM64 手机) | 6,200 ms | 1x |
| **GPU (Maleoon 920)** | **650 ms** | **9.5x** |

### GPU 各阶段耗时

| 阶段 | 总耗时 | GPU kernel | CPU/传输 | 占比 |
|------|--------|-----------|---------|------|
| Preprocess | 40 ms | 7 ms | 33 ms (readback) | 3% |
| TileBinner | 51 ms | 14 ms | 37 ms (prefix sum+upload) | 4% |
| **Sorter** | **846 ms** | **0 ms** | **846 ms (CPU sort+150MB传输)** | **67%** |
| Rasterizer | 327 ms | 314 ms | 13 ms (readback) | 26% |

## OpenCL Kernel 状态

| Kernel | 文件 | 调用 | GPU 耗时 | 说明 |
|--------|------|------|---------|------|
| `preprocess` | preprocess.cl | ✅ | 7.3 ms | 投影+SH+协方差 |
| `scatter` | scatter.cl | ✅ | 14.4 ms | 键值对生成 |
| `rasterize` | rasterize.cl | ✅ | 313.5 ms | tile alpha blending |
| `scan_blocks` | prefix_sum.cl | ❌ | — | GPU prefix sum 3-level 未实现 |
| `add_block_sums` | prefix_sum.cl | ❌ | — | 同上 |
| `radix_histogram` | radix_sort.cl | ❌ | — | O(N²) scatter 太慢 |
| `radix_scatter` | radix_sort.cl | ❌ | — | 同上 |
| `identify_tile_ranges` | radix_sort.cl | ❌ | — | CPU sort 路径中不需要 |
| `sort_within_tiles` | radix_sort.cl | ❌ | — | 部分排序质量不足 |

## 调试过程中发现的 12 个问题

详见 `docs/porting-issues-summary.md`，核心问题：

| # | 问题 | 严重性 | 根因 |
|---|------|--------|------|
| 1 | 旋转矩阵 R 转置 | Critical | GLM 列填充 vs C 行存储 |
| 2 | Jacobian J 列布局错 | Critical | 同上 |
| 3 | ViewProj 乘法顺序 | Critical | 列主序下 Proj*View |
| 4 | cameras.json C2W/W2C 混淆 | Critical | Python 变量名误导 |
| 5 | PLY 硬编码属性索引 | High | 不同 PLY 属性数量 |
| 6 | SH 超亮白斑 | Medium | 训练固有问题，clamp [0,1] |
| 7 | GPU kernel 漏写 out_rgb | Critical | 局部变量未写全局内存 |
| 8 | CL_MEM_WRITE_ONLY 跨 kernel | Critical | OpenCL buffer flag |
| 9 | GPU radix sort 稳定性 | High | atomic 破坏 pass 间稳定性 |
| 10 | GPU prefix sum 溢出 | High | 3-level scan 未实现 |
| 11 | x86 vs ARM64 浮点差异 | Critical | FMA 舍入差异 |
| 12 | Debug readback 拖慢性能 | Medium | 50MB 验证读回 |

## 文档清单

| 文档 | 位置 |
|------|------|
| GPU OpenCL 设计规格 | `docs/gpu-opencl-design.md` |
| GPU 实现计划 (10 tasks) | `docs/gpu-implementation-plan.md` |
| 移植问题总结 (12 bugs) | `docs/porting-issues-summary.md` |
| CPU 调试报告 | `docs/debugging-report.md` |
| GPU 优化报告 | `docs/gpu-optimization-report.md` |
| 抗锯齿分析 | `docs/antialiasing-analysis.md` |
| 默认配置对比 | `docs/default-config-comparison.md` |
| GPU/CPU 精度分析 | `docs/gpu-cpu-precision-analysis.md` |
| Kernel 调用状态 | `docs/kernel-dispatch-status.md` |

## 渲染结果

| 目录 | 内容 |
|------|------|
| `docs/renders/all_cameras_selected/` | basketball 76 相机 GPU 渲染 (8 samples) |
| `docs/renders/basket0_gpu_samples/` | basket0 76 相机 GPU 渲染 (8 samples) |
| `docs/renders/` | 各阶段调试/对比渲染图 |

## 环境变量控制

| 变量 | 默认 | 说明 |
|------|------|------|
| `--gpu` | CPU | 启用 OpenCL GPU 后端 |
| `--cl-info` | — | 打印 GPU 设备信息 |
| `AA=1` | OFF | 启用 Mip-Splatting 抗锯齿 |
| `BG_WHITE=1` | 黑色 | 白色背景 |
| `SH_DEGREE=N` | 模型值 | 覆盖 SH 阶数 |
| `DUMP_DIAG=1` | OFF | 输出诊断数据 |

## 下一步

1. **GPU 全 pipeline 排序** — 消除 846ms CPU sort + 150MB 传输瓶颈
2. **Rasterizer kernel 优化** — 从 314ms 降到 ~10ms
3. **实时交互** — NativeWindow + 触摸手势控制相机
4. **60 FPS 目标** — 需要从 1263ms 优化到 16.7ms (~75x)
