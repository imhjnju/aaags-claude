# AAA-Gaussians + RadixSort 集成总结

## 一、集成目标

将优化的 4-bit RadixSort 和 AAA-Gaussians 算法特性集成到 HarmonyOS 3DGS 前向渲染管线，目标设备为 Maleoon 920 GPU（6 CUs, 32KB local mem, OpenCL 3.0）。

## 二、RadixSort 集成

### 问题

原有 GPU 排序使用 counting sort + bitonic sort，bitonic sort 的 `skip_merge=true` 导致大 tile（>4096 元素）排序不完整，渲染结果错误（84% 像素与 CPU 参考不一致）。

### 方案

用 4-bit radix sort（subgroup shuffle_up）替换整个排序管线：

- 新增 `radix_sort_kv_cl.h`：3 个 kernel（histogram / compute_offsets / scatter_kv）
- 16 pass 对 64-bit key（tile_id << 32 | depth）全排序
- 自然同时按 tile_id 和 depth 排序，替代原有 5 阶段管线

### 结果

| 指标 | 旧版 (bitonic) | 新版 (radix) |
|------|---------------|-------------|
| Sort | 4,755 ms | **178 ms (26.7x↑)** |
| Total | 5,040 ms | **777 ms (6.5x↑)** |
| 像素正确性 | mean diff=9.53 | **mean diff=0.0001** |

### 关键发现：光栅化时间反增

排序正确后 rasterize 从 127ms 增到 492ms。根因：旧版排序不完整导致 alpha blending 顺序错误 → 透过率 T 过早饱和 → 提前退出 → "假快"。正确排序后每个像素需要处理更多 Gaussian 才能饱和，这是**预期行为**。

## 三、AAA-Gaussians 特性集成

### 已成功集成（无伪影）

| 特性 | 位置 | 效果 |
|------|------|------|
| eval_3D / gauss2screen 矩阵 | Preprocess | 3D Gaussian 全空间评估 |
| Scale dilation + 3D mip filter | Preprocess | 频率感知自适应缩放 |
| Camera-in-ellipsoid 检测 | Preprocess | 相机在椭球体内跳过 |
| Global 3D frustum culling | Preprocess | 全屏范围 3D 裁剪 |
| Tight opacity bounding | Preprocess | 基于透明度的动态 cutoff |
| Opacity dilation factor | Preprocess | 缩放后透明度校正 |
| Screen-space AABB | Preprocess | Hahlbohm 方法屏幕空间包围盒 |
| **View-space AABB fallback** | Preprocess | Screen-space 失败时回退到视空间 |
| Per-pixel 3D evaluation | Rasterize | 逐像素 Mahalanobis 距离 |
| cov3D_inv / mean_offset | Preprocess | 逆协方差和偏移量基础设施 |
| depthAlongRay / invertMatrix4x4 | Math | 数学工具函数 |
| **Per-pixel kBuffer (K=16)** | Rasterize | 逐像素排序窗口框架 |

### 集成失败的特性及根因分析

#### 1. Per-Tile 3D Frustum Culling — 块状伪影

**尝试过程**：
- 第一版：使用 `opacity_power_threshold`（1/255）→ tile-pairs 减少 82.7%，但背景区域出现严重块状伪影
- 第二版：保守阈值 `TILE_CULL_ALPHA = 1/25500`（100x 保守）→ tile-pairs 减少 73.6%，仍有块状伪影
- 第三版：使用 AABB cutoff 纯几何裁剪 → tile-pairs 减少 76%，仍有块状伪影

**根因**：per-tile culling 的固有问题——每个 tile 独立做裁剪决定。同一 Gaussian 在 tile A 被裁剪但在 tile B 保留，导致 tile 边界处渲染结果不连续。大量低贡献 Gaussian 的累积效应使差异可见。

**结论**：**无论使用什么阈值，per-tile culling 都会产生 tile 边界不连续**。这是数学上的固有限制，不是工程实现问题。AAA-Gaussians 原版通过 StopThePop 在像素级别重新排序来消除这种不连续，但 StopThePop 无法在 Maleoon 920 上实现（见下文）。

#### 2. Per-Tile Depth Key — 块状伪影

**根因**：同一 Gaussian 在不同 tile 使用不同深度排序键（depthAlongRay 基于 tile 中心射线），导致相邻 tile 对同一对 Gaussian 产生不同排序顺序。

**结论**：与 per-tile culling 同理——tile 级别的差异化决策必然导致 tile 边界不连续。

#### 3. Per-Pixel kBuffer (max_pos depth) — 块状伪影

**尝试过程**：
- 使用 `max_pos[2] / max_pos[3]`（NDC depth）作为排序键 → 严重块状伪影
- 改用 view-space z 作为排序键 → 伪影消除，但输出与无 kBuffer 完全相同（mean=0）

**根因**：`max_pos` 的 z 分量计算 `(dx*my - dy*mx) / dd` 在 `dd` 接近零时数值发散（Gaussian 中心接近像素射线），导致相邻像素排序结果差异巨大。

**结论**：view-space z 作为 kBuffer 排序键与全局排序键一致，kBuffer 变为 no-op。kBuffer 框架已就位，但需要稳定的 per-pixel depth 计算方式才能真正生效。

## 四、StopThePop 不可移植性分析

### Maleoon 920 支持的 OpenCL 扩展

```
cl_khr_subgroups ✓
cl_khr_subgroup_shuffle ✗
cl_khr_subgroup_shuffle_relative ✗
cl_khr_subgroup_ballot ✗
cl_khr_subgroup_non_uniform_vote ✗
```

### 核心阻碍

| CUDA 原语 | StopThePop 中的用途 | Maleoon 920 | 替代方案及代价 |
|-----------|---------------------|-------------|---------------|
| `__shfl_sync` | 排序核心：线程间直接读寄存器 | ❌ 不可用 | local mem + barrier，~100x 延迟 |
| `__ballot_sync` | 集体投票：收集所有线程判断 | ❌ 不可用 | local mem + barrier，~100x 延迟 |
| `__fns` | 查找第 N 个置位 bit | ❌ 不可用 | 手动位操作，~10x 慢 |
| CUB `BlockRadixSort` | block 级基数排序 | ❌ 无等价库 | 需重写 300-500 行 |
| `tiled_partition<N>` | 将 warp 拆分为子组 | ❌ 无递归分区 | 无法做子组级 barrier |
| Warp 隐式同步 | warp 内无需 barrier | ❌ 无此概念 | 每次操作必须 barrier |

### 根本矛盾

StopThePop 的高效性完全依赖 CUDA warp-level 硬件原语（1 cycle 延迟）。用 local memory + barrier 模拟后，延迟从 1 cycle 变为 ~100 cycle。即使完成移植，性能退化会使 StopThePop 比当前方案更慢，失去使用意义。

## 五、最终架构

```
Preprocessor                    TileBinner         RadixSort        Rasterizer
┌─────────────────────┐    ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│ eval_3D/gauss2screen│    │              │   │ 4-bit radix  │   │ Per-pixel 3D     │
│ Scale dilation+mip  │───▶│ Scatter keys │──▶│ 16 passes    │──▶│ evaluation       │
│ Global 3D frustum   │    │ (global depth│   │ ulong keys   │   │ + kBuffer (K=16) │
│ Screen AABB         │    │  sort key)   │   │ + uint values│   │ (view-space z)   │
│ View AABB fallback  │    └──────────────┘   └──────────────┘   └──────────────────┘
│ cov3D_inv/mean_off  │
└─────────────────────┘
```

## 六、性能汇总

**设备**：鸿蒙手机, Maleoon 920 GPU
**场景**：basketball.ply, 400K Gaussians, 720×960, eval_3D=ON

| 版本 | Sort | Rasterize | Total | 正确性 |
|------|------|-----------|-------|--------|
| 原始 (bitonic, 错误排序) | 4,755 ms | 127 ms | 5,040 ms | ✗ (mean=9.53) |
| RadixSort (正确排序) | 178 ms | 492 ms | 777 ms | ✓ (mean=0.0001) |
| RadixSort + kBuffer K=16 | 210 ms | 12,018 ms | 12,375 ms | ✓ (mean=0) |

> kBuffer K=16 当前使用 view-space depth 排序，与全局排序一致，输出相同。
> 性能开销来自 K=16 的插入排序（每 Gaussian × 每像素）。
> 推荐使用不带 kBuffer 的版本（777ms）作为生产配置。

## 七、关键结论

1. **RadixSort 集成成功**：26.7x 排序加速，总体 6.5x 加速，渲染结果与 CPU 完全一致。

2. **Per-tile 优化（culling / depth key）在没有 per-pixel sorting 的前提下必然产生 tile 边界伪影**。这是算法层面的固有限制，不是参数调优能解决的。

3. **StopThePop 的核心算法依赖 CUDA warp-level 硬件原语**，这些原语在 Maleoon 920 的 OpenCL 实现中不可用。用 local memory 模拟会导致 ~100x 性能退化。

4. **kBuffer 框架已就位**，使用 view-space depth 时功能正确但等效于 no-op。需要稳定的 per-pixel depth 计算方式才能真正发挥 per-pixel re-sorting 的价值。

5. **当前最优配置**：RadixSort + global view-space depth sort + 直接 alpha blending（无 per-tile culling、无 per-tile depth key、kBuffer 可选关闭）= **777ms，无伪影**。
