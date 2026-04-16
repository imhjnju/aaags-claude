# AAA-Gaussians 移植问题修复总结

本文档记录 AAA-Gaussians eval_3D 模式移植到鸿蒙 HarmonyOS (CPU + Maleoon 920 GPU) 过程中遇到的所有问题、根因分析和修复方案。

---

## 问题 1: CPU渲染出现 16×16 块状伪影

### 现象

CPU eval_3D 渲染 basketball.ply 时，输出图像出现明显的 16×16 像素方块状伪影，尤其在平滑区域（桌面、背景）可见。

### 根因

**AABB center 与 NDC 投影 center 的偏差。**

最初使用 `computeAABBScreen()` 返回的 `mean2D` 作为 tile 分配的中心点。但 AABB center 是在 gauss2screen 矩阵的屏幕空间中计算的，与标准 3DGS 使用的 `ndc2Pix()` 投影中心存在系统性偏差。这导致 Gaussian 被分配到错误的 tile 集合，在 tile 边界处产生不连续。

### 修复方案

Tile 分配使用 NDC 投影中心（与标准 2D 路径一致），AABB extent 仅用于确定 radius 大小。

### 涉及文件

- `src/cpu/preprocessor_cpu.cpp` — eval_3D tile 分配逻辑
- `src/gpu/kernels/preprocess.cl` — GPU eval_3D tile 分配逻辑

---

## 问题 2: GPU 渲染出现 tile 边界伪影（CPU/GPU 不一致）

### 现象

GPU eval_3D 渲染结果与 CPU 参考有较大差异（mean diff=9.53），tile boundary discontinuity 是 interior 的 2.6 倍。CPU 渲染无此问题。

### 根因

**GPU scatter kernel 残留了逐 Tile 3D 裁剪代码，但 CPU tile binner 已回退。**

### 修复方案

移除 `scatter.cl` 中的逐 tile 裁剪代码，与 CPU binner 保持一致。

### 修复效果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| CPU vs GPU mean diff | 9.53 | **0.48** |
| Boundary/interior ratio | 2.631 | **0.994** |

---

## 问题 3: 逐 Tile 3D 裁剪导致整块 tile 内容丢失

### 现象

启用逐 Tile 3D 裁剪后，渲染性能大幅提升，但图像出现整块 tile 变黑或颜色错误。

### 根因

**低透明度 Gaussian 的累积贡献被忽略。** 单个 Gaussian 的贡献低于阈值，但数百个这样的 Gaussian 累积后产生可见的颜色。逐 tile 裁剪将它们全部去除，导致整块 tile 丢失内容。

### 处理方案

**禁用逐 Tile 裁剪。** 需配合 StopThePop 分层排序才能避免伪影，而 StopThePop 依赖 CUDA warp 语义，无法在 OpenCL 上实现。

---

## 问题 4: Per-Tile Depth Key 导致块状伪影

### 现象

启用 per-tile depth key (depthAlongRay) 后，相邻 tile 边界处出现排序不连续。

### 根因

同一 Gaussian 在不同 tile 中使用不同的深度排序键。两个相邻 tile 可能对同一对 Gaussian 产生相反的排序顺序，导致边界处 alpha blending 结果不连续。

### 处理方案

**禁用 per-tile depth key**，回退到全局 view-space z 排序。全局排序虽不是每个像素最优，但保证 tile 间一致性，消除边界伪影。

---

## 问题 5: kBuffer Per-Pixel Depth 计算不稳定

### 现象

使用 `max_pos` 的 z 分量作为 kBuffer 排序键时，渲染出现严重方块伪影。

### 根因

当 `dd`（两个平面法向量叉积的模长平方）接近零时，除法结果发散，导致 per-pixel depth 数值不稳定。

### 修复尝试

1. 改用 view-space z 作为排序键 → 数值稳定，但 kBuffer K=4 窗口太小，pop 时机差异仍产生 tile 边界不连续
2. 最终回退：**禁用 kBuffer**，使用全局排序 + 直接混合

---

## 问题 6: Radix Sort 后光栅化时间反增

### 现象

| 指标 | 旧版 (bitonic) | 新版 (radix) |
|------|---------------|-------------|
| Sort | 4,755 ms | 178 ms |
| Rasterize | 127 ms | 492 ms |
| vs CPU 像素差异 | mean=9.53, 84% 不一致 | mean=0.0001, 近乎完美 |

排序加速 26.7x，但光栅化反而慢了 3.9x。

### 根因

**旧版 bitonic sort 的 `skip_merge = true` 导致排序不完整，光栅化"假快"。**

排序不完整的连锁效应：
1. Alpha blending 顺序错误 → 不透明的前景 Gaussian 被排到后面
2. 错误顺序下后方 Gaussian 先混合 → 透过率 T 过早饱和到 0.0001 阈值
3. `if (test_T < 0.0001f) break` 触发 early exit → rasterize 看起来快（127ms）
4. 但图像有 **84% 像素与 CPU 参考不一致**

新版 radix sort 完全正确排序后：
1. 前到后严格顺序 → 前景先混合
2. 每个像素处理更多 Gaussian 才能饱和 → rasterize 更慢（492ms）
3. 但与 CPU 参考**完全一致**（mean diff=0.0001）

### 结论

旧版 rasterize 的"高性能"是排序错误的副产物，不是真实性能。正确排序后光栅化时间增加是**预期行为**。

---

## 最终架构

```
Preprocessor → TileBinner → RadixSort → Rasterizer
     │                                       │
     ├─ eval_3D / gauss2screen               ├─ Per-pixel 3D evaluation
     ├─ Scale dilation + mip filter          └─ Direct alpha blending
     ├─ Global 3D frustum culling                (global sort order)
     ├─ Screen-space AABB
     └─ View-space AABB fallback (*)
```

(*) fallback: 仅在 screen-space AABB 失败时触发

### 性能汇总

**设备**: 鸿蒙手机, Maleoon 920 GPU (6 CUs, 32KB local mem)
**场景**: basketball.ply, 400K Gaussians, 720×960, camera 0

| 版本 | Total | Sort | Rasterize | 正确性 |
|------|-------|------|-----------|--------|
| 原始 (bitonic, 错误排序) | 5,040 ms | 4,755 ms | 127 ms | mean diff=9.53 |
| Radix Sort (正确排序) | 777 ms | 178 ms | 492 ms | mean diff=0.0001 |
| **总加速** | **6.5x** | **26.7x** | — | **正确** |

### 已禁用特性及原因

| 特性 | 禁用原因 |
|------|----------|
| Per-tile 3D frustum culling | 低透明度 Gaussian 累积丢失 → tile 内容消失 |
| Per-tile depth key | tile 间排序不一致 → 边界伪影 |
| Per-pixel kBuffer (K=4) | 窗口太小 + pop 时机差异 → 边界伪影 |

以上三个特性均需 StopThePop hierarchical per-pixel sorting 配合才能正确工作，而 StopThePop 依赖 CUDA warp 语义（`__shfl_sync`, `__ballot_sync`, cooperative groups），无法在 OpenCL/Maleoon GPU 上可靠实现。
