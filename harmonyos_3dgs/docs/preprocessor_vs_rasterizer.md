# Preprocessor vs Rasterizer 对比文档

> 文档版本：1.0  
> 最后更新：2026-03-31  
> 对应源码：`harmonyos_3dgs/src/cpu/` 下的预处理和光栅化模块

---

## 执行摘要

本文档详细对比了 3D Gaussian Splatting 训练管线中 **Preprocessor** 和 **Rasterizer** 两个核心模块的职责、数据流和梯度传递关系。

**核心区别**:

| 维度 | Preprocessor | Rasterizer |
|------|--------------|------------|
| **处理粒度** | 每个 Gaussian 独立 (O(N)) | 每个像素 (O(HW × 覆盖数)) |
| **正向功能** | 参数激活 → 投影 → Conic → SH 颜色 | Tile 内逐像素 Alpha 混合 |
| **反向输出** | `d_raw_*` (原始参数梯度) | `d_conics` (屏幕空间梯度) |
| **数据依赖** | 仅依赖单 Gaussian 状态 | 依赖 Tile 内所有覆盖像素 |

**梯度传递关系**:

```
Rasterizer Backward (先执行)
  d_image → d_conics, d_means2D, d_rgb, d_opacity_2d
                    │
                    ▼
Preprocessor Backward (后执行)
  d_conics → d_cov2D → d_cov3D → d_raw_scales, d_raw_rotations
  d_means2D → d_raw_positions
  d_rgb → d_raw_sh_coeffs
  d_opacity_2d → d_raw_opacities
```

---

## 目录

1. [正向传播对比](#1-正向传播对比)
2. [反向传播对比](#2-反向传播对比)
3. [数据结构对比](#3-数据结构对比)
4. [计算复杂度分析](#4-计算复杂度分析)
5. [内存访问模式](#5-内存访问模式)
6. [GPU 优化策略](#6-gpu-优化策略)

---

## 1. 正向传播对比

### 1.1 Preprocessor (阶段 1)

**文件**: `src/cpu/preprocessor_cpu.cpp`

**输入**:
- `RawGaussianParams`: 原始参数 (位置、缩放、旋转、SH 系数、不透明度)
- `Camera`: 相机参数 (viewproj 矩阵、内参)
- `RenderConfig`: 渲染配置 (抗锯齿、SH degree 等)

**输出**: `PreprocessOutput`

```cpp
struct PreprocessOutput {
    float* means2D;        // [N*2] 屏幕空间均值
    float* depths;         // [N] 视图空间深度
    float* conics;         // [N*3] 逆协方差矩阵上三角
    float* rgb;            // [N*3] SH 评估颜色
    float* opacities_2d;   // [N] 2D 不透明度 (含 h_scale)
    int* radii;            // [N] 屏幕空间半径
    int* tiles_touched;    // [N] 覆盖 Tile 数
    float* gauss2screen;   // [N*16] 3D→屏幕变换矩阵 (eval_3D 模式)
    float* cov3D_inv;      // [N*6] 逆 3D 协方差 (eval_3D 模式)
    float* mean_offset;    // [N*3] 相对于相机的偏移 (eval_3D 模式)
};
```

**处理流程**:

```
RawGaussianParams
    │
    │ 1. 参数激活
    │    - positions: 恒等 (无激活)
    │    - scales: exp()
    │    - rotations: 归一化四元数
    │    - opacities: sigmoid()
    │    - sh_coeffs: 恒等 (无激活)
    ▼
GaussianData (激活后参数)
    │
    │ 2. 视锥剔除 (z > 0.2)
    │ 3. 齐次投影 → NDC
    │ 4. 计算 cov3D = M^T · M (M = S · R)
    │ 5. 投影 cov3D → cov2D (含 frustum clamping)
    │ 6. +0.3 低通滤波 → det → conic = cov2D^(-1)
    │ 7. 计算屏幕半径和 Tile 覆盖范围
    │ 8. SH 评估 → RGB
    ▼
PreprocessOutput
```

**关键计算** (第 209-220 行):

```cpp
// 2D 协方差 (含 +0.3 低通滤波)
float det_cov = cov2d[0] * cov2d[2] - cov2d[1] * cov2d[1];
cov2d[0] += 0.3f;
cov2d[2] += 0.3f;
float det = cov2d[0] * cov2d[2] - cov2d[1] * cov2d[1];

// Conic 矩阵 (逆协方差)
float det_inv = 1.0f / det;
float conic[3] = {
    cov2d[2] * det_inv,      // c / det
    -cov2d[1] * det_inv,     // -b / det
    cov2d[0] * det_inv       // a / det
};
```

---

### 1.2 Rasterizer (阶段 3)

**文件**: `src/cpu/rasterizer_cpu.cpp`

**输入**:
- `PreprocessOutput`: 预处理结果
- `BinningOutput`: Tile 分配和排序结果

```cpp
struct BinningOutput {
    uint32_t* values_sorted;  // [M] 排序后的高斯 ID 列表
    uint32_t* tile_ranges;    // [grid_x*grid_y*2] 每个 Tile 的范围
};
```

**输出**: `rendered_image` [H*W*3]

**处理流程**:

```
PreprocessOutput + BinningOutput
    │
    │ 对每个 Tile:
    │   对每个像素 (px, py):
    │     1. 从排序列表收集覆盖该像素的 Gaussians
    │     2. 对每个 Gaussian 计算 power = -0.5*(conic·offset²)
    │     3. alpha = opacity * exp(power)
    │     4. front-to-back Alpha 混合:
    │        C = Σ α_i * T_i * c_i + T_N * c_bg
    ▼
rendered_image [H*W*3]
```

**关键计算** (第 86-92 行):

```cpp
float dx = pre.means2D[idx*2]   - (float)px;
float dy = pre.means2D[idx*2+1] - (float)py;
float con_a = pre.conics[idx*3];
float con_b = pre.conics[idx*3+1];
float con_c = pre.conics[idx*3+2];

// 2D 高斯求值
float power = -0.5f * (con_a*dx*dx + con_c*dy*dy) - con_b*dx*dy;
float alpha = std::min(0.99f, pre.opacities_2d[idx] * std::exp(power));

// Alpha 混合
C[0] += alpha * T * pre.rgb[idx*3+0];
C[1] += alpha * T * pre.rgb[idx*3+1];
C[2] += alpha * T * pre.rgb[idx*3+2];
T *= (1.0f - alpha);
```

---

## 2. 反向传播对比

### 2.1 Rasterizer Backward (先执行)

**文件**: `src/cpu/rasterizer_backward_cpu.cpp`

**输入**:
- `d_image` [H*W*3]: 像素颜色梯度
- `ForwardCache`: 前向缓存 (T_final, n_contrib)

**输出**: `RasterGradOutput`

```cpp
struct RasterGradOutput {
    float* d_means2D;       // [N*2] 屏幕空间均值梯度
    float* d_conics;        // [N*3] Conic 矩阵梯度 ← 传递给 Preprocessor
    float* d_rgb;           // [N*3] 颜色梯度
    float* d_opacities_2d;  // [N] 透明度梯度
};
```

**处理流程**:

```
d_image [H*W*3]
    │
    │ 对每个像素回放正向过程 (back-to-front):
    │   1. 从后向前遍历贡献者
    │   2. 计算 d_alpha (体渲染公式)
    │      d_alpha = T_i * (c_i - C_behind) · d_C
    │   3. 链式法则:
    │      - d_rgb += α * T * d_C
    │      - d_power = d_alpha * opacity * exp(power)
    │      - d_means2D += d_power * ∂power/∂(x,y)
    │      - d_conics += d_power * ∂power/∂(conic)
    ▼
RasterGradOutput
```

**d_conics 计算** (129-132 行):

```cpp
// Chain: power -> conics
rgrad.d_conics[gauss_idx*3]     += d_power * (-0.5f * dx * dx);
rgrad.d_conics[gauss_idx*3 + 1] += d_power * (-dx * dy);
rgrad.d_conics[gauss_idx*3 + 2] += d_power * (-0.5f * dy * dy);
```

---

### 2.2 Preprocessor Backward (后执行)

**文件**: `src/cpu/preprocessor_backward_cpu.cpp`

**输入**:
- `RasterGradOutput`: 光栅化梯度 (含 `d_conics`)

**输出**: `GradientOutput`

```cpp
struct GradientOutput {
    float* d_raw_positions;  // [N*3] 位置梯度
    float* d_raw_scales;     // [N*3] 缩放梯度
    float* d_raw_rotations;  // [N*4] 旋转梯度
    float* d_raw_sh_coeffs;  // [N*max_coeffs*3] SH 系数梯度
    float* d_raw_opacities;  // [N] 不透明度梯度
};
```

**处理流程**:

```
RasterGradOutput
    │
    │ 对每个 Gaussian 并行计算 4 条梯度链:
    │
    │ Chain 1 (协方差): d_conics → d_cov2D → d_cov3D → d_M → d_raw_scales, d_raw_rotations
    │                   ↘ d_T → d_J → d_p_view → d_raw_positions (间接贡献)
    │
    │ Chain 2 (颜色):   d_rgb → d_raw_sh_coeffs
    │
    │ Chain 3 (透明度): d_opacity_2d → d_raw_opacities (sigmoid 雅可比)
    │
    │ Chain 4 (位置):   d_means2D → d_raw_positions (投影反向)
    ▼
GradientOutput
```

**d_cov2D 计算** (295-306 行):

```cpp
// 从缓存读取 cov2D 元素
float a = cache.cov2D[i*3];
float b = cache.cov2D[i*3+1];
float c = cache.cov2D[i*3+2];
float det = cache.cov2D_det[i];
float inv_det = 1.0f / det;
float inv_det2 = inv_det * inv_det;

// 从 rgrad 读取 d_conics
float dc0 = rgrad.d_conics[i*3];
float dc1 = rgrad.d_conics[i*3+1];
float dc2 = rgrad.d_conics[i*3+2];

// Conic 逆矩阵求导
float d_a = dc0 * (-c*c * inv_det2)
          + dc1 * (b*c * inv_det2)
          + dc2 * (-b*b * inv_det2);

float d_b = dc0 * (2.0f*b*c * inv_det2)
          + dc1 * (-(a*c + b*b) * inv_det2)
          + dc2 * (2.0f*a*b * inv_det2);

float d_c = dc0 * (-b*b * inv_det2)
          + dc1 * (a*b * inv_det2)
          + dc2 * (-a*a * inv_det2);
```

---

## 3. 数据结构对比

### 3.1 输入数据

| 数据 | Preprocessor | Rasterizer |
|------|--------------|------------|
| 位置 | `raw_positions[N*3]` | `means2D[N*2]` (投影后) |
| 外观 | `raw_sh_coeffs[N*max_coeffs*3]` | `rgb[N*3]` (评估后) |
| 几何 | `raw_scales[N*3], raw_rotations[N*4]` | `conics[N*3]` (投影后) |
| 透明度 | `raw_opacities[N]` | `opacities_2d[N]` (激活后) |
| 组织 | 数组 of Structures | Structure of Arrays |

### 3.2 输出数据

| 数据 | Preprocessor | Rasterizer |
|------|--------------|------------|
| 主要输出 | `PreprocessOutput` (per-Gaussian) | `rendered_image[H*W*3]` (per-pixel) |
| 辅助输出 | `radii[N]`, `tiles_touched[N]` | `ForwardCache` (用于反向) |
| 梯度输出 (反向) | `GradientOutput` (原始参数) | `RasterGradOutput` (屏幕空间) |

### 3.3 内存占用 (以 1M Gaussians, 1080p 为例)

| 模块 | 数据 | 大小 (MB) |
|------|------|-----------|
| **Preprocessor** | means2D | 4 |
| | conics | 12 |
| | rgb | 12 |
| | opacities_2d | 4 |
| | radii, tiles_touched | 8 |
| **Rasterizer** | rendered_image | 13 (1080p × 3 × 4B) |
| | ForwardCache (T_final, n_contrib) | 8 |
| | values_sorted (M) | 4~40 (取决于覆盖) |

---

## 4. 计算复杂度分析

### 4.1 正向传播

| 模块 | 操作 | 复杂度 | 主导因素 |
|------|------|--------|----------|
| **Preprocessor** | 参数激活 | O(N) | Gaussian 数量 |
| | 视锥剔除 | O(N) | Gaussian 数量 |
| | 投影 + cov2D | O(N) | Gaussian 数量 |
| | SH 评估 | O(N × deg²) | SH degree |
| **Rasterizer** | 逐像素混合 | O(HW × G/tile) | 分辨率 × 每 Tile 覆盖数 |

**典型场景** (1M Gaussians, 1080p, 平均每 tile 覆盖 50):
- Preprocessor: ~1M 次操作
- Rasterizer: ~2M 像素 × 50 = ~100M 次操作

**Rasterizer 是计算瓶颈**。

### 4.2 反向传播

| 模块 | 操作 | 复杂度 |
|------|------|--------|
| **Rasterizer Backward** | 像素回放 + 链式法则 | O(HW × G/tile) |
| **Preprocessor Backward** | 4 条梯度链 (并行) | O(N) |

---

## 5. 内存访问模式

### 5.1 Preprocessor

**访问模式**: 顺序访问 (每 Gaussian 处理一次)

```cpp
for (int i = 0; i < N; i++) {
    // 读取: g.positions[i*3], g.scales[i*3], g.rotations[i*4], ...
    // 写入: out.means2D[i*2], out.conics[i*3], out.rgb[i*3], ...
}
```

**特点**:
- 适合 SIMD/SIMT 并行
- 缓存友好 (顺序读取)
- 无原子操作 (每线程写独立位置)

### 5.2 Rasterizer

**访问模式**: 随机访问 (多像素可能访问同一 Gaussian)

```cpp
for (int py = py_min; py < py_max; py++) {
    for (int px = px_min; px < px_max; px++) {
        for (uint32_t j = range_start; j < range_end; j++) {
            uint32_t idx = bin.values_sorted[j];  // 随机访问
            float dx = pre.means2D[idx*2] - px;   // 随机读取
            // ...
        }
    }
}
```

**特点**:
- 缓存不友好 (随机访问 `pre.*` 数组)
- 需要原子操作 (如果多像素同时更新同一 Gaussian 梯度)
- 适合共享内存优化 (GPU)

---

## 6. GPU 优化策略

### 6.1 Preprocessor GPU 策略

**优化重点**: 并行度最大化

```opencl
// OpenCL kernel: 每个 work-item 处理一个 Gaussian
__kernel void preprocess_kernel(__global const RawGaussianParams* params,
                                __global PreprocessOutput* out,
                                int N) {
    int idx = get_global_id(0);
    if (idx >= N) return;
    
    // 独立处理，无同步需求
    // ...
}
```

**关键点**:
- 每 Gaussian 一线程
- 无线程间依赖
- 适合 GPU 大规模并行

### 6.2 Rasterizer GPU 策略

**优化重点**: 共享内存 + 合并访问

```opencl
// OpenCL kernel: 每个 work-group 处理一个 Tile
__kernel void rasterize_kernel(__global const PreprocessOutput* pre,
                               __global const BinningOutput* bin,
                               __global float* out_image,
                               int tile_id) {
    // 共享内存：缓存当前 Tile 的 Gaussians
    __local float shared_means2D[MAX_GAUSSIANS * 2];
    __local float shared_conics[MAX_GAUSSIANS * 3];
    __local float shared_rgb[MAX_GAUSSIANS * 3];
    
    // 协作加载到共享内存
    // 同步
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // 逐像素处理
    int px = ..., py = ...;
    float C[3] = {0};
    float T = 1.0f;
    
    for (int j = range_start; j < range_end; j++) {
        int idx = bin.values_sorted[j];
        // 从共享内存读取 (快于全局内存)
        float mx = shared_means2D[idx*2];
        // ...
    }
}
```

**关键点**:
- 每 Tile 一个 work-group
- 共享内存缓存 per-Gaussian 数据
- 减少全局内存带宽压力

---

## 7. 完整数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                      FORWARD PASS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  RawGaussianParams                                              │
│    - raw_positions [N*3]                                        │
│    - raw_scales [N*3]                                           │
│    - raw_rotations [N*4]                                        │
│    - raw_sh_coeffs [N*max_coeffs*3]                             │
│    - raw_opacities [N]                                          │
│         │                                                       │
│         │ activate()                                            │
│         ▼                                                       │
│  GaussianData                                                   │
│         │                                                       │
│         │ [PREPROCESSOR] ─────────────────────────────┐         │
│         │  - 视锥剔除                                  │         │
│         │  - 投影 cov3D → cov2D                        │         │
│         │  - 计算 conic = cov2D^(-1)                   │         │
│         │  - SH 评估 → RGB                             │         │
│         ▼                                             │         │
│  PreprocessOutput                                     │         │
│    - means2D [N*2]                                    │         │
│    - conics [N*3] ────────────────────────────────────┤         │
│    - rgb [N*3]                                        │         │
│    - opacities_2d [N]                                 │         │
│         │                                             │         │
│         │ [BINNING] 排序 + Tile 分配                   │         │
│         ▼                                             │         │
│  BinningOutput                                        │         │
│    - values_sorted [M]                                │         │
│    - tile_ranges [grid*2]                             │         │
│         │                                             │         │
│         │ [RASTERIZER] ───────────────────────────────┤         │
│         │  - 逐像素 Alpha 混合                          │         │
│         ▼                                             │         │
│  rendered_image [H*W*3] ◄─────────────────────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
     │
     │ L1 Loss
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BACKWARD PASS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  d_image [H*W*3]                                                │
│         │                                                       │
│         │ [RASTERIZER BACKWARD] ──────────────────────┐         │
│         │  - 回放正向过程 (back-to-front)              │         │
│         │  - 计算 d_alpha (体渲染公式)                  │         │
│         │  - 链式法则: d_conics, d_means2D, d_rgb     │         │
│         ▼                                             │         │
│  RasterGradOutput                                     │         │
│    - d_conics [N*3] ──────────────────────────────────┤         │
│    - d_means2D [N*2]                                  │         │
│    - d_rgb [N*3]                                      │         │
│    - d_opacities_2d [N]                               │         │
│         │                                             │         │
│         │ [PREPROCESSOR BACKWARD] ────────────────────┤         │
│         │  Chain 1: d_conics → d_cov2D → d_cov3D     │         │
│         │                   → d_M → d_raw_scales     │         │
│         │  Chain 2: d_rgb → d_raw_sh_coeffs          │         │
│         │  Chain 3: d_opacity → d_raw_opacities      │         │
│         │  Chain 4: d_means2D → d_raw_positions      │         │
│         ▼                                             │         │
│  GradientOutput                                       │         │
│    - d_raw_positions [N*3] ◄─────────────────────────┤         │
│    - d_raw_scales [N*3] ◄────────────────────────────┤         │
│    - d_raw_rotations [N*4] ◄─────────────────────────┤         │
│    - d_raw_sh_coeffs [N*max_coeffs*3]                 │         │
│    - d_raw_opacities [N]                              │         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
     │
     │ Optimizer.step()
     ▼
RawGaussianParams (更新)
```

---

## 8. 总结

| 特性 | Preprocessor | Rasterizer |
|------|--------------|------------|
| **职责** | 参数 → 屏幕空间表示 | 屏幕空间表示 → 像素颜色 |
| **粒度** | per-Gaussian | per-pixel |
| **复杂度** | O(N) | O(HW × G/tile) |
| **内存访问** | 顺序 (缓存友好) | 随机 (需共享内存优化) |
| **反向输出** | 原始参数梯度 | 屏幕空间梯度 |
| **GPU 并行** | 每 Gaussian 一线程 | 每 Tile 一 block |

**梯度传递关系**: Rasterizer Backward 生成的 `d_conics` 是连接屏幕空间光栅化与几何参数优化的**关键桥梁**。Preprocessor Backward 接收 `d_conics` 后，通过 conic 逆矩阵求导、投影反向、协方差分解等步骤，最终得到原始参数的梯度。

理解两个模块的区别和联系对于调试训练问题、实现自定义损失函数或优化反向传播性能至关重要。
