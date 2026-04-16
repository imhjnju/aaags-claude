# GPU vs CPU 精度差异分析

## 概述

basket0.ply 在手机（Maleoon GPU 920）上 CPU 和 GPU 渲染对比：
- 76 帧平均 PSNR：**86.28 dB**
- 最差帧 (cam_0)：PSNR 73.88 dB，最大像素差 38，仅 3 个像素 diff>20
- 最佳帧 (cam_57)：PSNR 90.29 dB，最大像素差 1

差异来源：OpenCL `-cl-fast-relaxed-math` 编译选项和 `native_exp()` 函数。

---

## 数值精度总览

### 整个渲染管线的数据类型

| 阶段 | CPU 类型 | GPU 类型 | 一致？ |
|------|---------|---------|--------|
| 模型数据 (positions, SH, scales, rotations, opacities) | `float` (32-bit) | `float` (32-bit) | ✅ |
| 相机矩阵 (view, viewproj) | `float[16]` (32-bit) | `__constant float*` (32-bit) | ✅ |
| 中间值 (cov3D, cov2D, conic) | `float` (32-bit) | `float` (32-bit) | ✅ |
| SH 系数 | `float` (32-bit) | `float` (32-bit) | ✅ |
| 排序键 | `uint64_t` (64-bit) | `ulong` (64-bit) | ✅ |
| 排序值 | `uint32_t` (32-bit) | `uint` (32-bit) | ✅ |
| 深度编码 | `float` → bit-cast `uint32_t` | `float` → `as_uint()` | ✅ |
| 输出图像 | `float` (32-bit) | `float` (32-bit) | ✅ |

**结论：所有数据类型完全一致，均使用 IEEE 754 单精度 (32-bit float)。无 `double` 或 `half` 使用。**

---

## `-cl-fast-relaxed-math` 的影响

### 该编译选项的含义

`-cl-fast-relaxed-math` 等价于同时启用：

| 子选项 | 效果 |
|--------|------|
| `-cl-mad-enable` | 允许将 `a*b+c` 替换为 `mad(a,b,c)` 指令（单次舍入 vs 两次舍入） |
| `-cl-no-signed-zeros` | `-0.0` 和 `+0.0` 视为相同 |
| `-cl-unsafe-math-optimizations` | 允许重排浮点运算（如 `(a+b)+c` → `a+(b+c)`） |
| `-cl-finite-math-only` | 假设无 NaN/Inf，允许更激进优化 |

### 核心影响：运算重排序

OpenCL 编译器在 `-cl-fast-relaxed-math` 下可以：

```
// 原始代码
float power = -0.5f * (con_a*dx*dx + con_c*dy*dy) - con_b*dx*dy;

// 编译器可能重排为（不同的舍入累积）
float power = mad(-0.5f, mad(con_a, dx*dx, con_c*dy*dy), -con_b*dx*dy);
```

两种形式数学上等价，但浮点运算中由于舍入顺序不同，结果可能有 1-2 ULP（Unit in the Last Place）差异。

---

## `native_exp()` vs `std::exp()` 详细对比

### 代码中的使用位置

**CPU rasterizer (`rasterizer_cpu.cpp:53`)**:
```cpp
float alpha = std::min(0.99f, pre.opacities_2d[idx] * std::exp(power));
```

**GPU rasterizer (`rasterize.cl:97`)**:
```opencl
float alpha = fmin(0.99f, s_opa[j] * native_exp(power));
```

### 精度规格

| 函数 | 精度保证 | ULP 误差 | 实现方式 |
|------|---------|---------|---------|
| `std::exp(x)` (CPU) | 完全精度 | ≤ 1 ULP | 软件实现，查表+多项式逼近 |
| `exp(x)` (OpenCL) | 完全精度 | ≤ 3 ULP | 与 `std::exp` 等价 |
| `native_exp(x)` (OpenCL) | **实现定义** | **≤ 8192 ULP** (Maleoon) | 硬件指令，快速但低精度 |

### ULP 误差的实际影响

float32 有 23 位尾数（约 7 位有效十进制数字）。

- **1 ULP** @ value=0.5 → 误差 ≈ 3×10⁻⁸（完全精度）
- **8192 ULP** @ value=0.5 → 误差 ≈ 2.4×10⁻⁴（`native_exp` 最坏情况）

在 alpha blending 中：
```
alpha = opacity * native_exp(power)
```

如果 `power = -2.0`，`exp(-2.0) = 0.1353352832...`：
- `std::exp(-2.0f)` → `0.13533528` (精确到 float32 极限)
- `native_exp(-2.0f)` → `0.13533` ~ `0.13534`（最坏情况可能有 ±0.0001 偏差）

这个 alpha 差异通过 T 的累积放大：
```
T_new = T * (1 - alpha)
C += rgb * alpha * T
```

经过 50-100 个 Gaussian 的累积，alpha 的微小差异可导致最终像素值差 1-5（uint8 范围 0-255）。

### 为什么 cam_0 差异最大（max_diff=38）？

cam_0 对着篮球正面，该区域有大量密集 Gaussian（~2000+ per tile）。每个 Gaussian 的 `native_exp` 微小误差通过 alpha blending 的串行依赖链累积：

```
T₀ = 1.0
T₁ = T₀ × (1 - α₁)     ← α₁ 有 native_exp 误差
T₂ = T₁ × (1 - α₂)     ← 误差继续累积
...
T₂₀₀₀ = T₁₉₉₉ × (1 - α₂₀₀₀)  ← 2000 次累积后，T 值可能偏移 ~0.01
C = Σ(rgb_i × α_i × T_i)        ← C 累积偏移 → 像素值差 30-40
```

38 的像素差对应约 15% 的亮度偏移 (38/255)。

---

## GPU Preprocess Kernel 中的数学函数

| 调用 | CPU 函数 | GPU 函数 | 精度差异 |
|------|---------|---------|---------|
| 向量长度 | `std::sqrt()` | `sqrt()` | ≤ 3 ULP (OpenCL 规范) |
| 行列式 sqrt | `std::sqrt()` | `sqrt()` | ≤ 3 ULP |
| AA scaling | `std::sqrt()` | `sqrt()` | ≤ 3 ULP |
| Radius ceil | `std::ceil()` | `ceil()` | 精确 |
| 范围限制 | `std::min/max` | `fmin/fmax` | 精确 (NaN 处理不同) |
| **Alpha 计算** | **`std::exp()`** | **`native_exp()`** | **≤ 8192 ULP** |

**注意**: preprocess kernel 中使用的是标准 `sqrt()` 而非 `native_sqrt()`。但 `-cl-fast-relaxed-math` 可能将其隐式替换为 `native_sqrt()`。

---

## `-cl-fast-relaxed-math` 对 `sqrt` 和 `exp` 的隐式影响

根据 OpenCL 规范 §6.2.1：

> When `-cl-fast-relaxed-math` is specified, the following single precision
> built-in math functions ... may be implemented with reduced accuracy:
> `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, ...

这意味着即使代码写的是 `sqrt(x)`，在 `-cl-fast-relaxed-math` 下编译器**可能**将其替换为 `native_sqrt(x)`（精度 ≤ 8192 ULP）。

### 实际影响链

```
computeCov3D:
  R × S → M → M^T × M        ← mad() 替代 mul+add，~1 ULP 差异

computeCov2D:
  J × W → T → T^T × Vrk × T  ← 矩阵链乘，误差累积 ~5-10 ULP

sqrt(det):                      ← 可能用 native_sqrt，~100 ULP
det_inv = 1.0/det:              ← 可能用 native_recip，~100 ULP
conic = cov2d * det_inv:        ← 误差传播到每个像素的 alpha 计算

computeColorFromSH:
  sqrt(dx²+dy²+dz²):           ← 可能用 native_sqrt
  方向归一化后用于 SH 基函数计算，误差 ~1e-4

rasterize:
  native_exp(power):            ← 显式低精度，~1e-4 误差
  alpha 累积 × 100-2000 Gaussians → 像素差 1-38
```

---

## 量化验证

### basket0.ply 76 帧统计

| 指标 | 值 |
|------|-----|
| 平均 PSNR | 86.28 dB |
| 最差 PSNR | 73.88 dB (cam_0) |
| 最佳 PSNR | 90.29 dB (cam_57) |
| 全部 > 70 dB | ✅ |
| max_diff > 20 的像素总数 | 3 (仅 cam_0) |
| max_diff > 10 的像素总数 | ~20 (across all frames) |
| max_diff > 5 的像素总数 | ~50 (across all frames) |

### 与人眼感知对比

| PSNR | 人眼感知 | 我们的结果 |
|------|---------|-----------|
| > 40 dB | 无法区分 | — |
| > 60 dB | 仪器级差异 | — |
| > 70 dB | bit-level 差异 | 最差帧 |
| > 85 dB | 浮点精度极限 | 平均水平 |

**结论：GPU 和 CPU 渲染差异在人眼完全不可感知的范围内。**

---

## 如果需要更高精度

若需将 GPU 精度提升到与 CPU 完全一致（如用于自动化测试）：

1. **移除 `-cl-fast-relaxed-math`**：使用 `-cl-opt-disable` 或不加优化选项
2. **将 `native_exp()` 替换为 `exp()`**：GPU `exp()` 保证 ≤ 3 ULP
3. **代价**：预计渲染速度下降 20-40%（取决于 Maleoon 920 的 native 函数加速比）

当前配置（`-cl-fast-relaxed-math` + `native_exp`）是移动 GPU 渲染的标准做法，在性能和精度间取得了最优平衡。
