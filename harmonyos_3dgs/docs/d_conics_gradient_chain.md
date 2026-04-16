# d_conics 梯度传递链路文档

> 文档版本：1.0  
> 最后更新：2026-03-31  
> 涵盖模块：Rasterizer Backward → Preprocessor Backward

---

## 执行摘要

本文档详细追踪了梯度 `d_conics` 从损失函数到原始 Gaussian 参数的完整传递链路。`d_conics` 是连接**屏幕空间光栅化**与**几何参数优化**的关键桥梁。

**核心梯度链路**:

```
d_image (像素颜色梯度)
    │
    │ [Rasterizer Backward]
    │ d_conics = d_power * [-0.5dx², -dx*dy, -0.5dy²]
    ▼
d_conics [N*3]                    # Conic 矩阵梯度 (屏幕空间)
    │
    │ [Preprocessor Backward - Chain 1]
    │ d_cov2D = f(d_conics, det, a, b, c)
    ▼
d_cov2D [3]                       # 2D 协方差梯度
    │
    │ d_cov3D = T^T · d_cov2D · T
    ▼
d_cov3D [6]                       # 3D 协方差梯度
    │
    │ d_M = g(d_cov3D, M)
    ▼
d_raw_scales, d_raw_rotations     # 最终参数梯度
```

---

## 目录

1. [梯度源头：损失函数](#1-梯度源头损失函数)
2. [Rasterizer Backward: d_image → d_conics](#2-rasterizer-backward-d_image-d_conics)
3. [Preprocessor Backward: d_conics → d_cov2D](#3-preprocessor-backward-d_conics-d_cov2d)
4. [Cov2D → Cov3D 投影反向](#4-cov2d-cov3d-投影反向)
5. [Cov3D → Scale/Rotation 分解](#5-cov3d-scalerotation-分解)
6. [完整数据流图](#6-完整数据流图)
7. [代码位置索引](#7-代码位置索引)

---

## 1. 梯度源头：损失函数

### 1.1 L1 Loss

训练使用 L1 损失函数：

$$L = \frac{1}{n} \sum_{i=1}^{n} |C_{rendered}^{(i)} - C_{gt}^{(i)}|$$

其中 $C$ 是像素颜色 (RGB 三通道)，$n$ 是像素总数。

### 1.2 损失梯度

对像素颜色的梯度：

$$\frac{\partial L}{\partial C^{(i)}} = \frac{1}{n} \cdot \text{sign}(C_{rendered}^{(i)} - C_{gt}^{(i)})$$

**代码位置**: `harmonyos_3dgs/src/trainer.cpp` (第 52 行)

```cpp
// 计算 L1 损失和梯度
std::vector<float> d_image(n_pixels * 3);
for (int i = 0; i < n_pixels * 3; i++) {
    float diff = rendered[i] - gt_image[i];
    loss += std::abs(diff);
    d_image[i] = std::copysign(1.0f / n_pixels, diff);  // dL/dC
}
```

---

## 2. Rasterizer Backward: d_image → d_conics

### 2.1 函数签名

**文件**: `harmonyos_3dgs/src/cpu/rasterizer_backward_cpu.cpp`

```cpp
void RasterizerBackwardCPU::backward(
    const PreprocessOutput& pre,
    const BinningOutput& bin,
    const Camera& cam,
    const RenderConfig& cfg,
    const ForwardCache& cache,
    const float* d_image,          // 输入：像素颜色梯度 [H*W*3]
    RasterGradOutput& rgrad);      // 输出：rgrad.d_conics [N*3]
```

### 2.2 体渲染公式 (正向)

像素颜色由 front-to-back alpha 混合得到：

$$C = \sum_{i=1}^{N} c_i \cdot \alpha_i \cdot T_i$$

$$T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$$

其中：
- $c_i$ 是第 $i$ 个高斯的 RGB 颜色
- $\alpha_i = \text{opacity}_i \cdot \exp(\text{power}_i)$
- $\text{power}_i = -\frac{1}{2}(\text{conic}_0 \cdot dx^2 + \text{conic}_2 \cdot dy^2) - \text{conic}_1 \cdot dx \cdot dy$

### 2.3 反向传播推导

#### Step 1: d_C → d_alpha

对第 $i$ 个高斯的透明度求导：

$$\frac{\partial L}{\partial \alpha_i} = T_i \cdot (c_i - C_{behind}) \cdot \frac{\partial L}{\partial C}$$

其中 $C_{behind}$ 是第 $i$ 个高斯后面所有层的累积颜色。

**代码**: `rasterizer_backward_cpu.cpp` (101-103 行)

```cpp
float d_alpha = 0.0f;
for (int ch = 0; ch < 3; ch++)
    d_alpha += T_i * (pre.rgb[gauss_idx*3+ch] - accum_rec[ch]) * d_C[ch];
```

#### Step 2: d_alpha → d_power

由于 $\alpha = \text{opacity} \cdot \exp(\text{power})$：

$$\frac{\partial L}{\partial \text{power}} = \frac{\partial L}{\partial \alpha} \cdot \text{opacity} \cdot \exp(\text{power})$$

**代码**: (116 行)

```cpp
d_power = d_alpha * pre.opacities_2d[gauss_idx] * exp_power;
```

#### Step 3: d_power → d_conics

由于 $\text{power} = -\frac{1}{2}\text{conic}_0 \cdot dx^2 - \text{conic}_1 \cdot dx \cdot dy - \frac{1}{2}\text{conic}_2 \cdot dy^2$：

$$\frac{\partial L}{\partial \text{conic}_0} = \frac{\partial L}{\partial \text{power}} \cdot \left(-\frac{1}{2} dx^2\right)$$

$$\frac{\partial L}{\partial \text{conic}_1} = \frac{\partial L}{\partial \text{power}} \cdot (-dx \cdot dy)$$

$$\frac{\partial L}{\partial \text{conic}_2} = \frac{\partial L}{\partial \text{power}} \cdot \left(-\frac{1}{2} dy^2\right)$$

**代码**: (129-132 行)

```cpp
// Chain: power -> conics
rgrad.d_conics[gauss_idx*3]     += d_power * (-0.5f * dx * dx);
rgrad.d_conics[gauss_idx*3 + 1] += d_power * (-dx * dy);
rgrad.d_conics[gauss_idx*3 + 2] += d_power * (-0.5f * dy * dy);
```

### 2.4 输出汇总

**RasterGradOutput** 包含：

| 字段 | 形状 | 含义 |
|------|------|------|
| `d_rgb` | [N*3] | 颜色梯度 |
| `d_opacities_2d` | [N] | 透明度梯度 |
| `d_means2D` | [N*2] | 屏幕空间均值梯度 |
| `d_conics` | [N*3] | **Conic 矩阵梯度** ← 传递给 Preprocessor |

---

## 3. Preprocessor Backward: d_conics → d_cov2D

### 3.1 函数签名

**文件**: `harmonyos_3dgs/src/cpu/preprocessor_backward_cpu.cpp`

```cpp
void PreprocessorBackwardCPU::backward(
    const GaussianData& g,
    const Camera& cam,
    const RenderConfig& cfg,
    const ForwardCache& cache,
    const RasterGradOutput& rgrad,  // 输入：含 d_conics
    const RawGaussianParams& params,
    GradientOutput& grads);         // 输出：d_raw_*
```

### 3.2 读取 d_conics

**代码**: (267-269 行)

```cpp
float dc0 = rgrad.d_conics[i*3];    // d_loss/d(conic[0])
float dc1 = rgrad.d_conics[i*3+1];  // d_loss/d(conic[1])
float dc2 = rgrad.d_conics[i*3+2];  // d_loss/d(conic[2])
```

### 3.3 Conic 逆矩阵求导

正向公式：
$$\text{conic}[0] = \frac{c}{\det}, \quad \text{conic}[1] = \frac{-b}{\det}, \quad \text{conic}[2] = \frac{a}{\det}$$

其中 $\det = ac - b^2$，$a, b, c$ 是 cov2D 的上三角元素。

**代码**: (295-306 行)

```cpp
float inv_det2 = inv_det * inv_det;
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

### 3.4 输出 d_cov2D

$d_a, d_b, d_c$ 即为 2D 协方差矩阵的梯度：

```
d_cov2D = [d_a, d_b, d_c]  # 对应 cov2D[0], cov2D[1], cov2D[2]
```

---

## 4. Cov2D → Cov3D 投影反向

### 4.1 正向投影公式

3D 协方差通过透视投影变换为 2D：

$$\Sigma_{2D} = T^T \cdot \Sigma_{3D} \cdot T$$

其中 $T = W \cdot J$ 是 3×3 变换矩阵：
- $W$：视图旋转矩阵 (3×3)
- $J$：投影雅可比矩阵 (3×3)

### 4.2 反向传播

已知 $d_a, d_b, d_c$，求 $d\_Vrk$ (3D 协方差的中间表示)：

$$d\_Vrk[k][j] = d_a \cdot T[0][k] \cdot T[0][j] + d_b \cdot T[0][k] \cdot T[1][j] + d_c \cdot T[1][k] \cdot T[1][j]$$

**代码**: (363-369 行)

```cpp
float d_Vrk[3][3];
for (int k = 0; k < 3; k++)
    for (int j = 0; j < 3; j++) {
        d_Vrk[k][j] = d_a * T[0][k] * T[0][j]
                     + d_b * T[0][k] * T[1][j]
                     + d_c * T[1][k] * T[1][j];
    }
```

### 4.3 映射到上三角表示

3D 协方差使用上三角 6 元素存储：

```
cov3D[0] = Vrk[0][0]
cov3D[1] = Vrk[0][1] = Vrk[1][0]
cov3D[2] = Vrk[0][2] = Vrk[2][0]
cov3D[3] = Vrk[1][1]
cov3D[4] = Vrk[1][2] = Vrk[2][1]
cov3D[5] = Vrk[2][2]
```

**代码**: (374-380 行)

```cpp
d_cov3D[0] = d_Vrk[0][0];
d_cov3D[1] = d_Vrk[0][1] + d_Vrk[1][0];  // 对称累加
d_cov3D[2] = d_Vrk[0][2] + d_Vrk[2][0];
d_cov3D[3] = d_Vrk[1][1];
d_cov3D[4] = d_Vrk[1][2] + d_Vrk[2][1];
d_cov3D[5] = d_Vrk[2][2];
```

---

## 5. Cov3D → Scale/Rotation 分解

### 5.1 正向分解公式

3D 协方差由缩放和旋转构造：

$$\Sigma_{3D} = M^T \cdot M$$

其中 $M = S \cdot R$：
- $S = \text{diag}(s_x, s_y, s_z)$ 缩放矩阵
- $R$：旋转矩阵 (由四元数构造)

### 5.2 d_cov3D → d_M

$$\frac{\partial L}{\partial M[k][0]} = 2 \cdot d\_cov3D[0] \cdot M[k][0] + d\_cov3D[1] \cdot M[k][1] + d\_cov3D[2] \cdot M[k][2]$$

$$\frac{\partial L}{\partial M[k][1]} = d\_cov3D[1] \cdot M[k][0] + 2 \cdot d\_cov3D[3] \cdot M[k][1] + d\_cov3D[4] \cdot M[k][2]$$

$$\frac{\partial L}{\partial M[k][2]} = d\_cov3D[2] \cdot M[k][0] + d\_cov3D[4] \cdot M[k][1] + 2 \cdot d\_cov3D[5] \cdot M[k][2]$$

**代码**: (416-421 行)

```cpp
float d_M[3][3];
for (int k = 0; k < 3; k++) {
    d_M[k][0] = 2.0f*d_cov3D[0]*M[k][0] + d_cov3D[1]*M[k][1] + d_cov3D[2]*M[k][2];
    d_M[k][1] = d_cov3D[1]*M[k][0] + 2.0f*d_cov3D[3]*M[k][1] + d_cov3D[4]*M[k][2];
    d_M[k][2] = d_cov3D[2]*M[k][0] + d_cov3D[4]*M[k][1] + 2.0f*d_cov3D[5]*M[k][2];
}
```

### 5.3 d_M → d_scale

由于 $M[k][j] = s_k \cdot R[k][j]$：

$$\frac{\partial L}{\partial s_k} = \sum_{j=0}^2 \frac{\partial L}{\partial M[k][j]} \cdot R[k][j]$$

**代码**: (427-435 行)

```cpp
float d_scale[3];
float d_R[3][3];
for (int k = 0; k < 3; k++) {
    d_scale[k] = 0.0f;
    for (int j = 0; j < 3; j++) {
        d_scale[k] += d_M[k][j] * R[k][j];
        d_R[k][j] = d_M[k][j] * s[k];
    }
}
```

### 5.4 d_scale → d_raw_scale

由于使用了 exp 激活：$s = \exp(s_{raw})$：

$$\frac{\partial L}{\partial s_{raw}} = \frac{\partial L}{\partial s} \cdot s$$

**代码**: (440-442 行)

```cpp
for (int k = 0; k < 3; k++) {
    grads.d_raw_scales[i*3+k] = d_scale[k] * cfg.scale_modifier * g.scales[i*3+k];
}
```

### 5.5 d_R → d_raw_rotation

通过四元数归一化链式法则传递梯度。

**代码**: (444-519 行)

```cpp
// 1. d_R -> d_normalized_quat (455-503 行)
float d_qn[4] = {0, 0, 0, 0};
d_qn[0] += d_R[0][1] * 2.0f*z_q;  // ∂R[0][1]/∂r = 2z
// ... 其他项

// 2. d_normalized_quat -> d_raw_quat (505-519 行)
float dot_dqn_qn = d_qn[0]*r_q + d_qn[1]*x_q + d_qn[2]*y_q + d_qn[3]*z_q;
grads.d_raw_rotations[i*4+0] = (d_qn[0] - r_q * dot_dqn_qn) * inv_len;
// ... 其他分量
```

---

## 6. 完整数据流图

### 6.1 梯度传递链路

```
┌──────────────────────────────────────────────────────────────────┐
│                        FORWARD PASS                              │
├──────────────────────────────────────────────────────────────────┤
│ RawGaussianParams                                                │
│   - raw_positions [N*3]                                          │
│   - raw_scales [N*3]                                             │
│   - raw_rotations [N*4]                                          │
│   - raw_sh_coeffs [N*max_coeffs*3]                               │
│   - raw_opacities [N]                                            │
│        │                                                         │
│        │ activate()                                              │
│        ▼                                                         │
│ GaussianData                                                     │
│        │                                                         │
│        │ Preprocess                                              │
│        ▼                                                         │
│ conics [N*3] ──────────────────────────────────┐                │
│ means2D [N*2]                                  │                │
│ rgb [N*3]                                      │                │
│ opacities_2d [N]                               │                │
│        │                                       │                │
│        │ Rasterize                             │                │
│        ▼                                       │                │
│ rendered_image [H*W*3]                         │                │
└──────────────────────────────────────────────────────────────────┘
    │
    │ L1 Loss
    ▼
d_image [H*W*3]
    │
    │ [Rasterizer Backward]                                        │
    │ d_C ──► d_alpha ──► d_power ──► d_conics                    │
    ▼                                                              │
┌──────────────────────────────────────────────────────────────────┐
│ RasterGradOutput                                                 │
│   - d_conics [N*3]         ◄─────────────────────────────────────┤
│   - d_means2D [N*2]                                              │
│   - d_rgb [N*3]                                                  │
│   - d_opacities_2d [N]                                           │
└──────────────────────────────────────────────────────────────────┘
    │
    │ [Preprocessor Backward - Chain 1: Covariance]
    │ d_conics ──► d_cov2D ──► d_cov3D ──► d_M ──► d_scale, d_R
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ GradientOutput                                                   │
│   - d_raw_scales [N*3]                                           │
│   - d_raw_rotations [N*4]                                        │
│   - d_raw_positions [N*3]     (from Chain 4: d_means2D)          │
│   - d_raw_sh_coeffs [N*max_coeffs*3]  (from Chain 2: d_rgb)      │
│   - d_raw_opacities [N]       (from Chain 3: d_opacity)          │
└──────────────────────────────────────────────────────────────────┘
    │
    │ Optimizer.step()
    ▼
RawGaussianParams (更新)
```

### 6.2 关键中间变量表

| 变量 | 形状 | 来源 | 用途 |
|------|------|------|------|
| `d_image` | [H*W*3] | L1 Loss | 像素颜色梯度 |
| `d_alpha` | 标量/像素 | 体渲染反向 | 透明度梯度 |
| `d_power` | 标量/像素 | alpha = opacity*exp | 指数幂梯度 |
| **`d_conics`** | **[N*3]** | **Rasterizer** | **Conic 矩阵梯度** |
| `d_a, d_b, d_c` | 标量/Gaussian | Conic 逆矩阵 | 2D 协方差梯度 |
| `d_cov3D` | [6] | 投影反向 | 3D 协方差梯度 |
| `d_M` | [3][3] | Cov3D 分解 | 缩放旋转矩阵梯度 |
| `d_scale` | [3] | M = S*R | 缩放梯度 |
| `d_raw_scale` | [3] | exp 激活反向 | 原始缩放梯度 |

---

## 7. 代码位置索引

### 7.1 源文件

| 文件 | 功能 | 行号范围 |
|------|------|----------|
| `src/trainer.cpp` | 训练步进，损失计算 | 50-70 |
| `src/cpu/rasterizer_backward_cpu.cpp` | Rasterizer 反向 | 129-132 (d_conics) |
| `src/cpu/preprocessor_backward_cpu.cpp` | Preprocessor 反向 | 267-520 (Chain 1) |
| `src/gpu/rasterizer_backward_gpu.cpp` | GPU Rasterizer 反向 | - |
| `src/gpu/preprocessor_backward_gpu.cpp` | GPU Preprocessor 反向 | - |

### 7.2 头文件

| 文件 | 定义 |
|------|------|
| `include/train_types.h` | RasterGradOutput, GradientOutput |
| `include/types.h` | GaussianData, PreprocessOutput |
| `src/gpu/kernels/rasterize_backward.cl` | OpenCL Rasterizer 反向 kernel |
| `src/gpu/kernels/preprocess_backward.cl` | OpenCL Preprocessor 反向 kernel |

### 7.3 测试文件

| 文件 | 测试内容 |
|------|----------|
| `tests/test_rasterizer_backward.cpp` | Rasterizer 反向验证 |
| `tests/test_preprocessor_backward.cpp` | Preprocessor 反向验证 |
| `tests/test_train_pipeline.cpp` | 完整训练管线验证 |

---

## 8. 总结

`d_conics` 梯度是 3D 高斯泼溅训练中**屏幕空间与参数空间的桥梁**：

1. **生成端 (Rasterizer Backward)**: 从像素颜色梯度通过体渲染公式和 power 的链式法则生成
2. **消费端 (Preprocessor Backward)**: 通过 conic 逆矩阵求导转换为 2D 协方差梯度，再经投影反向传播到 3D 几何参数

理解这个梯度链路对于调试训练问题、实现自定义损失函数或优化反向传播性能至关重要。
