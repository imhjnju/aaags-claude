# 3DGS Antialiasing (Mip-Splatting) 分析

## 算法原理

3DGS 的 antialiasing 来自 **Mip-Splatting**（Yu et al., 2024）论文，核心思想是对 2D 投影高斯施加一个低通滤波器，防止高频 Gaussian 在远距离时产生锯齿。

### 实现代码

```cpp
// preprocessor_cpu.cpp, Step 6

// 原始 2D 协方差行列式（投影后、加滤波器前）
float det_cov = cov2d[0] * cov2d[2] - cov2d[1] * cov2d[1];

// 加 0.3 低通滤波器到对角线（等效于与 σ²=0.3 的高斯核卷积）
cov2d[0] += 0.3f;
cov2d[2] += 0.3f;

// 滤波后的行列式
float det_cov_plus_h = cov2d[0] * cov2d[2] - cov2d[1] * cov2d[1];

// Opacity 缩放因子
float h_conv_scaling = 1.0f;
if (antialiasing)
    h_conv_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h));

// 最终 opacity = 原始 opacity × h_conv_scaling
out.opacities_2d[i] = opacities[i] * h_conv_scaling;
```

### 数学解释

设原始 2D 协方差矩阵为 Σ，滤波核为 H = 0.3·I（各向同性高斯）。

- **滤波后协方差**: Σ' = Σ + H
- **Opacity 缩放**: `h = sqrt(det(Σ) / det(Σ + H))`

这个缩放因子保证了滤波前后 Gaussian 的积分（总能量）不变。当 Gaussian 在屏幕上很小时（det(Σ) 很小），h 接近 0，有效地降低了该 Gaussian 的不透明度。当 Gaussian 足够大时（det(Σ) >> det(H)），h ≈ 1，不影响原始不透明度。

**物理直觉**: 小的 Gaussian 在远处投影为亚像素级别，本应被"模糊"掉。低通滤波器将它们的尺寸膨胀到至少覆盖一个像素，同时降低 opacity 保持能量守恒，避免闪烁/锯齿。

---

## 定量对比

### 测试条件
- 模型: basket0.ply (784K Gaussians), basketball.ply (400K Gaussians)
- 相机: cameras.json id 0, 720×960
- 背景: 黑色
- 平台: PC x86, `-ffp-contract=off`

### basket0.ply 对比

| 指标 | AA OFF | AA ON | 变化 |
|------|--------|-------|------|
| 渲染时间 | 12,811 ms | 15,449 ms | **+20.6%** |
| 平均亮度 | 140.79 | 143.54 | +2.76 (+2.0%) |
| 边缘能量 (Sobel) | 39.78 | 39.01 | -1.9% |
| 白色像素 (>240) | 53 | **1,407** | **+2555%** |
| 平均饱和度 | 0.2017 | 0.1837 | -8.9% |
| PSNR (OFF vs ON) | — | — | **20.19 dB** |
| 最大像素差 | — | — | **248** |

### basketball.ply 对比

| 指标 | AA OFF | AA ON | 变化 |
|------|--------|-------|------|
| 渲染时间 | 31,479 ms | 33,781 ms | **+7.3%** |
| 平均亮度 | 144.15 | 142.68 | -1.47 (-1.0%) |
| 边缘能量 (Sobel) | 35.07 | 30.18 | **-13.9%** |
| 白色像素 (>240) | 15 | 11 | -26.7% |
| 平均饱和度 | 0.1795 | 0.1846 | +2.8% |
| PSNR (OFF vs ON) | — | — | **28.30 dB** |
| 最大像素差 | — | — | **171** |

---

## 视觉影响分析

### 1. 边缘平滑效果

**AA ON 减少了 13.9% 的边缘能量**（basketball），这意味着物体轮廓更平滑、锯齿感更少。从 diff map 可以看到，差异主要集中在物体边缘和纹理细节区域。

### 2. 白色斑块影响

AA ON 对 basket0 的白色像素**大幅增加**（53 → 1407）。这是因为：
- AA 的 `h_conv_scaling` 降低了小 Gaussian 的 opacity
- 原本覆盖较好的区域，opacity 降低后露出更多背景
- basket0 模型本身有大量超亮 Gaussian（DC > 1.0），AA 降低 opacity 后这些亮点占比更高

basketball 模型质量更好，AA 反而略微减少白色像素（15 → 11），因为平滑效果改善了覆盖。

### 3. 饱和度变化

basket0 开启 AA 后饱和度**下降 8.9%**。AA 通过降低 opacity 并膨胀 Gaussian，使相邻 Gaussian 的颜色更多地混合在一起，降低了颜色对比度。

basketball 饱和度略微上升（+2.8%），因为平滑效果去除了一些噪声，让真实颜色更突出。

### 4. 渲染性能

AA ON 增加 7-20% 的渲染时间。额外开销来自：
- `sqrt()` 和除法计算 `h_conv_scaling`
- 0.3f 加到协方差对角线后，Gaussian 椭圆变大，覆盖更多 tile
- 更多 tile-Gaussian pairs 需要排序和光栅化

---

## 何时开启/关闭 AA

| 场景 | 推荐 | 理由 |
|------|------|------|
| **高质量离线渲染** | ON | 减少边缘锯齿，输出更平滑 |
| **实时交互（60 FPS）** | OFF | 省 7-20% 性能，优先帧率 |
| **模型质量差（多超亮 Gaussian）** | OFF | AA 会恶化白色斑块问题 |
| **模型质量好（训练充分）** | ON | 纯粹改善视觉质量 |
| **远距离视角** | ON | 远处小 Gaussian 最需要 AA |
| **近距离视角** | 影响小 | 近处 Gaussian 投影大，AA 几乎不生效 |

---

## 对 0.3 滤波系数的理解

`cov2d[0] += 0.3f` 中的 0.3 不是任意选择：

- 它对应约 **0.55 像素的标准差** (√0.3 ≈ 0.548)
- 这意味着滤波器将任何 Gaussian 的最小有效尺寸扩展到至少约 0.55 像素
- 对于已经大于 2-3 像素的 Gaussian，0.3 的影响可忽略（det(Σ) >> 0.3）
- 对于亚像素 Gaussian（det(Σ) ≈ 0），h_conv_scaling → 0，有效隐藏它们

这个值是 Mip-Splatting 论文中的实验最优值，平衡了抗锯齿效果和细节保留。

---

## Diff Map 分析

差异放大 5 倍的 diff map 显示：

- **basket0**: 差异集中在篮球表面和边缘。球体上的高频纹理受 AA 影响最大，因为这些区域有大量小尺寸、高对比度的 Gaussian。背景办公室区域差异较小。

- **basketball**: 差异主要在球体边缘轮廓处。ANTA 字样区域有轻微差异。整体差异比 basket0 小得多（PSNR 28 vs 20 dB），说明 basketball 模型的 Gaussian 尺寸分布更合理。
