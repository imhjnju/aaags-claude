# AAA-Gaussians 算法移植文档

**论文**: [AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering](https://arxiv.org/abs/2504.12811) (arXiv:2504.12811, Graz University of Technology, 2025)

## 概述

AAA-Gaussians 在标准 3DGS 的基础上引入了 **全3D高斯评估** (eval_3D) 模式。论文的核心贡献分为三大类：
1. **透视正确的 3D 评估** (§3.1): 屏幕空间平面求交 + 分层排序
2. **自适应 3D 抗锯齿** (§3.2): 垂直子空间 Mip 滤波 + 训练时采样频率追踪
3. **稳定的包围与裁剪** (§3.3-3.4): 视空间包围盒 + 3D 视锥体逐Tile裁剪

本文档详细说明每个算法与移植代码的对应关系，并标注移植状态。

---

## 1. 数据结构扩展

### 1.1 filter_3D 属性

**算法**: AAA-Gaussians 在训练时为每个 Gaussian 计算一个 3D Mip 滤波系数 `filter_3D`，基于该 Gaussian 到所有训练相机的最小深度距离。

**AAA-GS 源码**: `AAA-Gaussians/utils/mip_filter.py` → `mip_filter_3d()`
```python
filter_3D = distance / focal_length * (0.3 ** 0.5)
```

**移植代码**:
- `include/types.h:44` — `GaussianData` 新增 `float* filter_3D` 字段
- `src/ply_loader.cpp:108-110` — 检测 PLY 文件中的 `filter_3D` 属性
- `src/ply_loader.cpp:127` — 按需分配内存
- `src/ply_loader.cpp:176-177` — 读取并存储（无需激活，直接存储原始值）

### 1.2 eval_3D 配置开关

**移植代码**:
- `include/types.h:61` — `RenderConfig` 新增 `bool eval_3D` 字段
- `src/main.cpp:370-376` — 自动检测（PLY含filter_3D则开启），可通过环境变量 `EVAL_3D=0/1` 覆盖

### 1.3 Gauss2Screen 矩阵输出

**移植代码**:
- `include/types.h:76-77` — `PreprocessOutput` 新增 `float* gauss2screen` (N×16) 和 `bool eval_3D`
- `src/gpu/preprocessor_gpu.h:20` — `PreprocessDeviceBuffers` 新增 `cl_mem gauss2screen`

---

## 2. 3D Mip 滤波与尺度膨胀 (Scale Dilation)

### 算法原理（论文 §3.2 "3D Gaussian Anti-Aliasing"）

标准 3DGS 使用固定的 Gaussian 尺度，远距离观察时会出现锯齿/走样。AAA-Gaussians 根据观察距离动态膨胀 Gaussian 尺度。

**关键创新 — 垂直子空间振幅缩放**：膨胀尺度后，如果简单按 3D 体积比缩放振幅，Gaussian 会过度透明。论文提出只考虑**视线方向垂直子空间**的协方差变化来缩放振幅：

```
Ĝ⊥(x) = √(|Σ⊥| / |Σ̂⊥|) × exp(-½(x-μ)ᵀ Σ̂⁻¹ (x-μ))
```

其中 Σ⊥ 是投影到视线垂直平面上的 2×2 协方差。实际计算中等价于：

```
dilation_factor = √( |Σ| × dᵀΣ⁻¹d  /  |Σ̂| × dᵀΣ̂⁻¹d )
```

d 是归一化的观察方向（从相机到 Gaussian 中心）。在代码中展开为：

```
viewray_gauss = R × normalize(mean3D - cam_pos)     // 视线在Gaussian局部空间中的方向
r = viewray_gauss ⊙ viewray_gauss                    // 逐元素平方
det_mul_ray_var     = dot(r, (s_y×s_z, s_z×s_x, s_x×s_y))  // 原始
det_mul_ray_var_dil = dot(r, (ŝ_y×ŝ_z, ŝ_z×ŝ_x, ŝ_x×ŝ_y))  // 膨胀后
dilation_factor = √(det_mul_ray_var / det_mul_ray_var_dil)
```

**尺度膨胀公式**：

```
v̂ = focal / z_view                     // 当前采样频率
v̂_train = focal_train / z_min_train    // 训练时最大采样频率 (存储为 filter_3D)
v̂' = min(v̂_train, v̂)                  // 有效采样频率 — 防止近距离过度收缩

scale_mip_filter = (z_view / focal)² × kernel_size    // 基于距离的滤波
scale_mip_filter = max(filter_3D², scale_mip_filter)   // 取训练下限
scale_dilated = sqrt(scale² + scale_mip_filter)
```

> **`filter_3D` 的含义**: `filter_3D = z_min / focal_max × √0.3`，编码了训练时该 Gaussian 能被观察到的最高分辨率。`max(filter_3D², ...)` 确保渲染时的滤波不会超过训练时的精度。

### AAA-GS 源码

`consistent_common.cuh:28-51` → `compute_gauss2screen()` 中的 `if (kernel_size > 0.0f)` 块

### 移植代码

**CPU**: `src/math_utils.cpp:489-519` → `computeGauss2Screen()` 中的 mip filter 计算
```cpp
float scale_mip = sq(mean_view[2] / focal) * kernel_size;  // 距离自适应
scale_mip = std::max(sq(filter_3d), scale_mip);            // 取训练时的下限
scale_dilated[i] = sqrt(scale[i]*scale[i] + scale_mip);    // 膨胀尺度
```

**GPU**: `src/gpu/kernels/preprocess.cl:397-420` — 同样逻辑的 OpenCL 实现

---

## 3. Gauss-to-Screen 矩阵构建

### 算法原理

标准 3DGS 将 3D 协方差投影为 2D 协方差（conic），然后在逐像素阶段用 2D 二次型计算贡献。AAA-Gaussians 改为构建一个 4×4 **Gauss-to-Screen 变换矩阵**，在逐像素阶段直接进行 3D 评估。

构建过程：
```
L = transpose(S_dilated × R)           # Gaussian局部坐标轴（含尺度）
gauss2world = [L | mean3D; 0 0 0 1]    # 列主序 4x4
viewport = [W/2  0   0  W/2-0.5]       # NDC → 像素坐标
           [ 0  H/2  0  H/2-0.5]
           [ 0   0   1    0    ]
           [ 0   0   0    1    ]
gauss2screen = transpose(viewport × viewproj × gauss2world)  # 存储为行主序
```

### AAA-GS 源码

`consistent_common.cuh:12-78` → `compute_gauss2screen()`

### 移植代码

**CPU**: `src/math_utils.cpp:474-582` → `computeGauss2Screen()`
- L矩阵构建: L530-532
- gauss2world 构建: L539-546
- viewport 矩阵: L550-557
- 矩阵链乘: L561-565 (`viewport × viewproj × gauss2world`)
- 转置存储: L568-570

**GPU**: `src/gpu/kernels/preprocess.cl:425-460`

---

## 4. 椭球体内相机检测

### 算法原理

如果相机位于 Gaussian 椭球体内部，渲染结果不稳定，需要跳过该 Gaussian。通过将相机位置变换到 Gaussian 局部坐标系，检查其到原点的距离平方是否小于 cutoff。

```
world2gauss = S_inv × R
campos_gauss = world2gauss × (cam_pos - mean3D)
if (dot(campos_gauss, campos_gauss) < cutoff) → 跳过
```

### AAA-GS 源码

`forward.cu:165-167`

### 移植代码

**CPU**: `src/cpu/preprocessor_cpu.cpp:70-103`
```cpp
campos_gauss[j] = (R[j][0]*diff[0] + R[j][1]*diff[1] + R[j][2]*diff[2]) / sd[j];
if (dist_sq < cutoff) continue;  // 相机在椭球体内
```

**GPU**: `src/gpu/kernels/preprocess.cl:462-470`

---

## 5. 3D 视锥体裁剪

### 算法原理

在屏幕空间中检测 Gaussian 对整个屏幕区域的最大贡献。如果最大贡献超过阈值，说明 Gaussian 在屏幕中完全不可见，提前剔除。

通过构造屏幕边界的平面方程，找到 Gaussian 在视锥体中贡献最大的点：
1. 如果 Gaussian 中心在屏幕内 → max_contrib = 0
2. 否则测试5条候选线/面（最近x平面、最近y平面、4条边交线），取最小的 Mahalanobis 距离

### AAA-GS 源码

`consistent_common.cuh:130-207` → `max_contrib_gaussian_frustum_3D()`
- 辅助函数: `max_contrib_plane()` (L80-87), `max_contrib_ray()` (L89-106)

### 移植代码

**CPU**: `src/math_utils.cpp:253-340` → `maxContribGaussianFrustum3D()`
- 平面最大贡献: `maxContribPlane()` (L210-222)
- 射线最大贡献: `maxContribRay()` (L186-207)
- 屏幕空间射线: `maxContribRayScreen()` (L237-248)
- 范围检测: `inScreenRange()` (L225-234)

**CPU 预处理调用**: `src/cpu/preprocessor_cpu.cpp:107-111`

**GPU**: `src/gpu/kernels/preprocess.cl:152-186` (内联实现)

---

## 5b. 逐 Tile 3D 裁剪 (Per-Tile Frustum Culling)

### 算法原理（论文 §3.4 "Frustum-Based Culling"）

第5节的 `maxContribGaussianFrustum3D` 用于全局预处理阶段（将 Gaussian 对整个屏幕做裁剪）。论文还提出了 **逐 Tile 级别**的3D裁剪：在 tile binning 阶段，对每个 (Gaussian, Tile) 组合，用该 Tile 对应的 3D 视锥体进行更精确的裁剪。

**标准 3DGS**: Tile binning 仅检查 2D 矩形重叠，所有在 bounding rect 内的 tile 都被分配。
**AAA-Gaussians**: 对每个 tile 构造 3D frustum，调用 `maxContribGaussianFrustum3D(tile_min, tile_max, g2s, depth)` 判断 Gaussian 是否真的对该 tile 有贡献。

```
对每个 (Gaussian, Tile) 对:
  tile_frustum = {tile_min_x, tile_min_y, tile_max_x, tile_max_y}
  max_contrib = maxContribGaussianFrustum3D(tile_frustum, gauss2screen)
  if max_contrib > opacity_power_threshold:
    跳过该 tile（不生成 key-value 对）
```

**裁剪优化策略 (2-plane + 3-edge)**:

全视锥体有 4 个平面（上下左右）和 4 条边（4个角的交线），朴素检查需要 4+4=8 次 `maxContribPlane/maxContribRay` 调用。论文优化为只检查 **2 个最近平面 + 3 条相关边**：

1. 根据 Gaussian 中心在屏幕空间的位置，确定 x 方向和 y 方向各一个最近平面
2. 检查: closer_plane_x 单独、closer_plane_y 单独
3. 检查: closer_x ∩ closer_y、closer_x ∩ other_y、other_x ∩ closer_y（3条边）
4. 取所有候选中的最小 Mahalanobis 距离

### AAA-GS 源码

- Tile-based culling 调用: `stopthepop/stopthepop_common.cuh:166-286` → `computeTilebasedCullingTileCount()`
- 在 key generation 中集成: `stopthepop/stopthepop_common.cuh:333-687` → `duplicateWithKeys_extended()`

### 移植状态

> **全局级裁剪**: ✅ 已移植 (`maxContribGaussianFrustum3D`，用于预处理阶段)
>
> **逐 Tile 级裁剪**: ❌ 已实现但回退。`maxContribGaussianFrustum3D` 对低透明度远距离 Gaussian 的裁剪过于激进，导致整块 tile 内容丢失（非边界伪影，而是整 tile 缺失）。
>
> 根因分析：大量低透明度 Gaussian 单独低于阈值但累积贡献可见；即使放宽阈值至 3x 仍有残余伪影。需要更精细的精度调优或改用不同的裁剪策略。
>
> GPU scatter kernel 已预留接口（`scatter.cl` 中 `eval_3D`/`gauss2screen`/`opacities_2d` 参数），待精度问题解决后可直接启用。

---

## 5c. 视空间包围盒 (View-Space Bounding, new_aabb)

### 算法原理（论文 §3.3 "Perspective Correct Bounding"）

论文指出屏幕空间 AABB (Hahlbohm 方法，第6节) 在 Gaussian 延伸到**视锥体后方**时会失败，导致 popping 伪影。提出改用**视空间角度**进行包围盒计算：

```
1. 在视空间中用角度 (θ, φ) 参数化切平面:
   πθ = (cos(θ), 0, -sin(θ), 0)ᵀ
   πφ = (0, cos(φ), -sin(φ), 0)ᵀ

2. 求解二次相切条件:
   θ₁,₂ = arctan((s₁,₃ ± √(s₁,₃² - s₁,₁s₃,₃)) / s₃,₃)
   其中 s_ij = ⟨t, T_i ⊙ T_j⟩

3. 旋转确保角度以 Gaussian 中心为界:
   (θ_μ - π) < θ₁ < θ_μ < θ₂ < (θ_μ + π)

4. 映射回像素空间:
   bounds_x = W/2 + focal_x × tan(θ₁,₂)
```

**关键优势**: 当 Gaussian 跨越视锥体近平面时，屏幕空间方法返回 false（丢弃），而视空间方法能正确给出全屏范围。

### AAA-GS 源码

`consistent_common.cuh:217-284` → `compute_aabb_view()`

### 移植状态

> **已实现但未启用**。`src/math_utils.cpp:398-472` 实现了 `computeAABBView()`，但由于 atan2/tan 的数值稳定性问题，实际使用的是 `computeAABBScreen()` (Hahlbohm 方法) 配合 NDC 投影中心做 tile 分配。
>
> 在当前测试场景 (basketball.ply) 中，屏幕空间方法已能正确处理。如果未来遇到 Gaussian 延伸到视锥体后方的场景（如极近距离观察），可能需要切换回视空间方法。

---

## 6. 屏幕空间 AABB 包围盒

### 算法原理

基于 Hahlbohm et al. 提出的方法，从 gauss2screen 矩阵直接计算 Gaussian 在屏幕空间的轴对齐包围盒 (AABB)。用于确定 Gaussian 覆盖的 tile 范围。

```
t = (cutoff, cutoff, cutoff, -1)
s = dot(t, T[3] ⊙ T[3])         // ⊙ 为逐元素乘
f = t / s
p = (dot(f, T[0]⊙T[3]), dot(f, T[1]⊙T[3]), dot(f, T[2]⊙T[3]))
h = p² - (dot(f, T[0]⊙T[0]), dot(f, T[1]⊙T[1]), dot(f, T[2]⊙T[2]))
extent = sqrt(max(0, h))
```

> **移植注意**: AABB center 和 NDC 投影 center 存在偏差，tile 分配必须使用 NDC 投影 center（与标准 2D 路径一致），AABB extent 仅用于确定 radius 大小。否则会产生 tile 边界处的块状伪影。

### AAA-GS 源码

`consistent_common.cuh:286-323` → `compute_aabb_screen()`

### 移植代码

**CPU**: `src/math_utils.cpp:342-389` → `computeAABBScreen()`

**CPU 预处理调用**: `src/cpu/preprocessor_cpu.cpp:113-134`
- NDC 投影中心用于 tile 分配: L115-119
- AABB extent 用于 radius: L126-127
- `getRect()` 做 tile 覆盖: L131

**GPU**: `src/gpu/kernels/preprocess.cl:477-509` (内联实现)

---

## 7. 逐像素 3D 平面交线评估

### 算法原理

这是 eval_3D 最核心的改进。标准 3DGS 在每个像素用 **2D conic** (2D 协方差逆矩阵) 计算 Gaussian 贡献：
```
// 标准 2D 路径
power = -0.5 × (a×dx² + c×dy²) - b×dx×dy
```

AAA-Gaussians 改为使用 **3D 平面交线**：对每个像素，用 gauss2screen 矩阵构造两个平面方程（x平面和y平面），找到这两个平面交线上距离 Gaussian 中心最近的点，计算该点的 Mahalanobis 距离。

```
// eval_3D 路径
plane_x = g2s[row0] - g2s[row3] × pixel_x     // x方向约束平面
plane_y = g2s[row1] - g2s[row3] × pixel_y     // y方向约束平面
d = cross(plane_x.xyz, plane_y.xyz)            // 交线方向
m = plane_x.w × plane_y.xyz - plane_x.xyz × plane_y.w
power = -0.5 × dot(m, m) / dot(d, d)          // Mahalanobis距离
```

**像素坐标**: 使用像素中心 `(px + 0.5, py + 0.5)`，与 viewport 矩阵的 `-0.5` 偏移一致。

### AAA-GS 源码

`hierarchical_render.cuh:528-540` — per-pixel evaluation
`consistent_common.cuh:89-106` — `max_contrib_ray()`

### 移植代码

**CPU**: `src/cpu/rasterizer_cpu.cpp:47-63`
```cpp
plane_x[k] = g2s[0*4+k] - g2s[3*4+k] * fpx;  // fpx = px + 0.5
plane_y[k] = g2s[1*4+k] - g2s[3*4+k] * fpy;
power = -0.5f * maxContribRayPixel(plane_x, plane_y, max_pos);
```

`src/math_utils.cpp:584-588` → `maxContribRayPixel()` (调用内部 `maxContribRay`)

**GPU**: `src/gpu/kernels/rasterize.cl:108-125`
- 从 shared memory 加载 g2s row0, row1, row3: L79-92
- 构造平面方程: L109-116
- 计算交叉积和 Mahalanobis 距离: L118-124

---

## 8. 分层排序与逐像素重排序 (StopThePop Hierarchical Sorting)

### 算法原理

标准 3DGS 对所有 Gaussian 按**全局 view-space z** 排序，同一 Gaussian 在所有 tile 中使用相同的 depth key。这会导致 **popping artifact**：当相机移动时，两个重叠的 Gaussian 的前后关系在全局排序中突变，导致渲染结果跳变。

AAA-Gaussians 引入了 **StopThePop** 机制，包含两层改进：

#### 8.1 逐 Tile 深度计算 (Per-Tile Depth)

同一 Gaussian 在不同 tile 中使用**不同的 depth key**。深度沿像素的观察射线计算，而非使用 Gaussian 中心的全局 z 值：

```
// depthAlongRay: 计算 Gaussian 中心沿观察射线的投影深度
viewdir_inv_cov = inv_cov3D × viewdir       // 3x3 逆协方差 × 视线方向
num = mean_offset · viewdir                  // Gaussian中心偏移在视线上的投影
den = viewdir · (inv_cov3D × viewdir)        // 归一化因子
depth = num / den
```

每个 tile 计算深度时可选择不同的目标位置：
- **VIEWSPACE_Z**: 直接使用全局 z（退化为标准模式）
- **PER_TILE_DEPTH_CENTER**: 使用 tile 中心位置的观察射线
- **PER_TILE_DEPTH_MAXPOS**: 使用 Gaussian 在 tile 中最大贡献位置的观察射线

#### 8.2 三级分层排序 (Hierarchical Per-Pixel Resort)

光栅化阶段使用三级缓冲区对 Gaussian 进行**逐像素重排序**，而非依赖预排序结果：

```
┌─────────────────────────────────────────────────┐
│  TAIL (Per-4×4 Tile)                            │
│  __shared__ float tail_depths[64]               │
│  批量加载 32 个 Gaussian，Bitonic Sort 排序      │
│  ↓ 推送最小的 4 个到 MID                         │
├─────────────────────────────────────────────────┤
│  MID (Per-4×4 Group)                            │
│  __shared__ float mid_depths[8~12]              │
│  归并排序插入，从 TAIL 接收                       │
│  ↓ 推送最小的到 HEAD                             │
├─────────────────────────────────────────────────┤
│  HEAD (Per-Thread/Pixel)                        │
│  Register float head_depths[4]                  │
│  插入排序，窗口大小 4                             │
│  ↓ blend_one(): 弹出最前面的进行 alpha 混合      │
└─────────────────────────────────────────────────┘
```

**处理流程**:
1. 从全局排序列表中批量加载 32 个 Gaussian 到 TAIL
2. TAIL 使用 Bitonic Sort 排序
3. 将 TAIL 中最小的 4 个通过 MID 传递到 HEAD
4. HEAD 满时弹出最前面的元素执行 alpha 混合
5. 重复直到所有 Gaussian 处理完毕
6. 依次排空 MID → HEAD，混合剩余元素

**核心排序算法**:
- `batcherSort()`: 32 元素 Bitonic Sort（用于 TAIL 层）
- `mergeSortRegToSmem()`: 有序插入（用于 MID 层接收）
- 插入排序: 4 元素窗口（用于 HEAD 层）

### AAA-GS 源码

| 功能 | 文件 | 行号 |
|------|------|------|
| 逐像素深度计算 | `stopthepop/stopthepop_common.cuh` | L34-64 `depthAlongRay()` |
| 逐 Tile depth key | `stopthepop/stopthepop_common.cuh` | L464-500 `tile_function` lambda |
| Tile-based culling | `stopthepop/stopthepop_common.cuh` | L166-286 `computeTilebasedCullingTileCount()` |
| 三级分层渲染主函数 | `stopthepop/hierarchical_render.cuh` | L207-1048 `sortGaussiansRayHierarchicaEvaluation()` |
| HEAD 层插入排序 | `stopthepop/hierarchical_render.cuh` | L567-577 |
| MID 层归并插入 | `stopthepop/hierarchical_render.cuh` | L24-70 `mergeSortRegToSmem()` |
| TAIL 层 Bitonic Sort | `stopthepop/hierarchical_render.cuh` | L158-192 `batcherSort()` |
| 混合弹出 | `stopthepop/hierarchical_render.cuh` | L319-420 `blend_one()` |
| 逐像素 k-buffer 排序 | `stopthepop/resorted_render.cuh` | L17-221 `renderkBufferCUDA()` |
| 全块 Radix Sort | `stopthepop/resorted_render.cuh` | L474-675 `renderSortedFullCUDA()` |

### 移植状态

> **当前未移植**。当前移植版本使用标准 3DGS 的全局 depth 排序（view-space z），在大多数场景下视觉效果可接受。
>
> 分层排序依赖 CUDA 的 warp-level 原语 (`__shfl_sync`, `__ballot_sync`)、CUB 库的 `BlockRadixSort`、以及 3D thread block 组织，这些在 OpenCL 中没有直接对应，需要重新设计：
>
> | CUDA 特性 | OpenCL 替代方案 |
> |-----------|-----------------|
> | `__shfl_sync` | `sub_group_shuffle` (需要 cl_khr_subgroups) |
> | `__ballot_sync` | `sub_group_ballot` |
> | CUB `BlockRadixSort` | 手写 local memory radix sort |
> | 3D thread blocks (16×4×4) | 扁平化为 1D work-group + 手动索引 |
>
> **建议移植优先级**: 若 popping artifact 在目标场景中不明显，可暂不移植。若需移植，建议先实现 Per-Tile Depth（预处理阶段改动较小），再按需实现分层排序。

---

## 9. 文件修改总览

| 文件 | 修改内容 | 新增行数 |
|------|----------|----------|
| `include/types.h` | GaussianData加filter_3D，RenderConfig加eval_3D，PreprocessOutput加gauss2screen | +4 |
| `include/math_utils.h` | 声明AAA-GS数学函数 | +38 |
| `src/math_utils.cpp` | 实现quat2mat, computeGauss2Screen, computeAABBScreen, maxContribGaussianFrustum3D, maxContribRayPixel等 | +443 |
| `src/ply_loader.cpp` | 加载filter_3D属性 | +12 |
| `src/main.cpp` | 自动检测eval_3D，EVAL_3D环境变量 | +15 |
| `src/cpu/preprocessor_cpu.cpp` | eval_3D预处理路径（完整流水线） | +100 |
| `src/cpu/rasterizer_cpu.cpp` | eval_3D逐像素评估路径 | +20 |
| `src/gpu/kernels/preprocess.cl` | GPU eval_3D预处理 | +200 |
| `src/gpu/kernels/rasterize.cl` | GPU eval_3D光栅化 | +50 |
| `src/gpu/preprocessor_gpu.h` | 加filter_3D和gauss2screen设备缓冲 | +2 |
| `src/gpu/preprocessor_gpu.cpp` | 上传filter_3D，传递eval_3D kernel参数 | +16 |
| `src/gpu/rasterizer_gpu.cpp` | 传递eval_3D和gauss2screen参数 | +5 |

---

## 9. 渲染流水线对比

### 标准 3DGS 流水线 (eval_3D=OFF)

```
对每个 Gaussian:
  1. 近平面裁剪 (z > 0.2)
  2. NDC投影 → 2D像素坐标
  3. computeCov3D (quaternion+scale → 3D协方差)
  4. computeCov2D (3D协方差 → 2D屏幕协方差)
  5. 抗锯齿 (+0.3对角线)
  6. 求逆 → conic (a, b, c)
  7. 特征值 → 屏幕半径 → tile覆盖
  8. SH颜色评估

对每个像素:
  power = -0.5*(a*dx² + c*dy²) - b*dx*dy
  alpha = opacity * exp(power)
  前到后混合
```

### AAA-Gaussians 完整流水线 (eval_3D=ON)

```
对每个 Gaussian (预处理):
  1. 近平面裁剪 (z > 0.2)
  2. 3D Mip 滤波 → scale_dilated, dilation_factor   [新增·已移植]
  3. 构建 gauss2screen 4×4 矩阵                      [替代 cov3D/cov2D·已移植]
  4. opacity *= dilation_factor                        [新增·已移植]
  5. 椭球体内相机检测                                   [新增·已移植]
  6. 3D 视锥体裁剪 (maxContribGaussianFrustum3D)      [新增·已移植]
  7. 屏幕空间 AABB (computeAABBScreen) → radius       [替代 特征值·已移植]
  8. NDC 投影中心 + getRect → tile覆盖                 [已移植]
  9. SH颜色评估 (同标准路径)                            [已移植]

Tile binning + 排序:
  · 标准: 全局 view-space z depth key                  [当前使用]
  · AAA-GS: Per-Tile depth key (depthAlongRay)         [未移植]

光栅化 (对每个像素):
  · 标准: 按预排序顺序直接混合                           [当前使用]
  · AAA-GS: 三级分层重排序 (TAIL→MID→HEAD)             [未移植]

  plane_x = g2s[row0] - g2s[row3] × (px+0.5)        [替代 2D conic·已移植]
  plane_y = g2s[row1] - g2s[row3] × (py+0.5)
  power = -0.5 × maxContribRay(plane_x, plane_y)
  alpha = opacity * exp(power)
  前到后混合
```

---

## 11. 消融实验与移植优先级（论文 Table 2, 4, 5）

### 各组件对渲染质量的影响

| 组件 | 去除后影响 | 移植状态 | 优先级 |
|------|-----------|----------|--------|
| 3D 评估 (eval_3D) | 大 FOV 下出现投影畸变，边缘失真 | ✅ 已移植 | — |
| 3D Mip 抗锯齿 | 分布内指标不变，但**缩放分辨率时 PSNR 下降 3.8dB** | ✅ 已移植 | — |
| dilation_factor (EWA) | 透明度不补偿导致远距离 Gaussian 过暗 | ✅ 已移植 | — |
| 分层排序 (StopThePop) | 指标微升 +0.06 PSNR，但**产生可见 popping 伪影** | ❌ 未移植 | 中 (视觉质量) |
| Per-Tile Depth key | 减少排序错误，配合分层排序使用 | ❌ 未移植 | 中 (配合分层排序) |
| 逐 Tile 3D 裁剪 | 无质量影响，但**渲染耗时增加 ~2.7×** (7.72→14.40ms) | ❌ 已实现但回退 | 高 (需精度修复) |
| 视空间包围盒 (new_aabb) | 视锥体边缘 Gaussian 被错误丢弃，popping | ⚠️ 已实现未启用 | 低 (当前场景无此问题) |
| 屏幕空间 AABB | 回退到特征值半径，tile覆盖过大 | ✅ 已移植 | — |
| 椭球体内相机检测 | 相机穿越 Gaussian 时渲染不稳定 | ✅ 已移植 | — |
| 3D 全局视锥体裁剪 | 不可见 Gaussian 不被提前剔除 | ✅ 已移植 | — |

### 性能数据（论文 Table 5，RTX 4090）

```
完整 AAA-GS:  7.72ms  (M360 indoor)
去掉3D裁剪:  14.40ms  (+87%)
标准 3DGS:    6.79ms  (MCMC baseline)
AAA-GS 开销:  仅 +13.8%
```

### 推荐的后续移植路线

1. **第一优先**: 逐 Tile 3D 裁剪 → GPU scatter kernel 中加入 `maxContribGaussianFrustum3D`，减少 tile-pair 数量，直接提升光栅化性能
2. **第二优先**: Per-Tile Depth key → tile binner 中为每个 tile 计算 `depthAlongRay`，改善排序一致性
3. **第三优先**: 分层排序 → 需要重写光栅化 kernel，用 OpenCL sub_group 原语替代 CUDA warp 原语
4. **可选**: 视空间包围盒 → 仅在极近距离观察场景时需要
