# AAA-Gaussians 算子分析文档

> 目标：为将 AAA-Gaussians（PyTorch + CUDA）移植到鸿蒙 Vulkan 平台提供算子清单，供后续制定测试计划使用。
>
> 分析基于源码版本：`/Graph/AAA-Gaussians`  
> 日期：2026-04-16

---

## 目录

1. [算法流水线概览](#1-算法流水线概览)
2. [Python 层算子](#2-python-层算子)
   - 2.1 [模型参数激活函数](#21-模型参数激活函数)
   - 2.2 [球谐函数 (Spherical Harmonics)](#22-球谐函数-spherical-harmonics)
   - 2.3 [3D MIP 滤波器](#23-3d-mip-滤波器)
   - 2.4 [损失函数](#24-损失函数)
   - 2.5 [几何工具函数](#25-几何工具函数)
   - 2.6 [高斯致密化 / MCMC 重定位](#26-高斯致密化--mcmc-重定位)
   - 2.7 [优化器](#27-优化器)
3. [CUDA 核算子](#3-cuda-核算子)
   - 3.1 [前向通道：预处理](#31-前向通道预处理-preprocesscuda)
   - 3.2 [前向通道：排序基础设施](#32-前向通道排序基础设施)
   - 3.3 [前向通道：渲染](#33-前向通道渲染-rendercuda)
   - 3.4 [后向通道：渲染](#34-后向通道渲染-rendercuda-backward)
   - 3.5 [后向通道：预处理](#35-后向通道预处理-preprocesscuda-backward)
   - 3.6 [KNN 初始化](#36-knn-初始化-simple-knn)
   - 3.7 [高斯重定位 CUDA 核](#37-高斯重定位-cuda-核-utilscu)
4. [AAA (EVAL_3D) 专有算子](#4-aaa-eval_3d-专有算子)
5. [算子分类汇总表](#5-算子分类汇总表)
6. [推理 vs 训练算子矩阵](#6-推理-vs-训练算子矩阵)
7. [移植优先级建议](#7-移植优先级建议)

---

## 1. 算法流水线概览

AAA-Gaussians 基于 3DGS，引入了 **真 3D 高斯求值（EVAL_3D）** 和多种精确排序模式（StopThePop）。流水线分两个阶段：

```
【推理/渲染阶段】
  ┌─────────────────────────────────────────────────────────────────┐
  │ CPU/Python                                                      │
  │  1. 相机参数计算（投影矩阵、视角矩阵）                              │
  │  2. 球谐→RGB（可选 Python 计算）                                  │
  │  3. 3D 协方差矩阵（可选 Python 计算）                              │
  └────────────────────────┬────────────────────────────────────────┘
                           │ CUDA 调用
  ┌────────────────────────▼────────────────────────────────────────┐
  │ CUDA                                                            │
  │  4. preprocessCUDA  → 视椎剔除、投影、SH→RGB、协方差计算          │
  │  5. Prefix Sum (CUB) → tile 偏移量                              │
  │  6. duplicateWithKeys → 生成 (tile|depth) 排序 key              │
  │  7. RadixSort (CUB)  → 按 tile+depth 排序                       │
  │  8. identifyTileRanges → 确定每个 tile 的高斯范围                │
  │  9. renderCUDA       → tile 内 alpha 合成，输出像素颜色           │
  └─────────────────────────────────────────────────────────────────┘

【训练额外步骤】
  10. 反向渲染 renderCUDA_backward → dL/d_color, dL/d_opacity, dL/d_mean2D
  11. computeCov2DCUDA_backward   → dL/d_cov3D, dL/d_mean3D
  12. preprocessCUDA_backward     → dL/d_sh, dL/d_scale, dL/d_rotation
  13. Loss (L1 + SSIM)
  14. Adam optimizer step
  15. MCMC 致密化：compute_relocation CUDA 核 + KNN
```

---

## 2. Python 层算子

### 2.1 模型参数激活函数

| 算子 | 数学表达 | 来源文件 | 输入/输出形状 | 推理/训练 |
|------|---------|---------|------------|---------|
| **Sigmoid**（opacity 激活） | `σ(x) = 1/(1+e⁻ˣ)` | `gaussian_model.py:35` | `[N,1]→[N,1]` | 两者 |
| **Exp**（scale 激活） | `e^x` | `gaussian_model.py:35` | `[N,3]→[N,3]` | 两者 |
| **L2 Normalize**（rotation 激活） | `q / ‖q‖` | `gaussian_model.py:43` | `[N,4]→[N,4]` | 两者 |
| **inverse_sigmoid** | `log(x/(1-x))` | `general_utils.py:18` | `[N,1]→[N,1]` | 训练 |
| **log**（scale 逆激活） | `log(x)` | `gaussian_model.py:36` | `[N,3]→[N,3]` | 训练 |

### 2.2 球谐函数 (Spherical Harmonics)

文件：`utils/sh_utils.py`，`cuda_rasterizer/forward.cu`（CUDA 版）

| 算子 | 说明 | 阶数 | 推理/训练 |
|------|------|------|---------|
| **eval_sh (Python)** | SH 系数 × 多项式基函数 → RGB | 0~4 阶，系数数 1/4/9/16/25 | 推理（可选） |
| **computeColorFromSH (CUDA)** | 同上，GPU 实现，deg 0~3 | 0~3 阶 | 推理 |
| **RGB2SH** | `(rgb-0.5)/C0` | — | 初始化 |
| **SH2RGB** | `sh*C0+0.5` | — | 初始化 |

**eval_sh 详细操作分解**（Python，阶数 0~3）：

```
deg=0: result = C0 * sh[0]                                [element-wise mul]
deg=1: += -C1*y*sh[1] + C1*z*sh[2] - C1*x*sh[3]         [mul, add]
deg=2: += C2[i] * (多项式组合) * sh[4..8]                 [mul, add]
deg=3: += C3[i] * (多项式组合) * sh[9..15]                [mul, add]
```

所有操作均可归结为：**标量乘加（FMA）+ element-wise 运算**。

### 2.3 3D MIP 滤波器

文件：`utils/mip_filter.py:5`（Python 预计算），`consistent_common.cuh`（CUDA 内联使用）

**mip_filter_3d**：

| 步骤 | 操作 | PyTorch 算子 |
|------|------|------------|
| 点云转相机空间 | `xyz_cam = xyz @ R + T` | `torch.matmul`, 广播加法 |
| 计算深度距离 | `torch.norm(xyz_cam, dim=1)` | L2 Norm |
| 深度有效性掩码 | `xyz_cam[:,2] > 0.2` | 比较，`torch.logical_and` |
| 屏幕空间投影 | `x/z * focal + W/2` | element-wise 除/乘/加 |
| 视野内判断 | `torch.logical_and/or` | 逻辑运算 |
| 最小深度规约 | `torch.min(distance[valid], z[valid])` | 条件规约 |
| Filter 计算 | `distance / focal_length * sqrt(0.3)` | element-wise 除/乘 |

### 2.4 损失函数

文件：`utils/loss_utils.py`，使用 `fused_ssim` 外部包

#### L1 Loss
```python
torch.abs(pred - gt).mean()
```
- 操作：element-wise subtract → abs → mean

#### L2 Loss（未在 train.py 直接使用，备用）
```python
((pred - gt) ** 2).mean()
```

#### SSIM（loss_utils.py 版本）
| 步骤 | PyTorch 算子 |
|------|------------|
| 高斯窗口生成 | `torch.exp`, `tensor.mm`（外积）|
| 局部均值 | `F.conv2d`（depthwise，11×11 高斯核）|
| 局部方差/协方差 | `F.conv2d` × 3 + element-wise 运算 |
| SSIM map 计算 | element-wise `*`, `/`, `+` |
| 输出 | `.mean()` |

> **注意**：实际训练使用 `fused_ssim` 包（CUDA 融合实现），不走上面的 Python 路径。

### 2.5 几何工具函数

文件：`utils/general_utils.py`，`utils/graphics_utils.py`

| 算子 | 操作 | 来源 |
|------|------|------|
| **build_rotation** | 四元数→旋转矩阵（9 个标量乘加） | `general_utils.py:78` |
| **build_scaling_rotation** | `L[:,i,i]=s[i]`, `L = R @ L` | `general_utils.py:101` |
| **strip_symmetric** | 对称矩阵 → 6 向量（上三角提取） | `general_utils.py:75` |
| **covariance_activation** | `L @ L.T`（批量 3×3 矩阵乘法） | `gaussian_model.py:31` |
| **getProjectionMatrix** | 透视投影矩阵构建 | `graphics_utils.py:51` |
| **getWorld2View2** | 视图矩阵构建（含 numpy.linalg.inv） | `graphics_utils.py:38` |

### 2.6 高斯致密化 / MCMC 重定位

文件：`scene/gaussian_model.py`，`utils/reloc_utils.py`

| 算子 | 操作 | 来源 |
|------|------|------|
| **get_opacity（筛死亡高斯）** | `opacity <= 0.005` 掩码 | `gaussian_model.py:132` |
| **multinomial 采样** | `torch.multinomial(probs, num, replacement=True)` | `gaussian_model.py:357` |
| **bincount** | `torch.bincount(sampled_idxs)` | `gaussian_model.py:360` |
| **compute_relocation_cuda** | 调用 CUDA 核计算新 opacity/scale | `reloc_utils.py:9` |
| **cat_tensors_to_optimizer** | `torch.cat` 扩展参数张量 | `gaussian_model.py:270` |
| **噪声扰动** | `torch.randn_like` + `torch.bmm` | `train.py:146` |
| **op_sigmoid（自定义）** | `1/(1+exp(-k*(x-x0)))` | `train.py:143` |

### 2.7 优化器

| 算子 | 说明 |
|------|------|
| **Adam** | 6 组参数（xyz, f_dc, f_rest, opacity, scaling, rotation），`eps=1e-15` |
| **指数 LR 调度** | xyz 参数的对数线性衰减 |

---

## 3. CUDA 核算子

### 3.1 前向通道：预处理 (`preprocessCUDA`)

**文件**：`cuda_rasterizer/forward.cu:70`  
**模板参数**：`<int C, bool TILE_BASED_CULLING, bool LOAD_BALANCING, bool EVAL_3D>`  
**线程配置**：每个高斯一个线程，`(P+255)/256` blocks × 256 threads

#### 3.1.1 标准 2D 模式 (EVAL_3D=false)

| 子算子 | 数学描述 | 文件位置 |
|--------|---------|---------|
| **Near Clipping** | `z_view < 0.2` 剔除 | `forward.cu:204` |
| **computeCov3D** | 四元数→旋转矩阵 R；S=diag(scale)；Σ=RSS^T R^T（上三角6元素） | `auxiliary.h` |
| **computeCov2D** | J（投影雅可比）× W（视图矩阵）→ T；cov2D = T^T × Σ × T（2×2对称矩阵） | `auxiliary.h` |
| **dilateCov2D** | 加低通滤波：`c_xx += 0.3`, `c_yy += 0.3`；计算行列式 | `auxiliary.h` |
| **computeConicOpacity** | `conic = inv(cov2D)` (2×2 逆)；opacity 乘以 EWA scaling factor | `auxiliary.h` |
| **世界坐标→NDC→像素** | `world2ndc(mean3D, proj)` → `ndc2Pix` | `auxiliary.h` |
| **屏幕空间 AABB** | 基于特征值计算 tile 覆盖矩形 | `forward.cu:241` |
| **computeColorFromSH (CUDA)** | SH 多项式展开（最高 3 阶），结果 clamp | `auxiliary.h` |
| **computeInvCov3D** | 计算 3D 协方差逆矩阵（用于 kbuffer 模式深度计算） | `auxiliary.h` |

#### 3.1.2 AAA 3D 模式 (EVAL_3D=true)

| 子算子 | 数学描述 | 文件位置 |
|--------|---------|---------|
| **compute_gauss2screen** | 构建高斯椭球→屏幕空间齐次变换矩阵（4×4）；含 MIP 滤波膨胀 | `stopthepop/consistent_common.cuh:12` |
| **max_contrib_gaussian_frustum_3D** | 计算高斯在视椎内的最大贡献位置（平面-射线求交） | `consistent_common.cuh:131` |
| **compute_aabb_screen** | 投影椭球到屏幕求精确 AABB（Hahlbohm et al. 方法） | `consistent_common.cuh:286` |
| **compute_aabb_view** | 视角空间角度分析求 AABB（备用方法） | `consistent_common.cuh:217` |
| **opacity 膨胀因子** | `proper_ewa_scaling`：乘以 `dilation_factor=sqrt(A/B)` | `consistent_common.cuh:50` |

#### 3.1.3 Tile-based Culling 模式 (TILE_BASED_CULLING=true)

| 子算子 | 说明 |
|--------|------|
| **computeTilebasedCullingTileCount** | 逐 tile 测试高斯 alpha 贡献，精确计算有效 tile 数 |
| **Load Balancing (LOAD_BALANCING=true)** | 用 warp 投票 (`__ballot_sync`) 判断整组线程是否可提前退出 |

---

### 3.2 前向通道：排序基础设施

| 算子 | 实现 | 输入/输出 | 说明 |
|------|------|---------|------|
| **Inclusive Prefix Sum** | `cub::DeviceScan::InclusiveSum` | `tiles_touched[P] → point_offsets[P]` | 计算每个高斯的 tile 写入偏移量 |
| **duplicateWithKeysCUDA** | `forward.cu:26`（简单版） | `means2D`, `depths`, `offsets` → `keys_unsorted[R]`, `values_unsorted[R]` | 生成 64-bit key = `tile_id(32bit) | depth(32bit)` |
| **duplicateWithKeys_extended** | `forward.cu` 扩展版（模板） | 同上，含 per-tile 深度变体（CENTER/MAXPOS） | StopThePop 排序 key 变体 |
| **Radix Sort (pairs)** | `cub::DeviceRadixSort::SortPairs` | `keys_unsorted → keys_sorted`, `values_unsorted → values_sorted` | 按 tile_id 和 depth 排序 |
| **identifyTileRanges** | `rasterizer_impl.cu:133` | `point_list_keys[R] → ranges[num_tiles]` | 扫描已排序 key 确定每 tile 的起止索引 |
| **checkFrustum** | `rasterizer_impl.cu:113` | `means3D, viewmatrix, projmatrix → present[P]` | 视椎剔除（markVisible 接口） |

---

### 3.3 前向通道：渲染 (`renderCUDA`)

**文件**：`cuda_rasterizer/forward.cu:322`，stopthepop 变体在 `stopthepop/*.cuh`  
**线程配置**：每个像素一个线程，`tile_grid(W/BLOCK_X, H/BLOCK_Y)` × `block(BLOCK_X=16, BLOCK_Y=16)`

#### 3.3.1 全局排序模式 (SortMode::GLOBAL) — 标准 alpha 合成

| 子算子 | 说明 |
|--------|------|
| **Shared Memory Tiling** | 协作从全局内存加载 Gaussian 数据到共享内存（BLOCK_SIZE 批次） |
| **2D 高斯核求值** | `power = -0.5*(cx*dx²+cz*dy²) - cy*dx*dy`；`alpha = opacity * exp(power)` |
| **3D 高斯核求值（EVAL_3D）** | `max_contrib_ray(plane_x, plane_y)`：射线与高斯椭球面交，求沿射线最近点 |
| **前向 Alpha 合成** | `T = T * (1-alpha)`；`C[ch] += color[ch] * alpha * T` |
| **Early termination** | `T < 0.0001` 则标记 done，`__syncthreads_count` 整块提前退出 |

#### 3.3.2 Per-pixel k-buffer 排序模式 (SortMode::PER_PIXEL_KBUFFER)

模板参数 `WINDOW` = 1/2/4/8/12/16/20/24（每像素局部排序窗口大小）

| 子算子 | 说明 |
|--------|------|
| **Per-pixel insertion sort** | 维护每像素的小型有序缓冲，边插入边合成 |
| **cov3D_inv 深度计算** | 使用 3D 协方差逆矩阵计算沿射线的精确深度 |

#### 3.3.3 分层排序模式 (SortMode::HIERARCHICAL)

模板参数 `HEAD_QUEUE_SIZE` × `MID_QUEUE_SIZE`（4×4 tile 级队列 + 像素级队列）

| 子算子 | 说明 |
|--------|------|
| **Hierarchical insertion sort** | 在 4×4 tile 级维护中间队列，再向单像素分发 |
| **HIER_CULLING** | 4×4 tile 级 alpha 提前剔除 |

---

### 3.4 后向通道：渲染 (`renderCUDA` backward)

**文件**：`cuda_rasterizer/backward.cu:662`  
**策略**：反向遍历排序列表（后到前），重放前向 alpha 计算后反向累积梯度

| 子算子 | 输出梯度 | 说明 |
|--------|---------|------|
| **颜色梯度 (dL/d_colors)** | `dL_dcolors[G, ch]` | `atomicAdd`，多个像素贡献同一高斯 |
| **不透明度梯度 (dL/d_opacity)** | `dL_dopacity[G]` | `G * dL_dalpha`，`atomicAdd` |
| **2D 中心梯度 (dL/d_mean2D)** — 2D 模式 | `dL_dmean2D[G].xy` | 高斯核对位置偏导，`atomicAdd` |
| **2D Conic 梯度 (dL/d_conic2D)** — 2D 模式 | `dL_dconic2D[G].{x,y,w}` | 高斯核对逆协方差偏导，`atomicAdd` |
| **gauss2screen 梯度 (dL/d_gauss2screen)** — 3D 模式 | `dL_dgauss2screen[G, 4×4]` | 复杂的齐次矩阵链式法则，`atomicAdd`×12 项 |

---

### 3.5 后向通道：预处理 (`preprocessCUDA` backward)

#### 3.5.1 2D 模式预处理后向

**文件**：`cuda_rasterizer/backward.cu`

| 核 | 输出梯度 | 主要计算 |
|----|---------|---------|
| **computeCov2DCUDA** | `dL_dcov3D[G,6]`, `dL_dmean3D[G]`, `dL_dopacity[G]` | conic→cov2D 逆；cov2D→T→Σ₃ 链式；EWA 缩放因子反向 |
| **preprocessCUDA_backward** | `dL_dmean3D[G]`, `dL_dsh[G,M]`, `dL_dscale[G]`, `dL_drot[G]` | 投影雅可比反向（mean2D→mean3D）；SH 多项式反向；协方差→scale/rotation 反向（computeCov3D device func） |

#### 3.5.2 3D 模式预处理后向 (`preprocessCUDA_3D`)

| 子算子 | 输出梯度 | 说明 |
|--------|---------|------|
| **SH backward** | `dL_dsh[G,M]`, 贡献到 `dL_dmean3D` | 同 2D 模式 |
| **computeGauss2Screen_backward** | `dL_dmean3D`, `dL_dscale`, `dL_drot` | gauss2screen 矩阵对 scale/rotation/mean 的梯度 |
| **computeGauss2Screen_ProperEWA_backward** | 同上 + `dL_dopacity` 修正 | 含 EWA 膨胀因子对 scale/rotation 的额外梯度项 |

---

### 3.6 KNN 初始化 (simple-knn)

**文件**：`submodules/simple-knn/simple_knn.cu`  
**用途**：初始化高斯 scale（从最近邻距离估计），仅在训练开始时调用一次

| 核 | 说明 |
|----|------|
| **coord2Morton** | 3D 点→30-bit Morton 码（空间填充曲线编码），每个维度 10bit |
| **boxMinMax** | 并行规约：每个 BLOCK_SIZE(1024) 点一组，求组内 min/max float3 |
| **boxMeanDist** | K=3 近邻距离计算：基于 Morton 排序后的空间局部性加速搜索 |
| **CUB 辅助**：Reduce Min/Max | `cub::DeviceReduce::Reduce`（全局 min/max） |
| **CUB 辅助**：RadixSort | `cub::DeviceRadixSort::SortPairs`（Morton 码排序） |
| **Thrust**：sequence | `thrust::sequence`（初始化索引序列） |

---

### 3.7 高斯重定位 CUDA 核 (`utils.cu`)

**文件**：`cuda_rasterizer/utils.cu:4`  
**用途**：MCMC densification，将一个高斯"分裂"为 N 个副本，计算新的 opacity 和 scale

| 算子 | 数学表达 | 说明 |
|------|---------|------|
| **新 opacity 计算** | `1 - (1-opacity_old)^(1/N)` | `powf` |
| **新 scale 计算** | 二项式累加：`Σ C(i-1,k) * (-1)^k / sqrt(k+1) * new_opacity^(k+1)` | 二重循环 + `powf`，查表（binoms 预计算） |

---

## 4. AAA (EVAL_3D) 专有算子

这是 AAA-Gaussians 相比基础 3DGS 的核心创新，移植时需重点关注：

| 算子 | 位置 | 功能 | Vulkan 映射难度 |
|------|------|------|---------------|
| **compute_gauss2screen** | `consistent_common.cuh:12` | 构建齐次 4×4 变换矩阵（含 MIP 滤波膨胀） | 中 |
| **max_contrib_gaussian_frustum_3D** | `consistent_common.cuh:131` | 视椎内最大高斯贡献计算（6 个平面-射线求交） | 高 |
| **compute_aabb_screen** | `consistent_common.cuh:286` | 椭球屏幕空间 AABB（Hahlbohm 方法，含平方根） | 中 |
| **compute_aabb_view** | `consistent_common.cuh:217` | 视角空间 AABB（三角函数，atan2） | 高 |
| **max_contrib_ray** | `consistent_common.cuh:89` | 射线与椭球面最近贡献点（两平面交线求解） | 中 |
| **3D renderCUDA** | `forward.cu:434` | 基于 gauss2screen 矩阵的 3D alpha 计算 | 高 |
| **3D backward gauss2screen** | `backward.cu:844` | gauss2screen 矩阵梯度（12 个 atomicAdd） | 高 |
| **computeGauss2Screen_backward** | `backward.cu:485` | gauss2screen 对 scale/rot/mean 的梯度 | 高 |

---

## 5. 算子分类汇总表

### 5.1 基础线性代数算子

| 算子 | 用途 | 来源 |
|------|------|------|
| `matmul` (2D/batched) | 协方差、旋转矩阵 | Python |
| `matmul` (4×4) | 投影变换 | CUDA device func |
| `matmul` (3×3) | 协方差、Jacobian | CUDA device func |
| `mat_transpose` | 协方差、梯度 | CUDA device func |
| `mat_inverse` (2×2) | 2D conic | CUDA device func |
| `mat_inverse` (3×3) | 3D cov inverse（kbuffer 模式） | CUDA device func |
| `vector_normalize` | 四元数归一化，视线方向 | Python + CUDA |
| `vector_dot` | 各处内积计算 | CUDA device func |
| `vector_cross` | 平面-射线求交（3D 模式） | CUDA device func |
| `outer_product` | SSIM 窗口生成，梯度计算 | Python |

### 5.2 特殊函数算子

| 算子 | 用途 | 精度要求 |
|------|------|---------|
| `exp` / `expf` | alpha 计算 | FP32 |
| `log` / `logf` | opacity threshold | FP32 |
| `sqrt` / `sqrtf` | 半径、AABB | FP32 |
| `pow` / `powf` | MCMC relocation | FP32 |
| `atan2` | 视角 AABB（view 空间） | FP32 |
| `tan` | FoV 转换 | FP32 |
| `sigmoid` | opacity 激活 | FP32 |

### 5.3 归约和扫描算子

| 算子 | 用途 | 实现 |
|------|------|------|
| **Inclusive Prefix Sum** | tile 偏移量 | `cub::DeviceScan::InclusiveSum` |
| **Radix Sort (64-bit key)** | tile+depth 排序 | `cub::DeviceRadixSort::SortPairs` |
| **Min Reduce** | KNN AABB，debug viz | `cub::DeviceReduce` |
| **Max Reduce** | KNN AABB，debug viz | `cub::DeviceReduce` |
| **AtomicAdd** | 后向梯度累积 | CUDA 原子操作 |
| **warp vote** (`__ballot_sync`) | Load balancing 提前退出 | CUDA warp 原语 |

### 5.4 内存访问模式

| 模式 | 位置 | 说明 |
|------|------|------|
| **Coalesced global read** | renderCUDA 共享内存加载 | 连续线程读连续内存 |
| **Shared memory tiling** | renderCUDA tile 协作加载 | `__shared__` 数据复用 |
| **Scatter atomicAdd** | 后向梯度写入 | 多像素写同一高斯 |
| **Gather** | 按排序索引读高斯属性 | 随机访问 |

---

## 6. 推理 vs 训练算子矩阵

| 算子 | 推理 | 训练 | 备注 |
|------|:----:|:----:|------|
| sigmoid / exp / normalize（激活） | ✅ | ✅ | |
| eval_sh / computeColorFromSH | ✅ | ✅ | |
| computeCov3D | ✅ | ✅ | |
| computeCov2D / dilateCov2D | ✅（2D模式） | ✅（2D模式） | EVAL_3D=true 时不用 |
| compute_gauss2screen | ✅（3D模式） | ✅（3D模式） | AAA 专有 |
| Prefix Sum（CUB） | ✅ | ✅ | |
| Radix Sort（CUB） | ✅ | ✅ | |
| identifyTileRanges | ✅ | ✅ | |
| renderCUDA (forward) | ✅ | ✅ | |
| renderCUDA (backward) | ❌ | ✅ | |
| computeCov2DCUDA (backward) | ❌ | ✅ | 2D模式 |
| preprocessCUDA_backward | ❌ | ✅ | |
| computeGauss2Screen backward | ❌ | ✅（3D模式） | AAA 专有 |
| L1 Loss / SSIM / fused_ssim | ❌ | ✅ | |
| Adam optimizer | ❌ | ✅ | |
| compute_relocation CUDA 核 | ❌ | ✅ | MCMC densification |
| KNN (simple_knn) | ❌ | ✅（初始化） | |
| mip_filter_3d | ❌ | ✅（保存点） | |

---

## 7. 移植优先级建议

基于推理流水线（鸿蒙端只需渲染，不需训练），优先级如下：

### P0：渲染关键路径（必须实现）

1. **preprocessCUDA**（单 Vulkan compute shader）
   - 依赖：`computeCov3D`, `computeCov2D`, `dilateCov2D`, `computeConicOpacity`（2D 模式）
   - AAA 增项：`compute_gauss2screen`, AABB 计算

2. **Prefix Sum**（Vulkan 实现，或用 subgroup/workgroup scan）

3. **duplicateWithKeys**（生成排序 key）

4. **Radix Sort**（Vulkan compute shader 或调用已有实现）
   - 可使用开源 Vulkan radix sort 库

5. **identifyTileRanges**

6. **renderCUDA GLOBAL 模式**（Vulkan fragment/compute shader）
   - 2D 模式：标准 3DGS conic 合成
   - 3D 模式（AAA）：ray-plane 射线求交 + 3D alpha 合成

### P1：质量提升（可选但重要）

7. **Per-pixel kbuffer 排序**（StopThePop 排序质量）

8. **computeColorFromSH**（SH 球谐评估，deg 0~3）

9. **MIP 滤波（filter_3D）**（抗锯齿）

### P2：仅训练需要（推理不需要）

10. 所有 backward CUDA 核
11. Adam optimizer
12. compute_relocation / KNN

---

## 附录：关键数据结构

| 参数 | 形状 | 说明 |
|------|------|------|
| `_xyz` | `[N, 3]` | 高斯中心位置 |
| `_features_dc` | `[N, 1, 3]` | SH DC 分量（基础颜色） |
| `_features_rest` | `[N, (deg+1)²-1, 3]` | SH 高阶分量 |
| `_opacity` | `[N, 1]` | 原始 logit（激活后为透明度） |
| `_scaling` | `[N, 3]` | 原始 log 缩放（激活后为轴长） |
| `_rotation` | `[N, 4]` | 单位四元数（wxyz） |
| `filter_3D` | `[N, 1]` | MIP 滤波器半径（由 camera 集合确定） |
| `cov3D` | `[N, 6]` | 3D 协方差上三角（前向缓存） |
| `cov3D_inv` | `[N, 3, float4]` | 3D 协方差逆（kbuffer 深度计算用） |
| `gauss2screen` | `[N, 16]` | gauss→screen 齐次矩阵（AAA 专有） |
| `conic_opacity` | `[N, float4]` | {a, b, c, opacity}（2D 模式） |
| `means2D` | `[N, 2]` | 投影后屏幕像素坐标 |
| `depths` | `[N]` | 排序用深度值 |
| `tiles_touched` | `[N]` | 每个高斯覆盖的 tile 数 |
| `point_list` | `[R]` | 排序后的高斯索引（R = sum of tiles_touched） |
| `ranges` | `[num_tiles, 2]` | 每个 tile 在 point_list 中的范围 |

---

*文档由代码库分析自动生成，如有疑问请参考对应源文件。*
