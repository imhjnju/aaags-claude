# VK 初始 Loss 不一致调查报告 (Session 10, 2026-04-23)

## 目标
VK 训练第一次迭代 loss 应与 CUDA/Python 完全一致。vulkan-3d worktree 已实现 42.8 dB PSNR，预期 loss 差距极小。

## 问题总览

| # | 问题 | 状态 | 影响 | 说明 |
|---|------|------|------|------|
| 1 | CHW/HWC 布局不匹配 | **已修复** | ~8% loss gap | rasterizer 输出 CHW，GT 以 HWC 加载 |
| 2 | Loss 函数不同 | **已确认** | +0.064 | VK 默认 `lambda_dssim=0.2`，Python 用纯 L1 |
| 3 | RNG/视角选择不同 | **已确认** | 不可比 | VK view=28 vs Python view=15 |
| 4 | 视图矩阵旋转方向 | **未解决** | 渲染完全不相关 | 可能是当前最大 bug |
| 5 | 渲染通道失衡 | **未解决** | R=0.018 G=0.058 B=0.003 | 视图矩阵错误的下游效应 |

---

## 已修复的问题

### 1. CHW/HWC 布局不匹配
- **文件**: `vk_train_main.cpp`
- **根因**: rasterizer 输出 `out_image[ch*HW+px]` (CHW)，`readPPM` 加载 GT 为 HWC `gt[(y*W+x)*3+c]`。`compute_combined_loss_gradient` 线性比较两者 → 像素映射错误
- **修复**: GT 加载后 HWC→CHW，保存前 CHW→HWC
- **代码位置**: `vk_train_main.cpp` 约第 522-529 行

### 2. Loss 函数不同 (无需修代码)
- VK 默认 `lambda_dssim=0.2` (L1+DSSIM)，Python 用 `l1_loss`
- DSSIM 贡献约 +0.064
- 对比时应使用相同 loss 函数

### 3. RNG/视角选择不同 (无需修代码)
- VK: `std::mt19937(42)` + 76 cameras → first view=28
- Python: `torch.randint(0,66)` → first view=15
- 对比时应强制使用同一视角

---

## 未解决的关键问题

### 4. 视图矩阵旋转方向错误 (最关键)

#### 现象
- 修复视图矩阵后，PSNR 仅 17.08 dB (之前 16.45 dB)
- 像素相关性 0.018 — 渲染结果与 CUDA 本质上不相关
- 出现纯绿色像素 [0, 0.42, 0] 而 CUDA 在同一位置为黑色

#### 根因分析 (本次 session 新发现)

**cameras.json 中 "rotation" 字段存储的是 R_w2c (世界→相机旋转)**

证据链:
1. Python `camera_to_JSON()` (`utils/camera_utils.py:62-82`) 中:
   ```python
   Rt[:3, :3] = camera.R.transpose()
   W2C = np.linalg.inv(Rt)
   rot = W2C[:3, :3]   # ← 这是 R_w2c
   ```
2. `camera.R` 来自 COLMAP，是 R_w2c
3. 最终 `W2C[:3, :3]` 就是 R_w2c

**CUDA 的视图矩阵实际使用 R_w2c**

CUDA `loadMatrix4x4` 从 PyTorch tensor 加载：
```cuda
M[i][j] = m[i * 4 + j]  // GLM 列主序：M[col][row] = ptr[col*4+row]
```

PyTorch 存储行主序 `w2v[row*4+col]`，所以 `M[col][row] = w2v[col*4+row]`

这意味着 `M[col][row]` = PyTorch 矩阵的第 `col` 行、第 `row` 列。即 CUDA 实际使用的数学矩阵是 PyTorch w2v 的**转置**。

Python `getWorld2View2` 返回的矩阵上三角 3x3 是 R_c2w (= R_w2c^T)。CUDA loadMatrix4x4 的隐式转置将其变为 R_w2c。

**VK 当前的存储方式**

`vk_train_main.cpp` 第 346-348 行 (前序 session 的 "修复"):
```cpp
// Column-major: store columns of R
entry.cam.view_matrix[0]=R[0][0]; entry.cam.view_matrix[1]=R[1][0]; entry.cam.view_matrix[2]=R[2][0]; // col 0
entry.cam.view_matrix[4]=R[0][1]; entry.cam.view_matrix[5]=R[1][1]; entry.cam.view_matrix[6]=R[2][1]; // col 1
entry.cam.view_matrix[8]=R[0][2]; entry.cam.view_matrix[9]=R[1][2]; entry.cam.view_matrix[10]=R[2][2]; // col 2
```

这里 R[row][col] 来自 cameras.json = R_w2c[row][col]。列主序存储后数学矩阵就是 R_w2c。

**但 CUDA 的 `transformPoint4x3` 使用的是 R_w2c 而不是 R_c2w**。所以这部分看起来对了...

**问题出在 W = transpose(mat3(viewmatrix))**

shader `computeCov2D` 中:
```glsl
mat3 W = transpose(mat3(cam.viewmatrix));
```

如果 viewmatrix 上三角 3x3 = R_w2c，则 W = R_w2c^T = R_c2w。

CUDA forward_common.h:93:
```cuda
// W = upper-left 3x3 of viewmatrix (after loadMatrix4x4 implicit transpose)
// CUDA 的 viewmatrix 已经是 R_w2c (因为隐式转置), 所以 W 直接取 3x3 = R_w2c
```

CUDA 不做 `transpose()` — 它直接使用 loadMatrix4x4 后的矩阵上三角 3x3。

**结论：VK shader 对 viewmatrix 做了一次额外的 transpose()，而 CUDA 不做。**

这意味着协方差计算用了错误的 W，导致 2D 投影的 splat 大小/形状错误 → 渲染完全不相关。

#### 修复方案 (待验证)

**方案 A**: 去掉 shader 中的 `transpose()`，直接使用 `mat3(cam.viewmatrix)`
- 需要理解 shader 中 W 的数学推导来确定正确方向

**方案 B**: 在 C++ 端存储 viewmatrix 时预转置，让 shader 的 `transpose()` 恢复正确值
- 即把前序 session 的 "修复" 改回旧版 (存行不存列)

**方案 C**: 直接打印 CUDA 和 VK 的 W 矩阵数值，确认差异后针对性修复

#### 需要做的验证
1. 打印 CUDA kernel 中 W (viewmatrix upper-left 3x3) 的 9 个数值
2. 打印 VK shader 中 `transpose(mat3(cam.viewmatrix))` 的 9 个数值
3. 确认是否转置了

#### 相关文件
- `vk_train_main.cpp:346-348` — view matrix 构建
- `preprocess.comp:761` — `mat3 W = transpose(mat3(cam.viewmatrix))`
- `preprocess.comp:731-778` — computeCov2D 函数
- `preprocess.comp:919-922` — p_view 计算 (用 preciseTransformRow)
- `AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h` — CUDA loadMatrix4x4
- `AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu` — CUDA forward kernel
- `AAA-Gaussians/utils/camera_utils.py:62-82` — cameras.json 生成

### 5. 渲染通道失衡 (问题 4 的下游效应)

#### 现象
- VK: R_mean=0.018, G_mean=0.058, B_mean=0.003
- CUDA: R_mean=0.039, G_mean=0.038, B_mean=0.037
- VK 出现纯绿 [0, 0.42, 0] 像素在 CUDA 黑色区域
- 总亮度 VK 为 CUDA 的 69%

#### 分析
- 大概率是视图矩阵错误导致的，不是独立的 SH/颜色 bug
- 错误的 W → 错误的协方差 → splat 覆盖错误的像素 → 通道间看起来不平衡
- 应先修问题 4 再验证此问题是否自动消失

---

## SH 数据流验证 (已确认正确)

本次 session 追踪了完整的 SH 数据流：

```
PLY 文件 → PLY loader (reorder channel-first → basis-interleaved)
→ GaussianData.sh_coeffs [N*K*3]
→ initRawFromGaussianData (memcpy, 无变换)
→ VulkanTrainer::raw_sh_coeffs_ (vector copy)
→ activate_params() (memcpy, 无变换)
→ g_.sh_coeffs → PreprocessorVulkan::process() (upload to GPU)
→ shader: sh[base + k*3 + ch] (匹配 CUDA 布局)
```

布局: `[N, K, 3]` 其中 K = max_coeffs = (sh_degree+1)^2，通道交错 R,G,B。
与 CUDA 完全一致，无 bug。

---

## 下一步行动

1. **验证视图矩阵 transpose 问题** — 打印 CUDA vs VK 的 W 矩阵数值
2. **修复视图矩阵** — 根据验证结果调整 (去掉 shader transpose 或改 C++ 存储)
3. **重新运行** — 确认 PSNR 接近 42.8 dB
4. **对比 loss** — 使用相同视角 + 纯 L1 loss，确认初始 loss 一致

---

## 关键参考文件

| 文件 | 作用 |
|------|------|
| `src/vulkan/vk_train_main.cpp` | VK 训练入口，camera 加载，视图矩阵构建 |
| `src/vulkan/shaders/preprocess.comp` | 预处理 compute shader，SH 评估，协方差计算 |
| `src/vulkan/preprocessor_vulkan.cpp` | 预处理器 C++ 实现，GPU buffer 管理 |
| `include/vulkan_trainer.h` | VulkanTrainer 类定义 |
| `src/vulkan_trainer.cpp` | VulkanTrainer 实现，参数激活 |
| `src/ply_loader.cpp` | PLY 加载，SH 重排序 |
| `AAA-Gaussians/utils/camera_utils.py` | cameras.json 生成 (确认 R_w2c) |
| `AAA-Gaussians/utils/graphics_utils.py` | getWorld2View2, getProjectionMatrix |
| `AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu` | CUDA forward kernel |
| `AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h` | CUDA loadMatrix4x4, transformPoint |
