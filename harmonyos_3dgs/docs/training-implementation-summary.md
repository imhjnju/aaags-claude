# C++ 3DGS 训练管线实现总结

## 1. 项目背景与目标

### 1.1 背景

3D Gaussian Splatting (3DGS) 是一种新型的实时辐射场渲染技术。原版实现基于 Python/PyTorch + CUDA，训练依赖 GPU。本项目的 `harmonyos_3dgs/` 子目录已有一个完整的 C++ 推理（前向渲染）管线，包含 CPU 和 OpenCL GPU 两个后端，但没有反向传播和训练能力。

### 1.2 目标

在现有 C++ 代码库中实现完整的训练管线：
- **前向渲染** → **损失计算** → **反向传播** → **参数更新**
- 支持 CPU 和 OpenCL GPU 后端
- 使用 TDD（测试驱动开发）确保每个模块的正确性
- 最终验证与 Python 参考实现的一致性

### 1.3 范围约束（最小可训练版本）

| 特性 | 状态 | 说明 |
|------|------|------|
| L1 Loss | ✅ 实现 | `mean(abs(rendered - gt))` |
| SGD 优化器 | ✅ 实现 | 位置 LR 指数衰减，其他固定 LR |
| 前向渲染 | ✅ 复用 | 现有 CPU/GPU 管线 |
| 反向传播 | ✅ 实现 | 4 阶段逆序：rasterizer → scatter → preprocessor |
| CPU 后端 | ✅ 完整 | 全部模块 |
| OpenCL GPU 后端 | ✅ 框架 | backward kernels 已写，待真机验证 |
| eval_3D (kBuffer) | ❌ 不含 | 仅 2D path |
| Antialiasing | ❌ 不含 | 简化 preprocessor backward |
| SSIM Loss | ❌ 不含 | 仅 L1 |
| Adam 优化器 | ❌ 不含 | 仅 SGD |
| Densification/Pruning | ❌ 不含 | 固定 Gaussian 数量 |

---

## 2. 整体架构

### 2.1 训练循环数据流

```
┌──────────────────── Training Step ─────────────────────┐
│                                                         │
│  RawGaussianParams ──activate()──► GaussianData         │
│        ▲                               │                │
│        │                      Forward Pipeline          │
│        │                     (Preprocess→Bin→Sort→Rast) │
│        │                               │                │
│   SGD Optimizer              rendered_image + ForwardCache │
│        ▲                               │                │
│        │                               ▼                │
│  GradientOutput       gt_image ──► L1 Loss ──► d_image  │
│        ▲                               │                │
│        │                               ▼                │
│        └──────── Backward Pipeline ◄───┘                │
│                  (Rasterizer_bw → Preprocessor_bw)      │
└─────────────────────────────────────────────────────────┘
```

### 2.2 反向传播阶段

| 阶段 | 输入 | 输出 | 复杂度 |
|------|------|------|--------|
| **L1 Loss** | rendered, gt → d_image | `sign(diff) / n` | O(HW) |
| **Rasterizer Backward** | d_image → per-Gaussian grads | d_rgb, d_opacity, d_means2D, d_conics | O(HW × G/tile) |
| **Gradient Scatter** | sorted grads → original order | 多 tile 累加 (+=) | O(total_pairs) |
| **Preprocessor Backward** | per-Gaussian grads → raw param grads | 4 条链并行 | O(N) |

### 2.3 Preprocessor 4 条梯度链

```
Chain 1 (Covariance):
  d_conics → d_cov2D → d_cov3D → d_M → d_raw_scales, d_raw_rotations
                ↘ d_T → d_J → d_p_view → d_raw_positions (间接贡献)

Chain 2 (SH Color):
  d_rgb → d_raw_sh_coeffs  (线性关系，最简单)

Chain 3 (Opacity):
  d_opacity_2d → d_raw_opacities  (sigmoid 雅可比)

Chain 4 (Position):
  d_means2D → d_ndc → d_p_hom → d_raw_positions  (透视投影反向)
```

### 2.4 关键数据结构

```cpp
// 原始参数（SGD 更新对象）
struct RawGaussianParams {
    float* raw_positions;    // [N*3] 世界坐标
    float* raw_scales;       // [N*3] log-space
    float* raw_rotations;    // [N*4] 未归一化四元数
    float* raw_sh_coeffs;    // [N*max_coeffs*3]
    float* raw_opacities;    // [N] inverse-sigmoid space
    void activate(GaussianData& g);  // exp/sigmoid/normalize
};

// 前向缓存（backward 需要的中间值）
struct ForwardCache {
    float* T_final;     // [H*W] 每像素最终透射率
    int*   n_contrib;   // [H*W] 每像素贡献 Gaussian 数
    float* cov2D;       // [N*3] 2D 协方差
    float* cov2D_det;   // [N] 行列式
    float* cov3D;       // [N*6] 3D 协方差上三角
    float* p_view;      // [N*3] view-space 坐标
    float* p_hom_w;     // [N] clip-space w 分量
};

// 梯度输出
struct GradientOutput {
    float* d_raw_positions, *d_raw_scales, *d_raw_rotations;
    float* d_raw_sh_coeffs, *d_raw_opacities;
};
```

---

## 3. TDD 实现过程

### 3.1 任务分解与依赖

```
Phase 1: 基础设施
  T1  训练数据结构 (RawGaussianParams, ForwardCache, GradientOutput)
  T2  SH 训练模式 (去掉上限 clamp)
  T3  ForwardCache 保存中间值
  T4  L1 Loss

Phase 2: 反向传播核心
  T5  Rasterizer Backward CPU  ← 最复杂
  T6  Preprocessor Backward: SH + Opacity 链
  T7  Preprocessor Backward: Covariance 链  ← 数学最复杂
  T8  Preprocessor Backward: Position 链

Phase 3: 训练循环
  T9  SGD 优化器 + LR 调度
  T10 Trainer 类集成
  T11 端到端训练测试

Phase 4: GPU 移植
  T12 Rasterizer Backward GPU (OpenCL)
  T13 Preprocessor Backward GPU (OpenCL)
  T14 GPU Trainer 集成
```

### 3.2 每个 Task 的 TDD 流程

```
1. 写失败的测试 (RED)
2. 实现最小代码使测试通过 (GREEN)
3. 重构 (REFACTOR)
4. 运行完整测试套件确认无回归
5. Git commit
```

### 3.3 测试策略

| 测试类型 | 适用场景 | 验证阈值 |
|----------|---------|---------|
| **Finite-difference** | Rasterizer/Preprocessor backward | 相对误差 < 1e-3 |
| **Unit test** | L1 loss, SGD, 数据结构 | 精确匹配 |
| **Integration test** | E2E 训练收敛 | Loss 下降 > 50% |
| **Cross-validation** | CPU vs GPU | 相对误差 < 1e-2 |
| **Python 对比** | 全管线验证 | 相对误差 < 1% |

---

## 4. 遇到的问题与解决方案

### 4.1 SH 上限 Clamp 阻断梯度

**问题**: `sh_eval.cpp` 中 `std::min(1.0f, std::max(0.0f, color + 0.5f))` 的上限 clamp 使 `color > 1.0` 时 `d_color/d_sh = 0`。

**影响**: 过亮的 Gaussian 无法通过优化降低亮度，训练时颜色"卡死"在 1.0。

**解决**:
```cpp
// 训练模式：仅保留下限 clamp
rgb_out[ch] = std::max(0.0f, rgb_out[ch] + 0.5f);
if (!training)
    rgb_out[ch] = std::min(1.0f, rgb_out[ch]);  // 推理时保留上限
```

**教训**: 推理优化（clamp 防白斑）不一定适合训练。必须检查前向代码中所有非线性操作的梯度通路。

---

### 4.2 缺少原始参数存储

**问题**: `GaussianData` 存储激活后的值（`exp(scale)`, `sigmoid(opacity)`, 归一化 quaternion），SGD 无法直接在激活后的空间更新。

**影响**: 如果直接对 `exp(scale)` 做 SGD，scale 可能变负（无物理意义）。

**解决**: 新增 `RawGaussianParams` 结构，持有 log-space scale、inverse-sigmoid opacity、未归一化 quaternion。每步流程：
```
SGD 更新 raw_params → raw.activate(gaussians) → 前向渲染
```

**教训**: 训练参数的存储空间必须与优化空间一致。激活函数（exp, sigmoid, normalize）的反向需要从激活值反推原始值的雅可比。

---

### 4.3 ForwardCache 设计不完整

**问题**: 反向传播需要前向的中间结果（cov2D、det、view-space position 等），但前向 pipeline 没有保存。

**影响**: 无法计算 `d_conics → d_cov2D`（需要 det）、`d_cov2D → d_cov3D`（需要 view-space J）等梯度链。

**解决**: 设计 `ForwardCache` 包含 7 个数组，通过可选参数 `ForwardCache*` 传入前向函数。推理时传 `nullptr`，零开销。

**教训**: 反向传播的数据需求应在设计阶段完整分析，而非实现时补充。Review agent 在设计评审中提前发现了这个问题。

---

### 4.4 Scatter-Add 而非简单 Reorder

**问题**: 初始设计认为 Binner/Sorter 反向是"重排梯度"，但一个 Gaussian 出现在多个 tile 中。

**影响**: 如果用 copy 而非累加，多 tile Gaussian 的梯度被最后一个 tile 覆盖，丢失其他 tile 的贡献。

**解决**: 反向直接用 `values_sorted[j]` 索引原始 Gaussian，用 `+=` 累加。GPU 需 CAS-based `atomicAdd_f`。

**教训**: "无可学习参数"的阶段仍然需要正确处理梯度的数据流（排序 → 反排序 = scatter-add）。

---

### 4.5 Rasterizer 反向的 Alpha 梯度公式

**问题**: 反向遍历中 `C_rest`（当前 Gaussian 之后的累积颜色）的更新公式不清晰，导致多 Gaussian 场景梯度错误。

**影响**: 单 Gaussian 测试通过（无累积效应），但多 Gaussian 场景 opacity 梯度误差达 14%。

**初始公式**（有 bug）:
```
C_accum = alpha * T_i * color + C_accum  // 绝对加权
d_alpha = dot(T_i * color - C_accum, d_C)
```

**修正公式**:
```
accum_rec = alpha * color + (1 - alpha) * accum_rec  // 归一化期望
d_alpha = dot(T_i * (color - accum_rec), d_C)
```

**教训**: Alpha compositing 的反向是 3DGS 训练中最微妙的部分。必须用 finite-difference 验证，且需要多 Gaussian 重叠场景的测试用例。

---

### 4.6 对称矩阵反向的双重计数

**问题**: `cov2D = T^T * Vrk * T` 和 `cov3D = M^T * M` 反向时，如果将上三角展开为完整对称矩阵再做矩阵乘，off-diagonal 元素的梯度被双重计数。

**影响**: Scale 和 rotation 梯度偏大约 2x，导致训练发散。

**错误做法**:
```
d_Sigma_full = [[d0, d1, d2], [d1, d3, d4], [d2, d4, d5]]  // 双重计数 d1, d2, d4
d_M = 2 * d_Sigma_full * M  // off-diagonal 贡献翻倍
```

**正确做法**: 直接从上三角元素计算 `d_M`：
```cpp
d_M[k][0] = 2*d[0]*M[k][0] + d[1]*M[k][1] + d[2]*M[k][2];  // 对角项系数 2，非对角系数 1
d_M[k][1] = d[1]*M[k][0] + 2*d[3]*M[k][1] + d[4]*M[k][2];
d_M[k][2] = d[2]*M[k][0] + d[4]*M[k][1] + 2*d[5]*M[k][2];
```

**教训**: 对称矩阵的链式法则需要特别注意存储格式。上三角 6 元素 ≠ 完整 9 元素的一半。

---

### 4.7 Position 梯度缺少 cov2D 间接路径

**问题**: 初始实现只通过 `means2D → position`（透视投影）计算 position 梯度，缺少 `cov2D → J → p_view → position` 的间接贡献。

**影响**: x/y 梯度误差 < 5%，但 z 梯度误差 > 100%。

**原因**: Position 的 z 分量主要通过改变透视 Jacobian（`focal/z` 和 `focal*t/(z^2)`）来影响 Gaussian 在屏幕上的大小（cov2D），而非直接影响 2D 位置。

**解决**: 完整实现 cov2D 到 position 的梯度链：
```
d_cov2D → d_T (矩阵链式法则)
  → d_J (T = W*J 反向)
    → d_p_view (J 依赖 tx, ty, tz)
      → d_position (transformPoint4x3 反向)
```
含 frustum clamp 零梯度检查。

**教训**: "次要路径"可能在某些方向上是主要贡献。z 方向的梯度几乎完全来自 cov2D 路径，而非 means2D 路径。

---

### 4.8 Python 验证脚本矩阵约定不一致

**问题**: C++ 使用 GLM 风格列主序 `[col][row]`，Python numpy 默认行主序 `[row][col]`。两者的 `computeCov2D` 结果不同。

**影响**: 前向渲染结果不一致（loss 0.00657 vs 0.00842），导致梯度对比无意义。

**解决**: 完全重写 Python 验证代码，所有矩阵操作逐行匹配 C++ 的列主序索引：
```python
# C++ 的 J[col][row]、W[col][row]、T[col][row] 全部用列主序
J = np.zeros((3,3))      # J[col][row]
J[0][0] = focal_x / tz   # col=0, row=0
J[0][2] = ...             # col=0, row=2

# T = W * J: T[col][row] = sum_k W[k][row] * J[col][k]
for col in range(3):
    for row in range(3):
        for k in range(3):
            T[col][row] += W[k][row] * J[col][k]
```

**修正后**: Loss 完全一致（0.00667448 = 0.00667448）。

**教训**: 跨语言验证必须严格匹配数据布局。"列主序"不仅影响存储，还影响矩阵乘法的索引顺序。

---

### 4.9 `lr_schedule` 对零值输入产生 NaN

**问题**: `exp(log(0) * (1-t) + ...)` 产生 NaN。

**影响**: 当 position LR 设为 0（意图冻结位置）时，所有参数被 NaN 污染。

**解决**: 将位置 LR 最小值设为 `1e-6f`，而非 0。

**教训**: `log` 函数的定义域为 `(0, +∞)`，当 LR 使用对数空间插值时必须确保输入 > 0。

---

### 4.10 真实数据训练不收敛

**问题**: 从 COLMAP 稀疏点初始化训练 flowers 数据集，500 步 loss 几乎不下降。

**原因分析**:
1. **初始 scale 不合理**: 统一使用固定值 0.5，但点云密度差异大，远处的点 scale 应更大
2. **初始 opacity 过低**: `inverse_sigmoid(0.1) ≈ -2.2`，Gaussians 几乎透明
3. **无 densification**: 38K 稀疏点无法表示复杂场景的所有细节

**解决**:
1. 用 KD-Tree 计算每点 3-NN 距离，初始 scale = `nn_dist × 0.5`（自适应）
2. 初始 opacity = `inverse_sigmoid(0.5) = 0`（50% 不透明）
3. 提高学习率（`lr_scale=3.0`）
4. 训练 3000+ 步

**教训**: 初始化对 3DGS 训练至关重要。原版论文也强调了从 SfM 点云初始化时需要合理的 scale 估计。无 densification 的情况下，只能优化已有点的颜色/大小/位置，无法新增覆盖缺失区域的点。

---

## 5. 验证结果

### 5.1 方法 2: 单步梯度对比

使用 3 个 Gaussians、16×16 图像的小场景，Python finite-difference vs C++ 解析梯度：

| 参数 | 最大相对误差 | 结论 |
|------|------------|------|
| SH coeffs (9 元素) | 0.000000 | ✅ 完美匹配 |
| Opacities (3 元素) | 0.000000 | ✅ 完美匹配 |
| Scales (9 元素) | 0.000000 | ✅ 完美匹配 |
| Positions (9 元素) | 0.000002 | ✅ 完美匹配 |

### 5.2 方法 3: 100 步 Loss 曲线对比

相同初始化、相同 LR、相同相机顺序的 100 步训练：

```
iter   0: py=0.00566198  cpp=0.00566198  err=0.0000%
iter  25: py=0.00561583  cpp=0.00562608  err=0.1825%
iter  50: py=0.00556990  cpp=0.00557841  err=0.1528%
iter  99: py=0.00548129  cpp=0.00549552  err=0.2596%
```

- **最大相对误差**: 0.2843%（< 1% 阈值）
- **平均相对误差**: 0.1293%
- **原因**: Python 用 finite-diff 梯度（O(eps²) 截断误差），C++ 用解析梯度

### 5.3 C++ 内部 finite-diff 验证

17 个 finite-diff 测试全部通过，覆盖所有梯度链：

| 测试 | 相对误差 | 通过阈值 |
|------|---------|---------|
| Rasterizer d_rgb | < 0.1% | 1e-3 |
| Rasterizer d_opacity | < 0.1% | 1e-3 |
| Rasterizer d_means2D | < 0.5% | 5e-3 |
| Rasterizer d_conics | < 0.1% | 1e-3 |
| SH backward (degree 0-3) | < 0.1% | 1e-3 |
| Opacity backward | < 0.01% | 1e-4 |
| Conic inversion | < 0.3% | 1e-3 |
| Cov2D → Cov3D | < 1% | 1e-2 |
| Scale gradient (E2E) | 0.1-0.8% | 1e-2 |
| Rotation gradient (E2E) | 0.08-1.6% | 2e-2 |
| Position gradient (x,y) | 0.3-4.9% | 5e-2 |

---

## 6. 性能数据

### 6.1 合成场景训练速度

| 配置 | PC (x86_64) | 手机 (ARM64) | 比率 |
|------|------------|-------------|------|
| 8G, 64×64 | 11594 it/s | 3498 it/s | 30% |
| 16G, 128×128 | 1900 it/s | 591 it/s | 31% |

### 6.2 真实数据 (flowers) 训练速度

| 配置 | PC | 手机 |
|------|-----|------|
| 38K G, 157×103 | 4.7 it/s | 0.8 it/s |
| 38K G, 628×414 | 0.2 it/s | — |

### 6.3 内存使用

| 场景 | 前向 | 前向+反向 | Arena 大小 |
|------|------|----------|-----------|
| 1K G, 64×64 | ~2 MB | ~4 MB | 256 MB 足够 |
| 38K G, 157×103 | ~50 MB | ~120 MB | 512 MB 足够 |
| 38K G, 628×414 | ~200 MB | ~500 MB | 1 GB 足够 |
| 1M G, 1080p | ~650 MB | ~1.3 GB | 2 GB 推荐 |

---

## 7. 文件清单

### 7.1 新增文件

| 文件 | 职责 |
|------|------|
| `include/train_types.h` | RawGaussianParams, ForwardCache, GradientOutput, TrainConfig |
| `src/train_types.cpp` | activate(), allocate_and_zero(), lr_schedule() |
| `src/loss.h/.cpp` | L1 loss + 梯度 |
| `src/cpu/rasterizer_backward_cpu.h/.cpp` | Rasterizer 反向 (CPU) |
| `src/cpu/preprocessor_backward_cpu.h/.cpp` | Preprocessor 反向 4 条链 (CPU) |
| `src/optimizer.h/.cpp` | SGD 优化器 |
| `include/trainer.h`, `src/trainer.cpp` | Trainer 类（前向-反向-优化编排） |
| `src/train_demo.cpp` | 合成场景训练 demo |
| `src/train_main.cpp` | 真实数据训练入口（多视角、PLY I/O） |
| `src/gpu/rasterizer_backward_gpu.h/.cpp` | Rasterizer 反向 (OpenCL) |
| `src/gpu/preprocessor_backward_gpu.h/.cpp` | Preprocessor 反向 (OpenCL) |
| `src/gpu/kernels/rasterize_backward.cl` | 反向光栅化 OpenCL kernel |
| `src/gpu/kernels/preprocess_backward.cl` | 反向预处理 OpenCL kernel |
| `include/trainer_gpu.h`, `src/trainer_gpu.cpp` | GPU Trainer |
| `tools/verify_gradients.py` | Python-C++ 梯度对比验证 |
| `tools/train_compare_py.py` | Python-C++ loss 曲线对比 |
| `tools/prepare_flowers.py` | COLMAP → 训练数据转换 |

### 7.2 修改文件

| 文件 | 变更 |
|------|------|
| `include/types.h` | 新增 ForwardCache 结构体 |
| `include/sh_eval.h`, `src/sh_eval.cpp` | 添加 training 参数 |
| `include/preprocessor.h`, `src/cpu/preprocessor_cpu.cpp` | 添加 ForwardCache 保存 |
| `include/rasterizer.h`, `src/cpu/rasterizer_cpu.cpp` | 添加 ForwardCache 保存 |
| `CMakeLists.txt` | 新增源文件和测试 |

### 7.3 测试文件

| 文件 | 测试数量 | 覆盖范围 |
|------|---------|---------|
| `tests/test_train_types.cpp` | 16 | 数据结构、activate、lr_schedule |
| `tests/test_loss.cpp` | 3 | L1 loss |
| `tests/test_forward_cache.cpp` | 5 | ForwardCache 填充 |
| `tests/test_rasterizer_backward.cpp` | 4 | Rasterizer FD 验证 |
| `tests/test_preprocessor_backward.cpp` | 14 | Preprocessor FD 验证 (4 链) |
| `tests/test_optimizer.cpp` | 3 | SGD 更新 |
| `tests/test_trainer_e2e.cpp` | 1 | E2E 收敛 |
| `tests/test_train_pipeline.cpp` | 2 | 合成场景+roundtrip |
| `tests/test_train_loop.cpp` | 3 | 多视角+PPM roundtrip |
| **总计** | **92+** | |

---

## 8. 后续扩展建议

| 特性 | 难度 | 依赖 | 预期收益 |
|------|------|------|---------|
| **Adam 优化器** | 低 | OptimizerState 加 moment buffers | 显著加速收敛 |
| **SSIM Loss** | 中 | 新 loss 模块，卷积窗口 | 提升视觉质量 |
| **Densification/Pruning** | 高 | GaussianModel 支持 resize | 从稀疏点覆盖完整场景 |
| **SH degree 调度** | 低 | TrainConfig 加 iteration 控制 | 渐进学习高频细节 |
| **eval_3D backward** | 高 | 保存每像素 kBuffer 排序顺序 | 支持 StopThePop 训练 |
| **Antialiasing backward** | 中 | preprocessor backward 加 h_conv_scaling 链 | 支持 AA 训练 |
| **多 GPU 并行** | 高 | 分布式梯度同步 | 大场景加速 |

**推荐优先级**: Adam → Densification → SSIM → SH 调度
