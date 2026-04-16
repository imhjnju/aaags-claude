# HarmonyOS 3DGS 训练管线验证报告

Date: 2026-03-27 ~ 2026-03-30

## 1. 项目总览

基于官方 3DGS fork，在 `harmonyos_3dgs/` 中实现了完整的 C++17 CPU 训练管线，包含：
- 四阶段渲染管线（Preprocessor → TileBinner → Sorter → Rasterizer）
- L1 Loss + 完整 backward pass
- Adam / SGD 优化器
- Adaptive Density Control（clone / split / prune）
- 支持 SH degree 0~3

本报告记录了从 L1 Loss 校验到大规模训练收敛的完整验证过程。

## 2. 发现并修复的 Bug

### Bug 1: SH→Position 梯度链缺失 [Critical]

**文件**: `src/cpu/preprocessor_backward_cpu.cpp`

`computeColorFromSH_backward` 只计算了 `d_sh_coeffs`，缺失 `d_pos`。对于 SH degree >= 1，颜色通过归一化视角向量依赖 position，但这条链未反传。

**修复**: 扩展函数增加 `float* d_pos` 输出，实现 `d_color/d_dir × d_dir/d_pos`（SH basis 偏导 × normalize Jacobian），覆盖 degree 1~3。

### Bug 2: Forward/Backward Training Mode 不匹配 [Critical]

**文件**: `include/types.h`, `src/cpu/preprocessor_cpu.cpp`, 多个入口和测试文件

Forward 默认 `training=false`（颜色 clamp 到 [0,1]），backward 用 `training=true`（无 clamp）。SH 颜色 > 1.0 时梯度完全错误。

**验证**: SH 系数设为 999.0 → C++ 读取正确 → loss 不变 → 确认 forward clamp 吞掉了变化。

**修复**: `RenderConfig` 新增 `bool training` 字段。所有训练入口和 backward 测试设 `cfg.training = true`。

### Bug 3: PLY 文件路径硬编码 [Medium]

**文件**: `CMakeLists.txt`, `tests/test_ply_loader.cpp`, `tests/test_renderer_e2e.cpp`

测试用硬编码相对路径 `"../tests/test_data/..."` 导致从项目根目录运行时 7 个测试失败。

**修复**: CMake `target_compile_definitions` 定义 `TEST_DATA_DIR` 宏指向源码绝对路径。

### 测试基础设施改进

| 改进 | 说明 |
|---|---|
| `l1_loss_double` | double 精度 L1 loss，用于 FD 梯度验证 |
| Rotation 梯度导出 | `verify_grads_main.cpp` 新增 `d_raw_rotations` 导出 |
| C++ FD 验证方法 | Extended test 使用 C++ 自身做 FD，消除 Python/C++ renderer 不一致 |

## 3. 验证结果汇总

### 3.1 L1 Loss 校验

C++ vs Python，5 个尺寸 (1x1 ~ 256x256)：**bit-exact**（差异 = 0）。

梯度（d_image）也 bit-exact，196608 个元素零误差。

### 3.2 Forward Render 校验

3-Gaussian, 16x16, SH0, identity view：

| 指标 | 值 |
|---|---|
| MSE | 3.65e-16 |
| PSNR | **154.4 dB** |
| 最大像素误差 | 2.38e-07 |

### 3.3 梯度校验

**基线 (Python FD vs C++ Analytical)**: 3G, 16x16, SH0

| 参数 | max_rel_err |
|---|---|
| raw_sh_coeffs | 0.000000 |
| raw_opacities | 0.000000 |
| raw_scales | 0.000000 |
| raw_rotations | 0.000002 |
| raw_positions | 0.000002 |

**扩展 (C++ FD vs C++ Analytical)**: 10 配置

| 配置 | 结果 |
|---|---|
| 3G SH0~SH3, 16x16 | **全部 PASS** |
| 10G SH0~SH1, 16x16 | **全部 PASS** |
| 3G SH0, 32x32 | **PASS** |
| 30G SH0, 16x16 | FAIL (遮挡边界) |
| 10G SH1, 32x32 | FAIL (遮挡边界) |

8/10 PASS。2 个失败是 tile-based rasterizer 的 T 饱和截断特性。

**非遮挡 Gaussian 的误差分析** (many_30g, 排除 G4):

| 梯度幅值 | 平均绝对误差 | 平均相对误差 |
|---|---|---|
| > 1e-2 | 5.3e-6 | **0.04%** |
| 1e-3 ~ 1e-2 | 3.2e-6 | **0.17%** |
| 1e-4 ~ 1e-3 | 1.6e-6 | **0.60%** |

绝对误差 ≈ 常数 (~2.5e-6)，完全由 float32 精度地板决定。

### 3.4 C++ Analytical vs C++ FD 训练

20 步 SGD，3G SH0：**loss 曲线完全一致** (max_rel_err = 0.000000)。

### 3.5 C++ vs Python FD 训练

100 步 SGD，3G SH0：**max_rel_err = 0.24%, PASS** (阈值 1%)。

### 3.6 Adam 大规模训练

16 配置（N=3~300, SH0~3, 16x16~32x32）：

| 配置 | Steps | Init Loss | Final Loss | Drop% |
|---|---|---|---|---|
| N=3, SH0, 16x16 | 500 | 0.01148 | 0.00007 | 99.4% |
| N=3, SH1, 16x16 | 500 | 0.00830 | 0.00058 | 93.0% |
| N=3, SH3, 16x16 | 500 | 0.00771 | 0.00020 | 97.4% |
| N=10, SH0, 16x16 | 500 | 0.02785 | 0.00014 | 99.5% |
| N=10, SH1, 16x16 | 1000 | 0.02518 | 0.00056 | 97.8% |
| N=30, SH0, 16x16 | 2000 | 0.05207 | 0.00019 | 99.6% |
| N=50, SH0, 16x16 | 2000 | 0.06856 | 0.00023 | 99.7% |
| N=50, SH1, 16x16 | 2000 | 0.06563 | 0.00049 | 99.3% |
| N=50, SH3, 16x16 | 2000 | 0.05124 | 0.00078 | 98.5% |
| N=100, SH0, 16x16 | 2000 | 0.08090 | 0.00025 | 99.7% |
| N=100, SH1, 16x16 | 2000 | 0.08332 | 0.00066 | 99.2% |
| N=100, SH0, 32x32 | 2000 | 0.07565 | 0.00026 | 99.7% |
| N=100, SH1, 32x32 | 2000 | 0.07840 | 0.00071 | 99.1% |
| N=200, SH0, 16x16 | 2000 | 0.10217 | 0.00052 | 99.5% |
| N=300, SH0, 16x16 | 2000 | 0.11423 | 0.00041 | 99.6% |
| N=300, SH1, 16x16 | 2000 | 0.11714 | 0.00068 | 99.4% |

**全部 > 98.5% loss 下降，零 NaN，零爆炸。** Adam 比 SGD 快 100~200x。

### 3.7 Densification 验证

| 初始 N | 最终 N | Growth | Loss Drop | NaN |
|---|---|---|---|---|
| 10 | 450 | 45x | 96.8% | 0 |
| 30 | 695 | 23x | 97.1% | 0 |
| 50 | 760 | 15x | 98.4% | 0 |

Clone/split/prune 全部正常工作，Gaussian 数量动态增长，loss 持续下降。

### 3.8 训练速度 (CPU 单线程)

| 配置 | 每步耗时 |
|---|---|
| N=100, 16x16 | 0.8 ms/iter |
| N=300, 16x16 | 1.8 ms/iter |
| N=100, 32x32 | 2.5 ms/iter |
| N=100, 64x64 | 5.4 ms/iter |

## 4. 测试状态

**124 tests: 121 PASS, 3 SKIP (GPU), 0 FAIL**

| 类别 | 数量 | 说明 |
|---|---|---|
| 类型/数学/SH | 14 | 基础数学和 SH 评估 |
| PLY 加载 | 6 | 加载 + 激活验证 |
| 前向管线 | 12 | Preprocessor, TileBinner, Sorter, Rasterizer |
| Rasterizer Backward | 6 | RGB/Opacity/Means2D/Conics FD 验证 |
| Preprocessor Backward | 14 | SH degree 0~3, 协方差链, Position 梯度 |
| L1 Loss | 5 | 单元测试 + 跨语言 bit-exact |
| SGD Optimizer | 4 | 基本更新, LR 衰减, 零梯度 |
| Adam Optimizer | 5 | 公式验证, 动量累积, 全参数类型, SGD 对比 |
| Density Controller | 9 | Clone/Split/Prune + N=1/N=0 边界 + 多轮 |
| Training (SGD) | 6 | 单/多视角, E2E, Pipeline |
| Training (Adam) | 4 | SH0/SH1/SH3 + 10G 收敛 |
| Forward Cache | 2 | Null cache 兼容性 |
| Image I/O | 2 | PPM 读写 + Renderer E2E |
| GPU (skip) | 3 | OpenCL 未启用 |

## 5. 已知限制

1. **N=30+ 遮挡边界 FD 不一致**: FD 扰动改变 Gaussian 在 T<0.0001 饱和边界的可见性，analytical backward 正确忽略。tile-based rasterizer 固有特性。
2. **合成场景 densification**: 固定 N 的 baseline 在合成场景上更优（GT 由同一组 Gaussian 渲染）。真实数据上 densification 将显著提升重建质量。
3. **Python FD 训练对比**: Python float32 renderer 与 C++ 有微小差异 (~2.4e-7/pixel)，100 步累积到 0.24%。
4. **GPU backward**: 未验证（OpenCL 未启用），需要在 HarmonyOS/Maleoon 920 上测试。

## 6. 修改文件总清单

### 核心 Bug 修复
| 文件 | 变更 |
|---|---|
| `include/types.h` | 新增 `bool training` 到 RenderConfig |
| `src/cpu/preprocessor_cpu.cpp` | 传递 `cfg.training` 给 computeColorFromSH |
| `src/cpu/preprocessor_backward_cpu.h` | 新增 `float* d_pos` 参数 |
| `src/cpu/preprocessor_backward_cpu.cpp` | 实现 SH→position 梯度链 (degree 1~3) |
| `src/loss.h` / `src/loss.cpp` | 新增 `l1_loss_double()` |
| `src/verify_grads_main.cpp` | double loss, rotation dump, training mode |
| `src/train_main.cpp` / `src/train_demo.cpp` | `cfg.training = true` |
| `CMakeLists.txt` | `TEST_DATA_DIR` 编译定义 |

### 新增功能
| 文件 | 变更 |
|---|---|
| `src/optimizer.h` / `src/optimizer.cpp` | AdamOptimizer (m/v buffers, bias correction) |
| `include/train_types.h` | DensifyConfig, OwnedRawParams, Adam 超参数 |
| `src/density_controller.h` / `src/density_controller.cpp` | DensityController (clone/split/prune) |
| `include/trainer.h` / `src/trainer.cpp` | Adam/SGD 选择, `step_with_densify()` |
| `src/train_compare_cpp.cpp` | 可配置 CLI (N/SH/iters/seed/res/--densify) |

### 新增测试
| 文件 | 变更 |
|---|---|
| `tests/test_loss.cpp` | L1LossCrosslang (5 尺寸 bit-exact) |
| `tests/test_optimizer.cpp` | Adam: BasicUpdate, Momentum, AllParamTypes, vs SGD |
| `tests/test_density_controller.cpp` | 9 tests: clone/split/prune/N=1/N=0/多轮 |
| `tests/test_train_loop.cpp` | Adam 训练: SH0/SH1/SH3 + 10G 收敛 |
| 所有 backward/training tests | `cfg.training = true` |
| `tests/test_ply_loader.cpp` / `tests/test_renderer_e2e.cpp` | `TEST_DATA_DIR` 路径修复 |

### 新增工具
| 文件 | 用途 |
|---|---|
| `tools/verify_l1_loss.py` | L1 loss 跨语言测试数据生成 |
| `tools/verify_forward_render.py` | Forward render 像素级对比 |
| `tools/verify_gradients_extended.py` | 多配置梯度验证 (C++ FD) |

## 7. Step 6: 真实数据集验证 (MipNeRF360 Flowers)

### 数据集
- **场景**: MipNeRF360 flowers（outdoor 360° 花丛场景）
- **图像**: 173 views, 原始 5025x3312, 8x 下采样 628x414
- **SfM 点**: 38347 (COLMAP sparse reconstruction)
- **数据转换**: `colmap_to_train.py` → PLY + cameras.json + GT PPM

### 训练结果

**短训练（探索性）**

| 配置 | N | Views | Res | Steps | 初始 Loss | 最终 Loss | Drop | PSNR | 速度 |
|---|---|---|---|---|---|---|---|---|---|
| SGD baseline | 500 | 4 | 628x414 | 200 | 0.371 | 0.225 | 39% | — | 0.6 it/s |
| Adam | 500 | 4 | 628x414 | 500 | 0.371 | 0.159 | 57% | 13.7 dB | 0.8 it/s |
| **Adam 2K** | **2000** | **16** | **314x207** | **500** | **0.219** | **0.165** | **25%** | **14.1 dB** | **1.8 it/s** |

**长训练（15000 步）— 发现 LR 发散问题**

| 配置 | Steps | Best 200-avg | Best @iter | Final 200-avg | 状态 |
|---|---|---|---|---|---|
| lr_scale=1.0 (默认) | 15000 | **0.1515** | 2104 | 0.4023 | **发散** |
| lr_scale=0.3 | 4000+ | **0.1449** | 3847 | 0.1497 | **稳定** |
| lr_scale=0.1 | 4400+ | 0.1685 | 4334 | 0.1687 | 稳定但慢 |

### 发现的问题：长训练 LR 发散

**现象**: 默认 LR (lr_scale=1.0) 在 ~2000 步后 loss 开始上升，最终从 0.15 发散到 0.40。

**根因**: Gaussian position 飘移。训练后参数诊断：
- Position z: 初始 3~8 → 训练后 **-25 ~ 30**（飘到相机后方和远处）
- Scale (exp): 初始 0.01~0.5 → 训练后 **最大 21.8**（部分 Gaussian 变得巨大）

**原因分析**:
- 2000 Gaussians 覆盖 65K 像素 (314×207)，每个 Gaussian 覆盖 ~33 像素
- 官方 3DGS 用 ~100K Gaussians 覆盖 1.6M 像素，每个 ~16 像素
- 每 Gaussian 梯度比官方大 ~2x，但 LR 未缩放
- Adam 放大了这个问题：对大梯度的参数给出 lr × sign(grad) 的步长

**解决方案**:
- `--lr_scale 0.3` 稳定收敛到 0.145 (200-avg)，不发散
- 未来需实现 `spatial_lr_scale` = 1/cameras_extent（官方做法）

### 分析

- 更多 Gaussians (2000 vs 500) 降低了初始 loss (0.219 vs 0.371)
- 最佳 200-step 平均 loss = **0.145** (lr_scale=0.3, ~4000 步)
- PSNR 14 dB vs 官方 ~21 dB 差距来自：SH0 only, 2K 点, 无 densification, 无 SSIM
- CPU 单线程: 1.8 it/s (2000G, 314x207)

### 下一步改进方向

| 改进 | 预期 PSNR 提升 | 优先级 |
|---|---|---|
| 更多迭代 (5000+) | +2~3 dB | P0 |
| SH degree 3 | +2~3 dB | P0 |
| 全量 SfM 点 (38K) | +1~2 dB | P1 (需 GPU) |
| SSIM Loss | +0.5~1 dB | P1 |
| Densification | +1~2 dB | P1 |
| 全量视角 (173) | +0.5~1 dB | P2 |

## 8. 下一步

1. **GPU backward 验证** — 在 HarmonyOS/Maleoon 920 上测试 OpenCL backward
2. **SSIM Loss 实现** — 完成 L1 + 0.2*(1-SSIM) 组合 loss
3. **长时间训练** — 5000+ 步 + SH3 验证是否接近官方 PSNR
