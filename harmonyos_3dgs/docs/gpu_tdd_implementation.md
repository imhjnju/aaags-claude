# GPU 算子 TDD 开发文档

## 概述

本文档记录了基于 TDD（测试驱动开发）模式的 3DGS 渲染器 GPU 算子实现过程，确保 GPU 实现与 CPU 实现严格一致。

## 开发原则

1. **测试先行**：在实现每个 GPU 算子之前，先编写 CPU-GPU 对比测试
2. **数值一致性**：GPU 输出与 CPU 输出的相对误差应 < 1e-4
3. **行为一致性**：训练模式和推理模式的差异处理（如 SH 颜色上限钳制）
4. **完整覆盖**：前向渲染和反向梯度传播都需要测试

## 已实现的 GPU 算子

### 1. Preprocessor Forward (GPU)

**文件**: `src/gpu/kernels/preprocess.cl`, `src/gpu/preprocessor_gpu.cpp`

**功能**: 3D→2D 投影、协方差计算、SH 颜色评估、视锥剔除

**测试**: `tests/test_preprocessor_gpu.cpp`
- `MatchesCPU_Basic`: 基础场景测试（5 个高斯球）
- `MatchesCPU_2DGaussian`: elongated 2D 高斯球测试
- `MatchesCPU_Eval3D`: 3D 评估路径测试（AAA-Gaussians）
- `Culling_BehindCamera`: 背后高斯球剔除测试
- `Culling_NearPlane`: 近平面高斯球测试

**验证结果**: 
- means2D: 相对误差 < 1e-4 ✓
- depths: 相对误差 < 1e-5 ✓
- conics: 相对误差 < 1e-4 ✓
- rgb: 相对误差 < 1e-4 ✓
- opacities_2d: 相对误差 < 1e-4 ✓
- radii: 完全匹配 ✓
- tiles_touched: 完全匹配 ✓

### 2. Preprocessor Backward (GPU)

**文件**: `src/gpu/kernels/preprocess_backward.cl`, `src/gpu/preprocessor_backward_gpu.cpp`

**功能**: 预处理器反向梯度传播，包含 4 条梯度链：
- Chain 1: d_conics → d_cov2D → d_cov3D → d_scales, d_rotations
- Chain 2: d_rgb → d_sh_coeffs
- Chain 3: d_opacity_2d → d_raw_opacity
- Chain 4: d_means2D → d_positions

**测试**: `tests/test_preprocessor_backward_gpu.cpp`
- `MatchesCPU`: CPU-GPU 梯度对比
- `SH_Degree0`: SH 度为 0 的最简情况测试
- `CulledGaussians`: 验证被剔除高斯球（radii<=0）梯度为零

**验证结果**:
- d_raw_positions: 相对误差 < 1e-2 ✓
- d_raw_scales: 相对误差 < 1e-2 ✓
- d_raw_rotations: 相对误差 < 1e-2 ✓
- d_raw_sh_coeffs: 相对误差 < 1e-2 ✓
- d_raw_opacities: 相对误差 < 1e-2 ✓

### 3. Rasterizer Forward (GPU)

**文件**: `src/gpu/kernels/rasterize.cl`, `src/gpu/rasterizer_gpu.cpp`

**功能**: 基于 Tile 的 Alpha 混合渲染，支持：
- 标准 2D 圆锥路径
- AAA-Gaussians 3D 评估路径（eval_3D）
- StopThePop per-pixel 深度排序（k-Buffer K=16）

**测试**: `tests/test_rasterizer_gpu.cpp`
- `MatchesCPU_SingleGaussian`: 单高斯球测试
- `MatchesCPU_MultipleGaussians`: 多高斯球（5 个）测试
- `LowAlphaGaussian`: 低 Alpha 高斯球剔除测试（alpha < 1/255）
- `TSaturation`: T 值饱和早停测试（T < 0.0001）

**验证结果**:
- 输出图像：最大绝对误差 < 1e-4 ✓

### 4. Rasterizer Backward (GPU)

**文件**: `src/gpu/kernels/rasterize_backward.cl`, `src/gpu/rasterizer_backward_gpu.cpp`

**功能**: 光栅化反向梯度传播
- 前向重放收集贡献者
- 反向遍历计算梯度
- CAS 原子浮点累加

**测试**: `tests/test_rasterizer_backward_gpu.cpp`
- `MatchesCPU`: CPU-GPU 梯度对比（8x8 图像，3 个高斯球）
- `SingleGaussian`: 单高斯球梯度验证
- `DepthOrdering`: 深度排序验证（前方高斯球梯度更大）

**验证结果**:
- d_means2D: 相对误差 < 1e-2 ✓
- d_conics: 相对误差 < 1e-2 ✓
- d_rgb: 相对误差 < 1e-2 ✓
- d_opacities_2d: 相对误差 < 1e-2 ✓

## 关键修复

### SH 颜色钳制行为修复

**问题**: GPU 的 `computeColorFromSH` 在训练模式下错误地应用了上限钳制（clamp to 1.0），导致与 CPU 行为不一致。

**修复**: 
1. 更新 `computeColorFromSH` 函数签名，添加 `training` 参数
2. 仅在推理模式（!training）下应用上限钳制
3. 训练模式下仅应用下限钳制（max(0, x+0.5)）

**代码变更**:
```cpp
// 旧代码（错误）
rgb_out[0]=fmin(1.0f,fmax(0.0f,rgb_out[0]+0.5f));

// 新代码（正确）
for (int ch=0;ch<3;ch++) {
    rgb_out[ch] = fmax(0.0f, rgb_out[ch] + 0.5f);
    if (!training) {
        rgb_out[ch] = fmin(1.0f, rgb_out[ch]);
    }
}
```

## 测试构建和运行

### 编译

```bash
# Desktop build with OpenCL and tests
cmake -B build -S harmonyos_3dgs \
    -DBUILD_TESTS=ON \
    -DENABLE_OPENCL=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 运行测试

```bash
# 运行所有测试
./build/gs3d_tests

# 运行特定 GPU 测试套件
./build/gs3d_tests --gtest_filter="PreprocessorGPU.*"
./build/gs3d_tests --gtest_filter="RasterizerGPU.*"
./build/gs3d_tests --gtest_filter="*BackwardGPU.*"

# 运行单个测试
./build/gs3d_tests --gtest_filter="PreprocessorGPU.MatchesCPU_Basic"
```

## 数值精度分析

### 误差来源

1. **浮点舍入**: GPU OpenCL 和 CPU x86 浮点运算顺序不同
2. **共享内存**: GPU 使用 shared memory 批量加载数据
3. **原子操作**: 反向传播使用 CAS 原子浮点累加

### 可接受误差范围

| 类型 | 绝对误差容限 | 相对误差容限 |
|------|-------------|-------------|
| 前向输出（颜色） | 1e-4 | 1e-3 |
| 前向输出（深度） | 1e-5 | 1e-4 |
| 反向梯度 | 1e-3 | 1e-2 |

## Git 提交记录

按照 Conventional Commits 规范：

```bash
# Preprocessor Forward GPU
git commit -m "feat(gpu): implement preprocessor forward OpenCL kernel

- Add preprocess.cl with per-Gaussian parallel processing
- Support both 2D conic and 3D eval_3D paths
- Match CPU behavior for training mode SH color clamping
- Add TDD tests in test_preprocessor_gpu.cpp"

# Preprocessor Backward GPU
git commit -m "feat(gpu): implement preprocessor backward OpenCL kernel

- Add preprocess_backward.cl with 4 gradient chains
- Chain 1: d_conics → d_cov3D → d_scales/d_rotations
- Chain 2: d_rgb → d_sh_coeffs
- Chain 3: d_opacity_2d → d_raw_opacity
- Chain 4: d_means2D → d_positions
- Add TDD tests in test_preprocessor_backward_gpu.cpp"

# Rasterizer Forward GPU
git commit -m "feat(gpu): implement rasterizer forward OpenCL kernel

- Add rasterize.cl with tile-based alpha blending
- Support StopThePop per-pixel depth sorting (k-Buffer K=16)
- Shared memory batch loading for efficiency
- Add TDD tests in test_rasterizer_gpu.cpp"

# Rasterizer Backward GPU
git commit -m "feat(gpu): implement rasterizer backward OpenCL kernel

- Add rasterize_backward.cl with forward replay + backward traversal
- CAS-based atomic float add for gradient accumulation
- Match CPU volumetric rendering gradient formula
- Add TDD tests in test_rasterizer_backward_gpu.cpp"

# Bug fix
git commit -m "fix(gpu): match CPU SH color clamping behavior in training mode

- GPU was incorrectly applying upper clamp (to 1.0) during training
- CPU only applies lower clamp (max(0, x+0.5)) in training mode
- Add training parameter to computeColorFromSH function
- Update kernel arguments in preprocessor_gpu.cpp"
```

## 下一步工作

1. **eval_3D 路径完整测试**: 当前测试主要覆盖 2D 路径，需要添加 3D 评估路径的专用测试
2. **性能基准测试**: 对比 CPU 和 GPU 的性能差异
3. **数值稳定性分析**: 在极端场景下（大量高斯球重叠）验证数值稳定性
4. **多 GPU 支持**: 探索多 GPU 并行渲染的可能性

## 参考资料

- [3DGS 渲染管线分析](docs/3dgs_render_pipeline_analysis.md)
- [GPU-CPU 精度分析](docs/gpu-cpu-precision-analysis.md)
- [梯度验证报告](docs/gradient-verification-report.md)
