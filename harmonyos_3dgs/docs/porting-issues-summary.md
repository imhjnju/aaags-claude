# 3DGS 鸿蒙移植全过程问题总结

## 概述

将 3D Gaussian Splatting 从 CUDA/Python 移植到鸿蒙手机（Maleoon GPU 920 + OpenCL）的完整过程中，共遇到 **12 个问题**，涵盖矩阵约定、浮点精度、GPU kernel、模型质量、性能优化等多个维度。

---

## Phase 1：CPU 渲染器移植（CUDA → C++）

### Bug #1：旋转矩阵 R 行列主序转置 [Critical]

**文件**: `src/math_utils.cpp` — `computeCov3D()`

**现象**: 渲染结果出现大量毛刺和边缘模糊

**根因**: CUDA/GLM 的 `glm::mat3(a,b,c,d,e,f,g,h,i)` 按**列**填充，但 C 二维数组 `float[3][3]` 按**行**存储。直接复制 9 个值导致 R 矩阵被转置。

**数学影响**: 使用 R^T 代替 R 得到 `R * S² * R^T` 而非正确的 `R^T * S² * R`。虽然两者都对称，但**值不同**——椭圆旋转方向反转。

**修复**: 交换 R 矩阵中所有含四元数标量 `r` 的符号项。

**教训**: **GLM 按列填充 vs C 数组按行存储**是最常见的移植陷阱。

---

### Bug #2：Jacobian 矩阵 J 列布局错误 [Critical]

**文件**: `src/math_utils.cpp` — `computeCov2D()`

**现象**: 2D 协方差投影不正确，Gaussian 椭圆形状异常

**根因**: GLM 构造器 `mat3(fx/z, 0, -fx*x/z², ...)` 中，每 3 个值是一**列**，不是一行。z 导数项 `-fx*x/z²` 应在 col0 的 row2，不是 col2 的 row0。

**修复**: 将导数项从 `J[2][0]` 移到 `J[0][2]`。

---

### Bug #3：View-Projection 矩阵乘法顺序 [Critical]

**文件**: `src/main.cpp`

**现象**: 场景完全不可见

**根因**: 列主序下正确的变换链是 `clip = Proj * View * world_pos`，应计算 `ViewProj = Proj * View`，但代码写成了 `View * Proj`。

---

### Bug #4：cameras.json C2W/W2C 旋转矩阵混淆 [Critical]

**文件**: `src/main.cpp` — `loadCameraJson()`

**现象**: Camera 0 视角向左下偏转

**根因**: Python 代码中变量名 `W2C` 实际是 `np.linalg.inv(W2C_matrix)` = C2W。cameras.json 存储的是 **C2W 旋转**，不是 W2C。加载后需要转置才能得到 W2C 的 view matrix。

**修复**: `view_matrix[col*4+row] = R_c2w[col][row]`（而非 `R_c2w[row][col]`），translation 用 `-R_c2w^T * cam_center`。

---

### Bug #5：PLY Loader 硬编码属性索引 [High]

**文件**: `src/ply_loader.cpp`

**现象**: 加载无法线的 PLY 文件（如 cactus.ply 59 属性）崩溃

**修复**: 改为 header-driven 解析，动态建立属性名→索引映射。

---

### Bug #6：SH 颜色超亮导致白色斑块 [Medium]

**文件**: `src/sh_eval.cpp`

**现象**: basket0 模型篮球表面出现白色斑块

**根因**: 18.3% 的 Gaussian 的 DC 颜色 > 1.0（max 4.05）。训练时 `clamp(0,1)` 在 loss 前，梯度为零导致模型不抑制极端值。138+ 个这样的 Gaussian 在同一像素累加后溢出。

**验证**: Python 3DGS `eval_sh` 产出完全相同的超亮值（max 4.803），确认是训练过程固有问题。

**修复**: 推理端 per-Gaussian 颜色 clamp 到 [0,1]。白斑减少 95%（3733 → 199 pixels）。

---

## Phase 2：GPU (OpenCL) 移植

### Bug #7：GPU kernel 遗漏 out_rgb 写入 [Critical]

**文件**: `src/gpu/kernels/preprocess.cl`

**现象**: GPU 渲染输出全黑

**根因**: preprocess kernel 的 Step 11 (Store outputs) 中，SH 颜色计算结果存在局部变量 `float rgb[3]` 中，但**忘记写入** `__global float* out_rgb` buffer。

**修复**: 添加 `out_rgb[i*3+ch] = rgb[ch]`。

**教训**: OpenCL kernel 中局部变量不会自动同步到全局内存，每个输出必须显式写入。

---

### Bug #8：CL_MEM_WRITE_ONLY 跨 kernel 读取 [Critical]

**文件**: `src/gpu/preprocessor_gpu.cpp`

**现象**: scatter 和 rasterize kernel 读取 preprocessor 输出时数据为零

**根因**: Preprocessor 输出 buffer 创建时用 `CL_MEM_WRITE_ONLY`，后续 kernel 读取属于未定义行为。

**修复**: 改为 `CL_MEM_READ_WRITE`。

**教训**: OpenCL buffer flag 是驱动优化提示，多 kernel pipeline 中应始终用 `CL_MEM_READ_WRITE`。

---

### Bug #9：GPU Radix Sort 稳定性破坏 [High]

**文件**: `src/gpu/kernels/radix_sort.cl`

**现象**: 使用 `atomic_add` 优化的 scatter 产出完全乱序的图像（PSNR 5 dB）

**根因**: Radix sort 要求 pass 间稳定性。`atomic_add` 的执行顺序不确定，同一 digit 的元素相对顺序被打乱，导致前一 pass 的排序结果丢失。

**修复**: 使用 thread-0 串行 rank 计算（O(N) per chunk），保证稳定性。

**教训**: GPU 排序优化必须保证稳定性，不能用非确定性的 atomic 操作替代。

---

### Bug #10：GPU Prefix Sum 大数组错误 [High]

**文件**: `src/gpu/tile_binner_gpu.cpp`

**现象**: GPU prefix sum 算出 `total_pairs=2517653`（正确值 6289717），CL error -14 崩溃

**根因**: Blelloch scan 的两级实现对超过 `512*512=262144` 元素的数组需要第三级，但代码只实现了两级。

**修复**: 回退到 CPU prefix sum（400K 元素只需 ~1ms，不值得复杂化 GPU 实现）。

---

## Phase 3：跨平台一致性

### Bug #11：x86 vs ARM64 浮点精度差异 [Critical]

**文件**: `CMakeLists.txt`

**现象**: PC x86 和 ARM64 手机渲染差异巨大（PSNR 20-28 dB，max diff 171-248 pixels）

**根因**: ARM NEON 的 **FMA (Fused Multiply-Add)** 指令将 `a*b+c` 合并为一次操作只舍入一次，而 x86 SSE 的 mul 和 add 分别舍入。这个微小差异（~1e-7）通过以下链条放大：
```
cov3D → cov2D (矩阵乘) → conic (矩阵求逆) → alpha (exp) → 颜色累加 → 排序顺序
```
当两个 Gaussian 深度几乎相同时，舍入差异导致排序翻转，alpha blending 前后顺序改变，像素颜色完全不同。

**修复**: `-ffp-contract=off` 编译选项禁止 FMA 合并，确保两平台舍入行为一致。

**效果**: PSNR 从 20-28 dB 提升到 **89-90 dB**（max diff 4-7，float32 精度极限）。

**教训**: 跨平台浮点一致性需要显式控制编译器的浮点优化选项。

---

### Bug #12：Debug Readback 导致排序性能虚高 [Medium]

**文件**: `src/gpu/sorter_gpu.cpp`

**现象**: GPU radix sort 报告 2143ms，远慢于 CPU sort (873ms)

**根因**: 排序后有一段 debug 代码读回全部 6.3M keys (50MB) 做正确性验证。这个 readback 本身消耗 ~1400ms。

**修复**: 移除 debug readback。实际 GPU sort 时间降至 779ms。

**教训**: 性能分析时必须区分实际计算和 debug 开销。

---

## 移植检查清单

移植 CUDA/GLM 到纯 C++/OpenCL 时：

- [ ] **GLM mat3 构造器**: 9 个参数按列填充，不是按行
- [ ] **C float[3][3]**: 索引约定是 `[row][col]`，与 GLM `[col][row]` 相反
- [ ] **矩阵乘法顺序**: 列主序下 `ViewProj = Proj * View`
- [ ] **cameras.json**: rotation 是 C2W 旋转，需要转置得到 W2C
- [ ] **OpenCL buffer flags**: 多 kernel pipeline 用 `CL_MEM_READ_WRITE`
- [ ] **OpenCL kernel 输出**: 每个输出变量必须显式写入 `__global` buffer
- [ ] **GPU 排序稳定性**: radix sort 不能用 non-deterministic atomic
- [ ] **跨平台浮点**: 加 `-ffp-contract=off` 禁止 FMA 合并
- [ ] **SH 颜色范围**: 推理端建议 clamp 到 [0,1] 防止累加溢出
- [ ] **PLY 加载**: header-driven 解析，不要硬编码属性索引

---

## 性能演进

| 阶段 | 总耗时 | 加速比 |
|------|--------|--------|
| CPU (PC x86) | 28,488 ms | 1x |
| CPU (ARM64 手机) | 12,953 ms | 2.2x |
| GPU v1 (初始移植) | 1,629 ms | 17.5x |
| GPU v2 (减 readback) | 1,515 ms | 18.8x |
| GPU v3 (去 debug) | 1,168 ms | 24.4x |
| GPU v4 (当前最优) | ~870 ms | **32.7x** |
| 目标 (60 FPS) | 16.7 ms | ~1700x |

**当前瓶颈**: Sort 827ms（CPU sort + 150MB 数据传输）+ Rasterizer 311ms（GPU kernel）
