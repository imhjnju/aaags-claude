# AAA-Gaussians 移植常见错误手册
**CUDA/PyTorch → HarmonyOS（OpenCL/Vulkan + Maleoon 920）**

> 本文档从 `harmonyos_3dgs/docs/` 的全部调试记录、bugfix 总结和架构分析中提炼而来。
> 凡是标注 **[Critical]** 的，都是直接导致黑屏/白屏/完全错误图像的必死 bug。

---

## 目录

1. [矩阵约定：最高频致命错误类](#1-矩阵约定最高频致命错误类)
2. [相机参数与变换链错误](#2-相机参数与变换链错误)
3. [OpenCL / Vulkan GPU 特有陷阱](#3-opencl--vulkan-gpu-特有陷阱)
4. [AAA-Gaussians 算法特有陷阱](#4-aaa-gaussians-算法特有陷阱)
5. [浮点精度与跨平台一致性](#5-浮点精度与跨平台一致性)
6. [排序算法正确性陷阱](#6-排序算法正确性陷阱)
7. [CUDA → OpenCL 移植的关键挑战（StopThePop 重点攻关）](#7-cuda--opencl-移植的关键挑战stoppthepop-重点攻关)
8. [性能分析的虚假数据陷阱](#8-性能分析的虚假数据陷阱)
9. [数据加载与格式处理](#9-数据加载与格式处理)
10. [训练管线特有问题](#10-训练管线特有问题)
11. [移植检查清单（Quick Reference）](#11-移植检查清单quick-reference)

---

## 1. 矩阵约定：最高频致命错误类

**这是从 CUDA/GLM 移植到 C++/OpenCL 时最容易踩的一类坑，几乎每个矩阵都需要单独核对。**

### 1.1 GLM mat3 构造器按列填充 vs C 数组按行存储 [Critical]

**现象**：渲染出现大量毛刺、边缘模糊、Gaussian 椭圆方向完全反转。

```cpp
// CUDA 原始代码：GLM 按「列」填充，mat3(col0, col1, col2)
glm::mat3 R = glm::mat3(
    1-2(yy+zz), 2(xy-rz), 2(xz+ry),   // ← 这 3 个是 column 0
    2(xy+rz),   1-2(xx+zz), 2(yz-rx), // ← 这 3 个是 column 1
    2(xz-ry),   2(yz+rx), 1-2(xx+yy)  // ← 这 3 个是 column 2
);
// 访问方式: R[col][row]，即 R[0][0]=1-2(yy+zz), R[0][1]=2(xy-rz)

// ❌ 错误移植：把相同 9 个值直接复制到 C 数组
float R[3][3] = {
    {1-2(yy+zz), 2(xy-rz), 2(xz+ry)},  // 被当作 row 0（实为 col 0）→ 转置了！
    ...
};

// ✅ 正确移植：需要将符号项做相应调整（交换 r 符号）
float R[3][3] = {
    {1-2(yy+zz), 2(xy+rz), 2(xz-ry)},  // row 0 = R_math[row=0, col=0..2]
    {2(xy-rz),   1-2(xx+zz), 2(yz+rx)},
    {2(xz+ry),   2(yz-rx), 1-2(xx+yy)}
};
```

**根因**：GLM 是列主序，`mat3(a,b,c, d,e,f, g,h,i)` 填充为 `col0=(a,b,c)`，`col1=(d,e,f)`，`col2=(g,h,i)`。C 的 `float[3][3]` 是行主序，同样的 9 个数直接粘贴，矩阵被转置。

> **警告**：协方差矩阵 Σ 是对称的，`R^T ≠ R` 但 `R^T × S² × R` 和 `R × S² × R^T` 都是对称的——它们的**值不同**，但形状都正确，很容易误判为"看起来没问题"。

---

### 1.2 Jacobian 矩阵 J 的导数项位置错误 [Critical]

**现象**：2D 协方差投影错误，Gaussian 在屏幕上的椭圆严重拉伸或方向异常。

```cpp
// CUDA：GLM 按列填充，focal_x/z 是 col0 的 row0，-(fx*x)/z² 是 col0 的 row2
glm::mat3 J = glm::mat3(
    focal_x/z, 0, -(fx*x)/z²,  // column 0（3个值是同一列的 row0,1,2）
    0, focal_y/z, -(fy*y)/z²,  // column 1
    0, 0, 0                     // column 2
);

// ❌ 错误移植：把每组3个值当作一行
float J[3][3] = {
    {focal_x/z, 0, 0},           // dz 项丢失！
    {0, focal_y/z, 0},           // dz 项丢失！
    {-(fx*x)/z², -(fy*y)/z², 0}  // dz 项错放到第3行
};

// ✅ 正确移植：导数项在 col0 和 col1 的 row2
float J[3][3] = {
    {focal_x/z, 0, -(fx*x)/z²},  // col 0
    {0, focal_y/z, -(fy*y)/z²},  // col 1
    {0, 0, 0}                     // col 2
};
```

---

### 1.3 各系统矩阵约定对照表

| 系统 | 存储约定 | 构造/访问 | 备注 |
|------|---------|---------|------|
| **PyTorch / NumPy** | 行主序 `[row][col]` | 按行填充 | 与 C 数组相同 |
| **GLM / CUDA** | 列主序 `mat[col][row]` | 按列填充 | `mat3(a,b,c,...)` 第一组是 col0 |
| **C `float[3][3]`** | 行主序 `arr[row][col]` | 按行填充 | 与 GLM 相反 |
| **OpenCL kernel** | 取决于实现 | 手动索引 | 统一约定后坚守 |

**黄金法则**：移植前先确定项目内部约定 `M[i][j]` 到底是 `M[row][col]` 还是 `M[col][row]`，然后对每个矩阵逐个验证。

---

## 2. 相机参数与变换链错误

### 2.1 ViewProjection 矩阵乘法顺序 [Critical]

**现象**：场景完全不可见，或位置完全错误。

```cpp
// ❌ 错误
mat4Mul(cam.view_matrix, proj, cam.viewproj_matrix);  // View * Proj

// ✅ 正确（列主序下变换链是 clip = Proj × View × world_pos）
mat4Mul(proj, cam.view_matrix, cam.viewproj_matrix);  // Proj * View
```

---

### 2.2 cameras.json 中 C2W/W2C 旋转矩阵混淆 [Critical]

**现象**：Camera 0 视角向左下偏转约 30°，多个相机方向系统性错误。

**根因**：Python 3DGS 代码中变量名 `W2C` 实际是 `np.linalg.inv(W2C_matrix)` = C2W。cameras.json 存储的是 **C2W 旋转**，不是 W2C。

```cpp
// ❌ 错误：把 C2W 当 W2C 直接使用
view_matrix[row*4+col] = R_from_json[row][col];

// ✅ 正确：C2W 需要转置得到 W2C 旋转部分
view_matrix[col*4+row] = R_c2w[col][row];
// translation: t = -R_c2w^T * cam_center
```

---

### 2.3 像素坐标偏移：像素中心 vs 像素角点

**现象**：eval_3D 模式下图像存在亚像素偏移，边缘 Gaussian 位置不准确。

AAA-Gaussians 的 viewport 矩阵使用 `-0.5` 偏移（将 NDC 映射到像素中心坐标），对应的逐像素评估必须使用像素中心 `(px + 0.5, py + 0.5)`：

```cpp
// ✅ 正确：使用像素中心
float fpx = (float)px + 0.5f;
float fpy = (float)py + 0.5f;
plane_x[k] = g2s[0*4+k] - g2s[3*4+k] * fpx;
plane_y[k] = g2s[1*4+k] - g2s[3*4+k] * fpy;
```

---

### 2.4 AABB Center 与 NDC 投影 Center 的偏差导致 Tile 块状伪影

**现象**：eval_3D 渲染出现明显的 16×16 像素方块状伪影。

**根因**：`computeAABBScreen()` 返回的 `mean2D`（AABB 中心）与标准 3DGS 的 `ndc2Pix()` 投影中心存在系统性偏差，用错误的中心点做 tile 分配导致 Gaussian 被分配到错误的 tile 集合。

```cpp
// ❌ 错误：用 AABB 中心做 tile 分配
float center_x = mean2D_aabb[0];
float center_y = mean2D_aabb[1];

// ✅ 正确：tile 分配用 NDC 投影中心，AABB extent 只用于计算半径
float center_x = ndc2Pix(p_ndc[0], W);  // 标准投影中心
float center_y = ndc2Pix(p_ndc[1], H);
float radius = max(extent_aabb[0], extent_aabb[1]);  // AABB 用于 radius
```

---

## 3. OpenCL / Vulkan GPU 特有陷阱

### 3.1 Kernel 输出变量必须显式写入全局内存 [Critical]

**现象**：GPU 渲染输出全黑，但 preprocessor 报告有效 Gaussian 数量正常。

```opencl
// ❌ 错误：计算完的颜色留在局部变量里，忘记写回
float rgb[3];
computeColorFromSH(/* ... */, rgb);
// rgb 计算完了，但没有 out_rgb[i*3+ch] = rgb[ch]！

// ✅ 正确：每个输出变量必须显式写入 __global buffer
for (int ch = 0; ch < 3; ch++) {
    out_rgb[i * 3 + ch] = rgb[ch];
}
```

> **规则**：OpenCL kernel 中局部变量不会自动同步到全局内存。每个输出字段逐一检查是否有对应的全局写入。

---

### 3.2 多 Kernel Pipeline 中 Buffer Flag 使用错误 [Critical]

**现象**：scatter 和 rasterize kernel 读取 preprocessor 输出时数据全零。

```cpp
// ❌ 错误：preprocessor 输出 buffer 用 WRITE_ONLY，后续 kernel 无法读取
cl_mem out_rgb = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, size, nullptr, &err);

// ✅ 正确：多 kernel pipeline 中的中间 buffer 用 READ_WRITE
cl_mem out_rgb = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, nullptr, &err);
```

> **规则**：`CL_MEM_WRITE_ONLY` / `CL_MEM_READ_ONLY` 是驱动的优化提示，在多 kernel pipeline 中所有中间 buffer 一律使用 `CL_MEM_READ_WRITE`。

---

### 3.3 GPU Prefix Sum 的级数不足导致大数组计算错误 [High]

**现象**：GPU prefix sum 输出 `total_pairs=2,517,653`，正确值是 `6,289,717`（严重低估）；随后 buffer 越界导致 `CL_ERROR -14`（无效访问）崩溃。

**根因**：Blelloch scan 的两级实现对超过 `512×512=262,144` 个元素的数组需要**第三级**，但代码只实现了两级。

```
400K elements / 512 per block = 782 blocks
  Level 1: scan 782 block sums → needs 2 blocks
  Level 2: scan 2 block sums → 1 value
  需要 3 级！代码只有 2 级 → 部分 block sums 没有被加上
```

**推荐做法**：对非瓶颈路径（如 tile 数前缀和，仅数千元素），直接用 CPU 串行 scan（400K 元素只需 ~1ms），优先保证正确性。

---

### 3.4 `-cl-fast-relaxed-math` 隐式影响精度

**现象**：关闭 fast-relaxed-math 后渲染结果有轻微偏差，开启后某些场景像素差达到 38（max_diff）。

该选项等价于同时开启：

| 子选项 | 隐式效果 |
|--------|---------|
| `-cl-mad-enable` | `a*b+c` → `mad()` 单次舍入 |
| `-cl-unsafe-math-optimizations` | 浮点运算可重排，`(a+b)+c → a+(b+c)` |
| `-cl-finite-math-only` | 假设无 NaN/Inf |

隐患：即使代码写的是 `sqrt(x)`，在 fast-relaxed-math 下编译器可能将其替换为 `native_sqrt(x)`（精度 ≤ 8192 ULP）。

> **建议**：调试期间去掉 `-cl-fast-relaxed-math`；性能优化时按需开启，同时用 `native_exp` 替换 `exp`，并接受约 PSNR 86 dB 的精度（人眼完全不可见）。

---

### 3.5 Bitonic Sort 的 Kernel Launch 次数过多导致性能反不如 CPU

**现象**：GPU bitonic sort 1038ms，CPU `std::sort` 750ms，GPU 反而更慢。

**根因**：6.3M 元素的 bitonic sort 需要约 276 次 kernel launch，每次 launch 有 ~0.1ms 的 dispatch overhead，仅 launch 开销就达 27.6ms，加上计算本身性能差。

```
Bitonic sort:  O(N × log²N) 次比较 → 276 次 kernel launch
GPU Radix sort: O(N × 8) passes → 24 次 kernel launch（8-bit）/ 16 次（4-bit）
```

> **结论**：大数据量排序优先使用 Radix Sort，不是 Bitonic Sort。Bitonic 适合 ≤16K 的小数组 in-tile 排序。

---

## 4. AAA-Gaussians 算法特有陷阱

### 4.1 逐 Tile 3D 裁剪与 StopThePop 的强依赖关系 [算法级]

**现象**：启用逐 Tile 3D 裁剪后渲染性能大幅提升，但图像出现整块 tile 变黑或颜色错误。

**根因**：低透明度 Gaussian 单独贡献低于裁剪阈值，但数百个这样的 Gaussian 累积后产生可见颜色。逐 tile 裁剪将它们全部去除。

更深的根因：**per-tile culling 必然产生 tile 边界不连续**——同一 Gaussian 在 tile A 被裁剪、在 tile B 保留，边界处渲染结果不连续。

> **结论**：逐 Tile 3D 裁剪**必须配合 StopThePop 逐像素重排序**才能正确工作。在 StopThePop 尚未移植完成前，应暂时禁用逐 Tile 裁剪，退回全局排序。StopThePop 移植是后续重点攻关方向（见第 7 节）。

---

### 4.2 Per-Tile Depth Key 独立使用会产生 Tile 边界伪影

**现象**：启用 per-tile depth key (depthAlongRay) 后，相邻 tile 边界处出现排序不连续的块状伪影。

**根因**：相邻两个 tile 对同一对 Gaussian 使用不同的深度键，可能产生相反的排序顺序，导致 alpha blending 在边界处突变。

> **结论**：per-tile depth key 同样**需要 StopThePop 配合**。单独启用只会引入伪影。StopThePop 移植完成后可同步启用。

---

### 4.3 kBuffer per-pixel depth 计算的数值发散

**现象**：使用 `max_pos` 的 z 分量作为 kBuffer 排序键时，渲染出现严重块状伪影。

**根因**：计算 per-pixel depth 需要用 `(dx*my - dy*mx) / dd`，当 `dd`（两个平面法向量叉积模长平方）接近零时（Gaussian 中心接近像素射线），除法结果发散，导致相邻像素排序结果差异巨大。

```cpp
// 有问题的代码
float depth = (plane_x[0]*plane_y[1] - plane_x[1]*plane_y[0]) / dd;
// dd = dot(cross(nx,ny), cross(nx,ny))，当 nx || ny 时 → 0

// 安全做法：加 epsilon 保护，或退回 view-space z
float depth = (dd > 1e-8f) ? /* 计算 */ : view_space_z;
```

---

### 4.4 Scale Dilation 的 Amplitude 补偿不可省略

**现象**：远距离 Gaussian 过度透明，场景远处细节丢失。

**算法**：膨胀尺度后若不补偿振幅，Gaussian 会过度透明。必须计算 `dilation_factor` 并乘以 opacity：

```cpp
// 不可省略的振幅补偿
float det_mul_ray_var     = dot(r, {sy*sz, sz*sx, sx*sy});
float det_mul_ray_var_dil = dot(r, {syd*szd, szd*sxd, sxd*syd});
float dilation_factor = sqrt(det_mul_ray_var / det_mul_ray_var_dil);
opacity_out = opacity_in * dilation_factor;  // 必须
```

---

### 4.5 `filter_3D` 属性的含义与存储格式

**常见误解**：`filter_3D` 是一个过滤强度，需要经过某种激活函数。

**实际**：`filter_3D` 存储的是原始值（`z_min / focal_max × √0.3`），直接从 PLY 读取，**无需任何激活**，作为训练时最小滤波尺度的下限使用：

```python
# 训练时计算（mip_filter.py）
filter_3D = distance / focal_length * (0.3 ** 0.5)
# 渲染时使用
scale_mip = max(filter_3D², (z_view / focal)² * kernel_size)
scale_dilated = sqrt(scale² + scale_mip)
```

---

### 4.6 Gauss2Screen 矩阵的行列主序与乘法顺序

**容易出错点**：`gauss2screen` 矩阵的构建需要严格按以下顺序，且最终以行主序存储：

```
L = transpose(S_dilated × R)           // Gaussian 局部坐标轴
gauss2world = [L | mean3D; 0 0 0 1]   // 列主序 4×4
viewport = [W/2  0  0  W/2-0.5; ...]  // 注意 -0.5 偏移
gauss2screen = transpose(viewport × viewproj × gauss2world)  // 行主序存储
```

任何一步的行列主序搞错都会导致逐像素 Mahalanobis 距离计算错误，表现为 Gaussian 形状异常或位置偏移。

---

## 5. 浮点精度与跨平台一致性

### 5.1 ARM FMA 指令导致 x86 vs ARM64 精度差异 [Critical]

**现象**：PC x86 和 ARM64 手机渲染差异巨大（PSNR 20-28 dB，最大像素差 248），虽然算法完全相同。

**根因**：ARM NEON 的 FMA（Fused Multiply-Add）将 `a*b+c` 合并为一次操作只舍入一次，而 x86 SSE 分别舍入两次。微小差异（~1e-7）通过以下链条放大：

```
cov3D → cov2D (矩阵乘) → conic (矩阵求逆) → alpha (exp) → 颜色累加
                                                           ↑
                              排序键由 depth 决定 → 排序翻转 → 完全不同的混合顺序
```

当两个 Gaussian 深度几乎相同时，舍入差异导致排序翻转，alpha blending 前后顺序改变，像素颜色完全不同。

**修复**：

```cmake
# CMakeLists.txt
target_compile_options(renderer PRIVATE -ffp-contract=off)
```

**效果**：PSNR 从 20-28 dB → **89-90 dB**（max diff 4-7，float32 精度极限）。

---

### 5.2 `native_exp()` vs `exp()` 的误差累积

**场景**：密集区域（2000+ Gaussian/tile）中，`native_exp` 的 ≤8192 ULP 误差通过 alpha blending 串行链累积：

```
T₀ = 1.0
T₁ = T₀ × (1 - α₁)     ← α₁ 有 native_exp 误差
...
T₂₀₀₀                   ← 2000 次累积后 T 可能偏移 ~0.01
C = Σ(rgb_i × α_i × T_i) ← 像素值差可达 30-40
```

| 函数 | 精度 | 典型误差 | 适用场景 |
|------|------|---------|---------|
| `std::exp()` (CPU) | ≤ 1 ULP | 精确 | 调试/参考 |
| `exp()` (OpenCL) | ≤ 3 ULP | 精确 | 精度优先 |
| `native_exp()` (OpenCL) | ≤ 8192 ULP | 最大 ~1e-4 | 性能优先（可接受） |

---

### 5.3 SH 颜色超亮导致白色斑块

**现象**：模型表面出现白色斑块（如篮球正面），Python 参考也有相同问题。

**根因**：训练时 clamp(0,1) 在 loss 之前，梯度为零，部分 Gaussian 的 DC 颜色训练到 >1.0（实测 max 4.05）。推理端多个超亮 Gaussian 累加后溢出。

**推理端必须加 clamp**：

```cpp
float r = clamp(sh_color[0], 0.0f, 1.0f);  // 必须在推理端加
float g = clamp(sh_color[1], 0.0f, 1.0f);
float b = clamp(sh_color[2], 0.0f, 1.0f);
```

**效果**：白斑像素从 3733 减少到 199（减少 95%）。

---

## 6. 排序算法正确性陷阱

### 6.1 Radix Sort 使用 `atomic_add` 破坏稳定性 [High]

**现象**：使用 atomic_add 优化 scatter 阶段后，图像出现完全乱序（PSNR 5 dB）。

**根因**：Radix sort 要求 pass 间稳定性。`atomic_add` 的执行顺序不确定，同一 digit 的元素相对顺序被打乱，前一 pass 的排序结果丢失。

```opencl
// ❌ 错误：atomic_add 破坏稳定性
int pos = atomic_add(&bucket_counters[digit], 1);
output[pos] = input[i];  // 同一 bucket 内顺序不确定

// ✅ 正确：使用预先计算好的确定性偏移量
int pos = offsets[wg_id * 16 + digit] + local_rank;
output[pos] = input[i];  // 稳定
```

---

### 6.2 排序错误导致光栅化"假高性能"

**现象**：修复排序后，光栅化时间从 127ms 增加到 492ms，看似性能倒退。

**解释**：这是**预期行为**，不是性能问题：

```
旧版（排序错误）:
  后方 Gaussian 先混合 → 透过率 T 过早饱和到 0.0001
  → if (T < 0.0001) break 提前退出 → 127ms（但图像有 84% 像素错误）

新版（排序正确）:
  前景先混合 → 每个像素需处理更多 Gaussian 才能饱和 → 492ms（图像正确）
```

> **规则**：光栅化时间与**排序正确性**强相关。排序越正确，光栅化越慢（但图像正确）。不要把"光栅化变慢"当作性能 bug 来 debug。

---

### 6.3 Bitonic Sort 跳过合并（`skip_merge=true`）导致大 tile 排序不完整

**现象**：tile 内 >4096 元素的 Gaussian 排序不完整，对应区域 84% 像素与 CPU 参考不一致。

**根因**：bitonic sort 的合并阶段被优化掉（`skip_merge=true`），对 >4096 元素的 tile 只做了部分排序。

> **规则**：任何排序优化都必须先验证正确性（CPU 参考对比 PSNR > 90 dB），再考虑性能。

---

## 7. CUDA → OpenCL 移植的关键挑战（StopThePop 重点攻关）

### 7.1 Maleoon 920 OpenCL 扩展现状

```
✅ cl_khr_subgroups              — 基础 subgroup 支持
⚠️ sub_group_shuffle_up          — 驱动在 -cl-std=CL3.0 下实际支持（RadixSort 已验证可用）
❌ cl_khr_subgroup_shuffle       — 未声明，待验证实际可用性
❌ cl_khr_subgroup_ballot        — 未声明，待验证实际可用性
❌ cl_khr_subgroup_non_uniform_vote — 未声明，待验证实际可用性
```

> **重要发现**：Maleoon 920 的驱动虽未声明 `cl_khr_subgroup_shuffle_relative`，但实际在 `-cl-std=CL3.0` 模式下支持 `sub_group_shuffle_up`（RadixSort 中已成功使用）。**其他 subgroup 操作（shuffle、ballot 等）也需要逐一实测验证实际可用性**，驱动声明的扩展列表可能不完整。

---

### 7.2 StopThePop 分层排序移植——重点攻关方向 🔥

StopThePop 是解锁逐 Tile 裁剪、Per-Tile Depth Key、kBuffer 等多项高级特性的前提，**是后续移植的最高优先级任务**。

**CUDA 原语 → OpenCL 替代方案分析：**

| CUDA 原语 | StopThePop 用途 | 替代方案 | 攻关难度 | 备注 |
|-----------|----------------|---------|---------|------|
| `__shfl_sync` | 排序核心：线程间直接读寄存器 | `sub_group_shuffle`（需实测）/ local mem + barrier | 高 | RadixSort 已验证 shuffle_up 可用，shuffle 很可能也支持 |
| `__ballot_sync` | 集体投票：收集线程判断 | `sub_group_ballot`（需实测）/ local mem + barrier | 高 | 核心依赖，需优先验证 |
| `__fns` | 查找第 N 个置位 bit | `popcount` + mask 手动位操作 | 中 | ~10-20 条指令替代，可行 |
| CUB `BlockRadixSort` | block 级基数排序 | 手写 local mem radix sort / bitonic sort | 中 | 已有 bitonic sort 实现可复用 |
| `tiled_partition<N>` | 将 warp 拆分为子组 | 手动计算 sub-group lane 索引 | 中 | 无子组级 barrier，需设计替代同步策略 |

**攻关路线建议：**

1. **第一步：实测验证** — 编写 micro-benchmark，在 Maleoon 920 上逐一测试 `sub_group_shuffle`、`sub_group_ballot` 等操作的实际可用性（驱动可能已实现但未声明）
2. **第二步：原型验证** — 如果 shuffle/ballot 实测可用，在小规模 tile（≤256 Gaussians）上实现 StopThePop 原型，验证正确性
3. **第三步：local mem 兜底方案** — 对不可用的原语，设计 local memory + barrier 替代方案，评估实际性能退化（可能远小于理论 100x，因为 Maleoon 的 barrier 实际开销需实测）
4. **第四步：混合策略** — 可用的原语直接使用，不可用的用 local mem 模拟，找到性能平衡点

**StopThePop 移植完成后将解锁：**
- ✅ 逐 Tile 3D 裁剪（预期减少 ~40% tile-pairs，大幅提升光栅化性能）
- ✅ Per-Tile Depth Key（消除 popping 伪影，提升视觉质量）
- ✅ kBuffer 逐像素深度排序（进一步消除排序伪影）

---

### 7.3 3D Thread Block 到 1D Work-Group 的映射

CUDA 的 3D thread block 组织（如 StopThePop 的 16×4×4）在 OpenCL 中需要手动展开（移植时注意索引映射）：

```opencl
// OpenCL 等价：扁平化为 1D work-group + 手动索引
int lid = get_local_id(0);        // 0..255
int lid_x = lid % 16;             // x 维度
int grp_y = (lid / 16) % 4;      // y 维度
int grp_z = lid / 64;             // z 维度
```

---

## 8. 性能分析的虚假数据陷阱

### 8.1 Debug Readback 导致排序性能虚高

**现象**：GPU radix sort 报告 2143ms，远慢于 CPU sort (873ms)；移除 debug 代码后降至 179ms。

**根因**：排序后的 debug 代码读回全部 6.3M keys (50MB) 做正确性验证。这个 readback 本身消耗 ~1400ms，被计入了排序时间。

```cpp
// ❌ 导致性能虚高的 debug 代码（生产版必须移除）
std::vector<uint64_t> keys_host(num_pairs);
clEnqueueReadBuffer(queue, keys_buf, CL_TRUE, 0,
    num_pairs * sizeof(uint64_t), keys_host.data(), 0, nullptr, nullptr);
verify_sort_order(keys_host);  // ~1400ms
```

---

### 8.2 GPU↔CPU 数据传输时间的掩藏效应

各阶段的"kernel 时间"和"总时间"可能差异巨大：

| 阶段 | Kernel 时间 | 实际总时间 | 非 Kernel 开销 |
|------|------------|-----------|---------------|
| Preprocess | 7.7 ms | 35.6 ms | readback 3.2MB → 27.9ms |
| TileBinner | 16.3 ms | 42.5 ms | CPU prefix sum + upload → 26.2ms |
| Sort | ~570 ms | 572.6 ms | 几乎全是 GPU |
| Rasterizer | 297.9 ms | 310.2 ms | readback 8.3MB → 12.3ms |

> **规则**：必须同时测量 kernel 时间和端到端时间，才能定位真正的瓶颈。

---

## 9. 数据加载与格式处理

### 9.1 PLY Loader 硬编码属性索引 [High]

**现象**：加载 59 属性的 PLY 文件时崩溃；加载不含法线的文件时数据错位。

**根因**：代码硬编码 `PROPS_PER_VERTEX = 62`，假设固定属性布局。

```cpp
// ❌ 错误：硬编码属性索引
float x = vertex_data[vertex_idx * 62 + 0];  // 假设 x 永远在第 0 列

// ✅ 正确：header-driven 解析
// 1. 解析 PLY header 中的 property 声明
// 2. 建立属性名 → 索引映射
// 3. 动态查找 "x", "y", "z", "f_dc_0", "filter_3D" 等属性
std::unordered_map<std::string, int> prop_index;
// ... 解析 header 填充映射 ...
float x = vertex_data[vertex_idx * num_props + prop_index["x"]];
```

---

### 9.2 `filter_3D` 属性缺失时的自动退化

AAA-Gaussians 的 PLY 文件包含 `filter_3D` 属性，标准 3DGS PLY 没有。代码必须检测并优雅退化：

```cpp
// 自动检测并决定是否启用 eval_3D
bool has_filter_3d = (prop_index.find("filter_3D") != prop_index.end());
cfg.eval_3D = has_filter_3d;  // 也可通过环境变量 EVAL_3D=0/1 覆盖
```

---

## 10. 训练管线特有问题

### 10.1 Forward/Backward training mode 不匹配 [Critical]

**现象**：训练时梯度为零或训练无法收敛（SH 颜色 > 1.0 的 Gaussian 无梯度）。

**根因**：Forward 默认 `training=false`（颜色 clamp 到 [0,1]），backward 用 `training=true`（无 clamp）。SH 颜色 > 1.0 时 forward clamp 导致变化不传播，梯度完全错误。

```cpp
// ✅ 训练时必须设置
RenderConfig cfg;
cfg.training = true;  // 禁用 clamp，使梯度正确传播
```

---

### 10.2 SH→Position 梯度链缺失 [Critical]

**现象**：SH degree ≥ 1 时 position 梯度全零，Gaussian 位置不收敛。

**根因**：颜色通过归一化视角向量依赖 position，这条链 `d_color/d_dir × d_dir/d_pos` 必须在 backward 中实现。SH degree 0 纯 DC 颜色不依赖方向，此 bug 不触发，容易漏测。

```cpp
// backward 必须实现
void computeColorFromSH_backward(
    ...,
    float* d_sh_coeffs,  // 已有
    float* d_pos         // 必须新增！SH degree >= 1 时 position 有梯度
);
```

---

### 10.3 学习率未随 Gaussian 密度缩放导致发散

**现象**：默认学习率在 ~2000 步后 loss 开始上升，最终发散（0.15 → 0.40）。Position z 飘移到相机后方，scale 膨胀到 21.8。

**根因**：Gaussian 数量少时每个 Gaussian 覆盖的像素更多，梯度幅值更大，但 LR 未缩放。

```python
# 官方 3DGS 的正确做法
spatial_lr_scale = 1.0 / cameras_extent  # 根据场景大小缩放
position_lr = initial_lr * spatial_lr_scale
```

**临时修复**：`--lr_scale 0.3` 可避免发散，但正确方案是实现 `spatial_lr_scale`。

---

### 10.4 Test Data 路径硬编码导致跨目录运行失败

```cpp
// ❌ 错误：相对路径在不同工作目录下失效
const char* ply_path = "../tests/test_data/cactus.ply";

// ✅ 正确：通过 CMake 编译定义注入绝对路径
// CMakeLists.txt:
// target_compile_definitions(tests PRIVATE TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/test_data")
const char* ply_path = TEST_DATA_DIR "/cactus.ply";
```

---

## 11. 移植检查清单（Quick Reference）

### 矩阵约定
- [ ] **GLM mat3 构造器**：9 个参数按**列**填充，不是按行
- [ ] **C `float[3][3]`**：索引约定是 `[row][col]`，与 GLM `[col][row]` **相反**
- [ ] **旋转矩阵 R**：从四元数构建时交换含 `r`（标量）的符号项
- [ ] **Jacobian J**：导数项 `-f×x/z²` 在 col0 的 row2，不在 col2 的 row0
- [ ] **ViewProj 乘法**：`ViewProj = Proj × View`（不是 View × Proj）
- [ ] **来自 Python 的矩阵**：传入 C++ 前确认是否需要转置

### 相机参数
- [ ] **cameras.json rotation**：存储的是 C2W 旋转，需要**转置**得到 W2C
- [ ] **像素坐标**：使用像素中心 `px + 0.5`，与 viewport 矩阵 `-0.5` 偏移一致
- [ ] **Tile 分配中心**：使用 NDC 投影中心，而非 AABB 中心

### OpenCL / Vulkan GPU
- [ ] **Kernel 输出**：每个输出变量必须显式写入 `__global` buffer
- [ ] **Buffer flags**：多 kernel pipeline 中的中间 buffer 用 `CL_MEM_READ_WRITE`
- [ ] **Prefix sum**：确认级数足够（元素数 > `WG_SIZE²` 需要第三级）
- [ ] **精度选项**：`-ffp-contract=off` 禁止 FMA 合并（跨平台一致性）
- [ ] **排序稳定性**：radix sort 不能用 non-deterministic atomic 替代确定性偏移

### AAA-Gaussians 特有
- [ ] **`filter_3D`**：从 PLY 直接读取，无需激活函数
- [ ] **dilation_factor**：scale 膨胀后必须补偿 opacity（不可省略）
- [ ] **逐 Tile 裁剪**：StopThePop 移植完成前暂时禁用
- [ ] **Per-Tile Depth Key**：StopThePop 移植完成前暂时禁用
- [ ] **StopThePop**：重点攻关——先实测 Maleoon 920 的 subgroup 操作实际可用性，再逐步实现

### 数据加载
- [ ] **PLY 解析**：header-driven 动态属性映射，不要硬编码属性索引
- [ ] **SH 颜色推理端**：推理时 clamp 到 [0,1] 防止累加溢出

### 训练管线
- [ ] **training mode**：forward 和 backward 必须使用相同的 `cfg.training = true`
- [ ] **SH→Position 梯度**：SH degree ≥ 1 时 backward 必须实现 position 梯度链
- [ ] **学习率缩放**：需要实现 `spatial_lr_scale = 1/cameras_extent`

### 性能分析
- [ ] **移除 debug readback**：性能测试前移除所有大数据 GPU→CPU 读回
- [ ] **同时测量 kernel + 端到端时间**：传输开销可能掩盖真正瓶颈
- [ ] **排序正确后光栅化变慢是正常现象**：不要把正确排序当性能 bug

---

## 附录：已验证的最优配置（Maleoon 920）

```
场景: basketball.ply, 400K Gaussians, 720×960
配置: eval_3D=ON, 4-bit RadixSort, 全局 view-space depth, 直接 alpha blending
      (禁用: per-tile culling, per-tile depth key, kBuffer)

阶段耗时:
  Preprocess:   35 ms
  TileBinner:   29 ms
  Sort:        178 ms  ← 4-bit radix sort (原 bitonic: 4755ms)
  Rasterize:   492 ms
  Total:       777 ms  (6.5x vs 修复前)

正确性: CPU vs GPU mean diff = 0.0001（float32 精度极限）
```

**性能目标路线图**（到 60 FPS = 16.7ms）：

| 优先级 | 方案 | 预期收益 |
|--------|------|---------|
| **P0** 🔥 | **StopThePop 移植攻关**（实测 subgroup → 原型 → 兜底方案） | 解锁下方三项 |
| P0 | 逐 Tile 裁剪（StopThePop 完成后启用） | -40% tile-pairs |
| P0 | Per-Tile Depth Key（StopThePop 完成后启用） | 消除 popping |
| ✅ 已完成 | GPU Radix Sort | 排序 26.7x |
| P1 | Rasterizer kernel 优化：batch size↑, native_exp | -50% |
| P1 | GPU prefix sum 修复三级 scan | 消除 CPU 往返 |
| P2 | CL-GL interop 直接显示 | -12ms |
