# HarmonyOS 3DGS CPU 渲染器 — 调试报告

## 概述

在将 3DGS CUDA 渲染器移植到纯 C++ CPU 实现的过程中，遇到了 7 个 bug。其中最关键的 2 个与 **行主序/列主序矩阵约定混淆** 直接相关，这也是从 CUDA/GLM 移植到纯 C++ 时最容易犯的错误。

---

## Bug #1: 旋转矩阵 R 的行列主序转置错误 [Critical]

**文件:** `src/math_utils.cpp` — `computeCov3D()`

### 现象

渲染结果出现大量"毛刺"和边缘模糊，Gaussian 椭圆方向明显错误。

### 根因分析

CUDA 代码使用 GLM 构造旋转矩阵：

```cpp
// CUDA — GLM 按列填充: mat3(col0..., col1..., col2...)
glm::mat3 R = glm::mat3(
    1-2(yy+zz), 2(xy-rz), 2(xz+ry),   // ← 这是 column 0
    2(xy+rz), 1-2(xx+zz), 2(yz-rx),     // ← 这是 column 1
    2(xz-ry), 2(yz+rx), 1-2(xx+yy)      // ← 这是 column 2
);
// GLM 访问: R[col][row], 即 R[0][0]=1-2(yy+zz), R[0][1]=2(xy-rz)
```

移植到 C 数组时，直接复制了相同的 9 个值：

```cpp
// 错误 — C 数组是 [row][col]，直接复制变成了转置！
float R[3][3] = {
    {1-2(yy+zz), 2(xy-rz), 2(xz+ry)},   // ← 被当作 row 0
    {2(xy+rz), 1-2(xx+zz), 2(yz-rx)},
    {2(xz-ry), 2(yz+rx), 1-2(xx+yy)}
};
```

**问题本质：** GLM 的 9 个参数按列填充，但 C 二维数组 `float[3][3]` 按行存储。直接复制值导致 R 矩阵被转置。

由于 `Sigma = M^T * M`，使用 R^T 代替 R 会得到 `R * S^2 * R^T` 而不是 `R^T * S^2 * R`。虽然两者都是对称矩阵，但 **它们的值不同** — 椭圆的旋转方向被反转。

### 修复

将 C 数组的值排列为正确的 `R_math(row, col)` 布局，即交换 R 矩阵中所有关于 `r`（四元数标量部分）的符号：

```cpp
// 修复后 — R[row][col] = R_math(row, col)
float R[3][3] = {
    {1-2(yy+zz), 2(xy+rz), 2(xz-ry)},   // 注意符号变化
    {2(xy-rz), 1-2(xx+zz), 2(yz+rx)},
    {2(xz+ry), 2(yz-rx), 1-2(xx+yy)}
};
```

### 影响

修复前后对比 — 边缘毛刺完全消除，椭圆方向正确。

---

## Bug #2: Jacobian 矩阵 J 的列布局错误 [Critical]

**文件:** `src/math_utils.cpp` — `computeCov2D()`

### 现象

2D 协方差投影不正确，Gaussian 在屏幕上的椭圆形状异常（过度拉伸或方向错误）。

### 根因分析

CUDA 代码：

```cpp
// CUDA — GLM 按列填充
glm::mat3 J = glm::mat3(
    focal_x/z, 0, -(fx*x)/z²,   // column 0: 包含 dz 导数项
    0, focal_y/z, -(fy*y)/z²,    // column 1: 包含 dz 导数项
    0, 0, 0                       // column 2: 全零
);
```

GLM 填充后 J 的实际布局（`J[col][row]`）：
- `J[0] = (focal_x/z, 0, -(fx*x)/z²)` — 导数项在 col0 的 row2
- `J[1] = (0, focal_y/z, -(fy*y)/z²)` — 导数项在 col1 的 row2
- `J[2] = (0, 0, 0)`

移植时的错误：

```cpp
// 错误 — 导数项被放到了第 3 列 (J[2])，而非第 1、2 列的第 3 行
float J[3][3] = {
    {focal_x/z, 0, 0},                    // col 0: 缺少 dz 项！
    {0, focal_y/z, 0},                     // col 1: 缺少 dz 项！
    {-(fx*x)/z², -(fy*y)/z², 0}           // col 2: dz 项错放在这里
};
```

**误解来源：** 把 CUDA 中的 3 行参数理解为"第一行是 col0 的 row0,row1"，漏掉了第 3 个值是同列的 row2 而非下一列的 row0。

### 修复

```cpp
// 修复后 — J[col][row]
float J[3][3] = {
    {focal_x/z, 0, -(fx*x)/z²},           // col 0: 导数项在 row 2 ✓
    {0, focal_y/z, -(fy*y)/z²},            // col 1: 导数项在 row 2 ✓
    {0, 0, 0}                               // col 2: 全零 ✓
};
```

### 影响

这个 bug 导致投影 Jacobian 完全错误，T = W * J 的结果不正确，最终 2D 协方差 `cov = T^T * Vrk * T` 产生的椭圆形状偏差极大。

---

## Bug #3 & #4: View-Projection 矩阵乘法顺序 + 缺少 LookAt 构建 [Critical]

**文件:** `src/main.cpp`
**修复提交:** `940f825`

### 现象

场景完全不可见或位置完全错误。

### 根因分析

两个问题叠加：

1. **View 矩阵未构建** — 使用了单位矩阵作为 view matrix，相机固定在原点看向 +Z
2. **ViewProj 乘法顺序错误** — 计算了 `view * proj` 而非 `proj * view`

正确的变换链（列主序）：`clip_pos = Proj * View * world_pos`，所以 `ViewProj = Proj * View`。

### 修复

```cpp
// 1. 使用 buildLookAt 构建正确的 view matrix
buildLookAt(eye, center, up, cam.view_matrix);

// 2. 正确的乘法顺序: viewproj = proj * view
mat4Mul(proj, cam.view_matrix, cam.viewproj_matrix);
```

---

## Bug #5: PLY Loader 硬编码属性索引 [High]

**文件:** `src/ply_loader.cpp`
**修复提交:** `b9d6cc2`

### 现象

加载 `cactus.ply`（59 属性，无法线）时崩溃或数据错位，因为 loader 假定固定 62 个属性。

### 根因分析

原始代码：
```cpp
static constexpr int PROPS_PER_VERTEX = 62;  // 硬编码
// 属性索引也是硬编码的: positions at 0-2, normals at 3-5, ...
```

不同 PLY 文件可能有不同的属性集（有/无法线、不同数量的 SH 系数等）。

### 修复

改为 header-driven 解析：解析 PLY header 中的 `property` 声明，动态建立属性名→索引映射。

---

## Bug #6: CLI 参数解析歧义 [Medium]

**文件:** `src/main.cpp`
**修复提交:** `5722fd6`

### 现象

无法同时支持 `./render model.ply 800 600`（自定义分辨率）和 `./render model.ply cameras.json 0`（JSON 相机）。

### 修复

通过检查第二个参数是否以 `.json` 结尾来区分两种模式。

---

## Bug #7: 缺少抗锯齿和异常值剔除 [Medium]

**文件:** `src/cpu/preprocessor_cpu.cpp`, `src/main.cpp`
**修复提交:** `5722fd6`

### 现象

渲染结果有锯齿和零星散点伪影。

### 修复

1. 启用 Mip-splatting 低通滤波（antialiasing）
2. 剔除屏幕半径超过图像尺寸的异常 Gaussian

---

## 经验总结：行主序 vs 列主序的陷阱

### 核心规则

| 系统 | 存储约定 | 构造方式 |
|------|---------|---------|
| **PyTorch / NumPy** | 行主序 `[row][col]` | 按行填充 |
| **GLM / CUDA** | 列主序 `mat[col][row]` | 按列填充 |
| **C float[3][3]** | 行主序 `arr[row][col]` | 按行填充 |

### 关键陷阱

**陷阱 1: GLM mat3 构造器**

```cpp
glm::mat3 M = glm::mat3(a, b, c, d, e, f, g, h, i);
// 填充为: col0=(a,b,c), col1=(d,e,f), col2=(g,h,i)
// M[0][0]=a, M[0][1]=b, M[0][2]=c  (column 0)
```

如果直接把这 9 个值复制到 C 数组 `float M[3][3] = {{a,b,c},{d,e,f},{g,h,i}}`，
则 `M[0][0]=a, M[0][1]=b, M[0][2]=c`，但这是 **row 0**，不是 column 0！

**结果：矩阵被转置。**

**陷阱 2: 对称矩阵的假安全感**

协方差矩阵 Vrk 是对称的，所以 Vrk 的转置不影响结果。但 `R^T * S^2 * R ≠ R * S^2 * R^T`，即使两者都是对称矩阵。不要因为最终结果对称就忽略中间矩阵的转置问题。

### 移植检查清单

移植 CUDA/GLM 代码到纯 C++ 时，对每个矩阵检查：

- [ ] GLM `mat3(a,b,c,d,e,f,g,h,i)` — 参数是按列填充的吗？
- [ ] C `float M[3][3]` — 你的索引约定是 `[row][col]` 吗？
- [ ] 矩阵乘法 — `A*B` 在 GLM 中是标准列主序乘法 (`result[col][row]`)
- [ ] View matrix — `column[3]` 存储了什么？(列主序: translation 在 indices 12-14)
- [ ] 如果矩阵来自 Python/PyTorch — 需要在传入 C++ 前转置

---

## 修复效果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 边缘质量 | 大量毛刺伪影 | 干净清晰 |
| Gaussian 形状 | 方向错误/拉伸 | 正确投影 |
| 多角度一致性 | 部分角度错误 | 6 个角度全部正确 |
| PLY 兼容性 | 仅支持 62 属性文件 | 任意属性布局 |

修复后的渲染结果与参考图 (`cactus_vanilla.png`) 质量一致。
