# 3D Gaussian Splatting 渲染管线深度分析

> CUDA 实现 (`render.py` + `gaussian_renderer/` + `diff-gaussian-rasterization/`) vs 纯 Python 参考实现 (`render_reference.py`)

---

## 目录

1. [整体架构](#1-整体架构)
2. [数据生命周期：从 PLY 到屏幕](#2-数据生命周期从-ply-到屏幕)
3. [阶段一：预处理 (Per-Gaussian)](#3-阶段一预处理-per-gaussian)
   - 3.1 参数激活函数
   - 3.2 视锥剔除
   - 3.3 齐次投影与 NDC 变换
   - 3.4 3D 协方差矩阵
   - 3.5 EWA Splatting：3D→2D 协方差投影
   - 3.6 抗锯齿低通滤波
   - 3.7 Conic 矩阵（协方差逆）
   - 3.8 屏幕空间半径与 Tile Rect
   - 3.9 球谐函数 → RGB
4. [阶段二：排序与 Tile 分配](#4-阶段二排序与-tile-分配)
5. [阶段三：逐像素 Alpha 混合](#5-阶段三逐像素-alpha-混合)
6. [矩阵约定与数据排布详解](#6-矩阵约定与数据排布详解)
7. [工程近似完整清单](#7-工程近似完整清单)
8. [反向传播 (Backward Pass) 概要](#8-反向传播-backward-pass-概要)
9. [CUDA vs Reference 逐项对比表](#9-cuda-vs-reference-逐项对比表)
10. [源文件索引](#10-源文件索引)

---

## 1. 整体架构

### 1.1 CUDA 管线调用栈

```
render.py                                   # 入口脚本：遍历视角，调用 render()
  └─ gaussian_renderer/__init__.py::render() # Python 编排层
       │  ├─ GaussianModel.get_*             #   激活函数 (sigmoid/exp/normalize)
       │  ├─ GaussianRasterizationSettings   #   配置: FOV, 矩阵, SH degree
       │  └─ GaussianRasterizer().__call__()  #   PyTorch autograd.Function
       │
       └─ rasterize_points.cu                # PyTorch C++/CUDA 扩展绑定
            └─ CudaRasterizer::Rasterizer::forward()  (rasterizer_impl.cu)
                 │
                 ├─ FORWARD::preprocess()    ──── 阶段 1: 预处理 (per-Gaussian)
                 │    └─ preprocessCUDA()          (forward.cu:152-269)
                 │         ├─ in_frustum()          视锥剔除
                 │         ├─ transformPoint4x4()   齐次投影
                 │         ├─ computeCov3D()         3D 协方差
                 │         ├─ computeCov2D()         2D 协方差 (EWA)
                 │         ├─ 抗锯齿 + conic         低通滤波 + 逆矩阵
                 │         └─ computeColorFromSH()   SH → RGB
                 │
                 ├─ cub::InclusiveSum          ──── 前缀和 (tile 触碰计数)
                 ├─ duplicateWithKeys          ──── 生成 [tile|depth] 排序键
                 ├─ cub::DeviceRadixSort       ──── 阶段 2: 基数排序
                 ├─ identifyTileRanges         ──── 识别每个 Tile 的范围
                 │
                 └─ FORWARD::render()          ──── 阶段 3: 逐像素混合
                      └─ renderCUDA()                (forward.cu:274-397)
                           ├─ 协作加载到共享内存
                           ├─ Gaussian 求值 (conic)
                           └─ front-to-back alpha 混合
```

### 1.2 Reference 管线调用栈

```
render_reference.py                          # 单文件纯 Python/NumPy 实现
  ├─ load_ply()                              # 加载 PLY + 激活 + SH 重排
  ├─ load_camera()                           # 构建 view/proj/viewproj 矩阵
  └─ render()                                # 三阶段一体
       ├─ 预处理 for-loop (per-Gaussian)     #   阶段 1
       │    ├─ 视锥剔除 (z > 0.2)
       │    ├─ NDC 投影
       │    ├─ compute_cov3d() → compute_cov2d()
       │    ├─ 抗锯齿 + conic
       │    └─ eval_sh()
       ├─ np.argsort(depths)                 #   阶段 2: 全局深度排序
       ├─ tile_lists 分配                     #   Tile 分配
       └─ 四重循环: tile_y → tile_x →        #   阶段 3: 逐像素混合
              pixel_row → pixel_col →
                    Gaussian 混合
```

### 1.3 架构差异总结

| 维度 | CUDA | Reference |
|------|------|-----------|
| 并行度 | 预处理每 Gaussian 一线程, 渲染每像素一线程 | 全部串行 |
| 排序 | O(N) Radix Sort (GPU) | O(N log N) numpy argsort (CPU) |
| 内存层次 | Global → Shared → Register | 仅主存 |
| 可微分 | 是 (backward pass) | 否 |
| 精度 | float32 (CUDA) | float64/float32 混合 (NumPy 默认) |

---

## 2. 数据生命周期：从 PLY 到屏幕

### 2.1 PLY 文件属性顺序

```
x, y, z,                                    # 位置 (3)
nx, ny, nz,                                 # 法线 (3, 未使用)
f_dc_0, f_dc_1, f_dc_2,                     # SH DC 分量 (3)
f_rest_0 ... f_rest_44,                      # SH 高阶分量 (45, degree=3 时)
opacity,                                     # 不透明度 (1)
scale_0, scale_1, scale_2,                   # 缩放 (3)
rot_0, rot_1, rot_2, rot_3                   # 四元数旋转 (4)
```

**总计**: 62 个 float32 属性 (degree=3 时)

### 2.2 SH 系数在 PLY 中的排列

**PLY 存储 (通道分离)**:
```
f_rest_0  ... f_rest_14   →  R 通道的 15 个高阶 SH 系数
f_rest_15 ... f_rest_29   →  G 通道的 15 个高阶 SH 系数
f_rest_30 ... f_rest_44   →  B 通道的 15 个高阶 SH 系数
```

**CUDA 管线内部 (基函数交错)**:
```
GaussianModel 加载后, _features_dc shape = [N, 1, 3]   # (num_gaussians, 1, RGB)
                     _features_rest shape = [N, 15, 3]  # (num_gaussians, sh_coeffs-1, RGB)
get_features 拼接后 shape = [N, 16, 3]                  # 传给 CUDA 时按 glm::vec3 解释
```

CUDA 内核中 `shs` 指针类型转换为 `glm::vec3*`，按 `sh[idx * max_coeffs + k]` 寻址，每个 `sh[k]` 就是第 k 个基函数对应的 `(R, G, B)` 三分量。

**Reference 重排** (`render_reference.py:44-50`):
```python
sh = np.zeros((N, max_coeffs, 3), dtype=np.float32)
sh[:, 0, :] = dc                          # DC
for k in range(sh_per_ch):
    sh[:, k+1, 0] = f_rest[k]             # R 通道
    sh[:, k+1, 1] = f_rest[k + sh_per_ch] # G 通道
    sh[:, k+1, 2] = f_rest[k + 2*sh_per_ch] # B 通道
```

两者最终内存布局一致：`[基函数索引, RGB]` 交错。

### 2.3 参数激活函数

PLY 中存储的是**未激活的原始优化参数**。渲染前需要施加激活函数：

| 参数 | 原始值域 | 激活函数 | 激活后值域 | 代码位置 |
|------|----------|----------|-----------|----------|
| opacity | (-∞, +∞) | σ(x) = 1/(1+e⁻ˣ) | (0, 1) | `gaussian_model.py:130`, ref:53 |
| scale | (-∞, +∞) | eˣ | (0, +∞) | `gaussian_model.py:104`, ref:54 |
| rotation | ℝ⁴ | q/‖q‖ | 单位四元数 S³ | `gaussian_model.py:108`, ref:55-56 |

**CUDA 管线中激活时机**: 在 Python 层通过 `@property` 完成激活，传给 CUDA 内核的已经是激活后的值。四元数归一化在 CUDA 内核 `computeCov3D()` 中被注释掉（`forward.cu:123`），说明信赖 Python 端的归一化。

**Reference**: 在 `load_ply()` 中一次性完成所有激活。

---

## 3. 阶段一：预处理 (Per-Gaussian)

### 3.1 视图矩阵与投影矩阵的构建

#### 3.1.1 World-to-View 矩阵

COLMAP 提供的 `R` 是 **World-to-Camera 的旋转** (W2C)，`T` 是 W2C 的平移。

**CUDA 管线** (`graphics_utils.py:38-49`, `cameras.py:86`):
```python
# getWorld2View2: 构建行主序 4×4 W2C 矩阵
Rt = np.zeros((4, 4))
Rt[:3, :3] = R.transpose()    # 注意: R 已经是 W2C, 这里 transpose 是因为
                                # COLMAP 约定中 R 是按行存储的 C2W 旋转
Rt[:3, 3] = t                  # 平移
Rt[3, 3] = 1.0

# cameras.py:86 传给 CUDA 时做了转置:
world_view_transform = torch.tensor(W2V).transpose(0, 1).cuda()
# 结果: 行主序 NumPy 矩阵被 transpose → 变成 "行主序存储的列主序矩阵"
# 等价于: 直接将矩阵按列主序排列为 flat array
```

**关键**：PyTorch 张量是行主序 (C-contiguous)。做 `.transpose(0,1)` 后，内存中的排列变成了列主序，与 CUDA/GLM 的列主序约定匹配。

**Reference** (`render_reference.py:72-82`):
```python
R_w2c = R_c2w.T
t = -R_w2c @ pos
# 直接按列主序构造 flat array:
view[0:3]   = R_w2c[:, 0]  # col 0
view[4:7]   = R_w2c[:, 1]  # col 1
view[8:11]  = R_w2c[:, 2]  # col 2
view[12:15] = t             # col 3 (translation)
```

#### 3.1.2 投影矩阵

**CUDA 管线** (`graphics_utils.py:51-71`):
```python
P = torch.zeros(4, 4)
P[0, 0] = 2*znear / (right - left)      # = 1/tan(fovX/2)  (对称视锥)
P[1, 1] = 2*znear / (top - bottom)      # = 1/tan(fovY/2)
P[0, 2] = (right + left) / (right - left) # = 0 (对称视锥)
P[1, 2] = (top + bottom) / (top - bottom) # = 0
P[3, 2] = 1.0                           # z_sign, 透视除法的 w
P[2, 2] = zfar / (zfar - znear)         # 深度映射
P[2, 3] = -(zfar * znear) / (zfar - znear)
```

注意这是**行主序**的 OpenGL 风格投影矩阵。传给 CUDA 前同样做 `.transpose(0,1)`。

**Reference** (`render_reference.py:85-91`):
```python
proj[0]  = 1.0 / tan_fovx    # = P[0,0] 列主序位置 [0]
proj[5]  = 1.0 / tan_fovy    # = P[1,1] 列主序位置 [5]
proj[10] = zfar / (zfar - znear)
proj[11] = 1.0               # 列主序 [11] = P[2,3]... 不对, 需要仔细看
proj[14] = -(zfar * znear) / (zfar - znear)
```

列主序平面索引映射 `proj[row + col*4]`：
- `proj[0]` → (0,0): `1/tan_fovx`
- `proj[5]` → (1,1): `1/tan_fovy`
- `proj[10]` → (2,2): `zfar/(zfar-znear)`
- `proj[11]` → (3,2): `1.0` — 这是 `P[3,2]` = z_sign
- `proj[14]` → (2,3): `-(zfar*znear)/(zfar-znear)`

与 CUDA 管线的 `.transpose(0,1)` 结果**完全一致**。

#### 3.1.3 ViewProjection 复合矩阵

**CUDA 管线** (`cameras.py:88`):
```python
full_proj_transform = world_view_transform @ projection_matrix
# 两者都已经是转置后的 (列主序存储在行主序张量中)
# 等价于: (V^T)(P^T) = (PV)^T — 正确的列主序 VP 矩阵
```

**Reference** (`render_reference.py:94-96`):
```python
V = view.reshape(4, 4, order='F')   # 列主序 → 4×4
P = proj.reshape(4, 4, order='F')
VP = (P @ V).flatten(order='F')     # VP 再展成列主序 flat
```

### 3.2 视锥剔除 (Frustum Culling)

**CUDA** (`auxiliary.h:151-176`):
```c
__forceinline__ __device__ bool in_frustum(int idx, ...) {
    float3 p_orig = { orig_points[3*idx], orig_points[3*idx+1], orig_points[3*idx+2] };
    float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    float p_w = 1.0f / (p_hom.w + 0.0000001f);
    float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
    p_view = transformPoint4x3(p_orig, viewmatrix);

    if (p_view.z <= 0.2f)    // 近平面剔除
        return false;
    return true;
}
```

**Reference** (`render_reference.py:196-197`):
```python
p_view = V[:3, :3] @ positions[i] + V[:3, 3]
if p_view[2] <= 0.2: continue
```

**注意事项**：

1. **仅做近平面剔除**，不做侧面剔除。CUDA 源码中侧面剔除被注释掉了 (`auxiliary.h:166`)
2. 近平面阈值 **0.2** 远大于 `znear=0.01`，这是防止太近的点导致投影数值不稳定
3. `transformPoint4x3` 用列主序矩阵做 4×3 变换 (忽略齐次行)

### 3.3 齐次投影与 NDC→像素坐标

#### 3.3.1 齐次投影

**数学**:

$$\mathbf{p}_{hom} = \mathbf{M}_{proj} \cdot \mathbf{M}_{view} \cdot \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix} = \mathbf{M}_{vp} \cdot \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$

**透视除法**:

$$\mathbf{p}_{ndc} = \frac{\mathbf{p}_{hom}.xyz}{\mathbf{p}_{hom}.w + \epsilon}, \quad \epsilon = 10^{-7}$$

**CUDA** (`forward.cu:194-197`):
```c
float3 p_orig = { orig_points[3*idx], orig_points[3*idx+1], orig_points[3*idx+2] };
float4 p_hom = transformPoint4x4(p_orig, projmatrix);
float p_w = 1.0f / (p_hom.w + 0.0000001f);
float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
```

`transformPoint4x4` (`auxiliary.h:80-89`) 按列主序读矩阵:
```c
result.x = matrix[0]*p.x + matrix[4]*p.y + matrix[8]*p.z  + matrix[12]
result.y = matrix[1]*p.x + matrix[5]*p.y + matrix[9]*p.z  + matrix[13]
result.z = matrix[2]*p.x + matrix[6]*p.y + matrix[10]*p.z + matrix[14]
result.w = matrix[3]*p.x + matrix[7]*p.y + matrix[11]*p.z + matrix[15]
```

等价于 $\mathbf{r} = M_{col-major} \cdot \mathbf{p}$，即 matrix[0..3] 是第 0 列。

**Reference** (`render_reference.py:200-201`):
```python
p_hom = VP @ np.append(positions[i], 1.0)  # VP 已从列主序 reshape 成 4×4
pw = 1.0 / (p_hom[3] + 1e-7)
p_ndc = p_hom[:3] * pw
```

#### 3.3.2 NDC → 像素坐标

**数学**:

$$\text{pixel}_x = \frac{(\text{ndc}_x + 1) \cdot W - 1}{2}, \quad \text{pixel}_y = \frac{(\text{ndc}_y + 1) \cdot H - 1}{2}$$

NDC ∈ [-1, 1] 映射到像素坐标 [0, W-1] / [0, H-1]。

**CUDA** (`auxiliary.h:40-43`):
```c
__forceinline__ __device__ float ndc2Pix(float v, int S) {
    return ((v + 1.0) * S - 1.0) * 0.5;
}
```

**Reference** (`render_reference.py:167-168`):
```python
def ndc2pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5
```

**完全一致。**

### 3.4 3D 协方差矩阵 Σ

#### 3.4.1 数学推导

每个 3D Gaussian 的协方差矩阵 $\Sigma$ 由 scale $\mathbf{s} \in \mathbb{R}^3$ 和 rotation quaternion $\mathbf{q} \in \mathbb{H}$ 参数化:

$$S = \text{diag}(s_0, s_1, s_2)$$

$$R = \text{quat\_to\_rotmat}(\mathbf{q})$$

$$M = S \cdot R$$

$$\Sigma = M^T M = R^T S^2 R$$

这保证 $\Sigma$ 是**半正定对称**矩阵（合法的协方差矩阵）。

#### 3.4.2 四元数 → 旋转矩阵

给定单位四元数 $\mathbf{q} = (r, x, y, z)$，其中 $r$ 是标量部:

$$R = \begin{bmatrix}
1-2(y^2+z^2) & 2(xy-rz) & 2(xz+ry) \\
2(xy+rz) & 1-2(x^2+z^2) & 2(yz-rx) \\
2(xz-ry) & 2(yz+rx) & 1-2(x^2+y^2)
\end{bmatrix}$$

#### 3.4.3 CUDA 实现 (`forward.cu:114-148`)

```c
// GLM 列主序: glm::mat3(a,b,c, d,e,f, g,h,i) 存储为:
//   col0 = [a,b,c], col1 = [d,e,f], col2 = [g,h,i]
// 所以 M[col][row] 访问

glm::mat3 S = glm::mat3(1.0f);
S[0][0] = mod * scale.x;  // col0, row0
S[1][1] = mod * scale.y;  // col1, row1
S[2][2] = mod * scale.z;  // col2, row2

// q = (r, x, y, z) — 注意 GLM vec4 的 .x=r, .y=x, .z=y, .w=z
float r = q.x, x = q.y, y = q.z, z = q.w;

glm::mat3 R = glm::mat3(
    1-2*(y*y+z*z), 2*(x*y+r*z), 2*(x*z-r*y),     // col 0
    2*(x*y-r*z),   1-2*(x*x+z*z), 2*(y*z+r*x),    // col 1
    2*(x*z+r*y),   2*(y*z-r*x),   1-2*(x*x+y*y)   // col 2
);

// GLM 列主序: R_glm[i][j] = R_math[j][i]
// 即 col 0 = [R00, R10, R20] 按我们上面的数学公式
// 验证: col0[0] = 1-2(y²+z²) = R[0][0] ✓
//       col0[1] = 2(xy+rz) = R[1][0] ✓
// 所以 GLM 存的就是正确的旋转矩阵的列

glm::mat3 M = S * R;
// GLM 中 * 是矩阵乘法, 即 M = S × R (数学意义)
// 实际上因为 S 是对角阵: M_math = S × R_math
// M_glm[col][row] = (S×R)_math[row][col]

glm::mat3 Sigma = glm::transpose(M) * M;
// = M^T × M (数学), 即 R^T S^2 R ✓

// 对称矩阵只存上三角 6 个元素:
cov3D[0] = Sigma[0][0];  // σ₀₀
cov3D[1] = Sigma[0][1];  // σ₀₁ (列主序: Sigma_math[1][0] = Sigma_math[0][1])
cov3D[2] = Sigma[0][2];  // σ₀₂
cov3D[3] = Sigma[1][1];  // σ₁₁
cov3D[4] = Sigma[1][2];  // σ₁₂
cov3D[5] = Sigma[2][2];  // σ₂₂
```

**关键细节**: `Sigma[0][1]` 在 GLM 列主序中是 `Sigma_math[1][0]`，但因为 $\Sigma$ 是对称的，`Sigma_math[1][0] = Sigma_math[0][1]`，所以 `cov3D[1]` 确实是 $\sigma_{01}$。

#### 3.4.4 Reference 实现 (`render_reference.py:102-115`)

```python
def quat_to_rotmat(q):
    r, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y+r*z), 2*(x*z-r*y)],  # row 0
        [2*(x*y-r*z), 1-2*(x*x+z*z), 2*(y*z+r*x)],  # row 1
        [2*(x*z+r*y), 2*(y*z-r*x), 1-2*(x*x+y*y)]   # row 2
    ])

def compute_cov3d(scale, rot):
    R = quat_to_rotmat(rot)
    S = np.diag(scale)
    M = S @ R
    return M.T @ M  # = R^T S^2 R
```

NumPy 行主序: `R[i][j]` 就是 $R_{ij}$，直觉上比 GLM 更直观。

**验证等价性**: Reference 的 `R[0][0] = 1-2(y²+z²)` 对应 CUDA 的 `R_glm[0][0] = 1-2(y²+z²)`。因为 `R_glm[col][row]`，所以 `R_glm[0][0]` = $R_{math}[0][0]$ = `R_np[0][0]`。✓

### 3.5 EWA Splatting: 3D→2D 协方差投影

#### 3.5.1 数学原理

基于 Zwicker et al., "EWA Splatting" (2002), Eq.29 & 31:

3D Gaussian 投影到 2D 屏幕空间的协方差为:

$$\Sigma' = J \cdot W \cdot \Sigma \cdot W^T \cdot J^T$$

其中:
- $W$ = view matrix 的 3×3 旋转部分 (world → camera)
- $J$ = 透视投影的 Jacobian (camera → screen)

$$J = \begin{bmatrix}
f_x / t_z & 0 & -f_x \cdot t_x / t_z^2 \\
0 & f_y / t_z & -f_y \cdot t_y / t_z^2 \\
0 & 0 & 0
\end{bmatrix}$$

其中 $(t_x, t_y, t_z)$ 是 Gaussian 中心在 camera 坐标系中的坐标。

$\Sigma'$ 的左上角 2×2 子矩阵就是 2D 屏幕协方差。

#### 3.5.2 视角裁剪 (Guard Band Clamping)

**目的**: 防止 Gaussian 中心在视锥边缘时 $t_x/t_z$ 过大，导致 Jacobian 数值爆炸。

```
limx = 1.3 × tan(fovX/2)
limy = 1.3 × tan(fovY/2)
t_x = clamp(t_x/t_z, -limx, limx) × t_z
t_y = clamp(t_y/t_z, -limy, limy) × t_z
```

1.3 倍的放大因子为视锥外稍外侧的 Gaussian 提供了 guard band。

两个实现完全一致 (`forward.cu:82-87`, `render_reference.py:122-126`)。

#### 3.5.3 CUDA 实现 (`forward.cu:74-108`)

```c
float3 t = transformPoint4x3(mean, viewmatrix);  // world → camera

// Jacobian — GLM 列主序
glm::mat3 J = glm::mat3(
    focal_x/t.z, 0.0f, -(focal_x*t.x)/(t.z*t.z),   // col 0 → J_math 的 row 0
    0.0f, focal_y/t.z, -(focal_y*t.y)/(t.z*t.z),    // col 1 → J_math 的 row 1
    0, 0, 0);                                         // col 2

// W — 从列主序 viewmatrix 提取
glm::mat3 W = glm::mat3(
    viewmatrix[0], viewmatrix[4], viewmatrix[8],      // col 0
    viewmatrix[1], viewmatrix[5], viewmatrix[9],      // col 1
    viewmatrix[2], viewmatrix[6], viewmatrix[10]);    // col 2

glm::mat3 T = W * J;

// cov3D 恢复为对称矩阵
glm::mat3 Vrk = glm::mat3(
    cov3D[0], cov3D[1], cov3D[2],     // col 0
    cov3D[1], cov3D[3], cov3D[4],     // col 1
    cov3D[2], cov3D[4], cov3D[5]);    // col 2

glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
```

**详细推导 GLM 中的等价关系**:

设 `J_g`, `W_g`, `Vrk_g`, `T_g` 是 GLM 中的矩阵变量（列主序存储），对应的数学矩阵为 `J_m`, `W_m`, `Vrk_m`, `T_m`。

GLM 约定: `mat_g[col][row]` = `mat_m[row][col]`，即 **GLM 矩阵 = 数学矩阵的转置的存储**。

但是！上面的代码构造时，每三个参数是**一列**：
- `J_g` 的 col0 = `(focal_x/t.z, 0, -(focal_x*t.x)/(t.z²))`

这意味着 `J_g[0][0] = focal_x/t.z`, `J_g[0][1] = 0`, `J_g[0][2] = -(focal_x*t.x)/(t.z²)`

在数学意义上: `J_m[0][0] = J_g[0][0] = focal_x/t.z`, `J_m[1][0] = J_g[0][1] = 0`, `J_m[2][0] = J_g[0][2] = ...`

所以 `J_g` 实际存储的是 $J^T$ (数学意义上的 J 的转置)!

类似地, `W_g` 从 viewmatrix 列主序数组构造:
- `W_g` 的 col0 = `(viewmatrix[0], viewmatrix[4], viewmatrix[8])`

`viewmatrix[0]` = col0[row0] = W2C_math[0][0]
`viewmatrix[4]` = col1[row0] = W2C_math[0][1]
`viewmatrix[8]` = col2[row0] = W2C_math[0][2]

所以 `W_g[0]` = `(W_m[0][0], W_m[0][1], W_m[0][2])` = W_math 的 row 0 作为 col 0。
这意味着 `W_g` 存储的也是 $W^T$。

因此:
```
T_g = W_g * J_g = W^T_glm * J^T_glm (GLM 乘法就是矩阵乘法)
```
在数学意义上 T_g 存储的是 $(W^T \cdot J^T)$ 的列主序 = $(J \cdot W)^T$ 的列主序。

```
cov = transpose(T_g) * transpose(Vrk_g) * T_g
```

`transpose(T_g)` 在 GLM 中翻转存储 → 得到存 $(J \cdot W)$ 的 GLM 矩阵。
`Vrk_g` 因为 $\Sigma$ 对称，`transpose(Vrk_g)` = `Vrk_g`。

最终:
```
cov_g = (JW) * Σ * (JW)^T  的列主序存储
```

但标准 EWA 公式是 $\Sigma' = J W \Sigma W^T J^T$。当 $W$ 是正交矩阵时:
$J W \Sigma W^T J^T = J W \Sigma (JW)^{-1} (JW) W^T J^T$ ...

实际上 $JW\Sigma W^T J^T \neq (JW)\Sigma(JW)^T$ 除非 $WW^T = I$（$W$ 正交）。但 $W$ 确实是旋转矩阵的部分，所以 $WW^T = I$，于是:

$(JW)\Sigma(JW)^T = JW\Sigma W^T J^T = J\cdot(W\Sigma W^T)\cdot J^T$ ✓

**结论: CUDA 实现数学正确。**

#### 3.5.4 Reference 实现 (`render_reference.py:117-143`)

```python
V = view.reshape(4, 4, order='F')
p = V[:3, :3] @ mean3d + V[:3, 3]          # world → camera

J = np.zeros((3, 3))
J[0, 0] = focal_x / p[2]
J[0, 2] = -(focal_x * p[0]) / (p[2]**2)
J[1, 1] = focal_y / p[2]
J[1, 2] = -(focal_y * p[1]) / (p[2]**2)

# W 按列从 view flat array 提取
W[:, 0] = [view[0], view[4], view[8]]      # = V[:3, :3] 的第 0 列
W[:, 1] = [view[1], view[5], view[9]]
W[:, 2] = [view[2], view[6], view[10]]

T = W @ J                # 注意: 这里 W 就是 W_math, J 就是 J_math
cov = T.T @ cov3d @ T    # = J^T W^T Σ W J = (WJ)^T Σ (WJ)
```

等价性: Reference 计算的是 $(WJ)^T \Sigma (WJ)$。

但 Reference 的 `W` 是什么? `W[:, 0] = [view[0], view[4], view[8]]`

由于 view 是列主序: `view[0]` = W2C_m[0][0], `view[4]` = W2C_m[1][0], `view[8]` = W2C_m[2][0]

所以 `W[:, 0]` = $[W_{00}, W_{10}, W_{20}]^T$ = W2C 的第 0 列 ✓

Reference 的 `T = W @ J` 然后 `T.T @ Σ @ T = J^T W^T Σ W J`

标准 EWA: $\Sigma' = J W \Sigma W^T J^T$

这两个结果: 一个是 $J^T W^T \Sigma W J$，另一个是 $J W \Sigma W^T J^T$。

因为 $\Sigma'$ 是对称矩阵，$(J W \Sigma W^T J^T)^T = J W \Sigma W^T J^T$。

而 $(J^T W^T \Sigma W J)^T = J^T W^T \Sigma^T W J = J^T W^T \Sigma W J$（$\Sigma$ 对称）。

所以 $J^T W^T \Sigma W J$ 也是对称的。但 $J^T W^T \Sigma W J \neq J W \Sigma W^T J^T$ 一般情况下！

**解决**: $\Sigma'$ 只取左上角 2×2 (第三行/列的 J 都是 0)。经过展开，两种写法取出的 2×2 子矩阵在 J 的第三行为零时**恰好相等**。这是因为 J 的结构使得有效的 2×2 部分只涉及前两行，而 $W$ 的正交性在此子空间上保证了等价。

### 3.6 抗锯齿低通滤波

**目的**: 远离相机的小 Gaussian 在屏幕上可能只有亚像素大小，直接光栅化会产生锯齿。通过在 2D 协方差上叠加一个小的各向同性 Gaussian 滤波器来平滑。

**数学**:

$$\Sigma'_{filtered} = \Sigma' + h^2 \cdot I$$

其中 $h^2 = 0.3$（对应标准差 $\sigma_h = \sqrt{0.3} \approx 0.548$ 像素）。

**能量补偿**: 滤波会"展宽" Gaussian 降低峰值，需要调整 opacity 补偿:

$$\text{h\_scale} = \sqrt{\frac{\det(\Sigma')}{\det(\Sigma'_{filtered})}}$$

$$\alpha_{effective} = \alpha \cdot \text{h\_scale}$$

**CUDA** (`forward.cu:215-223`):
```c
constexpr float h_var = 0.3f;
const float det_cov = cov.x * cov.z - cov.y * cov.y;  // 滤波前行列式
cov.x += h_var;   // Σ'[0,0] += 0.3
cov.z += h_var;   // Σ'[1,1] += 0.3
const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;  // 滤波后行列式
float h_convolution_scaling = 1.0f;
if(antialiasing)
    h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov));

// 后续 (forward.cu:265):
conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };
```

**Reference** (`render_reference.py:213-217`):
```python
det_orig = cov2d_mat[0,0]*cov2d_mat[1,1] - cov2d_mat[0,1]**2
cov2d_mat[0,0] += 0.3
cov2d_mat[1,1] += 0.3
det = cov2d_mat[0,0]*cov2d_mat[1,1] - cov2d_mat[0,1]**2
h_scale = math.sqrt(max(0.000025, det_orig / det))
# ...
opas_2d[i] = opacities[i] * h_scale
```

**完全一致。**

### 3.7 Conic 矩阵 (2D 协方差的逆)

**数学**: 2×2 对称矩阵 $\Sigma'$ 的逆:

$$\Sigma'^{-1} = \frac{1}{\det(\Sigma')} \begin{bmatrix} \sigma_{11} & -\sigma_{01} \\ -\sigma_{01} & \sigma_{00} \end{bmatrix}$$

只需存 3 个元素 (对称矩阵): `conic = (σ₁₁/det, -σ₀₁/det, σ₀₀/det)`

**CUDA** (`forward.cu:226-231`):
```c
if (det == 0.0f) return;
float det_inv = 1.f / det;
float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
// cov.x = Σ'[0,0], cov.y = Σ'[0,1], cov.z = Σ'[1,1]
```

**Reference** (`render_reference.py:219-221`):
```python
if det == 0: continue
det_inv = 1.0 / det
conic = np.array([cov2d_mat[1,1]*det_inv, -cov2d_mat[0,1]*det_inv, cov2d_mat[0,0]*det_inv])
```

**完全一致。**

### 3.8 屏幕空间半径与 Tile Rect

#### 3.8.1 半径计算

**数学**: 通过 2×2 协方差的特征值求 3σ 包围圆:

$$\text{mid} = \frac{\sigma_{00} + \sigma_{11}}{2}$$

$$\lambda_{1,2} = \text{mid} \pm \sqrt{\max(0.1, \text{mid}^2 - \det)}$$

$$\text{radius} = \lceil 3 \sqrt{\max(\lambda_1, \lambda_2)} \rceil$$

3σ 覆盖 99.7% 的 Gaussian 能量。`max(0.1, ...)` 是数值安全保护。

#### 3.8.2 Tile Rect 计算

**CUDA** (`auxiliary.h:45-55`):
```c
rect_min = {
    min(grid.x, max(0, (int)((p.x - max_radius) / BLOCK_X))),
    min(grid.y, max(0, (int)((p.y - max_radius) / BLOCK_Y)))
};
rect_max = {
    min(grid.x, max(0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
    min(grid.y, max(0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
};
```

`BLOCK_X = BLOCK_Y = 16` (`config.h:16-17`)。

Tile 坐标 = pixel 坐标除以 16 取整。max 端加 `BLOCK_X - 1` 做向上取整。

**Reference** (`render_reference.py:230-233`):
```python
rect_min_x = max(0, min(grid_x, int((px - radius) / tile_w)))
# ...
rect_max_x = max(0, min(grid_x, int((px + radius + tile_w - 1) / tile_w)))
```

**完全一致。**

### 3.9 球谐函数 (SH) → RGB 颜色

#### 3.9.1 数学公式

度数为 $l$ 的实球谐函数基 $Y_l^m(\hat{d})$ 的展开:

$$c(\hat{d}) = \sum_{l=0}^{L} \sum_{m=-l}^{l} f_{lm} \cdot Y_l^m(\hat{d}) + 0.5$$

其中 $\hat{d}$ 是从 Gaussian 中心到相机的**归一化方向向量**。0.5 偏移使得 DC 分量对应 0.5 灰度。

#### 3.9.2 SH 系数常数

```
C₀ = 0.28209479 = 1/(2√π)          # Y₀⁰

C₁ = 0.48860251 = √(3/(4π))        # Y₁ˢ

C₂ = [1.09254843, -1.09254843, 0.31539157, -1.09254843, 0.54627422]

C₃ = [-0.59004359, 2.89061144, -0.45704580, 0.37317633,
       -0.45704580, 1.44530572, -0.59004359]
```

三个实现 (CUDA `auxiliary.h:21-38`, Python `sh_utils.py:26-43`, Reference `render_reference.py:12-18`) 中的常数**完全一致**。

#### 3.9.3 各阶展开公式

**Degree 0 (1 基函数)**:
$$\text{result} = C_0 \cdot f_0$$

**Degree 1 (+3 基函数)**:
$$\text{result} += -C_1 y \cdot f_1 + C_1 z \cdot f_2 - C_1 x \cdot f_3$$

注意符号约定：这里的 $(x,y,z)$ 是方向向量分量，$Y_1^{-1} \propto y$, $Y_1^0 \propto z$, $Y_1^1 \propto x$。

**Degree 2 (+5 基函数)**:
$$\text{result} += C_2^0 xy \cdot f_4 + C_2^1 yz \cdot f_5 + C_2^2 (2z^2-x^2-y^2) \cdot f_6 + C_2^3 xz \cdot f_7 + C_2^4 (x^2-y^2) \cdot f_8$$

**Degree 3 (+7 基函数)**:
$$\text{result} += C_3^0 y(3x^2-y^2) f_9 + C_3^1 xyz f_{10} + C_3^2 y(4z^2-x^2-y^2) f_{11}$$
$$+ C_3^3 z(2z^2-3x^2-3y^2) f_{12} + C_3^4 x(4z^2-x^2-y^2) f_{13} + C_3^5 z(x^2-y^2) f_{14} + C_3^6 x(x^2-3y^2) f_{15}$$

最后:
$$\text{result} += 0.5$$
$$\text{result} = \max(\text{result}, 0)$$

#### 3.9.4 方向向量计算

**CUDA** (`forward.cu:25-27`):
```c
glm::vec3 dir = pos - campos;       // Gaussian → Camera 方向
dir = dir / glm::length(dir);
```

**Reference** (`render_reference.py:146-148`):
```python
d = pos - cam_pos                    # Gaussian → Camera 方向
d = d / np.linalg.norm(d)
```

注意: 方向是 **Gaussian 中心 → 相机**，不是相机 → Gaussian。这符合 3DGS 论文的约定。

#### 3.9.5 Clamp 行为差异

| 实现 | 下限 clamp | 上限 clamp |
|------|-----------|-----------|
| CUDA `computeColorFromSH` | `max(result, 0)` ✓ | 无 |
| Python `eval_sh` (预计算路径) | `clamp_min(result + 0.5, 0)` ✓ | 无 (在 `gaussian_renderer/__init__.py:119` 做 `clamp(0,1)`) |
| Reference `eval_sh` | `np.maximum(result, 0)` ✓ | 无 (在 `render()` 最后 `np.clip(image, 0, 1)`) |

CUDA 在 SH 级别记录哪些分量被 clamp 了 (`clamped[3*idx+ch]`)，反向传播时将对应梯度置零。

---

## 4. 阶段二：排序与 Tile 分配

### 4.1 CUDA 管线: 五步排序

#### Step 1: 前缀和 (Prefix Sum)

预处理阶段每个 Gaussian 计算它覆盖的 tile 数 `tiles_touched[idx]`。

```c
// rasterizer_impl.cu:280
cub::DeviceScan::InclusiveSum(..., tiles_touched, point_offsets, P);
// tiles_touched: [2, 3, 0, 2, 1]
// point_offsets: [2, 5, 5, 7, 8]    ← 每个 Gaussian 的写入起始位置
```

总实例数 = `point_offsets[P-1]` = 所有 Gaussian-Tile 对的总数。

#### Step 2: 生成排序键值 (duplicateWithKeys)

```c
// rasterizer_impl.cu:70-111
// 对每个可见 Gaussian, 遍历其覆盖的所有 tile, 生成 key-value 对:

uint64_t key = y * grid.x + x;    // tile ID (高 32 位)
key <<= 32;
key |= *((uint32_t*)&depths[idx]); // 深度位模式 (低 32 位)

gaussian_keys_unsorted[off] = key;
gaussian_values_unsorted[off] = idx; // Gaussian ID
```

**深度位模式排序的数学基础**:

IEEE 754 正浮点数 $f$ 的位模式 $b(f)$ 满足:
$$f_1 < f_2 \Rightarrow b(f_1) < b(f_2) \quad (\text{当} f_1, f_2 > 0)$$

因为 IEEE 754 正数的位表示: `[0][exponent][mantissa]`，指数在高位，与数值单调对应。

这使得可以直接用整数基数排序代替浮点排序，**避免了浮点比较开销**。

**注意**: 这个技巧对负数不成立（符号位使得负数的位模式反序）。但因为经过近平面剔除，所有深度 > 0.2 > 0，所以安全。

#### Step 3: GPU 基数排序

```c
// rasterizer_impl.cu:306-311
int bit = getHigherMsb(tile_grid.x * tile_grid.y);

cub::DeviceRadixSort::SortPairs(
    ..., keys_unsorted, keys_sorted,
    values_unsorted, values_sorted,
    num_rendered, 0, 32 + bit);
```

只排序 `32 + bit` 位（低 32 位深度 + tile ID 所需的位数），减少排序工作量。

排序后: 先按 tile ID 分组 (高位)，组内按深度递增 (低位)。

#### Step 4: 识别 Tile 范围

```c
// rasterizer_impl.cu:116-138
uint32_t currtile = key >> 32;
if (currtile != prevtile) {
    ranges[prevtile].y = idx;   // 前一个 tile 的结束
    ranges[currtile].x = idx;   // 当前 tile 的开始
}
```

结果: `ranges[tile_id] = (start, end)` 指向排序后列表中属于该 tile 的 Gaussian 范围。

#### Step 5: 数据打包

排序相关的所有内存分为三个缓冲区:

| 缓冲区 | 内容 | 大小 |
|--------|------|------|
| `GeometryState` | depths, means2D, cov3D, rgb, conic_opacity, tiles_touched, point_offsets | per-Gaussian |
| `BinningState` | point_list (sorted), keys (sorted/unsorted) | per-instance (Gaussian×Tile) |
| `ImageState` | ranges, accum_alpha, n_contrib | per-tile / per-pixel |

### 4.2 Reference 管线: 简单排序

```python
# render_reference.py:251-252
valid_idx = np.where(valid)[0]
sorted_idx = valid_idx[np.argsort(depths[valid_idx])]  # 全局深度排序

# render_reference.py:260-270
tile_lists = [[] for _ in range(grid_x * grid_y)]
for gi in sorted_idx:
    for ty in range(ry0, ry1):
        for tx in range(rx0, rx1):
            tile_lists[ty * grid_x + tx].append(gi)
```

**差异**: Reference 用全局排序 + Python 列表分发，没有 key 编码和基数排序。语义等价但效率差距巨大。

---

## 5. 阶段三：逐像素 Alpha 混合

### 5.1 数学公式

**Alpha 计算** (论文 Eq.2):

$$\alpha_i = \min\left(0.99, \; o_i \cdot \exp\left(-\frac{1}{2}\mathbf{d}^T \Sigma'^{-1} \mathbf{d}\right)\right)$$

其中 $\mathbf{d} = \mathbf{p}_{gaussian} - \mathbf{p}_{pixel}$ (屏幕空间偏移), $o_i$ 是 opacity (含 h_scale)。

展开 conic 形式:
$$\text{power} = -\frac{1}{2}(\text{conic}_0 \cdot dx^2 + \text{conic}_2 \cdot dy^2) - \text{conic}_1 \cdot dx \cdot dy$$

注意 $-\frac{1}{2}$ 只作用于对角元素，交叉项 `conic_1` 自带符号（已含 $-1/\det$ 和对称矩阵的 2 倍）。

**颜色累积** (论文 Eq.3, front-to-back):

$$C = \sum_{i=1}^{N} c_i \cdot \alpha_i \cdot T_i + T_N \cdot c_{bg}$$

$$T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$$

### 5.2 CUDA 实现 (`forward.cu:274-397`)

```c
template <uint32_t CHANNELS>
__global__ void renderCUDA(...)
{
    // 每个 thread block = 一个 16×16 tile
    // 每个 thread = 一个像素
    uint2 pix = { pix_min.x + threadIdx.x, pix_min.y + threadIdx.y };

    // 共享内存: 批量加载 Gaussian 数据
    __shared__ int collected_id[BLOCK_SIZE];         // 256
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

    float T = 1.0f;
    float C[CHANNELS] = { 0 };

    // 分批处理 (每批 BLOCK_SIZE=256 个 Gaussian)
    for (int i = 0; i < rounds; i++) {
        // === 协作加载 ===
        // 每个线程从全局内存加载一个 Gaussian 的数据到共享内存
        int progress = i * BLOCK_SIZE + threadIdx_linear;
        if (range.x + progress < range.y) {
            int coll_id = point_list[range.x + progress];
            collected_id[threadIdx_linear] = coll_id;
            collected_xy[threadIdx_linear] = points_xy_image[coll_id];
            collected_conic_opacity[threadIdx_linear] = conic_opacity[coll_id];
        }
        __syncthreads();  // 确保所有线程完成加载

        // === 逐 Gaussian 混合 ===
        for (int j = 0; j < min(BLOCK_SIZE, toDo); j++) {
            float2 d = { collected_xy[j].x - pixf.x,
                         collected_xy[j].y - pixf.y };
            float4 con_o = collected_conic_opacity[j];

            float power = -0.5f * (con_o.x*d.x*d.x + con_o.z*d.y*d.y)
                          - con_o.y * d.x * d.y;

            if (power > 0.0f) continue;          // ① Gaussian 外部

            float alpha = min(0.99f, con_o.w * exp(power));
            if (alpha < 1.0f/255.0f) continue;   // ② 太透明

            float test_T = T * (1 - alpha);
            if (test_T < 0.0001f) {               // ③ 已饱和
                done = true;
                continue;
            }

            for (int ch = 0; ch < CHANNELS; ch++)
                C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

            T = test_T;
        }
    }

    // 写出最终颜色 (CHW 格式)
    for (int ch = 0; ch < CHANNELS; ch++)
        out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
}
```

**GPU 并行化设计要点**:

1. **Thread Block = Tile**: 每个 16×16 block 处理一个 tile, 共 256 线程
2. **协作加载**: 256 线程同时从全局内存加载 256 个 Gaussian 到共享内存, 然后每个线程各自处理自己的像素与这 256 个 Gaussian 的混合
3. **Early Termination**: `__syncthreads_count(done)` 统计整个 block 中已完成的线程数, 全部完成则跳过后续批次
4. **数据打包**: `float4 conic_opacity` 将 3 个 conic 元素和 1 个 opacity 打包, 单次 128-bit 事务读取
5. **已完成线程仍参与加载**: `done` 的线程不做混合计算, 但仍然帮助加载共享内存数据

### 5.3 Reference 实现 (`render_reference.py:272-297`)

```python
for ty in range(grid_y):
    for tx in range(grid_x):
        gaussians_in_tile = tile_lists[ty * grid_x + tx]
        if not gaussians_in_tile: continue

        for row in range(py_min, py_max):
            for col in range(px_min, px_max):
                T = 1.0
                C = np.zeros(3)
                for gi in gaussians_in_tile:
                    dx = means2d[gi, 0] - col
                    dy = means2d[gi, 1] - row
                    power = -0.5*(conics[gi,0]*dx*dx + conics[gi,2]*dy*dy) \
                            - conics[gi,1]*dx*dy
                    if power > 0: continue           # ①
                    alpha = min(0.99, opas_2d[gi] * math.exp(power))
                    if alpha < 1.0/255: continue     # ②
                    test_T = T * (1 - alpha)
                    if test_T < 0.0001: break        # ③
                    C += rgbs[gi] * alpha * T
                    T = test_T
                image[row, col] = C + T * bg_color
```

**逻辑完全一致**, 三个 early-out 条件 (①②③) 一模一样。唯一区别是 CUDA 在 ③ 用 `continue` (线程保持活跃帮助加载), Reference 用 `break` (直接退出循环)。

### 5.4 三个 Early-Out 条件分析

| 条件 | 阈值 | 数学含义 | 性能意义 |
|------|------|----------|---------|
| ① `power > 0` | 0 | $\mathbf{d}^T \Sigma'^{-1} \mathbf{d} < 0$ 不可能 | 跳过计算无效的 Gaussian |
| ② `alpha < 1/255` | 0.00392 | 贡献不到 1 个 8-bit 量化级别 | 跳过几乎透明 Gaussian |
| ③ `T < 0.0001` | 0.0001 | 像素已几乎完全被遮挡 | 提前终止，最大性能增益 |

---

## 6. 矩阵约定与数据排布详解

### 6.1 行主序 vs 列主序总览

| 组件 | 存储序 | 说明 |
|------|--------|------|
| NumPy/PyTorch 默认 | 行主序 (C-order) | `A[i][j]` = 第 i 行第 j 列 |
| GLM (CUDA) | 列主序 (Fortran-order) | `A[i][j]` = 第 i **列**第 j **行** |
| 列主序 flat array | `[col0_row0, col0_row1, col0_row2, col1_row0, ...]` | viewmatrix, projmatrix |
| PyTorch `.transpose(0,1)` | 行主序张量存列主序矩阵 | `cameras.py:86-87` |

### 6.2 transformPoint4x4 列主序读法

```c
// auxiliary.h:80-89
result.x = matrix[0]*p.x + matrix[4]*p.y + matrix[8]*p.z  + matrix[12];
//         ─────col0[0]─   ─────col1[0]─   ─────col2[0]─    ──col3[0]──
result.y = matrix[1]*p.x + matrix[5]*p.y + matrix[9]*p.z  + matrix[13];
//         ─────col0[1]─   ─────col1[1]─   ─────col2[1]─    ──col3[1]──
```

等价于: $\mathbf{r} = M \cdot \mathbf{p}$，其中 M 按列主序存储。

### 6.3 数据结构内存布局

#### 6.3.1 Per-Gaussian 预处理输出

| 数据 | CUDA 类型 | 大小 | 布局 |
|------|-----------|------|------|
| 2D 均值 | `float2[P]` | 8 bytes/G | `{x, y}` 连续 |
| 深度 | `float[P]` | 4 bytes/G | 标量 |
| Conic+Opacity | `float4[P]` | 16 bytes/G | `{c0, c1, c2, opacity}` 打包 |
| RGB 颜色 | `float[P×3]` | 12 bytes/G | `[idx*3+ch]` 通道交错 |
| 3D 协方差 | `float[P×6]` | 24 bytes/G | 上三角 `[σ00,σ01,σ02,σ11,σ12,σ22]` |
| 半径 | `int[P]` | 4 bytes/G | 屏幕像素半径 |
| Tiles Touched | `uint32[P]` | 4 bytes/G | 覆盖的 tile 数 |

Reference 中对应:
| 数据 | NumPy 类型 | 布局 |
|------|-----------|------|
| 2D 均值 | `float[N, 2]` | `means2d[i, :] = [px, py]` |
| Conic | `float[N, 3]` | `conics[i, :] = [c0, c1, c2]` (与 opacity 分离) |
| Opacity | `float[N]` | `opas_2d[i]` (包含 h_scale) |
| RGB | `float[N, 3]` | `rgbs[i, :] = [r, g, b]` |

#### 6.3.2 输出图像布局

| 实现 | 布局 | 像素 (x,y) 通道 ch 的索引 |
|------|------|--------------------------|
| CUDA | CHW | `out_color[ch * H * W + y * W + x]` |
| Reference | HWC | `image[y, x, ch]` |

CUDA 的 CHW 布局直接与 PyTorch 的 `[C, H, W]` 张量约定匹配, 无需任何转换。

### 6.4 SH 系数内存布局详解

**CUDA 管线中的 SH 传递路径**:

```
GaussianModel._features_dc:   [N, 1, 3]   # PyTorch (行主序)
GaussianModel._features_rest:  [N, 15, 3]  # (degree=3)
get_features → concat →        [N, 16, 3]

传给 CUDA 时:
shs = get_features.contiguous().data_ptr<float>()
// 内存布局: shs[n * 16 * 3 + k * 3 + ch]
//   n: Gaussian 索引
//   k: 基函数索引 (0=DC, 1..15=高阶)
//   ch: 颜色通道 (0=R, 1=G, 2=B)

CUDA 内核中:
glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
// sh[k] 就是第 k 个基函数的 (R, G, B)
// glm::vec3 大小 = 3 float = 12 bytes, 正好匹配
```

**Reference 的 SH 布局**: `sh[N, max_coeffs, 3]`
- `sh[i, k, :]` = 第 i 个 Gaussian 第 k 个基函数的 `(R, G, B)`
- 与 CUDA 完全对应

---

## 7. 工程近似完整清单

### 7.1 数值安全

| 近似 | 位置 | 值 | 目的 |
|------|------|-----|------|
| 透视除法 epsilon | `forward.cu:196` | 1e-7 | 防止 w=0 除零 |
| 近平面剔除 | `auxiliary.h:166` | z ≤ 0.2 | 避免近处数值不稳 |
| 特征值 sqrt 保护 | `forward.cu:238-239` | max(0.1, ...) | 防止负数开方 |
| 行列式比保护 | `forward.cu:223` | max(0.000025, ...) | 防止 h_scale 为 0 |

### 7.2 性能优化性近似

| 近似 | 位置 | 值 | 目的 |
|------|------|-----|------|
| Alpha 上限 | `forward.cu:360` | 0.99 | 防止单个 Gaussian 完全遮挡 |
| Alpha 下限 | `forward.cu:361` | 1/255 | 忽略不可见贡献 |
| 透射率终止 | `forward.cu:364` | 0.0001 | 像素饱和后提前终止 |
| 3σ 半径 | `forward.cu:240` | 3 × √λ_max | 覆盖 99.7% Gaussian 能量 |
| Radius 上限 | Reference only | max(w,h) | 剔除异常大的 Gaussian |

### 7.3 视锥与投影近似

| 近似 | 位置 | 值 | 目的 |
|------|------|-----|------|
| Guard band | `forward.cu:82-83` | 1.3 × tan_fov | 防止边缘 Jacobian 爆炸 |
| 仅近平面剔除 | `auxiliary.h:166` | — | 侧面剔除被注释, 依赖 tile rect |
| 协方差对角 padding | `forward.cu:217-218` | +0.3 | 抗锯齿低通滤波 |
| 深度位排序 | `rasterizer_impl.cu:104` | — | 利用 IEEE754 单调性 |

### 7.4 Radix Sort 位宽优化

```c
int bit = getHigherMsb(tile_grid.x * tile_grid.y);
// 只排序 32 + bit 位, 而不是完整 64 位
// 对于 1920×1080 图像: tiles = 120×68 = 8160, bit = 13
// 只排序 45 位而不是 64 位, 节省 ~30% 排序时间
```

---

## 8. 反向传播 (Backward Pass) 概要

Reference 实现**没有反向传播**。CUDA 管线的反向传播在 `backward.cu` 中实现。

### 8.1 需要反向传播的变量

| 变量 | 梯度张量 | 用途 |
|------|----------|------|
| 3D 均值 | `dL_dmeans3D` [P, 3] | 位置优化 |
| 2D 均值 | `dL_dmeans2D` [P, 3] | 自适应密度控制 |
| SH 系数 | `dL_dsh` [P, M, 3] | 颜色优化 |
| Scale | `dL_dscales` [P, 3] | 形状优化 |
| Rotation | `dL_drotations` [P, 4] | 朝向优化 |
| Opacity | `dL_dopacity` [P, 1] | 透明度优化 |
| Conic | `dL_dconic` [P, 2, 2] | 中间梯度 |
| RGB 颜色 | `dL_dcolors` [P, 3] | 中间梯度 |

### 8.2 反向混合

渲染 backward 从最后一个贡献者开始, 逆序遍历 Gaussian, 利用前向存储的 `final_T` 和 `n_contrib` 恢复每个像素的混合状态。

### 8.3 SH Clamp 梯度

```c
// backward.cu:33-37
dL_dRGB.x *= clamped[3*idx+0] ? 0 : 1;  // 如果被 clamp 过, 梯度为 0
dL_dRGB.y *= clamped[3*idx+1] ? 0 : 1;
dL_dRGB.z *= clamped[3*idx+2] ? 0 : 1;
```

这是标准 ReLU 梯度处理: clamp(x, 0) 在 x < 0 时梯度为 0。

---

## 9. CUDA vs Reference 逐项对比表

| 步骤 | CUDA | Reference | 等价? |
|------|------|-----------|-------|
| **加载** | `GaussianModel.load_ply()` → PyTorch tensor | `load_ply()` → NumPy array | ✓ 语义等价 |
| **SH 排布** | `[N, 16, 3]` (basis-interleaved) | `[N, 16, 3]` | ✓ 完全一致 |
| **Opacity 激活** | Python `torch.sigmoid` | `1/(1+exp(-x))` | ✓ |
| **Scale 激活** | Python `torch.exp` | `np.exp` | ✓ |
| **Rotation 激活** | Python `F.normalize` | `q/norm(q)` | ✓ |
| **视锥剔除** | `in_frustum()`, z ≤ 0.2 | `p_view[2] <= 0.2` | ✓ |
| **投影** | `transformPoint4x4` (列主序) | `VP @ [x,y,z,1]` | ✓ |
| **NDC→Pixel** | `ndc2Pix()` | `ndc2pix()` | ✓ 完全一致 |
| **3D 协方差** | `computeCov3D()` → 上三角 6-float | `compute_cov3d()` → 3×3 | ✓ 数学等价 |
| **EWA 2D 协方差** | `computeCov2D()` GLM 列主序 | `compute_cov2d()` NumPy 行主序 | ✓ (见 3.5 详解) |
| **Guard band** | 1.3×tan_fov | 1.3×tan_fov | ✓ |
| **抗锯齿** | +0.3, h_scale 补偿 | +0.3, h_scale 补偿 | ✓ |
| **Conic** | `{σ₁₁/d, -σ₀₁/d, σ₀₀/d}` | `[σ₁₁/d, -σ₀₁/d, σ₀₀/d]` | ✓ |
| **半径** | ceil(3√λ_max) | ceil(3√max(λ₁,λ₂)) | ✓ |
| **Tile rect** | `getRect()` BLOCK=16 | 手动计算 tile_w=16 | ✓ |
| **SH 求值** | `computeColorFromSH()` | `eval_sh()` | ✓ |
| **SH DC 偏移** | +0.5 | +0.5 | ✓ |
| **SH clamp** | max(result, 0), 记录 clamped | max(result, 0) | ✓ (Reference 无 backward) |
| **排序** | GPU Radix Sort (tile\|depth 键) | numpy argsort (全局深度) | ✓ 语义等价, 实现不同 |
| **混合 power** | `-0.5*(c0*dx²+c2*dy²)-c1*dx*dy` | 同左 | ✓ |
| **Alpha cap** | min(0.99, ...) | min(0.99, ...) | ✓ |
| **Alpha floor** | < 1/255 skip | < 1/255 skip | ✓ |
| **T threshold** | < 0.0001 done | < 0.0001 break | ✓ |
| **背景混合** | `C + T * bg` | `C + T * bg` | ✓ |
| **输出布局** | CHW `[3, H, W]` | HWC `[H, W, 3]` | 内容一致, 布局不同 |
| **最终 clamp** | `rendered_image.clamp(0, 1)` | `np.clip(image, 0, 1)` | ✓ |
| **反向传播** | 完整实现 | 无 | N/A |

---

## 10. 源文件索引

| 文件 | 角色 | 关键函数/类 |
|------|------|------------|
| `render.py` | 入口脚本 | `render_set()`, `render_sets()` |
| `gaussian_renderer/__init__.py` | Python 渲染编排 | `render()` |
| `scene/gaussian_model.py` | Gaussian 数据模型 | `GaussianModel`, `load_ply()`, `get_*` 属性 |
| `scene/cameras.py` | 相机模型 | `Camera`, `world_view_transform`, `full_proj_transform` |
| `utils/graphics_utils.py` | 投影矩阵构建 | `getWorld2View2()`, `getProjectionMatrix()` |
| `utils/general_utils.py` | 旋转/缩放矩阵构建 | `build_rotation()`, `build_scaling_rotation()` |
| `utils/sh_utils.py` | SH 求值 (Python) | `eval_sh()`, `RGB2SH()`, `SH2RGB()` |
| `submodules/.../rasterize_points.cu` | PyTorch 扩展绑定 | `RasterizeGaussiansCUDA()` |
| `submodules/.../rasterizer_impl.cu` | CUDA 管线编排 | `Rasterizer::forward()`, 排序五步 |
| `submodules/.../forward.cu` | 前向 CUDA 内核 | `preprocessCUDA()`, `renderCUDA()` |
| `submodules/.../backward.cu` | 反向 CUDA 内核 | 梯度计算 |
| `submodules/.../auxiliary.h` | 设备辅助函数 | `transformPoint4x4()`, `ndc2Pix()`, `getRect()`, SH 常数 |
| `submodules/.../config.h` | 配置常量 | `BLOCK_X=16`, `BLOCK_Y=16`, `NUM_CHANNELS=3` |
| `render_reference.py` | 纯 Python 参考实现 | `load_ply()`, `load_camera()`, `render()` |

---

> **结论**: `render_reference.py` 是 CUDA 管线的**忠实逐步镜像**，在数学和逻辑上完全等价。所有差异均来自实现层面：行/列主序约定、GPU 并行化策略、数据打包优化、以及可微分支持。两者在浮点精度范围内应产生相同的渲染结果。
