# render_reference.py 逐函数实现说明

对应源码：`harmonyos_3dgs/tools/render_reference.py`

这个文件是一个纯 Python + NumPy 的 3D Gaussian Splatting 参考渲染器。它的目标不是性能，而是尽量逐步复现 CUDA 版 `forward.cu` 的前向逻辑，作为对照实现来检查中间结果、定位 C++/GPU 端偏差。

## 整体执行流程

程序入口在 `main()`，整体流程如下：

1. 读取命令行参数
2. 调用 `load_ply()` 加载高斯模型
3. 调用 `load_camera()` 加载相机参数并构造矩阵
4. 调用 `preprocess()` 对所有高斯做屏幕空间预处理
5. 调用 `render_image()` 做逐像素 alpha 合成
6. 保存 PNG，输出一些统计量

从职责上看，这个文件可以分成四层：

- 数据加载：`load_ply()`、`load_camera()`
- 基础数学：`transform_point4x3()`、`transform_point4x4()`、`ndc2pix()`
- 核心几何/颜色计算：`compute_cov3d()`、`compute_cov2d()`、`compute_sh()`
- 渲染流水线：`preprocess()`、`render_image()`、`main()`

---

## 顶部常量

### `SH_C0`, `SH_C1`, `SH_C2`, `SH_C3`

这些是球谐函数的固定系数，用于把每个 Gaussian 存储的 SH 系数恢复成 RGB 颜色。

它们不是经验参数，而是直接对应 CUDA 实现中的常量。这个文件里之所以手工写出来，是为了保证和原始实现的数值一致。

---

## 1. `load_ply(path)`

### 功能

从 PLY 文件中读取每个 Gaussian 的参数，并整理成后续渲染阶段可直接使用的数据结构。

### 输入

- `path`：PLY 文件路径

### 输出

返回一个字典，主要包含：

- `positions`：每个高斯的三维中心，形状 `[N, 3]`
- `sh_coeffs`：重排后的 SH 系数，形状 `[N, max_coeffs, 3]`
- `scales`：每个高斯的缩放
- `rotations`：归一化后的四元数
- `opacities`：sigmoid 后的不透明度
- `sh_degree`：SH 阶数
- `max_coeffs`：每个通道的总 SH 系数数
- `count`：高斯总数

### 实现步骤

#### 1. 读取 PLY 头部

函数先逐行读取头部，直到 `end_header`。在此过程中提取两类信息：

- 顶点数 `n`
- 所有 `property float ...` 的属性名

这样可以知道后续二进制区一共有多少个浮点数，以及每一列对应什么语义。

#### 2. 一次性读取二进制主体

通过：

```python
raw = np.frombuffer(f.read(n * nf * 4), dtype=np.float32).reshape(n, nf)
```

把 PLY 顶点数据读成一个二维数组：

- 行：每个 Gaussian
- 列：每个属性

这一步非常高效，而且避免了逐点解析。

#### 3. 建立属性名到列下标的映射

```python
pi = {name: i for i, name in enumerate(props)}
```

后续就能用 `pi['x']`、`pi['opacity']` 这种方式按名字读取列。

#### 4. 提取三维位置

从 `x, y, z` 三列拼成 `positions`。

#### 5. 提取 SH 的 DC 项

从 `f_dc_0`, `f_dc_1`, `f_dc_2` 三列提取 RGB 的 0 阶 SH 系数。

#### 6. 推断 SH 阶数

程序通过检查 `f_rest_0`, `f_rest_1`, ... 是否存在，统计剩余 SH 项的数量，并据此反推 SH 阶数：

- 至少 45 项，视为 3 阶
- 至少 24 项，视为 2 阶
- 至少 9 项，视为 1 阶
- 否则为 0 阶

#### 7. 重排 SH 系数布局

PLY 中的 SH 系数按通道连续存储，而后面渲染逻辑更适合按“基底序号”访问，因此这里把数据重排成：

`[N, max_coeffs, 3]`

也就是：

- 第二维表示第几个 SH 基底
- 第三维表示 RGB 三个通道

这一步是为了让 `compute_sh()` 可以直接按基底累加颜色。

#### 8. 做参数激活

这里把训练时的参数表示转成渲染时需要的真实值：

- `opacity` 用 sigmoid 激活
- `scale` 用 `exp` 激活
- `rotation` 读成四元数后做归一化

也就是说，这个函数不是简单“读文件”，而是在做“加载 + 解码 + 重排 + 激活”。

---

## 2. `load_camera(json_path, cam_id)`

### 功能

从 `cameras.json` 中读取指定相机，构造视图矩阵、投影矩阵以及组合矩阵。

### 输入

- `json_path`：相机 JSON 路径
- `cam_id`：要加载的相机编号

### 输出

返回一个字典，包含：

- `view_matrix`
- `viewproj_matrix`
- `cam_pos`
- `tan_fovx`, `tan_fovy`
- `width`, `height`
- `focal_x`, `focal_y`

### 实现要点

#### 1. 根据 `id` 找到目标相机

读取整个 JSON 后，筛选 `id == cam_id` 的那一项。

#### 2. 读取相机位姿与内参

包括：

- `position`
- `rotation`
- `fx`, `fy`
- `width`, `height`

注释里明确写了：JSON 里保存的是 `C2W` 旋转，而不是 `W2C`。

#### 3. 构造世界到相机的旋转和平移

由于相机文件给的是 `R_c2w`，要变成 view matrix 里的 `R_w2c`，需要转置：

```python
R_w2c = R_c2w.T
t = -R_w2c @ pos
```

这样构造出的外参矩阵才能把世界坐标点变到相机坐标系。

#### 4. 构造列主序 view matrix

这里不是用普通二维矩阵，而是长度为 16 的一维数组，按 column-major 布局存储。这是为了尽可能模拟 CUDA/GLM 的内存和索引方式。

#### 5. 由焦距得到视场的正切值

```python
tan_fovx = w / (2.0 * fx)
tan_fovy = h / (2.0 * fy)
```

程序并不显式求 `fov`，而是直接保存 `tan(fov/2)`，因为后续投影和裁剪更常用这个值。

#### 6. 构造投影矩阵

这里使用带有 `znear` 和 `zfar` 的透视投影矩阵，仍然采用列主序存储。

#### 7. 计算 `viewproj = proj * view`

内部定义了一个 `mat4mul()`，专门用来做列主序 4x4 乘法。这里不直接用普通二维矩阵乘法，就是为了避免行列约定不一致导致结果偏差。

---

## 3. `transform_point4x3(p, m)`

### 功能

用列主序 4x4 矩阵变换一个三维点，默认齐次坐标 `w = 1`，但只返回结果的前三维。

### 输入

- `p`：三维点
- `m`：列主序 4x4 矩阵，按长度为 16 的一维数组存储

### 输出

- 变换后的三维向量 `[x', y', z']`

### 用途

主要用于：

- 把高斯中心从世界坐标变到相机坐标
- 供 `compute_cov2d()` 使用中间 view-space 坐标

这个函数保留了和 CUDA 版本一致的索引方式。

---

## 4. `transform_point4x4(p, m)`

### 功能

和 `transform_point4x3()` 类似，但返回完整的齐次坐标 `[x', y', z', w']`。

### 用途

主要用于透视投影前的齐次变换。后续在 `preprocess()` 中会对返回结果做透视除法：

```python
p_ndc = p_hom[:3] * (1.0 / p_hom[3])
```

这一步得到标准化设备坐标 NDC。

---

## 5. `ndc2pix(v, S)`

### 功能

把一个 NDC 坐标值映射到像素坐标。

### 输入

- `v`：范围大致在 `[-1, 1]` 的 NDC 坐标
- `S`：对应轴向的像素尺寸，比如图像宽或高

### 输出

- 对应的像素坐标值

### 作用

在 `preprocess()` 中，Gaussian 中心先从 3D 点投影到 NDC，再通过这个函数映射到图像平面上的 `px` 和 `py`。

---

## 6. `compute_cov3d(scale, mod, rot)`

### 功能

根据高斯的局部缩放和旋转，构造其三维协方差矩阵。

### 输入

- `scale`：三个轴向的缩放
- `mod`：全局尺度修正系数
- `rot`：四元数旋转

### 输出

返回 3D 协方差矩阵的 6 个独立分量：

```python
[Sigma00, Sigma01, Sigma02, Sigma11, Sigma12, Sigma22]
```

### 实现逻辑

#### 1. 应用缩放修正

```python
sx, sy, sz = mod * scale[0], mod * scale[1], mod * scale[2]
```

#### 2. 用四元数构造旋转矩阵

这里的公式是按 CUDA/GLM 的列主序约定写的。虽然在 Python 里最终是 NumPy 数组，但作者非常强调这里不是随便写一个旋转矩阵，而是要和原实现的布局一致。

#### 3. 构造尺度矩阵 `S`

```python
S = np.diag([sx, sy, sz])
```

#### 4. 形成局部到世界的形状矩阵 `M`

```python
M = S @ R
```

这表示高斯先按局部主轴缩放，再施加旋转。

#### 5. 构造协方差矩阵

```python
Sigma = M.T @ M
```

由于协方差是对称矩阵，只返回上三角的 6 个元素即可。

### 意义

这一函数把“训练参数里的形状表示”变成“渲染可用的三维椭球形状表示”。这是后续做屏幕投影的基础。

---

## 7. `compute_cov2d(mean3d, cov3d, vm, focal_x, focal_y, tan_fovx, tan_fovy)`

### 功能

将三维高斯的协方差矩阵投影到图像平面，得到二维椭圆协方差。

### 输入

- `mean3d`：高斯中心位置
- `cov3d`：三维协方差的 6 个分量
- `vm`：view matrix
- `focal_x`, `focal_y`：焦距
- `tan_fovx`, `tan_fovy`：视场半角正切值

### 输出

返回二维协方差的 3 个独立分量：

```python
[cov_xx, cov_xy, cov_yy]
```

### 实现逻辑

#### 1. 把高斯中心变到 view space

```python
t = transform_point4x3(mean3d, vm)
```

#### 2. 对投影方向做限制

程序先计算 `t[0] / t[2]` 和 `t[1] / t[2]`，然后根据 `1.3 * tan_fov` 做 clamp。这个处理主要是为了与 CUDA 端行为对齐，并降低边缘处数值不稳定。

#### 3. 构造投影雅可比 `J`

`J` 用来描述 3D 小扰动在透视投影下如何变化成 2D 小扰动，也就是投影的局部线性近似。

#### 4. 从 view matrix 中取旋转线性部分 `W`

这里忽略平移，只保留线性变换部分，因为协方差传播只与线性项有关。

#### 5. 计算变换矩阵 `T = W * J`

作者没有依赖 NumPy 的默认二维矩阵乘法，而是手工用三重循环实现，目的是继续保持与 GLM/column-major 的一致性。

#### 6. 恢复 3D 协方差矩阵

把 6 个独立分量拼回完整的对称矩阵 `Vrk`。

#### 7. 做协方差传播

核心公式是：

```python
cov = T^T * Vrk * T
```

这是线性变换下协方差的标准传播形式。

### 意义

这一函数把“3D 椭球”转成“屏幕上的 2D 椭圆”，后续像素着色时计算高斯权重完全依赖这里的结果。

---

## 8. `compute_sh(degree, sh_coeffs, pos, cam_pos)`

### 功能

根据相机方向和球谐系数，计算当前 Gaussian 的 RGB 颜色。

### 输入

- `degree`：SH 阶数
- `sh_coeffs`：当前高斯的 SH 系数，形状 `[max_coeffs, 3]`
- `pos`：高斯中心位置
- `cam_pos`：相机位置

### 输出

- 一个三维 RGB 向量

### 实现逻辑

#### 1. 计算观察方向

```python
d = pos - cam_pos
d = d / np.linalg.norm(d)
```

得到单位方向向量 `(x, y, z)`。

#### 2. 累加 0 阶 SH 项

```python
result = SH_C0 * sh[0]
```

#### 3. 若阶数足够，继续累加 1 阶、2 阶、3 阶项

程序按固定公式逐项累加，完全对应 CUDA 实现里的 `computeColorFromSH`。

#### 4. 最后整体加偏置 `0.5`

```python
result += 0.5
```

这并不是数学上 SH 展开必须有的步骤，而是原始实现定义中的一部分。

#### 5. 只做下界裁剪

```python
return np.maximum(result, 0.0)
```

这里有个重要细节：

- 只裁到下界 0
- 不裁上界 1

原因是 CUDA 原实现也是这样，最终统一在保存图像前再 clip 到 `[0, 1]`。

### 意义

这个函数负责把每个高斯的视角相关外观恢复出来，是颜色计算的核心。

---

## 9. `preprocess(model, cam, scale_modifier=1.0, antialiasing=True)`

### 功能

对所有高斯做渲染前预处理，把原始模型参数转换成屏幕空间 splatting 所需的参数。

### 输入

- `model`：`load_ply()` 返回的模型字典
- `cam`：`load_camera()` 返回的相机字典
- `scale_modifier`：全局尺寸修正
- `antialiasing`：是否启用抗锯齿补偿

### 输出

返回一个列表，每个元素对应一个可见 Gaussian，包含：

- `idx`
- `px`, `py`
- `depth`
- `conic`
- `rgb`
- `opacity`
- `radius`
- `rect_min`, `rect_max`

### 实现流程

#### 1. 准备焦距和 tile 网格信息

程序先从相机字典中推回 `focal_x`、`focal_y`，并固定每个 tile 大小为 `16x16`。

#### 2. 遍历每个 Gaussian

对每个高斯依次做可见性检查、投影和参数计算。

#### 3. 做视锥裁剪

```python
p_view = transform_point4x3(pos, cam['view_matrix'])
if p_view[2] <= 0.2:
    continue
```

如果中心点在相机后方或过近，就直接跳过。

#### 4. 做透视投影并转到像素坐标

```python
p_hom = transform_point4x4(pos, cam['viewproj_matrix'])
p_w = 1.0 / (p_hom[3] + 1e-7)
p_ndc = p_hom[:3] * p_w
px = ndc2pix(p_ndc[0], cam['width'])
py = ndc2pix(p_ndc[1], cam['height'])
```

这一步得到 Gaussian 中心在屏幕上的位置。

#### 5. 计算 3D 协方差

调用 `compute_cov3d()`。

#### 6. 计算 2D 协方差

调用 `compute_cov2d()`。

#### 7. 做抗锯齿修正

程序先计算原二维协方差的行列式 `det_orig`，然后对对角项各加 `0.3`：

```python
cov2d[0] += 0.3
cov2d[2] += 0.3
```

这样会让屏幕空间核略微变宽，降低锯齿。

如果启用抗锯齿，则计算：

```python
h_scale = math.sqrt(max(0.000025, det_orig / det_plus))
```

这个系数最后会乘到透明度上，用来在核变宽后做能量补偿。

#### 8. 将二维协方差转成 conic 形式

通过二维协方差的逆矩阵构造：

```python
conic = [a, b, c]
```

后续逐像素计算高斯指数时会直接使用这个形式。

#### 9. 估计屏幕覆盖半径

程序通过二维协方差的特征值估计一个覆盖范围，再取 `3 sigma` 作为半径。这是一个典型的裁剪近似，可以把高斯的有效影响范围限定在有限像素区域内。

#### 10. 计算覆盖的 tile 范围

根据中心像素和半径，算出这个 Gaussian 覆盖哪些 `16x16` tile。如果一个 tile 都碰不到，直接跳过。

#### 11. 计算 SH 颜色

调用 `compute_sh()` 获得当前视角下的 RGB。

#### 12. 计算最终透明度

```python
opa = model['opacities'][i] * h_scale
```

即原始透明度再乘抗锯齿补偿因子。

#### 13. 写入结果列表

最终只保留后续渲染所需的屏幕空间信息。

### 意义

`preprocess()` 是整个渲染前向里最关键的准备步骤。它把每个 3D Gaussian 转成一个可直接投到屏幕上、并参与 alpha 合成的 2D splat。

---

## 10. `render_image(preprocessed, cam, bg_color=np.zeros(3))`

### 功能

对预处理后的高斯做 tile-based alpha blending，生成最终图像。

### 输入

- `preprocessed`：`preprocess()` 输出的可见高斯列表
- `cam`：相机信息
- `bg_color`：背景颜色，默认黑色

### 输出

- 浮点图像数组，形状 `[H, W, 3]`

### 实现流程

#### 1. 计算 tile 网格尺寸

仍然按 `16x16` 切分图像。

#### 2. 按 tile 对 Gaussian 分桶

程序遍历每个 Gaussian，把它加入所有覆盖到的 tile 的列表中。

这样做的好处是：某个像素只需要考虑所在 tile 的高斯，不需要对全场景高斯逐个测试。

#### 3. 对每个 tile 按深度排序

```python
tiles[key].sort(key=lambda g: g['depth'])
```

这保证了之后的 alpha 合成按从前到后的顺序进行。

#### 4. 对每个 tile 中的每个像素逐个合成

对一个像素，先初始化：

- `T = 1.0`：当前剩余透过率
- `C = [0, 0, 0]`：累计颜色

然后遍历当前 tile 的所有 Gaussian。

#### 5. 计算像素相对 Gaussian 中心的偏移

```python
dx = g['px'] - px_coord
dy = g['py'] - py
```

#### 6. 用 conic 计算二维高斯指数

```python
power = -0.5 * (con[0]*dx*dx + con[2]*dy*dy) - con[1]*dx*dy
```

如果 `power > 0.0`，直接跳过，说明这个值不符合预期的高斯衰减形式。

#### 7. 计算 alpha

```python
alpha = min(0.99, g['opacity'] * math.exp(power))
```

并且对很小的 alpha 做剪枝：

```python
if alpha < 1.0/255.0:
    continue
```

这是一个常见的性能优化。

#### 8. 进行透过率更新和提前终止

```python
test_T = T * (1.0 - alpha)
if test_T < 0.0001:
    break
```

如果像素几乎已经完全不透明，就不再继续叠加更后面的高斯。

#### 9. 累加颜色

```python
C += g['rgb'] * alpha * T
T = test_T
```

这就是标准的前向 alpha 合成形式。

#### 10. 混合背景颜色

最终像素写成：

```python
image[py, px_coord] = C + T * bg_color
```

### 意义

这个函数是最终成像阶段。它把每个屏幕空间椭圆高斯按深度排序后，逐像素累加成最终图像。

---

## 11. `main()`

### 功能

程序入口，负责串联整个渲染流程。

### 实现流程

#### 1. 检查命令行参数

要求至少提供：

- 模型 PLY 路径
- 相机 JSON 路径
- 相机 ID

输出文件名可选，默认是 `reference.png`。

#### 2. 加载模型

调用 `load_ply()`，并打印高斯数量和 SH 阶数。

#### 3. 加载相机

调用 `load_camera()`，并打印图像分辨率和焦距。

#### 4. 预处理所有高斯

调用 `preprocess()`，并统计可见高斯数量和耗时。

#### 5. 输出首个可见 Gaussian 的关键中间量

如果存在可见高斯，会打印：

- 二维位置 `pos2d`
- 深度 `depth`
- `conic`
- `rgb`
- `opacity`
- `radius`

这些输出是这个参考实现的一个重要用途：便于与 C++/CUDA 实现逐项比对。

#### 6. 调用 `render_image()` 生成图像

统计渲染耗时。

#### 7. 保存图像

程序先把浮点图像裁剪到 `[0, 1]`，再转换成 `uint8`，最后通过 PIL 保存为 PNG。

#### 8. 输出简单统计量

最后会统计“几乎纯白”的像素数量：

```python
white_px = np.sum(np.all(img_uint8 > 245, axis=2))
```

这个值通常可以用来快速判断画面是否有大面积爆白。

---

## 12. `if __name__ == '__main__':`

这是标准 Python 入口保护。

含义是：

- 直接运行这个脚本时，执行 `main()`
- 如果把它作为模块导入，则不会自动执行

---

## 这个文件最核心的理解点

### 1. 它是“参考实现”，不是“高性能实现”

整个文件大量使用显式循环、手工矩阵乘法、列主序索引，都是为了和 CUDA 行为更一致，而不是追求 Python 端速度。

### 2. `preprocess()` 是最关键的准备阶段

真实逐像素合成之前，已经完成了：

- 视锥裁剪
- 中心投影
- 3D 到 2D 的协方差传播
- 屏幕覆盖范围估计
- tile 覆盖范围计算
- SH 颜色恢复

### 3. `render_image()` 的本质是屏幕空间椭圆高斯叠加

它不是三角形栅格化，也不是光线追踪，而是对每个像素累加多个二维高斯的透明度和颜色贡献。

### 4. 中间量对比是这个文件的重要用途

`main()` 会打印第一个可见 Gaussian 的关键数值，这对于调试 C++/CUDA 实现非常有价值，因为你可以逐项检查：

- 投影位置是否一致
- 协方差/圆锥参数是否一致
- SH 颜色是否一致
- 不透明度和半径是否一致

---

## 一个简短总结

如果把这个文件压缩成一句话，它做的事情就是：

“读取 3D Gaussian 模型和相机参数，把每个 3D 高斯投影成屏幕上的 2D 椭圆，再按深度顺序逐像素做 alpha 合成，输出一张参考图像。”

如果继续深入阅读，最值得重点跟的三个函数是：

1. `compute_cov2d()`：理解 3D 椭球如何变成 2D 椭圆
2. `preprocess()`：理解前向流水线如何把模型参数转成屏幕空间参数
3. `render_image()`：理解最终像素颜色是如何累加出来的