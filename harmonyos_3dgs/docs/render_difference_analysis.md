# CUDA vs Reference 渲染差异根因分析

> 针对 `render.py`(CUDA 管线) 与 `render_reference.py`(纯 Python) 产出图片不一致的深入排查

---

## 结论先行

**数学逻辑完全等价，差异来自 6 个数值层面的根因**。按影响程度排序：

| 排名 | 根因 | 影响程度 | 差异类型 |
|------|------|----------|----------|
| **#1** | Alpha 混合累积精度 (float64 vs float32) | **高** | 全局色偏/色差 |
| **#2** | `exp()` 函数实现差异 | **高** | 边缘/半透明区域 |
| **#3** | render_cpu_reference.py 视图变换 bug | **致命** (仅该文件) | 全图错误 |
| **#4** | 浮点运算顺序 & FMA | **中** | 随机像素级噪声 |
| **#5** | 排序等深度稳定性 | **低** | 极少数像素 |
| **#6** | Guard band 边界舍入 | **低** | 画面边缘 |

---

## #1 Alpha 混合累积精度差异 [影响: 高]

**这是最主要的差异来源。**

### Root render_reference.py 中的混合循环

```python
# render_reference.py:284-297
T = 1.0              # ← Python float = float64
C = np.zeros(3)      # ← NumPy 默认 float64 数组

for gi in gaussians_in_tile:
    # opas_2d[gi] 是 float32, math.exp 返回 float64
    alpha = min(0.99, opas_2d[gi] * math.exp(power))

    # rgbs[gi] 是 float32, 乘以 float64 的 alpha 和 T → 提升为 float64
    C += rgbs[gi] * alpha * T
    T = test_T              # T 始终是 float64
```

### CUDA 混合循环

```c
// forward.cu:315-377
float T = 1.0f;             // ← float32
float C[CHANNELS] = { 0 };  // ← float32

float alpha = min(0.99f, con_o.w * exp(power));  // 全部 float32
C[ch] += features[...] * alpha * T;              // float32 累加
```

### 差异量化

| 操作 | CUDA (float32) | Reference (float64) | 相对误差 |
|------|---------------|---------------------|----------|
| 单次 α×T 乘法 | ~10⁻⁷ | ~10⁻¹⁶ | 10⁹ 倍精度差 |
| 累加 100 个 Gaussian | ~10⁻⁵ | ~10⁻¹⁴ | 误差被放大 |
| 累加 500 个 Gaussian | ~10⁻³~10⁻⁴ | ~10⁻¹² | **可见色差** |

对于密集重叠区域（500+ Gaussians/pixel），float32 累加误差可达 `±0.001`，对应 8-bit 图像的 `±0.25` 量化级，肉眼可见。

### 验证方法

```python
# 将 Reference 改为 float32 累加，看差异是否消失：
T = np.float32(1.0)
C = np.zeros(3, dtype=np.float32)
alpha = np.float32(min(0.99, np.float32(opas_2d[gi]) * np.float32(math.exp(power))))
```

---

## #2 exp() 函数实现差异 [影响: 高]

### GPU vs CPU 的 exp() 精度

| 平台 | exp() 实现 | 精度 | ULP 误差 |
|------|-----------|------|----------|
| CUDA `__expf()` | 快速近似 (多项式) | ~2 ULP | ±2×10⁻⁷ 相对误差 |
| CUDA `exp()` (默认) | 完整精度 | ~1 ULP | ±10⁻⁷ |
| CPU `math.exp` (Python) | libm 双精度 | < 1 ULP (float64) | ±10⁻¹⁶ |

### 关键场景: power ≈ 0 (Gaussian 中心附近)

```
power = -0.001
CUDA:  exp(-0.001f) = 0.999000499f  (float32)
CPU:   exp(-0.001)  = 0.999000499833...  (float64)
差异: ≈ 3×10⁻¹⁰ → 截断到 float32 后无差异
```

### 关键场景: power ≈ -5 (Gaussian 边缘)

```
power = -5.0
CUDA:  exp(-5.0f) = 0.006737947f     (float32, 可能有 1 ULP 差异)
CPU:   exp(-5.0)  = 0.0067379469990...  (float64 → float32: 0.006737947f)
```

此处差异很小，但 **乘以 opacity 后取 alpha 阈值判断** 可能导致像素级差异：

```python
alpha = opacity * exp(power)
# 如果 alpha 恰好在 1/255 ≈ 0.00392 附近:
# CUDA: alpha = 0.003921 → 跳过 (< 1/255)
# CPU:  alpha = 0.003923 → 保留 (> 1/255)
# 该 Gaussian 在此像素的贡献被完全不同对待!
```

### 这种阈值效应的影响

每个被错误跳过/保留的 Gaussian 贡献约 `±rgb × alpha × T` 的色差。对于半透明边缘区域 (T ≈ 0.5, alpha ≈ 0.004)，单个 Gaussian 贡献的误差约 0.002（约 0.5 个 8-bit 量化级），但**累积多个这样的边界情况后差异可达数个量化级**。

---

## #3 render_cpu_reference.py 的视图变换 BUG [影响: 致命]

**注意：此 bug 仅存在于 `render_cpu_reference.py`，不影响根目录的 `render_reference.py`。**

### Bug 位置: compute_cov2d() 中的点变换

```python
# render_cpu_reference.py:143-147
vm = viewmatrix  # vm = W2C.transpose(0,1) — 即 W2C 的转置

# BUG: 用行主序索引读转置后的矩阵
t[0] = vm[0,0]*mean3d[0] + vm[0,1]*mean3d[1] + vm[0,2]*mean3d[2] + vm[0,3]
```

### 矩阵存储分析

```python
# world_view_transform = getWorld2View2(R, T).transpose(0,1)
# getWorld2View2 返回:
# W2C = [[R_w2c[0,0], R_w2c[0,1], R_w2c[0,2], t[0]],
#        [R_w2c[1,0], R_w2c[1,1], R_w2c[1,2], t[1]],
#        [R_w2c[2,0], R_w2c[2,1], R_w2c[2,2], t[2]],
#        [0,          0,          0,          1    ]]

# vm = W2C.T (转置):
# vm = [[R_w2c[0,0], R_w2c[1,0], R_w2c[2,0], 0    ],   ← 注意平移跑到第4行!
#       [R_w2c[0,1], R_w2c[1,1], R_w2c[2,1], 0    ],
#       [R_w2c[0,2], R_w2c[1,2], R_w2c[2,2], 0    ],
#       [t[0],       t[1],       t[2],       1    ]]
```

### 代码实际计算的结果

```python
t[0] = vm[0,0]*x + vm[0,1]*y + vm[0,2]*z + vm[0,3]
     = R_w2c[0,0]*x + R_w2c[1,0]*y + R_w2c[2,0]*z + 0
     # ↑ 旋转矩阵列混乱      ↑ 平移丢失!
```

### 正确的应该是

```python
# 方法1: 用转置回来的矩阵
W2C = vm.T
t = W2C[:3, :3] @ mean3d + W2C[:3, 3]

# 方法2: 利用转置后的布局 (CUDA 列主序读法)
t[0] = vm[0,0]*x + vm[1,0]*y + vm[2,0]*z + vm[3,0]
t[1] = vm[0,1]*x + vm[1,1]*y + vm[2,1]*z + vm[3,1]
t[2] = vm[0,2]*x + vm[1,2]*y + vm[2,2]*z + vm[3,2]
```

### 影响

两个错误叠加：**旋转矩阵的行列混用 + 平移向量完全丢失**。导致 camera-space 坐标完全错误，进而影响:
1. Jacobian J 的所有非零元素
2. Guard band 裁剪
3. 深度值 (用于排序)

**但此 bug 仅在 render_cpu_reference.py 中**。根目录的 render_reference.py 使用列主序 flat array + `order='F'` reshape，不存在此问题。

### 验证: render_reference.py 的正确做法

```python
# render_reference.py:117-120 — 正确!
V = view.reshape(4, 4, order='F')  # 列主序 → 正确的 4x4 矩阵
p = V[:3, :3] @ mean3d + V[:3, 3]  # R_w2c @ pos + t ✓
```

---

## #4 浮点运算顺序与 FMA [影响: 中]

### Fused Multiply-Add (FMA)

CUDA GPU 硬件支持 FMA: `a * b + c` 在单条指令中完成，中间结果不截断到 float32。

```c
// GPU 实际执行 (FMA):
C[ch] += features[...] * alpha * T;
// 可能被编译为: fma(features[...] * alpha, T, C[ch])
// 中间结果 features*alpha 保持 float64 精度!
```

CPU 上 Python 的 `*` 和 `+` 是分开执行的，每步截断:

```python
tmp1 = rgbs[gi] * alpha     # 截断为 float64
tmp2 = tmp1 * T             # 截断为 float64
C += tmp2                   # 截断为 float64
```

### 影响

FMA 的差异通常是 ±1 ULP (约 10⁻⁷ 相对误差)。累积 N 个 Gaussian 后差异约 √N × ULP ≈ 10⁻⁵~10⁻⁶，一般不可见。但在特殊情况下（如对消严重时），差异可被放大。

### 运算顺序差异

CUDA 在共享内存中分批处理 256 个 Gaussian。批次边界处的浮点累加顺序与 Reference 的逐个处理不同。由于浮点加法不满足结合律 `(a+b)+c ≠ a+(b+c)`，相同的 Gaussian 集合可能产生微小不同的累加结果。

---

## #5 排序等深度稳定性 [影响: 低]

### 详细分析

已在前一轮讨论。补充一个被忽略的细节：

CUDA 排序的 key 是 `tile_id << 32 | depth_bits`。同一个 Gaussian 在不同 tile 中使用相同的 depth_bits，所以它在所有覆盖的 tile 中的顺序是一致的。

Reference 使用全局排序再分发到 tile，同样保证 tile 内顺序一致。

**唯一差异：相同深度（极罕见）时的 tie-breaking。** CUDA radix sort (稳定) 保持输入顺序，numpy argsort (不稳定) 不保证。

### 次要影响：深度精度

```python
# Reference: depth = p[2] (float32 从矩阵运算得到)
# CUDA: depths[idx] = p_view.z (float32 从 transformPoint4x3 得到)
```

如果中间计算精度不同（如 Reference 的 NumPy 在某些平台用 float64 做矩阵乘），相同 Gaussian 可能得到微小不同的深度值，导致排序顺序不同。

**排序顺序翻转的后果：** 两个深度极为接近的 Gaussian A、B 交换顺序后：

```
原顺序: C = c_A * α_A + c_B * α_B * (1-α_A) + ...
翻转后: C = c_B * α_B + c_A * α_A * (1-α_B) + ...
```

当 depth_A ≈ depth_B 时，两者通常也在空间上重叠，贡献相似的颜色，所以差异很小。

---

## #6 Guard Band 与 Tile Rect 边界舍入 [影响: 低]

### Guard band 裁剪

```python
txtz = min(limx, max(-limx, p[0]/p[2])) * p[2]
```

当 `p[0]/p[2]` 恰好在 `±limx` 边界时，float32 vs float64 的除法可能给出不同的裁剪结果。被裁剪的 Gaussian 其 Jacobian 和 2D 协方差会略有不同。

### Tile rect 边界

```python
rect_min_x = max(0, min(grid_x, int((px - radius) / tile_w)))
```

`px` 是浮点数，`radius` 是整数。如果 `(px - radius) / tile_w` 恰好是整数边界（如 3.99999 vs 4.00001），int() 截断可能给出不同的 tile 分配。

**后果：** 一个 Gaussian 被分配到不同的 tile 集合，某些 tile 的像素会多/少一个 Gaussian 的贡献。

---

## 诊断方法：逐步隔离差异

### Step 1: 消除精度差异

将 Reference 的所有中间变量强制为 float32：

```python
T = np.float32(1.0)
C = np.zeros(3, dtype=np.float32)
alpha = np.float32(min(0.99, np.float32(opas_2d[gi]) * np.float32(np.exp(np.float32(power)))))
C += np.float32(rgbs[gi] * alpha * T)
```

如果改为 float32 后差异显著缩小，则 **#1 是主因**。

### Step 2: 比较中间值

对同一个 Gaussian，打印两个管线的中间值：

```python
# 每个 Gaussian 打印:
print(f"G[{i}]: cov3d={cov3d[:6]}")
print(f"G[{i}]: cov2d=({cov2d[0,0]:.8f}, {cov2d[0,1]:.8f}, {cov2d[1,1]:.8f})")
print(f"G[{i}]: conic=({conic[0]:.8f}, {conic[1]:.8f}, {conic[2]:.8f})")
print(f"G[{i}]: depth={depth:.8f}, radius={radius}")
print(f"G[{i}]: rgb=({rgb[0]:.6f}, {rgb[1]:.6f}, {rgb[2]:.6f})")
```

对应 CUDA 端可在 `preprocessCUDA` 的 `if(idx == TARGET_IDX)` 中加 `printf`。

### Step 3: 比较排序结果

```python
# Reference 排序后的前 20 个 Gaussian 在 tile (10,10) 中的顺序
tile_id = 10 * grid_x + 10
for g in tile_lists[tile_id][:20]:
    print(f"  G[{g}]: depth={depths[g]:.8f}")
```

### Step 4: 单像素对比

选取差异最大的像素，打印两个管线在该像素的完整混合过程：

```python
# 对目标像素 (col, row)，打印每个 Gaussian 的贡献:
for gi in gaussians_in_tile:
    dx, dy = means2d[gi, 0] - col, means2d[gi, 1] - row
    power = ...
    alpha = ...
    print(f"  G[{gi}]: dx={dx:.4f} dy={dy:.4f} power={power:.6f} "
          f"alpha={alpha:.6f} T={T:.6f} contrib={rgb*alpha*T}")
```

---

## 预期差异量级

基于以上分析，两个实现之间的差异预期：

| 指标 | 预期值 | 说明 |
|------|--------|------|
| PSNR | 35~45 dB | 如果仅有精度差异 |
| PSNR | 25~35 dB | 如果叠加少量排序/tile 边界差异 |
| 最大像素误差 | 3~10/255 | 密集重叠区域 |
| 平均像素误差 | 0.5~2/255 | 全图平均 |

**如果 PSNR < 25 dB 或存在结构性差异（而非随机噪声），则大概率存在 bug 而非精度差异。**

---

## render_cpu_reference.py 独有的 Bug 总结

如果你对比的是 `render_cpu_reference.py`（而非根目录的 `render_reference.py`），则还存在 **#3 视图变换 bug**，会导致全图错误。

修复方法：

```python
# render_cpu_reference.py:143-147 — 当前有 bug
# 替换为:
W2C = vm.T  # 转回正确的 W2C 矩阵
t = W2C[:3, :3] @ mean3d + W2C[:3, 3]
```

同样需要修复 `compute_cov2d` 中的 W 提取：

```python
# 当前:
W = vm[:3, :3]  # = R_c2w，但后续公式假设它是按 CUDA 列主序约定

# 应该改为直接按 CUDA 的读法提取:
# (因为后续 T = W @ J 和 cov = T.T @ Σ @ T 的公式在 W = R_c2w 时也能给正确结果，
#  这部分实际上是对的 — 但 transform point 的部分是错的)
```

核心修复只需改 `compute_cov2d` 中的点变换和主循环中的视锥测试。
