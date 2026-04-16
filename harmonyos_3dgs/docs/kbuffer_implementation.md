# kBuffer Per-Pixel Sorting 实现文档

## 一、算法原理

kBuffer 是 StopThePop 论文中 per-pixel 排序的简化实现。核心思想：每个像素维护一个大小为 K 的深度排序窗口，在全局排序的基础上修正局部排序错误。

### 与全局排序的区别

全局排序使用**一个深度值**（view-space z）对所有 tile 内的 Gaussian 排序。但同一 Gaussian 在不同像素处的"真实深度"可能不同（3D 椭球体的投影深度因像素位置而异）。kBuffer 在每个像素维护一个小窗口，允许在窗口范围内重新排列 Gaussian 的混合顺序。

### 数据结构

每个像素（线程）持有 K 个槽位，每个槽位存储：

```
struct KEntry {
    float depth;    // 排序键（当前为 view-space z）
    float alpha;    // 该 Gaussian 在此像素的 alpha 贡献
    float r, g, b;  // 该 Gaussian 的 RGB 颜色
};
```

GPU 实现中用 5 个独立数组替代结构体以避免寄存器压力：

```opencl
float kb_depth[K];   // 排序键
float kb_alpha[K];   // alpha
float kb_r[K], kb_g[K], kb_b[K];  // RGB
int kb_count;        // 当前缓冲区元素数
```

### 算法流程

```
对 tile 内每个 Gaussian（按全局排序顺序）:
    1. 计算该 Gaussian 在当前像素的 alpha 贡献
    2. 如果 alpha < 1/255, 跳过
    3. 获取排序键 pix_depth（当前为 view-space z）

    4. 如果 kBuffer 已满 (kb_count >= K):
       a. 弹出 buffer[0]（最近的元素）
       b. 将 buffer[0] 执行 alpha blending:
          C += rgb * alpha * T
          T *= (1 - alpha)
       c. 将 buffer 左移一位, kb_count--

    5. 将新元素按 depth 排序插入 kBuffer:
       a. 二分/线性搜索找到插入位置 pos
       b. 将 [pos, kb_count) 右移一位
       c. 在 pos 处写入新元素
       d. kb_count++

处理完所有 Gaussian 后:
    按深度顺序弹出 kBuffer 中剩余的所有元素并 alpha blending
```

## 二、GPU 实现 (rasterize.cl)

### 关键代码段

#### 2.1 数据加载（shared memory）

```opencl
// 每个 work-item 加载一个 Gaussian 的数据到 shared memory
uint gid = values_sorted[load_idx];
s_rgb_r[local_id] = rgb[gid * 3];
s_rgb_g[local_id] = rgb[gid * 3 + 1];
s_rgb_b[local_id] = rgb[gid * 3 + 2];
s_opa[local_id] = opacities_2d[gid];

if (eval_3D) {
    // 加载 gauss2screen 矩阵的 row0, row1, row3（row2 不需要）
    int base = gid * 16;
    s_g2s_r0[local_id*4 .. +3] = gauss2screen[base + 0..3];
    s_g2s_r1[local_id*4 .. +3] = gauss2screen[base + 4..7];
    s_g2s_r3[local_id*4 .. +3] = gauss2screen[base + 12..15];
    // 加载 view-space depth 用于 kBuffer 排序
    s_depth[local_id] = depths[gid];
}
```

**说明**：`depths` 是新增的 kernel 参数，从 `PreprocessDeviceBuffers::depths` 传入。每个 Gaussian 的 view-space z 在预处理阶段已计算好。

#### 2.2 3D per-pixel evaluation（alpha 计算）

```opencl
// 构造两个平面方程
float px0..px3 = g2s_row0 - g2s_row3 * pixel_x;
float py0..py3 = g2s_row1 - g2s_row3 * pixel_y;

// 两平面交线的 Plücker 坐标
float d = px.xyz × py.xyz;   // 方向分量（叉积）
float m = px.w * py.xyz - px.xyz * py.w;  // 矩分量

// Mahalanobis 距离 = |m|² / |d|²
float contrib = (mx² + my² + mz²) / (dx² + dy² + dz²);
float power = -0.5 * contrib;

// alpha = opacity * exp(power)
float alpha = min(0.99, opacity * exp(power));
```

**物理含义**：`power` 是该 Gaussian 3D 椭球体在当前像素射线上的最大贡献点处的 Mahalanobis 距离的负一半。越接近中心，power 越接近 0，alpha 越大。

#### 2.3 kBuffer 溢出处理（pop front + blend）

```opencl
if (kb_count >= KBUF_K) {
    // 弹出 buffer[0]（深度最小/最近的元素）并混合
    float fa = kb_alpha[0];
    float test_T = T * (1.0f - fa);
    if (test_T < 0.0001f) { done = true; break; }  // T 饱和

    float w = fa * T;
    C_r += kb_r[0] * w;    // front-to-back blending
    C_g += kb_g[0] * w;
    C_b += kb_b[0] * w;
    T = test_T;

    // 左移删除 buffer[0]
    for (int k = 0; k < KBUF_K - 1; k++) {
        kb_depth[k] = kb_depth[k+1];
        kb_alpha[k] = kb_alpha[k+1];
        kb_r[k] = kb_r[k+1]; kb_g[k] = kb_g[k+1]; kb_b[k] = kb_b[k+1];
    }
    kb_count = KBUF_K - 1;
}
```

**关键设计**：先弹出再插入，保证每个 Gaussian 都被处理，不会丢失。

#### 2.4 排序插入

```opencl
// 从后向前搜索插入位置
int pos = kb_count;
for (int k = kb_count - 1; k >= 0; k--) {
    if (kb_depth[k] > pix_depth) pos = k; else break;
}

// 右移腾出位置
for (int k = kb_count; k > pos; k--) {
    kb_depth[k] = kb_depth[k-1];
    kb_alpha[k] = kb_alpha[k-1];
    kb_r[k] = kb_r[k-1]; kb_g[k] = kb_g[k-1]; kb_b[k] = kb_b[k-1];
}

// 插入
kb_depth[pos] = pix_depth;
kb_alpha[pos] = alpha;
kb_r[pos] = s_rgb_r[j]; kb_g[pos] = s_rgb_g[j]; kb_b[pos] = s_rgb_b[j];
kb_count++;
```

**复杂度**：每次插入 O(K)，K=16 时为常数。总复杂度 O(N*K)，N 为 tile 内 Gaussian 数。

#### 2.5 kBuffer 刷新（处理完所有 Gaussian 后）

```opencl
if (eval_3D) {
    for (int k = 0; k < kb_count; k++) {
        float fa = kb_alpha[k];
        float test_T = T * (1.0f - fa);
        if (test_T < 0.0001f) break;
        float w = fa * T;
        C_r += kb_r[k] * w;
        C_g += kb_g[k] * w;
        C_b += kb_b[k] * w;
        T = test_T;
    }
}
```

## 三、CPU 实现 (rasterizer_cpu.cpp)

CPU 版本逻辑相同，使用结构体数组：

```cpp
constexpr int KBUF_K = 16;
struct KEntry { float depth; uint32_t idx; float alpha; };
KEntry kbuf[KBUF_K];
int kcount = 0;
```

CPU 版本在 flush 时通过 `idx` 查找 `pre.rgb[gid*3+ch]` 获取颜色，而非像 GPU 一样在 kBuffer 中存储 RGB。这节省了 CPU 侧的内存但增加了间接访问。

## 四、数据流

```
PreprocessorGPU                  SorterGPU                   RasterizerGPU
┌──────────────┐            ┌──────────────┐            ┌──────────────────┐
│ depths[N]    │──(GPU)──→  │              │            │ s_depth[256]     │
│ gauss2screen │──(GPU)──→  │ RadixSort    │──(GPU)──→  │ s_g2s_r0/r1/r3  │
│   [N*16]     │            │ keys+values  │            │ s_rgb/s_opa      │
│ rgb[N*3]     │──(GPU)──→  │              │            │                  │
│ opacities_2d │──(GPU)──→  │              │            │ kb_depth[16]     │ ← 寄存器
│   [N]        │            │              │            │ kb_alpha[16]     │ ← 寄存器
└──────────────┘            └──────────────┘            │ kb_r/g/b[16]    │ ← 寄存器
                                                        └──────────────────┘
```

**关键路径**：
1. `depths` 在 preprocess kernel 中计算（`out_depths[i] = p_view[2]`）
2. 通过 `PreprocessDeviceBuffers::depths` 保持在 GPU 上
3. rasterizer_gpu.cpp 将其作为 kernel 参数传入
4. kernel 中加载到 `s_depth[BLOCK_SIZE]` shared memory
5. 每个线程在处理 Gaussian 时读取 `s_depth[j]` 作为排序键

## 五、host 侧绑定 (rasterizer_gpu.cpp)

```cpp
// 新增的 kernel 参数（arg index = 16）
CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &pre_dev->depths));
```

kernel 签名对应：

```opencl
__kernel void rasterize(
    ...                             // arg 0-12: 原有参数
    const int eval_3D,              // arg 13
    __global const float* gauss2screen,  // arg 14
    __global const float* depths         // arg 15 ← 新增
)
```

## 六、寄存器与 shared memory 用量

### 寄存器（每线程）

| 数组 | 大小 | 类型 | 寄存器数 |
|------|------|------|---------|
| kb_depth | K=16 | float | 16 |
| kb_alpha | K=16 | float | 16 |
| kb_r | K=16 | float | 16 |
| kb_g | K=16 | float | 16 |
| kb_b | K=16 | float | 16 |
| kb_count | 1 | int | 1 |
| **合计** | | | **81** |

加上原有寄存器（T, C_r/g/b, fpx/fpy, power, alpha 等约 20 个），每线程约 100 个寄存器。Maleoon 920 的寄存器文件大小未公开，K=16 可能导致寄存器溢出到 local memory。

### Shared memory（每 work-group）

| 数组 | 大小 | 字节 |
|------|------|------|
| s_rgb_r/g/b | 256×3 | 3,072 |
| s_opa | 256 | 1,024 |
| s_g2s_r0/r1/r3 | 256×4×3 | 12,288 |
| s_depth | 256 | 1,024 |
| s_xy_x/y, s_conic_a/b/c | 256×5 | 5,120 |
| s_wg_done | 1 | 4 |
| **合计** | | **22,532** |

Maleoon 920 local memory = 32,512 bytes，剩余约 10KB 余量。

## 七、当前限制与未来方向

### 排序键的选择

当前使用 **view-space z** 作为 kBuffer 排序键，与全局 radix sort 的排序键一致。因此 kBuffer 不会改变混合顺序，输出与无 kBuffer 完全相同。

要发挥 kBuffer 的真正价值，需要使用 **per-pixel 3D depth** 作为排序键。AAA-Gaussians 原版使用 `max_pos` 的 z 分量（3D 最大贡献点的 NDC 深度），但该计算在 `dd` 接近零时数值不稳定。

### 可能的改进方向

1. **稳定的 per-pixel depth**：用 `depthAlongRay`（已实现的数学函数）代替 `max_pos` z 分量，在 rasterizer 中使用 `cov3D_inv` 计算每个 Gaussian 在当前像素射线上的投影深度
2. **减小 K**：K=8 或 K=4 减少寄存器压力，以质量换性能
3. **条件启用**：通过环境变量或配置项控制是否启用 kBuffer

### 性能特征

| 模式 | Rasterize | 正确性 |
|------|-----------|--------|
| 无 kBuffer（直接混合） | 492 ms | ✓ mean=0.0001 vs CPU |
| kBuffer K=16 (view-space z) | 12,018 ms | ✓ mean=0 vs 无 kBuffer |

性能退化 ~24x，主要来自 K=16 的插入排序（每 Gaussian × 每像素 O(K) 操作）。
