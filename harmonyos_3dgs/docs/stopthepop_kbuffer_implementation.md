# StopThePop + kBuffer 算法实现文档

## 一、算法概述

StopThePop 的核心目标：消除 3DGS tile-based 渲染中因全局排序近似导致的 **popping 伪影**。

传统 3DGS 排序使用单一全局深度（view-space z），但同一 Gaussian 在不同像素处的"真实深度"不同（3D 椭球体沿不同射线的交点深度不同）。StopThePop 通过 **per-pixel depth** 重新排序，在每个像素处使用该 Gaussian 沿该像素射线的真实投影深度。

### 本实现方案

在 OpenCL/Maleoon 920 上实现 StopThePop 的 **kBuffer** 变体：
- 每像素维护一个大小为 K=16 的深度排序窗口
- 排序键：`depthAlongRay`（Gaussian 中心沿像素射线的投影深度）
- 替代 CUDA 原版的 warp-level shuffle 实现，使用寄存器数组 + 插入排序

## 二、depthAlongRay 算法

### 数学推导

给定 Gaussian 中心偏移 `μ = pos - cam_pos`（世界坐标），逆协方差 `Σ⁻¹`，像素射线方向 `d`：

```
t = (μᵀ Σ⁻¹ d) / (dᵀ Σ⁻¹ d)
```

这是 Gaussian 中心沿射线 `d` 的 **Mahalanobis 投影深度**——Gaussian 对该射线上的最大贡献点的参数化位置。

### 输入数据

| 数据 | 来源 | 维度 | 说明 |
|------|------|------|------|
| `cov3D_inv` | Preprocessor | N×6 | 逆 3D 协方差（对称上三角：[00,01,02,11,12,22]） |
| `mean_offset` | Preprocessor | N×3 | `gaussian_pos - cam_pos`（世界坐标） |
| `ray_dir` | Rasterizer 逐像素计算 | 3 | 从相机经过像素中心的归一化世界坐标射线方向 |

### 射线方向计算

通过逆视图投影矩阵 `inverse_vp` 将像素坐标反投影到世界坐标：

```
ndc = (2 * pixel_x / width - 1, 2 * pixel_y / height - 1, 0, 1)
world_point = inverse_vp * ndc  (齐次除法)
ray_dir = normalize(world_point - cam_pos)
```

每个像素计算一次 `ray_dir`，对 tile 内所有 Gaussian 复用。

## 三、GPU 实现 (rasterize.cl)

### 3.1 Kernel 签名

```opencl
__kernel void rasterize(
    // 原有参数 (arg 0-12)
    __global const float* means2D,      // [N*2]
    __global const float* conics,       // [N*3]
    __global const float* rgb,          // [N*3]
    __global const float* opacities_2d, // [N]
    __global const uint*  values_sorted,// [total_pairs]
    __global const uint*  tile_ranges,  // [num_tiles*2]
    const int width, const int height,
    const float bg_r, const float bg_g, const float bg_b,
    const int grid_x,
    __global float* out_image,          // [W*H*3]
    // AAA-Gaussians (arg 13-15)
    const int eval_3D,
    __global const float* gauss2screen, // [N*16]
    __global const float* depths,       // [N] view-space z (fallback)
    // StopThePop (arg 16-19)
    __global const float* cov3D_inv,    // [N*6] 逆协方差
    __global const float* mean_offset,  // [N*3] pos - cam_pos
    __constant const float* inverse_vp, // [16] 逆视图投影
    __constant const float* cam_pos     // [3] 相机位置
)
```

### 3.2 Shared Memory 布局

```
__local float s_rgb_r[256];           // 1024 B
__local float s_rgb_g[256];           // 1024 B
__local float s_rgb_b[256];           // 1024 B
__local float s_opa[256];             // 1024 B
__local float s_g2s_r0[256*4];       // 4096 B  (eval_3D)
__local float s_g2s_r1[256*4];       // 4096 B  (eval_3D)
__local float s_g2s_r3[256*4];       // 4096 B  (eval_3D)
__local float s_depth[256];           // 1024 B  (fallback depth)
__local float s_cov3d_inv[256*6];    // 6144 B  (StopThePop)
__local float s_mean_off[256*3];     // 3072 B  (StopThePop)
// 2D 专用（与 eval_3D 互斥使用）
__local float s_xy_x/y[256];         // 2048 B
__local float s_conic_a/b/c[256];    // 3072 B
__local int   s_wg_done;             // 4 B
────────────────────────────────────
eval_3D 总计: ~27,604 B / 32,512 B (84.9% 利用率)
```

### 3.3 Per-Pixel 射线计算（每像素一次）

```opencl
float ray_dir[3] = {0, 0, 1};
if (eval_3D && inverse_vp != 0 && px < width && py < height) {
    float ndc_x = (2.0f * fpx) / (float)width - 1.0f;
    float ndc_y = (2.0f * fpy) / (float)height - 1.0f;
    float p4[4] = {ndc_x, ndc_y, 0.0f, 1.0f};
    float wp[4];
    for (int i = 0; i < 4; i++)
        wp[i] = inverse_vp[i]*p4[0] + inverse_vp[4+i]*p4[1]
              + inverse_vp[8+i]*p4[2] + inverse_vp[12+i]*p4[3];
    float wi = 1.0f / (wp[3] + 1e-10f);
    ray_dir[0] = wp[0]*wi - cam_pos[0];
    ray_dir[1] = wp[1]*wi - cam_pos[1];
    ray_dir[2] = wp[2]*wi - cam_pos[2];
    float rl = sqrt(ray_dir[0]*ray_dir[0] + ray_dir[1]*ray_dir[1]
                  + ray_dir[2]*ray_dir[2]);
    if (rl > 1e-10f) { ray_dir[0]/=rl; ray_dir[1]/=rl; ray_dir[2]/=rl; }
}
```

### 3.4 Batch 加载（每批 256 个 Gaussian）

```opencl
// 标准数据
s_rgb_r/g/b[lid] = rgb[gid*3 + 0/1/2];
s_opa[lid] = opacities_2d[gid];

if (eval_3D) {
    // gauss2screen 矩阵 (row0, row1, row3)
    s_g2s_r0[lid*4..+3] = gauss2screen[gid*16 + 0..3];
    s_g2s_r1[lid*4..+3] = gauss2screen[gid*16 + 4..7];
    s_g2s_r3[lid*4..+3] = gauss2screen[gid*16 + 12..15];
    s_depth[lid] = depths[gid];  // fallback

    // StopThePop 数据
    if (cov3D_inv != 0) {
        s_cov3d_inv[lid*6..+5] = cov3D_inv[gid*6..+5];
        s_mean_off[lid*3..+2] = mean_offset[gid*3..+2];
    }
}
```

### 3.5 Per-Pixel depthAlongRay 计算

```opencl
// 从 shared memory 读取该 Gaussian 的 cov3D_inv 和 mean_offset
float ci0..ci5 = s_cov3d_inv[j*6 + 0..5];
float mo0..mo2 = s_mean_off[j*3 + 0..2];

// Σ⁻¹ × ray_dir（对称矩阵乘向量）
float sv0 = ci0*ray_dir[0] + ci1*ray_dir[1] + ci2*ray_dir[2];
float sv1 = ci1*ray_dir[0] + ci3*ray_dir[1] + ci4*ray_dir[2];
float sv2 = ci2*ray_dir[0] + ci4*ray_dir[1] + ci5*ray_dir[2];

// t = (μᵀ Σ⁻¹ d) / (dᵀ Σ⁻¹ d)
float num = mo0*sv0 + mo1*sv1 + mo2*sv2;
float den = ray_dir[0]*sv0 + ray_dir[1]*sv1 + ray_dir[2]*sv2;

float pix_depth = s_depth[j];  // fallback: view-space z
if (fabs(den) > 1e-10f) {
    float ptd = num / den;
    if (ptd > 0.0f) pix_depth = ptd;
}
```

### 3.6 kBuffer 溢出处理 + 排序插入

```opencl
// 1. 溢出时弹出最近元素混合
if (kb_count >= KBUF_K) {
    blend(kb_alpha[0], kb_r[0], kb_g[0], kb_b[0]);  // front-to-back
    shift_left(kb_*);  // 左移删除 [0]
    kb_count--;
}

// 2. 排序插入（从后向前找位置）
int pos = kb_count;
for (int k = kb_count-1; k >= 0; k--)
    if (kb_depth[k] > pix_depth) pos = k; else break;
shift_right(kb_*, pos, kb_count);  // 右移腾位
kb_*[pos] = new_entry;
kb_count++;
```

### 3.7 kBuffer 刷新

```opencl
if (eval_3D) {
    for (int k = 0; k < kb_count; k++) {
        blend(kb_alpha[k], kb_r[k], kb_g[k], kb_b[k]);
        if (T < 0.0001f) break;
    }
}
```

## 四、CPU 实现 (rasterizer_cpu.cpp)

### 4.1 射线计算

```cpp
float ray_dir[3] = {0, 0, 1};
if (pre.cov3D_inv) {
    pixelToWorldDir(fpx, fpy, cam.width, cam.height,
                    inverse_vp, cam.cam_pos, ray_dir);
}
```

### 4.2 Per-Pixel Depth

```cpp
float pix_depth = pre.depths[idx];  // fallback
if (pre.cov3D_inv) {
    float ptd = depthAlongRay(&pre.cov3D_inv[idx*6],
                              &pre.mean_offset[idx*3], ray_dir);
    if (ptd > 0.0f) pix_depth = ptd;
}
```

### 4.3 kBuffer 结构

CPU 使用结构体（GPU 用独立数组避免寄存器压力）：

```cpp
struct KEntry { float depth; uint32_t idx; float alpha; };
KEntry kbuf[16];
```

CPU 在 flush 时通过 `idx` 查找 `pre.rgb[gid*3+ch]`，不在 kBuffer 中存 RGB。

## 五、Host 绑定 (rasterizer_gpu.cpp)

```cpp
// StopThePop per-pixel depth 参数
CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &pre_dev->cov3D_inv));
CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &pre_dev->mean_offset));

// 每帧计算 inverse_vp 并上传
float inverse_vp[16];
invertMatrix4x4(cam.viewproj_matrix, inverse_vp);
cl_mem d_inv_vp = ctx_.createBuffer(16 * sizeof(float), CL_MEM_READ_ONLY);
ctx_.writeBuffer(d_inv_vp, inverse_vp, 16 * sizeof(float));

cl_mem d_cam_pos = ctx_.createBuffer(3 * sizeof(float), CL_MEM_READ_ONLY);
ctx_.writeBuffer(d_cam_pos, cam.cam_pos, 3 * sizeof(float));

CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_inv_vp));
CL_CHECK(cl.clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &d_cam_pos));

// kernel 执行后释放临时 buffer
ctx_.cl().clReleaseMemObject(d_inv_vp);
ctx_.cl().clReleaseMemObject(d_cam_pos);
```

## 六、性能特征

| 配置 | Rasterize | Total | 说明 |
|------|-----------|-------|------|
| 无 kBuffer | 492 ms | 777 ms | 全局 view-space z 排序 |
| kBuffer K=16 (view-space z) | 12,019 ms | 12,375 ms | kBuffer 无效果（排序键相同） |
| **kBuffer K=16 (depthAlongRay)** | **13,369 ms** | **13,715 ms** | **per-pixel 重排序生效** |

### 性能瓶颈分析

1. **K=16 插入排序**：每个 Gaussian × 每个像素做 O(K) 插入 = O(N×K) 每像素
2. **depthAlongRay 计算**：每个 Gaussian × 每像素 6 次乘法 + 3 次加法 + 1 次除法
3. **Shared memory 带宽**：每个 Gaussian 加载 6+3=9 个额外 float
4. **寄存器压力**：K=16 × 5 数组 = 80 个寄存器/线程

### 优化方向

| 优化 | 预期加速 | 复杂度 |
|------|---------|--------|
| 减小 K（K=4 或 K=8） | 2-4x | 简单 |
| subgroup shuffle 替代插入排序 | 2-3x | 中等（需 `sub_group_shuffle`） |
| 与 per-tile culling 组合 | 3-4x | 简单（减少 Gaussian 数量） |

## 七、正确性验证

| 对比 | Mean Diff | Max Diff | 说明 |
|------|-----------|----------|------|
| GPU StopThePop vs CPU StopThePop | 0.10 | 74 | GPU/CPU 一致 |
| StopThePop vs 全局排序 baseline | 1.92 | 159 | per-pixel 重排序生效 |
| StopThePop 视觉检查 | — | — | 无块状伪影 |

## 八、剩余未集成的 AAA 特性

| 特性 | 原因 | 影响 |
|------|------|------|
| Per-Tile Depth (Center/MaxPos) | 已验证导致 tile 边界伪影 | 排序一致性（需 StopThePop 配合） |
| Per-Tile Culling | 已验证导致 tile 内容丢失 | 性能优化（需 StopThePop 配合） |
| Hierarchical 3-level Sorting | 需 `tiled_partition`/warp barrier | 性能（比 kBuffer 快但更复杂） |
| Per-Pixel Full Sort (CUB) | 需 CUB BlockRadixSort | 完全正确排序（性能更差） |
| Distance-based Sort Order | 简单但收益不明确 | 可选排序模式 |
| Cooperative Load Balancing | 需 CUDA cooperative groups | GPU 利用率优化 |
| Proper EWA Scaling Flag | 已通过 dilation_factor 实现等价 | 配置灵活性 |

**结论**：所有可在 OpenCL/Maleoon 920 上可靠实现的推理相关 AAA 特性已全部集成。剩余特性要么已验证会产生伪影（per-tile culling/depth），要么依赖 CUDA 专属原语（hierarchical sorting/load balancing），要么收益不明确（distance sort）。
