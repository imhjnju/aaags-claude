# AAA-Gaussians 待移植优化方案 — 详细实现计划

## 待移植清单

| # | 优化方案 | 影响 | 涉及阶段 |
|---|----------|------|----------|
| 1 | 逐Tile 3D裁剪 | 减少~47%渲染耗时 | Tile Binning |
| 2 | Per-Tile Depth Key | 改善排序一致性，消除popping | Tile Binning |
| 3 | 分层排序 (StopThePop) | 消除popping伪影 | Rasterization |
| 4 | 视空间包围盒 (new_aabb) | 修复近平面边缘Gaussian丢弃 | Preprocessing |

---

## 方案 1: 逐Tile 3D裁剪

### 问题

当前 tile binning 阶段，一个 Gaussian 只要其 bounding rect 覆盖了某个 tile，就会为该 tile 生成一个 key-value 对。但实际上很多 Gaussian 对覆盖的 tile 贡献为零（非轴对齐椭球体的轴对齐包围盒远大于实际覆盖范围）。多余的 tile-pair 浪费排序和光栅化算力。

### 算法

对每个 (Gaussian, Tile) 组合，调用 `maxContribGaussianFrustum3D(tile_min, tile_max, g2s, depth)` 判断该 Gaussian 是否对该 tile 有实际贡献。无贡献则不生成 key-value 对。

```
对每个有效 Gaussian i:
  for tile (x, y) in bounding_rect[i]:
    tile_min = (x * TILE_W, y * TILE_H)
    tile_max = ((x+1)*TILE_W - 1, (y+1)*TILE_H - 1)
    max_contrib = maxContribGaussianFrustum3D(tile_min, tile_max, gauss2screen[i])
    if max_contrib <= opacity_power_threshold[i]:
      生成 key-value 对    // 该 tile 有贡献
    else:
      跳过                  // 该 tile 无贡献
```

### 需要修改的文件

#### CPU: `src/cpu/tile_binner_cpu.cpp`

**当前逻辑** (L52-75): 对 bounding rect 中的每个 tile 无条件生成 key-value 对。

**修改方案**:

1. `bin()` 函数签名不变，但需要访问 `PreprocessOutput` 中的 `gauss2screen` 和 `opacities_2d`

2. 在 key 生成循环中加入裁剪判断:

```cpp
// 在 for (int y = rect_min[1]; y < rect_max[1]; y++) 循环内
if (pre.eval_3D && pre.gauss2screen) {
    float tile_min_x = (float)(x * cfg.tile_w);
    float tile_min_y = (float)(y * cfg.tile_h);
    float tile_max_x = (float)((x+1) * cfg.tile_w - 1);
    float tile_max_y = (float)((y+1) * cfg.tile_h - 1);
    float max_depth;
    float mc = maxContribGaussianFrustum3D(
        tile_min_x, tile_min_y, tile_max_x, tile_max_y,
        &pre.gauss2screen[i * 16], max_depth);
    float opa_threshold = std::log(pre.opacities_2d[i] / (1.0f/255.0f));
    if (mc > opa_threshold)
        continue;  // 跳过该 tile
}
```

3. **注意**: 裁剪后实际生成的 pair 数量 < `tiles_touched[i]` 的总和。需要在第一遍循环中先计算实际 tile 数（含裁剪），然后重新做 prefix sum。或者改为两遍循环:
   - 第一遍: 计算每个 Gaussian 的实际 tile 数（替代 `tiles_touched`）
   - 第二遍: prefix sum + 填充 key-value

**更简单的方案**: 在 preprocessor 阶段就将裁剪后的 tile 数写入 `tiles_touched`。这样 binner 的 prefix sum 和 key 生成自然匹配。

**推荐方案**: 在预处理阶段 (`preprocessor_cpu.cpp`) 计算 `tiles_touched` 时即进行逐 tile 裁剪:

```cpp
// 替代当前的: tiles_touched = (rect_max_y - rect_min_y) * (rect_max_x - rect_min_x)
int actual_tiles = 0;
for (int ty = rect_min_arr[1]; ty < rect_max_arr[1]; ty++) {
    for (int tx = rect_min_arr[0]; tx < rect_max_arr[0]; tx++) {
        float tmin_x = (float)(tx * cfg.tile_w);
        float tmin_y = (float)(ty * cfg.tile_h);
        float tmax_x = (float)((tx+1) * cfg.tile_w - 1);
        float tmax_y = (float)((ty+1) * cfg.tile_h - 1);
        float md;
        float mc = maxContribGaussianFrustum3D(tmin_x, tmin_y, tmax_x, tmax_y,
                                                gauss2screen, md);
        if (mc <= opacity_power_threshold) actual_tiles++;
    }
}
out.tiles_touched[i] = actual_tiles;
```

然后在 `tile_binner_cpu.cpp` 中也加入同样的裁剪判断（保证 key 数量匹配 prefix sum）。

#### GPU: `src/gpu/kernels/preprocess.cl` + `src/gpu/kernels/scatter.cl`

**preprocess.cl**: 在 eval_3D 路径中，`tiles_touched` 改为逐 tile 裁剪后的计数:

```opencl
int actual_tiles = 0;
for (int ty = rmy; ty < rMy; ty++) {
    for (int tx = rmx; tx < rMx; tx++) {
        float tmin[2] = {(float)(tx*tile_w), (float)(ty*tile_h)};
        float tmax[2] = {(float)((tx+1)*tile_w-1), (float)((ty+1)*tile_h-1)};
        float md;
        float mc = maxContribGaussianFrustum3D(tmin[0],tmin[1],tmax[0],tmax[1], g2s, &md);
        if (mc <= opt) actual_tiles++;
    }
}
out_tiles_touched[i] = actual_tiles;
```

**scatter.cl**: 加入 `gauss2screen` 和 `opacities_2d` 参数，在 key 生成循环中做同样的裁剪:

```opencl
__kernel void scatter(
    ...,
    const int eval_3D,
    __global const float* gauss2screen,   // [N*16]
    __global const float* opacities_2d    // [N]
) {
    ...
    for (int y = rect_min_y; y < rect_max_y; y++) {
        for (int x = rect_min_x; x < rect_max_x; x++) {
            if (eval_3D) {
                // 逐 Tile 3D 裁剪
                float tmin_x = (float)(x * tile_w);
                float tmin_y = (float)(y * tile_h);
                float tmax_x = (float)((x+1) * tile_w - 1);
                float tmax_y = (float)((y+1) * tile_h - 1);
                float md;
                float mc = maxContribGaussianFrustum3D(tmin_x, tmin_y, tmax_x, tmax_y,
                    gauss2screen + idx*16, &md);
                float opa_thresh = log(opacities_2d[idx] / (1.0f/255.0f));
                if (mc > opa_thresh) continue;
            }
            // 生成 key-value 对
            ...
        }
    }
}
```

#### Host: `src/gpu/tile_binner_gpu.h/.cpp`

- scatter kernel 增加 3 个参数: `eval_3D`, `gauss2screen`, `opacities_2d`
- 从 `PreprocessDeviceBuffers` 获取 `gauss2screen` 设备缓冲
- 需要注意: scatter.cl 中要内联 `maxContribGaussianFrustum3D` 的全部代码（或在 scatter.cl 开头 include 这些函数）

### 性能预期

- tile-pair 数量减少 ~40%（论文数据）
- 排序耗时相应减少（排序是 O(n log n)）
- 光栅化耗时减少（每个 tile 处理更少 Gaussian）
- 预处理/scatter 阶段略微增加（每个 Gaussian-Tile 对多一次 frustum 检查）
- 总体性能提升显著

---

## 方案 2: Per-Tile Depth Key

### 问题

当前所有 tile 中同一 Gaussian 使用相同的 depth key（view-space z）。但从不同 tile 观察，同一 Gaussian 的有效深度不同。这导致两个重叠的 Gaussian 在不同 tile 中的排序可能不一致，产生 tile 边界处的 popping 伪影。

### 算法

为每个 (Gaussian, Tile) 组合计算独立的 depth key，使用 `depthAlongRay()` 函数：

```
对每个 (Gaussian i, Tile (x,y)):
    tile_center = ((x+0.5)*TILE_W, (y+0.5)*TILE_H)
    viewdir = normalize(pix2world(tile_center) - cam_pos)
    depth = depthAlongRay(inv_cov3D, viewdir)
```

`depthAlongRay` 计算 Gaussian 中心沿观察射线的投影深度：
```
viewdir_inv_cov = inv_cov3D × viewdir
num = mean_offset · viewdir
den = viewdir · viewdir_inv_cov
depth = num / den
```

### 需要新增的数据

1. **3D 逆协方差矩阵** `cov3D_inv`: 当前未存储。需要在预处理阶段计算并存储。

   对 eval_3D 模式，可以从 gauss2screen 推导。但更直接的方式是在预处理中额外计算:
   ```
   cov3D = R × S² × Rᵀ    // 已有 computeCov3D
   cov3D_inv = inverse(cov3D)  // 6 floats (对称矩阵)
   ```

2. **逆 ViewProj 矩阵** `inverse_vp`: 用于 `pix2world()` 将像素坐标转回世界坐标。每帧计算一次。

### 需要修改的文件

#### `include/types.h`

- `PreprocessOutput` 新增: `float* cov3D_inv` (N×6，对称矩阵上三角)

#### `src/cpu/preprocessor_cpu.cpp`

在 eval_3D 路径中计算并存储 `cov3D_inv`:
```cpp
// 利用已有的 scale_dilated 和 R 计算
// cov3D = R × diag(sd²) × Rᵀ
// cov3D_inv = R × diag(1/sd²) × Rᵀ
float inv_sd2[3] = {1.0f/(sd[0]*sd[0]), 1.0f/(sd[1]*sd[1]), 1.0f/(sd[2]*sd[2])};
// cov3D_inv[0] = R[0][0]²/sd0² + R[1][0]²/sd1² + R[2][0]²/sd2²  ... (6 元素)
```

#### `src/cpu/tile_binner_cpu.cpp`

在 key 生成循环中，为每个 tile 计算独立的 depth:
```cpp
if (pre.eval_3D && pre.cov3D_inv) {
    float tile_cx = (x + 0.5f) * cfg.tile_w;
    float tile_cy = (y + 0.5f) * cfg.tile_h;
    // pix2world: 像素坐标 → 世界坐标
    float world_pt[3];
    pix2world(tile_cx, tile_cy, cam.width, cam.height, inverse_vp, world_pt);
    float viewdir[3] = {world_pt[0]-cam.cam_pos[0], ...};
    normalize(viewdir);
    float depth = depthAlongRay(&pre.cov3D_inv[i*6], viewdir);
    memcpy(&depth_bits, &depth, 4);
}
```

#### GPU: `scatter.cl`

同样逻辑，每个 tile 计算独立的 depth key。需要传入:
- `cov3D_inv` 缓冲 (N×6 float)
- `inverse_vp` 矩阵 (16 float, uniform)
- `cam_pos` (3 float, uniform)

### 注意事项

- Per-Tile Depth 的收益需要配合分层排序才能充分体现
- 单独使用 Per-Tile Depth（不配合分层排序）也能改善排序一致性，减轻 popping
- 性能开销: 每个 Gaussian-Tile 对增加一次 `depthAlongRay` 计算（~15 FLOPS），影响很小

---

## 方案 3: 分层排序 (StopThePop Hierarchical Sorting)

### 问题

即使使用 Per-Tile Depth，全局排序仍然是近似的——一个 tile 内的所有像素共享相同的排序。两个在某些像素处前后关系相反的 Gaussian，在全局排序中只能有一个顺序。

### 算法

在光栅化阶段，对每个像素使用三级缓冲区进行**逐像素重排序**:

```
TAIL (64 elements, shared memory per 4×4 tile):
  Bitonic Sort 对 32 个元素排序

MID (8-12 elements, shared memory per 4×4 group):
  从 TAIL 接收最小元素，归并插入

HEAD (4 elements, registers per thread/pixel):
  从 MID 接收最小元素，插入排序
  满时弹出最前面的执行 alpha 混合
```

### CUDA → OpenCL 映射

| CUDA 概念 | OpenCL 等价 | 说明 |
|-----------|-------------|------|
| 3D thread block (16×4×4) | 1D work-group (256) + 手动索引 | `lid_x = lid%16, grp_y = (lid/16)%4, grp_z = lid/64` |
| `__shfl_sync(mask, val, lane)` | `sub_group_shuffle(val, lane)` | 需要 `cl_khr_subgroups` 扩展 |
| `__ballot_sync(mask, pred)` | `sub_group_ballot(pred)` | 返回 `uint` 而非 `uint32_t` |
| `__popc(x)` | `popcount(x)` | OpenCL 内置 |
| `__fns(mask, base, offset)` | 手动实现 | `find nth set bit` |
| CUB `BlockRadixSort` | 手写 local memory radix sort | 或用 bitonic sort |
| `__shared__` | `__local` | 相同语义 |
| warp (32 threads) | sub_group (通常 16-64) | Maleoon GPU 需查询实际 sub_group size |

### 需要修改的文件

#### `src/gpu/kernels/rasterize.cl` — 完全重写光栅化 kernel

当前 kernel: 256 threads/WG，每 thread 处理 1 pixel，简单前到后混合。

新 kernel 结构:
```opencl
#define TILE_W 16
#define TILE_H 16
#define WG_SIZE 256
#define SUB_TILE 4    // 4×4 sub-tile
#define HEAD_SIZE 4
#define MID_SIZE 8
#define TAIL_SIZE 64

__kernel void rasterize_hierarchical(...)
{
    int lid = get_local_id(0);
    int tile_id = get_group_id(0);

    // 像素位置
    int px = (tile_id % grid_x) * TILE_W + (lid % TILE_W);
    int py = (tile_id / grid_x) * TILE_H + (lid / TILE_W);

    // Sub-tile 索引 (4×4)
    int sub_x = (lid % TILE_W) / SUB_TILE;  // 0-3
    int sub_y = (lid / TILE_W) / SUB_TILE;  // 0-3
    int sub_id = sub_y * (TILE_W/SUB_TILE) + sub_x;  // 0-15
    int in_sub = (lid % SUB_TILE) + ((lid / TILE_W) % SUB_TILE) * SUB_TILE;  // 0-15

    // HEAD: per-thread registers
    float head_depths[HEAD_SIZE];
    int head_ids[HEAD_SIZE];
    int head_count = 0;

    // MID: per-sub-tile shared memory
    __local float mid_depths[16][MID_SIZE];   // 16 sub-tiles
    __local int mid_ids[16][MID_SIZE];
    __local int mid_count[16];

    // TAIL: per-sub-tile shared memory
    __local float tail_depths[16][TAIL_SIZE];
    __local int tail_ids[16][TAIL_SIZE];
    __local int tail_count[16];

    // 批量加载 + 排序 + 推送 循环
    for (batch = range_start; batch < range_end; batch += BATCH_SIZE) {
        // 1. 协作加载到 TAIL
        // 2. Bitonic sort TAIL
        // 3. 将 TAIL 最小元素推送到 MID
        // 4. 将 MID 最小元素推送到 HEAD
        // 5. HEAD 满时 blend_one()
    }

    // 排空 MID → HEAD → blend
}
```

#### `src/cpu/rasterizer_cpu.cpp` — 添加 CPU 分层排序参考实现

CPU 版本可以用更简单的方式实现: 对每个像素维护一个小的排序窗口 (k-buffer)，而不是三级分层。

```cpp
if (pre.eval_3D) {
    // Per-pixel k-buffer sorting
    constexpr int K = 16;
    struct Entry { float depth; uint32_t idx; float alpha; };
    Entry buffer[K];
    int buf_count = 0;

    for (uint32_t j = range_start; j < range_end; j++) {
        // 计算 depth 和 alpha
        // 插入排序到 buffer
        // buffer 满时弹出最前面的混合
    }
}
```

### 性能注意

- Maleoon GPU 920 的 sub_group size 需要运行时查询 (`CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`)
- shared memory (local memory) 用量增加：每个 WG 额外 ~16 sub-tiles × (TAIL+MID) ≈ 16×72×8 ≈ 9KB
- 需确认 Maleoon GPU 的 local memory 上限（通常 32-64KB）
- 如果 sub_group 扩展不可用，可退化为纯 local memory 实现（性能略低）

---

## 方案 4: 视空间包围盒 (View-Space Bounding)

### 问题

当前使用 `computeAABBScreen()` (Hahlbohm 方法) 计算屏幕空间 AABB。当 Gaussian 延伸到相机近平面后方时，该方法返回 false，导致 Gaussian 被错误丢弃，产生 popping。

### 算法

在视空间中用角度参数化切平面，求解二次相切条件:

```
θ₁,₂ = arctan((s₁,₃ ± √(s₁,₃² - s₁,₁s₃,₃)) / s₃,₃)
bounds = W/2 + focal × tan(θ₁,₂)
```

### 当前状态

`computeAABBView()` 已实现 (`src/math_utils.cpp:398-472`)，但由于数值稳定性问题未启用。具体问题:

1. `atan2` 多值性处理 — while 循环旋转角度到正确范围
2. 判别式为负时（椭球体与轴相交）— 回退到全屏范围
3. 边界情况（θ_mu ≈ 0 或 ±π）时角度规范化可能出错

### 修复方向

1. 检查 `normalizeAngle()` 的边界处理
2. 对比 CUDA 版本 `compute_aabb_view` 逐行验证
3. 添加数值保护: `theta_mu` 接近 0 时的特殊处理
4. 作为 `computeAABBScreen` 的 fallback: 先尝试 screen-space，失败时切换到 view-space

### 需要修改的文件

- `src/math_utils.cpp:398-472` — 修复 `computeAABBView` 数值问题
- `src/cpu/preprocessor_cpu.cpp` — 添加 fallback 逻辑:
  ```cpp
  if (!computeAABBScreen(gauss2screen, cutoff, mean2D_aabb, extent_aabb)) {
      // Screen-space AABB 失败，尝试 view-space
      if (!computeAABBView(gauss2view, p_view, focal_x, focal_y, W, H, cutoff, mean2D_aabb, extent_aabb))
          continue;
  }
  ```
- GPU kernel 同步修改

---

## 实现顺序建议

```
阶段1: 逐Tile 3D裁剪 (方案1)
  ├── CPU tile_binner + preprocessor 修改
  ├── GPU scatter kernel 修改
  ├── 验证: tile-pair 数量减少, 渲染结果不变
  └── 预期收益: GPU 渲染耗时减少 ~40%

阶段2: Per-Tile Depth Key (方案2)
  ├── 新增 cov3D_inv 计算和存储
  ├── CPU/GPU tile_binner 修改 depth 计算
  ├── 验证: 排序更一致, tile 边界伪影减轻
  └── 预期收益: 视觉质量提升 (配合阶段3效果更好)

阶段3: 分层排序 (方案3) [可选]
  ├── 重写 GPU 光栅化 kernel
  ├── CPU 添加 k-buffer 参考实现
  ├── 验证: popping 伪影消除
  └── 预期收益: 视觉质量显著提升

阶段4: 视空间包围盒 (方案4) [可选]
  ├── 修复 computeAABBView 数值问题
  ├── 添加 screen/view AABB fallback
  └── 预期收益: 近距离场景稳定性
```

## 验证方法

每个方案实现后:
1. 渲染 `basketball.ply` camera ID=0，与当前输出对比 (max diff < 5 可接受)
2. 检查 tile-pair 数量变化（方案1应减少 ~40%）
3. 检查 tile 边界处像素值连续性（方案2应改善）
4. 多角度渲染检查 popping（方案3应消除）
