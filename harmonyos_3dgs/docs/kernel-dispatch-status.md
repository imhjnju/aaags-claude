# OpenCL Kernel 调用状态与耗时报告

## 测试条件

- 模型: basketball.ply (400,000 Gaussians, SH degree 3)
- 相机: cameras.json id 0 (720×960)
- 设备: Maleoon GPU 920 (6 CUs, 512 max WG, 32KB local mem)
- Antialiasing: OFF
- 总渲染时间: **1263 ms**

---

## 全部 9 个 OpenCL Kernel 状态

| # | Kernel 名称 | 文件 | 编译 | 前向渲染调用 | GPU 耗时 | 备注 |
|---|------------|------|------|-------------|---------|------|
| 1 | `preprocess` | preprocess.cl | ✅ | **✅ 调用** | **7.3 ms** | 每 Gaussian 一个 work-item |
| 2 | `scatter` | scatter.cl | ✅ | **✅ 调用** | **14.4 ms** | 生成 tile-Gaussian 键值对 |
| 3 | `scan_blocks` | prefix_sum.cl | ✅ | ❌ 未调用 | — | GPU prefix sum 有 bug，已回退 CPU |
| 4 | `add_block_sums` | prefix_sum.cl | ✅ | ❌ 未调用 | — | 同上 |
| 5 | `radix_histogram` | radix_sort.cl | ✅ | ❌ 未调用 | — | GPU radix sort O(N²) scatter 太慢，已回退 CPU sort |
| 6 | `radix_scatter` | radix_sort.cl | ✅ | ❌ 未调用 | — | 同上 |
| 7 | `identify_tile_ranges` | radix_sort.cl | ✅ | ❌ 未调用 | — | CPU sort 路径中在 CPU 侧计算 tile ranges |
| 8 | `sort_within_tiles` | radix_sort.cl | ✅ | ❌ 未调用 | — | 4-pass 排序方案中使用，当前未启用 |
| 9 | `rasterize` | rasterize.cl | ✅ | **✅ 调用** | **313.5 ms** | 每 tile 一个 work-group (256 threads) |

**实际调用的 GPU kernel: 3 个 / 9 个**

---

## 渲染管线各阶段耗时分解

```
┌─────────────────────────────────────────────────────────────┐
│                    总耗时: 1263 ms                            │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ Preprocess│ TileBinner│  Sorter  │Rasterizer│                 │
│  39.6 ms  │  50.9 ms  │ 846.1 ms │ 326.6 ms │                 │
│   3.1%    │   4.0%    │  67.0%   │  25.9%   │                 │
├──────────┼──────────┼──────────┼──────────┤                 │
│          │          │          │          │                 │
│ GPU:7.3ms│ GPU:14.4 │ CPU:全部  │GPU:313.5 │                 │
│ CPU:32.3 │ CPU:36.5 │ 846.1ms  │ CPU:13.1 │                 │
│ (readback│ (prefix  │ (readback│ (readback│                 │
│  +alloc) │  sum+    │  +sort+  │  +alloc) │                 │
│          │  upload) │  upload) │          │                 │
└──────────┴──────────┴──────────┴──────────┘
```

### 详细分解

#### 1. Preprocess (39.6 ms, 3.1%)

| 操作 | 耗时 | 类型 |
|------|------|------|
| `preprocess` kernel | **7.3 ms** | GPU |
| readback radii (400K × 4B = 1.6MB) | ~5 ms | GPU→CPU |
| readback tiles_touched (400K × 4B = 1.6MB) | ~5 ms | GPU→CPU |
| 模型上传 (首帧) / kernel arg 设置 | ~22 ms | CPU+GPU |

#### 2. TileBinner (50.9 ms, 4.0%)

| 操作 | 耗时 | 类型 |
|------|------|------|
| CPU prefix sum (400K 元素) | ~1 ms | CPU |
| upload offsets (400K × 4B = 1.6MB) | ~5 ms | CPU→GPU |
| `scatter` kernel | **14.4 ms** | GPU |
| buffer 分配 (6.3M pairs) | ~30 ms | GPU |

#### 3. Sorter (846.1 ms, 67.0%) ← 主要瓶颈

| 操作 | 耗时 | 类型 |
|------|------|------|
| readback keys_unsorted (6.3M × 8B = 50MB) | ~200 ms | GPU→CPU |
| readback values_unsorted (6.3M × 4B = 25MB) | ~100 ms | GPU→CPU |
| CPU std::sort (6.3M pairs) | ~200 ms | CPU |
| upload keys_sorted (50MB) | ~200 ms | CPU→GPU |
| upload values_sorted (25MB) | ~100 ms | CPU→GPU |
| upload tile_ranges + CPU 计算 | ~46 ms | CPU+GPU |

**Sort 全程无 GPU kernel 调用，100% CPU + 数据传输。**

#### 4. Rasterizer (326.6 ms, 25.9%)

| 操作 | 耗时 | 类型 |
|------|------|------|
| `rasterize` kernel | **313.5 ms** | GPU |
| readback output image (720×960×3×4B = 8.3MB) | ~13 ms | GPU→CPU |

---

## GPU Kernel 调用详情

### preprocess (7.3 ms)

```
Global size: 400,128 (rounded up from 400,000)
Local size:  256
Work-groups: 1,563
每个 work-item 处理: 1 Gaussian
计算: frustum cull → NDC 投影 → cov3D → cov2D → AA → conic → radius → SH eval → 存储
```

### scatter (14.4 ms)

```
Global size: 400,128
Local size:  256
Work-groups: 1,563
每个 work-item 处理: 1 Gaussian
计算: 对有效 Gaussian 的每个覆盖 tile，生成 (tile_id<<32|depth_bits, gauss_idx) 键值对
输出: 6,289,717 对到 keys_unsorted/values_unsorted buffer
```

### rasterize (313.5 ms)

```
Global size: 691,200 (2,700 tiles × 256 threads/tile)
Local size:  256 (16×16 pixels per tile)
Work-groups: 2,700
每个 work-group 处理: 1 tile (平均 ~2,330 Gaussians)
计算: batch 加载 Gaussian 到 local memory → 逐像素 alpha blending → 写输出
Batches/tile: ~9 (2330/256)
Barriers/tile: ~18 (2 per batch)
```

---

## 未调用 Kernel 的原因

| Kernel | 未调用原因 | 问题详情 |
|--------|-----------|---------|
| `scan_blocks` | GPU prefix sum 算错 total_pairs | 3-level scan 未实现，2-level 对 400K 元素溢出 |
| `add_block_sums` | 同上 | — |
| `radix_histogram` | GPU radix sort scatter O(N²) 比 CPU sort 更慢 | 55ms/pass × 8 = 440ms GPU compute + 300ms sync |
| `radix_scatter` | 同上 | rank 计算 O(wg_size²) 是瓶颈 |
| `identify_tile_ranges` | CPU sort 路径中在 CPU 侧做 | — |
| `sort_within_tiles` | 4-pass 方案质量不足 (PSNR 16 dB) | 64 passes 限制不够，2330 passes 导致 GPU watchdog 超时 |

---

## 性能优化路线图

### 当前: 3 个 GPU kernel + CPU sort (1263 ms)

```
GPU kernel 总耗时: 335 ms (26.5%)
CPU + 传输耗时:    928 ms (73.5%) ← 主要是 sort 的 150MB 数据传输
```

### 目标: 全 GPU pipeline (预计 ~50-100 ms)

要实现 60 FPS (16.7 ms)，需要：

1. **消除 Sort 的 150MB 数据传输** (省 ~600 ms)
   - 需要正确高效的 GPU 排序算法
   - 方案: GPU counting sort (利用 tile_id 有限范围)

2. **消除 Preprocess readback** (省 ~10 ms)
   - GPU prefix sum 修复后，无需读回 radii/tiles_touched

3. **Rasterizer kernel 优化** (313 ms → ~10 ms)
   - 当前 2700 tiles × 9 batches × 256 threads = 有限并行度
   - 需要: 更大 batch size、warp-level 优化、减少 barrier

4. **消除图像 readback** (省 ~13 ms)
   - CL-GL interop 直接显示，无需读回
