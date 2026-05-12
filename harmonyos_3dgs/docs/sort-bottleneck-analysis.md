# Sort 瓶颈分析与逐 Kernel Profiling

## 2026-05-09 Vulkan/Fuchsia packed-keyval update

The current Vulkan path has an opt-in Fuchsia radix-sort carrier path controlled by `GS3D_USE_FUCHSIA_SORT=1`. The safe merge-training-master integration keeps this path opt-in and conservative: Fuchsia sort can publish sorted packed keyvals and GPU-generated tile ranges, but host `keys_sorted`/`values_sorted` mirrors are still reconstructed for the existing rasterizer/backward consumers.

Measured on NVIDIA Tegra Thor, basketball cam0, `runs/full_cap500000_step30000_vk_cuda3373529_literal_match/vk_trained_cap500000_step30000.ply`, `--eval_3d 1 --proper_ewa 1 --parity_mode 0`, 3 runs each:

| Mode | Render mean | Sorter mean | Wall mean | Output vs fallback |
|------|-------------|-------------|-----------|--------------------|
| `GS3D_USE_FUCHSIA_SORT=0` | 1548.5 ms | 654.9 ms | 2.46 s | reference |
| `GS3D_USE_FUCHSIA_SORT=1` | 1121.6 ms | 98.2 ms | 1.88 s | deterministic, but not bit-identical: 41.08 dB, RMSE 0.00883, max abs 0.7987 |

Conclusion: Fuchsia sort removes most of the sort-stage time for this scene, but because full-scene output still differs from the fallback path, it is not a default parity-safe replacement yet. Further GPU-only rasterizer/backward handoff should first resolve or explicitly characterize the ordering difference.

## 测试条件

- 模型: basketball.ply (400K Gaussians, 270K valid, 6.3M tile-pairs)
- 相机: cameras.json id 0 (720×960, 2700 tiles)
- 设备: Maleoon GPU 920 (6 CUs, 512 max WG, 32KB local mem)
- Antialiasing: OFF
- 总渲染时间: **961 ms**

---

## 全管线逐 Kernel Profiling

### 前向渲染调用的全部 Kernel

| # | Kernel | 文件 | GPU 耗时 | 阶段 | 说明 |
|---|--------|------|---------|------|------|
| 1 | `preprocess` | preprocess.cl | **7.7 ms** | Preprocess | 投影+SH+cov2D, 400K work-items |
| 2 | `scatter` (binner) | scatter.cl | **16.3 ms** | TileBinner | 键值对生成, 400K work-items |
| 3 | `count_per_tile` | counting_sort.cl | **~8 ms** | Sort Phase1 | atomic 计数, 6.3M work-items |
| 4 | `prefix_sum_tiles` | counting_sort.cl | **~1 ms** | Sort Phase2 | 单线程 scan, 2700 tiles |
| 5 | `scatter_by_tile` | counting_sort.cl | **~190 ms** | Sort Phase3 | atomic scatter, 6.3M work-items |
| 6 | `identify_tile_ranges` | radix_sort.cl | **~6 ms** | Sort Phase4 | tile 边界识别, 6.3M work-items |
| 7 | `bitonic_sort_tiles` | tile_sort.cl | **~320 ms** | Sort Phase5a | local-memory bitonic, ≤4096/tile |
| 8 | `merge_sorted_runs` | tile_sort.cl | **~25 ms** | Sort Phase5b | 大 tile 合并, >4096/tile |
| 9 | `rasterize` | rasterize.cl | **297.9 ms** | Rasterizer | tile alpha blending, 2700 WGs |

**实际调用: 9 个 kernel，全部运行在 GPU 上。无 CPU 回退。**

### 各阶段耗时汇总

| 阶段 | 总耗时 | GPU Kernel | 非 Kernel 开销 | 占比 |
|------|--------|-----------|---------------|------|
| Preprocess | 35.6 ms | 7.7 ms | 27.9 ms (readback 3.2MB) | 3.7% |
| TileBinner | 42.5 ms | 16.3 ms | 26.2 ms (CPU prefix sum + upload) | 4.4% |
| **Sort** | **572.6 ms** | **~570 ms** | **~2 ms** | **59.6%** |
| Rasterizer | 310.2 ms | 297.9 ms | 12.3 ms (readback 8.3MB) | 32.3% |
| **Total** | **961 ms** | **~892 ms** | **~69 ms** | |

---

## Sort 内部瓶颈分析

```
Sort 总耗时: 572.6 ms
├── Phase1 count_per_tile:    11.0 ms  ( 1.9%)  ← 快
├── Phase2 prefix_sum:         1.4 ms  ( 0.2%)  ← 快
├── Phase3 scatter_by_tile:  201.7 ms  (35.2%)  ← 瓶颈 #2
├── Phase4 tile_ranges:        8.8 ms  ( 1.5%)  ← 快
└── Phase5 in-tile sort:     347.4 ms  (60.7%)  ← 瓶颈 #1
    ├── bitonic_sort_tiles:  ~320 ms   (tiles ≤ 4096 元素)
    └── merge_sorted_runs:    ~25 ms   (tiles > 4096 元素)
```

### 瓶颈 #1: In-tile Bitonic Sort (347 ms, 60.7%)

**问题**: 2700 tiles 中每个 tile 平均 2330 个 Gaussian，需要 bitonic sort。

- Bitonic sort 复杂度: O(N × log²N) 比较
- 2330 元素 → padded 到 4096 → log²(4096) = 144 步
- 每步 2048 次比较 × 256 线程 = 8 次串行比较/线程
- 144 步 × 8 = 1152 串行比较/线程/tile
- 每步需要 `barrier(CLK_LOCAL_MEM_FENCE)`
- **144 次 barrier × 2700 tiles → 388,800 次 barrier 调用**

**为什么慢**: Maleoon 920 的 barrier 开销较高，加上 local memory bank conflict。

### 瓶颈 #2: Scatter by Tile (202 ms, 35.2%)

**问题**: 6.3M 次 atomic 全局内存写入。

- 每个元素做 `atomic_add(&tile_counters[tile_id], 1)` 获取写入位置
- 然后写 `keys_sorted[pos]` 和 `values_sorted[pos]` — **随机全局写**
- 6.3M × 12 bytes (8B key + 4B value) = 75MB 随机写
- Maleoon 920 全局内存带宽约 30 GB/s，但随机写带宽仅 ~2-5 GB/s
- 75MB / 3 GB/s ≈ 25ms 理论下界，实际 202ms（atomic 争用 + cache miss）

---

## 下一步优化方向

### 优化 1: 减少 In-tile Sort 的 Barrier 开销

**方案 A: 动态 Padded Size**
- 当前所有 tile pad 到 4096（log²=144 步）
- 多数 tile 只有 500-2000 元素，pad 到最近的 power-of-2 减少步数
- 1024 元素只需 log²(1024)=100 步，2048 需 121 步

**方案 B: 用 Odd-Even Sort 替代 Bitonic（小 tile）**
- 对 ≤256 元素的 tile，odd-even sort 256 passes 比 bitonic 更高效（无 padding）

### 优化 2: 减少 Scatter 的随机写

**方案 A: 分桶 Scatter**
- 将 scatter 分为多轮，每轮只处理部分 tile，提高 cache 局部性

**方案 B: 预排序 + 顺序写**
- 先按 tile_id 分桶计数（已有），然后每个 WG 只处理自己桶内的元素

### 优化 3: 消除剩余 CPU 开销

| 操作 | 当前耗时 | 优化方案 | 预计节省 |
|------|---------|---------|---------|
| Preprocess readback (radii+tiles_touched) | 28 ms | GPU prefix sum 修复 | -28 ms |
| TileBinner CPU prefix sum + upload | 26 ms | GPU prefix sum 修复 | -26 ms |
| Rasterizer readback (output image) | 12 ms | CL-GL interop | -12 ms |
| **总计** | **66 ms** | | **-66 ms** |

### 优化 4: Rasterizer Kernel (298 ms)

- 2700 tiles × 9 batches × barrier = 主要开销
- `values_sorted[j]` 间接索引导致全局内存随机读
- 可优化: 增大 batch size、减少 barrier、coalesced 内存访问

### 预期效果

| 优化 | 预计节省 | 新耗时 |
|------|---------|--------|
| 当前基线 | — | 961 ms |
| In-tile sort 优化 | -150 ms | ~811 ms |
| Scatter 优化 | -100 ms | ~711 ms |
| 消除 CPU 开销 | -66 ms | ~645 ms |
| Rasterizer 优化 | -100 ms | ~545 ms |
| **全部优化** | **~416 ms** | **~545 ms** |
