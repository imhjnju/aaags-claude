# GPU 渲染优化报告 — Maleoon GPU 920

## 目标

将 3DGS 渲染从 1515ms 优化到 16.7ms（60 FPS），在华为鸿蒙手机 Maleoon GPU 920 上实现实时渲染。

## 设备信息

| 参数 | 值 |
|------|-----|
| GPU | Maleoon 920 |
| Compute Units | 6 |
| Max Work-Group Size | 512 |
| Local Memory | 32 KB |
| Max Buffer Alloc | 1 GB |
| OpenCL Version | 1.2+ |

## 测试模型

- **basketball.ply**: 400,000 Gaussians, SH degree 3
- **Camera**: cameras.json id 0 (720×960)
- **有效 Gaussian**: 270,074 (67.5%)
- **Tile-Gaussian pairs**: 6,289,717

---

## 优化历程

### Phase 0: CPU 基线

| 平台 | 渲染时间 |
|------|---------|
| PC x86_64 | 16,365 ms |
| ARM64 手机 CPU | 13,787 ms |

### Phase 1: 初始 GPU 移植 (1629ms, **8.5x 加速**)

将四个管线阶段移植到 OpenCL：
- Preprocessor: GPU kernel (投影 + SH + 协方差)
- TileBinner: CPU prefix sum + GPU scatter
- Sorter: CPU std::sort fallback
- Rasterizer: GPU tile-based alpha blending kernel

**遇到的问题：**

#### Bug 1: GPU 渲染全黑
- **现象**: 渲染输出完全黑色，但 preprocessor 报告 270K 有效 Gaussian
- **根因**: `preprocess.cl` 的 Step 11 (Store outputs) 遗漏了 `out_rgb` 的写入。SH 颜色在 `float rgb[3]` 局部变量中计算完成，但未存储到 `__global float* out_rgb` buffer
- **修复**: 添加 `out_rgb[i*3+ch] = rgb[ch]`
- **教训**: OpenCL kernel 中局部变量不会自动同步到全局内存，每个输出必须显式写入

#### Bug 2: CL_MEM_WRITE_ONLY buffer 跨 kernel 读取
- **现象**: Scatter kernel 和 rasterize kernel 读取 preprocessor 输出时数据为零
- **根因**: Preprocessor 的输出 buffer 创建时使用 `CL_MEM_WRITE_ONLY`，后续 kernel 作为输入读取属于未定义行为
- **修复**: 改为 `CL_MEM_READ_WRITE`
- **教训**: OpenCL buffer flag 是给驱动的优化提示，在多 kernel pipeline 中应始终用 `CL_MEM_READ_WRITE`

### Phase 2: 消除 CPU readback (1515ms, **9.1x 加速**)

#### 优化 2a: 减少 Preprocessor readback
- **优化前**: 读回全部 7 个数组（~20MB for 400K Gaussians）
- **优化后**: 只读回 `radii` 和 `tiles_touched`（~3.2MB）
- **效果**: Preprocessor 38ms → 35ms

#### 优化 2b: TileBinner 零 readback
- **优化前**: 读回 unsorted keys/values（~96MB for 6.3M pairs）给 CPU sort
- **优化后**: Scatter kernel 直接写 GPU buffer，rasterizer 直接从 GPU 读
- **效果**: TileBinner 215ms → 29ms（**7.4x**）

#### 优化 2c: Rasterizer 直接读 device_data
- **优化前**: Rasterizer 从 CPU 数组读数据
- **优化后**: 直接从 PreprocessDeviceBuffers 和 BinningDeviceBuffers 读
- **效果**: Rasterizer 589ms → 247ms（**2.4x**）

**遇到的问题：**

#### Bug 3: GPU Blelloch prefix sum 计算错误
- **现象**: GPU prefix sum 输出 total_pairs=2,517,653（正确值 6,289,717）
- **根因**: 400K 元素 / 512 per block = 782 blocks。扫描 782 block sums 需要 2 个子 block，即需要 3 级 scan。代码只实现了 2 级
- **修复**: 回退 CPU prefix sum（400K 元素串行扫描 ~1ms，不是瓶颈）
- **教训**: 多级 parallel scan 的边界条件容易出错，对于非瓶颈路径优先保证正确性

#### Bug 4: GPU bitonic sort 比 CPU 更慢
- **现象**: GPU bitonic sort 1038ms vs CPU std::sort 750ms
- **根因**: 6.3M 元素的 bitonic sort 需要 log2(8M) × (log2(8M)+1)/2 ≈ 276 次 kernel launch。每次 launch 有 ~0.1ms 的 dispatch overhead，276 × 0.1ms = 27.6ms 仅是 launch 开销。加上每次 kernel 的实际计算，总时间超过 CPU
- **修复**: 回退 CPU sort（后续用 GPU radix sort 替换）
- **教训**: GPU 算法的 kernel launch 次数是关键性能指标。O(N·log²N) 的 bitonic sort 虽然并行度高，但 launch 次数太多。O(N·k) 的 radix sort（k=8 passes）只需 ~24 次 launch，更适合 GPU

---

## 当前 Profiling 数据

```
Preprocess:  35.0 ms  ( 2.3%)  — GPU kernel + readback(radii, tiles_touched)
TileBinner:  29.0 ms  ( 1.9%)  — CPU prefix sum + upload offsets + GPU scatter
Sorter:     873.0 ms  (57.6%)  — 75MB GPU→CPU + std::sort + 75MB CPU→GPU  ← 主瓶颈
Rasterizer: 579.0 ms  (38.2%)  — GPU rasterize kernel + 14MB image readback
─────────────────────────────────
Total:     1515.0 ms
```

### 瓶颈分析

| 阶段 | 计算耗时 | 传输耗时 | 优化空间 |
|------|---------|---------|---------|
| Preprocess | ~25ms (kernel) | ~10ms (3.2MB read) | 中 |
| TileBinner | ~5ms (scatter) | ~24ms (1.6MB write + kernel) | 小 |
| **Sorter** | **~200ms (sort)** | **~670ms (150MB 双向)** | **极大** |
| **Rasterizer** | **~560ms (kernel)** | **~19ms (14MB read)** | **大** |

---

## 下一步优化计划

### P0: GPU Radix Sort (消除 873ms → 目标 ~15ms)

**算法**: 8-bit radix sort, 8 passes for 64-bit keys

每个 pass 包含 3 个 kernel：
1. **Histogram**: 每个 work-group 统计 256 个 bucket 的计数
2. **Prefix Sum**: 对 histogram 做全局 prefix sum 得到 scatter 偏移
3. **Scatter**: 每个元素根据当前 digit 和偏移写入新位置

总计 8 × 3 = 24 次 kernel launch（vs bitonic 的 276 次）。

**预期性能**:
- 每个 pass 处理 6.3M 元素: ~1-2ms
- 8 passes: ~10-15ms
- 消除 150MB GPU↔CPU 传输: 省 ~670ms

**实现要点**:
```
Pass i (digit = bits [i*8, i*8+7]):
  1. histogram_kernel: 每个 WG 对 N/num_WG 元素统计 256-bucket histogram
     输出: histogram[num_WG × 256]
  2. prefix_sum_kernel: 对 histogram 做 exclusive scan
     输出: offsets[num_WG × 256]
  3. scatter_kernel: 每个元素根据 digit 和 offset 写入输出位置
     交替使用两组 buffer (ping-pong)
```

### P0: Rasterizer Kernel 优化 (579ms → 目标 ~10ms)

当前 rasterizer 的问题分析：
- 每个 tile 的 work-group (256 threads) 处理平均 ~1000 Gaussians
- 每批次 (batch) 加载 256 个 Gaussian 到 local memory
- ~4 批次/tile × 2700 tiles = ~10800 kernel work-items

**优化方向**:

1. **增大 batch size**: Maleoon 920 有 32KB local memory。当前每批次用 256 × (8+12+12+4) = 9KB。可以增大到 512 或更大
2. **使用 `native_exp`**: 替代精确的 `exp()` 函数，精度足够（alpha blending 容忍误差）
3. **减少 barrier 调用**: 合并 load + compute 循环，减少同步开销
4. **Warp-level 优化**: 利用 Maleoon 的 warp size 减少不必要的同步
5. **Early exit 优化**: 如果整个 work-group 的 T 都 < 0.0001，提前退出

### P1: 消除剩余 readback

| Readback | 大小 | 可消除方式 |
|----------|------|-----------|
| radii + tiles_touched | 3.2MB | GPU prefix sum 修复后不需读回 |
| 最终图像 | 14MB | 如果实时显示，可用 CL-GL interop 直接显示 |

### P2: Preprocess Kernel 优化

- SH 评估用 `native_sqrt`, `native_rsqrt`
- 减少 register 压力：SH degree 3 使用 48 个 SH 系数，可分批加载
- 协方差投影中的矩阵运算可用 `mad()` (multiply-add) 指令

### P3: Pipeline 级优化

- **异步执行**: 利用 OpenCL command queue 的异步特性，overlap 传输与计算
- **多 queue**: 一个 compute queue + 一个 transfer queue，实现计算/传输并行
- **Buffer 复用**: 避免帧间重复分配，完全池化

---

## 性能目标路线图

| 阶段 | 预期总耗时 | 加速比 vs CPU |
|------|-----------|-------------|
| 当前 | 1515 ms | 9x |
| +GPU Radix Sort | ~650 ms | 21x |
| +Rasterizer 优化 | ~100 ms | 138x |
| +全 GPU pipeline | ~30 ms | 460x |
| +异步/overlap | ~16 ms | **860x (60 FPS)** |
