# StopThePop 无法集成的具体原因分析

## Maleoon 920 GPU 实际支持的 OpenCL 扩展

```
cl_khr_global_int32_base_atomics
cl_khr_global_int32_extended_atomics
cl_khr_local_int32_base_atomics
cl_khr_local_int32_extended_atomics
cl_khr_byte_addressable_store
cl_khr_3d_image_writes
cl_khr_int64_base_atomics
cl_khr_int64_extended_atomics
cl_khr_fp16
cl_khr_icd
cl_khr_egl_image
cl_khr_image2d_from_buffer
cl_khr_depth_images
cl_khr_subgroups                    ← 支持基础 subgroup
cl_khr_il_program
cl_khr_priority_hints
cl_khr_throttle_hints
cl_khr_external_memory
cl_khr_external_memory_dma_buf
cles_khr_int64
```

**缺失的关键扩展：**
- ❌ `cl_khr_subgroup_shuffle` — 无 `sub_group_shuffle()`
- ❌ `cl_khr_subgroup_shuffle_relative` — 无 `sub_group_shuffle_up/down()`
- ❌ `cl_khr_subgroup_ballot` — 无 `sub_group_ballot()`
- ❌ `cl_khr_subgroup_non_uniform_vote` — 无 `sub_group_any/all()`

> 注：我们的 RadixSort 使用了 `sub_group_shuffle_up`，能编译运行是因为 Maleoon 920 的 OpenCL 驱动虽然没有声明 `cl_khr_subgroup_shuffle_relative`，但实际在 `-cl-std=CL3.0` 模式下支持该指令。这是驱动的非标准行为，不可靠地泛化到其他 subgroup 操作。

---

## StopThePop 使用的 CUDA 原语 vs Maleoon 920 支持

### 1. `__ballot_sync(mask, predicate)` — 🔴 不可移植

**用途**：投票——将 warp 内所有线程的 predicate 结果收集为一个 bitmask。

**StopThePop 中的使用场景**：
- `stopthepop_common.cuh:211` — 判断哪些线程有有效 Gaussian
- `stopthepop_common.cuh:278` — 活跃线程掩码（load balancing）
- `stopthepop_common.cuh:389` — 协作加载中的有效性检查
- `stopthepop_common.cuh:577` — 确定哪些线程需要参与归并排序
- `hierarchical_render.cuh:918,939` — 层级排序中的投票

**OpenCL 状态**：需要 `cl_khr_subgroup_ballot`，**Maleoon 920 不支持**。

**替代方案**：用 local memory 模拟（每线程写 0/1 到 shared array → barrier → 读取）。开销：1 次 barrier + N 次 local memory 访问。对 StopThePop 的高频调用（每个 Gaussian 一次），这会成为严重瓶颈。

---

### 2. `__shfl_sync(mask, value, lane)` — 🔴 不可移植

**用途**：Shuffle——直接读取 warp 内指定 lane 的寄存器值，无需经过 shared memory。

**StopThePop 中的使用场景**：
- `stopthepop_common.cuh:220-243` — `mergeSortInto()` 核心函数，将新元素插入已排序序列
- `stopthepop_common.cuh:586-590` — `shflRankingLocal()` 用 shuffle 实现线程间排名

**OpenCL 状态**：需要 `cl_khr_subgroup_shuffle`，**Maleoon 920 不支持**。

**替代方案**：用 local memory 模拟（写入 → barrier → 读取）。但 `mergeSortInto` 在一次调用中做多次 shuffle（循环内），每次需要一个 barrier。这会将 shuffle 的 1-cycle 延迟变为 ~100 cycle 的 barrier，性能退化 ~100x。

---

### 3. `__fns(mask, base, offset)` — 🔴 不可移植

**用途**：Find N-th Set bit——在 bitmask 中找到第 N 个置位的 bit 位置。

**StopThePop 中的使用场景**：
- `stopthepop_common.cuh:218` — 在 `mergeSortInto` 中确定插入位置
- `stopthepop_common.cuh:584` — 在 ranking 中确定线程排名

**OpenCL 状态**：无任何扩展提供此功能。

**替代方案**：手动位操作循环（`popcount` + mask）实现。~10-20 条指令 vs CUDA 原生 1 条。可行但增加寄存器压力。

---

### 4. `__syncthreads_and(predicate)` — 🟡 可模拟但有开销

**用途**：Block-level barrier + AND reduction。所有线程同步，并返回所有 predicate 的 AND。

**StopThePop 中的使用场景**：
- `resorted_render.cuh:125,398` — 检查所有线程是否都已完成（提前退出优化）

**OpenCL 状态**：无直接等价。

**替代方案**：`barrier()` + local memory reduction。2 次 barrier + 1 次 atomic 操作。可行，开销可接受。

---

### 5. CUB `BlockRadixSort` — 🔴 无等价库

**用途**：Block 级别的高效基数排序，用于对 tile 内所有 Gaussian 做全排序。

**StopThePop 中的使用场景**：
- `resorted_render.cuh:503` — PER_PIXEL_FULL 模式的核心排序

**OpenCL 状态**：无 CUB 等价库。需要从零实现 block-level radix sort（~300-500 行代码）。

**替代方案**：可用 local memory bitonic sort 代替（已有实现）。但 CUB radix sort 对大数据集（>256 元素）效率更高。

---

### 6. Cooperative Groups `tiled_partition<N>()` — 🟡 部分可模拟

**用途**：将 warp 拆分为更小的协作组（如 16 线程、4 线程），每个组内部做独立同步。

**StopThePop 中的使用场景**：
- `hierarchical_render.cuh:299-301` — 创建 warp/halfwarp/4-thread 分区
- 整个层级排序的核心：HEAD (4 线程) / MID (16 线程) / TAIL (32 线程) 三级缓冲

**OpenCL 状态**：`cl_khr_subgroups` 提供基础 subgroup 操作，但不支持递归分区（tiled_partition）。

**替代方案**：手动计算 sub-group 内的 lane 索引，但无法做 sub-group 级别的 barrier（OpenCL barrier 是 workgroup 级别的）。这是核心限制。

---

## 总结

| CUDA 原语 | 使用频率 | OpenCL 可移植性 | 性能影响 |
|-----------|---------|-----------------|---------|
| `__ballot_sync` | 高（每 Gaussian 1次） | 🔴 不可用，需 local mem 模拟 | ~100x 退化 |
| `__shfl_sync` | 极高（排序核心） | 🔴 不可用，需 local mem 模拟 | ~100x 退化 |
| `__fns` | 高 | 🔴 不可用，需手动位操作 | ~10x 退化 |
| `__syncthreads_and` | 低 | 🟡 可模拟 | 2x 退化 |
| CUB BlockRadixSort | PER_PIXEL_FULL 专用 | 🔴 需重写 | ~2x 退化 |
| `tiled_partition` | 层级排序核心 | 🟡 部分可模拟 | 无子组 barrier |

### 根本阻碍

StopThePop 的核心算法是 **warp-level 协作排序**：

1. 32 个线程（1 个 warp）共同维护一个排序缓冲区
2. 通过 `__shfl_sync` 在线程间传递数据（0 延迟）
3. 通过 `__ballot_sync` 做集体决策（1 cycle）
4. 无需 shared memory 或 barrier（warp 内隐式同步）

在 OpenCL 上：
1. 没有 warp 级别隐式同步
2. `__shfl_sync` 替代为 local memory + barrier（~100x 延迟）
3. `__ballot_sync` 替代为 local memory + barrier（~100x 延迟）
4. 每次排序操作从 ~1 cycle 变成 ~100 cycle

**结论**：即使完成移植，性能退化会使 StopThePop 比当前的全局排序 + 直接混合更慢，失去使用意义。StopThePop 的高效性完全依赖 CUDA 的 warp-level 硬件原语，这些原语在 Maleoon 920 的 OpenCL 实现中不可用。
