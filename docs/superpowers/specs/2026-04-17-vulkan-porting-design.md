# AAA-Gaussians → Vulkan 完整移植设计

> **状态**：已通过 brainstorming 7 节设计确认。进入 writing-plans 阶段前的最终 spec。
> **日期**：2026-04-17
> **负责人**：harmonyos_3dgs 主代理

---

## Context

### 项目目标

将 AAA-Gaussians（Anti-Aliased 3D Gaussian Splatting）从 Python + CUDA 移植到 Vulkan Compute Shader + 图形管线。要求训练 10000 步后，第 10000 步渲染图像与原 Python+CUDA 版本收敛质量一致（PSNR / SSIM 指标无降级）。

### 关键约束

- **数据集**：`/home/robota/Downloads/basketball/`
- **初始点云**：`/home/robota/Downloads/basketball/sparse/0/points3D.ply`
- **训练步数**：10000 步
- **致密化**：步 5100, 5200, ..., 9900（`densify_from_iter=5000, densify_until_iter=10000, interval=100`；条件 `step > from && step < until && step%interval==0`；实际触发 49 次）
- **所有超参数与原 AAA-Gaussians 完全一致**，除致密化 schedule 因项目约束而偏离
- **Adam eps = 1e-15**（Python 值，非 PyTorch 默认 1e-8）
- **SP-2/3/4 固定配置硬错误**：`eval_3D=false`、`tile_size=16x16`、`SortMode=GLOBAL`、`training=true`

### 现有基础

- 137 个单元测试全部通过，构建清洁
- 抽象接口已定义：`Preprocessor` / `TileBinner` / `Sorter` / `Rasterizer`（`include/*.h`）
- Python baseline 环境可用：`/home/robota/miniconda3/envs/aaa-gs`
- 已有 Vulkan 迁移分析：`dev_notes/master_plan/vulkan_migration_plan.md`
- 已知陷阱文档：`PORTING_PITFALLS.md`

### 既有代码的定位（避免误读为验收基准）

- **CUDA（`AAA-Gaussians/`）**：**唯一 golden 来源**。所有数值验收对齐 CUDA 导出的 `.npy` artifact。
- **OpenCL 后端（`src/gpu/`，10 kernel）**：legacy 实现，**不作为** Vulkan 验收基准。仅在调试期可作**局部实现 sanity 参考**（例如检查某个计算的思路），不在验收链上。
- **CPU 参考（`src/cpu/*`）**：同上，legacy；**不作为** Vulkan 验收基准。仅在开发者需要单步调试时作为**实现 sanity 参考**。
- 本 spec 的所有 "golden"、"容差"、"对齐" 术语均指向 CUDA 导出，不指向 OpenCL 或 CPU。

### 核心方法论

1. **TDD 强制**：每个 shader 先写测试再实现
2. **Vulkan vs CUDA golden**（非 vs CPU）：所有验收对齐 CUDA 导出的 `.npy` 中间张量，而非 CPU 参考实现
3. **Pass-first 设计**：以 compute pass 和 buffer schema 为起点，类是适配细节
4. **双阈值容差**：`max abs error` + `max rel error` 同时检查
5. **独立 sanity checks**：排序不变量、正定性、有限差分 spot check，防止"参考和实现同时错但一致"盲点
6. **Adam 路径**：CPU Adam 仅作临时 bring-up 工具，**不进 SP-4 最终验收**；`adam_step.comp` compute shader 是 SP-4 正式产出

---

## 1. 整体架构与子项目分解

### 1.1 核心验收模型

```
Python 层（调度）        CUDA 扩展层（内部张量 hook）
  dump_tool.py     →    diff-gaussian-rasterization 修改
  step/cam/seed/         → GeometryState/BinningState/ImageState 解析
  output_dir                → 命名 tensor 返回
                           ↓
                     checkpoints/step{N}/cam{K}/*.npy + manifest.json
                           ↓ 磁盘
C++ 测试程序（ctest）
  1. npy_reader → CPU buffer（golden）
  2. Vulkan 算子执行 → GPU buffer
  3. GPU→CPU download
  4. 双阈值误差比对（abs + rel）
  5. Sanity checks（独立不变量验证）
```

### 1.2 双阈值容差标准

| 算子类型 | max abs error | max rel error |
|---------|---------------|---------------|
| 预处理（投影、SH） | 1e-5 | 1e-4 |
| 排序 key/value | 0（精确匹配）| — |
| tile ranges | 0 | — |
| 光栅化像素 | 1e-5 | 1e-4（≈ PSNR > 100dB）|
| 梯度 | 1e-4 | 1e-3 |
| Adam 矩、参数 | 1e-6 | 1e-4 |

**排序 tie-break**：以 CUDA（CUB DeviceRadixSort）实际输出为 golden，Vulkan 复现相同顺序。SP-0 前置任务验证 CUB determinism；若不稳定则降级为"tile 分组正确 + 渲染一致"。

### 1.3 Sanity Checks

- 排序不变量：`sorted keys[i] ≤ sorted keys[i+1]`
- 正定性：`conic 矩阵 det > 0`
- transmittance 单调性：`T[px]` 序列单调不增
- 有限差分 spot check：随机 5 高斯 × 各参数维度，数值微分验梯度

### 1.4 数据集两档

| 档位 | 内容 | 用途 |
|------|------|------|
| **Tiny fixtures** | 5~50 合成高斯，固定 seed | 逐算子开发（SP-1~SP-3）|
| **Basketball checkpoints** | 真实数据，ladder: 1/10/100/500/1000/2000 | 集成验收（SP-4~SP-6）|

### 1.5 子项目分解

| 编号 | 子项目 | 产出 | 依赖 |
|------|--------|------|------|
| **SP-0** | CUDA Golden Infrastructure | dump_tool.py + npy_reader.h + manifest + 全量 golden | 无 |
| **SP-1** | Vulkan 基础设施 | VulkanContext/Buffer/Pipeline + hello-world 测试 | SP-0 |
| **SP-2** | 前向管线 | preprocess/sort/rasterize shaders + 逐算子 golden | SP-1 |
| **SP-3** | 反向管线 | backward shaders + 梯度 golden + FD spot check | SP-2 |
| **SP-4** | 训练管线（全 GPU） | TrainerVulkan + adam_step.comp + DSSIM + MCMC + schedule | SP-3 |
| **SP-5** | 2000 步一致性验证 | Checkpoint ladder 对齐报告 | SP-4 |
| **SP-6** | 10000 步 Soak + 性能 | PSNR/SSIM 对比 + 性能报告 | SP-5 |

每个子项目独立走完：**spec → writing-plans → TDD 实现 → review**。

---

## 2. SP-0：CUDA Golden Infrastructure

### 2.1 两层结构

```
SP-0
├── Python 调度层       tools/dump_tool.py（step/cam/seed/output_dir 控制）
├── CUDA 扩展修改层     diff-gaussian-rasterization/
│                        ├── rasterizer_impl.cu (不动核心)
│                        ├── rasterize_points.cu (新增 materialize_dump 函数)
│                        └── ext.cpp (新增 Python 绑定)
└── C++ 验收基础设施    tests/golden/npy_reader.h / compare.h / manifest.h
```

### 2.2 侵入面最小化

| 方向 | 修改点 |
|------|--------|
| 前向 | `rasterize_points.cu` **新增** `materialize_dump(geomBuffer, binningBuffer, imgBuffer, P, R, num_tiles, H, W, requires_gauss2screen)` 函数，解析三个原始 buffer 返回命名 tensor dict |
| 后向 | `RasterizeGaussiansBackwardCUDA()` 加 `dump_mode: bool` 参数；`True` 时 return tuple 末尾追加 `dL_dconic`（[P,2,2]）|
| Python | 新加参数命名 `dump_mode`，与现有 `debug`（snapshot 用）**完全隔离** |

`Rasterizer::forward`、`Rasterizer::backward` 核心接口**不动**。

### 2.3 精确 Artifact Schema

**前向 tensor 清单**（来自源码实测）：

| tensor | 来源 | dtype | shape |
|--------|------|-------|-------|
| `preprocess_means2D` | `GeometryState.means2D` (float2*) | f32 | [P, 2] |
| `preprocess_depths` | `GeometryState.depths` | f32 | [P] |
| `preprocess_conic_opacity` | `GeometryState.conic_opacity` (float4*) | f32 | [P, 4] packed {a,b,c,opacity} |
| `preprocess_rgb` | `GeometryState.rgb` | f32 | [P, 3] |
| `preprocess_radii` | `radii` (前向返回) | i32 | [P] |
| `preprocess_tiles_touched` | `GeometryState.tiles_touched` | u32 | [P] |
| `preprocess_point_offsets` | `GeometryState.point_offsets` | u32 | [P] |
| `preprocess_gauss2screen` | `GeometryState.gauss2screen` | f32 | [P, 16]（eval_3D=true 才有）|
| `sort_keys_unsorted` | `BinningState.point_list_keys_unsorted` | u64 | [R] |
| `sort_keys_sorted` | `BinningState.point_list_keys` | u64 | [R] |
| `sort_values_unsorted` | `BinningState.point_list_unsorted` | u32 | [R] |
| `sort_values_sorted` | `BinningState.point_list` | u32 | [R] |
| `sort_tile_ranges` | `ImageState.ranges` (uint2*) | u32 | [num_tiles, 2] |
| `rasterize_n_contrib` | `ImageState.n_contrib` | u32 | [H×W] |
| `rasterize_transmittance` | `ImageState.accum_alpha` | f32 | [H×W] |
| `rasterize_image` | 前向返回 `color` | f32 | [3, H, W] CHW |

**后向 tensor 清单**：

| tensor | dtype | shape | 是否当前已返回 |
|--------|-------|-------|-------------|
| `backward_d_means2D` | f32 | [P, **3**] | ✅ |
| `backward_d_colors` | f32 | [P, 3] | ✅ |
| `backward_d_conic` | f32 | [P, **2, 2**] | ❌ `dump_mode=true` 追加 |
| `backward_d_opacity` | f32 | [P, 1] | ✅ |
| `backward_d_means3D` | f32 | [P, 3] | ✅ |
| `backward_d_cov3D` | f32 | [P, 6] | ✅ |
| `backward_d_sh` | f32 | [P, M, 3] | ✅ |
| `backward_d_scales` | f32 | [P, 3] | ✅ |
| `backward_d_rotations` | f32 | [P, 4] | ✅ |
| `backward_d_gauss2screen` | f32 | [P, 4, 4] | **不纳入 SP-3 验收** |
| `backward_dL_dout_color` | f32 | [3, H, W] CHW | Python 侧 `rendered_image.grad`（需 `retain_grad()`）|

**训练附加 tensor**（SP-4 新增）：

| tensor | 说明 |
|--------|------|
| `step{N}_l1_{scalar, dL_dimage}` | L1 loss 标量与梯度 |
| `step{N}_ssim_{map, scalar, dL_dimage}` | fused_ssim 中间与梯度 |
| `step{N}_reg_{d_opacity, d_scales}` | 正则梯度贡献 |
| `step{N}_adam_{group}_{m, v, param_after}` | 6 组 × 3 种 |
| `step{N}_lr_active, step{N}_sh_degree_active` | LR 和 SH schedule 当前值 |
| `step{N}_relocation_{indices, new_opacity, new_scale}` | 致密化步 |
| `step{N}_noise_randn, step{N}_cam_index` | RNG artifacts（for PythonReplay）|
| `step{N}_densify_reinit_idx, step{N}_densify_add_idx` | multinomial 结果（for PythonReplay）|
| `step{N}_xyz_after_adam, step{N}_xyz_after_noise` | xyz 两态分离 |

### 2.4 `npy_reader.h` 接口

```cpp
enum class NpyDtype { float32, int32, int64, uint32, uint64 };

struct NpyArray {
    std::vector<uint8_t> raw;
    std::vector<size_t>  shape;
    NpyDtype             dtype;
    size_t  numel() const;
    float*    f32()   { return reinterpret_cast<float*>(raw.data()); }
    int32_t*  i32()   { return reinterpret_cast<int32_t*>(raw.data()); }
    uint32_t* u32()   { return reinterpret_cast<uint32_t*>(raw.data()); }
    uint64_t* u64()   { return reinterpret_cast<uint64_t*>(raw.data()); }
    int64_t*  i64()   { return reinterpret_cast<int64_t*>(raw.data()); }
};
NpyArray load_npy(const std::string& path);
void     assert_shape(const NpyArray&, std::vector<size_t>);
void     assert_dtype(const NpyArray&, NpyDtype);
```

### 2.5 目录结构

```
checkpoints/
└── step000001/
    └── cam0000/
        ├── manifest.json
        ├── preprocess_*.npy
        ├── sort_*.npy
        ├── rasterize_*.npy
        └── backward_*.npy
```

`manifest.json` 记录：step, camera_idx, seed, reference_commit, config_hash, 每 artifact 的 {filename, operator, tensor, shape, dtype, layout}。

### 2.6 SP-0 前置任务

- 锁定 **fused_ssim** 源码版本（写入 `dev_notes/dependencies.md`）。若无法拿到源，退化为 numerical probing + 等价 pure PyTorch SSIM。在此锁定前，SP-4 SSIM 部分仅接口实现，不进入验收。
- 验证 CUB `DeviceRadixSort::SortPairs` 是否 deterministic（同输入多次运行一致性）；若不稳定则排序验收降级。
- Python 侧 `rendered_image.retain_grad()` 是导出 `backward_dL_dout_color` 的必要条件，写入 dump_tool 规范。

---

## 3. SP-1：Vulkan 基础设施

### 3.1 VulkanContext：能力前置检查

初始化时必须完成以下检查并缓存：

```cpp
struct VulkanDeviceCapabilities {
    uint32_t max_push_constants_size;              // ≥ 128
    uint32_t max_compute_workgroup_invocations;    // ≥ 256
    uint32_t max_compute_shared_memory_size;       // ≥ 16384
    uint32_t max_compute_workgroup_size[3];        // ≥ {256,256,64}
    uint32_t subgroup_size;
    VkShaderStageFlags     subgroup_supported_stages;  // 必须含 COMPUTE
    VkSubgroupFeatureFlags subgroup_supported_ops;
    bool     has_shader_atomic_float;              // VK_EXT_shader_atomic_float
    uint32_t api_version;                          // ≥ VK_API_VERSION_1_1
};
```

**设备选择流程**：枚举 → 过滤（必需能力）→ 按类型排序 → 取第一个。过滤失败输出诊断信息。

**类型优先级（目标机 vs 开发机分离）**：

| 场景 | 默认优先级 | 触发条件 |
|------|----------|---------|
| 开发机（如 Tegra Thor，双 GPU 环境）| DISCRETE > INTEGRATED > CPU | 未设 `GS3D_VK_DEVICE` 环境变量时默认 |
| 目标机（Maleoon 920，HarmonyOS）| INTEGRATED > DISCRETE > CPU | 交叉编译或显式 `--prefer-integrated` |

**显式 override**（覆盖自动策略）：
- `GS3D_VK_DEVICE=<index>`：直接指定 physical device 索引（`vkEnumeratePhysicalDevices` 返回顺序）
- `GS3D_VK_DEVICE_NAME=<substring>`：按设备名字串匹配
- 任一 override 生效时跳过类型排序，仅做能力过滤

SP-6 性能报告必须记录实际选中的设备名（从 `VkPhysicalDeviceProperties::deviceName` 读）。

`VulkanContext` 是**普通 RAII 类，无内置单例**。单例放测试 helper。

### 3.2 VulkanBuffer

每次 upload/download 按需创建 staging buffer，立即销毁。**SP-1 正确性优先方案，非最终架构**；若 SP-2+ 出现瓶颈再升级为 staging allocator。

### 3.3 VulkanPipeline：两层 dispatch

**SPIR-V 输入字节流**（匹配 `xxd -i` 输出 `unsigned char[]`）：

```cpp
VulkanPipeline(const uint8_t* spirv_bytes, size_t spirv_byte_size,
               std::vector<BindingDesc> bindings,
               uint32_t push_constant_size,          // 运行时检查 ≤ 设备上限
               VulkanContext& ctx);
```

**两层 dispatch**：
- `dispatch_sync(...)`：同步，内置 begin/end/submit/wait，用于 hello 测试和 bring-up
- `record(VkCommandBuffer, ...)`：录制到外部命令缓冲区，SP-2+ 链式用
- `insert_compute_barrier(cmd)`：compute-to-compute SSBO 屏障 helper

### 3.4 SPIR-V 嵌入

CMake 规则：`glslc -O --target-env=vulkan1.1 *.comp → xxd -i → .h`；C++ 直接 `#include`，无运行时 I/O。

### 3.5 hello.comp 契约

```glsl
#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer A { float a[]; };
layout(set=0, binding=1) buffer B { float b[]; };
layout(set=0, binding=2) buffer C { float c[]; };
layout(push_constant) uniform PC { uint n; } pc;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    c[i] = a[i] + b[i];
}
```

3 SSBO（2 输入 1 输出）+ push constant `n` + 越界保护。作为后续所有管线基础设施的最小稳定红线。

---

## 4. SP-2：前向管线

### 4.1 Pass-first 分层

**6 个 Pass**：

| Pass | Shader |
|------|--------|
| PreprocessPass | `preprocess.comp` |
| PrefixScanPass | `prefix_sum.comp`（通用 scan，tiles_touched 和 radix bucket 共用）|
| ScatterPass | `scatter.comp`（duplicateWithKeys）|
| RadixSortPass | `radix_sort_count.comp` + `radix_sort_scatter.comp` + 复用 PrefixScanPass |
| TileRangePass | `tile_range.comp` |
| RasterizePass | `rasterize.comp` |

**4 个适配类**（组合，不定义新行为）：

| 类（继承） | 组合的 Pass |
|----------|------------|
| `PreprocessorVulkan : Preprocessor` | PreprocessPass |
| `TileBinnerVulkan : TileBinner` | PrefixScanPass（对 tiles_touched）+ ScatterPass |
| `SorterVulkan : Sorter` | RadixSortPass（内含 PrefixScanPass 对 histogram 复用）+ TileRangePass |
| `RasterizerVulkan : Rasterizer` | RasterizePass |

**API 分层**：适配类 override 基类同步方法；每 Pass 暴露 `record(cmd, ...)`。

### 4.2 RadixSort 算法闭环

4-bit radix，16 passes。每 pass 三步：

```
for bit in {0, 4, 8, ..., 60}:
    Step 1 [radix_sort_count.comp]:
        每 workgroup 构建局部 16 桶 histogram
    Step 2 [prefix_sum.comp 复用]:
        对 histograms 数组 exclusive scan → bucket_offsets
    Step 3 [radix_sort_scatter.comp]:
        按 bucket_offsets 稳定写入 keys_out, values_out
```

### 4.3 Schema 分层

| Pass | Golden schema | 设备内部 schema |
|------|-------------|---------------|
| PreprocessPass | packed `conic_opacity[N,4]` | packed float4 SSBO |
| PrefixScanPass | `point_offsets[N]` u32 | 同 |
| ScatterPass | `keys_unsorted[R]` u64, `values_unsorted[R]` u32 | 同 |
| RadixSortPass | `keys_sorted[R]` u64, `values_sorted[R]` u32 | 中间 histograms 不进 golden |
| TileRangePass | `tile_ranges[num_tiles, 2]` u32 | 同 |
| RasterizePass | `image[3,H,W]`, `transmittance[H×W]`, `n_contrib[H×W]` | 分离数组共享内存 |

**公共 deinterleave**：`PreprocessorVulkan::process()` 非链式调用时 download 后拆 packed 到公共 `PreprocessOutput.conics[N,3] + opacities_2d[N]`；链式（`record()`）全程 packed。

### 4.4 固定配置硬错误

```cpp
if (cfg.eval_3D) throw;                              // SP-2 Phase 2 才启用
if (cfg.tile_w != 16 || cfg.tile_h != 16) throw;
if (cfg.antialiasing) throw;
// SortMode != GLOBAL, tile_based_culling, load_balancing → throw
```

### 4.5 Training 语义

SP-2 固定 `training=true`（SH 不 clamp）。通过 specialization constant `spec_training=1` 注入。未来 `training=false` 用 `spec_training=0` 重编译，不改源码。

### 4.6 Camera UBO（std140，去冗余）

```glsl
layout(std140, binding=12) uniform CameraUBO {
    mat4 viewmatrix;          // 64B
    mat4 projmatrix;          // 64B
    mat4 inv_viewprojmatrix;  // 64B
    vec4 campos_pad;          // 16B, xyz=campos, w=0
    vec4 fov_size;            // 16B, x=tan_fovx, y=tan_fovy, z=W, w=H
};  // 224B
// focal_x = fov_size.z*0.5/fov_size.x  shader 内推导
```

`static_assert(sizeof(CameraUBO) == 224)` 主机侧校验。

### 4.7 Rasterize 共享内存（分离数组，禁 struct）

```glsl
shared float s_mean_x[256], s_mean_y[256];
shared float s_conic_a[256], s_conic_b[256], s_conic_c[256];
shared float s_opacity[256];
shared float s_rgb_r[256], s_rgb_g[256], s_rgb_b[256];
shared uint  s_gid[256];
// 总 10.24KB（< 16KB）
```

**禁止** `vec2/vec3/vec4/struct` 避免 std430 默认对齐填充。

### 4.8 完整 Binding 表

#### preprocess.comp（`local_size_x=256`）

| binding | 方向 | 内容 | shape | dtype |
|---------|-----|------|-------|-------|
| 0 | in SSBO | positions | [N,3] | f32 |
| 1 | in SSBO | scales | [N,3] | f32 |
| 2 | in SSBO | rotations | [N,4] | f32 |
| 3 | in SSBO | opacities | [N] | f32 |
| 4 | in SSBO | sh | [N,M,3] | f32 |
| 5 | in SSBO | filter_3D | [N] | f32 |
| 6 | out SSBO | means2D | [N,2] | f32 |
| 7 | out SSBO | depths | [N] | f32 |
| 8 | out SSBO | conic_opacity_packed | [N,4] | f32 |
| 9 | out SSBO | rgb | [N,3] | f32 |
| 10 | out SSBO | radii | [N] | i32 |
| 11 | out SSBO | tiles_touched | [N] | u32 |
| 12 | UBO | CameraUBO | 224B | |

Push constants (24B)：`num_gaussians, sh_degree, sh_coeffs_per_g, num_tiles_x, num_tiles_y, scale_modifier(f32)`
Specialization constants：`spec_training=1, spec_eval_3D=0`

#### prefix_sum.comp（通用 scan）

| binding | 方向 | 内容 | shape | dtype |
|---------|-----|------|-------|-------|
| 0 | in | input_array | [N] | u32 |
| 1 | out | output_array | [N] | u32 |
| 2 | inout | workgroup_sums | [num_wg] | u32 |

Push constants (16B)：`num_elements, phase(0/1/2), stride, _pad`。三次 dispatch 完成 3 级 Blelloch。

#### scatter.comp

| binding | 方向 | 内容 | shape | dtype |
|---------|-----|------|-------|-------|
| 0 | in | means2D | [N,2] | f32 |
| 1 | in | depths | [N] | f32 |
| 2 | in | radii | [N] | i32 |
| 3 | in | point_offsets | [N] | u32 |
| 4 | in | tiles_touched | [N] | u32 |
| 5 | out | keys_unsorted | [R] | u64 |
| 6 | out | values_unsorted | [R] | u32 |

Push constants (16B)：`num_gaussians, num_tiles_x, tile_w, tile_h`

#### radix_sort_count.comp

| binding | 方向 | 内容 | shape | dtype |
|---------|-----|------|-------|-------|
| 0 | in | keys_in | [R] | u64 |
| 1 | out | histograms | [num_wg × 16] | u32 |

Push constants (16B)：`num_elements, current_bit, _pad, _pad`

#### radix_sort_scatter.comp

| binding | 方向 | 内容 | shape | dtype |
|---------|-----|------|-------|-------|
| 0 | in | keys_in | [R] | u64 |
| 1 | in | values_in | [R] | u32 |
| 2 | in | bucket_offsets | [num_wg × 16] | u32 |
| 3 | out | keys_out | [R] | u64 |
| 4 | out | values_out | [R] | u32 |

每 pass 后 `keys_in ↔ keys_out` ping-pong。

#### tile_range.comp

| binding | 方向 | 内容 | shape | dtype |
|---------|-----|------|-------|-------|
| 0 | in | keys_sorted | [R] | u64 |
| 1 | out | tile_ranges | [num_tiles, 2] | u32 |

Push constants (16B)：`num_elements, num_tiles, _pad, _pad`

#### rasterize.comp（`local_size_x=16, local_size_y=16`）

| binding | 方向 | 内容 | shape | dtype |
|---------|-----|------|-------|-------|
| 0 | in | values_sorted | [R] | u32 |
| 1 | in | tile_ranges | [num_tiles,2] | u32 |
| 2 | in | means2D | [N,2] | f32 |
| 3 | in | conic_opacity_packed | [N,4] | f32 |
| 4 | in | rgb | [N,3] | f32 |
| 5 | out | out_image | [3,H,W] | f32 |
| 6 | out | transmittance | [H×W] | f32 |
| 7 | out | n_contrib | [H×W] | u32 |
| 8 | UBO | RasterUBO(bg_color vec4) | 16B | |

Push constants (20B)：`num_gaussians, image_width, image_height, num_tiles_x, num_tiles_y`

### 4.9 集成链路

```cpp
VulkanCommandBuffer cmd; cmd.begin();
pre.record(cmd, ...);       insert_compute_barrier(cmd);
bin.record(cmd, ...);       insert_compute_barrier(cmd);  // prefix+scatter
sort.record(cmd, ...);      insert_compute_barrier(cmd);  // radix×16 + tile_range
rast.record(cmd, ...);
cmd.end(); cmd.submit_and_wait();
```

---

## 5. SP-3：反向管线

### 5.1 Pass-first 分层

| Pass | Shader |
|------|--------|
| RasterizeBackwardPass | `rasterize_backward.comp` |
| PreprocessBackwardPass | `preprocess_backward.comp` |

适配类（非虚，对齐 `RasterizerBackwardCPU`/`PreprocessorBackwardCPU`）：`RasterizerBackwardVulkan`、`PreprocessorBackwardVulkan`。

### 5.2 固定配置硬错误

```cpp
if (cfg.eval_3D) throw;       // d_gauss2screen 不纳入 SP-3
if (!cfg.training) throw;     // 固定 training=true
if (cfg.tile_w != 16 || cfg.tile_h != 16) throw;
```

### 5.3 `atomicAdd(float)` 双路径

```glsl
#if HAS_ATOMIC_FLOAT_EXT
    #extension GL_EXT_shader_atomic_float : require
    #define atomicAddFloat(mem, val) atomicAdd(mem, val)
#else
    void atomicAddFloat(inout uint mem_u, float val) {
        uint expected = mem_u;
        while (true) {
            float old_val = uintBitsToFloat(expected);
            uint  new_bits = floatBitsToUint(old_val + val);
            uint  actual  = atomicCompSwap(mem_u, expected, new_bits);
            if (actual == expected) break;
            expected = actual;
        }
    }
#endif
```

**两套 SPIR-V 离线编译**：`*_native.spv` / `*_cas.spv`，由 SP-1 capability 检测选择。CAS 路径下相关 SSBO 声明为 `uint[]`，位重解释访问。

### 5.4 完整 Binding 表

#### rasterize_backward.comp（`local_size_x=16, local_size_y=16`）

| binding | 方向 | 内容 | shape | dtype |
|---------|------|------|-------|-------|
| 0 | in | values_sorted | [R] | u32 |
| 1 | in | tile_ranges | [num_tiles,2] | u32 |
| 2 | in | means2D | [N,2] | f32 |
| 3 | in | conic_opacity_packed | [N,4] | f32 |
| 4 | in | rgb | [N,3] | f32 |
| 5 | in | transmittance | [H×W] | f32 |
| 6 | in | n_contrib | [H×W] | u32 |
| 7 | in | dL_dout_color（**CHW contiguous**）| [3,H,W] | f32 |
| 8 | inout | dL_dmeans2D | [N,**3**] | f32/u32* |
| 9 | inout | dL_dconic_packed | [N,**4**] | f32/u32* |
| 10 | inout | dL_drgb | [N,3] | f32/u32* |
| 11 | inout | dL_dopacity | [N,1] | f32/u32* |
| 12 | UBO | CameraUBO | 224B | |

*CAS 路径 u32[]，native 路径 f32[]。

Push constants (20B)：`num_gaussians, image_width, image_height, num_tiles_x, num_tiles_y`
Specialization constants：`spec_eval_3D=0, spec_atomic_float_ext=0/1`

每次 backward 前 host 侧 `vkCmdFillBuffer(dL_d*, 0)` + barrier。

#### preprocess_backward.comp（`local_size_x=256`）

| binding | 方向 | 内容 | shape | dtype |
|---------|------|------|-------|-------|
| 0 | in | positions | [N,3] | f32 |
| 1 | in | scales | [N,3] | f32 |
| 2 | in | rotations | [N,4] | f32 |
| 3 | in | sh | [N,M,3] | f32 |
| 4 | in | filter_3D | [N] | f32 |
| 5 | in | radii | [N] | i32 |
| 6 | in | dL_dmeans2D | [N,3] | f32/u32* |
| 7 | in | dL_dconic_packed | [N,4] | f32/u32* |
| 8 | out | dL_dmeans3D | [N,3] | f32 |
| 9 | out | dL_dcov3D | [N,6] | f32 |
| 10 | out | dL_dsh | [N,M,3] | f32 |
| 11 | out | dL_dscales | [N,3] | f32 |
| 12 | out | dL_drotations | [N,4] | f32 |
| 13 | UBO | CameraUBO | 224B | |

Push constants (16B)：`num_gaussians, sh_degree, sh_coeffs_per_g, scale_modifier(f32)`
Specialization constants：`spec_training=1, spec_eval_3D=0, spec_atomic_float_ext=0/1`（同步 rasterize_backward 的 CAS 布局）

### 5.5 Schema 一致性

Golden 与设备完全同布局，无 deinterleave：
- dL_dmeans2D [P,3]（CUDA 分配 [P,3]，只写 xy）
- dL_dconic [P,2,2] 内存连续 [P,4]
- 其余标准形状

### 5.6 关键实现细节

- **SH→position 梯度链**（PORTING_PITFALLS §10）：degree≥1 时 `dL/dpos += dL/drgb × drgb/dviewdir × dviewdir/dpos`，必须显式实现
- **dL_dmeans2D 第 3 列恒 0**
- **dL_dconic 4 分量全写**（a11,a12,a21,a22 对称但都写）
- **被裁剪高斯跳过**：radii[i]==0 → `d_*[i]` 保持 0
- **Training 模式 SH 不 clamp**（与 SP-2 前向一致）

### 5.7 Golden 验收

**Layer 1 主验收**：单 Pass 对 CUDA golden。
**Layer 2 FD spot check**：独立交叉验证。

**标量 loss** 锁定（防漂移）：
```
L_scalar(image) = sum over (c,y,x) of image[c,y,x] * mask[c,y,x]
mask = fixed random ±1 pattern from seed=42, shape [3,H,W]
```

**Epsilon 规则**：xyz=1e-4, scale=1e-5, rotation=1e-5, opacity=1e-5, sh=1e-4。
**FD 容差**：`|fd_grad - computed_grad| / (|fd_grad| + 1e-8) < 1e-2`。

**双路径矩阵**：

| 测试文件 | native atomic | CAS fallback |
|---------|--------------|-------------|
| `test_rasterize_backward_pass_vk.cpp` | ✅ | ✅ |
| `test_preprocess_backward_pass_vk.cpp` | ✅ | ✅ |
| `test_backward_pipeline_vk.cpp` | ✅ | ✅ |

CAS 路径通过环境变量 `FORCE_CAS_PATH=1` 强制加载 CAS SPIR-V。Layout sanity 测试：同 dL_dout_color 按 CHW 上传应匹配 golden，按 HWC 应明显失败（PSNR < 20dB）。

### 5.8 集成链路

```cpp
VulkanCommandBuffer cmd; cmd.begin();
for (buf : {d_means2D, d_conic_packed, d_rgb, d_opacity,
             d_means3D, d_cov3D, d_sh, d_scales, d_rotations})
    cmd.fill(buf, 0);
insert_compute_barrier(cmd);
rast_bwd.record(cmd, ...);   insert_compute_barrier(cmd);
pre_bwd.record(cmd, ...);
cmd.end(); cmd.submit_and_wait();
```

---

## 6. SP-4：训练管线

### 6.1 Pass 清单

| Pass | Shader | 用途 |
|------|--------|------|
| L1LossPass | `l1_loss.comp` | L1 loss + dL/dimage |
| SSIMForwardPass | `ssim_forward.comp` | fused_ssim map + 标量 |
| SSIMBackwardPass | `ssim_backward.comp` | DSSIM 梯度 |
| RegLossPass | `reg_loss.comp` | opacity/scale 正则梯度 |
| AdamPass | `adam_step.comp` | 6 参数组各 dispatch |
| NoiseInjectionPass | `noise_injection.comp` | xyz 扰动 |
| RelocationPass | `mcmc_relocation.comp` | 新 opacity/scale（binomial）|

**TrainerVulkan**：组合上述 + SP-2/SP-3 全部 pass。新顶层类（不继承 CPU `Trainer`）。

### 6.2 每步训练流程（严格对齐 train.py:80-148）

```
per step:
  A. xyz_lr = lr_scheduler(step)                               # train.py:78
  B. if step % 1000 == 0: oneupSHdegree()                      # train.py:81-82
  C. 若 viewpoint_stack 空则 refill；randint(pop)              # train.py:85-87

  D. [GPU] 前向链 → rendered_image
  E. [GPU] L1LossPass → dL_L1
  F. [GPU] SSIMForwardPass + SSIMBackwardPass → dL_SSIM
  G. [CPU weights] 合并 dL_dimage = (1-λ)*L1 + λ*SSIM
  H. [GPU] 反向链（清零所有 d_* buffer，然后从 dL_dimage 填充梯度）
  I. [GPU] RegLossPass → d_opacity, d_scales 在 H 输出基础上累加正则项
           （严禁放在 H 之前：H 的清零会抹掉 RegLossPass 的贡献）

  J. [densify] if step < 10000 && step > 5000 && step%100==0:
        relocate_gs(...)      # 先 relocate
        add_new_gs(cap_max)   # 再 add new
  K. [GPU] AdamPass × 6
  L. [GPU] zero_grad
  M. [GPU] NoiseInjectionPass（用 Adam 后激活参数重建协方差）
```

**关键顺序**：
- RegLossPass **在** 反向链 **之后**（对齐 train.py:104-108：Python 把 reg 并入 loss 再 `loss.backward()`，反向末态含 reg 梯度）
- densify **在** Adam **之前**
- noise **在** Adam **之后**

### 6.3 AdamPass

```glsl
layout(push_constant) uniform PC {
    uint  num_elements;
    float lr;
    float beta1;           // 0.9
    float beta2;           // 0.999
    float eps;             // 1e-15
    float one_minus_bc1;   // 1 - beta1^t, CPU 预算
    float one_minus_bc2;   // 1 - beta2^t, CPU 预算
    uint  _pad;
} pc;
// m_hat = new_m / pc.one_minus_bc1
```

**CPU 预算 bias-correction**，避免 shader 内 `pow(beta, step)`。

**6 参数组**：xyz, f_dc, f_rest, opacity, scaling, rotation（各自 LR 和 shape；xyz 指数衰减，其他常数）。

### 6.4 Loss

- **L1LossPass**：前向标量 + dL/dimage 合并一 pass。
- **fused_ssim 对齐**：必须跟 Python `fused_ssim`，不走朴素 SSIM。**前置：SP-0 锁定版本**。
- **权重合并**：两 pass 各自接收 `weight` push constant（0.8 / 0.2），通过 `vkCmdFillBuffer(0)` + 两 pass `atomicAdd` 合并到同一 `dL_dout_color`。

### 6.5 MCMC 致密化

**边界条件**（对齐 train.py:130-131）：
```
step < densify_until_iter  &&  step > densify_from_iter  &&  step % interval == 0
```
SP-4 固定 `from=5000, until=10000, interval=100` → 触发步 **5100, 5200, ..., 9900（49 次）**。

**relocate_gs**（gaussian_model.py:363-389）：
1. CPU: dead_mask, alive_mask, multinomial 采样 reinit_idx（weighted by alive opacity）
2. GPU RelocationPass: new_opacity, new_scaling（binomial 级数）
3. GPU scatter: xyz/features/rotation[dead] = 源值；opacity/scaling[dead] = 逆激活(new)
4. GPU 源位更新: opacity/scaling[reinit_idx] = 逆激活(new)
5. Adam reset: m/v[reinit_idx] = 0

**add_new_gs**（gaussian_model.py:391-411）：
1. CPU: N_tgt = min(cap_max, int(1.05*N)); probs 来自 **relocate 后** opacity
2. GPU RelocationPass on add_idx
3. 源位更新: opacity/scaling[add_idx] = 逆激活(new)
4. Buffer 扩容 + 末尾 num_new 槽填源值（opacity/scaling 用 new 逆激活）
5. Adam state: m/v[add_idx]=0；末尾 num_new 自然为 0

### 6.6 LR / SH schedule（CPU）

```python
def get_xyz_lr(step, init=1.6e-4, final=1.6e-6, max_steps=30000,
               delay_steps=0, delay_mult=0.01):
    # sin warmup + log-linear decay
    ...

active_sh_degree = min(step // 1000, 3)
```

每步 CPU 计算，作为 push constant 传入。

### 6.7 NoiseInjectionPass（Adam 后激活重建协方差）

```glsl
// Input: xyz, sc_raw[N,3], rot_raw[N,4], op_raw[N], noise_in[N,3]
// 1. scaling = exp(sc_raw), rotation = normalize(rot_raw), opacity = sigmoid(op_raw)
// 2. L = R @ diag(scaling)  (3x3)
// 3. cov = L @ L^T
// 4. op_sig = 1 / (1 + exp(-k*(1 - sigmoid(op_raw) - x0)))
// 5. raw_noise = noise_in[i] * op_sig * noise_lr * xyz_lr
// 6. final_noise = cov @ raw_noise
// 7. xyz[i] += final_noise
```

**关键**：在 Adam **之后** 执行，在 shader 内从 raw 参数**重建**协方差，**不用** 旧 Cov3D 缓存。

`noise_in` 来源取决于 RNG 模式（见 §7）。

### 6.8 RegLossPass（链式通过激活函数）

```glsl
// L_op = λ_op * mean(sigmoid(o)), L_sc = λ_sc * mean(exp(s))
// scaling shape [N, 3] —— k 枚举 3 个分量（0, 1, 2），严禁写成 k < 4 或 k <= 3
float sig_o = 1.0 / (1.0 + exp(-op_raw[i]));
d_op[i] += pc.opacity_reg * sig_o * (1.0 - sig_o) / float(pc.N);
for (int k = 0; k < 3; ++k) {
    float exp_s = exp(sc_raw[i*3 + k]);
    d_sc[i*3 + k] += pc.scale_reg * exp_s / float(3 * pc.N);
}
```

**输入是 raw opacity/scaling**（未激活），链式经激活函数。
**严格要求**：scaling 只有 3 个维度（x/y/z），循环上界固定 `k < 3`；超出即越界写，会污染下一个高斯的 `d_sc[0]`。

### 6.9 超参数（SP-4 固定）

| 参数 | 值 | 来源 |
|------|-----|------|
| lambda_dssim | 0.2 | Python |
| opacity_reg / scale_reg | 0.01 / 0.01 | Python |
| beta1 / beta2 | 0.9 / 0.999 | Python |
| **eps** | **1e-15** | Python |
| noise_lr | 5e5 | Python |
| op_sigmoid_k / x0 | 100 / 0.995 | Python |
| max_sh_degree | 3 | Python |
| **densify_from_iter** | **5000** | 项目约束 |
| **densify_until_iter** | **10000** | 项目约束 |
| densification_interval | 100 | Python |
| dead_threshold | 0.005 | Python |
| growth_ratio | 1.05 | Python |
| cap_max | 3_000_000 | Python 默认 |

### 6.10 RNG 模式

| 模式 | 用途 | 语义 |
|------|------|------|
| `PythonReplay` | SP-4 / SP-5 严格验证 | 重放 Python dump 的 **stochastic artifacts**（cam_index, noise_randn, multinomial 结果），Vulkan 不做实际采样 |
| `NativeVulkan` | SP-6 soak | 自生 RNG（PCG-32 + Box-Muller）用于 camera 和 noise；**multinomial 保持 CPU 精确**（std::discrete_distribution，算法等价 torch.multinomial） |

`PythonReplay` 不是 seed replay 或 algorithm replay，是 **artifact replay**：磁盘读取 Python 采样好的结果作为确定性输入。

### 6.11 Golden 验收（SP-4 自身）

- 各 Pass 单元测试：见 §2.3 tensor 清单
- `test_trainer_step_vk.cpp`：单步端到端，6 参数组 param/m/v + xyz_after_adam + xyz_after_noise 对齐 Python golden
- `test_densification_vk.cpp`：致密化触发步独立测试，buffer 形状/值/Adam 重置对齐

---

## 7. SP-5 一致性验证 + SP-6 Soak & 性能

### 7.1 职责对比

| | SP-5 | SP-6 |
|---|------|------|
| 目标 | 正确性门 | 工程可用性 + 性能 |
| 步数 | 2000 | 10000 |
| RNG | `PythonReplay` | `NativeVulkan` |
| 验收 | 逐元素严格对齐 | PSNR/SSIM 收敛质量 |
| 前置 | SP-4 绿 | SP-5 绿 |

### 7.2 SP-5：2000 步 Checkpoint Ladder

**Ladder**：`1, 10, 100, 500, 1000, 2000`

每 ladder 步验收矩阵：

| 对象 | 容差 | 状态定义 |
|------|------|---------|
| rendered_image | abs<1e-5, rel<1e-4 | 该步前向输出 |
| param（除 xyz） | abs<1e-6, rel<1e-4 | Adam 后 |
| m, v（各 6 组） | abs<1e-6, rel<1e-4 | Adam 后 |
| **xyz_after_adam** | abs<1e-6, rel<1e-4 | Adam 后，noise 前（中间态）|
| **xyz_after_noise** | abs<1e-5, rel<1e-4 | NoiseInjectionPass 后，本步末态 |
| L1/SSIM/reg scalar | abs<1e-6, rel<1e-5 | |
| N（高斯数）| 精确相等 | |

**失败处理**：二分定位首次偏离步（需 SP-0 支持 `--dump_steps` 按需导出）。

**产出**：`dev_notes/sp5_validation_report.md` + git tag `sp5-ok-<commit>`。

### 7.3 SP-6：10000 步 Soak + 性能

**Python 参考**（需 `--eval` 切分 train/test；使用 AAA-Gaussians `train.py` 现有 CLI）：
```bash
conda run -n aaa-gs python AAA-Gaussians/train.py \
  -s /home/robota/Downloads/basketball \
  -m dev_notes/ground_truth/sp6/ \
  --iterations 10000 \
  --densify_from_iter 5000 --densify_until_iter 10000 \
  --densification_interval 100 --eval \
  --test_iterations 10000 --save_iterations 10000
```

注：`-m` / `--model_path` 是 AAA-Gaussians `__init__.py:52-59` 提供的输出目录参数；**不存在** `--output_path`。`dump_tool.py` 如需新参数应作为独立 CLI，不混入 `train.py`。

**Vulkan 侧**：
```bash
./build/gs3d_train_vk \
  --source_path /home/robota/Downloads/basketball \
  --ply_path ...points3D.ply --iterations 10000 \
  --densify_from_iter 5000 --densify_until_iter 10000 \
  --densification_interval 100 --eval \
  --rng_mode native_vulkan --output output/sp6
```

**Native RNG 作用域**：camera sampling + noise randn 用 PCG-32；**multinomial 保持 CPU 精确**（避免训练语义偏置）。

**指标**（仅 test split）：

| 指标 | 目标 |
|------|------|
| Test PSNR | ≥ CUDA Test PSNR − 0.5 dB |
| Test SSIM | ≥ 0.98 |
| Test L1 | ≤ CUDA Test L1 × 1.1 |
| N @ step 10000 | 差 ≤ 5% |
| L1 train 末值（最后 100 步均值）| 差 ≤ 10% |

移除 "Train PSNR"（原定义不清）。

**性能采样分 bucket**：

| Bucket | 步数 |
|--------|------|
| Normal step | ≈ 9951 |
| Densify step | 49 |

每 pass 报 mean/p50/p90/p99。总时间 = `normal_mean × 9951 + densify_mean × 49`。

**产出**：`dev_notes/sp6_soak_report.md` + git tag `sp6-ok-<commit>`。

### 7.4 README（终版接口草案）

`harmonyos_3dgs/README_VULKAN.md`（实施阶段定稿）：

```markdown
# AAA-Gaussians Vulkan 移植
> 状态：本文档为最终交付接口草案。具体命令名、CMake target、输出目录在 SP-5/SP-6
> 实施阶段最终锁定。

## 构建（草案）
cmake -B build -DBUILD_TESTS=ON -DENABLE_VULKAN=ON
cmake --build build

## 单元测试
ctest --test-dir build --output-on-failure

## 生成 CUDA golden（草案）
conda activate aaa-gs
python tools/dump_tool.py --fixture=tiny --output=tests/golden/tiny
python tools/dump_tool.py --fixture=basketball --output=dev_notes/ground_truth \
    --ladder=1,10,100,500,1000,2000,10000

## SP-5 / SP-6 验证（草案）
./build/gs3d_sp5_validate --golden=... --rng_mode=python_replay
./build/gs3d_train_vk --source_path=... --iterations=10000 ...
python tools/compare_renders.py --ref=... --pred=... --metrics=psnr,ssim,l1
```

### 7.5 交付物

| 类别 | 产出 |
|------|------|
| **代码** | `src/vulkan/` 全部 shader + Vulkan 类 + TrainerVulkan；`tools/dump_tool.py`；`tests/golden/npy_reader.h` |
| **测试** | 每 pass 一份 `test_*_vk.cpp`；集成链；单步 trainer；致密化；FD spot check |
| **Golden** | `tests/golden/tiny/` + `dev_notes/ground_truth/{sp5,sp6}/` |
| **文档** | `README_VULKAN.md`、`dev_notes/sp5_validation_report.md`、`dev_notes/sp6_soak_report.md`、`dev_notes/dependencies.md`（fused_ssim 版本）、`dev_notes/rng_contract.md`（PythonReplay / NativeVulkan 契约）|
| **Git tag** | `sp5-ok-<commit>` / `sp6-ok-<commit>` |

---

## 关键文件路径

| 用途 | 路径 |
|------|------|
| 抽象接口 | `harmonyos_3dgs/include/{preprocessor,rasterizer,tile_binner,sorter}.h` |
| 当前公共类型 | `harmonyos_3dgs/include/types.h`（PreprocessOutput / BinningOutput / ForwardCache）|
| OpenCL 参考 | `harmonyos_3dgs/src/gpu/kernels/*.h` |
| CPU 参考 | `harmonyos_3dgs/src/cpu/*.{h,cpp}` |
| CUDA 参考（不改）| `AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/*.cu` |
| CUDA 绑定（SP-0 修改点）| `AAA-Gaussians/submodules/diff-gaussian-rasterization/rasterize_points.cu`、`ext.cpp` |
| Python 训练循环 | `AAA-Gaussians/train.py`（L80-148）|
| Python Gaussian 模型 | `AAA-Gaussians/scene/gaussian_model.py`（L340-412 densify）|
| 陷阱清单 | `PORTING_PITFALLS.md` |
| 算子分析 | `harmonyos_3dgs/docs/aaa_operators_analysis.md` |
| Vulkan 迁移分析 | `dev_notes/master_plan/vulkan_migration_plan.md` |

新增文件（SP 实施时创建）：

```
harmonyos_3dgs/
├── src/vulkan/
│   ├── vk_context.{h,cpp}
│   ├── vk_buffer.{h,cpp}
│   ├── vk_pipeline.{h,cpp}
│   ├── preprocessor_vk.{h,cpp}
│   ├── tile_binner_vk.{h,cpp}
│   ├── sorter_vk.{h,cpp}
│   ├── rasterizer_vk.{h,cpp}
│   ├── rasterizer_backward_vk.{h,cpp}
│   ├── preprocessor_backward_vk.{h,cpp}
│   ├── trainer_vk.{h,cpp}
│   └── shaders/*.comp
├── tests/
│   ├── test_vulkan_context.h
│   ├── test_*_pass_vk.cpp            (每 pass 一份)
│   ├── test_forward_pipeline_vk.cpp
│   ├── test_backward_pipeline_vk.cpp
│   ├── test_trainer_step_vk.cpp
│   ├── test_densification_vk.cpp
│   └── golden/npy_reader.h, compare.h, manifest.h
tools/dump_tool.py
dev_notes/{sp5_validation_report.md, sp6_soak_report.md,
           dependencies.md, rng_contract.md}
```

---

## 验证

### 阶段门

| 阶段 | 验证方式 |
|------|---------|
| SP-0 | Python `dump_tool.py --fixture=tiny` 运行成功；npy_reader ctest pass；fused_ssim 版本锁定 |
| SP-1 | `ctest -R VulkanHello` 通过；能力检查拒绝不合规设备 |
| SP-2 | 每 pass `ctest -R *_pass_vk` 通过；`test_forward_pipeline_vk` 通过 |
| SP-3 | `test_*_backward_pass_vk` 双路径通过；`test_backward_fd_spot_check_vk` 通过 |
| SP-4 | `test_trainer_step_vk`（单步）+ `test_densification_vk` 通过 |
| SP-5 | 6 ladder 步全部绿；`dev_notes/sp5_validation_report.md` 签名 |
| SP-6 | Test PSNR/SSIM 达标；性能报告生成；`dev_notes/sp6_soak_report.md` 签名 |

### 端到端命令（终版草案）

```bash
# 构建（开发机）
cd harmonyos_3dgs && cmake -B build -DBUILD_TESTS=ON -DENABLE_VULKAN=ON \
  && cmake --build build

# 单元测试
ctest --test-dir build --output-on-failure

# CUDA golden（Python 环境）
conda run -n aaa-gs python tools/dump_tool.py --fixture=tiny --output=tests/golden/tiny
conda run -n aaa-gs python tools/dump_tool.py --fixture=basketball \
  --output=dev_notes/ground_truth --ladder=1,10,100,500,1000,2000,10000

# SP-5 一致性
./build/gs3d_sp5_validate --golden=dev_notes/ground_truth/sp5 --rng_mode=python_replay

# SP-6 soak
./build/gs3d_train_vk --source_path=/home/robota/Downloads/basketball \
  --iterations=10000 --densify_from_iter=5000 --densify_until_iter=10000 \
  --eval --rng_mode=native_vulkan --output=output/sp6

# 对比报告
python tools/compare_renders.py --ref=dev_notes/ground_truth/sp6/step010000 \
  --pred=output/sp6/step010000 --metrics=psnr,ssim,l1
```
