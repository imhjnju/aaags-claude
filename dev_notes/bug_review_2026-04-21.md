# 代码审查报告：潜在 Bug 汇总

- **日期**：2026-04-21
- **审查范围**：harmonyos_3dgs 全部 Vulkan pipeline + 训练逻辑
- **审查基线**：commit `6c14876` (Merge SP-2 through SP-7)，252/252 tests pass
- **方法**：4 个并行子审查 agent（trainer pipeline / GPU Adam / preprocessor+sorter / MCMC densification）+ 4 个验证 agent 对 HIGH 级发现逐一二次确认

---

## 概览

| 级别 | 数量 | 说明 |
|------|------|------|
| BUILD BREAK | 1 | 构建失败，阻塞所有开发 |
| HIGH | 4 | 确认的正确性 bug，影响训练质量或存在 UB |
| MEDIUM | 6 | 规范违规、方向性错误、边界缺失 |
| LOW | 7 | 设计/性能/健壮性改进 |

当前测试套件未暴露上述 HIGH 级 bug 的原因：

- H1（SH 布局）：测试中 `sh_degree_warmup=1000`，使 `active_sh_degree` 始终为 0，rest 梯度为零
- H2/H3（density_controller OOB）：越界读到的内存通常为 0，静默跳过
- H4（scale reg 3x）：无 Python-vs-C++ 的正则化梯度对比测试

---

## B0 — BUILD BREAK：CMakeLists 引用不存在的源文件

- **位置**：`harmonyos_3dgs/CMakeLists.txt:261`
- **问题**：`add_executable(gs3d_vk_render src/vulkan/vk_render_main.cpp)` 引用了不存在的文件
- **现象**：`cmake -B build` 报 "Cannot find source file"，构建完全失败
- **修复**：若该工具已迁移或弃用，删除这 3 行 CMake 定义；若需要保留，从正确的分支恢复源文件

---

## HIGH 级 — 确认的正确性 Bug

### H1：SH DC/rest 内存布局错误（Adam 更新区域错位）

- **位置**：`src/vulkan_trainer.cpp` 行 119-125, 224-228, 417-421, 480-484, 669-677
- **根因**：`raw_sh_coeffs_` 采用 AoS 布局 `[N × max_coeffs × 3]`，每个 Gaussian 的全部 SH 系数连续存储。Gaussian `i` 的 DC 位于偏移 `i * max_coeffs * 3`。但代码将前 `N*3` 个 float 当作全部 DC、后续当作全部 rest 分别上传到两个 Adam group（不同学习率）。

  对 N=2, max_coeffs=4 的例子：
  ```
  AoS 实际布局: [G0_dc, G0_r1, G0_r2, G0_r3, G1_dc, G1_r1, G1_r2, G1_r3]
                 (每个 token = 3 floats)

  代码认为的布局:
    DC  group = [G0_dc, G0_r1]      ← 错！包含了 G0 的第一个 rest 系数
    Rest group = [G0_r2, G0_r3, G1_dc, G1_r1, G1_r2, G1_r3]  ← 错！包含了 G1 的 DC
  ```

- **影响**：
  - Adam group 1（DC, lr=2.5e-3）更新了错误的内存区间
  - Adam group 2（rest, lr=1.25e-4）更新了错误的内存区间
  - GPU 下载后 `raw_sh_coeffs_` 布局被破坏，后续每步渲染颜色错乱
  - **sh_degree=0 时无影响**（max_coeffs=1，DC 即为全部）
  - **sh_degree >= 1 时训练产出错误颜色**

- **遮蔽原因**：当前所有测试使用 `sh_degree_warmup=1000`，使 `active_sh_degree` 在 100 步测试中始终为 0，rest 梯度恒为零

- **修复方案**：
  - 方案 A（推荐）：将 `raw_sh_coeffs_` 拆为 `raw_sh_dc_[N*3]` + `raw_sh_rest_[N*rest_coeffs*3]`，与 PyTorch 的 `_features_dc` / `_features_rest` 对齐
  - 方案 B：上传/下载时做 AoS↔SoA 转置

### H2：densify_and_split 中 avg_grads 越界读取

- **位置**：`src/density_controller.cpp:85-90`
- **根因**：调用链 `densify_and_prune → densify_and_clone → densify_and_split`。`densify_and_clone` 追加了新 Gaussian，使 `params.count()` 增大。随后 `densify_and_split` 在第 85 行取 `int N = params.count()`（clone 后的值），但 `avg_grads` 大小仍为原始 N。循环对 `i >= 原始N` 读取 `avg_grads[i]` 为未定义行为。

- **影响**：越界读到的值通常为 0（堆未初始化区域），使 `avg_grads[i] < threshold` 为 true 而跳过，实际等效于"不 split 克隆出的 Gaussian"——行为碰巧正确但机制是 UB

- **修复**：在 `densify_and_prune` 中 clone 前捕获原始 N，传入 split 限制循环上界
  ```cpp
  // densify_and_prune 中:
  const int N_orig = params.count();
  densify_and_clone(params, avg_grads, ...);
  densify_and_split(params, avg_grads, ..., N_orig);  // 只迭代原始范围
  ```

### H3：compact() 中 to_remove 越界读取

- **位置**：`src/density_controller.cpp:87,152` + `include/train_types.h:120-124`
- **根因**：`densify_and_split` 中 `to_remove` 大小为 clone 后 N。split 循环通过 `params.append()` 追加新 Gaussian 后，`params.count()` > `to_remove.size()`。第 152 行调用 `params.compact(to_remove)` 时，`compact()` 以 `count()` 为循环上界，对追加部分读取 `remove_mask[src]` 越界。

- **影响**：`std::vector<bool>` 为 bit-packed 存储，越界读到的 bit 不确定。新 split 出的子 Gaussian 可能被错误删除。

- **修复**：compact 前扩展 to_remove
  ```cpp
  to_remove.resize(params.count(), false);  // 保留新子 Gaussian
  params.compact(to_remove);
  ```

### H4：Scale 正则化梯度偏大 3 倍

- **位置**：`src/vulkan_trainer.cpp:379, 386`
- **根因**：Python 参考为 `loss += scale_reg * torch.abs(get_scaling).mean()`，其中 `get_scaling` shape 为 `[N, 3]`，`.mean()` 无参数时对全部 `N×3` 个元素求均值，因此梯度除数为 `N×3`。C++ 代码用 `inv_N = 1.0f / N`，缺少 `×3` 因子。

  ```
  Python:  dL/d_raw_sc[i,k] = scale_reg / (N * 3) * exp(raw_sc[i,k])
  C++:     dL/d_raw_sc[i,k] = scale_reg / N       * exp(raw_sc[i,k])  ← 3x 偏大
  ```

- **影响**：Scale 正则化过强 3 倍，Gaussian 被过度压缩，渲染细节丢失。Opacity 正则化不受影响（shape `[N,1]`，`/N` 正确）。

- **修复**：
  ```cpp
  // 行 386: scale 专用系数
  const float scale_coeff = tcfg_.scale_reg / static_cast<float>(N_ * 3);
  ```
  同时更新行 372 的注释（注释本身也写了错误公式）。

---

## MEDIUM 级 — 正确性/规范问题

### M1：build_L 协方差矩阵方向错误（位置噪声注入）

- **位置**：`src/train_utils.cpp:13`
- **根因**：C++ 的 `quat2mat`（math_utils.cpp:158）产出的旋转矩阵 `R_cpp` 是 Python `build_rotation` 的转置（`R_cpp = R_py^T`）。`build_L` 照搬了 Python 公式 `L[i][j] = R[i][j] * s[j]`（即 `L = R @ diag(s)`），但在 C++ 中这产出 `Σ = R_cpp @ S² @ R_cpp^T`，而预处理器的协方差为 `Σ = R_cpp^T @ S² @ R_cpp`。二者对各向异性 Gaussian 的 off-diagonal 元素符号相反。

- **影响**：MCMC 位置噪声沿错误方向注入，降低各向异性 Gaussian 的训练效果。各向同性 Gaussian 不受影响。

- **修复**：`L[i][j] = R[j][i] * act_scale[j]`（转置 R 索引）。

### M2：位置噪声使用过期的激活值

- **位置**：`src/vulkan_trainer.cpp:571-609`
- **根因**：`inject_position_noise()` 读取 `act_opacities_` / `act_scales_` / `act_rotations_`，这些值在步初 `activate_params()` 时计算（Adam 更新前）。Adam 完成后未重新调用 `activate_params()`。Python 参考通过 `@property` 实时计算。
- **影响**：门控和协方差滞后一个 Adam 步。学习率较小时影响可控。
- **修复**：在 Adam 下载后、噪声注入前调用 `activate_params()`，或在 `inject_position_noise` 内联计算 `sigmoid/exp/normalize`。

### M3：vkMapMemory 传入 size=0

- **位置**：`src/vulkan/vulkan_adam.cpp:99-100`
- **根因**：`zero_moments()` 当 `g.n == 0` 时 `n_bytes = 0`，`upload()` 调用 `vkMapMemory(size=0)` 违反 `VUID-vkMapMemory-size-00680`。
- **影响**：Vulkan 验证层报错；大多数驱动静默忽略。
- **修复**：`if (g.n == 0) continue;`

### M4：2-level prefix scan 的 readonly 别名 UB

- **位置**：`src/vulkan/prefix_scan_pass.cpp:78-82` + `shaders/prefix_sum.comp:37`
- **根因**：`bind_buffers_2level` 将 `wg_sums1` 同时绑定为 `readonly buffer InputArray`（binding 0）和可写 `buffer OutputArray`（binding 1）。通过 `readonly` 别名写入是 Vulkan 规范 UB，即使 shared memory 做了中介。
- **影响**：实际驱动工作正常，但理论上驱动可能利用 `readonly` 做缓存优化导致数据不一致。
- **修复**：移除 `prefix_sum.comp` 中 `InputArray` 的 `readonly` 限定符。

### M5：CPU fallback sort 无 tile_id 边界检查

- **位置**：`src/vulkan/sorter_vulkan.cpp:66`
- **根因**：GPU path 的 `tile_range.comp:45` 有 `if (tile_id >= num_tiles) return` 保护，CPU fallback 无此检查。
- **影响**：若 key 中 tile_id 超出合法范围，导致 `tile_ranges[]` OOB 写。
- **修复**：添加 `if (cur_tile >= static_cast<uint32_t>(num_tiles)) continue;`

### M6：total_pairs 为 int，大场景溢出风险

- **位置**：`src/vulkan/sorter_vulkan.cpp:170`
- **根因**：`static_cast<int>(R)` 当 R > `INT_MAX` 时为有符号整数溢出（UB）。
- **影响**：当前小场景不触发；生产规模可能触发。
- **修复**：改为 `uint32_t` 或 `int64_t`。

---

## LOW 级 — 设计/性能/健壮性

| # | 位置 | 问题 | 建议 |
|---|------|------|------|
| L1 | `vulkan_trainer.cpp:319` | CHW→HWC 转换每步分配整帧拷贝（~8MB at 720×960） | TODO 已标记；长期改 `rasterize.comp` 直接输出 HWC |
| L2 | `vulkan_trainer.cpp:712` | `reallocate_for_n` 重置 Adam m/v 但不重置 `step_count_`，新 Gaussian 偏差校正被低估 | 与 Python 参考对齐（检查 PyTorch optimizer state 在 densification 后的行为） |
| L3 | `vulkan_trainer.cpp:504-508` | `grad_means2D_accum_` 用 3D 位置梯度范数代替 Python 的 2D 屏幕空间梯度 | 改为投影后 2D 梯度或添加注释说明差异 |
| L4 | `density_controller.cpp:189` | 屏幕尺寸剪枝跳过 clone/split 新 Gaussian | 下次 densification 间隔会捕获；低优先级 |
| L5 | `prefix_sum.comp:54-83` | `blelloch_exclusive_scan_256()` 返回值仅 lane 255 有效，API 契约脆弱 | 增加文档注释或重构返回方式 |
| L6 | `vulkan_adam.h:17` | Adam epsilon=1e-15 导致类 sign-SGD 行为 | 如与 Python 参考一致则保留并注释；否则改为 1e-8 |
| L7 | `vulkan_trainer.cpp:89,718` | `N_ * rest_coeffs * 3` int 乘法可能溢出 | 改为 `static_cast<uint32_t>(N_) * rest_coeffs * 3` |

---

## 推荐修复优先级

```
阶段 1 — 恢复构建 + 训练正确性
  B0  删除/恢复 vk_render_main.cpp CMake 目标
  H1  SH DC/rest 拆分为独立数组
  H4  Scale 正则化除数改为 N*3
  M1  build_L 转置修复

阶段 2 — 消除 UB
  H2  densify_and_split 循环上界限制为原始 N
  H3  compact 前 to_remove.resize(params.count(), false)
  M3  zero_moments 跳过 n=0 的 group

阶段 3 — 规范合规 + 防御性加固
  M2  噪声注入前重新激活参数
  M4  prefix_sum.comp 移除 readonly
  M5  CPU fallback 添加 tile_id 边界检查
  M6  total_pairs 改为 uint32_t
```

---

## 附录：验证矩阵

| Bug | 初审 Agent | 二次验证 | 结论 |
|-----|-----------|---------|------|
| H1 SH 布局 | Trainer agent | 独立 agent 逐行追踪 AoS 布局 | **确认** |
| H2 avg_grads OOB | MCMC agent | 独立 agent 追踪 clone→split 调用链 | **确认** |
| H3 to_remove OOB | MCMC agent | 同上 | **确认** |
| H4 scale reg 3x | Adam agent | 独立 agent 对比 Python `train.py:107` | **确认** |
| M1 build_L 转置 | Adam agent | 独立 agent 数学推导 + quat2mat 对比 | **确认** |
