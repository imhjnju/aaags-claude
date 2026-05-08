# VK↔CUDA Parity Progress — 2026-04-28 to 2026-04-29

## 一句话结论

这两天定位到的、真正导致 scan1 独立训练 VK↔CUDA 比对长期不通过的主要移植问题，是 **2D rasterizer/backward 的 `n_contrib` 语义不一致**：CUDA 记录的是“最后一个实际参与 blend 的候选 Gaussian 在 tile 遍历中的 1-based 位置”，而 Vulkan/CPU 之前记录的是“实际 blend 的 Gaussian 个数”。这个差异会让 backward replay 回放错误的候选区间，优先扰动 `d_rgb`，再进入 SH 梯度并通过 Adam 累积放大。

修复后，scan1 camera0 的最终渲染 VK↔CUDA PSNR 明显改善：

| 训练步数 | 修复前 VK↔CUDA PSNR | 修复后 VK↔CUDA PSNR | 结论 |
|---:|---:|---:|---|
| 10 | 90.91 dB | 107.136986 dB | 基本闭合 |
| 200 | 22.42 dB | 49.822707 dB | 大幅改善 |
| 1000 | 8.24 dB | 27.773822 dB | 不再崩坏，但仍有轨迹分叉 |

1000 step 修复后 scan1 结果：CUDA-vs-GT `19.753679 dB`，VK-vs-GT `19.868585 dB`，VK-vs-CUDA `27.773822 dB`。也就是说，最终 GT 质量已经接近，剩余差异更像长程数值轨迹分叉，而不是一个已经定位到的公式级错误。

## 问题优先级总览

| 优先级 | 问题 | 类型 | 是否主因 | 状态 |
|---|---|---|---|---|
| P0 | `n_contrib` replay-boundary 语义错误 | Vulkan/CPU 2D rasterizer/backward 移植 bug | **是，scan1 失败主因** | 已修复并验证 |
| P1 | fair CLI 路径配置不一致 | harness/CLI 配置 bug | 不是核心算法主因，但会污染比对 | 已修复 |
| P1 | SH Adam DC/REST 分组上传布局错误 | trainer 参数布局 bug | 是早期 trajectory/SH parity 主因 | 已修复 |
| P2 | scale/rotation 100-step 后梯度相对误差增长 | 数值放大/病态初始化 | 不是已证实移植 bug | 继续观察 |
| P2 | eval_3D CLI 没有真正构造 eval_3D trainer | CLI 配置 bug | eval_3D smoke 误判主因 | 已修复 |
| P2 | eval_3D backward 缺失 | 功能缺口 | eval_3D 训练阻塞主因 | 已实现 parity-mode 路径 |
| P3 | eval_3D shared memory 超 32 KiB 风险 | shader 资源风险 | 不是数值不通过主因 | 已规避 |

## 1. P0 主因：`n_contrib` 语义不一致

### 现象

在 scan1 camera0、1600×1200、28,747 Gaussians 的独立训练中，初始 render 是一致的，但 step1 开始出现 SH 梯度漂移。进一步回溯发现：

- `T_final` 对齐；
- position / opacity / scale / rotation 梯度在 step1 对齐或为 0；
- 真正第一个稳定分裂点落在 raster backward replay 相关的 `n_contrib`；
- 差异先进入 `d_rgb`，再传到 SH 梯度。

### 根因

CUDA 的 `n_contrib` 不是 blended Gaussian 的数量，而是 tile candidate traversal 中“最后一个实际 blend 的候选项位置”。被 power/alpha 跳过的候选项虽然不 blend，但如果它们出现在最后 blend 候选项之前，仍然占据 traversal 位置。

之前 Vulkan/CPU 的 2D 路径把 `n_contrib` 当作“实际 blend 个数”来保存和 replay，于是 backward 从后向前回放时使用了错误的截止边界。

### 修复

已同步修改 2D Vulkan 和 CPU reference：

- `harmonyos_3dgs/src/vulkan/shaders/rasterize.comp`
  - 记录 candidate traversal position；
  - 存储最后一个实际 blend 的 candidate position。

- `harmonyos_3dgs/src/vulkan/shaders/rasterize_backward.comp`
  - 将 `n_contrib` 解释为 replay boundary；
  - backward 只回放到 CUDA 同义的最后贡献候选位置。

- `harmonyos_3dgs/src/cpu/rasterizer_cpu.cpp`
  - CPU forward cache 对齐 CUDA/Vulkan 2D 语义。

- `harmonyos_3dgs/src/cpu/rasterizer_backward_cpu.cpp`
  - CPU backward 按 replay boundary 回放。

- `harmonyos_3dgs/include/types.h`
  - 更新 `n_contrib` contract 注释。

- `harmonyos_3dgs/tests/test_rasterizer_backward_vulkan.cpp`
  - 更新 fixture 注释和断言语义。

### 验证

- `RasterizerBackwardVulkan.MatchesCPU_TinyFixture` PASS。
- 修复后 full CTest 通过，后续 eval_3D 和 CLI wiring 后最新基线为 `276/276 PASS`。
- scan1 VK↔CUDA PSNR 从 `90.91 / 22.42 / 8.24 dB` 改善到 `107.136986 / 49.822707 / 27.773822 dB`。

## 2. fair CLI 比对路径的配置问题

这些问题不是 scan1 P0 主因，但会让 CLI 级 fair comparison 看起来“比 focused harness 差很多”，必须排除，否则会误判为 shader 算法错误。

### 2.1 `sh_degree_warmup=0` 没有从 step1 生效

问题：CUDA golden 从 step1 使用 full SH，但 VulkanTrainer 的第一次 forward 没有正确尊重 `sh_degree_warmup=0`。

修复：`VulkanTrainer` 从第一步开始按配置启用 full SH。

验证：10/100-step parity harness 重新通过。

### 2.2 VK CLI projection matrix 与 CUDA/`camera_utils.cpp` 不一致

问题：CLI 路径中的 projection 构造和 CUDA reference 使用的 camera utility 不一致。

修复：`vk_train_main.cpp` projection 对齐 CUDA / `camera_utils.cpp`。

验证：fair 10-step comparison 对齐到 `VK vs CUDA 43.14 dB`，不再被 camera/projection 配置污染。

### 2.3 `model.free()` 过早释放 trainer 使用的数据

问题：`vk_train_main.cpp` 在构造/使用 trainer 前释放了 model storage，导致潜在悬挂指针风险。

修复：不再在 trainer 使用数据前调用 `model.free()`。

验证：focused build 和 10/100-step parity 测试通过。

### 2.4 render mode 不一致：CUDA 用 2D，VK render 默认硬编码 eval_3D

问题：之前 200-step fair report 中，CUDA final render 是 `eval_3D=false`，但 `gs3d_vk_render` 硬编码走 `eval_3D=true`，导致最终图像对比混入 render-mode mismatch。

修复：`gs3d_vk_render` 增加显式 `--eval_3d 0|1`，fair comparison 脚本传 `--eval_3d 0`。

验证：重新 render 后确认仍有训练轨迹差异，但不再是 render mode 混用造成的假差异。

## 3. 早期重要 bug：SH Adam DC/REST 分组布局错误

这个问题是更早一轮 10-step/trajectory parity 不稳定的主要原因。它不是 scan1 S13 的 P0 主因，但属于这轮 parity 工作中非常关键的已关闭问题。

### 现象

3-step trajectory 分析中，G[2] post-Adam SH 出现确定性的 `469%` relative diff。每次运行都是同一个 Gaussian、同一组元素出问题，说明不是 atomicAdd 随机噪声。

### 根因

3DGS 的 SH 使用两个 Adam group：

- DC：lr = `2.5e-3`
- REST：lr = DC / 20

CPU 侧 SH buffer 布局是 interleaved `[N,K,3]`。VulkanTrainer 之前用 `memcpy` 按 float index 把前 `N*3` 个 float 当成 DC，这在 `K=16` 时是错的：前 `N*3` float 实际包含 G[0] 的全部 48 个 SH 和 G[1] 的前 4 个 float，不是所有 Gaussian 的 DC。

结果是部分 Gaussian 的 DC 被放进 REST Adam group，用了 1/20 的学习率。

### 修复

在 `vulkan_trainer.cpp` 中增加并使用 gather/scatter helpers：

- `sh_gather_dc`
- `sh_gather_rest`
- `sh_scatter_dc`
- `sh_scatter_rest`

覆盖构造、oracle reset、gradient upload、post-Adam download、reallocate 等路径。

### 验证

- 3-step SH max-element rel diff：`469% → 0.015%`。
- step2 loss rel diff：`1.10e-3 → 5.5e-7`。
- 10-step trajectory 所有 group bounded。

## 4. scale/rotation 仍然增长的原因判断

100-step parity 里 scale/rotation 的梯度相对误差仍会增长：

- step100 `g_sca l2_rel = 4.517e-1`
- step100 `g_rot l2_rel = 8.811e-1`

目前判断：这不是已经定位到的公式移植错误，而是数值病态点被 Adam 放大的表现。

证据：

- step1 的关键链路大多对齐；
- rotation 在 identity quaternion + isotropic scale 初始化附近，解析梯度接近 0；
- CUDA actual kernel 会返回约 `1e-11` 的 rotation residual，而 VK 在某些位置抵消为 exactly zero；
- Adam 使用 `eps=1e-15`，会把这种近零 residual 转成可见 raw-rotation update。

已排除/负结果：

- conic off-diagonal convention 是成对约定，不能单独改；
- scalarizing `preprocess_backward.comp` 的 `W/J/T/Vrk/VT` 只让 drift 轻微变化，排除了简单 GLSL `mat3` layout 主因。

后续建议：不要直接改 shader 公式。下一步应该做 controlled same-state experiment，或者使用非退化初始化，先消除 near-zero rotation cusp。

## 5. eval_3D 相关进展

### 5.1 CLI smoke 的误判

问题：最初 `gs3d_vk_train --eval_3d 1` 只设置了 `RenderConfig::eval_3D`，没有设置 `VkTrainingConfig::eval_3D`。但 trainer 构造时是根据 `VkTrainingConfig` 专门化 preprocessor/rasterizer 的，所以那次 smoke 并没有构造真正的 eval_3D trainer。

修复：CLI 将 `--eval_3d` 同时传给 `RenderConfig` 和 `VkTrainingConfig`。

### 5.2 scan1 init PLY 缺少 `filter_3D`

问题：`/tmp/scan1_init_3dgs.ply` 没有 AAA `filter_3D` property，eval_3D forward 行为不完整。

修复：生成 `/tmp/scan1_init_3dgs_filter3d.ply`，追加 `filter_3D = min_valid_depth / max_focal * sqrt(0.3)`。

验证：白背景 eval_3D forward render 有 1,450,525 个非白像素，说明 forward 确实产生 transmittance。

### 5.3 eval_3D backward 缺失

问题：真正构造 eval_3D trainer 后，训练会到达 backward unsupported guard。

修复：新增独立 eval_3D backward 路径，而不是混入已修好的 2D shader：

- `harmonyos_3dgs/src/vulkan/shaders/rasterize_backward_eval3d.comp`
- `harmonyos_3dgs/src/vulkan/shaders/preprocess_backward_eval3d.comp`
- 对应 pass wrapper/header
- `RasterGradOutput` 增加 eval_3D 的 `d_gauss2screen`
- raster backward 输出 `d_rgb` / `d_opacity` / `d_gauss2screen`
- preprocess backward 消耗 `d_gauss2screen`，回传 raw pos/scale/rot/SH/opacity
- CUDA extension 增加非破坏性 debug binding `rasterize_gaussians_backward_dump_g2s`，因为原 Python API 内部会计算 `d_gauss2screen` 但不会返回，无法直接做中间量 parity

验证：

- `VulkanTrainer.Eval3DOneStepSmoke` PASS。
- `VulkanTrainer.Eval3DStepRequiresParityMode` PASS。
- `VulkanTrainer.Eval3DStep1RawGradientParity` PASS。
- tiny fixture raw gradients 与 CUDA 在 `2e-3` L2-relative / `2e-5` max-absolute 内。
- 直接中间量对比：
  - `d_rgb l2_rel=1.76e-3`, max_abs `1.09e-5`
  - `d_opacity l2_rel=1.52e-4`, max_abs `1.82e-6`
  - `d_gauss2screen l2_rel=5.63e-4`, max_abs `2.65e-5`

### 5.4 eval_3D non-parity path 暂时禁止训练

问题：默认 eval_3D forward 使用 Vulkan HEAD/sub-tile re-sort replay，而当前 backward 只验证了 CUDA-style parity replay。

修复：

- 内部 `VulkanTrainer::step` 对 `eval_3D && !parity_mode` 抛错；
- CLI 的 `--eval_3d 1` training 自动映射到 `parity_mode=true`；
- `forward_only` 仍可用于非 parity forward 诊断。

### 5.5 SH→position gradient 审计

问题：看起来 CUDA eval_3D 调用了 `computeColorFromSH`，似乎应该把 SH view-direction gradient 加到 position。

调查结论：CUDA eval_3D 后续 `computeGauss2Screen*` 使用赋值 `dL_dmean[idx] = ...`，覆盖了前面 `computeColorFromSH` 对 mean 的累加。因此 CUDA 实际上丢掉了这部分 SH→mean contribution。

实验：把 2D SH→position 逻辑加到 Vulkan 后，pos parity 从 `1.16e-3` 恶化到 `9.52e-2` L2-relative。

决定：Vulkan eval_3D 暂时保留 CUDA overwrite 行为，不移植这部分 2D 逻辑。

### 5.6 eval_3D shared memory 风险

问题：统一 forward shader 的 eval_3D path 中，`sub_order[16][256]` 使用 32-bit slot 时，shared memory 保守估计超过 32 KiB。

修复：把 sub-tile permutation packing 成每个 `uint` 里 4 个 8-bit batch index。

验证：

- `batch_size <= 256`，8-bit index 足够；
- packing 后保守 shared memory footprint 约 31.1 KiB；
- 子代理复核无 blocking 问题；
- full CTest `276/276 PASS`。

### 5.7 scan1 eval_3D smoke

命令记录在 `dev_notes/scan1_n_contrib_parity_s13.md`。

结果：scan1 camera0、filtered PLY、10-step eval_3D training 跑通：

- artifact: `/tmp/scan1_vk_eval3d_filter_10_postbwd/`
- loss: `0.665102 → 0.660058`
- final render PSNR vs JPEG-converted GT: `2.664990 dB`

这是 smoke，不是 CUDA eval_3D parity 结论。

## 6. 最新 100/1000-step fair comparison 结果

### 100-step basketball cam0

`VkVsCudaBasketball100Step.PerStepParity` PASS，并输出逐步 loss、gradient、backward chain、Adam m/v 诊断。

关键结果：

- step100 VK loss: `0.383244`
- step100 CUDA loss: `0.383523`
- abs diff: `2.796e-4`
- fair render：
  - CUDA-vs-GT: `7.96 dB`
  - VK-vs-GT: `7.94 dB`
  - VK-vs-CUDA: `54.51 dB`
  - gap CUDA−VK: `0.02 dB`

artifact: `/tmp/compare_fair_2000/`

### 1000-step basketball cam0

使用全新目录重跑，避免旧 artifact 混合：`/tmp/compare_fair_1000_20260429/`。

关键结果：

- CUDA final loss: `0.046733`
- VK final loss: `0.046796`
- CUDA-vs-GT: `21.05 dB`
- VK-vs-GT: `21.16 dB`
- VK-vs-CUDA: `28.71 dB`
- gap CUDA−VK: `-0.11 dB`

结论：1000 step 后两边最终 GT 质量基本一致，VK 甚至高 `0.11 dB`。但 VK-vs-CUDA render 只有 `28.71 dB`，说明训练轨迹仍分叉；这与 scale/rotation 近零梯度被 Adam 放大的判断一致。

## 7. 给确认主因用的判断

如果问题是：“为什么之前 scan1 10/200/1000 step VK↔CUDA 比对不通过、1000 step VK-vs-CUDA 掉到 8.24 dB？”

答案：**主因是 `n_contrib` 语义错误导致 backward replay 边界不一致。**

证据链：

1. 初始 render 对齐，说明不是 loader 或 camera 的第一层错误。
2. step1 `T_final` 对齐，但 `n_contrib` 不对齐。
3. step1 position/opacity/scale/rotation 梯度对齐或为 0，主要差异进入 SH。
4. CUDA 源码和实际 dump 表明 `n_contrib` 是 last blended candidate position，不是 blended count。
5. 修复该语义后，scan1 VK↔CUDA PSNR：
   - 10 step：`90.91 → 107.14 dB`
   - 200 step：`22.42 → 49.82 dB`
   - 1000 step：`8.24 → 27.77 dB`
6. 后续 100/1000-step basketball fair comparison 显示两边最终 GT 质量已经基本一致。

如果问题是：“为什么之前 CLI/fair comparison 看起来不可信或更差？”

答案：那主要是 harness/config 问题：full SH warmup、projection matrix、premature `model.free()`、render `eval_3D` mode 混用。这些已经修复。

如果问题是：“为什么 1000 step 后 VK 和 CUDA 仍不是完全同一张图？”

答案：目前没有新的确定公式 bug。剩余更像数值轨迹分叉：identity quaternion + isotropic scale 附近 rotation/scale 梯度极小，CUDA 的 `~1e-11` residual 与 VK 的 exact zero 会被 Adam `eps=1e-15` 放大。下一步应做 same-state / 非退化初始化实验，而不是直接改 shader 数学。

## 8. 当前状态

- 2D scan1 主因已修复。
- eval_3D backward parity-mode 路径已打通并有 tiny CUDA parity 测试。
- full CTest 最新基线：`276/276 PASS`。
- 100-step basketball fair comparison：VK-vs-CUDA `54.51 dB`，GT gap `0.02 dB`。
- 1000-step basketball fair comparison：VK-vs-CUDA `28.71 dB`，GT gap `-0.11 dB`。
- 1000-step basketball eval_3D comparison：使用 `/tmp/basketball_init_3dgs_filter3d.ply` 和 `eval_3D=True`，CUDA final loss `0.044052`，VK final loss `0.044153`，CUDA-vs-GT `21.35 dB`，VK-vs-GT `21.32 dB`，VK-vs-CUDA `28.47 dB`，GT gap `0.03 dB`。注意：当前 VK trained PLY 保存会丢失 `filter_3D`，所以 eval_3D 对比使用训练过程直接输出的 final render，而不是从 saved PLY 重渲染。
- 不应把剩余 1000-step render trajectory 分叉误判为新的已定位 porting bug；需要 controlled experiment 才能继续归因。
