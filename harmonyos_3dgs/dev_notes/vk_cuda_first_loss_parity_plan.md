# VK vs CUDA Training 首次 Loss Parity Plan (S10, 2026-04-24)

## 目标
相同输入（basket-aaa PLY + cam0 + COLMAP GT 图）下，VK 训练第一次 forward+loss 与 CUDA 参考 rel_diff < 1e-4。

## 配置（双方统一）
- `eval_3D = true`（VK 需要从硬编码 false 改成可配置）
- `sort_mode = 0`（CUDA 覆盖默认 3；VK 需要 `parity_mode` 关掉子 tile 重排）
- `near_clipping = false`
- `lambda_dssim = 0.0`（Phase 1/2 纯 L1）
- `background = (0,0,0)`

## 固定 Fixture
- PLY: `/home/robota/h00813233/Graph/aaags-claude/basket-aaa.ply` (400K Gaussian, SH3)
- Cameras: `/home/robota/Downloads/basketball/_sp0_dump_output/cameras.json`
- Cam0: 720×960, img_name=78899858295079
- GT: `/home/robota/Downloads/basketball/images/78899858295079.jpg`（COLMAP 原图）
- CUDA env: `conda run -n aaa-gs` (AAA DGR fork 已装)

## 交付物路径
- CUDA dump: `harmonyos_3dgs/tools/dump_cuda_training_step.py`
- Golden: `harmonyos_3dgs/tests/golden/basketball/cuda_ref/step_0001/`
- VK test: `harmonyos_3dgs/tests/test_vk_vs_cuda_first_loss.cpp`

## Phase 0 执行（当前）

### 0.1 VK 配置改造 + 0.2 中间态 capture（合并，避免 merge conflict）
文件:
- `include/train_types.h`: `VkTrainingConfig` 新增 `bool eval_3D = true`, `bool parity_mode = false`
- `src/vulkan_trainer.cpp:31`: 从 `tcfg.eval_3D` 读
- `src/vulkan/shaders/rasterize.comp`: 加 spec const `spec_disable_subtile_resort`
- `src/vulkan/rasterize_pass.cpp`: 传递 spec const
- `include/vulkan_trainer.h`: 新增 `enable_intermediate_capture()` + accessor 族
- `src/vulkan_trainer.cpp`: 在 step() 中 readback GPU 中间态（仅当 capture enabled）

### 0.3 CUDA dump 脚本
输出到 `tests/golden/basketball/cuda_ref/step_0001/`:
- `view_matrix.npy`, `proj_matrix.npy`, `config.json`
- `means2D.npy`, `conic_opacity.npy`, `rgb_colors.npy`, `radii.npy`
- `sorted_ids_per_tile.npy`
- `T_final.npy`, `n_contrib.npy`
- `rendered_image.npy` (CHW)
- `gt_image.npy` (CHW, 来自 COLMAP png)
- `l1_loss.npy`

### 0.4 VK 对比测试骨架
`tests/test_vk_vs_cuda_first_loss.cpp`:
- load golden
- run VK with parity config
- Phase 1 Gates I1-I4（输入）
- Phase 2 Gates P1-P7（forward 分阶段）
- Phase 3 Gates L1-L3（loss）

## 零回退约束
Phase 0.1 改完后，`./build/gs3d_vk_tests --gtest_filter=VkVsCudaBasketball.*`（当前 54.8 dB 基线）**必须零回退**。
`parity_mode=false` 时行为应与改前完全一致。

## 风险
- basket-aaa SH degree 3 + 40万 Gaussian：dump 文件较大（means2D~3MB, rgb~5MB, sorted_ids 视 tile 数而定）
- FP 顺序差：期望 Phase 2 P5 之前 PSNR 能推到 70+ dB；Phase 2 P5 本身允许有 FP 顺序差（< MSE 1e-5）
