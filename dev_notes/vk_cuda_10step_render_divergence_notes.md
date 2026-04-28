# VK/CUDA 10-step 渲染差异与 step-1 rotation 残差调查记录

日期：2026-04-28

## 背景

当前目标是定位 Vulkan 训练与 CUDA/PyTorch 参考训练在短程和长程训练后的差异来源。此前已对齐了一批 fair-path 配置：

- `sh_degree_warmup=0` 表示从 step 1 起使用完整 SH。
- `vk_train_main.cpp` 的相机/投影语义对齐 CUDA / `camera_utils.cpp`。
- `VulkanTrainer` 构造前不再提前 `model.free()`。
- `gs3d_vk_render` 支持严格 `--eval_3d 0|1`，fair render 使用 `eval_3D=false`。
- 10-step 与 100-step focused parity 测试包含 loss、梯度、参数、Adam `m/v`、以及 backward-chain diagnostics。

## 关键发现

### 1. 100-step 轨迹最早可观测分叉在 step 1 rotation 梯度

在 100-step 诊断中，step 1 的 rotation 链路表现为：

- CUDA `diag_pre_d_qn.npy` / `grad_rot.npy` 有 `~1e-11` 量级的非零残差。
- VK `captured_bwd_d_qn()` / `captured_grad_rotations()` 为精确 0。
- 由于 Adam `eps=1e-15`，CUDA 的 `~1e-12` 到 `~1e-11` raw rotation 梯度会在 step 1 被放大成 `~1e-3` 的 raw rotation 更新。

典型数值：

```text
CUDA step1 diag_pre_d_qn:
  nonzero = 1554
  max_abs = 3.406e-12
  l2      = 1.057e-11

CUDA step1 grad_rot:
  nonzero = 1554
  max_abs = 3.227e-12
  l2      = 1.101e-11

CUDA step1 param_rot - identity:
  max_abs = 9.997e-4
  l2      = 3.427e-2
```

Adam step1 公式可复现这个放大：

```text
param_new ≈ raw - lr * grad / (abs(grad) + eps)
max error vs saved CUDA param_rot.npy ≈ 2.64e-10
```

### 2. step-1 rotation 残差的比较来源是匹配的

已核对的导数层级：

- CUDA `diag_pre_d_qn.npy` 是 `_C.rasterize_gaussians_backward_dump` 返回的 `d_rotations`，对应 `dL/d(q_normalized)`。
- VK `captured_bwd_d_qn()` 是 `preprocess_backward.comp` 在 raw-quaternion normalization chain 之前写出的 `d_qn_*`，也对应 `dL/d(q_normalized)`。
- CUDA `grad_rot.npy` 是 PyTorch autograd 后的 `raw_rotation.grad`，对应 `dL/d(raw_rotation)`。
- VK `captured_grad_rotations()` 是 `grads.d_raw_rotations`，也对应 `dL/d(raw_rotation)`。

已核对的输入来源：

- CUDA 与 VK 都从 identity quaternion `(1,0,0,0)` 初始化。
- CUDA 与 VK 都使用 isotropic raw scale `log(0.03)`。
- CUDA 与 VK 都使用 `proper_ewa=false`、`eval_3D=false`、L1-only loss。
- CUDA diagnostic 的 second forward 在 step 1 与 autograd forward 完全一致：

```text
first vs second color max_abs = 0
first vs second color l2      = 0
radii first vs second neq     = 0
first L1 grad vs second sign grad max_abs = 0
```

因此，step 1 的 rotation 残差不是由 diagnostic second-forward 混用不同图像或不同 loss-gradient 导致。

### 3. step-1 rotation 残差不是稳定数学信号

使用完全相同的 CUDA forward buffers 和完全相同的 `dL_dout_color`，重复调用 `_C.rasterize_gaussians_backward_dump` 6 次，得到：

```text
dqn max_abs 每次约 2.95e-12 到 3.57e-12
相邻两次 dqn max_abs 差值约 4.07e-12 到 6.40e-12
```

这说明 CUDA step-1 rotation residual 是 identity quaternion + isotropic scale 退化点上的浮点消去 / 执行顺序 / atomic 级别 sub-ULP noise，不是稳定的语义梯度。

### 4. step-1 rotation 残差对 step2/step10 CUDA loss 几乎没有直接影响

用 CUDA step1 后参数作为 step2 forward 输入，只把 rotation 重置为 identity：

```text
recorded_step2_loss              0.5712935328
rerender_step2_normal            0.5712935328
rerender_step2_rotation_identity 0.5712935328
delta_vs_normal                  0.0
```

用 CUDA step9 后参数作为 step10 forward 输入，只把 rotation 重置为 identity：

```text
recorded_step10_loss              0.5666342378
rerender_step10_normal            0.5666342378
rerender_step10_rotation_identity 0.5666343570
delta_vs_normal                   +1.192e-7
```

从头训练 10 步，并在 step1 Adam 后中和 rotation 残差：

```text
step 2 delta after reset rotation param/state ≈ +5.96e-8
step 5 delta after reset rotation param/state = 0
step10 delta after reset rotation param/state = 0
```

结论：step1 的 CUDA rotation residual 会污染 raw rotation / Adam rotation state，但它不是 step2 或 step10 forward loss 差异的直接来源。

### 5. 10-step 渲染图确实已经肉眼可见不同，但需要区分输出来源

读取 `/tmp` 下已有 10-step 对比图后发现有两类结果。

#### 旧输出：`/tmp/compare_existing_10step`

```text
VK vs CUDA PSNR = 16.97 dB
MAE             = 0.07618
max_abs         = 0.6588
> 5/255 pixels  = 51.18%
>25/255 pixels  = 31.55%
```

肉眼现象：一侧仍像稀疏点云/暗图，另一侧已有模糊篮球轮廓。

这个目录很可能是 stale 或来自旧配置，不能作为当前 fair-path 修复后的权威比较。

#### fair-path 修复后输出：`/tmp/compare_fair_10_after_fix`

```text
VK vs CUDA PSNR = 43.03 dB
MAE             = 0.00318
max_abs         = 0.0706
> 5/255 pixels  = 4.96%
>10/255 pixels  = 1.38%
```

肉眼现象：两边都已经是模糊篮球轮廓，但亮度、模糊程度、局部点分布仍可见差异。

因此需要修正之前过窄的说法：

- “step1 rotation residual 对 CUDA 自己 step10 forward loss 基本无影响”成立。
- “10-step VK/CUDA 渲染图已经有可见差异”也成立。
- 这两个事实不矛盾；图像差异不能直接归因于 step1 rotation residual。

## 已排除的问题

1. **SH DC/REST Adam group layout bug**
   - 已修复。此前 SH interleaved `[N,K,3]` 被错误按连续 float 切 DC/REST，导致不同 Gaussian 的 DC/REST 落入错误 Adam group。
   - 修复后 10-step SH/Adam 轨迹大幅收敛。

2. **Adam timing / group order 独立错误**
   - 当前证据显示 Adam `m/v` 主要跟随梯度差异，没有独立 timing 或 group-order bug 的证据。

3. **conic off-diagonal 单边 2x 问题**
   - VK raster backward 的 full `d_conics[1]` 与 preprocess backward 的 full-parameter convention 是成对的。
   - CUDA 写 half offdiag 并在后续 symmetric chain 中补 2x。
   - 不能单独改其中一边。

4. **GLSL `mat3` layout 是主要原因**
   - scalarizing `preprocess_backward.comp` 中 `W/J/T/Vrk/VT` 后，100-step scale/rotation drift 只轻微变化，不是主因。

5. **CUDA diagnostic second-forward 输入混用**
   - step1 已验证 second forward 的 color/radii/L1 grad 与 autograd forward 完全一致。

6. **step1 rotation residual 直接造成 step2/step10 loss 差异**
   - 已通过 CUDA 内部 paired controls 排除。

## 当前未解决问题

1. **10-step fair render 仍有可见图像差异**
   - fair-after-fix 图像 PSNR 约 43 dB，局部差异仍可见。
   - 需要确认差异来自：
     - 训练 step 内部 forward image 已经不同；
     - 训练参数差异导致 render 不同；
     - PLY save/load 转换差异；
     - render CLI 与 trainer forward 配置仍有差异；
     - 或旧 `/tmp` 输出污染判断。

2. **100-step scale/rotation 梯度和 Adam moment drift**
   - 100-step 时 `g_sca/g_rot/m_sca/m_rot/v_sca/v_rot` drift 明显。
   - 但 step1 rotation cusp 已确认是 CUDA near-zero noise，不应直接按 VK math bug 修。

3. **VK/CUDA forward baseline 仍未达到目标**
   - 已有 VK↔CUDA same-ply forward PSNR 约 54.84 dB，目标 ≥60 dB。
   - Gate_P2–P7 / Gate_L1 仍需要继续实现和排查。

## 推荐解决方案 / 下一步

### A. 先建立权威 10-step 图像比较，清除 stale 输出干扰

不要继续使用 `/tmp/compare_existing_10step` 作为当前结论依据。应重新生成一个带时间戳的新目录，例如：

```bash
python harmonyos_3dgs/tools/compare_vk_cuda_fair.py \
  --steps 10 \
  --outdir /tmp/compare_fair_10_YYYYMMDD_HHMMSS
```

要求记录：

- VK/CUDA training command。
- VK/CUDA render command。
- `eval_3D`。
- `proper_ewa`。
- camera id / image name。
- 输出 PLY 路径。
- 渲染图 PSNR/MAE/max_abs。

### B. 区分“训练内部 forward image”与“导出后 render image”

在 `VulkanTrainer::step()` 和 CUDA dumper 中保存 step10 训练内部 forward image：

- CUDA：autograd `rendered`，即 loss 使用的 tensor。
- VK：`VulkanTrainer` forward 后的 `image_`。

比较四张图：

1. CUDA train-forward step10 image。
2. VK train-forward step10 image。
3. CUDA final PLY render image。
4. VK final PLY render image。

判定：

- 如果 1 vs 2 已明显不同：问题在训练 forward / 参数。
- 如果 1 vs 2 接近，但 3 vs 4 不同：问题在 PLY export/import 或 render CLI。
- 如果 VK train-forward vs VK render 不同：问题在 VK trainer/render config mismatch。
- 如果 CUDA train-forward vs CUDA render 不同：问题在 CUDA save/render path。

### C. 对 step1 rotation cusp 做控制实验，而不是直接改 shader math

可选实验：

1. **中和 near-zero rotation gradient**
   - 在 CUDA 和 VK 对照中，把 `abs(g_rot) < threshold` 的 rotation 梯度置 0。
   - 观察 10/100-step drift 是否显著下降。

2. **非退化初始化**
   - 使用轻微非 identity rotation 或 anisotropic scale。
   - 避开 identity quaternion + isotropic scale 的 analytical-zero cusp。
   - 如果 drift 消失，记录为 Adam-amplified degeneracy policy 问题。

3. **提高 Adam eps 控制实验**
   - 仅用于诊断，不作为最终实现。
   - 例如测试 `eps=1e-8` 是否消除 step1 rotation state 放大。

### D. 继续实现 forward parity gates

当前 same-ply forward 仍约 54.84 dB，建议继续 Gate_P2–P7：

- Gate_P2: ConicOpacity。
- Gate_P3: SH-evaluated RGB。
- Gate_P4: Radii。
- Gate_P5: SortedIds。
- Gate_P6: TFinal/NContrib。
- Gate_P7: RenderedImage。

这比继续追 step1 rotation residual 更可能解释 10-step 图像可见差异。

## 当前结论

1. step1 rotation residual 的来源已经确认：它是 CUDA 在退化初始化点上的 nondeterministic sub-ULP cancellation noise，VK 为精确 0。
2. Adam `eps=1e-15` 会把这个 near-zero 残差放大到 raw rotation state，但该影响在 step2/step10 forward loss 上几乎不可见。
3. 10-step render 图像确实已有可见差异；但旧 `/tmp/compare_existing_10step` 很可能是 stale/非 fair 输出，不能作为当前修复后的主证据。
4. 当前应把诊断重心转向权威 fresh fair render、训练内部 forward image vs 导出 render image、以及未完成的 forward parity gates，而不是把 step1 rotation residual 当作 VK 数学 bug 修掉。
