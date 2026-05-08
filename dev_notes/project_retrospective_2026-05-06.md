# harmonyos_3dgs 项目阶段复盘

> 时间范围：Vulkan 迁移与 VK↔CUDA parity 闭环推进阶段  
> 复盘目标：按时间顺序沉淀整体项目进展、关键里程碑、突破原因，以及后续团队可以复用的方法论。

## 总览

这个阶段的核心目标不是单纯“让 Vulkan 版本能跑”，而是把 AAA-Gaussians / 3DGS 的 CUDA 参考实现逐层迁移到 C++/Vulkan，并且建立可证明的 parity 闭环：

1. 先搭基础设施，让工程可构建、可测试、可重复运行。
2. 再完成前向渲染，让 Vulkan 能产出可比较图像。
3. 再搭反向传播和训练框架，让端到端训练能跑通。
4. 同时建设 dump / golden / intermediate-state 设施，把黑盒差异拆成可观测证据。
5. 逐层完成反向传播、Adam、训练轨迹、保存/重渲染和全数据集比较。
6. 最后打开 CUDA intermediate-state gate，把输入、前向中间态、最终图像、标量 loss 都纳入门控。

整体突破来自一条主线：**先明确语义合同，再建立可观测性，最后用严格 gate 收敛差异**。

---

## 时间标尺

> 说明：部分早期基础设施工作没有单独日期标签，以下时间按代码注释、golden、session state 和当前阶段证据归并，用于团队复盘时建立先后关系。

- **早期阶段：基础设施与 Vulkan forward 起步** — 建立 C++17/CMake/Vulkan compute 骨架、测试体系、CPU/Python/CUDA/VK 比较入口。
- **2026-04-21 前后：篮球 cam0 forward 基线快速抬升** — proper EWA、k-buffer、tile culling 等前向语义修正把 VK↔CUDA 从早期低 PSNR 推到 40+dB 量级。
- **2026-04-24 前后：eval_3D near-plane / table ROI 关键突破** — 移除与 CUDA 配置不一致的 near-plane cull 后，篮球 cam0 same-ply forward 达到约 54.84 dB，bottom-left table ROI 明显闭合。
- **2026-05-01 前后：端到端训练与全数据比较闭环** — MCMC densification、terminal no-update、filter_3D、view schedule、proper_ewa / sh_degree CLI 等训练契约完成，full-dataset/feature-matrix gate 进入可复用状态。
- **2026-05-06：CUDA intermediate-state gate 全开** — first-loss I1-I4、P1-P7、L1 全部 PASS，render parity 达到 103.72 dB，scalar L1 diff 约 4.69e-7。

## 时间线与关键里程碑

| 时间/阶段 | 阶段 | 核心成果 | 代表性证据 | 突破原因 |
|---|---|---|---|---|
| 早期 | 1. 基础设施搭建 | C++/CMake/Vulkan 工程骨架、测试基线、核心 Vulkan 封装 | VulkanContext/Buffer/Shader/Pipeline 可用，测试体系稳定增长 | 先把“可构建、可测试、可重复”的底座打稳 |
| 2026-04 中下旬 | 2. 前向渲染完成 | preprocess + sort + rasterize 前向链路跑通 | CPU↔VK 后续达到 99.15 dB；篮球 VK↔CUDA 基线推进到 54.84 dB | 将不可见图形误差转化为 PSNR/中间态误差 |
| 2026-04 下旬 | 3. 反向传播框架完成 | raster/preprocess backward、ForwardCache、VulkanTrainer、Adam 接入 | 训练可端到端执行，loss 可下降 | 先搭完整链路，再逐层校验梯度来源 |
| 2026-04 下旬至 2026-05-06 | 4. dump 设施完成 | CUDA/Vulkan 中间态 dump、golden、manifest、first-loss gate 梯子 | I1-I4、P1-P7、L1 逐步打开 | 把黑盒训练差异拆成输入、前向、raster、render、loss 证据 |
| 2026-04 下旬至 2026-05 初 | 5. 反向传播比较完成 | 梯度、Adam 后参数、短轨迹比较闭环 | SH layout bug 修复后 step2 loss diff 1.10e-3 → 5.5e-7 | 用逐层 dump 排除猜测，定位真实 layout/语义错位 |
| 2026-05-01 前后 | 6. 端到端训练完成 | MCMC densification、final-step no-update、proper_ewa、view schedule、全数据训练契约 | 5000 步 cam0 MCMC 计数 24680/24680；全 76 视角 1000 步计数 3513/3513 | 统一训练、保存、重渲染、GT 选择和 schedule 语义 |
| 2026-05-06 | 7. 基本比较完成 | 单视角、全数据、feature matrix、first-loss gate 全链路比较 | P7 render parity 103.72 dB；L1 scalar diff 约 4.69e-7；first-loss 12/12 PASS | 从结果比较升级为过程比较和 gate 化回归 |

---

## 1. 基础设施搭建

### 完成内容

项目最早的关键任务是把 HarmonyOS/Vulkan 目标拆成可持续开发的工程结构：

- C++17 + CMake 构建体系。
- Vulkan compute 基础封装：context、buffer、shader module、pipeline、descriptor 等。
- 测试体系和 golden fixture 结构。
- CPU / Python / CUDA / Vulkan 多实现之间的比较入口。

### 关键意义

这个阶段的价值在于建立了后续所有 parity 工作的“地基”：

- 每次改动可以快速 rebuild / ctest。
- 每个里程碑都能被测试数量、golden、PSNR 或中间态误差证明。
- Vulkan 代码不再是一次性实验，而是可被 review、可回归、可扩展的工程模块。

### 突破原因

突破不是来自某个 shader 优化，而是来自**工程闭环先行**：先保证代码能长期迭代，再追求数值一致性。

---

## 2. 前向渲染完成

### 完成内容

前向渲染阶段完成了 Vulkan 版 3DGS 的核心 forward pipeline：

- `preprocess.comp`：坐标变换、协方差、投影、半径、颜色、tile 覆盖等。
- sort/binning：按 tile 和 depth 构建渲染顺序。
- `rasterize.comp`：alpha blending、transmittance、贡献记录、图像输出。

### 代表性结果

- CPU↔VK 前向渲染最终达到约 **99.15 dB**，说明 CPU/Vulkan 同语义 forward 基本闭合。
- 篮球 cam0 VK↔CUDA same-ply forward 达到 **54.84 dB**，距离 60 dB 目标只剩约 5.16 dB。
- 后续 P1-P7 gate 进一步确认：大量剩余差异不是“整条 forward 坏了”，而是集中在可解释的边界、排序、early termination 和 rare-pixel drift 上。

### 关键突破

1. **proper_ewa / dilation 语义对齐**  
   早期 PSNR 差距并不只是 shader 精度问题，而是 `proper_ewa` 相关配置与 CUDA golden 实际生成配置不一致。修复后 CPU↔VK forward 大幅闭合。

2. **eval_3D parity mode 明确化**  
   CUDA eval_3D 默认仍保持 GLOBAL sort/no tile culling，而 Vulkan 默认 eval_3D 曾使用 StopThePop tile-depth 语义。引入 parity mode 后，比较目标变得明确。

3. **n_contrib 语义修正**  
   CUDA 的 `n_contrib` 是最后参与 blending 的 candidate position/replay boundary，不是简单 blended count。这个修正同时影响 forward 和 backward replay。

---

## 3. 反向传播框架搭建完成

### 完成内容

反向传播阶段完成了：

- `rasterize_backward.comp`
- `preprocess_backward.comp`
- forward cache / backward cache
- raw parameter gradient 回传
- GPU Adam 更新
- `VulkanTrainer::step()` 训练主循环

这标志着 Vulkan port 从“只能渲染”进入“能训练”的阶段。

### 关键挑战

反向传播的难点不是单个公式，而是链条很长：

```text
image loss
  → raster backward
  → d_rgb / d_opacity / d_conic / d_means2D / d_depth / d_gauss2screen
  → preprocess backward
  → raw position / scale / rotation / SH / opacity
  → Adam
  → next-step render
```

任何一处语义、布局或 reduction 顺序错位，都会在后续训练中被 Adam 放大。

### 突破原因

突破来自“框架先完整，再局部验证”：

- 先让 end-to-end step 能执行。
- 再把每个梯度 group、每个 Adam group、每个 step 的状态拆开对比。
- 避免只看最终 loss，因为最终 loss 很容易掩盖中间梯度问题。

---

## 4. dump 设施完成

### 完成内容

dump 设施是整个项目后半段的关键转折点。它把 CUDA/Vulkan 的比较从“最终图像差多少”升级成“每一层状态差多少”：

- 输入 gate：view matrix、projection matrix、raw params、config flags。
- 前向中间态 gate：means2D、conic opacity、RGB colors、radii、sorted IDs。
- raster state gate：`T_final`、`n_contrib`。
- render/loss gate：`rendered_image.npy`、`l1_loss.npy`。

### 已打开的 first-loss gate

| Gate | 含义 | 当前状态 |
|---|---|---|
| I1 | ViewMatrix | PASS |
| I2 | ProjMatrix | PASS |
| I3 | RawParams | PASS |
| I4 | ConfigFlags | PASS |
| P1 | Means2D | PASS |
| P2 | ConicOpacity | PASS |
| P3 | RgbColors | PASS |
| P4 | Radii | PASS |
| P5 | SortedIds | PASS |
| P6 | TFinalNContrib | PASS |
| P7 | RenderedImage | PASS |
| L1 | L1Loss | PASS |

### 关键突破

1. **artifact 缺失从 skip 改为 assert**  
   如果 golden dump 目录存在但关键 artifact 缺失，测试应失败而不是跳过，避免假绿。

2. **从 count budget 升级为 identity/evidence budget**  
   P6 不只断言 27 个 set-difference tile，而是锁定具体 tile IDs，防止“一个旧问题消失、一个新问题出现但数量不变”。

3. **从图像差异追溯到 raster 证据**  
   P7 的 rare pixel drift 不是孤立接受，而是与 P6 的 `T_final` / `n_contrib` 边界漂移证据相连。

---

## 5. 完成反向传播的比较

### 完成内容

反向传播比较从单步梯度扩展到多层次：

- step1 gradient/loss parity。
- per-group gradient comparison。
- post-Adam parameter comparison。
- step3 / step10 / step100 trajectory diagnostics。
- eval_3D backward smoke 和 parity-mode backward replay。

### 关键 bug：SH Adam group layout

最重要的突破之一是 SH Adam group 布局问题：

- 3DGS 的 SH 参数是 `[N, K, 3]` interleaved。
- Adam 分组中 DC 和 REST 使用不同学习率。
- 原实现按连续 float memcpy 拆 DC/REST，导致 DC/REST 实际错位。
- 修复为显式 gather/scatter DC/REST。

### 代表性结果

- 3-step SH 最大元素相对误差：**469% → 0.015%**。
- step2 loss 相对差：**1.10e-3 → 5.5e-7**。
- 10-step trajectory 中所有主要 group 都被约束到可接受范围。

### 关键方法论

这次突破的关键不是“怀疑 atomicAdd”，而是反过来证明：

- 若 drift 是 deterministic 且每次同一个 Gaussian 出问题，通常不是 atomic noise。
- L2 norm 可能掩盖单元素巨大偏差，所以必须加入 per-element max relative / abs diagnostics。
- 先定位第一处分叉，再修对应语义，不做大范围猜测式修改。

---

## 6. 端到端训练完成

### 完成内容

端到端训练阶段解决的是“训练过程整体语义一致”：

- `VulkanTrainer::step()` 完整 forward/backward/Adam。
- terminal no-update step：匹配 CUDA harness 最后一次 forward/backward 但不 optimizer step 的语义。
- MCMC densification：relocate/add/growth、Adam slot preserve/zero、opacity reset。
- `filter_3D` 在 eval_3D + MCMC + PLY 保存中的传播。
- `--view_schedule` 和 `--require_all_gt 1`：确保 CUDA/Vulkan 多视角训练使用同一 camera 序列与 GT 集合。
- `--proper_ewa` / `--sh_degree`：训练、保存、重渲染配置一致。

### 代表性结果

- 5000-step 篮球 cam0 MCMC：CUDA/VK final count **24680 / 24680**。
- 5000-step eval_3D + proper_ewa + MCMC：VK saved vs CUDA **29.37 dB**，VK saved vs train **59.04 dB**。
- 全 76 视角 1000-step eval_3D + proper_ewa + MCMC：CUDA/VK final count **3513 / 3513**，VK saved vs CUDA **27.99 dB**，VK saved vs train final view **59.24 dB**。

### 关键突破

一个典型问题是“训练 final render 好，但保存 PLY 再渲染坏”。最后证明根因不是 PLY 序列化，而是 render config 不一致：

- training 使用 `proper_ewa=false`。
- standalone render 默认走了不同配置。
- 加入 `--proper_ewa 0|1` 后，同一 saved PLY 以匹配配置重渲染，可与 training-final render 达到 **74.68 dB**。

这说明端到端训练不只要比较 loss，还要比较：

```text
training final render
saved PLY
standalone render config
GT/camera/view schedule
final-step optimizer semantics
```

---

## 7. 基本比较完成

### 完成内容

项目已经形成多层比较矩阵：

1. CPU↔VK same-ply forward。
2. VK↔CUDA same-ply forward。
3. VK↔CUDA first-loss intermediate gates。
4. VK↔Python / VK↔CUDA backward gradients。
5. post-Adam 参数比较。
6. 10/100/1000-step training trajectory。
7. cam0 与 full-dataset 训练比较。
8. feature matrix：DSSIM、LR decay、SH warmup 等训练功能开关。

### 代表性结果

- CPU↔VK forward：约 **99.15 dB**。
- 篮球 cam0 VK↔CUDA same-ply forward：约 **54.84 dB**。
- P7 rendered image parity：**103.724446 dB**。
- Gate_L1 scalar loss：VK/CUDA scalar abs diff 约 **4.69e-7**，first-loss gate **12/12 PASS**。
- 全数据 feature matrix：baseline / DSSIM / LR decay / SH warmup 均通过 count 与 PSNR gate。

### 关键突破

“基本比较完成”的意义不是所有数值都 bit-exact，而是：

- 哪些差异来自可解释边界条件已经被标注。
- 哪些差异来自 reduction / float accumulation 已经被单独 budget。
- 哪些路径已经进入固定 gate，未来回归会被自动捕获。
- 剩余工作可以明确聚焦到更高层训练质量或特定 numerical drift，而不是重新怀疑基础设施。

---

## 我认为最关键的注意点

### 1. 不要只看最终图像或最终 loss

最终图像好不代表中间状态对；最终图像坏也不代表训练坏。必须拆成：

```text
input → preprocess → bin/sort → raster state → rendered image → scalar loss → backward → Adam → trajectory
```

每一层都需要自己的证据。

### 2. 语义差异比浮点误差更常见

很多“大误差”最后不是 Vulkan 精度问题，而是语义不一致：

- `proper_ewa` 配置。
- eval_3D parity mode vs tile-depth mode。
- `n_contrib` 是 replay boundary，不是 count。
- final iteration 是否执行 Adam。
- saved PLY render config 是否匹配 training config。
- SH DC/REST Adam group layout。

### 3. L2 norm 不足以证明 parity

L2 可能被大多数正常元素稀释，掩盖单个元素的巨大错误。后续 gate 应同时看：

- max abs。
- max relative。
- first bad index。
- bad count。
- 稳定身份（tile id / gaussian id / pixel id）。

### 4. “预算”必须有证据来源

可以接受 bounded drift，但必须说明：

- drift 是否集中在边界。
- 是否与上游 gate 证据一致。
- 数量、位置、最大值是否固定。
- 如果 drift 移动，测试是否会失败。

P6/P7 的做法就是：不仅限制数量，还锁定 tile IDs / first bad / max drift index。

### 5. dump 设施是项目加速器

没有 dump 时，只能猜；有 dump 后，可以问更精确的问题：

- 输入是否一致？
- sort key 是否一致？
- active set 是否一致？
- raster replay boundary 是否一致？
- render 差异是否来自前一层已知 drift？

这是从“调 shader”升级到“做系统工程”的关键转折。

---

## 可以改进的地方

### 1. 更早建立机器可读的 milestone metrics

这次很多关键结果已经保存在测试输出、captains log 和 session state 里，但早期阶段仍有部分指标依赖叙述性记录。后续建议每个阶段都固定输出一份 JSON/CSV metrics，包括：commit、数据集、camera、配置、PSNR、loss、bad count、最大误差位置和运行命令。

### 2. 把 golden 生成配置固化为 artifact contract

`proper_ewa`、`eval_3D parity mode`、near clipping、SH degree、background、training/eval mode 等配置曾多次成为差异来源。后续 golden 目录应强制带完整 manifest，并在测试里 assert manifest 与当前运行配置一致，避免“图像 golden 是对的，但生成语义不可见”。

### 3. 中间态 gate 可以更早铺设

forward 阶段早期主要依赖最终图像 PSNR，直到 dump 设施完善后才真正加速定位。以后迁移新模块时，应优先放下 input / preprocess / sort / raster / render / scalar loss 的最小 gate，即使预算很宽，也比只看最终图像更容易收敛。

### 4. Diagnostic harness 与 regression gate 需要分层

这次有些测试既承担诊断输出，又承担回归门禁，容易让预算、日志和断言混在一起。后续可以把 harness 分成两类：一类是固定、安静、稳定的 CI gate；另一类是高信息量诊断工具，允许输出 top-k、heatmap、raw dump 和空间分解。

### 5. 对“可接受 drift”保留身份级证据

P6/P7 后期已经从 count budget 升级到 tile id / pixel id / first bad / max index。这个做法应前移到所有 numerical budget：只接受有固定身份、固定范围、固定上游解释的 drift；否则 count 不变也可能掩盖问题迁移。

### 6. 更早做配置矩阵而不是单点追误差

saved PLY 白块、proper EWA mismatch、full-dataset schedule 等问题都说明单点实验容易误判。后续遇到 gap 时，优先跑小型配置矩阵：CUDA/VK、training/render、proper_ewa on/off、parity/default、single/full dataset，把“配置错”先排除掉。

### 7. 将复盘产物纳入团队知识库

本复盘不应只是阶段总结，还应拆出可执行条目：默认 gate 模板、golden manifest 模板、dump artifact 命名规范、review checklist、常见语义错位清单。这样下一次迁移新 kernel 或新训练功能时，团队可以直接复用流程。

---

## 当前状态与后续方向

当前已经完成：

- Vulkan forward/backward/training 主链路。
- MCMC densification 与 full-dataset training harness。
- first-loss gate 从 I1 到 P7/L1 全部打开，当前 **12/12 PASS**。
- 端到端训练、保存、重渲染和多视角比较闭环。

后续重点建议：

1. 将 first-loss ladder 作为长期回归门禁保留，当前 I1-I4、P1-P7、L1 已全部 PASS。
2. 继续把已知 numerical drift 与训练轨迹 drift 分层归因。
3. 对 full-dataset / feature matrix 的关键配置保留机器可读 metrics，形成团队长期基线。
4. 把“语义合同 + dump 证据 + gate budget”作为后续所有 Vulkan/CUDA parity 工作的默认流程。
