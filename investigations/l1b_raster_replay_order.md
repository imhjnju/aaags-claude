# L1b Raster Replay Order Evidence

## Scope

Remaining L1b same-ply basketball cam0 VK↔CUDA forward gap after first-loss intermediate gates P1-P7/L1 are closed.

## Observations

- Top-level config matches the CUDA golden for `eval_3D=true`, `proper_ewa=true`, `parity_mode=false`.
- CUDA current render is bit-exact with the stored basketball cam0 golden.
- Touched Gaussian identity, tile ranges, candidate sets, RGB for touched Gaussians, `n_contrib`, and `T_final` are not sufficient to explain the highest-error pixels.
- High-error pixels exist where tile-level candidate order/count/transmittance are effectively identical but RGB differs, pointing inside the eval_3D replay cascade.

## Reference evidence

CUDA production basketball settings use hierarchical sort mode with eval_3D enabled and queue sizes equivalent to:

- `HEAD_WINDOW = 4`
- `MID_WINDOW = 8`
- `CULL_ALPHA = true`
- `EVAL_3D = true`

The CUDA dispatch path selects `sortGaussiansRayHierarchicalCUDA_forward<NUM_CHANNELS, HEAD_QUEUE_SIZE, MID_QUEUE_SIZE, HIER_CULLING, EVAL_3D, ...>` from `AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`.

The CUDA replay implementation in `stopthepop/hierarchical_render.cuh` uses a full cascade:

- per-4x4 TAIL queues,
- 2x2 MID queues,
- per-pixel HEAD queues,
- hierarchical 4x4 alpha/frustum culling,
- tail overflow draining through MID into HEAD.

## Vulkan evidence

`harmonyos_3dgs/src/vulkan/shaders/rasterize.comp` production eval_3D path documents itself as a simplified approximation:

- MID queue is elided.
- 4x4 alpha culling is not applied in the sub-tile sort step.
- streaming tail overflow is replaced by full-batch sort and sweep.
- the comment says `HEAD_W=4 matches CUDA exactly`, but the code currently sets `HEAD_W = 8`.

## Current hypothesis

The remaining L1b gap is dominated by production eval_3D replay-order mismatch rather than preprocess, binning, or top-level tile sort.

## Experiment: HEAD window alignment

Changing only Vulkan `HEAD_W` from 8 to 4 was tested because CUDA production settings use `HEAD_WINDOW=4`.

Result:

- Build succeeded and recompiled `rasterize.comp`.
- `VkVsCudaBasketball.Cam0_PsnrAtLeastBaseline` regressed to **52.074 dB** versus the prior **~54.73 dB** baseline.
- The change was reverted.

Conclusion: Vulkan's wider `HEAD_W=8` is compensating for the missing CUDA MID/tail-drain cascade. Matching the HEAD window alone is not a valid fix; the next diagnostic should instrument or emulate the CUDA TAIL→MID→HEAD replay order rather than tuning HEAD width in isolation.

## Experiment: near-depth HEAD tie tolerance

For worst pixel `(453,52)`, VK replay order differs from tile order by a high-alpha near-depth pair:

- VK replay: `... 260620, 218837, 182148, 355888 ...`
- tile/CUDA-matching order: `... 260620, 182148, 218837, 355888 ...`

Offline replay with VK preprocess dumps shows:

- VK replay sequence reconstructs the VK pixel within `~5e-7` RMSE.
- tile order reconstructs the CUDA pixel for this probe.

Changing the HEAD insertion predicate from `id < kbuf_depth[i]` to `id + 1e-6 < kbuf_depth[i]` was tested to preserve stream order for near-equal depths.

Result:

- `VkVsCudaBasketball.Cam0_PsnrAtLeastBaseline` regressed to **51.245 dB**.
- The change was reverted.

Conclusion: the probe proves the gap is replay-order sensitive, but a global fixed epsilon is too broad and disrupts legitimate depth sorting. The next step is to quantify replay-order differences across high-error pixels and then match CUDA's hierarchical queue semantics, not add a tolerance heuristic.

## Top-error replay-order quantification

After reverting both experiments, build plus `VkVsCudaBasketball.Cam0_PsnrAtLeastBaseline:VkVsCudaBasketball.Cam0_NContribDump` passed and restored the baseline:

- `Cam0_PsnrAtLeastBaseline`: **54.730 dB**.
- Probe `(453,52)` VK replay count: `14`.
- Probe VK replay GIDs: `145552 397629 166619 221332 283819 289947 125398 175573 236196 260620 218837 182148 355888 255866`.

Offline replay over the top 30 VK-vs-CUDA error pixels using VK preprocess dumps showed:

- VK replay sideband reconstructs VK color in **30/30** top-error pixels.
- raw tile order is closer to CUDA than VK replay in **25/30** top-error pixels.
- raw tile order exactly reconstructs CUDA in **1/30** top-error pixels.

Conclusion: the remaining gap is not a shader math/color/preprocess issue at these pixels; it is the order in which valid contributions are blended. CUDA is also not equivalent to raw tile order globally. The likely missing semantic is CUDA's streaming TAIL→MID→HEAD queue/drain cadence.

## Experiment: offline MID replay simulator v1

Added `harmonyos_3dgs/tools/diagnose_l1b_mid_replay.py` as an offline diagnostic before changing production Vulkan rendering.

The simulator uses:

1. VK preprocess dumps (`gauss2screen`, opacity, RGB), VK tile candidate stream, VK replay sideband, and the stored CUDA image.
2. The same per-pixel eval_3D alpha/depth math used by the current Vulkan shader.
3. A first approximation of CUDA `MID_WINDOW=8` TAIL→MID queue/drain cadence:
   - per-4x4 TAIL sorted at the 4x4 center,
   - 16-at-a-time TAIL overflow/final drains,
   - per-2x2 MID backlog of four entries merged with each new group of four,
   - comparison through both `HEAD_WINDOW=4` and current Vulkan `HEAD_W=8`.
4. No CUDA 4x4 alpha/frustum culling yet; this is isolated MID/drain evidence, not a full CUDA cascade clone.

Validation:

- `python3 -m py_compile harmonyos_3dgs/tools/diagnose_l1b_mid_replay.py` passed.
- `python3 harmonyos_3dgs/tools/diagnose_l1b_mid_replay.py --top 30`:
  - replay sideband reconstructs VK with max RMSE `5.07e-7`.
  - wins: `raw_tile=14`, `mid_h4=7`, `mid_h8=9`, `vk_replay=0`.
- `python3 harmonyos_3dgs/tools/diagnose_l1b_mid_replay.py --top 100`:
  - wins: `vk_replay=2`, `raw_tile=31`, `mid_h4=30`, `mid_h8=37`.
  - mean CUDA-color RMSE: `vk_replay=0.092780`, `raw_tile=0.061830`, `mid_h4=0.055119`, `mid_h8=0.047803`.
  - exact/near-exact (`<1e-4`) reconstructions: `raw_tile=2`, `mid_h4=11`, `mid_h8=11`.

For probe `(453,52)`, both MID variants reproduce the CUDA-matching swap:

- VK replay: `... 260620, 218837, 182148, 355888 ...`
- raw/MID replay: `... 260620, 182148, 218837, 355888 ...`

Conclusion: CUDA-style MID/drain cadence explains a substantial part of the remaining production eval_3D gap and beats raw tile order on the top-error set, especially with current Vulkan `HEAD_W=8`. It is still incomplete because raw tile order wins a large minority of pixels and CUDA production also has 4x4 alpha/frustum culling.

## Experiment: offline MID replay simulator v2 with CUDA tail culling

Extended `diagnose_l1b_mid_replay.py` with `--tail-alpha-cull`:

- ports CUDA `max_contrib_gaussian_frustum_3D<true,3,3>` into the script's row-major `gauss2screen` convention,
- applies `CULL_ALPHA=true` at the per-4x4 TAIL insertion stage,
- uses uncropped sort depth for TAIL/MID ordering and keeps final z-range rejection in the per-pixel blend evaluation,
- keeps both `HEAD_WINDOW=4` and `HEAD_W=8` comparison outputs.

Validation:

- `python3 -m py_compile harmonyos_3dgs/tools/diagnose_l1b_mid_replay.py` passed.
- `python3 harmonyos_3dgs/tools/diagnose_l1b_mid_replay.py --top 100 --tail-alpha-cull`:
  - wins: `vk_replay=1`, `raw_tile=9`, `mid_h4=89`, `mid_h8=1`.
  - mean CUDA-color RMSE: `vk_replay=0.092780`, `raw_tile=0.061830`, `mid_h4=0.008797`, `mid_h8=0.023334`.
  - exact/near-exact (`<1e-4`) reconstructions: `raw_tile=2`, `mid_h4=78`, `mid_h8=11`.

Conclusion: the dominant missing production semantic is now strongly identified as CUDA's `CULL_ALPHA=true` TAIL→MID→HEAD cascade with `HEAD_WINDOW=4`. The previous Vulkan `HEAD_W=8` was compensating for the missing MID/cull cascade; once the cull+MID semantics are present, CUDA's `HEAD_WINDOW=4` is the best match.

## Implementation: production shader cascade port

Ported the production non-parity `eval_3D` path in `harmonyos_3dgs/src/vulkan/shaders/rasterize.comp` from the simplified per-batch sub-tile permutation to a CUDA-style full-tile private replay stream:

- 4x4 TAIL frustum/alpha culling using the same `max_contrib_gaussian_frustum_3D<true,3,3>` semantics validated by the offline simulator.
- TAIL stages candidates in CUDA-like 32-input chunks, keeps a 64-entry depth-stable queue, and drains the front 16 when occupancy exceeds 32.
- Per-2x2 MID queue uses `MID_WINDOW=8` and emits the front 4 into HEAD after each merge.
- Per-pixel HEAD now uses `HEAD_WINDOW=4` and records replay sideband only when entries are actually composited.
- The `spec_disable_subtile_resort == 1u` parity path and the 2D conic path were left unchanged.

Validation:

- `cmake --build harmonyos_3dgs/build --target gs3d_vk_tests -j$(nproc)` passed.
- `VkVsCudaBasketball.Cam0_PsnrAtLeastBaseline` improved from **54.730 dB** to **60.696 dB**.
- `VkVsCudaBasketball.Cam0_NContribDump` passed and regenerated replay dumps.
- `python3 harmonyos_3dgs/tools/diagnose_l1b_mid_replay.py --build-dir harmonyos_3dgs/build --top 100 --tail-alpha-cull` confirmed replay sideband still reconstructs Vulkan output with max RMSE **2.60e-7** and mean RMSE **6.10e-8**.
- `VkVsCudaFirstLoss.*` passed **12/12**, preserving parity-mode gates.
- Targeted eval_3D trainer/forward smoke tests passed **5/5**.
- Targeted CTest regex `VkVsCudaBasketball|VkVsCudaFirstLoss|VulkanTrainer|ForwardPipeline|RasterizePass|TileBinnerVulkan|McmcDensify` passed **37/37**.
- Initial full CTest passed **315/315** in **220.53s**.
- +1/+2 review found one blocking shader safety issue: bounded TAIL/MID insertion could write out of bounds when the queue was full and a worse candidate sorted after all residents. Fixed both helpers to drop that candidate instead of writing at `TAIL_W`/`MID_W`.
- Review follow-ups also refreshed stale replay/binding comments and strengthened `VulkanTrainer.Eval3DNonParityStepSmoke` to assert replay sideband count equals summed `n_contrib`.
- Post-fix targeted regression `VulkanTrainer.Eval3DNonParityStepSmoke:VkVsCudaBasketball.Cam0_PsnrAtLeastBaseline:VkVsCudaBasketball.Cam0_NContribDump:VkVsCudaFirstLoss.*` passed **15/15**; PSNR remained **60.696 dB**.
- +1/+2 re-review found no remaining blockers.
- Final full CTest passed **315/315** in **243.58s**.

Conclusion: the L1b production forward gate is now open above the 60 dB target and review/test-audit blockers are cleared. The remaining top-error pixels are a much smaller residual distribution; do not tune constants blindly. Any further improvement should compare current shader replay against the offline simulator per top-error pixel and isolate cull/order differences one layer at a time.
