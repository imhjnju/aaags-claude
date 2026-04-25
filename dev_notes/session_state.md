# Session State

## Current Phase
**VK↔CUDA cascade sort equivalence harness** (active S10, 2026-04-23). Plan i: build per-level trace comparator before attempting the VK 3-level port. Ultimate target unchanged: VK eval_3D ≡ CUDA, closing the 17.2 dB PSNR gap.

## Test Counts
- Baseline: 243 passing + 1 VK-vs-CUDA basketball (PSNR=42.8 dB, baseline=42.1 dB) as of S8 2026-04-21.
- Added this session: `DumpCascadeFixtures.Basket_Cam0` (runs VK pre/binner/sorter, dumps .npy fixtures for CUDA+VK shared input).

## Cascade Equivalence Harness — Task Status (S10)
| Phase | Task | Status | Artifact |
|-------|------|--------|----------|
| 1  | VK preprocess dump gtest             | DONE | `tests/test_dump_cascade_fixtures.cpp`, `tests/golden/npy_writer.h` — passing, 44 MB fixture in `build/cascade_trace/` |
| 2a | CUDA trace header (NoOp + Device)    | DONE | `AAA-Gaussians/.../stopthepop/cascade_trace.h` |
| 2a | Patch `hierarchical_render.cuh`       | DONE | 6 edits: `#include`, default `TraceT=NoOp` template arg + arg, HEAD_BLEND + HEAD_INS + MID + TAIL snapshots, new `sortGaussiansRayHierarchicalCUDA_forward_traced` kernel |
| 2b | CUDA pybind entry + setup.py         | DONE | new `rasterize_hier_traced.cu`, `rasterize_points.h` decl, `ext.cpp` `m.def("rasterize_hierarchical_traced", ...)`, `setup.py` source list — compiling |
| 2c | `dump_cuda_cascade_trace.py`         | DONE | `AAA-Gaussians/.../tools/dump_cuda_cascade_trace.py` — awaiting successful build to test |
| 2d | CUDA-vs-CUDA sanity (mechanism)      | DONE | `SelfCompare_Cuda_vs_Cuda` passes: TAIL 688 non-empty snapshots match, HEAD_INS 969/1024 pixels match, HEAD_BLEND 969/1024 match, zero gid/depth/alpha deltas |
| 3  | Comparator gtest skeleton            | DONE | `tests/test_cascade_equivalence.cpp` — compiles in VK test binary, Self/VsCuda both SKIP until CUDA trace + VK trace produced |
| 4A | VK trace SSBO plumbing + dispatch + test | DONE | 17 SSBOs at set=0 bindings 14..30, `spec_trace_enabled=constant_id 1`, `rasterize_traced()` method, `DumpVkCascadeTrace.Basket_Cam0` passes, `Vk_vs_Cuda` FAILs with expected all-zero pattern |
| 4B | VK TAIL fill + bitonic64 + trace hook | PARTIAL | 79 exact match; cadence off — state machine needed, merged into 4B+C |
| 4B+C | VK TAIL+MID state machine (cadence correct, snap 0 matches) | DONE (partial gid_mm) | tail_wcur match; TAIL gid_mm=20049 / MID gid_mm=40816 pending HEAD — architectural coupling found |
| 4D+E+flush | VK HEAD state machine (front4OneFromMid + blend_one + MID-drain-thru-HEAD + end-flush) | IN PROGRESS | combines plan §6 D/E/F after analysis showed CUDA cascade is single state machine, not layerable |
| 4G-I | PSNR validation / Maleoon / perf | FUTURE | — |

## Path A — Defensive Config Unification (S10, 2026-04-25)

VK now reads `configs/aaa.json` via `splatting::SplattingSettings` and asserts
that every JSON value matches the value the VK shaders are hard-coded against.
Mismatches throw at `RasterizerVulkan` construction; VK refuses configs it has
not implemented rather than silently rendering with the wrong behaviour.

**Unified (JSON value == VK hard-coded value, asserted):**
- `sort_settings.sort_mode = HIERARCHICAL` (the only path; cascade rasterizer)
- `sort_settings.sort_order = PER_TILE_DEPTH_MAXPOS` (matches scatter.comp eval_3D depth-along-ray key)
- `sort_settings.queue_sizes = {per_pixel:4, tile_2x2:8, tile_4x4:64}` (matches `HEAD_W=4`, `s_mid_depth[16*4*8]`, `TAIL_SLOTS=64`)
- `culling_settings.{rect_bounding,tight_opacity_bounding,tile_based_culling,hierarchical_4x4_culling} = true` (all baked into preprocess.comp / scatter.comp / rasterize.comp without runtime gates)
- `load_balancing = true` (VK tile-binner + scatter expand each Gaussian over touched tiles unconditionally)
- `proper_ewa_scaling = true` (preprocess.comp scales opacity by dilation_factor unconditionally)
- `new_aabb = true` (compute_aabb_screen path is dead code in VK — see preprocess.comp:1030 comment, gotchas.md)

**Runtime-honoured (no assertion; pipeline reacts to JSON value):**
- `eval_3D` — drives `spec_eval_3D` specialization constant + `RasterizerVulkan(ctx, eval_3D=…)` plumbing.
- `near_clipping` — preprocess.comp 2D path gates the `p_view.z<=0.2` cull on this flag (master merge 1b02ca2).

**Still hard-coded (not yet config-driven; would need shader changes to vary):**
- All four queue-size constants are baked as literals + `shared` array sizes in `rasterize.comp`. Changing them needs spec constants + dynamic shared-mem sizing.
- Sort mode/order other than HIERARCHICAL/PER_TILE_DEPTH_MAXPOS would require a different rasterize shader entirely (stopthepop's GLOBAL/PER_PIXEL_FULL/PER_PIXEL_KBUFFER paths are not ported).

**Files (added):** `include/splatting_settings.h`, `src/splatting_settings.cpp`, `tests/test_config_loader.cpp`, `tests/test_data/aaa.json` (vendored fallback for worktrees without the AAA submodule).

**Files (modified):** `CMakeLists.txt` (move nlohmann_json fetch to global scope; add splatting_settings.cpp; register test), `include/vulkan/rasterizer_vulkan.h` (new SplattingSettings ctor), `src/vulkan/rasterizer_vulkan.cpp` (impl).

**Tests:** 12 new ConfigLoader tests pass; 4 reference tests still pass (`VkVsCudaBasketball.Cam0_PsnrAtLeastBaseline`, `RasterizerVulkan.Rasterize_TinyFixture`, `CascadeEquivalence.SelfCompare_Cuda_vs_Cuda`, `DumpVkCascadeTrace.Basket_Cam0`); 169 gs3d_tests pass.

**Limitations:** legacy `RasterizerVulkan(ctx, bool eval_3D)` ctor is preserved and bypasses the validator — call sites that have not migrated still rely on VK's hard-coded behaviour matching aaa.json. Migration of train/main entry points to the SplattingSettings ctor is a follow-up.

## Design Decisions (S10, user-confirmed)
- **Trace granularity**: 细 — 4-level (TAIL-post-merge, MID-post-merge, HEAD-insert, HEAD-blend).
- **Input scale**: 中等 — 4 tiles from basket-aaa cam0, pair count 300-2000.
- **Comparison**: 精确 — gid sequence bit-exact, depth ε=1e-5, alpha ε=1e-6.
- **Equal-depth tie**: unordered (CUDA `batcherSort<32>` has no secondary key → tie is hardware-dependent).
- **Submodule policy**: patch `diff-gaussian-rasterization`; guard with default template arg + `NoOpCascadeTrace` so existing callers zero-cost-compatible.
- **Shared input**: VK preprocess output drives both sides (.npy in `${CMAKE_BINARY_DIR}/cascade_trace/`).

## Milestones

## Milestones
| Milestone | Status | Sessions | Summary |
|-----------|--------|----------|---------|
| SP-0: CUDA golden infra | DONE | — | CPU reference + FD test harness |
| SP-1: Vulkan infra | DONE | S2 | VulkanContext/Buffer/Shader/Pipeline; TDD gate |
| SP-2: Vulkan forward pipeline | DONE | S2 | preprocess.comp + sort + rasterize.comp |
| SP-3: Vulkan backward pipeline | DONE | S3 | rasterize_backward.comp + preprocess_backward.comp |
| SP-4: Training integration | DONE | S4 | ForwardCache caching, GPU Adam skeleton (CpuAdam), VulkanTrainer, 216 tests |
| SP-5: GPU optimizer + hyperparams | DONE | S5 | GPU Adam kernel, LR+SH schedules, DSSIM, MCMC densification, basketball E2E smoke test, 229 tests |
| SP-6: Training gaps closed | DONE | S7 | T1-T5 done; 243/243 + basketball loss-decrease pass |
| VK-CUDA parity (PSNR≥60dB) | IN PROGRESS | S8 | 25.3→42.8 dB; proper_ewa, tile_culling, sub-tile sort done. Need persistent TAIL buffer |
| M0: Foundation | IN PROGRESS | — | Interleaved with Vulkan migration |

## VK-CUDA Parity Task Status (Session 8)
| Task | Status | PSNR Impact | Commit |
|------|--------|-------------|--------|
| Harness (render_single.py + gtest) | DONE | baseline=25.3 | a4dc240, fa6adc9 |
| proper_ewa_scaling (eval_3D + 2D) | DONE | +11.6 dB | ed303bd |
| rect_bounding + tight_opacity_bounding | DONE | 0 dB | 6ff257b |
| tile_based_culling (INVALID sentinel) | DONE | +3.4 dB | 83ca5b1 |
| Hierarchical sub-tile TAIL re-sort | DONE | +2.3 dB | b4cd239 |
| HEAD_W=8, subtile_cx+2.0, z/w key | DONE | +0.2 dB | 72bd4bf |
| Persistent cross-batch TAIL buffer | TODO | ? | — |
| Verify ≥60 dB + lock baseline | TODO | — | — |

## Python → C++ Gaps Closed (SP-6)
1. **Opacity reg**: `dL/d_raw_opacity += (0.01/N)*sig*(1-sig)` — after backward, before Adam upload
2. **Scale reg**: `dL/d_raw_scale += (0.01/N)*exp(raw_sc)` — per-component
3. **Position noise**: `Sigma @ N(0,1) * op_sigmoid(1-opacity) * noise_lr * pos_lr` — after Adam download
4. **Spatial LR**: `pos_lr = spatial_lr_scale * lr_schedule(...)` — default scale=1.0

## Latest Sessions

### S10 — 2026-04-23 → 2026-04-24 (current)
- Opened cascade sort equivalence harness (design ratified with user as Plan i).
- Phase 1 DONE: `DumpCascadeFixtures.Basket_Cam0` passes on Tegra. Dump = 44 MB (N=400k, 2700 tiles; 4 selected tile IDs: 680 719 720 721; counts 301-366). Fixture at `harmonyos_3dgs/build/cascade_trace/`.
- Phase 2a DONE: `cascade_trace.h` (NoOp + Device trace writers, per-level claim/snapshot helpers with correct NoOp fallback) + 6 edits to `hierarchical_render.cuh` — `#include`, default `TraceT=NoOp` template arg, HEAD_BLEND / HEAD_INS / MID / TAIL snapshots, new `sortGaussiansRayHierarchicalCUDA_forward_traced` kernel. Existing callers untouched (zero behavior change).
- Phase 2b DONE: `rasterize_hier_traced.cu` (new, bypasses CUDA preprocess; takes VK-preprocessed tensors directly), `rasterize_points.h` decl, `ext.cpp` pybind (`rasterize_hierarchical_traced`), `setup.py` source list. Python rebuild iterating on compile fixes (include order, `using namespace CudaRasterizer`, constexpr int64_t locals instead of through-instance constants).
- Phase 2c DONE: `tools/dump_cuda_cascade_trace.py` — loads fixture, casts u32→int32 for binding, calls `_C.rasterize_hierarchical_traced`, saves 15 trace tensors + output color + final_T to `cuda/` subdir.
- Phase 3 DONE (skeleton): `test_cascade_equivalence.cpp` + `tests/golden/npy_writer.h` already in CMake. Self-compare (CUDA vs CUDA) and Vk-vs-Cuda tests both gated on fixture presence → SKIP if traces missing.
- REMAINING this session: (a) green the CUDA build (4th attempt in flight), (b) run Python script end-to-end to verify trace generation, (c) run SelfCompare gtest to validate comparator.
- FUTURE: Phase 4 VK 3-level cascade port itself — expected to consume this harness.

### S8 — 2026-04-20 (current)
- SP-7 T1-T5 complete (CB chaining, persistent buffers): 248 tests pass
- Basketball test: changed to 100 steps (no densification) PSNR=5.61 dB, finite+positive assertion
- VK vs Python gradient comparison: 3 new tests implemented (task 54-56)
  - `VkVsPyReference.Step1GradientAndLoss`: loss rel_diff=3.6e-7, all grad norms 0.0000 diff
  - `VkVsPyReference.ConvergenceTable100Steps`: both converge 0.042→0.002 (100 steps)
- **BUG FIXED**: rasterize.comp writes CHW but loss/backward expect HWC — fixed in vulkan_trainer.cpp
  - Gradient norms were 45-60% below Python reference before fix
  - See gotchas.md for full details
- Python reference dump: tools/dump_tiny_reference.py + tests/golden/tiny/py_ref/ (1000 files)
- VulkanTrainer gradient capture: enable_gradient_capture() + captured_grad_*() accessors

### S7 — 2026-04-19
- SP-6 T1-T4 complete via subagent-driven development
- 243 tests pass (non-basketball)
- Basketball 100-step loss-decrease validated

### S6 — 2026-04-19
- SP-5 Task 6 (basketball E2E) completed and committed
- 229/229 tests pass
- Basketball 100-step background validation PASSED (248.7s, loss decreased)

### S5 — 2026-04-18 to 2026-04-19
- SP-5 Tasks 1-6 complete via subagent-driven development
- GPU Adam (adam_step.comp + VulkanAdam), LR schedule, SH warmup, DSSIM analytical gradient, MCMC densification, basketball smoke test
- Key perf finding: ~2.5 s/step on Tegra due to sync-per-dispatch; SP-7 will chain CBs

### S4 — 2026-04-18
- SP-4 all 8 tasks complete + extra cov2D/Part C bug fix
- 216/216 tests passing

### S3 — 2026-04-17 to 2026-04-18
- SP-3 complete: rasterize_backward.comp + preprocess_backward.comp
- All 208 SP-3 tests passing before SP-4

### S2 — 2026-04-17
- SP-1 complete (Vulkan infra) + SP-2 complete (forward pipeline)

### S1 — 2026-04-16
- Installed dev harness (CLAUDE.md, WORKFLOW.md, PROJECT.md, skills, hooks, memory)
