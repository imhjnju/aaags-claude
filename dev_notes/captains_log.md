# Captain's Log

## Session 4 — 2026-04-20

AAA-GS reference render pipeline wired up.

Goals:
- [x] Load `basket-aaa.ply` (400k Gaussians, sh_degree=3, filter_3D=True) via GaussianModel
- [x] Parse camera ID 0 from `harmonyos_3dgs/cameras.json` (720×960, fx≈706, fy≈707)
- [x] Reconstruct R, T from C2W pose stored in cameras.json
- [x] Enable all AAA features via `configs/aaa.json` (proper_ewa_scaling, eval_3D, rect_bounding, tight_opacity_bounding, tile_based_culling, hierarchical_4x4_culling, load_balancing)
- [x] Render: 181,943/400,000 visible Gaussians, pixel range [0, ~1.19] → clamped to PNG
- [x] Saved render script as `tools/render_single.py`

Key decisions:
- Must run in `conda run -n aaa-gs` (AAA rasterizer built for Python 3.10; system is 3.13)
- cameras.json stores C2W rotation+position → must invert to get W2C (R, T) for getWorld2View2
- `render_output.png` is a generated artifact → added to .gitignore

## Session 3 — 2026-04-20

SP-4 code review (Tasks 1–8, two-round dual-reviewer).

### SP-4 Tasks 1–4 Review (earlier)
Scope: Forward cache export, d_raw_opacities binding, T_MIN guard rewrite, quaternion Jacobian fix.

Verdict: **CONCERNS** (no blocking bugs, 5 fix-now items)

Fix-now items:
- [ ] F1 `test_preprocessor_backward_vulkan.cpp:263` — uninitialized rotations qx/qy/qz in `CulledGaussianZeroGrad` (UB)
- [ ] F2 `test_backward_pipeline_vk.cpp` — no `d_raw_opacities` assertion in N=103 integration test
- [ ] F3 `preprocessor_backward_vulkan.cpp:121` — stale comment "not read by shader — SP-3 scope" (now false)
- [ ] F4 `rasterize_backward.comp:160-167` — comment says "all Gaussians processed" but T_MIN early-exit was added
- [ ] F5 `test_preprocessor_backward_vulkan.cpp:66` — `MatchesCPU_TinyGolden` uses pre-normalized quat; doesn't exercise P1-1 fix

Deferred:
- Layer-2 cache buffer getters missing (document as TODO for SP-5)
- `d_raw_positions` ~0.8 absolute divergence vs CPU — pre-existing, needs root-cause investigation

### SP-4 Tasks 5–8 Review
Scope: means2D/cov2D cache fix, vkDeviceWaitIdle teardown, CpuAdam optimizer, VulkanTrainer.

Verdict: **REWORK** (1 blocking bug found)

Fix-now items:
- [ ] **F1 BLOCKING** `vulkan_trainer.cpp` — CHW vs HWC image layout mismatch. Rasterizer writes CHW; L1 loss loop and `dL_dpixels_` treat buffer as HWC. Tests use all-zero target (bug masked). Real target → wrong gradients from step 1.
- [ ] F2 `vk_buffer.cpp:27-31` — `vkDeviceWaitIdle(ctx_.device())` called without null-guard; latent crash if buffer outlives context
- [ ] F4 `test_cpu_adam.cpp` — `ConvergesToMinimum` never calls `CpuAdam::step()`; fix to use actual API
- [ ] F5 `cpu_adam.h/cpp` — no null guard for `g.grad == nullptr` in `step()`
- [ ] F6 `test_training_step_vk.cpp` — no position gradient assertion; `d_raw_positions` not independently verified

Deferred perf:
- F3 `vk_buffer.cpp` — ~20× redundant `vkDeviceWaitIdle` per backward call (O(N_buffers)); remove from buffer dtor

Verified correct:
- Adam formula (bias correction 1-indexed, ε outside sqrt, no weight_decay)
- cov2D/det cache wiring (binding 17/18 forward → 20/21 backward, all consistent)
- Gradient zeroing (all 20 GPU SSBOs zeroed before dispatch including d_raw_opacities)
- ForwardCache threading (5 fields downloaded after forward, uploaded before backward)
- Training step order (forward→sort→rasterize→L1→rasterize_bwd→preprocess_bwd→Adam)
- SH degree-3 backward math vs CPU reference

## Session 2 — 2026-04-18

Resume verification + state correction.

- [x] Build verified: all targets compile clean
- [x] Tests verified: 187 total, 100% pass (0 failures)
- [x] Hooks verified: commit-gate, edit-test-gate, state.cjs — all OK
- [x] Workflow state: clean (dirty=false, no pending review)
- [x] MEMORY.md corrected: test count 140 → 187
- [x] session_state.md updated with actual baseline
- SP-1 already merged (prior sessions): VulkanContext, Buffer, Shader, ComputePipeline, TDD gate
- SP-2 plan committed: preprocess.comp + PreprocessorVK
- Next: implement SP-2 (TDD: write test_preprocessor_vk.cpp first, then shader + host)

## Session 1 — 2026-04-16

Project harness initialized for harmonyos_3dgs.

Goals:
- [x] M0 kickoff: dev harness installed (CLAUDE.md, WORKFLOW.md, PROJECT.md, skills, hooks, memory)
- [ ] M0: Verify build passes (`cmake -B build -DBUILD_TESTS=ON && cmake --build build`)
- [ ] M0: Verify test baseline (all 24 unit tests pass)
- [ ] M0: Write first spec section for highest-priority subsystem
