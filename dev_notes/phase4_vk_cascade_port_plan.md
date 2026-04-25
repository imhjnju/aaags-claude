# Phase 4 — VK 3-level Cascade Rasterizer Port: Design Plan

> Produced S10 (2026-04-24) by Plan subagent; trimmed & corrected to actual constants. Ultimate target: VK `rasterize.comp` eval_3D ≡ CUDA `hierarchical_render.cuh`, closing 17.2 dB PSNR gap (42.8 → ≥60 dB).

## Locked decisions (user-ratified)

| Key | Value | Source |
|---|---|---|
| Trace buffer capacity | `MAX_TAIL=512, MAX_MID=1024, MAX_HEAD_INS=4096, MAX_HEAD_BLEND=4096` (already oversized in `cascade_trace.h`; Q1 bump moot) | §7.3 |
| PSNR target | **60 dB stretch, hard goal** (no 55 dB softening) | §7.4 / Q2 |
| Maleoon shared-mem branch | lazy: query `maxComputeSharedMemorySize`, activate `BATCH_SIZE=128` spec-const only if <37 KiB | §7.5 / Q3 |
| Submodule strategy | patch `diff-gaussian-rasterization` in place (already done) | Phase 2a |
| Tie-break | CUDA `batcherSort<32>` has no secondary key → depth-tie unordered tolerated on VK side | §7.4 |

## Outstanding debts

- **B2 verification** (tracked as task #28): prove Phase 2a CUDA patch is bit-exact no-op. If CUDA gold is tainted, Milestone G PSNR check is invalid. Address before G.

---

## 1. CUDA → Vulkan GLSL Semantic Mapping

| CUDA construct | Vulkan GLSL equivalent | Notes |
|---|---|---|
| `cg::this_thread_block()` | implicit workgroup; `barrier()` + `memoryBarrierShared()` | One block == one workgroup |
| `block.thread_rank()` | `gl_LocalInvocationIndex` (0..255) | Linear rank |
| `cg::tiled_partition<32>` (warp) | No core GLSL equivalent. **Portable**: shared-memory emulation, indexed by logical warp id. **Fast path** (opt-in): `subgroup*` ops when `gl_SubgroupSize==32` | Tegra=32, Maleoon unknown → emulate by default |
| `cg::tiled_partition<16>` (halfwarp) | No native. Emulate via shared memory; all sub-warp syncs promote to workgroup `barrier()` | Perf cost, correctness-neutral |
| `cg::tiled_partition<4>` (head_group) | No native. 2×2 pixel quad exchange via shared-mem slots indexed by `quadrant_id` | Biggest porting pain — `shflRankingLocal<4>` has no GLSL analog |
| `__shfl_sync / __shfl_xor` | `subgroupShuffle*` (if size matches) or shared-mem exchange | Default to shared-mem for portability |
| `batcherSort<32>` | Bitonic sort on 32 elements in shared memory | No secondary key → tie unordered |
| `mergeSortRegToSmem<N>` | Per-lane `lower_bound` binary search over sibling half's keys in shared mem, write to rank slot | Used in `pushPullThroughMid` |
| `shflRankingLocal<4>` | Shared-mem rank-by-count over 4 quadrant pixels (all-to-all compare) | Used in `front4OneFromMid` |
| `__shared__ int; atomicAdd` | `shared int; atomicAdd(var,1)` | Identical semantics, needs `barrier()` before peer reads |
| `__ballot_sync` | `subgroupBallot` (uvec4) or shared int counter fallback | |
| `__syncwarp()` | No sub-workgroup sync → promote to `barrier()` | |
| `return` from thread needing future `barrier()` | **UNDEFINED BEHAVIOR** in GLSL — never return; gate with booleans | Critical rule |
| Dynamic `extern __shared__` | Must size shared arrays at compile time via spec-constants | Forces worst-case sizing |

**Fundamentally different idiom** to handle:
- Warp-divergent early exit → always reach barriers, gate body with `if (!pixel_done) { ... }`.
- Register-resident cooperative sort → pays shared-mem round-trip.

---

## 2. Workgroup Layout

CUDA uses `dim3(16,4,4)=256`. VK keeps `local_size(16,16,1)=256` (matches current `rasterize.comp`). Logical IDs are pure `gl_LocalInvocationID` arithmetic — **not** dependent on `gl_SubgroupSize`:

```glsl
uint pix_x = gl_LocalInvocationID.x;                        // 0..15
uint pix_y = gl_LocalInvocationID.y;                        // 0..15
uint subtile_x = pix_x / 4u;                                 // 0..3
uint subtile_y = pix_y / 4u;                                 // 0..3
uint subtile_id = subtile_y * 4u + subtile_x;                // 0..15 (= warp analog)
uint in_subtile = (pix_y & 3u) * 4u + (pix_x & 3u);          // 0..15
uint quadrant_id = ((pix_y & 3u) / 2u) * 2u
                 + ((pix_x & 3u) / 2u);                      // 0..3 (= head_group analog)
uint in_quadrant = (pix_y & 1u) * 2u + (pix_x & 1u);         // 0..3
```

Optional fast-path: spec-const `USE_SUBGROUP_FAST_PATH=1` replaces shared-mem exchanges with `subgroupShuffle*` when `gl_SubgroupSize==32`. Baseline works on any subgroup size.

---

## 3. Shared Memory Budget (per workgroup)

`BATCH_SIZE=256` preserved (from CUDA).

| Buffer | Purpose | Bytes |
|---|---|---|
| `s_g2s[256*16]` | gauss2screen 4×4 matrix (existing) | 16384 |
| `s_opa[256]` | pre-dilated opacity | 1024 |
| `s_rgb_r/g/b[256]` | colour channels (existing) | 3072 |
| `s_gid[256]` | original Gaussian id (for trace) | 1024 |
| `s_pixpos[256]` | packed 2D center cache | 1024 |
| `s_tail_depth[16*64]` | **TAIL** depth per subtile, 64 slots | 4096 |
| `s_tail_idx[16*64]` | TAIL index | 4096 |
| `s_tail_count[16]` | fill count per subtile TAIL | 64 |
| `s_mid_depth[16*4*8]` | **MID** depth per (subtile,quadrant), 8 slots | 2048 |
| `s_mid_idx[16*4*8]` | MID index | 2048 |
| `s_mid_count[16*4]` | fill count per (subtile,quadrant) | 256 |
| `s_merge_scratch[256]` | merge-sort scratch | 1024 |
| `s_rank_scratch[16*4*4]` | rank-sort scratch | 1024 |
| `s_trace_tail_slot`, `s_trace_mid_slot` | block-wide slot broadcasts | 8 |
| **Subtotal** |  | **~37.2 KiB** |

Tegra 48 KiB → OK. Maleoon unknown → §7.5 lazy-branch.

Current shader uses `sub_order[16][256]` (16 KiB) which goes away in the port — net +21 KiB over existing baseline.

---

## 4. Synchronization Plan

```
for each batch:
  (A) load_batch_to_shared()
  barrier()                               // [B1] after cooperative load
  (B) tail_fill_per_subtile()             // 16 subtiles in parallel
  barrier()                               // [B2] before TAIL sort
  (C) tail_sort_64_per_subtile()          // bitonic stages, fixed trip count
  barrier()                               // [B3] ★ TRACE HOOK 1: TAIL ★
  (D) push_through_mid_per_quad()         // per (subtile,quadrant) merge
  barrier()                               // [B4] ★ TRACE HOOK 2: MID ★
  (E) head_fill_and_blend_per_pixel()     // per-pixel, may trace HEAD_INS / HEAD_BLEND
  barrier()                               // [B5] end of batch
// Post-loop: flush remaining per-pixel HEAD (no barriers needed; per-pixel private)
```

**Uniform-control-flow rules** (critical — non-uniform barrier = VK_ERROR_DEVICE_LOST):
1. Every barrier site is outside any data-dependent `if`.
2. Per-pixel "done" gates write body only; `barrier()` always crossed.
3. `tail_sort_64` stage loop has fixed trip count (compile time); subtiles with 0 fill walk empty stages, still cross barriers.
4. Thread 0 is always alive (no `return`) → trace slot-broadcast `if (gl_LocalInvocationIndex==0u)` pattern is uniform.

---

## 5. Trace Hook Plan

Must produce 16 NPY files matching CUDA `DeviceCascadeTraceView` layout. Files (confirmed from `cascade_trace.h:87-121`):

| File | Shape | dtype |
|---|---|---|
| `slot_lookup.npy` | `[num_tiles]` | i32 (−1 or 0..K−1) |
| `tail_depths.npy`, `tail_ids.npy` | `[K, 512, 16, 64]` | f32 / i32 |
| `tail_wcur.npy` | `[K]` | u32 |
| `mid_depths.npy`, `mid_ids.npy` | `[K, 1024, 16, 4, 8]` | f32 / i32 |
| `mid_wcur.npy` | `[K]` | u32 |
| `head_ins_{depth,alpha}.npy` | `[K, 256, 4096]` | f32 |
| `head_ins_gid.npy` | `[K, 256, 4096]` | i32 |
| `head_ins_cursor.npy` | `[K, 256]` | u32 |
| `head_blend_{depth,alpha,T}.npy` | `[K, 256, 4096]` | f32 |
| `head_blend_gid.npy` | `[K, 256, 4096]` | i32 |
| `head_blend_cursor.npy` | `[K, 256]` | u32 |

Note `head_blend_T` — blend-time T value; Plan subagent's original draft missed this field.

### 5.1 Specialization constant

```glsl
layout(constant_id = 1) const uint spec_trace_enabled = 0u;
```

(Existing `constant_id = 0` is `spec_eval_3D`; extend `rasterize_spec::` namespace with `TRACE_ENABLED = 1`.)

All trace code paths guarded by `if (spec_trace_enabled == 1u) { ... }`. Spec-const folding by driver eliminates the path at specialization time — zero runtime cost when disabled.

### 5.2 SSBO bindings

Candidate layout — extend `rasterize_bind::`. If set=0 binding 14..28 is too many, promote trace block to set=1.

```
binding 14: TraceMeta (UBO, 32 B) { K, num_tiles, reserved[6] }
binding 15: SlotLookup (SSBO, num_tiles i32)  // slot_lookup: -1 or 0..K-1
binding 16: TailDepths                 (SSBO, K*512*16*64 f32)
binding 17: TailIds                    (SSBO, K*512*16*64 i32)
binding 18: TailWcur                   (SSBO, K u32)
binding 19: MidDepths                  (SSBO, K*1024*16*4*8 f32)
binding 20: MidIds                     (SSBO, K*1024*16*4*8 i32)
binding 21: MidWcur                    (SSBO, K u32)
binding 22: HeadInsDepth               (SSBO, K*256*4096 f32)
binding 23: HeadInsAlpha               (SSBO, K*256*4096 f32)
binding 24: HeadInsGid                 (SSBO, K*256*4096 i32)
binding 25: HeadInsCursor              (SSBO, K*256 u32)
binding 26: HeadBlendDepth             (SSBO, K*256*4096 f32)
binding 27: HeadBlendAlpha             (SSBO, K*256*4096 f32)
binding 28: HeadBlendT                 (SSBO, K*256*4096 f32)
binding 29: HeadBlendGid               (SSBO, K*256*4096 i32)
binding 30: HeadBlendCursor            (SSBO, K*256 u32)
```

When trace disabled: bind a single 4-byte `dummy4_buf` to bindings 14..30 to satisfy validation.

Total trace VRAM at K=4: ~146 MB. Trivial.

### 5.3 Slot-claim pattern

Mirrors CUDA `claim_tail_slot()` which does `atomicAdd(&view.tail_wcur[slot], 1u)`. In GLSL:

```glsl
if (spec_trace_enabled == 1u) {
    int k = SlotLookup[tile_id];
    if (k >= 0) {
        // Claim once per batch per block: leader increments wcur, broadcast via shared
        if (gl_LocalInvocationIndex == 0u) {
            s_trace_tail_slot = int(atomicAdd(TailWcur[k], 1u));
        }
        barrier();
        int t = s_trace_tail_slot;
        if (t >= 0 && t < 512) {
            // Each of 16 half-warp-equivalent threads writes 4 of 64 slots (strided)
            // matching CUDA's: for i in 0..3: write slot (rank + i*16)
            uint base = ((uint(k) * 512u + uint(t)) * 16u + subtile_id) * 64u;
            // thread_in_halfwarp = in_subtile (0..15); write 4 slots
            for (int i = 0; i < 4; ++i) {
                uint s = uint(in_subtile) + uint(i) * 16u;
                TailDepths[base + s] = s_tail_depth[subtile_id * 64u + s];
                TailIds[base + s]    = int(s_gid[s_tail_idx[subtile_id * 64u + s]]);
            }
        }
    }
}
barrier();
```

MID hook: same pattern with MID wcur + sub_y/sub_z/quad indexing.

HEAD_INS / HEAD_BLEND: per-pixel `cursor` is thread-private — each pixel maintains `uint ins_cur, blend_cur`; at write time index `HeadInsDepth[(k*256 + p)*4096 + cur]`, then `++cur`. At end of shader flush cursors to `HeadInsCursor[k*256+p]`.

### 5.4 Driver changes

Add `RasterizerVulkan::rasterize_traced(...)` parallel to `rasterize()`:
1. Create pipeline variant with `spec_trace_enabled=1`.
2. Upload SlotLookup (fill `-1`, write `k` for each selected tile).
3. `vkCmdFillBuffer` to zero all cursor/wcur buffers.
4. Dispatch as usual.
5. Readback → NPY via new `save_npy_*` mirror of `test_dump_cascade_fixtures.cpp`.

New test: `tests/test_dump_vk_cascade_trace.cpp` — writes `${CMAKE_BINARY_DIR}/cascade_trace/vk/*.npy`.

---

## 6. Incremental Milestones

Each milestone flips one level of `CascadeEquivalence.Vk_vs_Cuda` from FAIL-or-SKIP to clean.

### A — Trace plumbing (this session)
- Shader: `spec_trace_enabled` const + bindings 14..30 declarations. NO trace writes (stub).
- Driver: `rasterize_traced()` method; allocates, zero-inits, downloads, NPY-writes.
- New test: `tests/test_dump_vk_cascade_trace.cpp`.
- **Success**: `Vk_vs_Cuda` goes SKIP→FAIL with "A(cuda_val, cuda_gid) B(0, 0)" divergence. Comparator sees VK input.

### B+C — TAIL + MID state machine (combined; cannot be independent)
- **Architectural note**: CUDA TAIL is NOT a 64-wide bitonic sort. It is a **state machine** running on a **32-wide inner batch cadence** inside the 256-wide cooperative load:
  1. batcherSort<32> sorts the 32 new entries in-register
  2. mergeSortRegToSmem merges these 32 sorted with the existing persistent TAIL content (up to 64 slots)
  3. If TAIL would overflow past 64, the overflow front (smallest-depth entries) is pushed via `pushPullThroughMid` into MID, freeing TAIL slots
  4. Emit TAIL trace hook (one snapshot per inner 32-wide step)
- Because steps 3 and 4 are **tightly coupled** (TAIL cadence follows inner-batch cadence; overflow requires MID to exist as a sink), Milestones B and C cannot be independently achieved. They are merged here.
- Shader work:
  - Inner 32-wide loop inside existing 256-wide cooperative load
  - `tail_fill_per_subtile` (eval at subtile center via `max_contrib_ray`, push to scratch register array — not shared yet)
  - sort-32 of the new batch (Batcher's bitonic on 32; the current Milestone B code has a 64-bitonic we can adapt as sort-32 helper)
  - mergeSortRegToSmem<N> emulation: binary-search-based merge of sorted-32 into sorted-64 shared TAIL
  - Overflow detection: if persistent TAIL would exceed 64 slots, push front to MID via `push_through_mid_per_quad` (per quadrant, 8 slots; match `pushPullThroughMid` semantics)
  - Emit TAIL trace after each merge step
  - Emit MID trace after each drain-to-MID step
- Test work:
  - Fix `compare_mid` helper in `test_cascade_equivalence.cpp` (currently reuses `compare_tail` semantics — wrong for [K, 1024, 16, 4, 8] layout; must treat 8 slots per (subtile, quadrant) group)
- **Success**: `Vk_vs_Cuda` TAIL AND MID → gid_mm=0 d_mm=0 (ties unordered tolerated).

### D — HEAD_INS (huge)
- Shader: replace current ad-hoc HEAD (HEAD_W=8) with `shflRankingLocal<4>`-emulated: for each quadrant's 8 MID entries, 4 pixels do all-to-all count-rank in shared memory, distribute 4 smallest-depth survivors to per-pixel HEAD[4]. Emit HEAD_INS trace.
- **Success**: HEAD_INS → gid_mm=0 d_mm=0 a_mm=0.

### E — HEAD_BLEND (medium)
- Shader: when HEAD full + new incoming entry, blend front (smallest-depth) into `(C,T)`, pop; continue until `T<T_MIN`. Emit HEAD_BLEND trace (including `T_before`).
- **Success**: all 4 levels `Vk_vs_Cuda` green.

### F — final HEAD flush (small)
- Shader: post-last-batch flush of remaining per-pixel HEAD into final color. Re-verify trace clean + `test_rasterize_pass_vk` green.

### G — PSNR (small test-run, possibly large triage)
- Run `test_vk_vs_cuda_basketball` with trace disabled.
- **Stretch**: ≥60 dB (Q2 hard goal).
- If trace clean but PSNR<60 dB: triage tie-break determinism, FP contract, SH eval, preprocess drift.

### H — Maleoon (medium)
- Cross-compile, query `maxComputeSharedMemorySize`. <37 KiB → activate `BATCH_SIZE=128` spec-const (Q3 lazy-branch).
- Render basketball, cross-compare with Tegra output.

### I — perf (optional)
- Profile. If TAIL sort >30% and subgroupSize==32 → `USE_SUBGROUP_FAST_PATH=1` variant.

### Sequencing

A → B → C (+ compare_mid fix) → D → E → F → G → **B2 verification** (task #28) → H → I.

B2 verification **must** happen before Milestone G is accepted; otherwise a clean cascade trace could coincide with a tainted CUDA gold, hiding real divergence.

---

## 7. Risks & mitigations

### 7.1 Subgroup size variance (Maleoon)
Baseline is subgroup-size-agnostic. Fast path opt-in only.

### 7.2 Barrier-in-divergent-flow
Run `VK_LAYER_KHRONOS_validation` each milestone; any "barrier reached by non-uniform control flow" warning blocks merge.

### 7.3 Trace buffer overflow
If per-tile real TAIL events >512 or HEAD events >4096, CUDA silently truncates (cascade_trace.h:147 `snapshot_id >= MAX_TAIL_SNAPSHOTS) return`). Add runtime assert in `test_dump_cuda_cascade_trace.py` probe that checks `tail_wcur < 512` for all K. Escalate by re-bumping if it fires.

### 7.4 Depth-tie determinism
Comparator tolerates. If PSNR<60 dB with clean trace, consider adding `(depth, gid)` secondary key on VK side (over-constrains VK vs CUDA — user decision required before merging).

### 7.5 Shared memory on Maleoon
Mitigation: spec-const `BATCH_SIZE=128` path halves 256-sized buffers (~10 KiB saved). Activate only on devices reporting <37 KiB.

### 7.6 Unbound SSBO when disabled
Bind dummy 4-byte buffer to bindings 14..30 when trace off (existing dummy_buf pattern, e.g., `rasterizer_vulkan.cpp:215`).

---

## 8. Work unit ranking (rough scope)

| Rank | Milestone | Scope |
|---|---|---|
| 1 | D HEAD_INS | **Huge** — `shflRankingLocal<4>` full shared-mem emulation |
| 2 | B TAIL | **Large** — bitonic-64, first-time GLSL primitive |
| 3 | C MID | **Large** — mergeSortRegToSmem emulation + compare_mid fix |
| 4 | A plumbing | **Medium** — 17 bindings + dispatch + test |
| 5 | E HEAD_BLEND | **Medium** — mirrors existing HEAD overflow |
| 6 | G PSNR triage | **Variable** — small if clean, large if mismatch |
| 7 | H Maleoon | **Medium** — cross-compile + unknown device |
| 8 | F flush | **Small** |
| 9 | I perf | **Small-Medium, optional** |

---

## Critical files

- `harmonyos_3dgs/shaders/rasterize.comp` (verify exact path on Milestone A start)
- `harmonyos_3dgs/include/vulkan/preprocess_bindings.h` (binding namespace)
- `harmonyos_3dgs/src/vulkan/rasterize_pass.cpp`, `.h` (pass driver)
- `harmonyos_3dgs/src/vulkan/rasterizer_vulkan.cpp` (high-level driver + dummy_buf pattern)
- `harmonyos_3dgs/tests/test_cascade_equivalence.cpp` (comparator — needs compare_mid fix in C)
- `harmonyos_3dgs/tests/test_dump_cascade_fixtures.cpp` (reference for NPY writer usage)
- `harmonyos_3dgs/tests/golden/npy_writer.h` (NPY v1.0 writer)
