# Diagnostic Patches

Patches that modify code **outside** `harmonyos_3dgs/` — most commonly the
AAA-Gaussians CUDA rasterizer submodule (`AAA-Gaussians/submodules/diff-gaussian-rasterization`)
— for investigation purposes only. They are not part of the production build; do
not apply them to the shared submodule. Apply them to a per-worktree local clone
of the submodule when the corresponding diagnostic is needed, and revert before
shipping.

## `dgr_hierarchical_ncontrib.patch`

**Target**: `AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/stopthepop/hierarchical_render.cuh`

**What it does**: Enables the per-pixel `n_contrib` writer in the HIERARCHICAL
rasterizer path (`sortGaussiansRayHierarchicalCUDA_forward`). Uncomments three
lines so the hierarchical cascade increments a `num_blends` counter on each
successful blend and writes it to `img.n_contrib[pix_id]` at pixel finalize.

This is normally commented out in upstream DGR — only the GLOBAL/vanilla
`renderCUDA` path writes `n_contrib` by default (as `last_contributor`). For
HIERARCHICAL sort_mode (what `configs/aaa.json` uses, `sort_mode=3`), the
`n_contrib` buffer is left uninitialized without this patch.

**When to apply**: Any time we need to compare VK's per-pixel blend count
against CUDA's while the CUDA scene uses `sort_mode=HIERARCHICAL`. In the
Session 9 / white-table VK-vs-CUDA parity investigation, this was used to
confirm the bug was not a count-mismatch issue, which then pointed to
per-Gaussian preprocess differences.

### How to apply (per-worktree isolated build)

The safe pattern — **do not modify the shared submodule**:

1. Create a per-worktree git-worktree of the DGR submodule at a clean commit:
   ```bash
   cd AAA-Gaussians/submodules/diff-gaussian-rasterization
   git worktree add <worktree>/local-deps/dgr-clean HEAD
   ```
2. Apply this patch to the isolated worktree:
   ```bash
   cd <worktree>/local-deps/dgr-clean
   git apply <worktree>/dev_notes/diagnostic_patches/dgr_hierarchical_ncontrib.patch
   ```
3. Force a full `.cu` recompile (distutils doesn't track `.cuh` header deps):
   ```bash
   touch cuda_rasterizer/forward.cu cuda_rasterizer/backward.cu
   conda activate aaa-gs
   python setup.py build_ext --inplace
   ```
4. Route Python to the isolated build via `PYTHONPATH`:
   ```bash
   PYTHONPATH=<worktree>/local-deps/dgr-clean \
     python tools/dump_cuda_ncontrib.py ...
   ```

### How to revert

```bash
cd <worktree>/local-deps/dgr-clean
git checkout -- cuda_rasterizer/stopthepop/hierarchical_render.cuh
touch cuda_rasterizer/forward.cu && python setup.py build_ext --inplace
```

### Why not commit this upstream

The extra write costs a store per blend per pixel, which is not free on dense
scenes. Upstream DGR keeps it commented so production renders aren't taxed by
diagnostic-only counters.
