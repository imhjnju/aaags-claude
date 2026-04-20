# Captain's Log

## Session 2 — 2026-04-20

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

## Session 1 — 2026-04-16

Project harness initialized for harmonyos_3dgs.

Goals:
- [x] M0 kickoff: dev harness installed (CLAUDE.md, WORKFLOW.md, PROJECT.md, skills, hooks, memory)
- [ ] M0: Verify build passes (`cmake -B build -DBUILD_TESTS=ON && cmake --build build`)
- [ ] M0: Verify test baseline (all 24 unit tests pass)
- [ ] M0: Write first spec section for highest-priority subsystem
