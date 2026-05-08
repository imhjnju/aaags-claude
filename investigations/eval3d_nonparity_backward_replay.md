# eval_3D non-parity backward replay investigation

## Scope

User goal: enable Vulkan full-data AAA-GS end-to-end training in non-parity mode, with eval_3D/proper_ewa, matched training parameters, standard train/test split, and long MCMC densification.

This note covers the core blocker for `eval_3D && !parity_mode` training.

## Evidence

### Spec contracts

`spec/README.md` requires:

- §4 Tile Binner: every active Gaussian must appear in every tile it overlaps.
- §5 Sorter: Gaussians in a tile are sorted by depth.
- §6 Rasterizer: front-to-back alpha compositing with deterministic order.

For training, rasterizer backward must replay the exact forward compositing order; otherwise `T`, accumulated color, and alpha gradients correspond to a different image formation process.

### Current forward non-parity path

`harmonyos_3dgs/src/vulkan/shaders/rasterize.comp` eval_3D path has two behaviors:

- `spec_disable_subtile_resort == 1`: parity mode writes an identity sub-tile permutation and streams the raw `values_sorted` batch order.
- `spec_disable_subtile_resort == 0`: non-parity mode builds a per-subtile depth permutation, streams candidates through a bounded HEAD buffer, and flushes the shallowest entry when HEAD overflows.

Relevant implementation anchors:

- Sub-tile permutation construction: `rasterize.comp` around the `sub_order_packed` / `sub_count` path.
- HEAD insertion and overflow flush: `rasterize.comp` around `kbuf_depth`, `kbuf_alpha`, and `kbuf_size`.
- Final outputs store only `transmittance[px]` and `n_contrib[px]`.

### Current eval_3D backward path

`harmonyos_3dgs/src/vulkan/shaders/rasterize_backward_eval3d.comp` replays backwards by iterating raw `values_sorted` in reverse and using `n_contrib` as the last-contributor boundary.

That is valid for the parity/identity-order path, but it does not reconstruct the non-parity forward order because non-parity forward changes candidate order through:

1. per-subtile depth sorting;
2. per-pixel HEAD insertion sorting;
3. HEAD overflow flushing.

The backward shader currently does not receive either the final per-pixel blended Gaussian sequence or enough forward sideband state to invert that sequence cheaply.

### Existing guard is justified

`VulkanTrainer::step()` rejects `tcfg.eval_3D && !tcfg.parity_mode`. This prevents training with a backward replay order that can differ from the forward order.

Existing tests also encode this limitation: `VulkanTrainer.Eval3DStepRequiresParityMode` expects the throw. Current P5/P6/P7/L1 gates validate eval_3D parity-mode behavior, not non-parity backward replay correctness.

## Conclusion

It is unsafe to simply remove the non-parity guard. A correct implementation needs one of these evidence-backed mechanisms:

1. Store enough forward per-pixel contribution order for backward to replay exactly; or
2. Implement a slow deterministic backward replay that recomputes the non-parity forward candidate/HEAD process per pixel and emits the reverse contribution sequence; or
3. Change forward non-parity training to a replayable algorithm and prove it matches the intended AAA-GS non-parity behavior.

For this worktree, the safe immediate implementation is to expose `--parity_mode` explicitly and keep `eval_3D && !parity_mode` guarded until exact non-parity replay is implemented and tested.

## Next implementation target

- Expose `--parity_mode` in `gs3d_vk_train` so parity is no longer implicitly tied to `--eval_3d`.
- Keep the guard for `eval_3D && !parity_mode` with an error message that names the missing HEAD/sub-tile replay.
- Extend harness config to record the active parity mode.
- Do not claim non-parity eval_3D training is complete until a non-parity replay test replaces the current throw test.
