# n_contrib Backward Contract Investigation

## Trigger

Post-merge targeted validation failed in `VkVsPyReference.*` after the Py reference loss was realigned to `worktree-training` semantics. Step 1 forward loss matched (`py=0.0415892`, `vk=0.0415891`), but parameter gradients still had ~0.46–0.55 L2 relative error.

## Evidence

- `rasterize_backward.comp` reads `n_contrib[pix]` as `last_contrib` and documents it as CUDA's 1-based last-contributor position in the tile traversal, not a blend count.
- `worktree-training` `rasterize.comp` implements the 2D path by incrementing `contributor_pos` for every candidate before filters, then assigning `n = contributor_pos` only when the candidate actually blends.
- Current merge shader preserved `contributor_pos` in eval_3D raw replay, but the 2D conic path writes `n += 1u` after each blend. This preserves forward color/transmittance but changes the replay boundary consumed by backward.
- The failure pattern matches this contract break: forward loss is aligned while backward gradients are structurally wrong from step 1.

## Relevant spec/reference

- `spec/README.md` §6 requires front-to-back alpha compositing and early termination to match reference behavior.
- `rasterize_backward.comp` lines around the `last_contrib` load define the internal forward/backward replay contract.
- `worktree-training:harmonyos_3dgs/src/vulkan/shaders/rasterize.comp` is the passing reference state for `VkVsPyReference.Step1GradientAndLoss`.

## Fix

Restore the 2D conic path's `n_contrib` value to CUDA-style 1-based candidate position: increment `contributor_pos` for every candidate visited and assign `n = contributor_pos` when that candidate blends. Do not store the number of blended contributors.
