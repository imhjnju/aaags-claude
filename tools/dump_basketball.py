"""Basketball ladder dump — SP-0 T19 step=1 implementation.

Mirrors ``AAA-Gaussians/train.py`` lines 36-148 closely. We re-implement the
training step (not import ``train.py``) because ``train.py`` is a script, not
a module: its loop is tangled with argparse, tensorboard, progress bars, and
network GUI. The cost of a narrow re-implementation is accepting that if
``train.py`` changes its step ordering, this file must follow — guarded by
the dumped artifacts, which are bytewise reproducible from train.py.

Step ordering (must match ``train.py`` exactly):
  (a) ``xyz_lr = gaussians.update_learning_rate(iteration)``
  (b) ``if iteration % 1000 == 0: gaussians.oneupSHdegree()``
  (c) Pick camera *without replacement per epoch*:
        ``if not viewpoint_stack: viewpoint_stack = scene.getTrainCameras().copy()``
        ``cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))``
  (d) ``render_pkg = render(cam, gaussians, pipe, bg, splat_args=...)``
  (e) Loss = (1-λ)·L1 + λ·(1 - fused_ssim) + opacity_reg·|σ| + scale_reg·|s|
  (f) ``image.retain_grad(); loss.backward()``
  (g) Densify (SKIPPED at step=1 because densify_from_iter=5000 per SP-0)
  (h) ``optimizer.step(); optimizer.zero_grad(set_to_none=True)``
  (i) Noise injection: ``randn * op_sigmoid(1-σ) * noise_lr * xyz_lr``,
      then ``bmm(actual_covariance, noise)``, then ``_xyz.add_(noise)``

Artifacts dumped (step N, cam idx K written to ``stepNNNNNN/camKKKK/``):
  * 25 forward+backward tensors from materialize_dump + backward_dump
    (same schema as ``dump_tiny.py``)
  * ``step{N}_cam_index``           int32 [1]
  * ``step{N}_lr_active``           float32 [1]
  * ``step{N}_sh_degree_active``    int32 [1]
  * ``step{N}_l1_scalar``           float32 [1]
  * ``step{N}_ssim_scalar``         float32 [1]   (if fused_ssim present)
  * ``step{N}_adam_{group}_m``      float32, same shape as param
  * ``step{N}_adam_{group}_v``      float32, same shape as param
  * ``step{N}_adam_{group}_param_after`` float32, post-step parameter
    for group in {xyz, f_dc, f_rest, opacity, scaling, rotation}  (6×3 = 18)
  * ``step{N}_xyz_after_adam``      float32 [P, 3]  (before noise)
  * ``step{N}_xyz_after_noise``     float32 [P, 3]  (after noise)
  * ``step{N}_noise_randn``         float32 [P, 3]  (randn used)
  * ``step{N}_backward_dL_dout_color`` float32 [3, H, W]
"""
import os
import sys
from pathlib import Path
from random import randint, seed as random_seed

import numpy as np
import torch

# Make AAA-Gaussians importable (the submodule lives at repo_root/AAA-Gaussians).
AAA_ROOT = Path(__file__).resolve().parents[1] / "AAA-Gaussians"
if str(AAA_ROOT) not in sys.path:
    sys.path.insert(0, str(AAA_ROOT))

from tools.dump_npy import NpyDumper


def _build_args_namespace(source_path: str, ply_path: str, iterations: int,
                          densify_from_iter: int, densify_until_iter: int,
                          densification_interval: int):
    """Construct argparse.Namespace matching train.py's merged args.

    We invoke ``ModelParams(parser)`` etc. so the defaults come from the
    reference code, then override only what the caller needs. This matches
    ``train.py``'s discipline: never hardcode values that ``arguments/`` owns.
    """
    from argparse import ArgumentParser
    from arguments import ModelParams, OptimizationParams, PipelineParams

    parser = ArgumentParser(add_help=False)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # Parse empty argv to get defaults; we override via Namespace attrs below.
    args = parser.parse_args([])

    # Populate required fields (mirroring train.py's resolved Namespace).
    args.source_path = source_path
    args.model_path = str(Path(source_path) / "_sp0_dump_output")
    args.images = "images"
    args.resolution = -1
    args.white_background = False
    args.data_device = "cuda"
    args.eval = False
    args.sh_degree = 3
    # cap_max is required by train.py's early exit — value unused in
    # step=1 because densify_from_iter > 1.
    args.cap_max = 1_000_000

    # Optimization overrides per SP-0 plan (densify shifted to 5000/10000 so
    # ladder steps 1..2000 do not trigger densification).
    args.iterations = iterations
    args.densify_from_iter = densify_from_iter
    args.densify_until_iter = densify_until_iter
    args.densification_interval = densification_interval
    args.random_background = False

    # Return (lp_group, op_group, pp_group, full_args) — train.py uses
    # extract() groups for passing into training().
    return lp.extract(args), op.extract(args), pp.extract(args), args


def _build_splat_args():
    """Minimal ExtendedSettings for basketball — matches SP-0 sort_mode=GLOBAL.

    AAA-Gaussians's config.json for basketball isn't committed here, so we
    build the settings object directly. GLOBAL + Z_DEPTH is the default
    configuration used for the tiny fixture and the SP-0 golden pipeline.
    """
    from diff_gaussian_rasterization import (
        ExtendedSettings, SortSettings, SortMode, GlobalSortOrder,
        CullingSettings,
    )
    return ExtendedSettings(
        sort_settings=SortSettings(
            sort_mode=SortMode.GLOBAL,
            sort_order=GlobalSortOrder.Z_DEPTH,
        ),
        culling_settings=CullingSettings(),
        load_balancing=False,
        proper_ewa_scaling=False,
        eval_3D=False,
    )


def _op_sigmoid(x: torch.Tensor, k: float = 100.0, x0: float = 0.995):
    """Matches train.py:143 — sharp sigmoid gating for noise injection."""
    return 1.0 / (1.0 + torch.exp(-k * (x - x0)))


def _dump_adam_state(dumper: NpyDumper, optimizer, step: int):
    """Dump Adam m, v, and post-step param for each of the 6 param groups.

    Adam state lives in ``optimizer.state[param]`` keyed by the Parameter
    object itself (not by name). Param groups carry ``'name'`` so we can
    recover the mapping set up by ``GaussianModel.training_setup``.
    """
    for group in optimizer.param_groups:
        name = group.get("name", "unknown")
        # Each group has exactly one parameter (see training_setup).
        param = group["params"][0]
        state = optimizer.state.get(param, {})
        # exp_avg == m, exp_avg_sq == v. After optimizer.step() both exist.
        m = state.get("exp_avg")
        v = state.get("exp_avg_sq")
        if m is None or v is None:
            # Could happen if a group has no gradient this step; record
            # zeros of the right shape so the manifest stays consistent.
            m = torch.zeros_like(param.data)
            v = torch.zeros_like(param.data)
        dumper.dump(f"step{step}", f"adam_{name}_m", m.detach())
        dumper.dump(f"step{step}", f"adam_{name}_v", v.detach())
        dumper.dump(f"step{step}", f"adam_{name}_param_after", param.detach())


def _dump_forward_backward(dumper: NpyDumper, gaussians, viewpoint_cam,
                           splat_args, pipe, bg, image, dL_dout_color):
    """Re-run forward via raw _C to capture buffers + materialize_dump, then
    backward_dump. The autograd ``render()`` used for loss.backward() hides
    the buffers behind the Function boundary; re-running is deterministic
    (same params, same camera, same settings).
    """
    from diff_gaussian_rasterization import (
        _C, GaussianRasterizationSettings,
    )
    import math

    tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)

    rs = GaussianRasterizationSettings(
        image_height=int(viewpoint_cam.image_height),
        image_width=int(viewpoint_cam.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewpoint_cam.world_view_transform,
        projmatrix=viewpoint_cam.full_proj_transform,
        inv_viewprojmatrix=viewpoint_cam.full_proj_transform_inverse,
        sh_degree=gaussians.active_sh_degree,
        campos=viewpoint_cam.camera_center,
        prefiltered=False,
        settings=splat_args,
        render_depth=False,
        debug=pipe.debug,
    )

    colors_precomp = torch.Tensor([]).cuda()
    cov3Ds_precomp = torch.Tensor([]).cuda()
    filter3D = gaussians.filter_3D if gaussians.filter_3D is not None else torch.Tensor([]).cuda()

    fw_args = (
        rs.bg,
        gaussians.get_xyz,
        colors_precomp,
        gaussians.get_opacity,
        gaussians.get_scaling,
        gaussians.get_rotation,
        filter3D,
        rs.scale_modifier,
        cov3Ds_precomp,
        rs.viewmatrix,
        rs.projmatrix,
        rs.inv_viewprojmatrix,
        rs.tanfovx,
        rs.tanfovy,
        rs.image_height,
        rs.image_width,
        gaussians.get_features,
        rs.sh_degree,
        rs.campos,
        rs.prefiltered,
        rs.settings.to_dict(),
        rs.render_depth,
        rs.debug,
    )

    R, color, radii, geomBuf, binBuf, imgBuf = _C.rasterize_gaussians(*fw_args)
    torch.cuda.synchronize()
    assert R > 0, f"basketball forward produced R={R} — sort path empty"
    print(f"  materialize forward: R (num_rendered) = {R}")

    TILE = 16
    nt_x = (rs.image_width + TILE - 1) // TILE
    nt_y = (rs.image_height + TILE - 1) // TILE
    num_tiles = nt_x * nt_y

    out = _C.materialize_dump(
        geomBuf, binBuf, imgBuf,
        gaussians.get_xyz.shape[0],
        R, num_tiles,
        rs.image_height, rs.image_width,
        False, False,
    )
    out["rasterize_image"] = color
    out["preprocess_radii"] = radii

    # Backward dump — fed the same dL_dout_color captured from autograd.
    bw_args = (
        rs.bg,
        gaussians.get_xyz,
        radii,
        gaussians.get_opacity,
        colors_precomp,
        gaussians.get_scaling,
        gaussians.get_rotation,
        rs.scale_modifier,
        cov3Ds_precomp,
        rs.viewmatrix,
        rs.projmatrix,
        rs.inv_viewprojmatrix,
        rs.tanfovx,
        rs.tanfovy,
        color,
        dL_dout_color,
        gaussians.get_features,
        rs.sh_degree,
        rs.campos,
        geomBuf,
        R,
        binBuf,
        imgBuf,
        rs.settings.to_dict(),
        rs.debug,
    )
    (
        d_means2D, d_colors, d_opacity, d_means3D, d_cov3D,
        d_sh, d_scales, d_rotations, d_conic,
    ) = _C.rasterize_gaussians_backward_dump(*bw_args)
    torch.cuda.synchronize()

    # Dump forward (materialize_dump outputs plus the two extras).
    for key, ten in out.items():
        idx = key.index("_")
        dumper.dump(key[:idx], key[idx + 1:], ten)

    # Dump backward outputs.
    dumper.dump("backward", "dL_dout_color", dL_dout_color)
    dumper.dump("backward", "d_means2D",    d_means2D)
    dumper.dump("backward", "d_colors",     d_colors)
    dumper.dump("backward", "d_opacity",    d_opacity)
    dumper.dump("backward", "d_means3D",    d_means3D)
    dumper.dump("backward", "d_cov3D",      d_cov3D)
    dumper.dump("backward", "d_sh",         d_sh)
    dumper.dump("backward", "d_scales",     d_scales)
    dumper.dump("backward", "d_rotations",  d_rotations)
    dumper.dump("backward", "d_conic",      d_conic)


def _scalar_tensor(v, dtype=np.float32):
    """Wrap a Python/torch scalar as a shape-[1] ndarray (not shape-[])."""
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().item()
    return np.asarray([v], dtype=dtype)


def run_basketball(output_dir: str, steps, source_path: str, ply_path: str,
                   iterations: int, seed: int,
                   densify_from_iter: int, densify_until_iter: int,
                   densification_interval: int):
    """Run AAA-Gaussians training for up to ``max(steps)`` iterations.

    Dumps a full artifact set at each step in ``steps``. For SP-0 T19 this
    is validated for ``steps == [1]``; T20 will extend to the full ladder.

    Args:
        output_dir: Golden bundle root (written as
            ``{output_dir}/step{N:06d}/cam{cam_idx:04d}/``).
        steps: Sorted list of step numbers to dump at (1-indexed,
            matching ``train.py``'s iteration variable).
        source_path: Path to COLMAP scene (contains ``sparse/`` +
            ``images/``).
        ply_path: Ignored when Scene.create_from_pcd is used — the PLY is
            loaded by dataset_readers from ``source_path/sparse/0/points3D.ply``.
        iterations: Total training steps (loop upper bound).
        seed: Controls torch, numpy, random; stamped into each manifest.json.
        densify_from_iter, densify_until_iter, densification_interval:
            Passed through to opt; at step=1 all are inert (from=5000>1).
    """
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import l1_loss
    from utils.general_utils import safe_state, build_scaling_rotation
    try:
        from fused_ssim import fused_ssim
    except ImportError:
        print("WARNING: fused_ssim not installed; SSIM artifacts will be "
              "absent from manifest (T22 documents).", file=sys.stderr)
        fused_ssim = None

    # Deterministic seeding. ``safe_state(silent=True)`` redirects stdout's
    # prefix and seeds torch — we still set numpy and random ourselves.
    safe_state(silent=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random_seed(seed)

    # Argparse-equivalent namespaces (lp = model/data, op = optimization,
    # pp = pipeline). Order matches train.py.
    lp, op, pp, raw_args = _build_args_namespace(
        source_path, ply_path, iterations,
        densify_from_iter, densify_until_iter, densification_interval,
    )
    os.makedirs(lp.model_path, exist_ok=True)
    splat_args = _build_splat_args()

    # Scene init (same sequence as train.py:42-44).
    gaussians = GaussianModel(lp.sh_degree)
    scene = Scene(lp, gaussians)
    gaussians.training_setup(op)
    P_init = gaussians.get_xyz.shape[0]
    print(f"basketball scene: P={P_init} gaussians")

    # Background: white if lp.white_background else black. SP-0 uses black.
    bg_color = [1, 1, 1] if lp.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Git commit stamp for manifest provenance.
    try:
        import subprocess
        reference_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=AAA_ROOT, text=True
        ).strip()
    except Exception:
        reference_commit = "unknown"

    viewpoint_stack = None
    steps_set = set(int(s) for s in steps)
    max_step = max(steps_set)
    if iterations < max_step:
        print(f"WARNING: iterations={iterations} < max ladder step={max_step}; "
              "loop will stop before reaching all ladder points.",
              file=sys.stderr)

    for iteration in range(1, iterations + 1):
        # (a) Learning rate update.
        xyz_lr = gaussians.update_learning_rate(iteration)

        # (b) SH degree ramp (every 1000 iters).
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        sh_degree_active = gaussians.active_sh_degree

        # (c) Pick camera without replacement per epoch.
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        cam_idx_in_stack = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(cam_idx_in_stack)
        # `cam_index` recorded below is ``uid`` (stable across runs) — not
        # the position in the shuffled stack. That matches how SP-4 replay
        # will reference cameras.
        cam_uid = int(viewpoint_cam.uid)

        # (d) Render via autograd. We keep ``render_pkg["render"]`` live for
        # loss.backward() — its .grad becomes dL_dout_color at the backward.
        bg = torch.rand((3), device="cuda") if op.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pp, bg, splat_args=splat_args)
        image = render_pkg["render"]
        image.retain_grad()   # capture dL/d(image) for the dump

        # (e) Loss = (1-λ)·L1 + λ·(1 - SSIM) + opacity_reg + scale_reg.
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if fused_ssim is not None:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = torch.tensor(0.0, device="cuda")
        loss = (1.0 - op.lambda_dssim) * Ll1 + op.lambda_dssim * (1.0 - ssim_value)
        loss = loss + raw_args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
        loss = loss + raw_args.scale_reg * torch.abs(gaussians.get_scaling).mean()

        # (f) Backward.
        loss.backward()
        dL_dout_color = image.grad.detach().clone() if image.grad is not None else torch.zeros_like(image)

        # --- If this is a ladder step, dump forward+backward+loss artifacts
        #     BEFORE optimizer.step so param snapshots reflect the
        #     post-adam state via _dump_adam_state after the step.
        should_dump = iteration in steps_set
        if should_dump:
            dumper = NpyDumper(
                output_dir=output_dir,
                step=iteration,
                cam_idx=cam_uid,
                seed=seed,
                reference_commit=reference_commit,
                config_hash="",
            )
            step_tag = f"step{iteration}"
            # Training metadata.
            dumper.dump(step_tag, "cam_index",
                        np.asarray([cam_uid], dtype=np.int32))
            dumper.dump(step_tag, "lr_active",
                        _scalar_tensor(xyz_lr, np.float32))
            dumper.dump(step_tag, "sh_degree_active",
                        np.asarray([sh_degree_active], dtype=np.int32))
            dumper.dump(step_tag, "l1_scalar",
                        _scalar_tensor(Ll1, np.float32))
            if fused_ssim is not None:
                dumper.dump(step_tag, "ssim_scalar",
                            _scalar_tensor(ssim_value, np.float32))
            # Forward+backward CUDA artifacts (25 tensors identical in
            # schema to dump_tiny).
            _dump_forward_backward(
                dumper, gaussians, viewpoint_cam, splat_args, pp, bg,
                image, dL_dout_color,
            )

        with torch.no_grad():
            # (g) Densification — inert at step=1 under SP-0 defaults
            #     (densify_from_iter=5000). Left here for T20 fidelity.
            if iteration < op.densify_until_iter:
                if (iteration > op.densify_from_iter
                        and iteration % op.densification_interval == 0):
                    dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                    gaussians.relocate_gs(viewpoint_cam.camera_center,
                                          dead_mask=dead_mask)
                    gaussians.add_new_gs(cap_max=raw_args.cap_max)

            # (h) Optimizer step + zero_grad.
            # NOTE: train.py guards with ``if iteration < opt.iterations:`` so
            # the last iteration skips adam+noise. For the dump tool we
            # *always* run these when the step is in ``steps_set`` so that a
            # call with ``--iterations=N --ladder=N`` still captures
            # post-adam/post-noise artifacts — "step N" here means "state
            # after executing step N's full update". Non-ladder steps still
            # follow train.py's guard for faithful intermediate state.
            run_update = should_dump or (iteration < op.iterations)
            if run_update:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                if should_dump:
                    _dump_adam_state(dumper, gaussians.optimizer, iteration)
                    dumper.dump(step_tag, "xyz_after_adam",
                                gaussians._xyz.detach())

                # (i) Noise injection (train.py:141-148).
                L = build_scaling_rotation(gaussians.get_scaling,
                                           gaussians.get_rotation)
                actual_covariance = L @ L.transpose(1, 2)
                randn = torch.randn_like(gaussians._xyz)
                noise = randn * (_op_sigmoid(1 - gaussians.get_opacity)) \
                    * raw_args.noise_lr * xyz_lr
                noise = torch.bmm(actual_covariance,
                                  noise.unsqueeze(-1)).squeeze(-1)
                gaussians._xyz.add_(noise)

                if should_dump:
                    dumper.dump(step_tag, "noise_randn", randn.detach())
                    dumper.dump(step_tag, "xyz_after_noise",
                                gaussians._xyz.detach())
                    dumper.finalize()
                    print(f"  dumped step {iteration} -> "
                          f"{output_dir}/step{iteration:06d}/"
                          f"cam{cam_uid:04d}/")

        # Early exit once we've dumped all ladder steps — no point running
        # 2000 iters for ``--ladder=1``.
        if iteration >= max_step:
            break

    print(f"run_basketball complete: dumped steps {sorted(steps_set)}")
