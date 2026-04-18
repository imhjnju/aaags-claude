"""Run a tiny synthetic fixture and dump one forward+backward checkpoint.

Step=1, cam=0. Produces all preprocess/sort/rasterize/backward artifacts in
the schema required by SP-0 spec §2.3, committed alongside C++ tests.

Why this lives in tools/ and not tests/: the .npy bytes are the golden —
this module is the generator of record. The C++ round-trip test (T17)
consumes what this emits.

Run via:
    python -m tools.dump_tool --fixture=tiny \\
        --output=harmonyos_3dgs/tests/golden/tiny
"""
import torch

from tools.dump_npy import NpyDumper
from tools.dump_fixtures import build_tiny_scene


def _split_op_tensor(key: str):
    """Split ``'preprocess_means2D'`` -> ``('preprocess', 'means2D')``.

    ``materialize_dump`` returns keys as ``{op}_{tensor_name}`` where
    ``tensor_name`` may itself contain underscores (e.g. ``tile_ranges``),
    so we split on the *first* underscore only.
    """
    idx = key.index("_")
    return key[:idx], key[idx + 1:]


def run_tiny(output_dir: str, seed: int = 42):
    """Execute the tiny fixture forward+backward and dump all artifacts.

    Parameters
    ----------
    output_dir : str
        Root directory for the golden bundle. Files are written to
        ``{output_dir}/step000001/cam0000/``.
    seed : int
        Seed threaded through both ``build_tiny_scene`` (for scene RNG) and
        ``torch.manual_seed`` (for the backward's random ``dL_dout_color``).
    """
    from diff_gaussian_rasterization import (
        _C,
        GaussianRasterizationSettings,
        ExtendedSettings,
        SortSettings,
        SortMode,
        GlobalSortOrder,
        CullingSettings,
    )

    s = build_tiny_scene(seed=seed)
    # Deterministic dL_dout_color for the backward — seeded here (not inside
    # build_tiny_scene) because the gradient is part of the *test input*,
    # not the scene.
    torch.manual_seed(seed)

    settings = ExtendedSettings(
        sort_settings=SortSettings(
            sort_mode=SortMode.GLOBAL,
            sort_order=GlobalSortOrder.Z_DEPTH,
        ),
        culling_settings=CullingSettings(),
        load_balancing=False,
        proper_ewa_scaling=False,
        eval_3D=False,
    )

    rs = GaussianRasterizationSettings(
        image_height=s["H"],
        image_width=s["W"],
        tanfovx=s["tan_fovx"],
        tanfovy=s["tan_fovy"],
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=s["viewmatrix"],
        projmatrix=s["projmatrix"],
        inv_viewprojmatrix=s["inv_viewprojmatrix"],
        sh_degree=s["sh_degree"],
        campos=s["campos"],
        prefiltered=False,
        settings=settings,
        render_depth=False,
        debug=False,
    )

    # --- Forward via raw _C (bypass nn.Module wrapper to get buffers) ----
    colors_precomp = torch.Tensor([]).cuda()
    cov3Ds_precomp = torch.Tensor([]).cuda()

    fw_args = (
        rs.bg,
        s["means3D"],
        colors_precomp,
        s["opacities"],
        s["scales"],
        s["rotations"],
        s["filter_3D"],
        rs.scale_modifier,
        cov3Ds_precomp,
        rs.viewmatrix,
        rs.projmatrix,
        rs.inv_viewprojmatrix,
        rs.tanfovx,
        rs.tanfovy,
        rs.image_height,
        rs.image_width,
        s["sh"],
        rs.sh_degree,
        rs.campos,
        rs.prefiltered,
        rs.settings.to_dict(),
        rs.render_depth,
        rs.debug,
    )

    R, color, radii, geomBuf, binBuf, imgBuf = _C.rasterize_gaussians(*fw_args)
    torch.cuda.synchronize()
    assert R > 0, (
        f"tiny fixture produced R={R} (degenerate scene); sort path not "
        "exercised. Fix the fixture before committing golden bytes."
    )
    print(f"R (num_rendered) = {R}")

    TILE = 16
    nt_x = (s["W"] + TILE - 1) // TILE
    nt_y = (s["H"] + TILE - 1) // TILE
    num_tiles = nt_x * nt_y

    # requires_cov3D_inv=False, requires_gauss2screen=False — we only need
    # the SP-0 §2.3 baseline tensor set for the tiny fixture round-trip.
    out = _C.materialize_dump(
        geomBuf,
        binBuf,
        imgBuf,
        s["means3D"].shape[0],
        R,
        num_tiles,
        s["H"],
        s["W"],
        False,
        False,
    )

    # Keep final outputs that live outside materialize_dump's return map.
    out["rasterize_image"] = color       # [3, H, W], float32
    out["preprocess_radii"] = radii      # [P], int32

    # --- T7: Input tensors for C++ round-trip reconstruction ---------------
    # These are the exact inputs fed into rasterize_gaussians above; the
    # PreprocessorVulkan C++ test rebuilds GaussianData/Camera/RenderConfig
    # from these npys and compares its outputs against preprocess_*.npy.
    # Key names must match the `input_<tensor>` convention consumed by
    # _split_op_tensor (splits on the first '_' → op="input").
    out["input_positions"]          = s["means3D"]            # [N, 3] float32
    out["input_scales"]             = s["scales"]             # [N, 3] float32
    out["input_rotations"]          = s["rotations"]          # [N, 4] float32
    out["input_opacities"]          = s["opacities"]          # [N, 1] float32
    out["input_sh"]                 = s["sh"]                 # [N, M, 3] float32
    out["input_filter_3D"]          = s["filter_3D"]          # [N] float32
    out["input_viewmatrix"]         = s["viewmatrix"]         # [4, 4] float32
    out["input_projmatrix"]         = s["projmatrix"]         # [4, 4] float32
    out["input_inv_viewprojmatrix"] = s["inv_viewprojmatrix"] # [4, 4] float32
    out["input_campos"]             = s["campos"]             # [3]    float32
    # [tan_fovx, tan_fovy, W, H] — order matters (matches C++ consumer).
    out["input_fov_size"] = torch.tensor(
        [s["tan_fovx"], s["tan_fovy"],
         float(s["W"]), float(s["H"])], device="cuda")
    # [sh_degree, sh_coeffs_per_g, H, W] — as floats for uniform dtype.
    out["input_meta"] = torch.tensor(
        [float(s["sh_degree"]), float(s["sh_coeffs_per_g"]),
         float(s["H"]),          float(s["W"])], device="cuda")

    # --- Backward --------------------------------------------------------
    # Fixed random gradient mask via torch.manual_seed above — keeps the
    # golden bytes reproducible across dumper runs.
    dL_dout_color = torch.randn_like(color)

    bw_args = (
        rs.bg,
        s["means3D"],
        radii,
        s["opacities"],
        colors_precomp,       # colors (precomputed) — empty, use SH path
        s["scales"],
        s["rotations"],
        rs.scale_modifier,
        cov3Ds_precomp,       # cov3D_precomp — empty
        rs.viewmatrix,
        rs.projmatrix,
        rs.inv_viewprojmatrix,
        rs.tanfovx,
        rs.tanfovy,
        color,                # pixel_colors (forward output, reused)
        dL_dout_color,
        s["sh"],
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
        d_means2D,
        d_colors,
        d_opacity,
        d_means3D,
        d_cov3D,
        d_sh,
        d_scales,
        d_rotations,
        d_conic,
    ) = _C.rasterize_gaussians_backward_dump(*bw_args)
    torch.cuda.synchronize()

    # --- Dump ------------------------------------------------------------
    dumper = NpyDumper(output_dir, step=1, cam_idx=0, seed=seed)

    # Forward tensors (from materialize_dump + forward return).
    for key, ten in out.items():
        op, tensor = _split_op_tensor(key)
        dumper.dump(op, tensor, ten)

    # Backward input: dL_dout_color (tests need the same gradient input).
    dumper.dump("backward", "dL_dout_color", dL_dout_color)
    # Backward outputs — 9-tuple per rasterize_points.h §RasterizeGaussiansBackwardDumpCUDA.
    dumper.dump("backward", "d_means2D",   d_means2D)
    dumper.dump("backward", "d_colors",    d_colors)
    dumper.dump("backward", "d_opacity",   d_opacity)
    dumper.dump("backward", "d_means3D",   d_means3D)
    dumper.dump("backward", "d_cov3D",     d_cov3D)
    dumper.dump("backward", "d_sh",        d_sh)
    dumper.dump("backward", "d_scales",    d_scales)
    dumper.dump("backward", "d_rotations", d_rotations)
    dumper.dump("backward", "d_conic",     d_conic)

    dumper.finalize()
    print(f"tiny fixture dumped to {output_dir}/step000001/cam0000/")
