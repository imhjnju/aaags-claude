# tools/smoke_materialize_dump.py
"""Smoke test: exercise ``_C.materialize_dump`` end-to-end.

Drives the AAA-Gaussians CUDA rasterizer with a tiny synthetic scene, then
hands the raw ``geomBuffer`` / ``binningBuffer`` / ``imgBuffer`` to
``_C.materialize_dump`` (bound in Task 11) and asserts that every tensor
promised by the SP-0 spec section 2.3 is present with a sensible shape.

Scene setup mirrors ``tools/verify_cub_determinism.py`` (the CUB determinism
probe from Task 2 / 10): the AAA-Gaussians rasterizer consumes ``viewmatrix``
and ``projmatrix`` in *transposed* form (same convention as
``AAA-Gaussians/scene/cameras.py``'s ``world_view_transform``), so translation
lives at ``view[3, 0:3]`` rather than ``view[0:3, 3]``. A naive
``view[2, 3] = -5.0`` (math-form convention) places the world origin outside
the frustum and ``num_rendered`` collapses to 0 — no sort keys generated,
``sort_*`` tensors empty, and the smoke test becomes vacuous. We instead set
``view[3, 2] = 5.0`` and cluster Gaussians near world z=-3 so camera-space
z ≈ +2 (> the 0.2 near-plane guard in ``in_frustum``).

Image is 64x64 and N=50 to minimize compute; we only care that the Python
binding surface is wired correctly and returns the expected keys.
"""
import math
import sys


def main():
    # --- Preflight ------------------------------------------------------
    try:
        import torch
    except ImportError as e:
        sys.exit(f"Failed to import torch ({e}); are you in conda env aaa-gs?")

    if not torch.cuda.is_available():
        sys.exit("CUDA required (torch.cuda.is_available() is False)")

    try:
        from diff_gaussian_rasterization import (
            _C,
            GaussianRasterizationSettings,
            ExtendedSettings,
            SortSettings,
            SortMode,
            GlobalSortOrder,
            CullingSettings,
        )
    except ImportError as e:
        sys.exit(
            f"Failed to import diff_gaussian_rasterization ({e}); "
            "are you in conda env aaa-gs?"
        )

    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"CUDA version : {torch.version.cuda}")
    print(f"torch version: {torch.__version__}")

    # --- Build a tiny scene that actually renders -----------------------
    torch.manual_seed(0)
    device = torch.device("cuda")
    N = 50

    # Gaussians clustered near world z=-3 (see module docstring).
    means3D = torch.randn(N, 3, device=device) * 0.5
    means3D[:, 2] -= 3.0
    scales = torch.rand(N, 3, device=device) * 0.2
    rots = torch.nn.functional.normalize(
        torch.randn(N, 4, device=device), dim=1
    )
    opac = torch.rand(N, 1, device=device) * 0.5 + 0.3
    sh = torch.randn(N, 16, 3, device=device) * 0.1
    filt = torch.ones(N, device=device) * 0.01

    # Transposed W2V convention per AAA-Gaussians/scene/cameras.py:
    # translation at view[3, 0:3]. view[3, 2] = 5.0 puts world origin at
    # camera-space z=+5; Gaussians at world z=-3 land at cam z=+2 (in
    # frustum, past the 0.2 near-plane guard).
    view = torch.eye(4, device=device)
    view[3, 2] = 5.0

    # Real perspective projection, built in math form then transposed to
    # match the rasterizer's row-major-as-transposed storage convention.
    tan_fov = 1.0
    znear, zfar = 0.1, 100.0
    proj_math = torch.zeros(4, 4, device=device)
    proj_math[0, 0] = 1.0 / tan_fov
    proj_math[1, 1] = 1.0 / tan_fov
    proj_math[2, 2] = zfar / (zfar - znear)
    proj_math[2, 3] = -znear * zfar / (zfar - znear)
    proj_math[3, 2] = 1.0
    proj = proj_math.T.contiguous()

    # Combined view-projection (both already transposed) and its inverse.
    full = (view @ proj).contiguous()
    invvp = torch.linalg.inv(full).contiguous()

    settings = ExtendedSettings(
        sort_settings=SortSettings(
            sort_mode=SortMode.GLOBAL, sort_order=GlobalSortOrder.Z_DEPTH
        ),
        culling_settings=CullingSettings(),
        load_balancing=False,
        proper_ewa_scaling=False,
        eval_3D=False,
    )

    rs = GaussianRasterizationSettings(
        image_height=64,
        image_width=64,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=view,
        projmatrix=full,
        inv_viewprojmatrix=invvp,
        sh_degree=3,
        campos=torch.zeros(3, device=device),
        prefiltered=False,
        settings=settings,
        render_depth=False,
        debug=False,
    )

    # Empty tensors for the "unused" inputs (same pattern the nn.Module uses).
    colors_precomp = torch.Tensor([])
    cov3Ds_precomp = torch.Tensor([])

    args = (
        rs.bg,
        means3D,
        colors_precomp,
        opac,
        scales,
        rots,
        filt,
        rs.scale_modifier,
        cov3Ds_precomp,
        rs.viewmatrix,
        rs.projmatrix,
        rs.inv_viewprojmatrix,
        rs.tanfovx,
        rs.tanfovy,
        rs.image_height,
        rs.image_width,
        sh,
        rs.sh_degree,
        rs.campos,
        rs.prefiltered,
        rs.settings.to_dict(),
        rs.render_depth,
        rs.debug,
    )

    # --- Forward pass (raw _C entry, bypassing the nn.Module wrapper) ---
    (
        num_rendered,
        color,
        radii,
        geomBuffer,
        binningBuffer,
        imgBuffer,
    ) = _C.rasterize_gaussians(*args)
    torch.cuda.synchronize()
    print(f"num_rendered R = {num_rendered}")
    assert num_rendered > 0, (
        f"num_rendered={num_rendered}; scene does not exercise the sort "
        "path. Smoke test is vacuous — fix the scene before trusting it."
    )

    # --- Materialize dump ----------------------------------------------
    TILE = 16
    num_tiles_x = (rs.image_width + TILE - 1) // TILE
    num_tiles_y = (rs.image_height + TILE - 1) // TILE
    num_tiles = num_tiles_x * num_tiles_y
    print(f"num_tiles = {num_tiles} ({num_tiles_x}x{num_tiles_y})")

    # requires_cov3D_inv=False, requires_gauss2screen=False
    out = _C.materialize_dump(
        geomBuffer,
        binningBuffer,
        imgBuffer,
        N,
        num_rendered,
        num_tiles,
        rs.image_height,
        rs.image_width,
        False,
        False,
    )

    print("materialize_dump tensors:")
    for k, v in sorted(out.items()):
        print(f"  {k}: shape={list(v.shape)} dtype={v.dtype}")

    # --- Assert the SP-0 spec §2.3 contract ----------------------------
    required = [
        "preprocess_means2D",
        "preprocess_depths",
        "preprocess_conic_opacity",
        "preprocess_rgb",
        "preprocess_tiles_touched",
        "preprocess_point_offsets",
        "sort_tile_ranges",
        "rasterize_n_contrib",
        "rasterize_transmittance",
    ]
    if num_rendered > 0:
        required += [
            "sort_keys_unsorted",
            "sort_keys_sorted",
            "sort_values_unsorted",
            "sort_values_sorted",
        ]
    missing = [k for k in required if k not in out]
    assert not missing, f"missing tensors: {missing}"
    print("OK: all required tensors present")


if __name__ == "__main__":
    main()
