# tools/verify_cub_determinism.py
"""Verify CUB DeviceRadixSort gives bit-identical output across runs.

Hashes the raw ``binningBuffer`` bytes (sorted keys + sorted values) across
10 runs of the AAA-Gaussians CUDA rasterizer with a fixed-seed input scene.
If all 10 hashes match, CUB's radix sort is byte-deterministic on this GPU
for this workload. If they diverge, sort golden verification must downgrade
from exact byte-match to "tile-grouping consistent" (see spec section 1.2).

The probe calls ``_C.rasterize_gaussians`` directly (bypassing the
``nn.Module`` wrapper) so it can access the raw tuple
``(num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer)``
and hash the binningBuffer — which actually contains the sort output —
rather than the rendered image. Image pixels can collapse sort-order
differences (e.g. when all sort keys are unique, radix-sort output is
content-identical regardless of tie-breaking), so hashing the image
would give false DETERMINISTIC verdicts.
"""
import sys
import hashlib


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

    # --- Build a scene that exercises the sort meaningfully -------------
    torch.manual_seed(42)
    device = torch.device("cuda")

    # N=500 so we get tile overlap (many Gaussians per tile -> many sort keys).
    N = 500

    # Gaussians clustered in camera frustum (in front of camera at z=-3).
    means3D = torch.randn(N, 3, device=device) * 0.5 + torch.tensor(
        [0.0, 0.0, -3.0], device=device
    )
    scales = torch.randn(N, 3, device=device) * 0.1
    rots = torch.nn.functional.normalize(
        torch.randn(N, 4, device=device), dim=1
    )
    opac = torch.rand(N, 1, device=device)
    sh = torch.randn(N, 16, 3, device=device) * 0.1
    filt = torch.ones(N, device=device) * 0.01

    # Real view matrix. The rasterizer consumes viewmatrix as 4 columns of
    # 4 floats each read sequentially from PyTorch row-major storage — i.e.
    # it expects the *transpose* of a standard math-notation world-to-view
    # matrix (same convention as AAA-Gaussians' ``world_view_transform``,
    # which is built via ``getWorld2View2(...).transpose(0,1)``). In this
    # transposed layout, translation sits at ``view[3, 0:3]``. Setting
    # ``view[3, 2] = 5.0`` places world origin at camera-space z=+5
    # (camera forward is +Z; in_frustum requires p_view.z > 0.2f). Gaussians
    # at world z=-3 then land at camera-space z = -3 + 5 = +2, safely in
    # front of the camera.
    view = torch.eye(4, device=device)
    view[3, 2] = 5.0

    # Real perspective projection matrix (znear=0.1, zfar=100, fov=90deg).
    # Like view, the rasterizer expects the transposed form. We build P in
    # math convention first, then transpose it. The math form is:
    #   P[0,0]=1/tanfovx, P[1,1]=1/tanfovy,
    #   P[2,2]=zfar/(zfar-znear), P[2,3]=-znear*zfar/(zfar-znear),
    #   P[3,2]=1, P[3,3]=0.
    tanfovx = 1.0
    tanfovy = 1.0
    znear, zfar = 0.1, 100.0
    proj_math = torch.zeros(4, 4, device=device)
    proj_math[0, 0] = 1.0 / tanfovx
    proj_math[1, 1] = 1.0 / tanfovy
    proj_math[2, 2] = zfar / (zfar - znear)
    proj_math[2, 3] = -(znear * zfar) / (zfar - znear)
    proj_math[3, 2] = 1.0
    proj_math[3, 3] = 0.0
    proj = proj_math.transpose(0, 1).contiguous()

    # Full view-projection in the rasterizer's transposed convention:
    # ``full = view @ proj`` (both already transposed), matching
    # AAA-Gaussians scene/cameras.py.
    viewproj = view @ proj
    invvp = torch.linalg.inv(viewproj)

    # Camera position in world space = inverse-of-transposed-view row 3.
    # (AAA-Gaussians scene/cameras.py: ``world_view_transform.inverse()[3, :3]``.)
    campos = torch.linalg.inv(view)[3, :3].contiguous()

    settings = ExtendedSettings(
        sort_settings=SortSettings(
            sort_mode=SortMode.GLOBAL, sort_order=GlobalSortOrder.Z_DEPTH
        ),
        culling_settings=CullingSettings(),
        load_balancing=False,
        proper_ewa_scaling=False,
        eval_3D=False,
    )

    raster_settings = GaussianRasterizationSettings(
        image_height=256,
        image_width=256,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=view,
        projmatrix=viewproj,
        inv_viewprojmatrix=invvp,
        sh_degree=3,
        campos=campos,
        prefiltered=False,
        settings=settings,
        render_depth=False,
        debug=False,
    )

    # Empty tensors for the "unused" inputs (same pattern the nn.Module uses).
    colors_precomp = torch.Tensor([])
    cov3Ds_precomp = torch.Tensor([])

    args = (
        raster_settings.bg,
        means3D,
        colors_precomp,
        opac,
        scales,
        rots,
        filt,
        raster_settings.scale_modifier,
        cov3Ds_precomp,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
        raster_settings.inv_viewprojmatrix,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.image_height,
        raster_settings.image_width,
        sh,
        raster_settings.sh_degree,
        raster_settings.campos,
        raster_settings.prefiltered,
        raster_settings.settings.to_dict(),
        raster_settings.render_depth,
        raster_settings.debug,
    )

    # --- Untimed warmup (one pass) --------------------------------------
    num_rendered, _, _, _, _, _ = _C.rasterize_gaussians(*args)
    torch.cuda.synchronize()
    assert num_rendered > 0, (
        f"warmup: num_rendered={num_rendered}; scene does not exercise sort"
    )
    print(f"warmup rendered = {num_rendered} sort entries")

    # --- Measurement loop (10 runs, hash binningBuffer) -----------------
    hashes = []
    for run in range(10):
        (
            num_rendered,
            color,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = _C.rasterize_gaussians(*args)
        torch.cuda.synchronize()

        assert num_rendered > 0, (
            f"run {run}: num_rendered={num_rendered}; sort not exercised"
        )

        h = hashlib.sha256(
            binningBuffer.detach().cpu().numpy().tobytes()
        ).hexdigest()
        hashes.append(h)
        print(
            f"run {run}: rendered={num_rendered} "
            f"binningBuffer sha256={h}"
        )

    if len(set(hashes)) == 1:
        print(
            "DETERMINISTIC: all 10 runs produced identical binningBuffer "
            "(sort output is byte-identical)."
        )
        sys.exit(0)
    else:
        print(
            f"NON-DETERMINISTIC: {len(set(hashes))} distinct binningBuffer "
            "hashes in 10 runs."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
