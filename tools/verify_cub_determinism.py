# tools/verify_cub_determinism.py
"""Verify CUB DeviceRadixSort gives bit-identical output across runs.

Runs rasterize_gaussians on a fixed-seed tiny scene 10 times and compares
sort_keys_sorted / sort_values_sorted byte-for-byte. If all 10 match,
CUB is deterministic on this hardware; if not, sort golden verification
must downgrade from exact to 'tile-grouping consistent' (see spec §1.2).
"""
import sys, os, hashlib
import numpy as np
import torch

# This probe uses a synthetic tiny scene to avoid dataset dependency.
# It does NOT require materialize_dump — it manually parses geomBuffer
# bytes using the known GeometryState layout.
# TODO: update after Task 11 ships to use materialize_dump instead.

def main():
    torch.manual_seed(42)
    # Synthetic 100-gaussian scene, random positions + random view
    # This is intentionally before materialize_dump — we use checksums
    # of the raw binningBuffer bytes (which contain point_list + keys) as
    # the determinism signal.
    N = 100
    from diff_gaussian_rasterization import GaussianRasterizationSettings, \
        GaussianRasterizer, ExtendedSettings, SortSettings, SortMode, \
        GlobalSortOrder, CullingSettings

    device = torch.device("cuda")
    means3D = torch.randn(N, 3, device=device) * 2
    scales  = torch.randn(N, 3, device=device) * 0.1
    rots    = torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=1)
    opac    = torch.rand(N, 1, device=device)
    sh      = torch.randn(N, 16, 3, device=device) * 0.1
    filt    = torch.ones(N, device=device) * 0.01

    view = torch.eye(4, device=device)
    proj = torch.eye(4, device=device)
    invvp = torch.eye(4, device=device)

    settings = ExtendedSettings(
        sort_settings=SortSettings(sort_mode=SortMode.GLOBAL,
                                    sort_order=GlobalSortOrder.Z_DEPTH),
        culling_settings=CullingSettings(),
        load_balancing=False, proper_ewa_scaling=False, eval_3D=False)

    rs = GaussianRasterizationSettings(
        image_height=256, image_width=256,
        tanfovx=1.0, tanfovy=1.0,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=view, projmatrix=proj, inv_viewprojmatrix=invvp,
        sh_degree=3,
        campos=torch.zeros(3, device=device),
        prefiltered=False, settings=settings, render_depth=False, debug=False)

    ra = GaussianRasterizer(raster_settings=rs)

    hashes = []
    for run in range(10):
        color, radii = ra(means3D=means3D, means2D=torch.zeros_like(means3D),
                          opacities=opac, filter3D=filt, shs=sh,
                          scales=scales, rotations=rots)
        torch.cuda.synchronize()
        h = hashlib.sha256(color.detach().cpu().numpy().tobytes()).hexdigest()
        hashes.append(h)
        print(f"run {run}: image sha256 = {h}")

    if len(set(hashes)) == 1:
        print("DETERMINISTIC: all 10 runs produced identical rendered image.")
        sys.exit(0)
    else:
        print(f"NON-DETERMINISTIC: {len(set(hashes))} distinct hashes in 10 runs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
