#!/usr/bin/env python3
"""Regenerate the tiny-fixture 2D preprocess golden NPYs.

The existing tiny golden (preprocess_conic_opacity.npy etc.) was produced
before commits ed303bd / 6ff257b made Vulkan's 2D preprocess unconditionally
apply proper_ewa_scaling + tight_opacity_bounding + rect_bounding. This
script regenerates the goldens from CUDA with the same settings the Vulkan
path uses, so test_preprocess_pass_vk::MatchesCUDAGolden_Tiny agrees.

Usage:
    /home/robota/miniconda3/envs/aaa-gs/bin/python3.10 \
        tools/dump_tiny_preprocess_golden.py \
        [--fixture-dir tests/golden/tiny/step000001/cam0000]

Requires PYTHONPATH to point at the clean local DGR build (the shared
submodule is instrumented for S10 cascade tracing and ~100x slower).
"""

import argparse
import os
import sys
import math
import numpy as np
import torch


def load_fixture(d: str) -> dict:
    positions  = np.load(os.path.join(d, "input_positions.npy")).astype(np.float32)
    scales     = np.load(os.path.join(d, "input_scales.npy")).astype(np.float32)
    rotations  = np.load(os.path.join(d, "input_rotations.npy")).astype(np.float32)
    opacities  = np.load(os.path.join(d, "input_opacities.npy")).astype(np.float32)
    sh_coeffs  = np.load(os.path.join(d, "input_sh.npy")).astype(np.float32)
    filter_3D  = np.load(os.path.join(d, "input_filter_3D.npy")).astype(np.float32)
    viewmatrix = np.load(os.path.join(d, "input_viewmatrix.npy")).astype(np.float32)
    projmatrix = np.load(os.path.join(d, "input_projmatrix.npy")).astype(np.float32)
    inv_vp     = np.load(os.path.join(d, "input_inv_viewprojmatrix.npy")).astype(np.float32)
    campos     = np.load(os.path.join(d, "input_campos.npy")).astype(np.float32)
    fov_size   = np.load(os.path.join(d, "input_fov_size.npy")).astype(np.float32)
    meta       = np.load(os.path.join(d, "input_meta.npy")).astype(np.float32)

    return dict(
        positions=positions, scales=scales, rotations=rotations,
        opacities=opacities, sh_coeffs=sh_coeffs, filter_3D=filter_3D,
        viewmatrix=viewmatrix, projmatrix=projmatrix, inv_vp=inv_vp, campos=campos,
        sh_degree=int(meta[0]), H=int(meta[2]), W=int(meta[3]),
        tan_fovx=float(fov_size[0]), tan_fovy=float(fov_size[1]),
        N=positions.shape[0],
    )


def main():
    ap = argparse.ArgumentParser()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_fix = os.path.join(repo_root, "tests", "golden", "tiny", "step000001", "cam0000")
    ap.add_argument("--fixture-dir", default=default_fix)
    args = ap.parse_args()

    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings, GaussianRasterizer, ExtendedSettings, _C,
    )

    d = load_fixture(args.fixture_dir)
    print(f"Fixture: {args.fixture_dir}")
    print(f"  N={d['N']} {d['W']}x{d['H']} SH_deg={d['sh_degree']}")

    device = "cuda"
    means3D   = torch.tensor(d["positions"], device=device)
    scales_t  = torch.tensor(d["scales"],    device=device)
    rots_t    = torch.tensor(d["rotations"], device=device)
    opacs_t   = torch.tensor(d["opacities"], device=device)
    shs_t     = torch.tensor(d["sh_coeffs"], device=device)
    filter3d_t = torch.tensor(d["filter_3D"].reshape(-1, 1), device=device)
    viewmat_t = torch.tensor(d["viewmatrix"], device=device)
    projmat_t = torch.tensor(d["projmatrix"], device=device)
    invvp_t   = torch.tensor(d["inv_vp"],    device=device)
    campos_t  = torch.tensor(d["campos"],    device=device)
    bg = torch.zeros(3, device=device)

    # 2D path with proper_ewa_scaling + tight_opacity_bounding + rect_bounding
    # — matches Vulkan preprocess.comp's current behavior.
    splat = ExtendedSettings()
    splat.eval_3D = False
    splat.proper_ewa_scaling = True
    splat.culling_settings.tight_opacity_bounding = True
    splat.culling_settings.rect_bounding = True
    # tile_based_culling left default (false) for now; if radii/tiles_touched still
    # mismatch, we can try enabling it too.

    rs = GaussianRasterizationSettings(
        image_height=d["H"], image_width=d["W"],
        tanfovx=d["tan_fovx"], tanfovy=d["tan_fovy"],
        bg=bg, scale_modifier=1.0,
        viewmatrix=viewmat_t, projmatrix=projmat_t, inv_viewprojmatrix=invvp_t,
        sh_degree=d["sh_degree"], campos=campos_t,
        prefiltered=False, settings=splat, render_depth=False, debug=False,
    )

    means2D = torch.zeros_like(means3D, requires_grad=True)

    # Invoke low-level so we can grab the opaque buffers
    args_fwd = (
        bg, means3D, torch.Tensor([]).to(device), opacs_t, scales_t, rots_t, filter3d_t,
        1.0, torch.Tensor([]).to(device),
        viewmat_t, projmat_t, invvp_t,
        d["tan_fovx"], d["tan_fovy"], d["H"], d["W"],
        shs_t, d["sh_degree"], campos_t, False,
        splat.to_dict(), False, False,
    )
    torch.cuda.synchronize()
    num_rendered, color, radii, geomBuf, binBuf, imgBuf = _C.rasterize_gaussians(*args_fwd)
    torch.cuda.synchronize()
    P = int(d["N"])
    num_tiles = math.ceil(d["W"] / 16) * math.ceil(d["H"] / 16)
    print(f"  R={int(num_rendered)} P={P} num_tiles={num_tiles}")

    # requires_cov3D_inv is false for sort_mode=GLOBAL (the default) + sort_order=VIEWSPACE_Z
    # (the default).  requires_gauss2screen is splat.eval_3D (false here).
    out = _C.materialize_dump(geomBuf, binBuf, imgBuf,
                               P, int(num_rendered), num_tiles, d["H"], d["W"],
                               False, False)

    outdir = args.fixture_dir
    def save(name, key, cast=None):
        t = out[key].detach().cpu().numpy()
        if cast is not None:
            t = t.astype(cast)
        path = os.path.join(outdir, f"{name}.npy")
        np.save(path, t)
        print(f"  wrote {name}.npy  shape={t.shape} dtype={t.dtype}")

    save("preprocess_means2D",        "preprocess_means2D",        np.float32)
    save("preprocess_depths",         "preprocess_depths",         np.float32)
    save("preprocess_conic_opacity",  "preprocess_conic_opacity",  np.float32)
    save("preprocess_rgb",            "preprocess_rgb",            np.float32)
    save("preprocess_tiles_touched",  "preprocess_tiles_touched",  np.uint32)
    save("preprocess_point_offsets",  "preprocess_point_offsets",  np.uint32)
    # radii isn't in materialize_dump — it's the top-level tensor returned from
    # rasterize_gaussians.  Save that.
    np.save(os.path.join(outdir, "preprocess_radii.npy"),
            radii.detach().cpu().numpy().astype(np.int32))
    print(f"  wrote preprocess_radii.npy  shape={radii.shape} dtype=int32")

    # Rasterize-stage goldens (image + n_contrib + transmittance). The rendered
    # color is the top-level return value; n_contrib and transmittance come from
    # materialize_dump (n_contrib enabled in our local DGR build).
    np.save(os.path.join(outdir, "rasterize_image.npy"),
            color.detach().cpu().numpy().astype(np.float32))
    print(f"  wrote rasterize_image.npy  shape={color.shape} dtype=float32")
    save("rasterize_n_contrib",     "rasterize_n_contrib",     np.uint32)
    save("rasterize_transmittance", "rasterize_transmittance", np.float32)
    # Sort outputs (used by RasterizerVulkan.Rasterize_TinyFixture too)
    save("sort_values_sorted", "sort_values_sorted", np.uint32)
    save("sort_keys_sorted",   "sort_keys_sorted",   np.uint64)
    save("sort_tile_ranges",   "sort_tile_ranges",   np.uint32)


if __name__ == "__main__":
    main()
