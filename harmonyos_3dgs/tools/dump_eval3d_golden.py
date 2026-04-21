#!/usr/bin/env python3
"""dump_eval3d_golden.py — Generate CUDA eval_3D golden data for the tiny fixture.

Loads the tiny fixture inputs, runs the AAA-Gaussians CUDA rasterizer with
eval_3D=True and filter3D from the fixture, and saves the rendered image as
rasterize_image_eval3d.npy in [3,H,W] CHW float32 format.

Usage:
    /home/robota/miniconda3/envs/aaa-gs/bin/python3.10 tools/dump_eval3d_golden.py \
        [--fixture-dir tests/golden/tiny/step000001/cam0000] \
        [--output tests/golden/tiny/step000001/cam0000/rasterize_image_eval3d.npy]

Matrix convention notes:
    The fixture npy files store matrices in exactly the format expected by the
    CUDA GaussianRasterizationSettings API:
      - input_viewmatrix.npy  = world_view_transform = W2C.T   (cameras.py line 54)
      - input_projmatrix.npy  = full_proj_transform  = wvt @ proj_t (cameras.py line 56)
      - input_inv_viewprojmatrix.npy = inverse of full_proj_transform

    The inputs are already activated (sigmoid opacity, exp scales, normalized quats).
    The CUDA rasterizer is called with these directly — no re-activation needed.
"""

import argparse
import os
import sys

sys.path.insert(0, '/home/robota/h00813233/Graph/AAA-Gaussians/submodules/diff-gaussian-rasterization')

import numpy as np
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings, GaussianRasterizer, ExtendedSettings
)


def load_fixture(fixture_dir: str) -> dict:
    """Load tiny fixture npy files. Returns activated Gaussian params + camera data."""
    d = fixture_dir

    # Gaussian parameters (already activated: opacity=sigmoid(raw), scales=exp(raw),
    # rotations=normalized_quat, sh=raw coefficients)
    positions  = np.load(os.path.join(d, "input_positions.npy")).astype(np.float32)   # [N, 3]
    scales     = np.load(os.path.join(d, "input_scales.npy")).astype(np.float32)      # [N, 3]
    rotations  = np.load(os.path.join(d, "input_rotations.npy")).astype(np.float32)   # [N, 4]
    opacities  = np.load(os.path.join(d, "input_opacities.npy")).astype(np.float32)   # [N, 1]
    sh_coeffs  = np.load(os.path.join(d, "input_sh.npy")).astype(np.float32)          # [N, 16, 3]
    filter_3D  = np.load(os.path.join(d, "input_filter_3D.npy")).astype(np.float32)   # [N]

    # Camera matrices — stored in the exact format GaussianRasterizationSettings expects:
    #   viewmatrix = world_view_transform = W2C.T  (cameras.py: getWorld2View2(R,T).T)
    #   projmatrix = full_proj_transform  = wvt @ proj_t (cameras.py line 56)
    #   inv_viewprojmatrix = inverse of full_proj_transform
    viewmatrix     = np.load(os.path.join(d, "input_viewmatrix.npy")).astype(np.float32)     # [4, 4]
    projmatrix     = np.load(os.path.join(d, "input_projmatrix.npy")).astype(np.float32)     # [4, 4]
    inv_vp         = np.load(os.path.join(d, "input_inv_viewprojmatrix.npy")).astype(np.float32)  # [4, 4]
    campos         = np.load(os.path.join(d, "input_campos.npy")).astype(np.float32)         # [3]
    fov_size       = np.load(os.path.join(d, "input_fov_size.npy")).astype(np.float32)       # [4]
    meta           = np.load(os.path.join(d, "input_meta.npy")).astype(np.float32)           # [4]

    # Parse meta / fov_size
    # meta = [sh_degree, max_coeffs, H, W]
    # fov_size = [tan_fovx, tan_fovy, W, H]
    sh_degree  = int(meta[0])
    H          = int(meta[2])
    W          = int(meta[3])
    tan_fovx   = float(fov_size[0])
    tan_fovy   = float(fov_size[1])

    return dict(
        N=positions.shape[0],
        H=H, W=W,
        sh_degree=sh_degree,
        tan_fovx=tan_fovx,
        tan_fovy=tan_fovy,
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        sh_coeffs=sh_coeffs,
        filter_3D=filter_3D,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        inv_viewprojmatrix=inv_vp,
        campos=campos,
    )


def render_with_eval3d(data: dict) -> np.ndarray:
    """Run the CUDA rasterizer with eval_3D=True. Returns [3, H, W] float32 CHW image."""
    device = "cuda"

    # Move everything to CUDA tensors
    means3D   = torch.tensor(data["positions"],  dtype=torch.float32, device=device)   # [N, 3]
    scales_t  = torch.tensor(data["scales"],     dtype=torch.float32, device=device)   # [N, 3]
    rots_t    = torch.tensor(data["rotations"],  dtype=torch.float32, device=device)   # [N, 4]
    opacs_t   = torch.tensor(data["opacities"],  dtype=torch.float32, device=device)   # [N, 1]
    shs_t     = torch.tensor(data["sh_coeffs"],  dtype=torch.float32, device=device)   # [N, 16, 3]
    # filter_3D: the CUDA rasterizer expects [N, 1] or [N]
    filter3d_t = torch.tensor(
        data["filter_3D"].reshape(-1, 1), dtype=torch.float32, device=device          # [N, 1]
    )

    # Camera tensors — pass as-is from fixture (already in world_view_transform / full_proj form)
    viewmatrix_t  = torch.tensor(data["viewmatrix"],         dtype=torch.float32, device=device)
    projmatrix_t  = torch.tensor(data["projmatrix"],         dtype=torch.float32, device=device)
    inv_vp_t      = torch.tensor(data["inv_viewprojmatrix"], dtype=torch.float32, device=device)
    campos_t      = torch.tensor(data["campos"],             dtype=torch.float32, device=device)

    bg = torch.zeros(3, dtype=torch.float32, device=device)

    # ExtendedSettings with eval_3D=True
    splat_args = ExtendedSettings()
    splat_args.eval_3D = True

    raster_settings = GaussianRasterizationSettings(
        image_height=data["H"],
        image_width=data["W"],
        tanfovx=data["tan_fovx"],
        tanfovy=data["tan_fovy"],
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewmatrix_t,
        projmatrix=projmatrix_t,
        inv_viewprojmatrix=inv_vp_t,
        sh_degree=data["sh_degree"],
        campos=campos_t,
        prefiltered=False,
        settings=splat_args,
        render_depth=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means2D must have requires_grad=True (gradient hook needed by the autograd function)
    means2D = torch.zeros_like(means3D, requires_grad=True)

    with torch.no_grad():
        rendered, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs_t,
            colors_precomp=None,
            opacities=opacs_t,
            scales=scales_t,
            rotations=rots_t,
            filter3D=filter3d_t,
            cov3D_precomp=None,
        )

    # rendered: [3, H, W] float32 tensor
    return rendered.detach().cpu().numpy()


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_fixture = os.path.join(
        repo_root, "tests", "golden", "tiny", "step000001", "cam0000"
    )
    default_output = os.path.join(default_fixture, "rasterize_image_eval3d.npy")

    parser = argparse.ArgumentParser(
        description="Generate CUDA eval_3D golden for the tiny fixture."
    )
    parser.add_argument(
        "--fixture-dir", default=default_fixture,
        help=f"Fixture directory (default: {default_fixture})"
    )
    parser.add_argument(
        "--output", default=default_output,
        help=f"Output npy path (default: {default_output})"
    )
    args = parser.parse_args()

    print(f"Loading fixture from: {args.fixture_dir}")
    data = load_fixture(args.fixture_dir)
    print(f"  N={data['N']} Gaussians, {data['W']}x{data['H']}, SH degree={data['sh_degree']}")
    print(f"  tan_fovx={data['tan_fovx']:.4f}, tan_fovy={data['tan_fovy']:.4f}")
    print(f"  filter_3D range: [{data['filter_3D'].min():.4f}, {data['filter_3D'].max():.4f}]")
    print(f"  campos: {data['campos']}")

    print("Running CUDA rasterizer with eval_3D=True ...")
    rendered = render_with_eval3d(data)

    print(f"  rendered: shape={rendered.shape}, dtype={rendered.dtype}")
    print(f"  min={rendered.min():.6f}, max={rendered.max():.6f}")
    print(f"  mean={rendered.mean():.6f}, sum={rendered.sum():.4f}")
    non_zero = np.sum(rendered > 1e-6)
    print(f"  non-zero pixels: {non_zero} / {rendered.size}")

    # Compare with existing rasterize_image.npy (no eval_3D) if present
    no_eval3d_path = os.path.join(args.fixture_dir, "rasterize_image.npy")
    if os.path.exists(no_eval3d_path):
        ref = np.load(no_eval3d_path)
        diff = np.abs(rendered - ref)
        print(f"\n  vs rasterize_image.npy (no eval_3D):")
        print(f"    max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
        print(f"    ref sum={ref.sum():.4f}, eval3d sum={rendered.sum():.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, rendered)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
