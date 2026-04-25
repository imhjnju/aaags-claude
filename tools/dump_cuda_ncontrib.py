#!/usr/bin/env python3
"""Dump CUDA's per-pixel n_contrib buffer via diff-gaussian-rasterization's
materialize_dump entry point. Used for VK-vs-CUDA parity investigation.

IMPORTANT: set PYTHONPATH to the local clean DGR build before invoking:
  PYTHONPATH=/home/robota/.../white-table/local-deps/dgr-clean python ...
otherwise the instrumented shared build is used (100x slower).
"""

import sys
import os
import math
import json
import argparse

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKTREE_ROOT = os.path.dirname(_SCRIPT_DIR)
_MAIN_REPO = os.path.abspath(os.path.join(_WORKTREE_ROOT, os.pardir, os.pardir, os.pardir))
_AAA_DIR = os.path.join(_MAIN_REPO, "AAA-Gaussians")
sys.path.insert(0, _AAA_DIR)

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    ExtendedSettings,
    _C,
)
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def camera_from_json_entry(entry, device="cuda"):
    rot_c2w = np.array(entry["rotation"], dtype=np.float64)
    pos_c2w = np.array(entry["position"], dtype=np.float64)
    C2W = np.eye(4, dtype=np.float64)
    C2W[:3, :3] = rot_c2w
    C2W[:3, 3] = pos_c2w
    W2C = np.linalg.inv(C2W)
    R = W2C[:3, :3].T.astype(np.float32)
    T = W2C[:3, 3].astype(np.float32)
    W, H, fx, fy = entry["width"], entry["height"], entry["fx"], entry["fy"]
    fovx = 2.0 * math.atan(W / (2.0 * fx))
    fovy = 2.0 * math.atan(H / (2.0 * fy))
    world_view_transform = (
        torch.tensor(getWorld2View2(R, T), dtype=torch.float32).transpose(0, 1).to(device)
    )
    projection_matrix = (
        getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).to(device)
    )
    full_proj = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    return MiniCam(
        width=W, height=H, fovy=fovy, fovx=fovx, znear=0.01, zfar=100.0,
        world_view_transform=world_view_transform, full_proj_transform=full_proj,
    )


def _find_basket_aaa_ply():
    """Mirror tests/test_data_paths.h::find_basket_aaa_ply for the tooling side.
    Search env var, current worktree, master root, and known sibling worktrees.
    Returns "" if none exist (caller should fall back to argparse explicit --ply).
    """
    env = os.environ.get("BASKET_AAA_PLY", "")
    if env and os.path.exists(env):
        return env
    candidates = [
        os.path.join(_WORKTREE_ROOT, "basket-aaa.ply"),
        "/home/robota/h00813233/Graph/aaags-claude/basket-aaa.ply",
        "/home/robota/h00813233/Graph/AAA-Gaussians/basket-aaa.ply",
        "/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/vulkan_3d/basket-aaa.ply",
        "/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/training/basket-aaa.ply",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cam_id", type=int, default=0)
    p.add_argument("--ply", type=str,
                   default=_find_basket_aaa_ply(),
                   help="Path to basket-aaa.ply. If omitted, searches "
                        "$BASKET_AAA_PLY / current worktree / master / known sibling worktrees.")
    p.add_argument("--cameras", type=str,
                   default="/home/robota/Downloads/basketball/_sp0_dump_output/cameras.json")
    p.add_argument("--config", type=str,
                   default=os.path.join(_AAA_DIR, "configs", "aaa.json"))
    p.add_argument("--out", type=str,
                   default=os.path.join(_SCRIPT_DIR, "out", "cuda_n_contrib_cam0.raw"))
    args = p.parse_args()

    print(f"Loading Gaussians: {args.ply}")
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(args.ply)
    print(f"  {gaussians.get_xyz.shape[0]:,} Gaussians | filter_3D={gaussians.filter_3D is not None}")

    with open(args.cameras) as f:
        cams_j = json.load(f)
    cam = camera_from_json_entry(cams_j[args.cam_id])
    W, H = int(cam.image_width), int(cam.image_height)
    print(f"  Camera {args.cam_id}: {W}x{H}")

    splat_args = ExtendedSettings.from_json(args.config)

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    campos = cam.world_view_transform.inverse()[3, :3]
    inv_projmatrix = cam.full_proj_transform.inverse()

    means3D = gaussians.get_xyz
    P = means3D.shape[0]
    opacity = gaussians.get_opacity
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    shs = gaussians.get_features
    filter3D = gaussians.filter_3D

    # Empty placeholders for precomputed color / cov3D (we pass SH and scales/rot instead)
    empty = torch.Tensor([]).to("cuda")
    colors_precomp = empty
    cov3Ds_precomp = empty

    # Mirror _RasterizeGaussians.forward arg tuple exactly
    cfg_dict = splat_args.to_dict()
    fwd_args = (
        bg, means3D, colors_precomp, opacity, scales, rotations, filter3D,
        1.0,  # scale_modifier
        cov3Ds_precomp,
        cam.world_view_transform, cam.full_proj_transform, inv_projmatrix,
        math.tan(cam.FoVx * 0.5), math.tan(cam.FoVy * 0.5),
        H, W,
        shs,
        gaussians.active_sh_degree,
        campos,
        False,       # prefiltered
        cfg_dict,
        False,       # render_depth
        False,       # debug
    )

    print("Rasterizing...")
    torch.cuda.synchronize()
    import time
    t0 = time.time()
    num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = \
        _C.rasterize_gaussians(*fwd_args)
    torch.cuda.synchronize()
    print(f"  forward: {time.time() - t0:.2f}s, num_rendered (R)={int(num_rendered)}")

    num_tiles = math.ceil(W / 16) * math.ceil(H / 16)
    print(f"  num_tiles={num_tiles}, P={P}")

    # Both requires_cov3D_inv and requires_gauss2screen are true in eval_3D+new_aabb mode
    # (aaa.json has eval_3D=true and new_aabb=true); these control the chunk layout in
    # GeometryState.fromChunk. Passing false when the buffers were allocated with true
    # gives garbage indices.
    out = _C.materialize_dump(
        geomBuffer, binningBuffer, imgBuffer,
        int(P), int(num_rendered), int(num_tiles), int(H), int(W),
        True,  # requires_cov3D_inv
        True,  # requires_gauss2screen
    )
    n_contrib = out["rasterize_n_contrib"].to("cpu").to(torch.int32).numpy()
    assert n_contrib.size == H * W, f"n_contrib size {n_contrib.size} != H*W={H*W}"

    # Save as int32 little-endian to match VK dump format
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_contrib.astype("<i4").tofile(args.out)
    print(f"  dumped n_contrib int32 -> {args.out} ({n_contrib.size} elements)")

    # Also dump per-Gaussian preprocess outputs for VK-vs-CUDA diff.
    out_dir = os.path.dirname(args.out)
    def dump_t(name, key):
        arr = out[key].to("cpu").contiguous().numpy().astype("<f4")
        path = os.path.join(out_dir, f"cuda_pre_{name}.raw")
        arr.tofile(path)
        print(f"  dumped preprocess_{name} -> {path} shape={arr.shape}")
    dump_t("rgb",             "preprocess_rgb")
    dump_t("conic_opacity",   "preprocess_conic_opacity")  # [P,4] (a,b,c,opacity)
    dump_t("means2D",         "preprocess_means2D")
    dump_t("depths",          "preprocess_depths")
    # Also dump tiles_touched (uint32) — authoritative "was rendered" signal.
    tt = out["preprocess_tiles_touched"].to("cpu").contiguous().numpy().astype("<u4")
    path = os.path.join(out_dir, "cuda_pre_tiles_touched.raw")
    tt.tofile(path)
    print(f"  dumped preprocess_tiles_touched -> {path} shape={tt.shape}")

    # Sort outputs: per-tile Gaussian sequence + ranges
    if "sort_values_sorted" in out:
        sv = out["sort_values_sorted"].to("cpu").contiguous().numpy().astype("<u4")
        sv.tofile(os.path.join(out_dir, "cuda_bin_values_sorted.raw"))
        print(f"  dumped sort_values_sorted -> cuda_bin_values_sorted.raw shape={sv.shape}")
    if "sort_keys_sorted" in out:
        sk = out["sort_keys_sorted"].to("cpu").contiguous().numpy().astype("<u8")
        sk.tofile(os.path.join(out_dir, "cuda_bin_keys_sorted.raw"))
        print(f"  dumped sort_keys_sorted -> cuda_bin_keys_sorted.raw shape={sk.shape}")
    if "sort_tile_ranges" in out:
        tr = out["sort_tile_ranges"].to("cpu").contiguous().numpy().astype("<u4")
        tr.tofile(os.path.join(out_dir, "cuda_bin_tile_ranges.raw"))
        print(f"  dumped sort_tile_ranges -> cuda_bin_tile_ranges.raw shape={tr.shape}")

    # Stats: ROI vs full-image
    n_img = n_contrib.reshape(H, W)
    roi = n_img[720:960, 0:160]  # bottom-left 160x240
    print(f"\n[CUDA_NContrib ROI] shape={roi.shape} pixels={roi.size}")
    print(f"[CUDA_NContrib ROI] n_contrib p5={int(np.percentile(roi, 5))} "
          f"p50={int(np.percentile(roi, 50))} p95={int(np.percentile(roi, 95))} "
          f"max={int(roi.max())}")
    # Histogram in bins of 5 up to 100, then 100+
    bins = list(range(0, 101, 5)) + [10**9]
    hist, _ = np.histogram(roi, bins=bins)
    print("[CUDA_NContrib ROI] Histogram (bin width=5):")
    for i, c in enumerate(hist):
        lo = bins[i]; hi = bins[i+1]
        label = f"[{lo:3d}, inf)" if hi == 10**9 else f"[{lo:3d},{hi:3d})"
        print(f"  {label}: {int(c)} pixels")

    print(f"\n[CUDA_NContrib FULL] pixels={n_img.size}  "
          f"p5={int(np.percentile(n_img, 5))} p50={int(np.percentile(n_img, 50))} "
          f"p95={int(np.percentile(n_img, 95))} max={int(n_img.max())}")


if __name__ == "__main__":
    main()
