#!/usr/bin/env python3
"""
Render a single view using the AAA-Gaussians pipeline.

Loads basket-aaa.ply and renders camera ID 0 from harmonyos_3dgs/cameras.json
with all AAA-GS features enabled (configs/aaa.json).

Usage:
    cd <repo_root>
    python tools/render_single.py [--cam_id 0] [--output render_output.png] [--white_bg]
"""

import sys
import os
import argparse
import json
import math

import numpy as np
import torch
import torchvision

# ---------------------------------------------------------------------------
# Path setup — AAA-Gaussians lives in the main repo, not the worktree
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_WORKTREE_ROOT = os.path.dirname(_SCRIPT_DIR)          # …/worktrees/render
_MAIN_REPO    = os.path.abspath(
    os.path.join(_WORKTREE_ROOT, os.pardir, os.pardir, os.pardir)
)                                                        # …/aaags-claude
_AAA_DIR = os.path.join(_MAIN_REPO, "AAA-Gaussians")
if not os.path.isdir(_AAA_DIR):
    # fallback: worktree submodule initialized
    _AAA_DIR = os.path.join(_WORKTREE_ROOT, "AAA-Gaussians")

sys.path.insert(0, _AAA_DIR)

from diff_gaussian_rasterization import ExtendedSettings
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render as gs_render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def camera_from_json_entry(entry: dict, device: str = "cuda") -> MiniCam:
    """
    Reconstruct a MiniCam from a cameras.json entry.

    cameras.json stores the *camera-to-world* (C2W) transform:
      - "rotation": 3×3 C2W rotation matrix
      - "position": camera origin in world space

    Camera expects the *world-to-camera* (W2C) decomposition (R, T) as used
    by getWorld2View2, where W2C = [[R^T | T], [0 | 1]].
    """
    rot_c2w = np.array(entry["rotation"], dtype=np.float64)   # 3×3
    pos_c2w = np.array(entry["position"],  dtype=np.float64)  # (3,)

    C2W = np.eye(4, dtype=np.float64)
    C2W[:3, :3] = rot_c2w
    C2W[:3, 3]  = pos_c2w

    W2C = np.linalg.inv(C2W)
    # getWorld2View2 expects R s.t. W2C[:3,:3] == R.T
    R = W2C[:3, :3].T.astype(np.float32)
    T = W2C[:3, 3].astype(np.float32)

    W = entry["width"]
    H = entry["height"]
    fx = entry["fx"]
    fy = entry["fy"]

    fovx = 2.0 * math.atan(W / (2.0 * fx))
    fovy = 2.0 * math.atan(H / (2.0 * fy))

    znear, zfar = 0.01, 100.0

    world_view_transform = torch.tensor(
        getWorld2View2(R, T), dtype=torch.float32
    ).transpose(0, 1).to(device)

    projection_matrix = getProjectionMatrix(
        znear=znear, zfar=zfar, fovX=fovx, fovY=fovy
    ).transpose(0, 1).to(device)

    full_proj = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)

    return MiniCam(
        width=W,
        height=H,
        fovy=fovy,
        fovx=fovx,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj,
    )


class _Pipeline:
    """Minimal pipeline params matching PipelineParams defaults."""
    convert_SHs_python  = False
    compute_cov3D_python = False
    debug               = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Render one view with AAA-GS features")
    parser.add_argument("--cam_id",   type=int,   default=0,
                        help="Camera ID in cameras.json (default: 0)")
    parser.add_argument("--output",   type=str,   default="render_output.png",
                        help="Output image path (default: render_output.png)")
    parser.add_argument("--white_bg", action="store_true",
                        help="Use white background (default: black)")
    parser.add_argument("--ply",      type=str,
                        default=os.path.join(_WORKTREE_ROOT, "basket-aaa.ply"),
                        help="Path to .ply Gaussian model")
    parser.add_argument("--cameras",  type=str,
                        default=os.path.join(_WORKTREE_ROOT, "harmonyos_3dgs", "cameras.json"),
                        help="Path to cameras.json")
    parser.add_argument("--config",   type=str,
                        default=os.path.join(_AAA_DIR, "configs", "aaa.json"),
                        help="Path to AAA splatting config JSON")
    args = parser.parse_args()

    # ---- Gaussian model ----
    print(f"Loading Gaussians: {args.ply}")
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(args.ply)
    n = gaussians.get_xyz.shape[0]
    print(f"  {n:,} Gaussians | active SH degree: {gaussians.active_sh_degree}"
          f" | filter_3D: {gaussians.filter_3D is not None}")

    # ---- Camera ----
    print(f"Loading cameras: {args.cameras}")
    with open(args.cameras) as f:
        cam_list = json.load(f)
    cam_entry = next((c for c in cam_list if c["id"] == args.cam_id), None)
    if cam_entry is None:
        raise ValueError(f"Camera id={args.cam_id} not found in {args.cameras}")
    print(f"  Camera {args.cam_id}: '{cam_entry['img_name']}' "
          f"{cam_entry['width']}×{cam_entry['height']} "
          f"fx={cam_entry['fx']:.1f} fy={cam_entry['fy']:.1f}")
    camera = camera_from_json_entry(cam_entry)

    # ---- AAA features ----
    print(f"Loading AAA config: {args.config}")
    splat_args = ExtendedSettings.from_json(args.config)
    print(f"  proper_ewa_scaling={splat_args.proper_ewa_scaling}"
          f" | eval_3D={splat_args.eval_3D}"
          f" | rect_bounding={splat_args.culling_settings.rect_bounding}")

    # ---- Render ----
    bg = [1, 1, 1] if args.white_bg else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    print("Rendering…")
    with torch.no_grad():
        result = gs_render(camera, gaussians, _Pipeline(), background,
                           splat_args=splat_args)

    img = result["render"]
    visible = result["visibility_filter"].sum().item()
    print(f"  Visible Gaussians: {visible:,} / {n:,}")
    print(f"  Pixel range: [{img.min():.4f}, {img.max():.4f}]")

    # ---- Save ----
    out = args.output
    if not os.path.isabs(out):
        out = os.path.join(_WORKTREE_ROOT, out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torchvision.utils.save_image(img, out)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
