#!/usr/bin/env python3
# TEMP DIAGNOSTIC — basketball VK-vs-CUDA 10-step parity
"""Dump 10 Adam training steps of CUDA-backed PyTorch (diff_gaussian_rasterization)
on the COLMAP basketball PLY, mirroring VK's test_e2e_basketball.cpp init.

Per-step artifacts to <out_dir>/step_NNNN/:
    loss.npy                    (1,)  float32
    grad_pos.npy                (N,3) float32  d_loss/d_raw_xyz
    grad_sh_dc.npy              (N,1,3) float32 d_loss/d_features_dc
    grad_sh_rest.npy            (N,K-1,3) float32 d_loss/d_features_rest
    grad_op.npy                 (N,1) float32  d_loss/d_raw_opacity
    grad_sca.npy                (N,3) float32  d_loss/d_raw_scaling
    grad_rot.npy                (N,4) float32  d_loss/d_raw_rotation
    param_pos.npy               (N,3) float32  raw_xyz POST-Adam
    param_sh.npy                (N,K,3) float32 [features_dc; features_rest] POST-Adam
    param_op.npy                (N,1) float32  raw_opacity POST-Adam
    param_sca.npy               (N,3) float32  raw_scaling POST-Adam
    param_rot.npy               (N,4) float32  raw_rotation POST-Adam (unnormalized)

Plus a top-level <out_dir>/meta.json describing config.

Adam config:
    lr_pos=1.6e-4, lr_sh_dc=2.5e-3, lr_sh_rest=1.25e-4,
    lr_op=0.05, lr_sca=0.005, lr_rot=0.001
    betas=(0.9, 0.999), eps=1e-15
    proper_ewa_scaling=False, eval_3D=False, lambda_dssim=0
    sh_degree=3, max_coeffs=16, identity rot, init_raw_scale=log(0.03),
    init_raw_opacity=logit(0.1), SH DC = (rgb/255-0.5)/C0

Run with:
    conda activate aaa-gs && \
    python harmonyos_3dgs/tools/dump_basketball_cuda_10step.py --steps 10
"""

import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))           # harmonyos_3dgs
WORKTREE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, os.pardir))        # training
CLAUDE_ROOT = os.path.abspath(os.path.join(WORKTREE_ROOT, os.pardir, os.pardir))
AAA_ROOTS = [
    os.path.join(CLAUDE_ROOT, "AAA-Gaussians"),
    "/home/robota/h00813233/Graph/aaags-claude/AAA-Gaussians",
]
for p in AAA_ROOTS:
    if os.path.isdir(p):
        sys.path.insert(0, p)
        break

from diff_gaussian_rasterization import (  # noqa: E402
    GaussianRasterizationSettings,
    ExtendedSettings,
    GaussianRasterizer,
)


# ----------------------------------------------------------------------------
# COLMAP binary PLY loader (xyz + nxnynz + rgb), matches test_e2e_basketball.cpp
# ----------------------------------------------------------------------------
def load_colmap_ply(path):
    """Return list of (x, y, z, r, g, b uint8) tuples."""
    with open(path, "rb") as f:
        # Parse header
        line = f.readline().decode("ascii", errors="ignore").strip()
        if not line.startswith("ply"):
            raise RuntimeError(f"Not a PLY: {path}")
        n_vertices = 0
        is_binary_le = False
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("format binary_little_endian"):
                is_binary_le = True
            elif line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line == "end_header":
                break
        if not is_binary_le or n_vertices <= 0:
            raise RuntimeError("Unsupported PLY format")

        # Each vertex: float x,y,z,nx,ny,nz (24B) + uchar r,g,b (3B) = 27B
        stride = 24 + 3
        data = f.read(stride * n_vertices)

    pts = np.zeros((n_vertices, 3), dtype=np.float32)
    rgb = np.zeros((n_vertices, 3), dtype=np.uint8)
    for i in range(n_vertices):
        off = i * stride
        x, y, z = struct.unpack_from("<fff", data, off)
        r, g, b = struct.unpack_from("<BBB", data, off + 24)
        pts[i] = (x, y, z)
        rgb[i] = (r, g, b)
    return pts, rgb


# ----------------------------------------------------------------------------
# Camera builder — uses the SAME view/proj/cam_pos that test_e2e_basketball.cpp
# hardcodes, so VK and CUDA see byte-identical inputs.
# ----------------------------------------------------------------------------
def build_camera(W, H, device="cuda"):
    """Match test_e2e_basketball.cpp:272-329.

    cam.view_matrix is COLUMN-MAJOR storage; we reconstruct the row-major W2C
    then build world_view_transform = W2C.T (row-major numpy storage),
    matching how the CUDA-side rasterizer expects (and matching VK Camera
    layout — see dev_notes/.../matrix_layouts comment in
    dump_cuda_training_step.py).
    """
    fx = 706.089
    fy = 707.452

    fovx = 2.0 * math.atan(W / (2.0 * fx))
    fovy = 2.0 * math.atan(H / (2.0 * fy))
    tan_fovx = math.tan(fovx * 0.5)  # = (W/2)/fx
    tan_fovy = math.tan(fovy * 0.5)

    cam_pos = np.array([-0.42047721, 0.27240201, -0.21650667], dtype=np.float32)

    # Row-major W2C from the test:
    W2C = np.array([
        [-0.22993832,  0.03527074,  0.97256586,  0.10427535],
        [-0.38269602, -0.92211196, -0.05703769,  0.07792116],
        [ 0.89480284, -0.38531223,  0.22552685,  0.53003210],
        [ 0.0,         0.0,         0.0,         1.0       ],
    ], dtype=np.float32)

    # world_view_transform expected by DGR is W2C.T (so .reshape(-1) bytes match
    # column-major W2C — this is the same convention used by getWorld2View2().T).
    world_view_transform = torch.from_numpy(W2C.T.copy()).to(device)

    # Viewproj from the test (already column-major; convert to numpy row-major
    # = byte-identical to col-major full_proj DGR expects).
    viewproj_col_major = np.array([
        -0.45099131,  0.06917855,  1.90754958,  0.20452127,
        -0.56403945, -1.35906174, -0.08406543,  0.11484469,
         0.89489233, -0.38535077,  0.2255494,   0.52008411,
         0.89480284, -0.38531223,  0.22552685,  0.53003210,
    ], dtype=np.float32).reshape(4, 4)
    # The CPP file calls these "col0..col3" and stores them sequentially in
    # cam.viewproj_matrix[16] => bytes are col-major. To get the same bytes
    # in numpy row-major, we just use the same flat array.
    full_proj_transform = torch.from_numpy(viewproj_col_major.copy()).to(device)

    return {
        "W": W, "H": H,
        "fx": fx, "fy": fy,
        "fovx": fovx, "fovy": fovy,
        "tan_fovx": tan_fovx, "tan_fovy": tan_fovy,
        "cam_pos": cam_pos,
        "world_view_transform": world_view_transform,
        "full_proj_transform": full_proj_transform,
        "campos": torch.from_numpy(cam_pos).to(device),
    }


# ----------------------------------------------------------------------------
# Render via low-level GaussianRasterizer (autograd-enabled)
# ----------------------------------------------------------------------------
def render(cam, raw_xyz, features_dc, features_rest, raw_opacity,
           raw_scaling, raw_rotation, sh_degree_active, splat_args,
           bg_color, max_sh_degree=3):
    """Forward pass with autograd. Mirrors gaussian_renderer.render() but uses
    raw params (we apply activations here so we get gradients on the raw
    pre-activation tensors, matching VK's bookkeeping).
    """
    # Activations (must match GaussianModel):
    #   xyz: identity
    #   scaling: exp
    #   rotation: normalize
    #   opacity: sigmoid
    #   features: identity (SH coeffs are stored unactivated)
    means3D = raw_xyz                                                       # (N,3)
    scales = torch.exp(raw_scaling)                                         # (N,3)
    rotations = raw_rotation / torch.norm(raw_rotation, dim=-1, keepdim=True)  # (N,4)
    opacity = torch.sigmoid(raw_opacity)                                    # (N,1)
    # features_dc: (N,1,3); features_rest: (N,K-1,3); concat to (N,K,3)
    shs = torch.cat([features_dc, features_rest], dim=1)                    # (N,K,3)

    # Screenspace dummy for radii (autograd hook, not used here for grad)
    screenspace = torch.zeros_like(means3D, requires_grad=True)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(cam["H"]),
        image_width=int(cam["W"]),
        tanfovx=cam["tan_fovx"],
        tanfovy=cam["tan_fovy"],
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=cam["world_view_transform"],
        projmatrix=cam["full_proj_transform"],
        inv_viewprojmatrix=cam["full_proj_transform"].inverse(),
        sh_degree=int(sh_degree_active),
        campos=cam["campos"],
        prefiltered=False,
        settings=splat_args,
        render_depth=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered, radii = rasterizer(
        means3D=means3D,
        means2D=screenspace,
        opacities=opacity,
        filter3D=None,
        shs=shs,
        colors_precomp=None,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    return rendered, radii


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default="/home/robota/Downloads/basketball/sparse/0/points3D.ply")
    ap.add_argument("--gt_image",
                    default="/home/robota/Downloads/basketball/images/78899858295079.jpg")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--lr_pos",     type=float, default=1.6e-4)
    ap.add_argument("--lr_sh_dc",   type=float, default=2.5e-3)
    ap.add_argument("--lr_sh_rest", type=float, default=1.25e-4)
    ap.add_argument("--lr_op",      type=float, default=0.05)
    ap.add_argument("--lr_sca",     type=float, default=0.005)
    ap.add_argument("--lr_rot",     type=float, default=0.001)
    ap.add_argument("--out_dir",
                    default=os.path.join(REPO_ROOT, "tests", "golden",
                                         "basketball_cuda_ref_10step"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[init] out_dir = {args.out_dir}")

    # ------------------------------------------------------------------
    # 1. Load PLY (xyz + rgb only; matches test_e2e_basketball.cpp)
    # ------------------------------------------------------------------
    print(f"[1/4] Loading PLY: {args.ply}")
    pts_xyz, pts_rgb = load_colmap_ply(args.ply)
    N = pts_xyz.shape[0]
    print(f"       N = {N}")

    # ------------------------------------------------------------------
    # 2. Initialise raw params per VK convention
    # ------------------------------------------------------------------
    sh_deg = args.sh_degree
    K = (sh_deg + 1) ** 2  # 16 for sh_degree=3
    C0 = 0.28209479177387814

    init_raw_scale = math.log(0.03)        # = -3.5066
    init_raw_opacity = math.log(0.1 / 0.9)  # = logit(0.1) = -2.1972

    raw_xyz_np = pts_xyz.copy()
    raw_scaling_np = np.full((N, 3), init_raw_scale, dtype=np.float32)
    raw_opacity_np = np.full((N, 1), init_raw_opacity, dtype=np.float32)
    raw_rotation_np = np.zeros((N, 4), dtype=np.float32)
    raw_rotation_np[:, 0] = 1.0  # identity quat (w, x, y, z)

    # SH: features_dc[N,1,3]; features_rest[N,K-1,3]; SH layout in
    # GaussianModel is [N, channel, coeff] BUT the rasterizer wants
    # `shs` in [N, coeff, channel] order — see GaussianModel.get_features:
    #   features = torch.cat((self._features_dc, self._features_rest), dim=1)
    # where self._features_dc has shape [N, 1, 3]  (coeff_idx=0, channels=3)
    # and self._features_rest has shape [N, K-1, 3] (coeffs=1..K-1, channels=3).
    # So effectively features_dc[i,0,:] is the DC for Gaussian i (3 channels).
    sh_dc_np = np.zeros((N, 1, 3), dtype=np.float32)
    sh_rest_np = np.zeros((N, K - 1, 3), dtype=np.float32)
    for ch in range(3):
        sh_dc_np[:, 0, ch] = (pts_rgb[:, ch].astype(np.float32) / 255.0 - 0.5) / C0

    device = "cuda"
    raw_xyz = torch.tensor(raw_xyz_np, device=device, requires_grad=True)
    raw_scaling = torch.tensor(raw_scaling_np, device=device, requires_grad=True)
    raw_rotation = torch.tensor(raw_rotation_np, device=device, requires_grad=True)
    raw_opacity = torch.tensor(raw_opacity_np, device=device, requires_grad=True)
    features_dc = torch.tensor(sh_dc_np, device=device, requires_grad=True)
    features_rest = torch.tensor(sh_rest_np, device=device, requires_grad=True)

    # ------------------------------------------------------------------
    # 3. Camera + GT image
    # ------------------------------------------------------------------
    W, H = 720, 960
    cam = build_camera(W, H, device=device)
    print(f"[2/4] Loading GT image: {args.gt_image}")
    img = Image.open(args.gt_image).convert("RGB")
    if img.size != (W, H):
        img = img.resize((W, H), Image.BILINEAR)
    gt_chw = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)  # (3,H,W)
    gt = torch.from_numpy(gt_chw.copy()).to(device)

    bg = torch.zeros(3, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # 4. Optimizer (two-group SH split, matches VulkanTrainer + GaussianModel)
    # ------------------------------------------------------------------
    splat_args = ExtendedSettings()
    splat_args.proper_ewa_scaling = False
    splat_args.eval_3D = False     # match VK training path (backward unsupported in eval_3D)
    splat_args.load_balancing = False
    # Defaults: sort_settings/culling_settings (sort_mode 0 = GLOBAL).

    optimizer = torch.optim.Adam(
        [
            {"params": [raw_xyz],        "lr": args.lr_pos,     "name": "xyz"},
            {"params": [features_dc],    "lr": args.lr_sh_dc,   "name": "f_dc"},
            {"params": [features_rest],  "lr": args.lr_sh_rest, "name": "f_rest"},
            {"params": [raw_opacity],    "lr": args.lr_op,      "name": "opacity"},
            {"params": [raw_scaling],    "lr": args.lr_sca,     "name": "scaling"},
            {"params": [raw_rotation],   "lr": args.lr_rot,     "name": "rotation"},
        ],
        lr=0.0, eps=1e-15, betas=(0.9, 0.999),
    )

    # ------------------------------------------------------------------
    # 5. Save meta.json
    # ------------------------------------------------------------------
    meta = {
        "marker": "TEMP DIAGNOSTIC — basketball VK-vs-CUDA 10-step parity",
        "ply": os.path.abspath(args.ply),
        "gt_image": os.path.abspath(args.gt_image),
        "N": int(N),
        "W": W, "H": H,
        "sh_degree": sh_deg, "max_coeffs": K,
        "init_raw_scale": init_raw_scale,
        "init_raw_opacity": init_raw_opacity,
        "C0": C0,
        "rotation_init": "identity quaternion (w,x,y,z) = (1,0,0,0)",
        "lr": {
            "pos": args.lr_pos, "sh_dc": args.lr_sh_dc, "sh_rest": args.lr_sh_rest,
            "op": args.lr_op,   "sca":   args.lr_sca,   "rot":     args.lr_rot,
        },
        "betas": [0.9, 0.999], "eps": 1e-15,
        "proper_ewa_scaling": False,
        "eval_3D": False,
        "lambda_dssim": 0.0,
        "background": [0.0, 0.0, 0.0],
        "loss": "L1(rendered, gt).mean() over CHW pixels",
        "sh_degree_active_per_step":
            "all steps = sh_degree when sh_degree_warmup=0",
        "view_matrix_source": "test_e2e_basketball.cpp:290-305 (column-major W2C)",
        "viewproj_source":    "test_e2e_basketball.cpp:314-329 (column-major full_proj)",
    }

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    print(f"[3/4] Running {args.steps} steps...")
    losses = []
    t_start = time.time()
    for step in range(1, args.steps + 1):
        sh_degree_active = sh_deg
        t0 = time.time()
        rendered, radii = render(
            cam, raw_xyz, features_dc, features_rest,
            raw_opacity, raw_scaling, raw_rotation,
            sh_degree_active=sh_degree_active,
            splat_args=splat_args,
            bg_color=bg, max_sh_degree=sh_deg,
        )
        loss = torch.abs(rendered - gt).mean()
        loss_val = float(loss.item())

        optimizer.zero_grad(set_to_none=False)
        loss.backward()

        # Capture gradients BEFORE optimizer.step() (which updates params).
        grads = {
            "pos":      raw_xyz.grad.detach().cpu().contiguous().numpy().astype(np.float32),
            "sh_dc":    features_dc.grad.detach().cpu().contiguous().numpy().astype(np.float32),
            "sh_rest":  features_rest.grad.detach().cpu().contiguous().numpy().astype(np.float32),
            "op":       raw_opacity.grad.detach().cpu().contiguous().numpy().astype(np.float32),
            "sca":      raw_scaling.grad.detach().cpu().contiguous().numpy().astype(np.float32),
            "rot":      raw_rotation.grad.detach().cpu().contiguous().numpy().astype(np.float32),
        }

        optimizer.step()

        # Capture POST-Adam params and moment state.
        with torch.no_grad():
            params = {
                "pos":  raw_xyz.detach().cpu().contiguous().numpy().astype(np.float32),
                # features_dc + features_rest -> concat to (N,K,3) like GaussianModel.get_features
                "sh":   torch.cat([features_dc, features_rest], dim=1).detach().cpu()
                              .contiguous().numpy().astype(np.float32),
                "op":   raw_opacity.detach().cpu().contiguous().numpy().astype(np.float32),
                "sca":  raw_scaling.detach().cpu().contiguous().numpy().astype(np.float32),
                "rot":  raw_rotation.detach().cpu().contiguous().numpy().astype(np.float32),
            }
            adam_m = {
                "pos":     optimizer.state[raw_xyz]["exp_avg"].detach().cpu().contiguous().numpy().astype(np.float32),
                "sh_dc":   optimizer.state[features_dc]["exp_avg"].detach().cpu().contiguous().numpy().astype(np.float32),
                "sh_rest": optimizer.state[features_rest]["exp_avg"].detach().cpu().contiguous().numpy().astype(np.float32),
                "op":      optimizer.state[raw_opacity]["exp_avg"].detach().cpu().contiguous().numpy().astype(np.float32),
                "sca":     optimizer.state[raw_scaling]["exp_avg"].detach().cpu().contiguous().numpy().astype(np.float32),
                "rot":     optimizer.state[raw_rotation]["exp_avg"].detach().cpu().contiguous().numpy().astype(np.float32),
            }
            adam_v = {
                "pos":     optimizer.state[raw_xyz]["exp_avg_sq"].detach().cpu().contiguous().numpy().astype(np.float32),
                "sh_dc":   optimizer.state[features_dc]["exp_avg_sq"].detach().cpu().contiguous().numpy().astype(np.float32),
                "sh_rest": optimizer.state[features_rest]["exp_avg_sq"].detach().cpu().contiguous().numpy().astype(np.float32),
                "op":      optimizer.state[raw_opacity]["exp_avg_sq"].detach().cpu().contiguous().numpy().astype(np.float32),
                "sca":     optimizer.state[raw_scaling]["exp_avg_sq"].detach().cpu().contiguous().numpy().astype(np.float32),
                "rot":     optimizer.state[raw_rotation]["exp_avg_sq"].detach().cpu().contiguous().numpy().astype(np.float32),
            }

        # Save
        step_dir = os.path.join(args.out_dir, f"step_{step:04d}")
        os.makedirs(step_dir, exist_ok=True)
        np.save(os.path.join(step_dir, "loss.npy"),
                np.array([loss_val], dtype=np.float32))
        for k, v in grads.items():
            np.save(os.path.join(step_dir, f"grad_{k}.npy"), v)
        for k, v in params.items():
            np.save(os.path.join(step_dir, f"param_{k}.npy"), v)
        for k, v in adam_m.items():
            np.save(os.path.join(step_dir, f"adam_m_{k}.npy"), v)
        for k, v in adam_v.items():
            np.save(os.path.join(step_dir, f"adam_v_{k}.npy"), v)

        losses.append(loss_val)
        dt = time.time() - t0
        print(f"  step {step:2d}: loss={loss_val:.6f}  ({dt*1000:.1f} ms)")

    total = time.time() - t_start
    print(f"[4/4] Done. {args.steps} steps in {total:.2f}s ({total/args.steps*1000:.1f} ms/step avg)")

    meta["losses"] = losses
    meta["total_seconds"] = total
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
