#!/usr/bin/env python3
# TEMP DIAGNOSTIC — basketball ply downsampled parity
"""Dump Python autograd reference on the basketball point cloud at downsampled resolution.

Mirrors dump_tiny_reference.py but loads the basketball COLMAP PLY (N=2892,
sh_degree=3, max_coeffs=16) and the first image (78899858295079.jpg)
downsampled to a low resolution so per-pixel autograd terminates in finite
time. Used to verify that the recent fixes (SH Adam-group layout,
proper_ewa default flip, det floor + frustum 1.3x clamp alignment) hold
at scale (N=2892) — 145x the tiny fixture's N=20.

SH degree convention:
    forward_render_pytorch in train_pytorch_reference.py only uses
    raw_sh[:, :3] (DC) and ignores higher-order coefficients regardless of
    the sh_degree argument. To keep parity tractable on a per-pixel triple
    loop and avoid a per-Gaussian SH evaluator in autograd, we keep
    SH evaluation locked at degree 0 (DC only) on BOTH sides:
      - Python: sh_degree=0 in forward_render_pytorch
      - VK:     sh_degree_max=0, sh_degree_warmup=very large, render with
                cfg.sh_degree=0
    REST coefficients (k=1..15) are zero-initialized and receive zero gradient
    in both pipelines. The post-Adam SH array still has shape [N, 16, 3] and
    the Adam-group split is exercised end-to-end (DC group lr=2.5e-3, REST
    group lr=1.25e-4 with zero updates), which is exactly the path the
    layout-fix at scale needs to validate.

Adam group structure (matches VK two-group SH):
    group 0: positions [N,3]      lr = pos_lr_at_step(step) ~ 1.6e-4
    group 1: SH DC     [N,1,3]    lr = 2.5e-3       (DC of all N Gaussians)
    group 2: SH REST   [N,K-1,3]  lr = 2.5e-3 / 20  = 1.25e-4
    group 3: opacities [N,1]      lr = 0.05
    group 4: scales    [N,3]      lr = 0.005
    group 5: rotations [N,4]      lr = 0.001
The DC and REST tensors are SEPARATE leaf params and get re-stitched into
the full [N, max_coeffs*3] interleaved layout when saved.

Usage:
    python harmonyos_3dgs/tools/dump_basketball_ds_reference.py \
        --resolution 64x48 --steps 10 \
        --ply /home/robota/Downloads/basketball/sparse/0/points3D.ply \
        --image /home/robota/Downloads/basketball/images/78899858295079.jpg \
        --output-dir harmonyos_3dgs/tests/golden/basketball_ds
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

# Import the differentiable renderer (do NOT copy its code).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pytorch_reference import forward_render_pytorch


# ---------------------------------------------------------------------------
# Hyperparameters — match dump_tiny_reference.py (which matches VulkanTrainer)
# ---------------------------------------------------------------------------
ADAM_EPS = 1e-15
POS_LR_INIT = 1.6e-4
POS_LR_FINAL = 1.6e-6
MAX_STEPS_SCHEDULE = 30000
SH_DC_LR = 2.5e-3
SH_REST_LR = SH_DC_LR / 20.0   # 1.25e-4 (matches VK two-group split)
OPACITY_LR = 0.05
SCALE_LR = 0.005
ROTATION_LR = 0.001
ADAM_BETAS = (0.9, 0.999)


def pos_lr_at_step(step: int) -> float:
    t = min(1.0, max(0.0, step / float(MAX_STEPS_SCHEDULE)))
    return float(np.exp(np.log(POS_LR_INIT) * (1.0 - t) + np.log(POS_LR_FINAL) * t))


# ---------------------------------------------------------------------------
# PLY loader — mirrors test_e2e_basketball.cpp (binary_little_endian, 27B/vertex)
# ---------------------------------------------------------------------------
def load_basketball_ply(path: str):
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            b = f.read(1)
            if not b:
                raise RuntimeError(f"Unexpected EOF in PLY header: {path}")
            hdr += b
        N = None
        for line in hdr.decode().splitlines():
            if line.startswith("element vertex "):
                N = int(line.split()[2])
                break
        if N is None:
            raise RuntimeError(f"Could not find vertex count in PLY header: {path}")
        raw = f.read(N * 27)
        if len(raw) != N * 27:
            raise RuntimeError(f"PLY data truncated: got {len(raw)} bytes, expected {N*27}")
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(N, 27)
    pos = arr[:, 0:12].copy().view(np.float32).reshape(N, 3).astype(np.float32)
    rgb = arr[:, 24:27].astype(np.float32)  # uchar -> float
    return pos, rgb


# ---------------------------------------------------------------------------
# Initial Gaussians — matches test_e2e_basketball.cpp lines 188-247
# ---------------------------------------------------------------------------
SH_C0 = 0.28209479177387814

def make_initial_gaussians(pos_np: np.ndarray, rgb_np: np.ndarray, max_coeffs: int):
    N = pos_np.shape[0]
    init_raw_scale = float(np.log(0.03))
    init_raw_opacity = float(np.log(0.1 / 0.9))   # ~ -2.197
    raw_pos = pos_np.copy().astype(np.float32)                       # [N, 3]
    raw_sc  = np.full((N, 3), init_raw_scale, dtype=np.float32)      # [N, 3]
    raw_rot = np.zeros((N, 4), dtype=np.float32)                     # [N, 4]
    raw_rot[:, 0] = 1.0                                               # identity quat
    raw_op  = np.full((N, 1), init_raw_opacity, dtype=np.float32)    # [N, 1]
    raw_sh  = np.zeros((N, max_coeffs, 3), dtype=np.float32)         # [N, K, 3]
    # DC slot: index k=0 per Gaussian
    raw_sh[:, 0, 0] = (rgb_np[:, 0] / 255.0 - 0.5) / SH_C0
    raw_sh[:, 0, 1] = (rgb_np[:, 1] / 255.0 - 0.5) / SH_C0
    raw_sh[:, 0, 2] = (rgb_np[:, 2] / 255.0 - 0.5) / SH_C0
    return raw_pos, raw_sc, raw_rot, raw_op, raw_sh


# ---------------------------------------------------------------------------
# Camera (matches test_e2e_basketball.cpp lines 274-329 — pre-computed for
# COLMAP image 78899858295079.jpg). FOV is invariant of image resolution; the
# proj matrix stores 1/tan_fovx, 1/tan_fovy, so vpm is reusable as-is at any
# downsampled resolution. forward_render_pytorch derives focal_x_val from
# vpm[0,0]*W*0.5, which scales correctly with W.
# ---------------------------------------------------------------------------
def basketball_cam():
    vm = np.zeros(16, dtype=np.float32)
    vm[ 0] = -0.22993832; vm[ 1] = -0.38269602; vm[ 2] =  0.89480284; vm[ 3] = 0.0
    vm[ 4] =  0.03527074; vm[ 5] = -0.92211196; vm[ 6] = -0.38531223; vm[ 7] = 0.0
    vm[ 8] =  0.97256586; vm[ 9] = -0.05703769; vm[10] =  0.22552685; vm[11] = 0.0
    vm[12] =  0.10427535; vm[13] =  0.07792116; vm[14] =  0.53003210; vm[15] = 1.0
    vpm = np.zeros(16, dtype=np.float32)
    vpm[ 0] = -0.45099131; vpm[ 1] =  0.06917855; vpm[ 2] =  1.90754958; vpm[ 3] = 0.20452127
    vpm[ 4] = -0.56403945; vpm[ 5] = -1.35906174; vpm[ 6] = -0.08406543; vpm[ 7] = 0.11484469
    vpm[ 8] =  0.89489233; vpm[ 9] = -0.38535077; vpm[10] =  0.2255494;  vpm[11] = 0.52008411
    vpm[12] =  0.89480284; vpm[13] = -0.38531223; vpm[14] =  0.22552685; vpm[15] = 0.53003210
    cam_pos = np.array([-0.42047721, 0.27240201, -0.21650667], dtype=np.float32)
    return vm, vpm, cam_pos


# ---------------------------------------------------------------------------
# Run dump
# ---------------------------------------------------------------------------
def run_dump(args):
    os.makedirs(args.output_dir, exist_ok=True)
    W, H = args.W, args.H

    # ----- Load PLY + build initial Gaussians -----
    pos_np, rgb_np = load_basketball_ply(args.ply)
    N = pos_np.shape[0]
    sh_degree = 3
    max_coeffs = 16  # (sh_degree+1)^2

    raw_pos_np, raw_sc_np, raw_rot_np, raw_op_np, raw_sh_np = make_initial_gaussians(
        pos_np, rgb_np, max_coeffs)
    print(f"[init] N={N} sh_degree={sh_degree} max_coeffs={max_coeffs}", flush=True)
    print(f"[init] Resolution: {W}x{H}", flush=True)

    # ----- Load + downsample target image (BILINEAR) -----
    pil = Image.open(args.image).convert("RGB")
    print(f"[init] Source image {pil.size} -> downsample to ({W}, {H})", flush=True)
    pil_ds = pil.resize((W, H), Image.BILINEAR)
    gt_np = np.asarray(pil_ds, dtype=np.float32) / 255.0   # [H, W, 3]
    # Save BOTH HWC (Python convention) and CHW (VK convention) so the C++
    # parity test can load a byte-identical target without re-doing the
    # downsample (avoids PIL/stb resize behaviour drift).
    np.save(os.path.join(args.output_dir, "target_image_hwc.npy"),
            gt_np.astype(np.float32))
    np.save(os.path.join(args.output_dir, "target_image_chw.npy"),
            np.transpose(gt_np, (2, 0, 1)).astype(np.float32))

    # ----- Camera -----
    vm_np, vpm_np, cam_pos_np = basketball_cam()
    # Save camera + meta so the VK side reads the SAME matrices.
    # tan_fovx/tan_fovy are FOV-only, derived from native intrinsics
    # (fx=706.089, fy=707.452 at native 720x960). Resolution-invariant.
    tan_fovx = 720.0 / (2.0 * 706.08879017028869)   # 0.509851
    tan_fovy = 960.0 / (2.0 * 707.4516640627736)    # 0.678492
    np.save(os.path.join(args.output_dir, "input_viewmatrix.npy"),
            vm_np.astype(np.float32))
    np.save(os.path.join(args.output_dir, "input_projmatrix.npy"),
            vpm_np.astype(np.float32))
    np.save(os.path.join(args.output_dir, "input_campos.npy"),
            cam_pos_np.astype(np.float32))
    np.save(os.path.join(args.output_dir, "input_fov_size.npy"),
            np.array([tan_fovx, tan_fovy, float(W), float(H)], dtype=np.float32))
    np.save(os.path.join(args.output_dir, "input_meta.npy"),
            np.array([float(sh_degree), float(max_coeffs), float(H), float(W)],
                     dtype=np.float32))
    # Save initial raw params so the VK side ingests the SAME initialization
    # without re-deriving from PLY (decouples C++ test from PLY parsing).
    np.save(os.path.join(args.output_dir, "init_raw_pos.npy"),
            raw_pos_np.astype(np.float32))
    np.save(os.path.join(args.output_dir, "init_raw_sc.npy"),
            raw_sc_np.astype(np.float32))
    np.save(os.path.join(args.output_dir, "init_raw_rot.npy"),
            raw_rot_np.astype(np.float32))
    np.save(os.path.join(args.output_dir, "init_raw_op.npy"),
            raw_op_np.astype(np.float32).reshape(-1))   # [N]
    np.save(os.path.join(args.output_dir, "init_raw_sh.npy"),
            raw_sh_np.reshape(N, max_coeffs * 3).astype(np.float32))

    # ----- Build PyTorch leaf parameters -----
    device = "cpu"
    raw_pos = torch.tensor(raw_pos_np, dtype=torch.float32, device=device, requires_grad=True)
    raw_sc  = torch.tensor(raw_sc_np,  dtype=torch.float32, device=device, requires_grad=True)
    raw_rot = torch.tensor(raw_rot_np, dtype=torch.float32, device=device, requires_grad=True)
    raw_op  = torch.tensor(raw_op_np,  dtype=torch.float32, device=device, requires_grad=True)
    # SH: split into DC and REST as SEPARATE leaves so two Adam groups apply
    # different LRs (matches VK two-group split). At save time we re-stitch
    # them into the [N, K, 3] interleaved layout that VulkanTrainer expects.
    raw_sh_dc   = torch.tensor(raw_sh_np[:, 0:1, :].copy(), dtype=torch.float32,
                               device=device, requires_grad=True)   # [N, 1, 3]
    raw_sh_rest = torch.tensor(raw_sh_np[:, 1:, :].copy(),  dtype=torch.float32,
                               device=device, requires_grad=True)   # [N, K-1, 3]

    vm  = torch.tensor(vm_np,  dtype=torch.float32, device=device)
    vpm = torch.tensor(vpm_np, dtype=torch.float32, device=device)
    cam_pos = torch.tensor(cam_pos_np, dtype=torch.float32, device=device)
    bg = torch.zeros(3, device=device)
    gt = torch.tensor(gt_np, dtype=torch.float32, device=device)   # [H, W, 3]

    # ----- Adam optimizer (6 groups: pos, sh_dc, sh_rest, op, sca, rot) -----
    optimizer = torch.optim.Adam([
        {"params": [raw_pos],    "lr": pos_lr_at_step(1)},
        {"params": [raw_sh_dc],  "lr": SH_DC_LR},
        {"params": [raw_sh_rest],"lr": SH_REST_LR},
        {"params": [raw_op],     "lr": OPACITY_LR},
        {"params": [raw_sc],     "lr": SCALE_LR},
        {"params": [raw_rot],    "lr": ROTATION_LR},
    ], betas=ADAM_BETAS, eps=ADAM_EPS)

    print(f"[run] Starting {args.steps} steps. Output: {args.output_dir}", flush=True)
    t_start = time.time()

    for step in range(1, args.steps + 1):
        t_step0 = time.time()
        # 1-indexed pos lr
        optimizer.param_groups[0]["lr"] = pos_lr_at_step(step)
        optimizer.zero_grad()

        # Stitch DC + REST into the flat raw_sh tensor that
        # forward_render_pytorch expects: shape [N, max_coeffs*3].
        # forward_render_pytorch only reads raw_sh[:, :3] (DC), so REST is
        # passed but unused — its grad will be zero, which is the correct
        # behavior at sh_degree=0.
        # NOTE: torch.cat preserves grad through both leaves, so backward
        # accumulates grad onto raw_sh_dc; raw_sh_rest still gets zero grad.
        # We use cat so the Python forward sees the full [N, K, 3] layout
        # the VK side ingests.
        raw_sh_full = torch.cat([raw_sh_dc, raw_sh_rest], dim=1).reshape(N, max_coeffs * 3)

        rendered = forward_render_pytorch(
            raw_pos, raw_sc, raw_rot, raw_sh_full, raw_op,
            vm, vpm, cam_pos, W, H, bg, sh_degree=0
        )
        loss = torch.mean(torch.abs(rendered - gt))
        loss.backward()

        # Save grads BEFORE optimizer.step()
        prefix = os.path.join(args.output_dir, f"step_{step:04d}")
        np.save(f"{prefix}_loss.npy",
                np.array([loss.item()], dtype=np.float32))
        np.save(f"{prefix}_grad_pos.npy",
                raw_pos.grad.detach().numpy().reshape(N, 3).astype(np.float32))
        np.save(f"{prefix}_grad_sc.npy",
                raw_sc.grad.detach().numpy().reshape(N, 3).astype(np.float32))
        np.save(f"{prefix}_grad_rot.npy",
                raw_rot.grad.detach().numpy().reshape(N, 4).astype(np.float32))
        # SH grad: stitch DC + REST grads into [N, max_coeffs*3] (same layout
        # the VK side captures via captured_grad_sh()). raw_sh_rest grad is
        # zero in the sh_degree=0 forward path; we still write the full array
        # so VK and Python compare element-wise across all 16*3 slots.
        if raw_sh_dc.grad is None:
            grad_sh_dc = torch.zeros_like(raw_sh_dc)
        else:
            grad_sh_dc = raw_sh_dc.grad.detach()
        if raw_sh_rest.grad is None:
            grad_sh_rest = torch.zeros_like(raw_sh_rest)
        else:
            grad_sh_rest = raw_sh_rest.grad.detach()
        grad_sh_full = torch.cat([grad_sh_dc, grad_sh_rest], dim=1).reshape(N, max_coeffs * 3)
        np.save(f"{prefix}_grad_sh.npy",
                grad_sh_full.numpy().astype(np.float32))
        np.save(f"{prefix}_grad_op.npy",
                raw_op.grad.detach().numpy().reshape(N).astype(np.float32))

        optimizer.step()

        # Save params AFTER optimizer.step()
        np.save(f"{prefix}_raw_pos.npy",
                raw_pos.detach().numpy().reshape(N, 3).astype(np.float32))
        np.save(f"{prefix}_raw_sc.npy",
                raw_sc.detach().numpy().reshape(N, 3).astype(np.float32))
        np.save(f"{prefix}_raw_rot.npy",
                raw_rot.detach().numpy().reshape(N, 4).astype(np.float32))
        np.save(f"{prefix}_raw_op.npy",
                raw_op.detach().numpy().reshape(N).astype(np.float32))
        # Re-stitch post-Adam SH into [N, max_coeffs, 3] and save flat.
        raw_sh_full_post = torch.cat([raw_sh_dc.detach(), raw_sh_rest.detach()],
                                     dim=1).reshape(N, max_coeffs * 3)
        np.save(f"{prefix}_raw_sh.npy",
                raw_sh_full_post.numpy().astype(np.float32))

        dt = time.time() - t_step0
        print(f"  step {step:3d}/{args.steps}: loss={loss.item():.8f}  "
              f"pos_lr={pos_lr_at_step(step):.2e}  step_time={dt:.1f}s  "
              f"elapsed={(time.time()-t_start)/60.0:.1f}min", flush=True)

    print(f"\n[done] Saved {args.steps * 11} files to {args.output_dir}  "
          f"total={(time.time()-t_start)/60.0:.1f} min", flush=True)


def parse_resolution(s: str):
    s = s.lower().replace(" ", "")
    if "x" in s:
        a, b = s.split("x", 1)
        return int(a), int(b)
    raise ValueError(f"Bad --resolution: {s!r}; expected WxH like 64x48")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ply", required=True)
    p.add_argument("--image", required=True,
                   help="Path to first basketball image (78899858295079.jpg).")
    p.add_argument("--resolution", default="64x48",
                   help="Target resolution WxH (default 64x48).")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    W, H = parse_resolution(args.resolution)
    args.W, args.H = W, H
    run_dump(args)


if __name__ == "__main__":
    main()
