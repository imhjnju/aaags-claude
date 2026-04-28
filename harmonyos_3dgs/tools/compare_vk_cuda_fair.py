#!/usr/bin/env python3
"""Fair VK vs CUDA 2000-step comparison — unified conditions.

BOTH sides use:
  Init PLY   : /tmp/basketball_init_3dgs.ply  (same COLMAP-derived 3DGS init)
  Camera     : cam0 — 78899858295079.jpg  (single camera, same view matrix)
  Resolution : 720 × 960  (full, no downscaling)
  Steps      : 2000, no densification
  Loss       : L1 only  (lambda_dssim=0, same for both)
  Adam       : lr_pos=1.6e-4, lr_sh_dc=2.5e-3, lr_sh_rest=1.25e-4,
               lr_op=0.05, lr_sca=0.005, lr_rot=0.001
               betas=(0.9,0.999), eps=1e-15
  proper_ewa : False  (match parity-harness default)

Render cam: cam0 (same camera used for training)
Output in <outdir>/: cuda_render.png, vk_render.png, gt_cam0.png,
                     comparison.png (GT|CUDA|VK), report.txt

Usage:
  # first time (run both):
  conda activate aaa-gs && python tools/compare_vk_cuda_fair.py

  # skip one side:
  python tools/compare_vk_cuda_fair.py --skip_cuda
  python tools/compare_vk_cuda_fair.py --skip_vk
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)
BUILD_DIR  = os.path.join(REPO_DIR, "build")

INIT_PLY   = "/tmp/basketball_init_3dgs.ply"
CAM_JSON   = "/home/robota/Downloads/basketball/_sp0_dump_output/cameras.json"
GT_JPG     = "/home/robota/Downloads/basketball/images/78899858295079.jpg"
COLMAP_PLY = "/home/robota/Downloads/basketball/sparse/0/points3D.ply"

CAM0_IDX   = 0     # cameras.json index; img_name = "78899858295079"
steps      = 2000

# Adam hyperparameters — match VK train defaults from train_types.h
LR_POS     = 1.6e-4
LR_SH_DC   = 2.5e-3
LR_SH_REST = 1.25e-4
LR_OP      = 0.05
LR_SCA     = 0.005
LR_ROT     = 0.001
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS   = 1e-15

SH_DEGREE  = 3
K          = (SH_DEGREE + 1) ** 2   # 16 SH coefficients


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def save_png(arr: np.ndarray, path: str):
    u8 = (arr.clip(0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(u8).save(path)


def write_ppm_u8(path: str, hwc: np.ndarray):
    H, W = hwc.shape[:2]
    u8 = (hwc.clip(0, 1) * 255 + 0.5).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(f"P6\n{W} {H}\n255\n".encode())
        f.write(u8.tobytes())


def _aaa_root():
    for c in [
        os.path.join(REPO_DIR, "..", "..", "AAA-Gaussians"),
        "/home/robota/h00813233/Graph/aaags-claude/AAA-Gaussians",
    ]:
        c = os.path.abspath(c)
        if os.path.isdir(c):
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    raise RuntimeError("AAA-Gaussians not found")


# ---------------------------------------------------------------------------
# Camera parameters from cameras.json cam0
# ---------------------------------------------------------------------------
def load_cam0_params():
    """Return (R_c2w, T_w2c, tan_fovx, tan_fovy, W, H, gt_np)."""
    cam_data = json.load(open(CAM_JSON))
    c = cam_data[CAM0_IDX]
    assert c["img_name"] == "78899858295079", f"cam0 mismatch: {c['img_name']}"

    R_c2w   = np.array(c["rotation"], dtype=np.float64)   # 3×3 C2W
    cam_pos = np.array(c["position"],  dtype=np.float64)   # world pos
    T_w2c   = -(R_c2w.T @ cam_pos)                         # W2C translation

    W, H    = c["width"], c["height"]
    tan_fovx = W / (2.0 * c["fx"])
    tan_fovy = H / (2.0 * c["fy"])

    gt_np = np.array(Image.open(GT_JPG).convert("RGB"), dtype=np.float32) / 255.0
    assert gt_np.shape == (H, W, 3), f"GT shape {gt_np.shape} vs expected {(H,W,3)}"

    return R_c2w, T_w2c, tan_fovx, tan_fovy, W, H, gt_np


# ---------------------------------------------------------------------------
# CUDA training — inline PyTorch, same init PLY, same cam0
# ---------------------------------------------------------------------------
def run_cuda_training(outdir: str, steps: int) -> np.ndarray:
    import torch
    from torch import nn
    _aaa_root()

    from diff_gaussian_rasterization import ExtendedSettings
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene import GaussianModel
    from utils.loss_utils import l1_loss
    from arguments import OptimizationParams
    import argparse as ap

    print(f"\n=== CUDA Training ({steps} steps, cam0, 720×960) ===")
    t0 = time.time()

    R_c2w, T_w2c, tan_fovx, tan_fovy, W, H, gt_np = load_cam0_params()
    FoVx = 2.0 * math.atan(tan_fovx)
    FoVy = 2.0 * math.atan(tan_fovy)

    gt_tensor = torch.from_numpy(gt_np).permute(2, 0, 1).float().cuda()  # CHW

    # Camera object (3DGS convention: R=R_c2w, T=T_w2c)
    cam = Camera(
        colmap_id=0,
        R=R_c2w.astype(np.float32),
        T=T_w2c.astype(np.float32),
        FoVx=FoVx,
        FoVy=FoVy,
        image=gt_tensor,
        gt_alpha_mask=None,
        image_name="cam0",
        uid=0,
    )

    # Load Gaussians from shared init PLY
    gaussians = GaussianModel(sh_degree=SH_DEGREE)
    gaussians.load_ply(INIT_PLY)
    N = gaussians.get_xyz.shape[0]
    print(f"  Loaded {N} Gaussians from {INIT_PLY}")

    # Adam optimizer — match VK hyperparameters exactly
    params = [
        {"params": [gaussians._xyz],           "lr": LR_POS,     "name": "xyz"},
        {"params": [gaussians._features_dc],   "lr": LR_SH_DC,   "name": "f_dc"},
        {"params": [gaussians._features_rest], "lr": LR_SH_REST, "name": "f_rest"},
        {"params": [gaussians._opacity],       "lr": LR_OP,      "name": "opacity"},
        {"params": [gaussians._scaling],       "lr": LR_SCA,     "name": "scaling"},
        {"params": [gaussians._rotation],      "lr": LR_ROT,     "name": "rotation"},
    ]
    optimizer = torch.optim.Adam(params, lr=0,
                                  betas=ADAM_BETAS, eps=ADAM_EPS)

    # Pipeline config
    class PipeCfg:
        debug = False
        convert_SHs_python = False
        compute_cov3D_python = False

    pipe = PipeCfg()
    bg = torch.zeros(3, device="cuda")

    losses = []
    for step in range(1, steps + 1):
        es = ExtendedSettings()
        es.proper_ewa_scaling = False
        es.eval_3D = False

        render_pkg = render(cam, gaussians, pipe, bg, splat_args=es)
        image = render_pkg["render"]

        # L1 loss only (lambda_dssim=0)
        loss = l1_loss(image, gt_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lv = loss.item()
        losses.append(lv)
        if step <= 5 or step % 200 == 0:
            print(f"  [CUDA] step {step:4d}: loss={lv:.6f}  "
                  f"({time.time()-t0:.1f}s)")

    # Save trained PLY
    ply_path = os.path.join(outdir, "cuda_trained.ply")
    gaussians.save_ply(ply_path)
    print(f"  Saved PLY: {ply_path}")

    # Final render from cam0
    with torch.no_grad():
        es = ExtendedSettings()
        es.proper_ewa_scaling = False
        es.eval_3D = False
        final_pkg = render(cam, gaussians, pipe, bg, splat_args=es)
        rendered  = final_pkg["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    p = psnr(rendered, gt_np)
    elapsed = time.time() - t0
    print(f"[CUDA] Done: steps={steps}, final_loss={losses[-1]:.6f}, "
          f"PSNR={p:.2f} dB, time={elapsed:.1f}s")

    np.save(os.path.join(outdir, "cuda_render.npy"), rendered)
    save_png(rendered, os.path.join(outdir, "cuda_render.png"))
    return rendered


# ---------------------------------------------------------------------------
# VK training — subprocess + gs3d_vk_render
# ---------------------------------------------------------------------------
def run_vk_training(outdir: str, steps: int) -> np.ndarray:
    print(f"\n=== VK Training ({steps} steps, cam0, 720×960) ===")
    t0 = time.time()

    R_c2w, T_w2c, tan_fovx, tan_fovy, W, H, gt_np = load_cam0_params()

    # 1. Cam0-only cameras.json
    cam_data  = json.load(open(CAM_JSON))
    cam0_info = cam_data[CAM0_IDX]
    cam0_json_path = os.path.join(outdir, "cam0_only.json")
    with open(cam0_json_path, "w") as f:
        json.dump([cam0_info], f, indent=2)

    # 2. Convert GT to PPM
    ppm_dir  = os.path.join(outdir, "ppm_cam0")
    os.makedirs(ppm_dir, exist_ok=True)
    ppm_path = os.path.join(ppm_dir, cam0_info["img_name"] + ".ppm")
    write_ppm_u8(ppm_path, gt_np)
    print(f"  GT PPM: {ppm_path}  ({W}×{H})")

    # 3. VK training
    vk_train = os.path.join(BUILD_DIR, "gs3d_vk_train")
    vk_ply   = os.path.join(outdir, "vk_trained.ply")
    cmd = [
        vk_train,
        "--ply",        INIT_PLY,
        "--cameras",    cam0_json_path,
        "--gt_dir",     ppm_dir,
        "--output",     vk_ply,
        "--iterations", str(steps),
        "--log_every",  "200",
    ]
    print(f"  Running: {' '.join(cmd)}")
    ret = subprocess.run(cmd, capture_output=False, text=True)
    if ret.returncode != 0:
        raise RuntimeError(f"gs3d_vk_train failed (exit {ret.returncode})")

    elapsed_train = time.time() - t0
    print(f"  VK training done in {elapsed_train:.1f}s")

    # 4. VK render from cam0
    vk_render_bin = os.path.join(BUILD_DIR, "gs3d_vk_render")
    render_wd     = os.path.join(outdir, "vk_render_tmp")
    os.makedirs(render_wd, exist_ok=True)
    cmd2 = [vk_render_bin, vk_ply, cam0_json_path, "0", "--eval_3d", "0"]
    print(f"  Running: {' '.join(cmd2)}")
    ret2 = subprocess.run(cmd2, capture_output=False, text=True, cwd=render_wd)
    if ret2.returncode != 0:
        raise RuntimeError(f"gs3d_vk_render failed (exit {ret2.returncode})")

    # Load raw float CHW output
    raw_path = os.path.join(render_wd, "vk_float.raw")
    if os.path.exists(raw_path):
        data = np.fromfile(raw_path, dtype=np.float32)
        vk_np = data.reshape(3, H, W).transpose(1, 2, 0).clip(0, 1)
    else:
        # Fallback: load uint8 PPM
        from io import BytesIO
        ppm_out = os.path.join(render_wd, "output_vk.ppm")
        with open(ppm_out, "rb") as f:
            f.readline(); dims = f.readline().split(); f.readline()
            W2, H2 = int(dims[0]), int(dims[1])
            vk_np = np.frombuffer(f.read(), dtype=np.uint8).reshape(H2, W2, 3).astype(np.float32) / 255.0

    p = psnr(vk_np, gt_np)
    elapsed = time.time() - t0
    print(f"[VK] Done: PSNR={p:.2f} dB, time={elapsed:.1f}s")

    np.save(os.path.join(outdir, "vk_render.npy"), vk_np)
    save_png(vk_np, os.path.join(outdir, "vk_render.png"))
    return vk_np


# ---------------------------------------------------------------------------
# Compare and report
# ---------------------------------------------------------------------------
def compare_and_report(cuda_np, vk_np, outdir, steps):
    _, _, _, _, W, H, gt_np = load_cam0_params()
    save_png(gt_np, os.path.join(outdir, "gt_cam0.png"))

    p_cuda = psnr(cuda_np, gt_np)
    p_vk   = psnr(vk_np,   gt_np)
    p_diff = psnr(vk_np,   cuda_np)

    report = f"""
====================================================
  VK vs CUDA — FAIR {steps}-step Comparison
====================================================
  Init PLY    : {INIT_PLY}
  Camera      : cam0 — 78899858295079.jpg
  Resolution  : {W}×{H}  (full, same for both)
  Loss        : L1 only  (lambda_dssim=0)
  Adam lr_pos : {LR_POS:.1e}
  Adam lr_sh  : dc={LR_SH_DC:.1e}  rest={LR_SH_REST:.1e}
  Adam lr_op  : {LR_OP:.1e}
  proper_ewa  : False
====================================================
  PSNR (CUDA render vs GT)   = {p_cuda:6.2f} dB
  PSNR (VK   render vs GT)   = {p_vk:6.2f} dB
  PSNR (VK vs CUDA)          = {p_diff:6.2f} dB
  Gap (CUDA − VK)            = {p_cuda - p_vk:6.2f} dB
====================================================
"""
    print(report)
    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write(report)

    # Side-by-side: GT | CUDA | VK
    canvas = np.concatenate([gt_np, cuda_np, vk_np], axis=1)
    save_png(canvas, os.path.join(outdir, "comparison.png"))
    print(f"  Saved: {outdir}/comparison.png  (GT | CUDA | VK)")
    print(f"  Saved: {outdir}/report.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir",    default="/tmp/compare_fair_2000")
    parser.add_argument("--steps",     type=int, default=steps)
    parser.add_argument("--skip_cuda", action="store_true")
    parser.add_argument("--skip_vk",   action="store_true")
    args = parser.parse_args()

    for p, n in [(INIT_PLY, "init PLY"), (GT_JPG, "GT JPG"), (CAM_JSON, "cameras.json")]:
        if not os.path.exists(p):
            # Auto-generate init PLY if missing
            if n == "init PLY":
                print(f"  Generating init PLY from COLMAP ...")
                ret = subprocess.run([
                    sys.executable,
                    os.path.join(SCRIPT_DIR, "init_basketball_gaussian.py"),
                    "--ply", COLMAP_PLY, "--output", INIT_PLY,
                ], capture_output=False, text=True)
                if ret.returncode != 0 or not os.path.exists(INIT_PLY):
                    print(f"ERROR: failed to generate {INIT_PLY}", file=sys.stderr)
                    sys.exit(1)
            else:
                print(f"ERROR: {n} not found: {p}", file=sys.stderr)
                sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    # CUDA
    cuda_npy = os.path.join(args.outdir, "cuda_render.npy")
    if args.skip_cuda:
        if not os.path.exists(cuda_npy):
            raise RuntimeError(f"--skip_cuda but no {cuda_npy}")
        cuda_np = np.load(cuda_npy)
        print(f"  CUDA render loaded from cache: {cuda_np.shape}")
    else:
        _aaa_root()
        cuda_np = run_cuda_training(args.outdir, args.steps)

    # VK
    vk_npy = os.path.join(args.outdir, "vk_render.npy")
    if args.skip_vk:
        if not os.path.exists(vk_npy):
            raise RuntimeError(f"--skip_vk but no {vk_npy}")
        vk_np = np.load(vk_npy)
        print(f"  VK render loaded from cache: {vk_np.shape}")
    else:
        vk_np = run_vk_training(args.outdir, args.steps)

    compare_and_report(cuda_np, vk_np, args.outdir, args.steps)


if __name__ == "__main__":
    main()
