#!/usr/bin/env python3
"""Compare VK and CUDA training quality after 2000 steps on basketball (all cameras).

Both sides:
  Init PLY : /home/robota/Downloads/basketball/sparse/0/points3D.ply
  Dataset  : /home/robota/Downloads/basketball  (76 cameras, COLMAP text format)
  Steps    : 2000, no densification
  Final render: camera 0 (78899858295079.jpg) for both

Output in <outdir>/:
  cuda_render.png   — CUDA final render from cam0
  vk_render.png     — VK   final render from cam0
  gt_cam0.png       — ground truth cam0
  comparison.png    — side-by-side: GT | CUDA | VK  (2160px wide)
  report.txt        — PSNR table

Usage (from repo root, inside conda aaa-gs env):
  python harmonyos_3dgs/tools/compare_vk_cuda_2000step.py [--outdir /tmp/compare_2000]
  python harmonyos_3dgs/tools/compare_vk_cuda_2000step.py --skip_cuda  # only run VK
  python harmonyos_3dgs/tools/compare_vk_cuda_2000step.py --skip_vk   # only run CUDA
"""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_DIR     = os.path.dirname(SCRIPT_DIR)           # harmonyos_3dgs/
WORKTREE     = os.path.dirname(REPO_DIR)             # worktree root
BUILD_DIR    = os.path.join(REPO_DIR, "build")

COLMAP_DIR      = "/home/robota/Downloads/basketball"
COLMAP_PLY_PATH = os.path.join(COLMAP_DIR, "sparse/0/points3D.ply")
INIT_3DGS_PLY   = "/tmp/basketball_init_3dgs.ply"   # 3DGS-format init PLY for VK
IMG_DIR         = os.path.join(COLMAP_DIR, "images")
CAM_JSON        = os.path.join(COLMAP_DIR, "_sp0_dump_output/cameras.json")
GT_JPG          = os.path.join(IMG_DIR, "78899858295079.jpg")

STEPS = 2000
CAM0_IDX = 0   # index in cameras.json for 78899858295079

# ---------------------------------------------------------------------------
# AAA-Gaussians on path
# ---------------------------------------------------------------------------
def _add_aaa_root():
    for candidate in [
        os.path.join(WORKTREE, "..", "..", "AAA-Gaussians"),
        os.path.join(WORKTREE, "..", "AAA-Gaussians"),
        "/home/robota/h00813233/Graph/aaags-claude/AAA-Gaussians",
    ]:
        cand = os.path.abspath(candidate)
        if os.path.isdir(cand):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return cand
    raise RuntimeError("AAA-Gaussians not found")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def save_png(arr_hwc: np.ndarray, path: str):
    u8 = (arr_hwc.clip(0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(u8).save(path)


def write_ppm_u8(path: str, hwc: np.ndarray):
    """Write float32 HWC [0,1] as P6 uint8 PPM (what VK train can read)."""
    H, W = hwc.shape[:2]
    u8 = (hwc.clip(0, 1) * 255 + 0.5).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(f"P6\n{W} {H}\n255\n".encode())
        f.write(u8.tobytes())


def load_ppm(path: str) -> np.ndarray:
    """Load P6 uint8 PPM, return float32 HWC [0,1]."""
    with open(path, "rb") as f:
        magic = f.readline().strip()
        assert magic == b"P6", f"Expected P6, got {magic}"
        dims = f.readline().strip().split()
        W, H = int(dims[0]), int(dims[1])
        f.readline()  # maxval
        data = np.frombuffer(f.read(), dtype=np.uint8).copy()
    return data.reshape(H, W, 3).astype(np.float32) / 255.0


def load_vk_raw(raw_path: str, W: int, H: int) -> np.ndarray:
    """Load raw float CHW from vk_float.raw → float32 HWC [0,1]."""
    data = np.fromfile(raw_path, dtype=np.float32)
    assert data.size == 3 * H * W, f"Size mismatch: {data.size} vs {3*H*W}"
    return data.reshape(3, H, W).transpose(1, 2, 0).clip(0, 1)


# ---------------------------------------------------------------------------
# Convert all basketball JPG/PNG → PPM for VK train
# ---------------------------------------------------------------------------
def convert_images_to_ppm(ppm_dir: str):
    os.makedirs(ppm_dir, exist_ok=True)
    cam_data = json.load(open(CAM_JSON))
    img_names = {c["img_name"] for c in cam_data}

    converted = 0
    for ext in [".jpg", ".JPG", ".jpeg", ".png", ".PNG"]:
        for fname in os.listdir(IMG_DIR):
            stem, fext = os.path.splitext(fname)
            if fext.lower() != ext.lower().lower():
                continue
            if stem not in img_names:
                continue
            dst = os.path.join(ppm_dir, stem + ".ppm")
            if os.path.exists(dst):
                continue
            src = os.path.join(IMG_DIR, fname)
            img = np.array(Image.open(src).convert("RGB"), dtype=np.float32) / 255.0
            write_ppm_u8(dst, img)
            converted += 1

    print(f"  Converted {converted} images to PPM in {ppm_dir}")


# ---------------------------------------------------------------------------
# CUDA training via subprocess (train_reference_py.py)
# ---------------------------------------------------------------------------
def run_cuda_training(outdir: str) -> str:
    """Run CUDA training; returns cuda_out directory."""
    print(f"\n=== CUDA Training ({STEPS} steps, all cameras) ===")
    t0 = time.time()

    cuda_out = os.path.join(outdir, "cuda_train_out")
    os.makedirs(cuda_out, exist_ok=True)

    # Skip if already trained
    if os.path.exists(os.path.join(cuda_out, "final_render.npy")):
        print(f"  CUDA output already exists — skipping training.")
        return cuda_out

    aaa_root = _add_aaa_root()
    script = os.path.join(SCRIPT_DIR, "train_reference_py.py")
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = aaa_root + (":" + existing_pp if existing_pp else "")

    cmd = [
        sys.executable, script,
        "-s", COLMAP_DIR,
        "--iterations", str(STEPS),
        "--sh_degree", "3",
        "--resolution", "1",        # full resolution
        "--output", cuda_out,
        "--no_densify",
    ]
    print(f"  Running: {' '.join(cmd)}")
    print(f"  PYTHONPATH += {aaa_root}")
    proc = subprocess.run(cmd, capture_output=False, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"CUDA training failed (exit {proc.returncode})")

    print(f"[CUDA] Training done in {time.time()-t0:.1f}s.")
    return cuda_out


# ---------------------------------------------------------------------------
# Load CUDA render from saved npy (produced by train_reference_py.py)
# ---------------------------------------------------------------------------
def cuda_load_render(cuda_out: str, outdir: str) -> tuple:
    """Load final_render.npy and gt_cam0.npy saved by train_reference_py.py."""
    render_path = os.path.join(cuda_out, "final_render.npy")
    gt_path     = os.path.join(cuda_out, "gt_cam0.npy")

    if not os.path.exists(render_path):
        raise RuntimeError(f"CUDA final render not found: {render_path}")
    cuda_np = np.load(render_path)   # HWC float32 [0,1]

    cuda_gt_np = np.load(gt_path) if os.path.exists(gt_path) else None
    if cuda_gt_np is None:
        print("  Warning: gt_cam0.npy not found; using cam0 JPG as GT")
        cuda_gt_np = np.array(Image.open(GT_JPG).convert("RGB"),
                              dtype=np.float32) / 255.0

    np.save(os.path.join(outdir, "cuda_render.npy"), cuda_np)
    save_png(cuda_np, os.path.join(outdir, "cuda_render.png"))
    print(f"  CUDA render loaded: {cuda_np.shape}  "
          f"PSNR(vs cuda_gt)={psnr(cuda_np, cuda_gt_np):.2f} dB")
    return cuda_np, cuda_gt_np


# ---------------------------------------------------------------------------
# VK training via gs3d_vk_train
# ---------------------------------------------------------------------------
def run_vk_training(outdir: str, ppm_dir: str) -> str:
    """Run VK training; returns path to trained PLY."""
    print(f"\n=== VK Training ({STEPS} steps, all cameras) ===")
    t0 = time.time()

    vk_train = os.path.join(BUILD_DIR, "gs3d_vk_train")
    vk_ply   = os.path.join(outdir, "vk_trained.ply")

    cmd = [
        vk_train,
        "--ply",        INIT_3DGS_PLY,   # 3DGS-format init (not raw COLMAP PLY)
        "--cameras",    CAM_JSON,
        "--gt_dir",     ppm_dir,
        "--output",     vk_ply,
        "--iterations", str(STEPS),
        "--log_every",  "200",
        "--training_preset", "fast",
    ]
    print(f"  Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=False, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"gs3d_vk_train failed (exit {proc.returncode})")

    print(f"[VK] Training done in {time.time()-t0:.1f}s. PLY: {vk_ply}")
    return vk_ply


# ---------------------------------------------------------------------------
# VK render from cam0 via gs3d_vk_render
# ---------------------------------------------------------------------------
def vk_render_cam0(ply_path: str, outdir: str) -> np.ndarray:
    """Render VK model from cam0; returns float32 HWC [0,1]."""
    print(f"\n  Rendering VK model from cam0 ...")

    vk_render = os.path.join(BUILD_DIR, "gs3d_vk_render")
    render_wd  = os.path.join(outdir, "vk_render_tmp")
    os.makedirs(render_wd, exist_ok=True)

    cmd = [vk_render, ply_path, CAM_JSON, str(CAM0_IDX), "--eval_3d", "0", "--proper_ewa", "0"]
    print(f"  Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=False, text=True, cwd=render_wd)
    if proc.returncode != 0:
        raise RuntimeError(f"gs3d_vk_render failed (exit {proc.returncode})")

    raw_path = os.path.join(render_wd, "vk_float.raw")
    ppm_path = os.path.join(render_wd, "output_vk.ppm")

    # Try raw float first (higher precision)
    cam_data  = json.load(open(CAM_JSON))
    cam0_info = cam_data[CAM0_IDX]
    W, H = cam0_info["width"], cam0_info["height"]

    if os.path.exists(raw_path):
        vk_np = load_vk_raw(raw_path, W, H)
    elif os.path.exists(ppm_path):
        vk_np = load_ppm(ppm_path)
    else:
        raise RuntimeError("No render output from gs3d_vk_render")

    np.save(os.path.join(outdir, "vk_render.npy"), vk_np)
    save_png(vk_np, os.path.join(outdir, "vk_render.png"))
    print(f"  VK render saved: {outdir}/vk_render.png  ({W}×{H})")
    return vk_np


# ---------------------------------------------------------------------------
# Compare and report
# ---------------------------------------------------------------------------
def _resize_hwc(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize HWC float32 array to (H,W,3) using PIL bilinear."""
    u8 = (arr.clip(0, 1) * 255 + 0.5).astype(np.uint8)
    resized = np.array(Image.fromarray(u8).resize((W, H), Image.BILINEAR),
                       dtype=np.float32) / 255.0
    return resized


def compare_and_report(cuda_np, cuda_gt_np, vk_np, outdir):
    vk_gt_np = np.array(Image.open(GT_JPG).convert("RGB"), dtype=np.float32) / 255.0
    save_png(vk_gt_np, os.path.join(outdir, "gt_cam0.png"))
    save_png(cuda_gt_np, os.path.join(outdir, "gt_cuda_cam.png"))

    # PSNR each vs its own GT (ground truth for the rendered camera)
    p_cuda_gt = psnr(cuda_np, cuda_gt_np)
    p_vk_gt   = psnr(vk_np,   vk_gt_np)

    # Cross-comparison: resize to common size for pixel comparison
    H_vk, W_vk = vk_np.shape[:2]
    cuda_rs = _resize_hwc(cuda_np, H_vk, W_vk)
    p_vk_cuda = psnr(vk_np, cuda_rs)

    report = f"""
====================================================
  VK vs CUDA {STEPS}-step Training Comparison
  Scene   : basketball (76 cameras, COLMAP)
  Init    : points3D.ply  (COLMAP sparse)
  CUDA    : {cuda_np.shape[1]}×{cuda_np.shape[0]} — 3DGS train_cams[0]
  VK      : {vk_np.shape[1]}×{vk_np.shape[0]} — cameras.json cam0 (78899858295079)
====================================================
  PSNR (CUDA render vs its GT) = {p_cuda_gt:6.2f} dB
  PSNR (VK   render vs cam0 GT)= {p_vk_gt:6.2f} dB
  PSNR (VK vs CUDA, same size) = {p_vk_cuda:6.2f} dB
  (Note: if CUDA and VK rendered different views, VK-vs-CUDA PSNR is
   not purely a quality metric but still indicates visual similarity.)
====================================================
"""
    print(report)
    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write(report)

    # Side-by-side at VK resolution: cuda_gt | CUDA | VK | vk_gt
    H, W = H_vk, W_vk
    panels = [
        _resize_hwc(cuda_gt_np, H, W),
        cuda_rs,
        vk_np,
        vk_gt_np,
    ]
    canvas = np.concatenate(panels, axis=1)
    save_png(canvas, os.path.join(outdir, "comparison.png"))
    print(f"  Saved: {outdir}/comparison.png  (CUDA-GT | CUDA | VK | VK-GT)")
    print(f"  Saved: {outdir}/report.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir",    default="/tmp/compare_vk_cuda_2000")
    parser.add_argument("--skip_cuda", action="store_true")
    parser.add_argument("--skip_vk",   action="store_true")
    args = parser.parse_args()

    for path, name in [(COLMAP_PLY_PATH, "COLMAP PLY"), (GT_JPG, "GT image"), (CAM_JSON, "cameras.json")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    ppm_dir = os.path.join(args.outdir, "ppm")

    # Step 0a: Generate 3DGS-format init PLY for VK (if not exists)
    if not args.skip_vk and not os.path.exists(INIT_3DGS_PLY):
        init_script = os.path.join(SCRIPT_DIR, "init_basketball_gaussian.py")
        print(f"\n  Generating 3DGS init PLY: {INIT_3DGS_PLY}")
        ret = subprocess.run([sys.executable, init_script,
                              "--ply", COLMAP_PLY_PATH,
                              "--output", INIT_3DGS_PLY],
                             capture_output=False, text=True)
        if ret.returncode != 0:
            raise RuntimeError("init_basketball_gaussian.py failed")

    # Step 0b: Convert images to PPM (needed for VK train)
    if not args.skip_vk:
        convert_images_to_ppm(ppm_dir)

    # Step 2: CUDA training
    cuda_out      = os.path.join(args.outdir, "cuda_train_out")
    cuda_render_npy = os.path.join(args.outdir, "cuda_render.npy")
    cuda_gt_npy   = os.path.join(args.outdir, "cuda_gt.npy")
    if args.skip_cuda:
        if not os.path.exists(cuda_render_npy):
            raise RuntimeError(f"--skip_cuda but no existing render: {cuda_render_npy}")
        cuda_np    = np.load(cuda_render_npy)
        cuda_gt_np = np.load(cuda_gt_npy) if os.path.exists(cuda_gt_npy) else \
                     np.array(Image.open(GT_JPG).convert("RGB"), dtype=np.float32) / 255.0
        print(f"  Loaded existing CUDA render: {cuda_render_npy}")
    else:
        cuda_out = run_cuda_training(args.outdir)
        cuda_np, cuda_gt_np = cuda_load_render(cuda_out, args.outdir)
        np.save(cuda_gt_npy, cuda_gt_np)

    # Step 3: VK training + render
    vk_render_npy = os.path.join(args.outdir, "vk_render.npy")
    if args.skip_vk:
        if not os.path.exists(vk_render_npy):
            raise RuntimeError(f"--skip_vk but no existing render: {vk_render_npy}")
        vk_np = np.load(vk_render_npy)
        print(f"  Loaded existing VK render: {vk_render_npy}")
    else:
        vk_ply = run_vk_training(args.outdir, ppm_dir)
        vk_np  = vk_render_cam0(vk_ply, args.outdir)

    # Step 4: Compare
    compare_and_report(cuda_np, cuda_gt_np, vk_np, args.outdir)


if __name__ == "__main__":
    main()
