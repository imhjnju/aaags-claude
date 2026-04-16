#!/usr/bin/env python3
"""Compare official Python 3DGS training vs C++ training step by step.

Runs both on the same dataset with matching configurations, outputs per-step
loss comparison. Uses the official diff-gaussian-rasterization for Python side.

Usage:
    python harmonyos_3dgs/tools/compare_training_py_cpp.py \
        --data /home/robota/Downloads/360_extra_scenes/flowers \
        --iterations 1000 --resolution 8 --sh_degree 0
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
import torch
import torch.nn as nn
from torch import optim

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import l1_loss


def run_python_training(args):
    """Run official 3DGS Python training and collect per-step loss."""

    # Setup model params
    parser = argparse.ArgumentParser()
    mp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)

    # Override with our args
    sys.argv = [
        "train.py",
        "-s", args.data,
        "-m", os.path.join(args.output, "python"),
        "--iterations", str(args.iterations),
        "--sh_degree", str(args.sh_degree),
        "--resolution", str(args.resolution),
        "--eval",
    ]
    parsed = parser.parse_args(sys.argv[1:])
    dataset = mp.extract(parsed)
    opt = op.extract(parsed)
    pipe = pp.extract(parsed)

    # Override densification to match C++ (disabled for now)
    opt.densify_from_iter = args.iterations + 1  # never densify
    opt.densify_until_iter = 0

    # Setup
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cameras = scene.getTrainCameras()
    n_views = len(train_cameras)

    print(f"Python: {gaussians.get_xyz.shape[0]} Gaussians, {n_views} views, "
          f"SH degree {dataset.sh_degree}", file=sys.stderr)

    losses = []
    views_used = []
    for iteration in range(1, args.iterations + 1):
        # Pick view (same random order for fairness — use iteration as seed)
        torch.manual_seed(iteration)
        viewpoint_idx = torch.randint(0, n_views, (1,)).item()
        viewpoint_cam = train_cameras[viewpoint_idx]

        # Forward
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]

        # Loss (L1 only, no SSIM for simplicity)
        gt_image = viewpoint_cam.original_image.cuda()
        loss = l1_loss(image, gt_image)

        # Backward
        loss.backward()

        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            # Update LR
            gaussians.update_learning_rate(iteration)

        loss_val = loss.item()
        losses.append(loss_val)
        views_used.append(viewpoint_idx)

        if iteration <= 5 or iteration % (args.iterations // 20 or 1) == 0:
            print(f"  py iter {iteration:6d}: loss={loss_val:.6f} view={viewpoint_idx}",
                  file=sys.stderr)

    return losses, views_used, gaussians.get_xyz.shape[0]


def run_cpp_training(args):
    """Run C++ training and collect per-step loss from CSV output."""

    cpp_exe = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "build", "gs3d_train")
    if not os.path.exists(cpp_exe):
        cpp_exe = "./build/gs3d_train"
    if not os.path.exists(cpp_exe):
        print(f"ERROR: C++ executable not found at {cpp_exe}", file=sys.stderr)
        return None, None, 0

    # Need PLY + cameras.json + GT images in PPM format
    # Use colmap_to_train.py to convert
    out_dir = os.path.join(args.output, "cpp_data")
    os.makedirs(out_dir, exist_ok=True)

    convert_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "colmap_to_train.py")
    subprocess.run([
        sys.executable, convert_script,
        args.data, out_dir,
        "--resolution", str(args.resolution)
    ], capture_output=True)

    # Convert JPEG GT images to PPM
    gt_ppm_dir = os.path.join(out_dir, "gt_ppm")
    os.makedirs(gt_ppm_dir, exist_ok=True)
    gt_dir = os.path.join(out_dir, "gt_images")
    for f in os.listdir(gt_dir):
        if f.endswith((".JPG", ".jpg", ".png")):
            base = os.path.splitext(f)[0]
            src = os.path.join(gt_dir, f)
            # Follow symlinks
            if os.path.islink(src):
                src = os.path.realpath(src)
            dst = os.path.join(gt_ppm_dir, base + ".ppm")
            if not os.path.exists(dst):
                os.system(f'convert "{src}" "{dst}" 2>/dev/null')
            # Also create name.JPG.ppm symlink
            link = os.path.join(gt_ppm_dir, f + ".ppm")
            if not os.path.exists(link):
                os.symlink(base + ".ppm", link)

    ply_path = os.path.join(out_dir, "init_points.ply")
    cam_path = os.path.join(out_dir, "cameras.json")

    # Run C++ training with per-iteration logging
    cmd = [
        cpp_exe,
        "--ply", ply_path,
        "--cameras", cam_path,
        "--gt_dir", gt_ppm_dir,
        "--output", os.path.join(args.output, "cpp_trained.ply"),
        "--iterations", str(args.iterations),
        "--sh_degree", str(args.sh_degree),
        "--lr_scale", str(args.lr_scale),
        "--log_every", "1",
    ]

    print(f"C++ cmd: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.iterations * 10)

    # Parse losses from C++ output
    import re
    losses = []
    views_used = []
    n_gaussians = 0
    for line in result.stdout.split("\n"):
        m = re.search(r"loss=([\d.]+)\s+view=(\d+)", line)
        if m:
            losses.append(float(m.group(1)))
            views_used.append(int(m.group(2)))
        m2 = re.search(r"Loaded (\d+) Gaussians", line)
        if m2:
            n_gaussians = int(m2.group(1))

    if not losses:
        print(f"C++ stderr: {result.stderr[:500]}", file=sys.stderr)

    return losses, views_used, n_gaussians


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="COLMAP dataset directory")
    parser.add_argument("--output", default="train/compare_py_cpp", help="Output directory")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--sh_degree", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--lr_scale", type=float, default=1.0, help="C++ LR scale factor")
    parser.add_argument("--skip_python", action="store_true")
    parser.add_argument("--skip_cpp", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Run Python training
    py_losses, py_views, py_N = None, None, 0
    if not args.skip_python:
        print("=== Running Python (official 3DGS) training ===", file=sys.stderr)
        py_losses, py_views, py_N = run_python_training(args)
        np.save(os.path.join(args.output, "py_losses.npy"), np.array(py_losses))
        print(f"Python: {len(py_losses)} steps, N={py_N}", file=sys.stderr)
    else:
        py_losses = np.load(os.path.join(args.output, "py_losses.npy")).tolist()

    # Run C++ training
    cpp_losses, cpp_views, cpp_N = None, None, 0
    if not args.skip_cpp:
        print("\n=== Running C++ training ===", file=sys.stderr)
        cpp_losses, cpp_views, cpp_N = run_cpp_training(args)
        if cpp_losses:
            np.save(os.path.join(args.output, "cpp_losses.npy"), np.array(cpp_losses))
        print(f"C++: {len(cpp_losses) if cpp_losses else 0} steps, N={cpp_N}", file=sys.stderr)
    else:
        cpp_losses = np.load(os.path.join(args.output, "cpp_losses.npy")).tolist()

    # Compare
    if not py_losses or not cpp_losses:
        print("ERROR: Missing loss data", file=sys.stderr)
        return

    n = min(len(py_losses), len(cpp_losses))
    py = np.array(py_losses[:n])
    cpp = np.array(cpp_losses[:n])

    # Smoothed comparison (different views, so raw comparison is noisy)
    window = max(10, n // 20)
    py_smooth = np.convolve(py, np.ones(window) / window, mode="valid")
    cpp_smooth = np.convolve(cpp, np.ones(window) / window, mode="valid")

    print(f"\n{'='*70}")
    print(f"TRAINING COMPARISON: Python (official) vs C++")
    print(f"{'='*70}")
    print(f"  Python:  {py_N} Gaussians, {len(py_losses)} steps")
    print(f"  C++:     {cpp_N} Gaussians, {len(cpp_losses)} steps")
    print(f"  Dataset: {args.data}")
    print(f"  Config:  SH{args.sh_degree}, resolution=1/{args.resolution}")
    print()

    # Loss at checkpoints (smoothed)
    print(f"  {'Iter':>6s}  {'Python':>10s}  {'C++':>10s}  {'Diff%':>8s}")
    print(f"  {'-'*40}")
    for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        idx = min(int(frac * (len(py_smooth) - 1)), len(py_smooth) - 1)
        it = idx + window // 2
        rel = (cpp_smooth[idx] / py_smooth[idx] - 1) * 100 if py_smooth[idx] > 0 else 0
        print(f"  {it:6d}  {py_smooth[idx]:10.4f}  {cpp_smooth[idx]:10.4f}  {rel:+7.1f}%")

    # Final comparison
    py_final = np.mean(py[-window:])
    cpp_final = np.mean(cpp[-window:])
    print(f"\n  Final {window}-step avg: Python={py_final:.4f}  C++={cpp_final:.4f}")
    print(f"  C++ vs Python: {(cpp_final/py_final-1)*100:+.1f}%")

    # Convergence trend
    py_drop = (1 - py_smooth[-1] / py_smooth[0]) * 100
    cpp_drop = (1 - cpp_smooth[-1] / cpp_smooth[0]) * 100
    print(f"  Loss drop: Python={py_drop:.1f}%  C++={cpp_drop:.1f}%")

    # Output CSV
    csv_path = os.path.join(args.output, "comparison.csv")
    with open(csv_path, "w") as f:
        f.write("iteration,python_loss,cpp_loss\n")
        for i in range(n):
            f.write(f"{i+1},{py[i]:.6f},{cpp[i]:.6f}\n")
    print(f"\n  CSV: {csv_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
