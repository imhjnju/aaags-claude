#!/usr/bin/env python3
"""Reference Python training using official diff-gaussian-rasterization.

Outputs per-iteration CSV: iteration,loss,view_idx
Designed to be compared step-by-step with C++ gs3d_train output.

Usage:
    python harmonyos_3dgs/tools/train_reference_py.py \
        -s /path/to/colmap_dataset \
        --iterations 500 --sh_degree 0 --resolution 8
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import l1_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--sh_degree", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--output", default="train/ref_py")
    parser.add_argument("--no_densify", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Setup official 3DGS
    sys.argv = [
        "train.py", "-s", args.source,
        "-m", args.output,
        "--iterations", str(args.iterations),
        "--sh_degree", str(args.sh_degree),
        "--resolution", str(args.resolution),
        "--eval", "--depths", "",
    ]
    full_parser = argparse.ArgumentParser()
    mp = ModelParams(full_parser, sentinel=True)
    pp = PipelineParams(full_parser)
    op = OptimizationParams(full_parser)
    parsed = full_parser.parse_args(sys.argv[1:])
    dataset = mp.extract(parsed)
    opt = op.extract(parsed)
    pipe = pp.extract(parsed)

    # Disable densification
    if args.no_densify:
        opt.densify_from_iter = args.iterations + 1
        opt.densify_until_iter = 0

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    train_cams = scene.getTrainCameras()
    N = gaussians.get_xyz.shape[0]

    print(f"# Gaussians: {N}, Views: {len(train_cams)}, "
          f"SH: {dataset.sh_degree}, Res: 1/{args.resolution}", file=sys.stderr)

    # CSV header
    print("iteration,loss,view_idx,n_gaussians")

    for iteration in range(1, args.iterations + 1):
        # Random view (same seed scheme as official train.py)
        viewpoint_idx = torch.randint(0, len(train_cams), (1,)).item()
        viewpoint_cam = train_cams[viewpoint_idx]

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]
        gt = viewpoint_cam.original_image.cuda()
        loss = l1_loss(image, gt)

        loss.backward()

        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians.update_learning_rate(iteration)

        print(f"{iteration},{loss.item():.8f},{viewpoint_idx},{N}")

        if iteration <= 5 or iteration % 100 == 0:
            print(f"  iter {iteration}: loss={loss.item():.6f} view={viewpoint_idx}",
                  file=sys.stderr)

    print(f"Done. Final loss={loss.item():.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
