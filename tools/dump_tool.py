#!/usr/bin/env python3
"""SP-0 CUDA golden dumper.

Orchestrates step/camera/seed and calls AAA-Gaussians CUDA extension to
produce .npy golden artifacts at specified ladder steps.

Usage:
  python -m tools.dump_tool --fixture=tiny --output=harmonyos_3dgs/tests/golden/tiny
  python -m tools.dump_tool --fixture=basketball --output=dev_notes/ground_truth \
                             --ladder=1,10,100,500,1000,2000
  python -m tools.dump_tool --fixture=basketball --output=dev_notes/ground_truth \
                             --dump_steps=1234,1235  # on-demand for bisection
"""
import argparse
import subprocess
import sys
from pathlib import Path


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", choices=["tiny", "basketball"], required=True)
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--ladder", default="",
                   help="Comma-separated step numbers to dump (e.g. 1,10,100)")
    p.add_argument("--dump_steps", default="",
                   help="Additional on-demand step numbers (e.g. 1234,1235)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source_path", default="/home/robota/Downloads/basketball")
    p.add_argument("--ply_path",
                   default="/home/robota/Downloads/basketball/sparse/0/points3D.ply")
    p.add_argument("--iterations", type=int, default=2000,
                   help="Total training steps when fixture=basketball")
    p.add_argument("--densify_from_iter", type=int, default=5000)
    p.add_argument("--densify_until_iter", type=int, default=10000)
    p.add_argument("--densification_interval", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    steps = set()
    if args.ladder:
        steps.update(int(s) for s in args.ladder.split(",") if s)
    if args.dump_steps:
        steps.update(int(s) for s in args.dump_steps.split(",") if s)

    if args.fixture == "tiny":
        from tools.dump_tiny import run_tiny
        run_tiny(output_dir=str(out), seed=args.seed)
    elif args.fixture == "basketball":
        if not steps:
            print("--ladder or --dump_steps required for basketball", file=sys.stderr)
            sys.exit(2)
        from tools.dump_basketball import run_basketball
        run_basketball(output_dir=str(out), steps=sorted(steps),
                       source_path=args.source_path, ply_path=args.ply_path,
                       iterations=args.iterations, seed=args.seed,
                       densify_from_iter=args.densify_from_iter,
                       densify_until_iter=args.densify_until_iter,
                       densification_interval=args.densification_interval)
    else:
        raise ValueError(args.fixture)


if __name__ == "__main__":
    main()
