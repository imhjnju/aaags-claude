"""Basketball ladder dump — SP-0 T18 placeholder.

Runs AAA-Gaussians training for N steps, dumping at specified ladder steps.

T18 scope: skeleton only. T19 completes implementation.
"""
import sys
from pathlib import Path

AAA_ROOT = Path(__file__).resolve().parents[1] / "AAA-Gaussians"
sys.path.insert(0, str(AAA_ROOT))


def run_basketball(output_dir: str, steps, source_path: str, ply_path: str,
                   iterations: int, seed: int,
                   densify_from_iter: int, densify_until_iter: int,
                   densification_interval: int):
    raise NotImplementedError(
        "dump_basketball.py: placeholder. Full implementation in T19.")
