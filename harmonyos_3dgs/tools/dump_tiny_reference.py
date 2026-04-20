#!/usr/bin/env python3
"""Dump Python autograd reference for the tiny golden fixture.

Runs 100 steps of PyTorch training with VulkanTrainer-matching hyperparameters
and saves per-step loss, gradients (before Adam), and params (after Adam) as .npy.

Usage:
    python tools/dump_tiny_reference.py [--steps 100] [--output-dir tests/golden/tiny/py_ref]
"""

import argparse
import os
import sys

import numpy as np
import torch

# Import the differentiable renderer from the existing reference script.
# Do NOT copy its code — import it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pytorch_reference import forward_render_pytorch


# ---------------------------------------------------------------------------
# Hyperparameters — MUST match VulkanTrainer exactly
# ---------------------------------------------------------------------------
ADAM_EPS = 1e-15          # NOT 1e-8 — critical
POS_LR_INIT = 1.6e-4
POS_LR_FINAL = 1.6e-6
MAX_STEPS_SCHEDULE = 30000  # divisor for LR schedule (NOT number of training steps)
SH_DC_LR = 2.5e-3
OPACITY_LR = 0.05
SCALE_LR = 0.005
ROTATION_LR = 0.001
ADAM_BETAS = (0.9, 0.999)


def pos_lr_at_step(step: int) -> float:
    """Position LR schedule — 1-indexed step."""
    t = min(1.0, max(0.0, step / float(MAX_STEPS_SCHEDULE)))
    return float(np.exp(np.log(POS_LR_INIT) * (1.0 - t) + np.log(POS_LR_FINAL) * t))


def load_fixture(fixture_dir: str):
    """Load tiny golden fixture and convert activated params to raw (network) params."""
    d = fixture_dir

    positions = np.load(os.path.join(d, "input_positions.npy")).astype(np.float32)   # [20, 3]
    scales    = np.load(os.path.join(d, "input_scales.npy")).astype(np.float32)      # [20, 3]
    rotations = np.load(os.path.join(d, "input_rotations.npy")).astype(np.float32)   # [20, 4]
    opacities = np.load(os.path.join(d, "input_opacities.npy")).astype(np.float32)   # [20, 1]
    sh_coeffs = np.load(os.path.join(d, "input_sh.npy")).astype(np.float32)          # [20, 16, 3]
    vm_np     = np.load(os.path.join(d, "input_viewmatrix.npy")).astype(np.float32)  # [4, 4]
    vpm_np    = np.load(os.path.join(d, "input_projmatrix.npy")).astype(np.float32)  # [4, 4]
    campos_np = np.load(os.path.join(d, "input_campos.npy")).astype(np.float32)      # [3]
    fov_size  = np.load(os.path.join(d, "input_fov_size.npy")).astype(np.float32)    # [4]
    meta      = np.load(os.path.join(d, "input_meta.npy")).astype(np.float32)        # [4]

    N = positions.shape[0]
    # sh_degree = int(meta[0])  # stored but we train with degree 0 (DC only) matching C++ test
    max_coeffs = int(meta[1])   # 16 in the fixture
    H = int(meta[2])
    W = int(meta[3])

    # tan_fovx = fov_size[0], tan_fovy = fov_size[1]
    # W = int(fov_size[2]), H = int(fov_size[3])  — redundant with meta, both 64

    # Activated → Raw conversions
    # positions: identity (raw = activated)
    raw_pos_np = positions.copy()

    # scales: activated = exp(raw) → raw = log(activated)
    raw_sc_np = np.log(np.maximum(scales, 1e-12)).astype(np.float32)

    # rotations: identity (already normalized quat)
    raw_rot_np = rotations.copy()

    # opacities: activated = sigmoid(raw) → raw = logit(activated)
    op = np.clip(opacities, 1e-6, 1.0 - 1e-6)
    raw_op_np = (np.log(op / (1.0 - op))).astype(np.float32)   # [20, 1]

    # SH: reshape [N,16,3] → [N, max_coeffs*3]
    raw_sh_np = sh_coeffs.reshape(N, max_coeffs * 3).astype(np.float32)

    # The viewmatrix and projmatrix are stored as [4,4].
    # forward_render_pytorch expects flat [16] tensors (it calls .reshape(4,4) internally).
    vm_flat  = vm_np.flatten().astype(np.float32)
    vpm_flat = vpm_np.flatten().astype(np.float32)

    return dict(
        N=N,
        W=W,
        H=H,
        max_coeffs=max_coeffs,
        raw_pos_np=raw_pos_np,
        raw_sc_np=raw_sc_np,
        raw_rot_np=raw_rot_np,
        raw_op_np=raw_op_np,
        raw_sh_np=raw_sh_np,
        vm_flat=vm_flat,
        vpm_flat=vpm_flat,
        campos_np=campos_np,
    )


def run_dump(n_steps: int, output_dir: str, fixture_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    data = load_fixture(fixture_dir)
    N         = data["N"]
    W         = data["W"]
    H         = data["H"]
    max_coeffs = data["max_coeffs"]

    device = "cpu"  # Must be CPU — forward_render_pytorch uses per-pixel Python loop

    # Create PyTorch leaf parameters (requires_grad=True)
    raw_pos = torch.tensor(data["raw_pos_np"], dtype=torch.float32, device=device, requires_grad=True)
    raw_sc  = torch.tensor(data["raw_sc_np"],  dtype=torch.float32, device=device, requires_grad=True)
    raw_rot = torch.tensor(data["raw_rot_np"], dtype=torch.float32, device=device, requires_grad=True)
    raw_op  = torch.tensor(data["raw_op_np"],  dtype=torch.float32, device=device, requires_grad=True)
    raw_sh  = torch.tensor(data["raw_sh_np"],  dtype=torch.float32, device=device, requires_grad=True)

    vm       = torch.tensor(data["vm_flat"],    dtype=torch.float32, device=device)
    vpm      = torch.tensor(data["vpm_flat"],   dtype=torch.float32, device=device)
    cam_pos  = torch.tensor(data["campos_np"],  dtype=torch.float32, device=device)

    # Background: all zeros — matches C++ test_cpu_vk_compare.cpp
    bg = torch.zeros(3, device=device)

    # GT image: all zeros — matches C++ test_cpu_vk_compare.cpp
    gt = torch.zeros(H, W, 3, device=device)

    # Adam param groups (order matters — matches VK groups 0–4)
    optimizer = torch.optim.Adam([
        {"params": [raw_pos], "lr": pos_lr_at_step(1)},  # group 0: positions
        {"params": [raw_sh],  "lr": SH_DC_LR},            # group 1: sh DC
        {"params": [raw_op],  "lr": OPACITY_LR},          # group 2: opacities
        {"params": [raw_sc],  "lr": SCALE_LR},            # group 3: scales
        {"params": [raw_rot], "lr": ROTATION_LR},         # group 4: rotations
    ], betas=ADAM_BETAS, eps=ADAM_EPS)

    print(f"Starting dump: N={N}, {W}x{H}, max_coeffs={max_coeffs}, steps={n_steps}")
    print(f"Output dir: {output_dir}")

    for step in range(1, n_steps + 1):
        # Update position LR first (1-indexed)
        optimizer.param_groups[0]["lr"] = pos_lr_at_step(step)

        optimizer.zero_grad()

        # Forward render — sh_degree=0 for DC-only (matches C++ test)
        rendered = forward_render_pytorch(
            raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
            vm, vpm, cam_pos, W, H, bg, sh_degree=0
        )

        # L1 loss vs all-zeros GT
        loss = torch.mean(torch.abs(rendered - gt))
        loss.backward()

        # --- SAVE GRADIENTS (before optimizer.step()) ---
        prefix = os.path.join(output_dir, f"step_{step:04d}")

        np.save(f"{prefix}_loss.npy",
                np.array([loss.item()], dtype=np.float32))

        np.save(f"{prefix}_grad_pos.npy",
                raw_pos.grad.detach().numpy().reshape(N, 3).astype(np.float32))
        np.save(f"{prefix}_grad_sc.npy",
                raw_sc.grad.detach().numpy().reshape(N, 3).astype(np.float32))
        np.save(f"{prefix}_grad_rot.npy",
                raw_rot.grad.detach().numpy().reshape(N, 4).astype(np.float32))
        np.save(f"{prefix}_grad_sh.npy",
                raw_sh.grad.detach().numpy().reshape(N, max_coeffs * 3).astype(np.float32))
        np.save(f"{prefix}_grad_op.npy",
                raw_op.grad.detach().numpy().reshape(N).astype(np.float32))

        # Adam step
        optimizer.step()

        # --- SAVE PARAMS (after optimizer.step()) ---
        np.save(f"{prefix}_raw_pos.npy",
                raw_pos.detach().numpy().reshape(N, 3).astype(np.float32))
        np.save(f"{prefix}_raw_sc.npy",
                raw_sc.detach().numpy().reshape(N, 3).astype(np.float32))
        np.save(f"{prefix}_raw_rot.npy",
                raw_rot.detach().numpy().reshape(N, 4).astype(np.float32))
        np.save(f"{prefix}_raw_op.npy",
                raw_op.detach().numpy().reshape(N).astype(np.float32))

        if step <= 5 or step % 10 == 0 or step == n_steps:
            print(f"  step {step:4d}: loss={loss.item():.8f}  pos_lr={pos_lr_at_step(step):.2e}")

    print(f"\nDone. Saved {n_steps * 10} files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Dump PyTorch autograd reference for the tiny golden fixture."
    )
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Number of training steps to dump (default: 100)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests", "golden", "tiny", "py_ref"
        ),
        help="Directory to write .npy files (default: tests/golden/tiny/py_ref/)"
    )
    parser.add_argument(
        "--fixture-dir", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests", "golden", "tiny", "step000001", "cam0000"
        ),
        help="Directory containing the tiny fixture .npy files"
    )
    args = parser.parse_args()

    run_dump(n_steps=args.steps, output_dir=args.output_dir, fixture_dir=args.fixture_dir)


if __name__ == "__main__":
    main()
