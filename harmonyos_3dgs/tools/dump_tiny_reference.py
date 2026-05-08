#!/usr/bin/env python3
"""Dump Python autograd reference for the tiny golden fixture.

Runs 100 steps of PyTorch training with VulkanTrainer-matching hyperparameters
and saves per-step loss, gradients (before Adam), and params (after Adam) as .npy.

Usage:
    python tools/dump_tiny_reference.py [--steps 100] [--output-dir tests/golden/tiny/py_ref]

TEMP DIAGNOSTIC — Phase D.deep step-1 audit:
    --dump-step1-intermediates flag adds these files for step 1 only:
      step_0001_int_means2D.npy        [N, 2]
      step_0001_int_depths.npy         [N]
      step_0001_int_cov2D.npy          [N, 3]    (a, b, c) post-dilation
      step_0001_int_conic.npy          [N, 3]    (a, b, c) inverse cov2D
      step_0001_int_rgb.npy            [N, 3]
      step_0001_int_visible.npy        [N]      bool
      step_0001_int_T_final.npy        [H*W]    transmittance at final
      step_0001_int_rendered.npy       [3,H,W]  CHW
      step_0001_int_dLdimage.npy       [3,H,W]  CHW
      step_0001_int_p_view.npy         [N, 3]
      step_0001_int_cov3D.npy          [N, 6]   (xx, xy, xz, yy, yz, zz)
      step_0001_int_act_op.npy         [N]
      step_0001_int_act_scales.npy     [N, 3]
      step_0001_int_act_rotations.npy  [N, 4]   (post-normalize)
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


# TEMP DIAGNOSTIC — Phase D.deep step-1 audit
# ---------------------------------------------------------------------------
# Re-implementation of forward_render_pytorch that ALSO returns intermediates.
# Runs same math as the imported function. Used only when --dump-step1-intermediates.
def forward_render_with_intermediates(raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
                                       vm, vpm, cam_pos, W, H, bg, sh_degree=0):
    N = raw_pos.shape[0]
    device = raw_pos.device

    positions = raw_pos
    scales = torch.exp(raw_sc)
    quat_len = torch.norm(raw_rot, dim=1, keepdim=True).clamp(min=1e-12)
    rotations = raw_rot / quat_len
    opacities = torch.sigmoid(raw_op)

    SH_C0 = 0.28209479177387814
    rgb = torch.clamp(SH_C0 * raw_sh[:, :3] + 0.5, min=0.0)

    ones = torch.ones(N, 1, device=device)
    pos_h = torch.cat([positions, ones], dim=1)
    M_vm = vm.reshape(4, 4)
    Mt_vm = M_vm.T
    p_view = (Mt_vm[:3, :3] @ positions.T).T + Mt_vm[:3, 3]

    Mvp = vpm.reshape(4, 4).T
    p_hom = (Mvp @ pos_h.T).T
    p_w = 1.0 / (p_hom[:, 3:4] + 1e-7)
    p_ndc = p_hom[:, :3] * p_w
    pixel_x = ((p_ndc[:, 0] + 1.0) * W - 1.0) * 0.5
    pixel_y = ((p_ndc[:, 1] + 1.0) * H - 1.0) * 0.5

    r, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y + r*z), 2*(x*z - r*y),
        2*(x*y - r*z), 1 - 2*(x*x + z*z), 2*(y*z + r*x),
        2*(x*z + r*y), 2*(y*z - r*x), 1 - 2*(x*x + y*y),
    ], dim=1).reshape(N, 3, 3)

    S = torch.diag_embed(scales)
    M_sr = S @ R
    cov3D_full = M_sr.transpose(1, 2) @ M_sr  # [N, 3, 3]
    # Pack to [xx, xy, xz, yy, yz, zz]
    cov3D = torch.stack([
        cov3D_full[:, 0, 0], cov3D_full[:, 0, 1], cov3D_full[:, 0, 2],
        cov3D_full[:, 1, 1], cov3D_full[:, 1, 2], cov3D_full[:, 2, 2],
    ], dim=1)

    focal_x_val = Mvp[0, 0].item() * W * 0.5
    focal_y_val = Mvp[1, 1].item() * H * 0.5

    tz = p_view[:, 2]
    # Clamp tx/tz, ty/tz to ±1.3 * tan_fov before computing J (CUDA EWA splatting trick).
    # CUDA reference: cuda_rasterizer/forward_common.h:81-86.
    # Vulkan equivalent: src/vulkan/shaders/preprocess.comp:746-752.
    # focal = W/(2*tan_fov) ⇒ tan_fov = W/(2*focal); inverts focal_x_val above.
    tan_fovx = W / (2.0 * focal_x_val)
    tan_fovy = H / (2.0 * focal_y_val)
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = p_view[:, 0] / tz
    tytz = p_view[:, 1] / tz
    tx_clamped = torch.clamp(txtz, min=-limx, max=limx) * tz
    ty_clamped = torch.clamp(tytz, min=-limy, max=limy) * tz
    J = torch.zeros(N, 2, 3, device=device)
    J[:, 0, 0] = focal_x_val / tz
    J[:, 0, 2] = -focal_x_val * tx_clamped / (tz * tz)
    J[:, 1, 1] = focal_y_val / tz
    J[:, 1, 2] = -focal_y_val * ty_clamped / (tz * tz)
    W_mat = Mt_vm[:3, :3]
    T_mat = (W_mat.unsqueeze(0) @ J.transpose(1, 2)).transpose(1, 2)
    cov2D_full = T_mat @ cov3D_full @ T_mat.transpose(1, 2)
    cov2D_full = cov2D_full.clone()
    cov2D_full[:, 0, 0] = cov2D_full[:, 0, 0] + 0.3
    cov2D_full[:, 1, 1] = cov2D_full[:, 1, 1] + 0.3
    # Pack cov2D as (a, b, c) = (00, 01, 11)
    cov2D = torch.stack([
        cov2D_full[:, 0, 0], cov2D_full[:, 0, 1], cov2D_full[:, 1, 1],
    ], dim=1)

    det = cov2D_full[:, 0, 0] * cov2D_full[:, 1, 1] - cov2D_full[:, 0, 1] * cov2D_full[:, 1, 0]
    visible = (p_view[:, 2] > 0.2) & (det > 0)
    # CUDA reference: cuda_rasterizer/forward_common.h:137 — `float det_inv = 1.f / det;` (no clamp).
    # Visibility mask above (`det > 0`) gates non-positive dets. Match CUDA/VK exactly.
    conic_a = cov2D_full[:, 1, 1] / det
    conic_b = -cov2D_full[:, 0, 1] / det
    conic_c = cov2D_full[:, 0, 0] / det
    conic = torch.stack([conic_a, conic_b, conic_c], dim=1)

    # Sort by depth (depth = view-space z)
    depths = p_view[:, 2].clone()
    depths_for_sort = depths.clone()
    depths_for_sort[~visible] = 1e10
    order = torch.argsort(depths_for_sort)

    image = torch.zeros(H, W, 3, device=device)
    T_final_arr = torch.ones(H, W, device=device)
    means2d = torch.stack([pixel_x, pixel_y], dim=1)

    for py in range(H):
        for px in range(W):
            T_val = torch.ones(1, device=device)
            C = torch.zeros(3, device=device)
            for idx in order:
                if not visible[idx]:
                    continue
                dx = means2d[idx, 0] - float(px)
                dy = means2d[idx, 1] - float(py)
                power = -0.5 * (conic_a[idx]*dx*dx + conic_c[idx]*dy*dy) - conic_b[idx]*dx*dy
                if power > 0:
                    continue
                alpha = torch.clamp(opacities[idx] * torch.exp(power), max=0.99)
                if alpha < 1.0/255.0:
                    continue
                test_T = T_val * (1.0 - alpha)
                if test_T < 0.0001:
                    break
                C = C + rgb[idx] * alpha * T_val
                T_val = test_T
            C = C + T_val * bg
            image[py, px] = C
            T_final_arr[py, px] = T_val.item() if T_val.numel() == 1 else T_val[0]

    return dict(
        means2D=means2d.detach().cpu().numpy(),
        depths=depths.detach().cpu().numpy(),
        cov2D=cov2D.detach().cpu().numpy(),
        conic=conic.detach().cpu().numpy(),
        rgb=rgb.detach().cpu().numpy(),
        visible=visible.detach().cpu().numpy(),
        T_final=T_final_arr.detach().cpu().numpy(),
        rendered=image.detach().cpu().numpy(),  # [H,W,3]
        p_view=p_view.detach().cpu().numpy(),
        cov3D=cov3D.detach().cpu().numpy(),
        act_op=opacities.detach().cpu().numpy(),
        act_scales=scales.detach().cpu().numpy(),
        act_rotations=rotations.detach().cpu().numpy(),
    )


def run_dump(n_steps: int, output_dir: str, fixture_dir: str,
             dump_step1_intermediates: bool = False) -> None:
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

    # Background: all zeros
    bg = torch.zeros(3, device=device)

    # GT image: all zeros
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

        # TEMP DIAGNOSTIC — Phase D.deep step-1 audit
        # Re-run forward in eval mode (no autograd) to capture intermediates.
        # This MUST run before the autograd forward; values are deterministic.
        if dump_step1_intermediates and step == 1:
            with torch.no_grad():
                inters = forward_render_with_intermediates(
                    raw_pos.detach(), raw_sc.detach(), raw_rot.detach(),
                    raw_sh.detach(), raw_op.detach(),
                    vm, vpm, cam_pos, W, H, bg, sh_degree=0
                )
            int_prefix = os.path.join(output_dir, "step_0001_int")
            np.save(f"{int_prefix}_means2D.npy",       inters["means2D"].astype(np.float32))
            np.save(f"{int_prefix}_depths.npy",        inters["depths"].astype(np.float32))
            np.save(f"{int_prefix}_cov2D.npy",         inters["cov2D"].astype(np.float32))
            np.save(f"{int_prefix}_conic.npy",         inters["conic"].astype(np.float32))
            np.save(f"{int_prefix}_rgb.npy",           inters["rgb"].astype(np.float32))
            np.save(f"{int_prefix}_visible.npy",       inters["visible"].astype(np.int32))
            np.save(f"{int_prefix}_T_final.npy",       inters["T_final"].astype(np.float32))
            # CHW layout: rendered is [H,W,3] -> transpose to [3,H,W]
            np.save(f"{int_prefix}_rendered.npy",      np.transpose(inters["rendered"], (2,0,1)).astype(np.float32))
            np.save(f"{int_prefix}_p_view.npy",        inters["p_view"].astype(np.float32))
            np.save(f"{int_prefix}_cov3D.npy",         inters["cov3D"].astype(np.float32))
            np.save(f"{int_prefix}_act_op.npy",        inters["act_op"].astype(np.float32))
            np.save(f"{int_prefix}_act_scales.npy",    inters["act_scales"].astype(np.float32))
            np.save(f"{int_prefix}_act_rotations.npy", inters["act_rotations"].astype(np.float32))

        # Forward render — sh_degree=0 for DC-only (matches C++ test)
        rendered = forward_render_pytorch(
            raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
            vm, vpm, cam_pos, W, H, bg, sh_degree=0
        )

        # L1 loss vs all-zeros GT
        loss = torch.mean(torch.abs(rendered - gt))

        # TEMP DIAGNOSTIC — Phase D.deep step-1 audit
        # Capture dL/d(image) at step 1 by retaining the gradient on `rendered`.
        # rendered is [H, W, 3]; dL/d(rendered) = sign(rendered - gt) / numel.
        # We compute it analytically since loss is mean(|x|) — and also save
        # via retain_grad to verify autograd matches.
        if dump_step1_intermediates and step == 1:
            int_prefix = os.path.join(output_dir, "step_0001_int")
            with torch.no_grad():
                # mean(|x|) -> dL/dx = sign(x) / numel
                numel = float(rendered.numel())
                dL_dimage_hwc = torch.sign(rendered - gt) / numel
                # CHW
                np.save(f"{int_prefix}_dLdimage.npy",
                        np.transpose(dL_dimage_hwc.detach().cpu().numpy(), (2,0,1)).astype(np.float32))

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
        np.save(f"{prefix}_raw_sh.npy",
                raw_sh.detach().numpy().reshape(N, max_coeffs * 3).astype(np.float32))

        if step <= 5 or step % 10 == 0 or step == n_steps:
            print(f"  step {step:4d}: loss={loss.item():.8f}  pos_lr={pos_lr_at_step(step):.2e}")

    print(f"\nDone. Saved {n_steps * 11} files to {output_dir}")


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
        "--dump-step1-intermediates", action="store_true",
        help="TEMP DIAGNOSTIC: also dump forward intermediates and dL_dimage for step 1"
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

    run_dump(n_steps=args.steps, output_dir=args.output_dir, fixture_dir=args.fixture_dir,
             dump_step1_intermediates=args.dump_step1_intermediates)


if __name__ == "__main__":
    main()
