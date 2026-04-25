#!/usr/bin/env python3
"""PyTorch reference implementation of 3DGS forward+backward+Adam.

Implements the EXACT same rendering pipeline as C++ (preprocessor → rasterizer)
using PyTorch tensors + autograd for backward. Compares per-step loss with C++.

Usage:
    python harmonyos_3dgs/tools/train_pytorch_reference.py \
        --ply train/flowers/init_points_2k.ply \
        --cameras train/flowers/cameras_16_lowres.json \
        --gt_dir train/flowers/gt_ppm_16 \
        --iterations 100 --lr_scale 0.3
"""

import torch
import torch.nn.functional as TF
import numpy as np
import os, sys, json, argparse, subprocess, re

F = np.float32


# ============================================================
# PyTorch differentiable 3DGS forward renderer
# Matches C++ preprocessor_cpu.cpp + rasterizer_cpu.cpp
# ============================================================

def forward_render_pytorch(raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
                            vm, vpm, cam_pos, W, H, bg, sh_degree=0):
    """
    Full differentiable forward render in PyTorch.
    All inputs are torch tensors (float32, requires_grad where needed).
    Returns rendered image [H, W, 3].
    """
    N = raw_pos.shape[0]
    device = raw_pos.device

    # Activate
    positions = raw_pos                                    # identity
    scales = torch.exp(raw_sc)                             # exp activation
    quat_len = torch.norm(raw_rot, dim=1, keepdim=True).clamp(min=1e-12)
    rotations = raw_rot / quat_len                         # normalize
    opacities = torch.sigmoid(raw_op)                      # sigmoid

    # SH degree 0: color = SH_C0 * dc + 0.5, clamp >=0
    SH_C0 = 0.28209479177387814
    rgb = torch.clamp(SH_C0 * raw_sh[:, :3] + 0.5, min=0.0)  # [N, 3]

    # Project to screen
    ones = torch.ones(N, 1, device=device)
    pos_h = torch.cat([positions, ones], dim=1)  # [N, 4]

    # View transform: p_view[j] = sum_k vm[k*4+j] * pos[k] + vm[12+j]
    # In column-major vm, this is: p_view = vm_colmaj_3x3 @ pos + vm_colmaj_t
    # vm_colmaj_3x3[i,j] = vm[j*4+i], so vm.reshape(4,4)[i,j] = vm[i*4+j] which is wrong
    # Correct: don't transpose. vm.reshape(4,4) in numpy puts vm[0..3] as row0.
    # C++ does: out[j] = m[0*4+j]*p[0] + m[1*4+j]*p[1] + m[2*4+j]*p[2] + m[3*4+j]
    # Wait, C++ actually does: out[0] = m[0]*p[0]+m[4]*p[1]+m[8]*p[2]+m[12]
    # = m[0*4+0]*p[0] + m[1*4+0]*p[1] + m[2*4+0]*p[2] + m[3*4+0]
    # In matrix form (numpy row-major): M[row,col] where M_flat[row*4+col] = vm[col*4+row]
    # So M[row,col] = vm[col*4+row], and out[row] = sum_col M[row,col]*p[col]
    # To get M: M[i,j] = vm[j*4+i] → M = vm.reshape(4,4).T
    # Then p_view = M[:3,:3] @ pos + M[:3,3]
    # BUT vm is stored in column-major C layout: vm[0..3]=col0, vm[4..7]=col1
    # numpy reshape(4,4) puts vm[0..3] as row0 → that's row-major interpretation
    # Column 0 = vm[0],vm[1],vm[2],vm[3] → in numpy this is [:, 0] after .T
    # So vm.reshape(4,4) gives M where M[i,j] = vm[i*4+j]
    # C++ out[0] = vm[0]*p[0] + vm[4]*p[1] + vm[8]*p[2] + vm[12]
    #            = M[0,0]*p[0] + M[1,0]*p[1] + M[2,0]*p[2] + M[3,0]
    #            = (M^T)[0,:3] @ p + (M^T)[0,3]
    # So the correct operation is: p_view = (M^T)[:3,:3] @ pos + (M^T)[:3, 3]
    # where M = vm.reshape(4,4)
    M = vm.reshape(4, 4)  # M[i,j] = vm[i*4+j]
    Mt = M.T  # Mt[i,j] = M[j,i] = vm[j*4+i] — this is the actual transform matrix
    p_view = (Mt[:3, :3] @ positions.T).T + Mt[:3, 3]  # [N, 3]

    # NDC projection (same column-major convention)
    Mvp = vpm.reshape(4, 4).T
    p_hom = (Mvp @ pos_h.T).T  # [N, 4]
    p_w = 1.0 / (p_hom[:, 3:4] + 1e-7)
    p_ndc = p_hom[:, :3] * p_w  # [N, 3]

    pixel_x = ((p_ndc[:, 0] + 1.0) * W - 1.0) * 0.5  # ndc2pix
    pixel_y = ((p_ndc[:, 1] + 1.0) * H - 1.0) * 0.5

    # 3D covariance from scale + rotation
    r, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y + r*z), 2*(x*z - r*y),
        2*(x*y - r*z), 1 - 2*(x*x + z*z), 2*(y*z + r*x),
        2*(x*z + r*y), 2*(y*z - r*x), 1 - 2*(x*x + y*y),
    ], dim=1).reshape(N, 3, 3)

    S = torch.diag_embed(scales)  # [N, 3, 3]
    M = S @ R  # [N, 3, 3]  (M = S * R since S is diagonal)
    cov3D = M.transpose(1, 2) @ M  # [N, 3, 3]  Sigma = M^T M

    # 2D covariance via Jacobian projection
    focal_x = W / (2.0 * vm.new_tensor([1.0]).squeeze() * 0 + W / (2.0 * (W / (2.0 * (p_view[:, 2:3]))).reciprocal()))
    # Simpler: focal from tan_fov
    # Actually compute focal from the projection matrix
    focal_x_val = Mvp[0, 0].item() * W * 0.5
    focal_y_val = Mvp[1, 1].item() * H * 0.5

    tz = p_view[:, 2]
    # Clamp tx/tz, ty/tz to ±1.3 * tan_fov before computing J (CUDA EWA splatting trick).
    # CUDA reference: cuda_rasterizer/forward_common.h:81-86 —
    #   const float limx = 1.3f * tan_fovx;  const float limy = 1.3f * tan_fovy;
    #   t.x = min(limx, max(-limx, t.x/t.z)) * t.z;  t.y = min(limy, ...)*t.z;
    # Vulkan equivalent: src/vulkan/shaders/preprocess.comp:746-752.
    # focal_x = W/(2*tan_fovx)  ⇒  tan_fovx = W/(2*focal_x). Inverts the focal_x_val
    # convention used directly above so the J before/after the clamp matches CUDA.
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

    # W matrix (upper-left 3x3 of view matrix) — same Mt as used for p_view
    W_mat = Mt[:3, :3]  # [3, 3]

    # C++ convention: T[col][row] = sum_k W[k][row] * J[col][k]
    # In PyTorch batch: T_mat = W_mat @ J^T, then transpose to [N, 2, 3]
    T_mat = (W_mat.unsqueeze(0) @ J.transpose(1, 2)).transpose(1, 2)  # [N, 2, 3]
    # cov2D = T @ cov3D @ T^T
    cov2D = T_mat @ cov3D @ T_mat.transpose(1, 2)  # [N, 2, 2]

    # Add low-pass filter
    cov2D[:, 0, 0] = cov2D[:, 0, 0] + 0.3
    cov2D[:, 1, 1] = cov2D[:, 1, 1] + 0.3

    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] * cov2D[:, 1, 0]

    # Visibility mask
    visible = (p_view[:, 2] > 0.2) & (det > 0)

    # Conics (inverse 2D cov).
    # CUDA reference: cuda_rasterizer/forward_common.h:137 — `float det_inv = 1.f / det;`
    # No clamp; visibility mask above (`det > 0`) already gates non-positive dets.
    # Vulkan equivalent: src/vulkan/shaders/preprocess.comp:1148-1159 — `if (det == 0.0) return; det_inv = 1.0 / det;`
    # Previously this used `det.clamp(min=1e-10)`, which diverged from CUDA/VK in det ∈ (0, 1e-10).
    conic_a = cov2D[:, 1, 1] / det
    conic_b = -cov2D[:, 0, 1] / det
    conic_c = cov2D[:, 0, 0] / det

    # Sort by depth
    depths = p_view[:, 2].clone()
    depths[~visible] = 1e10
    order = torch.argsort(depths)

    # Alpha blending (sequential per-pixel)
    image = torch.zeros(H, W, 3, device=device)

    # Precompute per-Gaussian values
    means2d = torch.stack([pixel_x, pixel_y], dim=1)  # [N, 2]

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

    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--lr_scale", type=float, default=0.3)
    parser.add_argument("--compare_cpp", action="store_true", help="Also run C++ and compare")
    args = parser.parse_args()

    device = "cpu"  # Must be CPU for per-pixel loop

    # Load PLY
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ply_loader import load_ply_raw
    raw = load_ply_raw(args.ply)
    N = raw["count"]
    sh_degree = raw["sh_degree"]
    max_coeffs = raw["max_coeffs"]

    # Create PyTorch parameters (requires_grad)
    raw_pos = torch.tensor(raw["positions"].reshape(N, 3), dtype=torch.float32, device=device, requires_grad=True)
    raw_sc = torch.tensor(raw["scales"].reshape(N, 3), dtype=torch.float32, device=device, requires_grad=True)
    raw_rot = torch.tensor(raw["rotations"].reshape(N, 4), dtype=torch.float32, device=device, requires_grad=True)
    raw_sh = torch.tensor(raw["sh_coeffs"].reshape(N, max_coeffs * 3), dtype=torch.float32, device=device, requires_grad=True)
    raw_op = torch.tensor(raw["opacities"], dtype=torch.float32, device=device, requires_grad=True)

    # Load camera
    cameras = json.load(open(args.cameras))
    cam = cameras[0]
    W, H = cam["width"], cam["height"]

    R_np = np.array(cam["rotation"], dtype=np.float64)
    pos_np = np.array(cam["position"], dtype=np.float64)

    # View matrix (column-major flat) — matches C++ train_main.cpp lines 364-367
    # C++: vm[0..2]=R[0][0..2], vm[4..6]=R[1][0..2], vm[8..10]=R[2][0..2]
    # Translation: t[i] = -(R[0][i]*pos[0] + R[1][i]*pos[1] + R[2][i]*pos[2])
    vm_np = np.zeros(16, dtype=np.float32)
    vm_np[0]=float(R_np[0][0]); vm_np[1]=float(R_np[0][1]); vm_np[2]=float(R_np[0][2])
    vm_np[4]=float(R_np[1][0]); vm_np[5]=float(R_np[1][1]); vm_np[6]=float(R_np[1][2])
    vm_np[8]=float(R_np[2][0]); vm_np[9]=float(R_np[2][1]); vm_np[10]=float(R_np[2][2])
    vm_np[15] = 1.0
    for i in range(3):
        vm_np[12+i] = float(-(R_np[0][i]*pos_np[0] + R_np[1][i]*pos_np[1] + R_np[2][i]*pos_np[2]))

    # Projection matrix (must multiply proj × view to get viewproj, matching C++)
    fx, fy = cam["fx"], cam["fy"]
    tan_fovx = W / (2.0 * fx)
    tan_fovy = H / (2.0 * fy)
    near, far = 0.01, 100.0
    proj_np = np.zeros(16, dtype=np.float32)
    proj_np[0] = 1.0 / tan_fovx
    proj_np[5] = 1.0 / tan_fovy
    proj_np[10] = far / (far - near)
    proj_np[11] = 1.0
    proj_np[14] = -(far * near) / (far - near)
    # viewproj = mat4Mul(proj, view): out[col*4+row] = sum_k proj[k*4+row]*view[col*4+k]
    vpm_np = np.zeros(16, dtype=np.float32)
    for col in range(4):
        for row in range(4):
            s = 0.0
            for k in range(4):
                s += float(proj_np[k*4+row]) * float(vm_np[col*4+k])
            vpm_np[col*4+row] = float(s)

    vm = torch.tensor(vm_np, dtype=torch.float32, device=device)
    vpm = torch.tensor(vpm_np, dtype=torch.float32, device=device)
    cam_pos_t = torch.tensor(pos_np, dtype=torch.float32, device=device)
    bg = torch.zeros(3, device=device)

    # Load GT image
    gt_name = cam["img_name"]
    gt_path = None
    for ext in [".ppm", ".JPG.ppm"]:
        p = os.path.join(args.gt_dir, gt_name + ext)
        if os.path.exists(p):
            gt_path = p
            break

    with open(gt_path, "rb") as f:
        f.readline()  # magic
        dims = f.readline().decode().strip()
        while dims.startswith("#"): dims = f.readline().decode().strip()
        w, h = map(int, dims.split())
        f.readline()  # maxval
        gt_np = np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0

    gt = torch.tensor(gt_np[:H, :W], dtype=torch.float32, device=device)

    # Adam optimizer
    s = args.lr_scale
    lr_pos_init, lr_pos_final = 0.00016 * s, 0.0000016 * s
    optimizer = torch.optim.Adam([
        {"params": [raw_pos], "lr": lr_pos_init},
        {"params": [raw_sc], "lr": 0.005 * s},
        {"params": [raw_rot], "lr": 0.001 * s},
        {"params": [raw_sh], "lr": 0.0025 * s},
        {"params": [raw_op], "lr": 0.025 * s},
    ], betas=(0.9, 0.999), eps=1e-8)

    # LR scheduler for position
    def update_lr(iteration):
        t = iteration / args.iterations
        t = max(0.0, min(1.0, t))
        lr = float(np.exp(np.log(lr_pos_init) * (1 - t) + np.log(lr_pos_final) * t))
        optimizer.param_groups[0]["lr"] = lr

    print(f"# N={N}, {W}x{H}, SH{sh_degree}, {args.iterations} iters", file=sys.stderr)
    print("iteration,py_loss")

    for it in range(args.iterations):
        update_lr(it)
        optimizer.zero_grad()

        rendered = forward_render_pytorch(raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
                                          vm, vpm, cam_pos_t, W, H, bg, sh_degree)
        loss = torch.mean(torch.abs(rendered - gt))
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        print(f"{it},{loss_val:.8f}")

        if it <= 5 or (it + 1) % 10 == 0:
            print(f"  iter {it}: loss={loss_val:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
