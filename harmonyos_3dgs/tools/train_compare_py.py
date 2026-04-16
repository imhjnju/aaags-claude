#!/usr/bin/env python3
"""Method 3 verification: 100-step training comparison Python vs C++.
Uses the exact column-major forward renderer from verify_gradients.py.
Loads initial params and GT image from C++ binary dumps to ensure exact match.
Backward pass via finite-difference.
"""

import numpy as np
import sys
import os

F = np.float32
SH_C0 = F(0.28209479177387814)

def sigmoid_f32(x):
    """Sigmoid in float32, matching C++."""
    x = np.float32(x)
    return F(1.0) / (F(1.0) + np.float32(np.exp(float(-x))))

def sigmoid_arr(x):
    return np.array([sigmoid_f32(v) for v in x.flat], dtype=F).reshape(x.shape)

# ---- Exact C++ function ports (column-major [col][row]), all float32 ----

def transformPoint4x3(p, m):
    """All float32 ops."""
    return np.array([
        float(m[0])*float(p[0]) + float(m[4])*float(p[1]) + float(m[8])*float(p[2])  + float(m[12]),
        float(m[1])*float(p[0]) + float(m[5])*float(p[1]) + float(m[9])*float(p[2])  + float(m[13]),
        float(m[2])*float(p[0]) + float(m[6])*float(p[1]) + float(m[10])*float(p[2]) + float(m[14]),
    ], dtype=F)

def transformPoint4x4(p, m):
    return np.array([
        float(m[0])*float(p[0]) + float(m[4])*float(p[1]) + float(m[8])*float(p[2])  + float(m[12]),
        float(m[1])*float(p[0]) + float(m[5])*float(p[1]) + float(m[9])*float(p[2])  + float(m[13]),
        float(m[2])*float(p[0]) + float(m[6])*float(p[1]) + float(m[10])*float(p[2]) + float(m[14]),
        float(m[3])*float(p[0]) + float(m[7])*float(p[1]) + float(m[11])*float(p[2]) + float(m[15]),
    ], dtype=F)

def ndc2pix(v, S):
    return F(((float(v) + 1.0) * S - 1.0) * 0.5)

def computeCov3D(scale, mod, rot):
    sx = F(float(mod) * float(scale[0]))
    sy = F(float(mod) * float(scale[1]))
    sz = F(float(mod) * float(scale[2]))
    r, x, y, z = float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])
    R = np.zeros((3,3), dtype=F)
    R[0][0] = F(1 - 2*(y*y + z*z)); R[0][1] = F(2*(x*y + r*z)); R[0][2] = F(2*(x*z - r*y))
    R[1][0] = F(2*(x*y - r*z)); R[1][1] = F(1 - 2*(x*x + z*z)); R[1][2] = F(2*(y*z + r*x))
    R[2][0] = F(2*(x*z + r*y)); R[2][1] = F(2*(y*z - r*x)); R[2][2] = F(1 - 2*(x*x + y*y))
    M = np.zeros((3,3), dtype=F)
    ss = [sx, sy, sz]
    for i in range(3):
        for j in range(3):
            M[i][j] = F(float(ss[i]) * float(R[i][j]))
    cov3D = np.zeros(6, dtype=F)
    cov3D[0] = F(float(M[0][0])*float(M[0][0]) + float(M[1][0])*float(M[1][0]) + float(M[2][0])*float(M[2][0]))
    cov3D[1] = F(float(M[0][0])*float(M[0][1]) + float(M[1][0])*float(M[1][1]) + float(M[2][0])*float(M[2][1]))
    cov3D[2] = F(float(M[0][0])*float(M[0][2]) + float(M[1][0])*float(M[1][2]) + float(M[2][0])*float(M[2][2]))
    cov3D[3] = F(float(M[0][1])*float(M[0][1]) + float(M[1][1])*float(M[1][1]) + float(M[2][1])*float(M[2][1]))
    cov3D[4] = F(float(M[0][1])*float(M[0][2]) + float(M[1][1])*float(M[1][2]) + float(M[2][1])*float(M[2][2]))
    cov3D[5] = F(float(M[0][2])*float(M[0][2]) + float(M[1][2])*float(M[1][2]) + float(M[2][2])*float(M[2][2]))
    return cov3D

def computeCov2D(mean3d, cov3D, vm, focal_x, focal_y, tan_fovx, tan_fovy):
    t = transformPoint4x3(mean3d, vm)
    limx = 1.3 * float(tan_fovx)
    limy = 1.3 * float(tan_fovy)
    txtz = float(t[0]) / float(t[2])
    tytz = float(t[1]) / float(t[2])
    t[0] = F(min(limx, max(-limx, txtz)) * float(t[2]))
    t[1] = F(min(limy, max(-limy, tytz)) * float(t[2]))
    J = np.zeros((3,3), dtype=F)
    J[0][0] = F(float(focal_x) / float(t[2]))
    J[0][2] = F(-(float(focal_x) * float(t[0])) / (float(t[2]) * float(t[2])))
    J[1][1] = F(float(focal_y) / float(t[2]))
    J[1][2] = F(-(float(focal_y) * float(t[1])) / (float(t[2]) * float(t[2])))
    W_mat = np.zeros((3,3), dtype=F)
    W_mat[0][0] = vm[0]; W_mat[0][1] = vm[4]; W_mat[0][2] = vm[8]
    W_mat[1][0] = vm[1]; W_mat[1][1] = vm[5]; W_mat[1][2] = vm[9]
    W_mat[2][0] = vm[2]; W_mat[2][1] = vm[6]; W_mat[2][2] = vm[10]
    T = np.zeros((3,3), dtype=F)
    for col in range(3):
        for row in range(3):
            s = 0.0
            for k in range(3):
                s += float(W_mat[k][row]) * float(J[col][k])
            T[col][row] = F(s)
    Vrk = np.array([
        [cov3D[0], cov3D[1], cov3D[2]],
        [cov3D[1], cov3D[3], cov3D[4]],
        [cov3D[2], cov3D[4], cov3D[5]]
    ], dtype=F)
    tmp = np.zeros((3,3), dtype=F)
    for col in range(3):
        for row in range(3):
            s = 0.0
            for k in range(3):
                s += float(Vrk[row][k]) * float(T[col][k])
            tmp[col][row] = F(s)
    result = np.zeros((3,3), dtype=F)
    for col in range(3):
        for row in range(3):
            s = 0.0
            for k in range(3):
                s += float(T[row][k]) * float(tmp[col][k])
            result[col][row] = F(s)
    cov2D = np.array([result[0][0], result[1][0], result[1][1]], dtype=F)
    return cov2D, t

def computeSH_deg0(sh_dc, pos, cam_pos):
    result = np.array([SH_C0 * F(v) + F(0.5) for v in sh_dc], dtype=F)
    return np.maximum(F(0.0), result)

def forward_render(raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
                    vm, vpm, cam_pos, focal_x, focal_y,
                    tan_fovx, tan_fovy, W, H, bg):
    """Full forward matching C++ pipeline. All float32."""
    N = len(raw_pos)
    pos = raw_pos.astype(F).copy()
    scales = np.array([F(np.exp(float(v))) for v in raw_sc.flat], dtype=F).reshape(raw_sc.shape)
    rots = raw_rot.astype(F).copy()
    for i in range(N):
        ln = float(np.sqrt(sum(float(v)**2 for v in rots[i])))
        if ln < 1e-12:
            ln = 1e-12
        rots[i] = np.array([F(float(v)/ln) for v in rots[i]], dtype=F)
    opacities = np.array([sigmoid_f32(v) for v in raw_op.flat], dtype=F)

    vm_f = np.array(vm, dtype=F)
    vpm_f = np.array(vpm, dtype=F)
    cam_pos_f = np.array(cam_pos, dtype=F)
    focal_x_f = F(focal_x)
    focal_y_f = F(focal_y)
    tan_fovx_f = F(tan_fovx)
    tan_fovy_f = F(tan_fovy)

    means2d = np.zeros((N, 2), dtype=F)
    conics = np.zeros((N, 3), dtype=F)
    rgb = np.zeros((N, 3), dtype=F)
    op2d = np.zeros(N, dtype=F)
    depths = np.zeros(N, dtype=F)
    visible = np.zeros(N, dtype=bool)

    for i in range(N):
        p_view = transformPoint4x3(pos[i], vm_f)
        if float(p_view[2]) <= 0.2:
            continue
        depths[i] = p_view[2]
        p_hom = transformPoint4x4(pos[i], vpm_f)
        if abs(float(p_hom[3])) < 1e-7:
            continue
        p_ndc = np.array([F(float(p_hom[j])/float(p_hom[3])) for j in range(3)], dtype=F)
        means2d[i] = [ndc2pix(p_ndc[0], W), ndc2pix(p_ndc[1], H)]
        cov3d = computeCov3D(scales[i], F(1.0), rots[i])
        cov2d, _ = computeCov2D(pos[i], cov3d, vm_f, focal_x_f, focal_y_f, tan_fovx_f, tan_fovy_f)
        a = F(float(cov2d[0]) + 0.3)
        b = cov2d[1]
        c = F(float(cov2d[2]) + 0.3)
        det = F(float(a) * float(c) - float(b) * float(b))
        if float(det) <= 0:
            continue
        conics[i] = [F(float(c)/float(det)), F(-float(b)/float(det)), F(float(a)/float(det))]
        rgb[i] = computeSH_deg0(raw_sh.astype(F)[i, :3], pos[i], cam_pos_f)
        op2d[i] = opacities[i]
        visible[i] = True

    order = np.argsort(depths)
    image = np.zeros((H, W, 3), dtype=F)
    bg_f = np.array(bg, dtype=F)
    for py in range(H):
        for px in range(W):
            T_val = 1.0
            C = np.zeros(3, dtype=np.float64)
            for i in order:
                if not visible[i]:
                    continue
                dx = float(means2d[i, 0]) - float(px)
                dy = float(means2d[i, 1]) - float(py)
                ca, cb, cc = float(conics[i, 0]), float(conics[i, 1]), float(conics[i, 2])
                power = -0.5 * (ca*dx*dx + cc*dy*dy) - cb*dx*dy
                if power > 0:
                    continue
                alpha = min(0.99, float(op2d[i]) * np.exp(power))
                if alpha < 1.0/255.0:
                    continue
                for ch in range(3):
                    C[ch] += float(rgb[i, ch]) * alpha * T_val
                T_val *= (1.0 - alpha)
                if T_val < 0.0001:
                    break
            for ch in range(3):
                C[ch] += T_val * float(bg_f[ch])
            image[py, px] = np.array([F(v) for v in C], dtype=F)
    return image

def l1_loss(rendered, gt):
    """L1 loss matching C++: sum(|diff|) / n."""
    r = rendered.flatten().astype(F)
    g = gt.flatten().astype(F)
    n = len(r)
    inv_n = F(1.0 / float(n))
    s = F(0.0)
    for i in range(n):
        diff = float(r[i]) - float(g[i])
        s = F(float(s) + abs(diff))
    return float(F(float(s) * float(inv_n)))

def l1_loss_fast(rendered, gt):
    """Faster L1 loss (close to C++ but not bit-exact due to summation order)."""
    diff = rendered.flatten().astype(F) - gt.flatten().astype(F)
    n = len(diff)
    return float(np.sum(np.abs(diff)) / F(n))

def compute_fd_gradients(raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
                          vm, vpm, cam_pos, focal_x, focal_y,
                          tan_fovx, tan_fovy, W, H, bg, gt_image, eps=1e-4):
    """Compute gradients via finite difference for all raw parameters."""
    params_dict = {
        'raw_positions': raw_pos,
        'raw_scales': raw_sc,
        'raw_rotations': raw_rot,
        'raw_sh_coeffs': raw_sh,
        'raw_opacities': raw_op,
    }
    grads = {}
    for name, arr in params_dict.items():
        g = np.zeros(arr.size, dtype=np.float64)
        for idx in range(arr.size):
            old = float(arr.flat[idx])
            arr.flat[idx] = F(old + eps)
            rp = forward_render(raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
                         vm, vpm, cam_pos, focal_x, focal_y, tan_fovx, tan_fovy, W, H, bg)
            lp = l1_loss_fast(rp, gt_image)
            arr.flat[idx] = F(old - eps)
            rm = forward_render(raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
                         vm, vpm, cam_pos, focal_x, focal_y, tan_fovx, tan_fovy, W, H, bg)
            lm = l1_loss_fast(rm, gt_image)
            arr.flat[idx] = F(old)
            g[idx] = (lp - lm) / (2 * eps)
        grads[name] = g
    return grads


def main():
    N, W, H = 3, 16, 16
    bg = [0.0, 0.0, 0.0]
    num_iters = 100
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load exact float32 values from C++ binary dumps
    raw_pos = np.fromfile(os.path.join(base_dir, 'train/compare_raw_pos.bin'), dtype=F).reshape(N, 3)
    raw_sc  = np.fromfile(os.path.join(base_dir, 'train/compare_raw_sc.bin'), dtype=F).reshape(N, 3)
    raw_rot = np.fromfile(os.path.join(base_dir, 'train/compare_raw_rot.bin'), dtype=F).reshape(N, 4)
    raw_sh  = np.fromfile(os.path.join(base_dir, 'train/compare_raw_sh.bin'), dtype=F).reshape(N, 3)
    raw_op  = np.fromfile(os.path.join(base_dir, 'train/compare_raw_op.bin'), dtype=F)

    # Load camera params from C++ dump
    cam_data = np.fromfile(os.path.join(base_dir, 'train/compare_cam.bin'), dtype=F)
    vm = cam_data[0:16]
    vpm = cam_data[16:32]
    cam_pos = cam_data[32:35]
    tan_fovx = float(cam_data[35])
    tan_fovy = float(cam_data[36])
    focal_x = W / (2.0 * tan_fovx)
    focal_y = H / (2.0 * tan_fovy)

    # Load GT image from C++ (exact same pixels)
    gt_image = np.fromfile(os.path.join(base_dir, 'train/compare_gt_image.bin'), dtype=F).reshape(H, W, 3)

    # Verify initial forward pass matches
    rendered0 = forward_render(raw_pos.copy(), raw_sc.copy(), raw_rot.copy(),
                               raw_sh.copy(), raw_op.copy(),
                               vm, vpm, cam_pos, focal_x, focal_y,
                               tan_fovx, tan_fovy, W, H, bg)
    loss0_py = l1_loss_fast(rendered0, gt_image)
    print(f"# Python initial loss: {loss0_py:.8f}", file=sys.stderr)

    # Learning rates (float32 to match C++)
    lr_position = 0.001
    lr_feature  = 0.01
    lr_opacity  = 0.01
    lr_scaling  = 0.005
    lr_rotation = 0.001

    losses = []
    print("iteration,loss")
    for it in range(num_iters):
        rendered = forward_render(raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
                                  vm, vpm, cam_pos, focal_x, focal_y,
                                  tan_fovx, tan_fovy, W, H, bg)
        loss = l1_loss_fast(rendered, gt_image)
        losses.append(loss)
        print(f"{it},{loss:.8f}")

        # Finite-diff gradients
        grads = compute_fd_gradients(raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
                                      vm, vpm, cam_pos, focal_x, focal_y,
                                      tan_fovx, tan_fovy, W, H, bg, gt_image)

        # SGD update: param -= lr * grad (in float32 to match C++)
        for idx in range(raw_pos.size):
            raw_pos.flat[idx] = F(float(raw_pos.flat[idx]) - lr_position * grads['raw_positions'][idx])
        for idx in range(raw_sc.size):
            raw_sc.flat[idx] = F(float(raw_sc.flat[idx]) - lr_scaling * grads['raw_scales'][idx])
        for idx in range(raw_rot.size):
            raw_rot.flat[idx] = F(float(raw_rot.flat[idx]) - lr_rotation * grads['raw_rotations'][idx])
        for idx in range(raw_sh.size):
            raw_sh.flat[idx] = F(float(raw_sh.flat[idx]) - lr_feature * grads['raw_sh_coeffs'][idx])
        for idx in range(raw_op.size):
            raw_op.flat[idx] = F(float(raw_op.flat[idx]) - lr_opacity * grads['raw_opacities'][idx])

    return losses


def compare(cpp_csv, py_csv=None, py_losses=None):
    """Compare C++ and Python loss curves."""
    cpp_losses = []
    with open(cpp_csv) as f:
        for line in f:
            line = line.strip()
            if line.startswith('iteration') or not line:
                continue
            parts = line.split(',')
            cpp_losses.append(float(parts[1]))

    if py_losses is None:
        py_losses = []
        with open(py_csv) as f:
            for line in f:
                line = line.strip()
                if line.startswith('iteration') or not line:
                    continue
                parts = line.split(',')
                py_losses.append(float(parts[1]))

    print("\n=== Loss Curve Comparison (Python vs C++) ===", file=sys.stderr)
    rel_errors = []
    for i in range(min(len(py_losses), len(cpp_losses))):
        denom = max(abs(py_losses[i]), 1e-8)
        rel_err = abs(py_losses[i] - cpp_losses[i]) / denom
        rel_errors.append(rel_err)
        if i < 5 or i >= len(cpp_losses) - 5 or i % 20 == 0:
            print(f"  iter {i:3d}: py={py_losses[i]:.8f}  cpp={cpp_losses[i]:.8f}  rel_err={rel_err:.6f}", file=sys.stderr)

    max_err = max(rel_errors)
    avg_err = sum(rel_errors) / len(rel_errors)
    print(f"\n  Max relative error:  {max_err:.6f}", file=sys.stderr)
    print(f"  Mean relative error: {avg_err:.6f}", file=sys.stderr)
    print(f"  First loss: py={py_losses[0]:.8f}  cpp={cpp_losses[0]:.8f}", file=sys.stderr)
    print(f"  Last loss:  py={py_losses[-1]:.8f}  cpp={cpp_losses[-1]:.8f}", file=sys.stderr)
    print(f"\n  {'PASS' if max_err < 0.01 else 'FAIL'} (threshold: 1%)", file=sys.stderr)
    return max_err


if __name__ == '__main__':
    if '--compare' in sys.argv:
        idx = sys.argv.index('--compare')
        cpp_csv = sys.argv[idx + 1]
        py_csv = sys.argv[idx + 2] if len(sys.argv) > idx + 2 else None
        compare(cpp_csv, py_csv)
    else:
        losses = main()
        # Auto-compare if C++ CSV exists
        cpp_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'train', 'compare_cpp.csv')
        if os.path.exists(cpp_csv):
            compare(cpp_csv, py_losses=losses)
