#!/usr/bin/env python3
"""Verify C++ forward render matches Python reference pixel-by-pixel.

Uses the exact same float32 Python renderer from train_compare_py.py
(which was validated to match C++ conventions: column-major, same SH, etc.)

Usage:
    # Build first
    cmake -B build -S harmonyos_3dgs -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc)

    # Run verification
    python harmonyos_3dgs/tools/verify_forward_render.py

Expected output: per-pixel MSE < 1e-12, PSNR > 100 dB (bit-exact)
"""

import numpy as np
import subprocess
import os
import sys
import json

F = np.float32
SH_C0 = F(0.28209479177387814)


def sigmoid_f32(x):
    x = np.float32(x)
    return F(1.0) / (F(1.0) + np.float32(np.exp(float(-x))))


def transformPoint4x3(p, m):
    return np.array([
        float(m[0])*float(p[0]) + float(m[4])*float(p[1]) + float(m[8])*float(p[2]) + float(m[12]),
        float(m[1])*float(p[0]) + float(m[5])*float(p[1]) + float(m[9])*float(p[2]) + float(m[13]),
        float(m[2])*float(p[0]) + float(m[6])*float(p[1]) + float(m[10])*float(p[2]) + float(m[14]),
    ], dtype=F)


def transformPoint4x4(p, m):
    return np.array([
        float(m[0])*float(p[0]) + float(m[4])*float(p[1]) + float(m[8])*float(p[2]) + float(m[12]),
        float(m[1])*float(p[0]) + float(m[5])*float(p[1]) + float(m[9])*float(p[2]) + float(m[13]),
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
    R = np.zeros((3, 3), dtype=F)
    R[0][0] = F(1 - 2*(y*y + z*z)); R[0][1] = F(2*(x*y + r*z)); R[0][2] = F(2*(x*z - r*y))
    R[1][0] = F(2*(x*y - r*z)); R[1][1] = F(1 - 2*(x*x + z*z)); R[1][2] = F(2*(y*z + r*x))
    R[2][0] = F(2*(x*z + r*y)); R[2][1] = F(2*(y*z - r*x)); R[2][2] = F(1 - 2*(x*x + y*y))
    M = np.zeros((3, 3), dtype=F)
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
    J = np.zeros((3, 3), dtype=F)
    J[0][0] = F(float(focal_x) / float(t[2]))
    J[0][2] = F(-(float(focal_x) * float(t[0])) / (float(t[2]) * float(t[2])))
    J[1][1] = F(float(focal_y) / float(t[2]))
    J[1][2] = F(-(float(focal_y) * float(t[1])) / (float(t[2]) * float(t[2])))
    W_mat = np.zeros((3, 3), dtype=F)
    W_mat[0][0] = vm[0]; W_mat[0][1] = vm[4]; W_mat[0][2] = vm[8]
    W_mat[1][0] = vm[1]; W_mat[1][1] = vm[5]; W_mat[1][2] = vm[9]
    W_mat[2][0] = vm[2]; W_mat[2][1] = vm[6]; W_mat[2][2] = vm[10]
    T = np.zeros((3, 3), dtype=F)
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
    tmp = np.zeros((3, 3), dtype=F)
    for col in range(3):
        for row in range(3):
            s = 0.0
            for k in range(3):
                s += float(Vrk[row][k]) * float(T[col][k])
            tmp[col][row] = F(s)
    result = np.zeros((3, 3), dtype=F)
    for col in range(3):
        for row in range(3):
            s = 0.0
            for k in range(3):
                s += float(T[row][k]) * float(tmp[col][k])
            result[col][row] = F(s)
    cov2D = np.array([result[0][0], result[1][0], result[1][1]], dtype=F)
    return cov2D, t


def computeSH_deg0(sh_dc, pos, cam_pos):
    """SH degree 0 only. Matches C++ computeColorFromSH with degree=0."""
    result = np.array([SH_C0 * F(v) + F(0.5) for v in sh_dc], dtype=F)
    return np.maximum(F(0.0), result)


def forward_render(raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
                   vm, vpm, cam_pos, focal_x, focal_y,
                   tan_fovx, tan_fovy, W, H, bg):
    """Full forward matching C++ pipeline. All float32."""
    N = len(raw_pos)

    # Activate raw params (matching C++ RawGaussianParams::activate)
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
        p_w = F(1.0 / (float(p_hom[3]) + 1e-7))
        p_ndc = np.array([F(float(p_hom[j])*float(p_w)) for j in range(3)], dtype=F)
        means2d[i] = [ndc2pix(p_ndc[0], W), ndc2pix(p_ndc[1], H)]
        cov3d = computeCov3D(scales[i], F(1.0), rots[i])
        cov2d, _ = computeCov2D(pos[i], cov3d, vm_f, F(focal_x), F(focal_y), F(tan_fovx), F(tan_fovy))

        # det before +0.3 filter (for antialiasing h_conv_scaling, not used here)
        a = F(float(cov2d[0]) + 0.3)
        b = cov2d[1]
        c = F(float(cov2d[2]) + 0.3)
        det = F(float(a) * float(c) - float(b) * float(b))
        if float(det) == 0.0:
            continue
        det_inv = F(1.0 / float(det))
        conics[i] = [F(float(c)*float(det_inv)), F(-float(b)*float(det_inv)), F(float(a)*float(det_inv))]

        # Radius (eigenvalue-based)
        mid = F(0.5 * (float(a) + float(c)))
        disc = F(float(mid)*float(mid) - float(det))
        disc = max(F(0.1), disc)
        lambda1 = F(float(mid) + np.sqrt(float(disc)))
        lambda2 = F(float(mid) - np.sqrt(float(disc)))
        my_radius = int(np.ceil(3.0 * np.sqrt(float(max(lambda1, lambda2)))))

        rgb[i] = computeSH_deg0(raw_sh.astype(F)[i, :3], pos[i], cam_pos_f)
        op2d[i] = opacities[i]  # no antialiasing h_conv_scaling
        visible[i] = True

    # Sort by depth (matching C++ sorter)
    order = np.argsort(depths)

    image = np.zeros((H, W, 3), dtype=F)
    bg_f = np.array(bg, dtype=F)
    for py in range(H):
        for px in range(W):
            T_val = F(1.0)
            C = np.zeros(3, dtype=np.float64)  # accumulate in float64 for precision
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

                test_T = float(T_val) * (1.0 - alpha)
                if test_T < 0.0001:
                    break

                for ch in range(3):
                    C[ch] += float(rgb[i, ch]) * alpha * float(T_val)

                T_val = F(test_T)

            for ch in range(3):
                C[ch] += float(T_val) * float(bg_f[ch])
            image[py, px] = np.array([F(v) for v in C], dtype=F)

    return image


def write_bin(path, arr):
    arr.astype(np.float32).tofile(path)


def read_bin(path, count):
    return np.fromfile(path, dtype=np.float32, count=count)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, 'train', 'verify_forward')
    os.makedirs(out_dir, exist_ok=True)

    # Scene setup: 3 Gaussians at known positions, SH degree 0
    N, W, H = 3, 16, 16
    bg = [0.0, 0.0, 0.0]

    raw_pos = np.array([[0, 0, 5], [0.5, -0.3, 6], [-0.4, 0.2, 4.5]], dtype=np.float64)
    raw_sc = np.log(np.array([[0.3, 0.3, 0.3], [0.2, 0.4, 0.2], [0.5, 0.2, 0.3]], dtype=np.float64))
    raw_rot = np.array([[1, 0, 0, 0], [0.9, 0.1, 0.2, 0], [0.8, 0, 0.3, 0.1]], dtype=np.float64)
    raw_sh = np.array([[1.5, 0.8, 0.3], [0.5, 1.5, 0.8], [0.3, 0.5, 1.5]], dtype=np.float64)
    raw_op = np.array([2.0, 1.5, 1.0], dtype=np.float64)

    fov = 60.0
    tan_fov = float(np.tan(np.radians(fov/2)))
    focal_x = W / (2.0 * tan_fov)
    focal_y = H / (2.0 * tan_fov)
    cam_pos = np.array([0, 0, 0], dtype=np.float64)

    # Identity view matrix (column-major flat[16])
    vm = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    # Perspective projection (column-major)
    near, far = 0.01, 100.0
    r = tan_fov * near
    vpm = [0.0]*16
    vpm[0] = near / r
    vpm[5] = near / r
    vpm[10] = far/(far-near)
    vpm[11] = 1.0
    vpm[14] = -(far*near)/(far-near)

    # Python forward render
    print(f"Scene: {N} Gaussians, {W}x{H}, SH degree 0")
    print(f"Camera: identity view, fov={fov}, focal=({focal_x:.4f},{focal_y:.4f})")
    py_image = forward_render(raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
                              vm, vpm, cam_pos, focal_x, focal_y,
                              tan_fov, tan_fov, W, H, bg)
    print(f"Python rendered: sum={py_image.sum():.6f}, "
          f"min={py_image.min():.6f}, max={py_image.max():.6f}, "
          f"nonzero={np.count_nonzero(py_image)}/{py_image.size}")

    # Save inputs for C++
    write_bin(f'{out_dir}/raw_positions.bin', raw_pos.astype(F))
    write_bin(f'{out_dir}/raw_scales.bin', raw_sc.astype(F))
    write_bin(f'{out_dir}/raw_rotations.bin', raw_rot.astype(F))
    write_bin(f'{out_dir}/raw_sh_coeffs.bin', raw_sh.astype(F))
    write_bin(f'{out_dir}/raw_opacities.bin', raw_op.astype(F))
    write_bin(f'{out_dir}/view_matrix.bin', np.array(vm, dtype=F))
    write_bin(f'{out_dir}/viewproj_matrix.bin', np.array(vpm, dtype=F))
    write_bin(f'{out_dir}/py_rendered.bin', py_image.flatten().astype(F))

    # Use same image as GT (for loss computation, not important here)
    write_bin(f'{out_dir}/gt_image.bin', py_image.flatten().astype(F))

    json.dump({
        'N': N, 'W': W, 'H': H,
        'sh_degree': 0, 'max_coeffs': 1,
        'cam_pos': [0, 0, 0],
        'tan_fovx': tan_fov, 'tan_fovy': tan_fov,
        'bg_color': bg,
        'loss': 0.0,
    }, open(f'{out_dir}/meta.json', 'w'), indent=2)

    # Run C++
    cpp_exe = os.path.join(base_dir, 'build', 'gs3d_verify_grads')
    if not os.path.exists(cpp_exe):
        # Try parent directory
        cpp_exe = os.path.join(os.path.dirname(base_dir), 'build', 'gs3d_verify_grads')
    if not os.path.exists(cpp_exe):
        print(f"\nERROR: C++ executable not found. Build first:")
        print(f"  cmake -B build -S harmonyos_3dgs -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release")
        print(f"  cmake --build build -j$(nproc)")
        return False

    print(f"\nRunning C++ forward render...")
    result = subprocess.run([cpp_exe, out_dir], capture_output=True, text=True)
    print(result.stdout.rstrip())
    if result.returncode != 0:
        print(f"C++ FAILED: {result.stderr}")
        return False

    # Load C++ rendered image
    cpp_rendered_path = f'{out_dir}/cpp_rendered.bin'
    if not os.path.exists(cpp_rendered_path):
        print(f"ERROR: C++ rendered image not found at {cpp_rendered_path}")
        return False

    cpp_image = read_bin(cpp_rendered_path, H * W * 3).reshape(H, W, 3)
    print(f"C++ rendered:    sum={cpp_image.sum():.6f}, "
          f"min={cpp_image.min():.6f}, max={cpp_image.max():.6f}, "
          f"nonzero={np.count_nonzero(cpp_image)}/{cpp_image.size}")

    # Compare
    diff = np.abs(py_image.flatten().astype(np.float64) - cpp_image.flatten().astype(np.float64))
    mse = float(np.mean(diff**2))
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))
    nonzero_diffs = int(np.sum(diff > 0))

    if mse > 0:
        psnr = -10 * np.log10(mse)
    else:
        psnr = float('inf')

    print(f"\n{'='*60}")
    print(f"FORWARD RENDER COMPARISON: Python vs C++")
    print(f"{'='*60}")
    print(f"  Image size:     {W}x{H} ({H*W*3} elements)")
    print(f"  MSE:            {mse:.2e}")
    print(f"  PSNR:           {psnr:.1f} dB")
    print(f"  Max pixel err:  {max_err:.2e}")
    print(f"  Mean pixel err: {mean_err:.2e}")
    print(f"  Nonzero diffs:  {nonzero_diffs}/{H*W*3}")

    # Per-pixel detail for mismatches
    if nonzero_diffs > 0 and nonzero_diffs <= 20:
        print(f"\n  Mismatched pixels:")
        for idx in np.where(diff > 0)[0]:
            py_val = float(py_image.flat[idx])
            cpp_val = float(cpp_image.flat[idx])
            print(f"    [{idx}] py={py_val:.10f}  cpp={cpp_val:.10f}  diff={diff[idx]:.2e}")

    # Verdict
    threshold = 1e-6  # per-pixel max error threshold
    passed = max_err <= threshold
    print(f"\n  Threshold:      max_err <= {threshold}")
    print(f"  Result:         {'PASS' if passed else 'FAIL'}")
    print(f"{'='*60}")

    return passed


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
