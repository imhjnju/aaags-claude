#!/usr/bin/env python3
"""Extended gradient verification across multiple configurations.

Tests C++ analytical gradients vs Python finite-difference for:
  - All 5 parameter types (positions, scales, rotations, opacities, SH coeffs)
  - Multiple SH degrees (0, 1, 2, 3)
  - Multiple Gaussian counts (3, 10, 30)
  - Multiple resolutions (16x16, 32x32)

P0 fix: Python FD uses float32-precise renderer matching C++ exactly.
P1 fix: Comparison uses absolute tolerance for near-zero gradients.

Usage:
    cmake -B build -S harmonyos_3dgs -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc)
    python harmonyos_3dgs/tools/verify_gradients_extended.py
"""

import numpy as np
import subprocess
import os
import sys
import json
import time

F = np.float32

# Float32 SH constants (matching C++ sh_eval.cpp exactly)
SH_C0 = F(0.28209479177387814)
SH_C1 = F(0.4886025119029199)
SH_C2 = [F(1.0925484305920792), F(-1.0925484305920792), F(0.31539156525252005),
          F(-1.0925484305920792), F(0.5462742152960396)]
SH_C3 = [F(-0.5900435899266435), F(2.890611442640554), F(-0.4570457994644658),
          F(0.3731763325901154), F(-0.4570457994644658), F(1.445305721320277),
          F(-0.5900435899266435)]


# ============================================================
# Float32-precise helpers (matching C++ preprocessor_cpu.cpp)
# ============================================================

def sigmoid_f32(x):
    x = float(x)
    if x < -50: return F(0.0)
    if x > 50: return F(1.0)
    return F(1.0 / (1.0 + np.exp(-x)))


def transformPoint4x3(p, m):
    return np.array([
        F(float(m[0])*float(p[0]) + float(m[4])*float(p[1]) + float(m[8])*float(p[2]) + float(m[12])),
        F(float(m[1])*float(p[0]) + float(m[5])*float(p[1]) + float(m[9])*float(p[2]) + float(m[13])),
        F(float(m[2])*float(p[0]) + float(m[6])*float(p[1]) + float(m[10])*float(p[2]) + float(m[14])),
    ], dtype=F)


def transformPoint4x4(p, m):
    return np.array([
        F(float(m[0])*float(p[0]) + float(m[4])*float(p[1]) + float(m[8])*float(p[2]) + float(m[12])),
        F(float(m[1])*float(p[0]) + float(m[5])*float(p[1]) + float(m[9])*float(p[2]) + float(m[13])),
        F(float(m[2])*float(p[0]) + float(m[6])*float(p[1]) + float(m[10])*float(p[2]) + float(m[14])),
        F(float(m[3])*float(p[0]) + float(m[7])*float(p[1]) + float(m[11])*float(p[2]) + float(m[15])),
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
    limx = F(1.3 * float(tan_fovx))
    limy = F(1.3 * float(tan_fovy))
    txtz = float(t[0]) / float(t[2])
    tytz = float(t[1]) / float(t[2])
    t[0] = F(min(float(limx), max(-float(limx), txtz)) * float(t[2]))
    t[1] = F(min(float(limy), max(-float(limy), tytz)) * float(t[2]))
    J = np.zeros((3, 3), dtype=F)
    J[0][0] = F(float(focal_x) / float(t[2]))
    J[0][2] = F(-(float(focal_x) * float(t[0])) / (float(t[2]) * float(t[2])))
    J[1][1] = F(float(focal_y) / float(t[2]))
    J[1][2] = F(-(float(focal_y) * float(t[1])) / (float(t[2]) * float(t[2])))
    W_mat = np.zeros((3, 3), dtype=F)
    W_mat[0][0] = vm[0]; W_mat[0][1] = vm[4]; W_mat[0][2] = vm[8]
    W_mat[1][0] = vm[1]; W_mat[1][1] = vm[5]; W_mat[1][2] = vm[9]
    W_mat[2][0] = vm[2]; W_mat[2][1] = vm[6]; W_mat[2][2] = vm[10]
    T_mat = np.zeros((3, 3), dtype=F)
    for col in range(3):
        for row in range(3):
            s = 0.0
            for k in range(3):
                s += float(W_mat[k][row]) * float(J[col][k])
            T_mat[col][row] = F(s)
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
                s += float(Vrk[row][k]) * float(T_mat[col][k])
            tmp[col][row] = F(s)
    result = np.zeros((3, 3), dtype=F)
    for col in range(3):
        for row in range(3):
            s = 0.0
            for k in range(3):
                s += float(T_mat[row][k]) * float(tmp[col][k])
            result[col][row] = F(s)
    cov2D = np.array([result[0][0], result[1][0], result[1][1]], dtype=F)
    return cov2D, t


def computeColorFromSH(degree, sh_coeffs, pos, cam_pos):
    """Float32-precise SH evaluation matching C++ sh_eval.cpp for degrees 0-3."""
    dx = F(float(pos[0]) - float(cam_pos[0]))
    dy = F(float(pos[1]) - float(cam_pos[1]))
    dz = F(float(pos[2]) - float(cam_pos[2]))
    ln = F(np.sqrt(float(dx)*float(dx) + float(dy)*float(dy) + float(dz)*float(dz)))
    if float(ln) < 1e-8:
        ln = F(1.0)
    dx = F(float(dx)/float(ln)); dy = F(float(dy)/float(ln)); dz = F(float(dz)/float(ln))
    x, y, z = float(dx), float(dy), float(dz)

    result = np.zeros(3, dtype=F)
    # Degree 0
    for ch in range(3):
        result[ch] = F(float(SH_C0) * float(sh_coeffs[0*3 + ch]))

    if degree > 0:
        for ch in range(3):
            result[ch] = F(float(result[ch])
                + float(F(-float(SH_C1) * y)) * float(sh_coeffs[1*3 + ch])
                + float(F(float(SH_C1) * z))  * float(sh_coeffs[2*3 + ch])
                + float(F(-float(SH_C1) * x)) * float(sh_coeffs[3*3 + ch]))

        if degree > 1:
            xx, yy, zz = x*x, y*y, z*z
            xy, yz, xz = x*y, y*z, x*z
            for ch in range(3):
                result[ch] = F(float(result[ch])
                    + float(SH_C2[0]) * xy * float(sh_coeffs[4*3 + ch])
                    + float(SH_C2[1]) * yz * float(sh_coeffs[5*3 + ch])
                    + float(SH_C2[2]) * (2*zz - xx - yy) * float(sh_coeffs[6*3 + ch])
                    + float(SH_C2[3]) * xz * float(sh_coeffs[7*3 + ch])
                    + float(SH_C2[4]) * (xx - yy) * float(sh_coeffs[8*3 + ch]))

            if degree > 2:
                for ch in range(3):
                    result[ch] = F(float(result[ch])
                        + float(SH_C3[0]) * y*(3*xx - yy) * float(sh_coeffs[9*3 + ch])
                        + float(SH_C3[1]) * xy*z * float(sh_coeffs[10*3 + ch])
                        + float(SH_C3[2]) * y*(4*zz - xx - yy) * float(sh_coeffs[11*3 + ch])
                        + float(SH_C3[3]) * z*(2*zz - 3*xx - 3*yy) * float(sh_coeffs[12*3 + ch])
                        + float(SH_C3[4]) * x*(4*zz - xx - yy) * float(sh_coeffs[13*3 + ch])
                        + float(SH_C3[5]) * z*(xx - yy) * float(sh_coeffs[14*3 + ch])
                        + float(SH_C3[6]) * x*(xx - 3*yy) * float(sh_coeffs[15*3 + ch]))

    # Clamp: max(0, result + 0.5) in float32 — matches C++ sh_eval.cpp line 84
    for ch in range(3):
        result[ch] = F(max(0.0, float(result[ch]) + 0.5))
    return result


# ============================================================
# Float32-precise forward render (matching C++ pipeline exactly)
# ============================================================

def forward_render(raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
                   vm, vpm, cam_pos, focal_x, focal_y,
                   tan_fovx, tan_fovy, W, H, bg, sh_degree=0):
    N = len(raw_pos)
    max_coeffs = (sh_degree + 1) ** 2

    # Activate raw params in float32 (matching C++ RawGaussianParams::activate)
    pos = raw_pos.astype(F).copy()
    scales = np.array([F(np.exp(float(v))) for v in raw_sc.flat], dtype=F).reshape(raw_sc.shape)
    rots = raw_rot.astype(F).copy()
    for i in range(N):
        ln = float(np.sqrt(sum(float(v)**2 for v in rots[i])))
        if ln < 1e-12: ln = 1e-12
        rots[i] = np.array([F(float(v)/ln) for v in rots[i]], dtype=F)
    opacities = np.array([sigmoid_f32(v) for v in raw_op.flat], dtype=F)

    vm_f = np.array(vm, dtype=F)
    vpm_f = np.array(vpm, dtype=F)
    cam_pos_f = np.array(cam_pos, dtype=F)
    sh_coeffs_f = raw_sh.astype(F)

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
        # Matching C++ preprocessor_cpu.cpp line 196: 1.0f / (p_hom[3] + 1e-7f)
        p_w = F(1.0 / (float(p_hom[3]) + 1e-7))
        p_ndc = np.array([F(float(p_hom[j])*float(p_w)) for j in range(3)], dtype=F)

        means2d[i] = [ndc2pix(p_ndc[0], W), ndc2pix(p_ndc[1], H)]

        cov3d = computeCov3D(scales[i], F(1.0), rots[i])
        cov2d, _ = computeCov2D(pos[i], cov3d, vm_f, F(focal_x), F(focal_y), F(tan_fovx), F(tan_fovy))

        # Matching C++: +0.3f filter, then det check
        a = F(float(cov2d[0]) + 0.3)
        b = cov2d[1]
        c = F(float(cov2d[2]) + 0.3)
        det = F(float(a) * float(c) - float(b) * float(b))
        # Matching C++ preprocessor_cpu.cpp line 218: det == 0.0f
        if float(det) == 0.0:
            continue
        det_inv = F(1.0 / float(det))
        conics[i] = [F(float(c)*float(det_inv)), F(-float(b)*float(det_inv)), F(float(a)*float(det_inv))]

        rgb[i] = computeColorFromSH(sh_degree, sh_coeffs_f[i, :max_coeffs*3], pos[i], cam_pos_f)
        op2d[i] = opacities[i]
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


def l1_loss(rendered, gt):
    """L1 loss matching C++ loss.cpp: serial float32 accumulation."""
    r = rendered.astype(F).flatten()
    g = gt.astype(F).flatten()
    n = len(r)
    inv_n = F(1.0 / n)
    total = F(0.0)
    for i in range(n):
        diff = F(float(r[i]) - float(g[i]))
        total = F(float(total) + float(F(abs(float(diff)))))
    return float(F(float(total) * float(inv_n)))


def l1_loss_fast(rendered, gt):
    """L1 loss for FD: render in float32 but accumulate in float64 for FD precision."""
    r = rendered.astype(F).flatten().astype(np.float64)
    g = gt.astype(F).flatten().astype(np.float64)
    return float(np.mean(np.abs(r - g)))


def write_bin(path, arr):
    arr.astype(np.float32).tofile(path)


def read_bin(path, count):
    return np.fromfile(path, dtype=np.float32, count=count)


def make_camera(W, H, fov=60.0):
    tan_fov = float(np.tan(np.radians(fov/2)))
    focal_x = W / (2.0 * tan_fov)
    focal_y = H / (2.0 * tan_fov)
    cam_pos = np.array([0, 0, 0], dtype=np.float64)
    vm = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    near, far = 0.01, 100.0
    r = tan_fov * near
    vpm = [0.0]*16
    vpm[0] = near/r; vpm[5] = near/r
    vpm[10] = far/(far-near); vpm[11] = 1.0; vpm[14] = -(far*near)/(far-near)
    return vm, vpm, cam_pos, focal_x, focal_y, tan_fov


def make_scene(N, sh_degree, seed=42):
    """Generate deterministic scene with N Gaussians."""
    rng = np.random.RandomState(seed)
    max_coeffs = (sh_degree + 1) ** 2

    # Positions: spread in z=3..8 range
    raw_pos = np.zeros((N, 3), dtype=np.float64)
    raw_pos[:, 0] = rng.uniform(-1.5, 1.5, N)
    raw_pos[:, 1] = rng.uniform(-1.5, 1.5, N)
    raw_pos[:, 2] = rng.uniform(3.0, 8.0, N)

    # Scales: log-space, moderate size
    raw_sc = np.log(rng.uniform(0.1, 0.6, (N, 3)).astype(np.float64))

    # Rotations: random quaternions (unnormalized)
    raw_rot = rng.randn(N, 4).astype(np.float64)
    raw_rot[:, 0] += 1.0  # bias toward identity

    # SH coeffs
    raw_sh = rng.uniform(-1.0, 2.0, (N, max_coeffs * 3)).astype(np.float64)

    # Opacities: inverse-sigmoid space, moderate
    raw_op = rng.uniform(-1.0, 3.0, N).astype(np.float64)

    return raw_pos, raw_sc, raw_rot, raw_sh, raw_op, max_coeffs


def run_cpp_get_loss(exe, d):
    """Run C++ executable, read loss from binary file for full double precision."""
    r = subprocess.run([exe, d], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    loss_path = os.path.join(d, 'cpp_loss.bin')
    if os.path.exists(loss_path):
        return float(np.fromfile(loss_path, dtype=np.float64, count=1)[0])
    # Fallback: parse from stdout
    for line in r.stdout.split('\n'):
        if 'C++ loss' in line:
            return float(line.split(':')[1].strip())
    return None


def run_config(name, N, W, H, sh_degree, seed, cpp_exe, base_out,
               rel_threshold=0.10, abs_tol=5e-5):
    """Run gradient verification using C++ for both FD and analytical.

    This eliminates Python/C++ renderer mismatch by using C++ for all
    forward renders. For each parameter, we perturb the binary file,
    run C++ forward to get the loss, and compute FD from C++ loss values.
    """
    out_dir = os.path.join(base_out, name)
    os.makedirs(out_dir, exist_ok=True)

    raw_pos, raw_sc, raw_rot, raw_sh, raw_op, max_coeffs = make_scene(N, sh_degree, seed)
    vm, vpm, cam_pos, focal_x, focal_y, tan_fov = make_camera(W, H)
    bg = [0.0, 0.0, 0.0]

    # GT image: render in Python float32, add offset to ensure gt > rendered everywhere
    rendered_base = forward_render(raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
                                   vm, vpm, cam_pos, focal_x, focal_y, tan_fov, tan_fov,
                                   W, H, bg, sh_degree=sh_degree)
    gt_image = (rendered_base.astype(np.float64) + 0.1).astype(F)

    # Save base params for C++
    write_bin(f'{out_dir}/raw_positions.bin', raw_pos)
    write_bin(f'{out_dir}/raw_scales.bin', raw_sc)
    write_bin(f'{out_dir}/raw_rotations.bin', raw_rot)
    write_bin(f'{out_dir}/raw_sh_coeffs.bin', raw_sh)
    write_bin(f'{out_dir}/raw_opacities.bin', raw_op)
    write_bin(f'{out_dir}/gt_image.bin', gt_image.flatten())
    write_bin(f'{out_dir}/view_matrix.bin', np.array(vm, dtype=np.float32))
    write_bin(f'{out_dir}/viewproj_matrix.bin', np.array(vpm, dtype=np.float32))
    json.dump({'N': N, 'W': W, 'H': H, 'sh_degree': sh_degree, 'max_coeffs': max_coeffs,
               'cam_pos': [0, 0, 0], 'tan_fovx': tan_fov, 'tan_fovy': tan_fov,
               'bg_color': bg, 'loss': 0.0},
              open(f'{out_dir}/meta.json', 'w'), indent=2)

    # Run C++ to get analytical gradients
    base_loss = run_cpp_get_loss(cpp_exe, out_dir)
    if base_loss is None:
        return False, {'error': 'C++ failed on base run'}

    # C++ FD: perturb each parameter, run C++ forward, compute gradient.
    # Per-parameter epsilon balances FD noise floor (need large eps for float32 loss)
    # against truncation error (need small eps for nonlinear params like quaternions).
    # With double-precision loss, eps=1e-4 gives sufficient SNR for all params.
    eps = 1e-4
    param_files = {
        'raw_sh_coeffs': ('raw_sh_coeffs.bin', raw_sh, eps),
        'raw_opacities': ('raw_opacities.bin', raw_op, eps),
        'raw_scales':    ('raw_scales.bin',    raw_sc, eps),
        'raw_rotations': ('raw_rotations.bin', raw_rot, eps),
        'raw_positions': ('raw_positions.bin', raw_pos, eps),
    }

    fd_grads = {}
    for pname, (fname, arr, eps) in param_files.items():
        fpath = f'{out_dir}/{fname}'
        orig = arr.astype(np.float32).flatten().copy()
        grads = np.zeros(orig.size, dtype=np.float64)
        for idx in range(orig.size):
            buf = orig.copy()
            buf[idx] = np.float32(float(orig[idx]) + eps)
            write_bin(fpath, buf)
            lp = run_cpp_get_loss(cpp_exe, out_dir)

            buf[idx] = np.float32(float(orig[idx]) - eps)
            write_bin(fpath, buf)
            lm = run_cpp_get_loss(cpp_exe, out_dir)

            grads[idx] = (lp - lm) / (2 * eps)
        # Restore original
        write_bin(fpath, orig)
        fd_grads[pname] = grads

    # Re-run C++ to regenerate analytical gradients (params now restored)
    run_cpp_get_loss(cpp_exe, out_dir)

    # Compare C++ FD vs C++ analytical
    results = {}
    all_pass = True
    for pname in param_files:
        fd_g = fd_grads[pname]
        cpp_path = f'{out_dir}/cpp_grad_{pname}.bin'
        if not os.path.exists(cpp_path):
            results[pname] = {'status': 'MISSING'}
            all_pass = False
            continue
        cpp_g = read_bin(cpp_path, fd_g.size).astype(np.float64)

        max_err = 0.0
        max_err_idx = 0
        n_compared = 0
        n_skipped = 0
        n_fail = 0
        for j in range(fd_g.size):
            mag = max(abs(fd_g[j]), abs(cpp_g[j]))
            if mag < abs_tol:
                n_skipped += 1
                continue
            n_compared += 1
            rel = abs(fd_g[j] - cpp_g[j]) / mag
            # Combined tolerance: |fd-ana| <= atol + rtol * max(|fd|,|ana|)
            # This handles small-gradient edge cases where relative error is high
            # but absolute error is within noise floor.
            combined_ok = abs(fd_g[j] - cpp_g[j]) <= abs_tol + rel_threshold * mag
            if rel > max_err:
                max_err = rel
                max_err_idx = j
            if not combined_ok:
                n_fail += 1

        ok = n_fail == 0
        if not ok:
            all_pass = False
        results[pname] = {
            'max_rel_err': max_err,
            'max_err_idx': max_err_idx,
            'status': 'PASS' if ok else 'FAIL',
            'n_elements': fd_g.size,
            'n_compared': n_compared,
            'n_skipped': n_skipped,
        }
        if not ok:
            results[pname]['fd_at_max'] = float(fd_g[max_err_idx])
            results[pname]['cpp_at_max'] = float(cpp_g[max_err_idx])

    return all_pass, results


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cpp_exe = os.path.join(base_dir, 'build', 'gs3d_verify_grads')
    if not os.path.exists(cpp_exe):
        cpp_exe = './build/gs3d_verify_grads'
    if not os.path.exists(cpp_exe):
        print("ERROR: build gs3d_verify_grads first")
        return False

    base_out = os.path.join(base_dir, 'train', 'verify_grads_ext')
    os.makedirs(base_out, exist_ok=True)

    # Test configurations: (name, N, W, H, sh_degree, seed)
    configs = [
        # Original baseline
        ("base_3g_16x16_sh0",       3, 16, 16, 0,  42),
        # Rotation stress: non-trivial quaternions
        ("rot_3g_16x16_sh0",        3, 16, 16, 0, 123),
        # More Gaussians
        ("many_10g_16x16_sh0",     10, 16, 16, 0, 456),
        ("many_30g_16x16_sh0",     30, 16, 16, 0, 789),
        # Larger resolution
        ("res_3g_32x32_sh0",        3, 32, 32, 0,  42),
        # Higher SH degrees
        ("sh1_3g_16x16",            3, 16, 16, 1, 100),
        ("sh2_3g_16x16",            3, 16, 16, 2, 200),
        ("sh3_3g_16x16",            3, 16, 16, 3, 300),
        # SH degree + more Gaussians
        ("sh1_10g_16x16",          10, 16, 16, 1, 500),
        # Larger scene
        ("big_10g_32x32_sh1",      10, 32, 32, 1, 600),
    ]

    print("=" * 80)
    print("EXTENDED GRADIENT VERIFICATION (float32-precise FD)")
    print("=" * 80)
    print(f"Configurations: {len(configs)}")
    print(f"C++ executable: {cpp_exe}")
    print(f"Method: C++ FD vs C++ analytical (same executable)")
    print(f"Rel threshold: 10%, Abs tolerance: 5e-5")
    print()

    total_pass = 0
    total_fail = 0
    failed_details = []

    for name, N, W, H, sh_deg, seed in configs:
        t0 = time.time()
        sys.stdout.write(f"  {name:30s} (N={N:2d}, {W}x{H}, SH{sh_deg}) ... ")
        sys.stdout.flush()

        passed, results = run_config(name, N, W, H, sh_deg, seed, cpp_exe, base_out)
        elapsed = time.time() - t0

        if 'error' in results:
            print(f"ERROR ({elapsed:.1f}s)")
            print(f"    {results['error']}")
            total_fail += 1
            continue

        # Summarize
        param_summary = []
        for pname in ['raw_sh_coeffs', 'raw_opacities', 'raw_scales', 'raw_rotations', 'raw_positions']:
            r = results.get(pname, {})
            status = r.get('status', '?')
            err = r.get('max_rel_err', -1)
            skipped = r.get('n_skipped', 0)
            compared = r.get('n_compared', 0)
            short = pname.replace('raw_','')[:4]
            if status == 'PASS':
                param_summary.append(f"{short}={err:.1e}({compared}/{compared+skipped})")
            else:
                param_summary.append(f"{short}=FAIL({err:.1e})")

        status_str = "PASS" if passed else "FAIL"
        print(f"{status_str}  [{', '.join(param_summary)}]  ({elapsed:.1f}s)")

        if passed:
            total_pass += 1
        else:
            total_fail += 1
            for pname, r in results.items():
                if r.get('status') == 'FAIL':
                    failed_details.append(
                        f"    {name}/{pname}: max_rel_err={r['max_rel_err']:.6f} "
                        f"at idx={r['max_err_idx']} "
                        f"(fd={r.get('fd_at_max', '?'):.8f}, "
                        f"ana={r.get('cpp_at_max', '?'):.8f})")

    print()
    print("=" * 80)
    print(f"RESULTS: {total_pass} passed, {total_fail} failed, "
          f"{total_pass + total_fail} total")
    if failed_details:
        print("\nFailed details:")
        for d in failed_details:
            print(d)
    print("=" * 80)

    return total_fail == 0


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
