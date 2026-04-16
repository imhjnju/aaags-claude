#!/usr/bin/env python3
"""Verify C++ backward gradients against Python finite-difference.
All matrix operations use column-major [col][row] convention matching C++."""

import numpy as np
import subprocess, os, json

SH_C0 = 0.28209479177387814

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

# ---- Exact C++ function ports (column-major [col][row]) ----

def transformPoint4x3(p, m):
    """m: flat[16] column-major. m[col*4+row]."""
    return np.array([
        m[0]*p[0] + m[4]*p[1] + m[8]*p[2]  + m[12],
        m[1]*p[0] + m[5]*p[1] + m[9]*p[2]  + m[13],
        m[2]*p[0] + m[6]*p[1] + m[10]*p[2] + m[14],
    ], dtype=np.float64)

def transformPoint4x4(p, m):
    return np.array([
        m[0]*p[0] + m[4]*p[1] + m[8]*p[2]  + m[12],
        m[1]*p[0] + m[5]*p[1] + m[9]*p[2]  + m[13],
        m[2]*p[0] + m[6]*p[1] + m[10]*p[2] + m[14],
        m[3]*p[0] + m[7]*p[1] + m[11]*p[2] + m[15],
    ], dtype=np.float64)

def ndc2pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5

def computeCov3D(scale, mod, rot):
    """Exact port of C++ computeCov3D. rot = (r,x,y,z)."""
    sx, sy, sz = mod * scale[0], mod * scale[1], mod * scale[2]
    r, x, y, z = rot

    # R[row][col] — row-major storage matching C++
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y+r*z),   2*(x*z-r*y)],
        [2*(x*y-r*z),   1-2*(x*x+z*z), 2*(y*z+r*x)],
        [2*(x*z+r*y),   2*(y*z-r*x),   1-2*(x*x+y*y)]
    ], dtype=np.float64)

    # M[i][j] = S[i] * R[i][j]
    M = np.zeros((3,3), dtype=np.float64)
    for i in range(3):
        s = [sx, sy, sz][i]
        for j in range(3):
            M[i][j] = s * R[i][j]

    # Sigma = M^T * M, upper triangle
    cov3D = np.zeros(6, dtype=np.float64)
    cov3D[0] = M[0][0]*M[0][0] + M[1][0]*M[1][0] + M[2][0]*M[2][0]
    cov3D[1] = M[0][0]*M[0][1] + M[1][0]*M[1][1] + M[2][0]*M[2][1]
    cov3D[2] = M[0][0]*M[0][2] + M[1][0]*M[1][2] + M[2][0]*M[2][2]
    cov3D[3] = M[0][1]*M[0][1] + M[1][1]*M[1][1] + M[2][1]*M[2][1]
    cov3D[4] = M[0][1]*M[0][2] + M[1][1]*M[1][2] + M[2][1]*M[2][2]
    cov3D[5] = M[0][2]*M[0][2] + M[1][2]*M[1][2] + M[2][2]*M[2][2]
    return cov3D

def computeCov2D(mean3d, cov3D, vm, focal_x, focal_y, tan_fovx, tan_fovy):
    """Exact port of C++ computeCov2D. vm: flat[16] column-major.
    J, W, T stored as [col][row] matching C++."""
    t = transformPoint4x3(mean3d, vm)

    # Clamp to 1.3x frustum
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    t[0] = min(limx, max(-limx, txtz)) * t[2]
    t[1] = min(limy, max(-limy, tytz)) * t[2]

    # J[col][row] — column-major
    J = np.zeros((3,3), dtype=np.float64)
    J[0][0] = focal_x / t[2]
    J[0][2] = -(focal_x * t[0]) / (t[2] * t[2])
    J[1][1] = focal_y / t[2]
    J[1][2] = -(focal_y * t[1]) / (t[2] * t[2])

    # W[col][row] — column-major, from view_matrix
    W = np.zeros((3,3), dtype=np.float64)
    W[0][0] = vm[0]; W[0][1] = vm[4]; W[0][2] = vm[8]
    W[1][0] = vm[1]; W[1][1] = vm[5]; W[1][2] = vm[9]
    W[2][0] = vm[2]; W[2][1] = vm[6]; W[2][2] = vm[10]

    # T = W * J, both [col][row]
    # T[col][row] = sum_k W[k][row] * J[col][k]
    T = np.zeros((3,3), dtype=np.float64)
    for col in range(3):
        for row in range(3):
            for k in range(3):
                T[col][row] += W[k][row] * J[col][k]

    # Vrk from upper triangle
    Vrk = np.array([
        [cov3D[0], cov3D[1], cov3D[2]],
        [cov3D[1], cov3D[3], cov3D[4]],
        [cov3D[2], cov3D[4], cov3D[5]]
    ], dtype=np.float64)

    # tmp = Vrk * T: tmp[col][row] = sum_k Vrk[row][k] * T[col][k]
    tmp = np.zeros((3,3), dtype=np.float64)
    for col in range(3):
        for row in range(3):
            for k in range(3):
                tmp[col][row] += Vrk[row][k] * T[col][k]

    # result = T^T * tmp: result[col][row] = sum_k T[row][k] * tmp[col][k]
    result = np.zeros((3,3), dtype=np.float64)
    for col in range(3):
        for row in range(3):
            for k in range(3):
                result[col][row] += T[row][k] * tmp[col][k]

    cov2D = np.array([result[0][0], result[1][0], result[1][1]], dtype=np.float64)
    return cov2D, t

def computeSH_deg0(sh_dc, pos, cam_pos):
    d = pos - cam_pos
    norm = np.sqrt(np.sum(d**2))
    if norm < 1e-8:
        d = np.array([0,0,1], dtype=np.float64)
    else:
        d = d / norm
    result = SH_C0 * sh_dc + 0.5
    return np.maximum(0.0, result)

def forward_render(raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
                    vm, vpm, cam_pos, focal_x, focal_y,
                    tan_fovx, tan_fovy, W, H, bg):
    """Full forward matching C++ pipeline. All column-major."""
    N = len(raw_pos)
    pos = raw_pos.copy()
    scales = np.exp(raw_sc)
    rots = raw_rot.copy()
    for i in range(N):
        rots[i] = rots[i] / (np.linalg.norm(rots[i]) + 1e-12)
    opacities = sigmoid(raw_op)

    means2d = np.zeros((N, 2))
    conics = np.zeros((N, 3))
    rgb = np.zeros((N, 3))
    op2d = np.zeros(N)
    depths = np.zeros(N)
    visible = np.zeros(N, dtype=bool)

    for i in range(N):
        p_view = transformPoint4x3(pos[i], vm)
        if p_view[2] <= 0.2:
            continue
        depths[i] = p_view[2]

        p_hom = transformPoint4x4(pos[i], vpm)
        if abs(p_hom[3]) < 1e-7:
            continue
        p_ndc = p_hom[:3] / p_hom[3]
        means2d[i] = [ndc2pix(p_ndc[0], W), ndc2pix(p_ndc[1], H)]

        cov3d = computeCov3D(scales[i], 1.0, rots[i])
        cov2d, _ = computeCov2D(pos[i], cov3d, vm, focal_x, focal_y, tan_fovx, tan_fovy)

        a = cov2d[0] + 0.3
        b = cov2d[1]
        c = cov2d[2] + 0.3
        det = a * c - b * b
        if det <= 0:
            continue

        conics[i] = [c/det, -b/det, a/det]
        rgb[i] = computeSH_deg0(raw_sh[i, :3], pos[i], cam_pos)
        op2d[i] = opacities[i]
        visible[i] = True

    # Sort by depth
    order = np.argsort(depths)

    image = np.zeros((H, W, 3))
    for py in range(H):
        for px in range(W):
            T = 1.0
            C = np.zeros(3)
            for i in order:
                if not visible[i]:
                    continue
                dx = means2d[i, 0] - float(px)
                dy = means2d[i, 1] - float(py)
                ca, cb, cc = conics[i]
                power = -0.5 * (ca*dx*dx + cc*dy*dy) - cb*dx*dy
                if power > 0:
                    continue
                alpha = min(0.99, float(op2d[i]) * np.exp(power))
                if alpha < 1.0/255.0:
                    continue
                C += rgb[i] * alpha * T
                T *= (1.0 - alpha)
                if T < 0.0001:
                    break
            C += T * np.array(bg)
            image[py, px] = C
    return image

def l1_loss(rendered, gt):
    return float(np.mean(np.abs(rendered.flatten() - gt.flatten())))

def write_bin(path, arr):
    arr.astype(np.float32).tofile(path)

def read_bin(path, count):
    return np.fromfile(path, dtype=np.float32, count=count)

def main():
    np.random.seed(42)
    N, W, H = 3, 16, 16
    bg = [0.0, 0.0, 0.0]

    raw_pos = np.array([[0,0,5],[0.5,-0.3,6],[-0.4,0.2,4.5]], dtype=np.float64)
    raw_sc = np.log(np.array([[0.3,0.3,0.3],[0.2,0.4,0.2],[0.5,0.2,0.3]], dtype=np.float64))
    raw_rot = np.array([[1,0,0,0],[0.9,0.1,0.2,0],[0.8,0,0.3,0.1]], dtype=np.float64)
    raw_sh = np.array([[1.5,0.8,0.3],[0.5,1.5,0.8],[0.3,0.5,1.5]], dtype=np.float64)
    raw_op = np.array([2.0, 1.5, 1.0], dtype=np.float64)

    fov = 60.0
    tan_fov = float(np.tan(np.radians(fov/2)))
    focal_x = W / (2.0 * tan_fov)
    focal_y = H / (2.0 * tan_fov)
    cam_pos = np.array([0,0,0], dtype=np.float64)

    # Identity view (column-major flat[16])
    vm = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
    # Perspective (column-major, matching C++ buildPerspective)
    near, far = 0.01, 100.0
    r = tan_fov * near
    vpm = [0.0]*16
    vpm[0] = near / r       # [0][0]
    vpm[5] = near / r       # [1][1]
    vpm[10] = far/(far-near) # [2][2]
    vpm[11] = 1.0            # [2][3]
    vpm[14] = -(far*near)/(far-near) # [3][2]

    # Render GT (offset SH by 0.5)
    gt_sh = raw_sh.copy() + 0.5
    gt_image = forward_render(raw_pos.copy(), raw_sc.copy(), raw_rot.copy(),
                               gt_sh, raw_op.copy(), vm, vpm, cam_pos,
                               focal_x, focal_y, tan_fov, tan_fov, W, H, bg)

    loss0 = l1_loss(
        forward_render(raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
                       vm, vpm, cam_pos, focal_x, focal_y, tan_fov, tan_fov, W, H, bg),
        gt_image)

    print(f"Scene: {N} Gaussians, {W}x{H}")
    print(f"GT sum: {gt_image.sum():.4f}, Loss: {loss0:.8f}")

    # Finite-diff gradients
    print("\n=== Python FD Gradients (column-major) ===")
    eps = 1e-4
    params = {'raw_sh_coeffs': raw_sh, 'raw_opacities': raw_op,
              'raw_scales': raw_sc, 'raw_rotations': raw_rot,
              'raw_positions': raw_pos}

    py_grads = {}
    for name, arr in params.items():
        grads = np.zeros(arr.size, dtype=np.float64)
        for idx in range(arr.size):
            old = arr.flat[idx]
            arr.flat[idx] = old + eps
            lp = l1_loss(forward_render(raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
                         vm, vpm, cam_pos, focal_x, focal_y, tan_fov, tan_fov, W, H, bg), gt_image)
            arr.flat[idx] = old - eps
            lm = l1_loss(forward_render(raw_pos, raw_sc, raw_rot, raw_sh, raw_op,
                         vm, vpm, cam_pos, focal_x, focal_y, tan_fov, tan_fov, W, H, bg), gt_image)
            arr.flat[idx] = old
            grads[idx] = (lp - lm) / (2*eps)
        py_grads[name] = grads
        print(f"  {name}: {grads}")

    # Save for C++
    out = 'train/verify_grads'
    os.makedirs(out, exist_ok=True)
    write_bin(f'{out}/raw_positions.bin', raw_pos.astype(np.float32))
    write_bin(f'{out}/raw_scales.bin', raw_sc.astype(np.float32))
    write_bin(f'{out}/raw_rotations.bin', raw_rot.astype(np.float32))
    write_bin(f'{out}/raw_sh_coeffs.bin', raw_sh.astype(np.float32))
    write_bin(f'{out}/raw_opacities.bin', raw_op.astype(np.float32))
    write_bin(f'{out}/gt_image.bin', gt_image.astype(np.float32).flatten())
    write_bin(f'{out}/view_matrix.bin', np.array(vm, dtype=np.float32))
    write_bin(f'{out}/viewproj_matrix.bin', np.array(vpm, dtype=np.float32))
    json.dump({'N':N,'W':W,'H':H,'sh_degree':0,'max_coeffs':1,
               'cam_pos':[0,0,0],'tan_fovx':tan_fov,'tan_fovy':tan_fov,
               'bg_color':bg,'loss':loss0}, open(f'{out}/meta.json','w'), indent=2)
    for name, g in py_grads.items():
        write_bin(f'{out}/py_grad_{name}.bin', g.astype(np.float32))

    # Run C++
    cpp = './build/gs3d_verify_grads'
    if not os.path.exists(cpp):
        print(f"\nBuild {cpp} first!"); return
    print(f"\nRunning C++...")
    r = subprocess.run([cpp, out], capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); return

    # Compare
    print("=== Comparison (Python FD vs C++ Analytic) ===")
    all_pass = True
    for name in params:
        py_g = py_grads[name]
        cpp_path = f'{out}/cpp_grad_{name}.bin'
        if not os.path.exists(cpp_path):
            print(f"  {name}: MISSING"); all_pass = False; continue
        cpp_g = read_bin(cpp_path, py_g.size).astype(np.float64)

        max_err = 0
        for j in range(py_g.size):
            denom = max(abs(py_g[j]), abs(cpp_g[j]), 1e-7)
            rel = abs(py_g[j] - cpp_g[j]) / denom
            max_err = max(max_err, rel)

        ok = max_err < 0.10
        if not ok: all_pass = False
        print(f"  {name:20s}: max_rel_err={max_err:.6f}  [{'PASS' if ok else 'FAIL'}]")
        print(f"    py:  {py_g}")
        print(f"    cpp: {cpp_g}")

    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")

if __name__ == '__main__':
    main()
