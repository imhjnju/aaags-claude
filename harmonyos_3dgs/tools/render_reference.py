#!/usr/bin/env python3
"""Pure Python/PyTorch CPU reference renderer for 3DGS.
Follows the CUDA forward.cu logic exactly for ground-truth comparison."""

import struct, json, sys, time, math
import numpy as np
from PIL import Image

# ============ SH Constants (from forward.cu) ============
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
          -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
          0.3731763325901154, -0.4570457994644658, 1.445305721320277,
          -0.5900435899266435]

def load_ply(path):
    """Load PLY with same logic as our C++ loader."""
    with open(path, 'rb') as f:
        props = []
        n = 0
        for line in f:
            line = line.decode().strip()
            if line == 'end_header': break
            if line.startswith('element vertex '): n = int(line.split()[-1])
            if line.startswith('property float '): props.append(line.split()[-1])
        nf = len(props)
        raw = np.frombuffer(f.read(n * nf * 4), dtype=np.float32).reshape(n, nf)
    pi = {name: i for i, name in enumerate(props)}

    positions = np.stack([raw[:, pi['x']], raw[:, pi['y']], raw[:, pi['z']]], axis=1)

    # DC SH
    dc = np.stack([raw[:, pi['f_dc_0']], raw[:, pi['f_dc_1']], raw[:, pi['f_dc_2']]], axis=1)

    # Rest SH (channel-first in PLY)
    sh_rest_count = 0
    while f'f_rest_{sh_rest_count}' in pi:
        sh_rest_count += 1
    sh_degree = 3 if sh_rest_count >= 45 else (2 if sh_rest_count >= 24 else (1 if sh_rest_count >= 9 else 0))
    sh_per_ch = (sh_degree + 1)**2 - 1
    max_coeffs = (sh_degree + 1)**2

    # Reorder to basis-interleaved [N, max_coeffs, 3]
    sh_coeffs = np.zeros((n, max_coeffs, 3), dtype=np.float32)
    sh_coeffs[:, 0, :] = dc
    for k in range(sh_per_ch):
        sh_coeffs[:, 1+k, 0] = raw[:, pi[f'f_rest_{k}']]
        sh_coeffs[:, 1+k, 1] = raw[:, pi[f'f_rest_{sh_per_ch+k}']]
        sh_coeffs[:, 1+k, 2] = raw[:, pi[f'f_rest_{2*sh_per_ch+k}']]

    # Activations
    opacities = 1.0 / (1.0 + np.exp(-raw[:, pi['opacity']]))
    scales = np.exp(np.stack([raw[:, pi['scale_0']], raw[:, pi['scale_1']], raw[:, pi['scale_2']]], axis=1))
    rots = np.stack([raw[:, pi['rot_0']], raw[:, pi['rot_1']], raw[:, pi['rot_2']], raw[:, pi['rot_3']]], axis=1)
    rots = rots / np.linalg.norm(rots, axis=1, keepdims=True)

    return dict(positions=positions, sh_coeffs=sh_coeffs, scales=scales,
                rotations=rots, opacities=opacities, sh_degree=sh_degree,
                max_coeffs=max_coeffs, count=n)

def load_camera(json_path, cam_id):
    """Load camera from cameras.json — same logic as C++ loadCameraJson."""
    with open(json_path) as f:
        cams = json.load(f)
    cam = [c for c in cams if c['id'] == cam_id][0]

    pos = np.array(cam['position'], dtype=np.float64)
    # cameras.json stores C2W rotation (not W2C!)
    R_c2w = np.array(cam['rotation'], dtype=np.float64)
    fx, fy = cam['fx'], cam['fy']
    w, h = cam['width'], cam['height']

    tan_fovx = w / (2.0 * fx)
    tan_fovy = h / (2.0 * fy)

    # Build W2C view matrix: R_w2c = R_c2w^T, t = -R_w2c * pos
    R_w2c = R_c2w.T
    t = -R_w2c @ pos

    # Column-major 4x4
    view = np.zeros(16, dtype=np.float64)
    view[0], view[1], view[2] = R_w2c[0,0], R_w2c[1,0], R_w2c[2,0]
    view[4], view[5], view[6] = R_w2c[0,1], R_w2c[1,1], R_w2c[2,1]
    view[8], view[9], view[10] = R_w2c[0,2], R_w2c[1,2], R_w2c[2,2]
    view[12], view[13], view[14], view[15] = t[0], t[1], t[2], 1.0
    view[3] = view[7] = view[11] = 0.0

    # Projection matrix (column-major)
    znear, zfar = 0.01, 100.0
    proj = np.zeros(16, dtype=np.float64)
    proj[0] = 1.0 / tan_fovx
    proj[5] = 1.0 / tan_fovy
    proj[10] = zfar / (zfar - znear)
    proj[11] = 1.0
    proj[14] = -(zfar * znear) / (zfar - znear)

    # viewproj = proj * view (column-major multiply)
    def mat4mul(A, B):
        out = np.zeros(16, dtype=np.float64)
        for col in range(4):
            for row in range(4):
                s = 0.0
                for k in range(4):
                    s += A[k*4+row] * B[col*4+k]
                out[col*4+row] = s
        return out

    viewproj = mat4mul(proj, view)

    return dict(view_matrix=view.astype(np.float32), viewproj_matrix=viewproj.astype(np.float32),
                cam_pos=pos.astype(np.float32), tan_fovx=float(tan_fovx), tan_fovy=float(tan_fovy),
                width=w, height=h, focal_x=float(fx), focal_y=float(fy))

# ============ Core math (matching CUDA forward.cu EXACTLY) ============

def transform_point4x3(p, m):
    """Column-major 4x4 matrix * point (homogeneous w=1, return xyz)."""
    return np.array([
        m[0]*p[0] + m[4]*p[1] + m[8]*p[2] + m[12],
        m[1]*p[0] + m[5]*p[1] + m[9]*p[2] + m[13],
        m[2]*p[0] + m[6]*p[1] + m[10]*p[2] + m[14],
    ], dtype=np.float32)

def transform_point4x4(p, m):
    return np.array([
        m[0]*p[0] + m[4]*p[1] + m[8]*p[2] + m[12],
        m[1]*p[0] + m[5]*p[1] + m[9]*p[2] + m[13],
        m[2]*p[0] + m[6]*p[1] + m[10]*p[2] + m[14],
        m[3]*p[0] + m[7]*p[1] + m[11]*p[2] + m[15],
    ], dtype=np.float32)

def ndc2pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5

def compute_cov3d(scale, mod, rot):
    """Matches CUDA computeCov3D — GLM column-major convention."""
    sx, sy, sz = mod * scale[0], mod * scale[1], mod * scale[2]
    r, x, y, z = rot[0], rot[1], rot[2], rot[3]

    # GLM mat3 fills COLUMNS: R_glm[col][row]
    # R_math(row, col) = R_glm[col][row]
    # For C row-major storage: R[row][col] = R_math(row, col)
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y+r*z),   2*(x*z-r*y)],
        [2*(x*y-r*z),   1-2*(x*x+z*z), 2*(y*z+r*x)],
        [2*(x*z+r*y),   2*(y*z-r*x),   1-2*(x*x+y*y)]
    ], dtype=np.float32)

    S = np.diag([sx, sy, sz]).astype(np.float32)
    M = S @ R  # M[i][j] = S[i] * R[i][j]
    Sigma = M.T @ M  # upper triangle
    return np.array([Sigma[0,0], Sigma[0,1], Sigma[0,2],
                     Sigma[1,1], Sigma[1,2], Sigma[2,2]], dtype=np.float32)

def compute_cov2d(mean3d, cov3d, vm, focal_x, focal_y, tan_fovx, tan_fovy):
    """Matches CUDA computeCov2D — GLM column-major convention."""
    t = transform_point4x3(mean3d, vm)

    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    t[0] = min(limx, max(-limx, txtz)) * t[2]
    t[1] = min(limy, max(-limy, tytz)) * t[2]

    # J stored as [col][row] matching GLM
    J = np.array([
        [focal_x/t[2], 0, -(focal_x*t[0])/(t[2]*t[2])],
        [0, focal_y/t[2], -(focal_y*t[1])/(t[2]*t[2])],
        [0, 0, 0]
    ], dtype=np.float32)

    # W stored as [col][row]
    W = np.array([
        [vm[0], vm[4], vm[8]],
        [vm[1], vm[5], vm[9]],
        [vm[2], vm[6], vm[10]]
    ], dtype=np.float32)

    # T = W * J (column-major multiply)
    T = np.zeros((3, 3), dtype=np.float32)
    for col in range(3):
        for row in range(3):
            for k in range(3):
                T[col][row] += W[k][row] * J[col][k]

    Vrk = np.array([
        [cov3d[0], cov3d[1], cov3d[2]],
        [cov3d[1], cov3d[3], cov3d[4]],
        [cov3d[2], cov3d[4], cov3d[5]]
    ], dtype=np.float32)

    # cov = T^T * Vrk * T
    tmp = np.zeros((3, 3), dtype=np.float32)
    for col in range(3):
        for row in range(3):
            for k in range(3):
                tmp[col][row] += Vrk[row][k] * T[col][k]

    result = np.zeros((3, 3), dtype=np.float32)
    for col in range(3):
        for row in range(3):
            for k in range(3):
                result[col][row] += T[row][k] * tmp[col][k]

    return np.array([result[0][0], result[1][0], result[1][1]], dtype=np.float32)

def compute_sh(degree, sh_coeffs, pos, cam_pos):
    """Matches CUDA computeColorFromSH — NO upper clamp (original behavior)."""
    d = pos - cam_pos
    d = d / np.linalg.norm(d)
    x, y, z = d
    sh = sh_coeffs  # [max_coeffs, 3]

    result = SH_C0 * sh[0]
    if degree > 0:
        result += -SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3]
    if degree > 1:
        xx, yy, zz = x*x, y*y, z*z
        xy, yz, xz = x*y, y*z, x*z
        result += (SH_C2[0]*xy*sh[4] + SH_C2[1]*yz*sh[5] +
                   SH_C2[2]*(2*zz-xx-yy)*sh[6] + SH_C2[3]*xz*sh[7] +
                   SH_C2[4]*(xx-yy)*sh[8])
    if degree > 2:
        result += (SH_C3[0]*y*(3*xx-yy)*sh[9] + SH_C3[1]*xy*z*sh[10] +
                   SH_C3[2]*y*(4*zz-xx-yy)*sh[11] + SH_C3[3]*z*(2*zz-3*xx-3*yy)*sh[12] +
                   SH_C3[4]*x*(4*zz-xx-yy)*sh[13] + SH_C3[5]*z*(xx-yy)*sh[14] +
                   SH_C3[6]*x*(xx-3*yy)*sh[15])
    result += 0.5
    return np.maximum(result, 0.0)  # CUDA: only lower clamp

def preprocess(model, cam, scale_modifier=1.0, antialiasing=True):
    """Preprocess all Gaussians — matches CUDA preprocessCUDA."""
    N = model['count']
    focal_x = cam['width'] / (2.0 * cam['tan_fovx'])
    focal_y = cam['height'] / (2.0 * cam['tan_fovy'])
    TILE_W, TILE_H = 16, 16
    grid_x = (cam['width'] + TILE_W - 1) // TILE_W
    grid_y = (cam['height'] + TILE_H - 1) // TILE_H

    results = []
    for i in range(N):
        pos = model['positions'][i]

        # Frustum test
        p_view = transform_point4x3(pos, cam['view_matrix'])
        if p_view[2] <= 0.2:
            continue

        # NDC projection
        p_hom = transform_point4x4(pos, cam['viewproj_matrix'])
        p_w = 1.0 / (p_hom[3] + 1e-7)
        p_ndc = p_hom[:3] * p_w

        px = ndc2pix(p_ndc[0], cam['width'])
        py = ndc2pix(p_ndc[1], cam['height'])

        # 3D covariance
        cov3d = compute_cov3d(model['scales'][i], scale_modifier, model['rotations'][i])

        # 2D covariance
        cov2d = compute_cov2d(pos, cov3d, cam['view_matrix'], focal_x, focal_y,
                              cam['tan_fovx'], cam['tan_fovy'])

        # Antialiasing
        det_orig = cov2d[0] * cov2d[2] - cov2d[1] * cov2d[1]
        cov2d[0] += 0.3
        cov2d[2] += 0.3
        det_plus = cov2d[0] * cov2d[2] - cov2d[1] * cov2d[1]
        h_scale = 1.0
        if antialiasing:
            h_scale = math.sqrt(max(0.000025, det_orig / det_plus))

        det = det_plus
        if det == 0:
            continue
        det_inv = 1.0 / det
        conic = np.array([cov2d[2]*det_inv, -cov2d[1]*det_inv, cov2d[0]*det_inv], dtype=np.float32)

        # Screen radius
        mid = 0.5 * (cov2d[0] + cov2d[2])
        l1 = mid + math.sqrt(max(0.1, mid*mid - det))
        l2 = mid - math.sqrt(max(0.1, mid*mid - det))
        radius = int(math.ceil(3.0 * math.sqrt(max(l1, l2))))

        if radius > max(cam['width'], cam['height']):
            continue

        # Tile rect
        rect_min = [min(grid_x, max(0, int((px - radius) / TILE_W))),
                     min(grid_y, max(0, int((py - radius) / TILE_H)))]
        rect_max = [min(grid_x, max(0, int((px + radius + TILE_W - 1) / TILE_W))),
                     min(grid_y, max(0, int((py + radius + TILE_H - 1) / TILE_H)))]
        tiles = (rect_max[0] - rect_min[0]) * (rect_max[1] - rect_min[1])
        if tiles == 0:
            continue

        # SH color — NO upper clamp (matching CUDA exactly)
        rgb = compute_sh(model['sh_degree'], model['sh_coeffs'][i], pos, cam['cam_pos'])

        opa = model['opacities'][i] * h_scale

        results.append(dict(
            idx=i, px=px, py=py, depth=p_view[2],
            conic=conic, rgb=rgb, opacity=opa, radius=radius,
            rect_min=rect_min, rect_max=rect_max
        ))

    return results

def render_image(preprocessed, cam, bg_color=np.zeros(3)):
    """Tile-based alpha blending — matches CUDA renderCUDA."""
    W, H = cam['width'], cam['height']
    TILE_W, TILE_H = 16, 16
    grid_x = (W + TILE_W - 1) // TILE_W
    grid_y = (H + TILE_H - 1) // TILE_H

    # Bin Gaussians into tiles, sort by depth
    tiles = {}
    for g in preprocessed:
        for ty in range(g['rect_min'][1], g['rect_max'][1]):
            for tx in range(g['rect_min'][0], g['rect_max'][0]):
                key = ty * grid_x + tx
                if key not in tiles:
                    tiles[key] = []
                tiles[key].append(g)

    # Sort each tile by depth
    for key in tiles:
        tiles[key].sort(key=lambda g: g['depth'])

    image = np.zeros((H, W, 3), dtype=np.float32)

    for ty in range(grid_y):
        for tx in range(grid_x):
            tile_id = ty * grid_x + tx
            if tile_id not in tiles:
                # No gaussians — fill with bg
                px_min_x, px_min_y = tx * TILE_W, ty * TILE_H
                px_max_x = min(px_min_x + TILE_W, W)
                px_max_y = min(px_min_y + TILE_H, H)
                image[px_min_y:px_max_y, px_min_x:px_max_x] = bg_color
                continue

            gaussians = tiles[tile_id]
            px_min_x, px_min_y = tx * TILE_W, ty * TILE_H
            px_max_x = min(px_min_x + TILE_W, W)
            px_max_y = min(px_min_y + TILE_H, H)

            for py in range(px_min_y, px_max_y):
                for px_coord in range(px_min_x, px_max_x):
                    T = 1.0
                    C = np.zeros(3, dtype=np.float64)

                    for g in gaussians:
                        dx = g['px'] - px_coord
                        dy = g['py'] - py
                        con = g['conic']

                        power = -0.5 * (con[0]*dx*dx + con[2]*dy*dy) - con[1]*dx*dy
                        if power > 0.0:
                            continue

                        alpha = min(0.99, g['opacity'] * math.exp(power))
                        if alpha < 1.0/255.0:
                            continue

                        test_T = T * (1.0 - alpha)
                        if test_T < 0.0001:
                            break

                        C += g['rgb'] * alpha * T
                        T = test_T

                    image[py, px_coord] = C + T * bg_color

    return image

def main():
    if len(sys.argv) < 4:
        print("Usage: render_reference.py <model.ply> <cameras.json> <cam_id> [output.png]")
        sys.exit(1)

    ply_path = sys.argv[1]
    json_path = sys.argv[2]
    cam_id = int(sys.argv[3])
    out_path = sys.argv[4] if len(sys.argv) > 4 else 'reference.png'

    print(f"Loading {ply_path}...")
    model = load_ply(ply_path)
    print(f"  {model['count']} Gaussians, SH degree {model['sh_degree']}")

    print(f"Loading camera {cam_id} from {json_path}...")
    cam = load_camera(json_path, cam_id)
    print(f"  {cam['width']}x{cam['height']}, focal=({cam['focal_x']:.1f},{cam['focal_y']:.1f})")

    print("Preprocessing...")
    t0 = time.time()
    preprocessed = preprocess(model, cam)
    t1 = time.time()
    print(f"  {len(preprocessed)} visible Gaussians ({t1-t0:.1f}s)")

    # Dump some intermediate values for comparison
    if len(preprocessed) > 0:
        g = preprocessed[0]
        print(f"\n=== First visible Gaussian (idx={g['idx']}) ===")
        print(f"  pos2d=({g['px']:.2f}, {g['py']:.2f}), depth={g['depth']:.4f}")
        print(f"  conic=({g['conic'][0]:.6f}, {g['conic'][1]:.6f}, {g['conic'][2]:.6f})")
        print(f"  rgb=({g['rgb'][0]:.4f}, {g['rgb'][1]:.4f}, {g['rgb'][2]:.4f})")
        print(f"  opacity={g['opacity']:.4f}, radius={g['radius']}")

    print(f"\nRendering {cam['width']}x{cam['height']}...")
    t2 = time.time()
    image = render_image(preprocessed, cam, bg_color=np.zeros(3))
    t3 = time.time()
    print(f"  Render time: {t3-t2:.1f}s")

    # Clamp to [0,1] and save
    image = np.clip(image, 0.0, 1.0)
    img_uint8 = (image * 255 + 0.5).astype(np.uint8)
    Image.fromarray(img_uint8).save(out_path)
    print(f"Saved to {out_path}")

    # Statistics
    white_px = np.sum(np.all(img_uint8 > 245, axis=2))
    print(f"White pixels (>245): {white_px}")

if __name__ == '__main__':
    main()
