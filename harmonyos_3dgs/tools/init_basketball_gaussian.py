#!/usr/bin/env python3
"""Convert basketball COLMAP sparse PLY → 3DGS-initialized PLY for gs3d_vk_train.

Creates the same initialization as test_vk_vs_cuda_10step.cpp:
  raw_xyz   = COLMAP xyz
  SH DC     = (rgb/255 - 0.5) / C0
  opacity   = logit(0.1)
  scale     = log(0.03) * [1, 1, 1]
  rotation  = identity quaternion [1, 0, 0, 0]

Output PLY has all 3DGS properties: f_dc_0..2, f_rest_0..44, opacity,
scale_0..2, rot_0..3 — compatible with gs3d_vk_train / gs3d_vk_render.

Usage:
  python tools/init_basketball_gaussian.py \\
    --ply /home/robota/Downloads/basketball/sparse/0/points3D.ply \\
    --output /tmp/basketball_init_3dgs.ply
"""

import argparse
import math
import os
import struct
import sys

import numpy as np

SH_DEGREE = 3
K_COEFFS = (SH_DEGREE + 1) ** 2   # 16 per color channel
C0 = 0.28209479177387814           # SH DC factor


# ---------------------------------------------------------------------------
# COLMAP binary PLY loader
# ---------------------------------------------------------------------------
def load_colmap_ply(path: str):
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n = 0
        properties = []
        for line in header_lines:
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[1], parts[2]))

        fmt_map = {"float": "f", "double": "d", "uchar": "B",
                   "uint8": "B", "int": "i", "uint": "I"}
        fmt = "<" + "".join(fmt_map[t] for t, _ in properties)
        sz = struct.calcsize(fmt)

        xyz  = np.zeros((n, 3), dtype=np.float32)
        rgb8 = np.zeros((n, 3), dtype=np.uint8)

        prop_names = [nm for _, nm in properties]
        x_i = prop_names.index("x")
        y_i = prop_names.index("y")
        z_i = prop_names.index("z")
        r_i = next((prop_names.index(k) for k in ("red", "r", "diffuse_red") if k in prop_names), None)
        g_i = next((prop_names.index(k) for k in ("green", "g", "diffuse_green") if k in prop_names), None)
        b_i = next((prop_names.index(k) for k in ("blue", "b", "diffuse_blue") if k in prop_names), None)

        for i in range(n):
            row = struct.unpack(fmt, f.read(sz))
            xyz[i]  = (row[x_i], row[y_i], row[z_i])
            if r_i is not None:
                rgb8[i] = (row[r_i], row[g_i], row[b_i])

    print(f"Loaded {n} points from {path}")
    return xyz, rgb8


# ---------------------------------------------------------------------------
# Write 3DGS PLY (same format as GaussianModel.save_ply)
# ---------------------------------------------------------------------------
def write_3dgs_ply(path: str,
                   xyz:    np.ndarray,    # (N,3) float32
                   sh:     np.ndarray,    # (N,K,3) float32
                   op:     np.ndarray,    # (N,1) float32  raw opacity
                   scale:  np.ndarray,    # (N,3) float32  log scale
                   rot:    np.ndarray,    # (N,4) float32  quaternion
                   ):
    N = xyz.shape[0]
    K = sh.shape[1]

    # Build property list
    props = []
    props += [("x","float"), ("y","float"), ("z","float")]
    props += [("nx","float"), ("ny","float"), ("nz","float")]
    for c in range(3):
        props.append((f"f_dc_{c}", "float"))
    for k in range(1, K):
        for c in range(3):
            props.append((f"f_rest_{(k-1)*3+c}", "float"))
    props.append(("opacity", "float"))
    for j in range(3):
        props.append((f"scale_{j}", "float"))
    for j in range(4):
        props.append((f"rot_{j}", "float"))

    with open(path, "wb") as f:
        # Header
        lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {N}",
        ]
        for pname, ptype in props:
            lines.append(f"property {ptype} {pname}")
        lines.append("end_header")
        for line in lines:
            f.write((line + "\n").encode("ascii"))

        # Data
        normals = np.zeros((N, 3), dtype=np.float32)
        sh_dc   = sh[:, 0, :]                     # (N,3)
        sh_rest = sh[:, 1:, :].reshape(N, -1)     # (N,(K-1)*3)

        row = np.concatenate([
            xyz,           # x y z
            normals,       # nx ny nz
            sh_dc,         # f_dc_0..2
            sh_rest,       # f_rest_0..(K-1)*3-1
            op,            # opacity
            scale,         # scale_0..2
            rot,           # rot_0..3
        ], axis=1).astype(np.float32)

        f.write(row.tobytes())

    print(f"Saved 3DGS PLY: {path}  ({N} Gaussians, {K} SH coeffs)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply",    default="/home/robota/Downloads/basketball/sparse/0/points3D.ply")
    parser.add_argument("--output", default="/tmp/basketball_init_3dgs.ply")
    args = parser.parse_args()

    if not os.path.exists(args.ply):
        print(f"ERROR: PLY not found: {args.ply}", file=sys.stderr)
        sys.exit(1)

    xyz, rgb8 = load_colmap_ply(args.ply)
    N = xyz.shape[0]

    # SH DC from RGB (same as CUDA test init)
    rgb_f = rgb8.astype(np.float32) / 255.0    # (N,3) [0,1]
    sh_dc = ((rgb_f - 0.5) / C0)               # (N,3)

    sh = np.zeros((N, K_COEFFS, 3), dtype=np.float32)
    sh[:, 0, :] = sh_dc

    # Opacity: logit(0.1) = log(0.1/0.9)
    op = np.full((N, 1), math.log(0.1 / 0.9), dtype=np.float32)

    # Scale: log(0.03) × 3
    sc = np.full((N, 3), math.log(0.03), dtype=np.float32)

    # Rotation: identity quaternion [w=1, x=0, y=0, z=0]
    rot = np.zeros((N, 4), dtype=np.float32)
    rot[:, 0] = 1.0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_3dgs_ply(args.output, xyz, sh, op, sc, rot)


if __name__ == "__main__":
    main()
