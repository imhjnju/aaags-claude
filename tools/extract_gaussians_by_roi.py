#!/usr/bin/env python3
"""
Slice a Gaussian-splat .ply by screen-space ROI for diagnostic use.

Given a .ply trained with AAA-Gaussians and a cameras.json entry, project every
Gaussian into that camera's pixel grid, compute a conservative 2D AABB (same
math the CUDA reference uses in preprocessCUDA), and keep only Gaussians whose
AABB intersects a user-supplied pixel region. The kept records are written to a
new binary .ply whose raw per-Gaussian bytes are copied verbatim from the
source (no re-serialisation of SH coefficients, no precision loss).

This is a one-off diagnostic tool for VK-vs-CUDA rasterizer parity
debugging — it is pure CPU, uses numpy, and does not touch the rasterizer.

Math sources (studied and lifted — file:line references live inline):
  - AAA-Gaussians/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
      preprocessCUDA: lines 70-317 (radius, ndc2Pix, extent)
  - .../cuda_rasterizer/forward_common.h
      computeCov2D:   lines 73-106 (J·W·Σ·Wᵀ·Jᵀ, limx/limy clamp)
      dilateCov2D:    lines 108-131 (h_var = 0.3 low-pass)
      computeCov3D:   lines 149-171 (Σ = (S·R)ᵀ·(S·R))
      quat2mat:       auxiliary.h 106-119 (wxyz order)
  - .../cuda_rasterizer/auxiliary.h
      ndc2Pix:        lines 69-72   ((v+1)·S - 1)·0.5
  - AAA-Gaussians/utils/graphics_utils.py
      getWorld2View2, getProjectionMatrix
  - AAA-Gaussians/scene/gaussian_model.py
      scaling_activation = exp, rotation_activation = L2 normalize,
      opacity_activation = sigmoid

Usage:
    python tools/extract_gaussians_by_roi.py \
        --ply /path/to/basket-aaa.ply \
        --cameras /path/to/cameras.json \
        --cam_id 0 \
        --roi 0,720,160,960 \
        --out tools/out/subset.ply \
        --manifest tools/out/subset.manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Camera conversion — mirrors tools/render_single.py::camera_from_json_entry
# ---------------------------------------------------------------------------

def _getWorld2View2(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Port of AAA-Gaussians/utils/graphics_utils.py:38-49.

    Input:  R (3x3) with R == W2C[:3,:3].T, t (3,) == W2C[:3,3].
    Output: W2C 4x4 in row-major numpy (same layout graphics_utils returns).
    """
    Rt = np.zeros((4, 4), dtype=np.float64)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    # getWorld2View2 also round-trips through C2W with optional translate/scale;
    # with the defaults (translate=0, scale=1) that is an identity round-trip,
    # so we return Rt directly — matches render_single.py behaviour.
    return Rt.astype(np.float32)


def _getProjectionMatrix(znear: float, zfar: float, fovX: float, fovY: float) -> np.ndarray:
    """Port of graphics_utils.py:51-71. Returns 4x4 (numpy, row-major)."""
    tanHalfY = math.tan(fovY / 2.0)
    tanHalfX = math.tan(fovX / 2.0)
    top = tanHalfY * znear
    right = tanHalfX * znear
    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] = 2.0 * znear / (2.0 * right)
    P[1, 1] = 2.0 * znear / (2.0 * top)
    P[0, 2] = 0.0
    P[1, 2] = 0.0
    P[2, 2] = zfar / (zfar - znear)
    P[3, 2] = 1.0
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P.astype(np.float32)


def build_camera(entry: dict, znear: float = 0.01, zfar: float = 100.0) -> dict:
    """Reconstruct camera matrices from cameras.json entry.

    Matches render_single.py::camera_from_json_entry conventions exactly:
      - entry["rotation"] is the 3x3 C2W rotation matrix.
      - entry["position"] is the camera origin in world.
      - The rasterizer stores world_view_transform and full_proj_transform
        TRANSPOSED (see render_single.py:82-92); we preserve that so our
        mean2D computation matches the CUDA ndc2Pix path.
    """
    rot_c2w = np.asarray(entry["rotation"], dtype=np.float64)
    pos_c2w = np.asarray(entry["position"], dtype=np.float64)

    C2W = np.eye(4, dtype=np.float64)
    C2W[:3, :3] = rot_c2w
    C2W[:3, 3] = pos_c2w
    W2C = np.linalg.inv(C2W)

    R = W2C[:3, :3].T.astype(np.float32)
    T = W2C[:3, 3].astype(np.float32)

    W = int(entry["width"])
    H = int(entry["height"])
    fx = float(entry["fx"])
    fy = float(entry["fy"])
    fovx = 2.0 * math.atan(W / (2.0 * fx))
    fovy = 2.0 * math.atan(H / (2.0 * fy))

    w2v = _getWorld2View2(R, T)                           # 4x4 standard layout
    proj = _getProjectionMatrix(znear, zfar, fovx, fovy)  # 4x4 standard layout

    # render_single.py transposes each then bmms. full_proj = (w2v.T) @ (proj.T).
    # Then CUDA's world2ndc does viewproj * vec4(p,1) == (w2v.T @ proj.T) * vec(p,1),
    # which equals (proj @ w2v).T * vec(p,1), i.e. CUDA reads transposed.
    # We want to project directly in numpy, so we'll compute the "normal"
    # full_proj = proj @ w2v (not transposed), and call it as full_proj @ p_h.
    full_proj_std = proj @ w2v  # standard (non-transposed) 4x4

    return {
        "W": W, "H": H, "fx": fx, "fy": fy,
        "fovx": fovx, "fovy": fovy,
        "tan_fovx": math.tan(fovx / 2.0),
        "tan_fovy": math.tan(fovy / 2.0),
        "znear": znear, "zfar": zfar,
        "W2C": w2v.astype(np.float64),        # 4x4, world -> camera
        "full_proj": full_proj_std.astype(np.float64),  # 4x4, world -> clip
    }


# ---------------------------------------------------------------------------
# PLY reader / writer — binary_little_endian, all float32
# ---------------------------------------------------------------------------

def parse_ply_header(path: str) -> Tuple[bytes, int, int, list]:
    """Return (raw_header_bytes, header_byte_len, n_vertex, property_names).

    Only supports binary_little_endian float-only vertex element (the AAA-GS
    export format). Keeps raw header so we can re-emit an identical prefix.
    """
    hdr = bytearray()
    n_vertex = -1
    props: list[str] = []
    with open(path, "rb") as f:
        # Read line by line until end_header\n.
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF reading header of {path}")
            hdr += line
            s = line.decode("ascii", errors="replace").strip()
            if s.startswith("element vertex"):
                n_vertex = int(s.split()[-1])
            elif s.startswith("property "):
                parts = s.split()
                # Guard: we only support scalar float properties.
                if parts[1] != "float":
                    raise ValueError(f"Unsupported property type in {path}: {s!r}")
                props.append(parts[2])
            elif s == "end_header":
                break
        header_byte_len = f.tell()
    if n_vertex < 0:
        raise ValueError(f"{path}: no 'element vertex N' in header")
    return bytes(hdr), header_byte_len, n_vertex, props


def load_ply_raw(path: str):
    """Return (header_bytes, header_len, n, props, raw_payload_ndarray).

    raw_payload is a 2-D bytes view of shape (n, stride) so we can slice rows
    and concatenate them verbatim for the subset write.
    """
    header_bytes, header_len, n, props = parse_ply_header(path)
    stride = 4 * len(props)  # every prop is float32
    size = os.path.getsize(path) - header_len
    expected = n * stride
    if size != expected:
        raise ValueError(
            f"{path}: payload size {size} != expected {expected} "
            f"(n={n}, stride={stride})"
        )
    with open(path, "rb") as f:
        f.seek(header_len)
        payload = f.read(expected)
    raw = np.frombuffer(payload, dtype=np.uint8).reshape(n, stride).copy()
    return header_bytes, header_len, n, props, raw


def extract_columns(raw: np.ndarray, props: list, names: list) -> np.ndarray:
    """View per-Gaussian float32 fields by name — returns (N, len(names)) float32.
    """
    idx = [props.index(nm) for nm in names]
    flat = raw.view(np.float32).reshape(raw.shape[0], -1)
    return flat[:, idx].astype(np.float32, copy=True)


def rewrite_header(header: bytes, n_new: int) -> bytes:
    """Replace 'element vertex N' with the new count, keep everything else."""
    lines = header.split(b"\n")
    for i, ln in enumerate(lines):
        if ln.startswith(b"element vertex "):
            lines[i] = f"element vertex {n_new}".encode("ascii")
            break
    return b"\n".join(lines)


# ---------------------------------------------------------------------------
# Screen-space AABB — lifted from forward_common.h + forward.cu
# ---------------------------------------------------------------------------

def quat2mat_wxyz(q: np.ndarray) -> np.ndarray:
    """Vectorised port of auxiliary.h:106-119.

    Input  q: (N,4) with order (w, x, y, z) — matches .ply rot_0..3 and the
              rasterizer's glm::vec4 rotations layout.
    Output R: (N,3,3).
    """
    r = q[:, 0]; x = q[:, 1]; y = q[:, 2]; z = q[:, 3]
    N = q.shape[0]
    R = np.empty((N, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - r * z)
    R[:, 0, 2] = 2.0 * (x * z + r * y)
    R[:, 1, 0] = 2.0 * (x * y + r * z)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - r * x)
    R[:, 2, 0] = 2.0 * (x * z - r * y)
    R[:, 2, 1] = 2.0 * (y * z + r * x)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R


def compute_cov3d(scale: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Port of forward_common.h::computeCov3D (lines 149-171).

    The CUDA code computes M = S · R, Σ = Mᵀ · M. Using S = diag(s) that gives
    Σ = Rᵀ · diag(s²) · R. We replicate that exactly.

    scale: (N,3) post-activation (already exp'd), R: (N,3,3).
    Returns: (N,3,3) symmetric.
    """
    # M = S · R means M[i, j] = s[i] * R[i, j]  (row scaling)
    M = scale[:, :, None] * R  # (N,3,3)
    Sigma = np.einsum("nki,nkj->nij", M, M)  # M^T @ M per batch
    return Sigma


def compute_cov2d_upper(mean_view: np.ndarray, cov3d: np.ndarray,
                         W_mat: np.ndarray, focal_x: float, focal_y: float,
                         tan_fovx: float, tan_fovy: float) -> np.ndarray:
    """Port of forward_common.h::computeCov2D (lines 73-106).

    CUDA does (glm column-major semantics):
        J = [[fx/tz, 0, -fx·tx/tz²],
             [0, fy/tz, -fy·ty/tz²],
             [0, 0, 0]]
        W = transpose(viewmatrix[:3])   -- i.e. world->camera 3x3
        T = W · J                       -- column-major multiply
        cov = Tᵀ · Vrkᵀ · T

    Translated to row-major numpy, this is the EWA projection.

    Returns the upper-left 2x2 as (N,3) packing (a, b, c) = (cov[0,0],
    cov[0,1], cov[1,1]).
    """
    t = mean_view.copy()
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    tz = t[:, 2]
    txtz = t[:, 0] / tz
    tytz = t[:, 1] / tz
    t[:, 0] = np.clip(txtz, -limx, limx) * tz
    t[:, 1] = np.clip(tytz, -limy, limy) * tz

    N = t.shape[0]
    J = np.zeros((N, 3, 3), dtype=np.float64)
    J[:, 0, 0] = focal_x / tz
    J[:, 0, 2] = -(focal_x * t[:, 0]) / (tz * tz)
    J[:, 1, 1] = focal_y / tz
    J[:, 1, 2] = -(focal_y * t[:, 1]) / (tz * tz)
    # row 2 is zero

    # W_mat here is the world->camera 3x3 in standard row-major (same
    # semantics as glm::transpose(glm::mat3(viewmatrix)) — glm stores column
    # major, so transpose(glm_mat3_from_row_major_viewmatrix) recovers the
    # row-major viewmatrix block, which is what we pass in).
    #
    # CUDA computes T = W · J with column-major product, equivalent to
    # row-major J · W transposed-conventions math; but the final result
    # cov = Tᵀ · Vrk · T is invariant under that detail when Vrk is symmetric.
    # Practically: the end-product cov2D is (J W Σ Wᵀ Jᵀ)[:2,:2].
    WΣ = np.einsum("ij,njk->nik", W_mat, cov3d)          # W · Σ         (3,3)·(N,3,3)
    WΣWt = np.einsum("nij,kj->nik", WΣ, W_mat)           # W · Σ · Wᵀ
    JWΣWt = np.einsum("nij,njk->nik", J, WΣWt)           # J · (W Σ Wᵀ)
    cov = np.einsum("nij,nkj->nik", JWΣWt, J)            # (·) · Jᵀ
    # Upper-left 2x2
    a = cov[:, 0, 0]
    b = cov[:, 0, 1]
    c = cov[:, 1, 1]
    return np.stack([a, b, c], axis=1)


def ndc_to_pix(ndc: np.ndarray, S: int) -> np.ndarray:
    """auxiliary.h:69-72 ndc2Pix: ((v + 1) · S - 1) · 0.5."""
    return ((ndc + 1.0) * S - 1.0) * 0.5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--cam_id", type=int, required=True)
    ap.add_argument("--roi", required=True,
                    help="x_min,y_min,x_max,y_max (inclusive-exclusive, y=0 at top)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pad", type=int, default=2,
                    help="Pad the per-Gaussian AABB by this many pixels (default 2)")
    ap.add_argument("--lowpass", choices=["cuda_classic", "mip_3dgs"], default="cuda_classic",
                    help=("Low-pass filter on screen-space cov: cuda_classic adds 0.3·I "
                          "to the 2x2 diag (matches forward_common.h::dilateCov2D, "
                          "h_var=0.3). mip_3dgs uses a kernel-size-aware adjustment that "
                          "the AAA fork only applies inside the 3D covariance pipeline "
                          "(see forward.cu:149 kernel_size=0.3f and compute_gauss2screen). "
                          "For diagnostic picking, cuda_classic is the conservative choice."))
    ap.add_argument("--opacity_threshold", type=float, default=0.0,
                    help="[Legacy] If >0, also filter out Gaussians whose sigmoid(opacity_raw) < threshold. "
                         "Only affects the count; does not change geometry math. Default 0. "
                         "Prefer --min-opacity for new usage; when both are given --min-opacity wins.")
    ap.add_argument("--min-opacity", dest="min_opacity", type=float, default=0.0,
                    help="After sigmoid activation, drop Gaussians with opacity < min_opacity. "
                         "Default 0.0 (no-op). Populates manifest n_culled_low_opacity.")
    ap.add_argument("--filter", dest="filter_mode", choices=["aabb", "center"], default="aabb",
                    help="Selection rule. aabb (default): keep if projected 2D AABB "
                         "(pad-expanded) intersects the ROI — preserves legacy behavior. "
                         "center: keep if the projected 2D center lies inside the ROI "
                         "expanded by --pad on each side (radius ignored).")
    args = ap.parse_args()

    # --min-opacity takes precedence over legacy --opacity_threshold when set.
    effective_min_opacity = args.min_opacity if args.min_opacity > 0.0 else args.opacity_threshold

    roi = [int(v) for v in args.roi.split(",")]
    if len(roi) != 4:
        raise SystemExit(f"--roi expects 4 comma-separated ints, got {args.roi!r}")
    x_min, y_min, x_max, y_max = roi

    # ---- Camera ----
    with open(args.cameras) as f:
        cams = json.load(f)
    entry = next((c for c in cams if c["id"] == args.cam_id), None)
    if entry is None:
        raise SystemExit(f"cam_id {args.cam_id} not in {args.cameras}")
    cam = build_camera(entry)
    W, H = cam["W"], cam["H"]
    print(f"Camera {args.cam_id}: {W}x{H} fx={cam['fx']:.2f} fy={cam['fy']:.2f}")

    # ---- PLY ----
    header_bytes, header_len, N, props, raw = load_ply_raw(args.ply)
    print(f"PLY: {N} Gaussians, {len(props)} props, stride={raw.shape[1]} bytes")

    xyz = extract_columns(raw, props, ["x", "y", "z"]).astype(np.float64)
    scale_raw = extract_columns(raw, props, ["scale_0", "scale_1", "scale_2"]).astype(np.float64)
    rot_raw = extract_columns(raw, props, ["rot_0", "rot_1", "rot_2", "rot_3"]).astype(np.float64)
    opacity_raw = extract_columns(raw, props, ["opacity"]).astype(np.float64).ravel()

    # ---- Activations (gaussian_model.py:35-43) ----
    scale = np.exp(scale_raw)                                             # exp
    q_norm = rot_raw / np.linalg.norm(rot_raw, axis=1, keepdims=True)     # L2 normalize
    opacity = 1.0 / (1.0 + np.exp(-opacity_raw))                          # sigmoid (for bookkeeping)

    # ---- 3D covariance: Σ = (S·R)ᵀ · (S·R) ---- (forward_common.h:149-171)
    R = quat2mat_wxyz(q_norm)
    cov3d = compute_cov3d(scale, R)

    # ---- Camera-space center + near culling ---- (preprocessCUDA near clip)
    # EVAL_3D path uses mean3D_view.z < 0.2f (forward.cu:136); classical path
    # uses mean3D_view.z < 0.2f (forward.cu:204). znear=0.01 is the projection
    # clipping; we follow the rasterizer culling threshold of 0.2 to match
    # what actually reaches the screen. This is the more conservative (fewer
    # false negatives) cull since 0.2 > 0.01.
    Z_NEAR_CULL = 0.2
    xyz_h = np.concatenate([xyz, np.ones((N, 1), dtype=np.float64)], axis=1)   # (N,4)
    mean_view = (xyz_h @ cam["W2C"].T)[:, :3]                                  # (N,3)
    behind_mask = mean_view[:, 2] <= Z_NEAR_CULL
    n_behind = int(behind_mask.sum())

    # ---- 2D covariance upper-left (computeCov2D + dilateCov2D) ----
    W_mat = cam["W2C"][:3, :3]  # world->camera rotation, row-major
    cov2d_abc = compute_cov2d_upper(
        mean_view, cov3d, W_mat,
        focal_x=cam["fx"], focal_y=cam["fy"],
        tan_fovx=cam["tan_fovx"], tan_fovy=cam["tan_fovy"],
    )
    a = cov2d_abc[:, 0]; b = cov2d_abc[:, 1]; c = cov2d_abc[:, 2]

    if args.lowpass == "cuda_classic":
        # forward_common.h:113-115 — h_var = 0.3f added to diag.
        a = a + 0.3
        c = c + 0.3
    else:  # mip_3dgs
        # The AAA fork applies its Mip-Splatting-style filter *inside* the
        # 3D→screen pipeline (forward.cu:149 kernel_size=0.3f going through
        # compute_gauss2screen). For a CPU proxy we still need SOME low-pass
        # on the 2D cov — emulate by adding kernel_size² (= 0.09) on the diag
        # and document as an approximation. Strictly speaking we over-filter
        # here, keeping MORE Gaussians, which is the cheap direction for
        # diagnostics.
        a = a + 0.09
        c = c + 0.09

    det = a * c - b * b
    # preprocessCUDA lines 241-244:
    #   const float mid = 0.5*(a+c);
    #   const float lambda = mid + sqrt(max(0.01, mid*mid - det));
    #   radius = extent * sqrt(lambda);  with extent = 3.33f (tight_opacity_bounding defaults off)
    mid = 0.5 * (a + c)
    lam = mid + np.sqrt(np.maximum(0.01, mid * mid - det))
    radius_pix = 3.33 * np.sqrt(np.maximum(lam, 0.0))
    # Match radii[idx] = (int)ceil(radius) — forward.cu:313.
    radius_ceil = np.ceil(radius_pix)

    # ---- Screen-space center via full_proj + ndc2Pix ---- (preprocessCUDA:250-252)
    clip = xyz_h @ cam["full_proj"].T  # (N,4)
    w_clip = clip[:, 3] + 1e-7         # world2ndc uses +1e-7 guard
    ndc_x = clip[:, 0] / w_clip
    ndc_y = clip[:, 1] / w_clip
    cx = ndc_to_pix(ndc_x, W)
    cy = ndc_to_pix(ndc_y, H)

    # ---- Selection rule ----
    if args.filter_mode == "aabb":
        # Legacy behavior: keep if projected 2D AABB (radius expanded by --pad)
        # intersects the ROI. Half-open interval [min, max).
        r = radius_ceil + args.pad
        aabb_xmin = cx - r
        aabb_xmax = cx + r
        aabb_ymin = cy - r
        aabb_ymax = cy + r
        in_region = (
            (aabb_xmax >= x_min) &
            (aabb_xmin <  x_max) &
            (aabb_ymax >= y_min) &
            (aabb_ymin <  y_max)
        )
    else:  # "center"
        # Keep if projected 2D center lies inside the ROI expanded by --pad on
        # each side. Radius ignored entirely — this gives a much tighter subset
        # when huge near-camera splats have centers far outside the ROI.
        in_region = (
            (cx >= (x_min - args.pad)) &
            (cx <  (x_max + args.pad)) &
            (cy >= (y_min - args.pad)) &
            (cy <  (y_max + args.pad))
        )

    # Apply near-culling to avoid including Gaussians that CUDA throws out.
    keep = in_region & (~behind_mask)

    # Opacity filter — always computed so manifest n_culled_low_opacity is
    # populated regardless of threshold. When threshold is 0.0, no Gaussian
    # can satisfy opacity < 0.0 (sigmoid is in [0,1]), so count stays 0.
    if effective_min_opacity > 0.0:
        low_op = opacity < effective_min_opacity
    else:
        low_op = np.zeros_like(opacity, dtype=bool)
    # Count culled among the geometrically-in-region, not-behind pool so the
    # number reflects "otherwise kept but opacity-dropped".
    n_low_op = int((low_op & keep).sum())
    keep = keep & (~low_op)

    sel = np.nonzero(keep)[0]
    n_sel = int(sel.size)
    print(f"Selected {n_sel} / {N} ({100.0 * n_sel / N:.3f}%)"
          f"  behind-camera culled: {n_behind}"
          f"  low-opacity culled: {n_low_op}")

    # ---- Write subset .ply ----
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new_header = rewrite_header(header_bytes, n_sel)
    # Slice raw bytes row-wise to preserve exact per-Gaussian encoding.
    subset = raw[sel]  # (n_sel, stride) uint8
    with open(args.out, "wb") as f:
        f.write(new_header)
        f.write(subset.tobytes())
    print(f"Wrote subset -> {args.out} ({os.path.getsize(args.out)} bytes)")

    # ---- Manifest ----
    def _pct(arr, q):
        return float(np.percentile(arr, q)) if arr.size else float("nan")

    sel_depth = mean_view[sel, 2] if n_sel else np.zeros(0)
    sel_op = opacity[sel] if n_sel else np.zeros(0)
    sel_xyz = xyz[sel] if n_sel else np.zeros((0, 3))
    if n_sel:
        bbox_min = [float(v) for v in sel_xyz.min(axis=0)]
        bbox_max = [float(v) for v in sel_xyz.max(axis=0)]
    else:
        bbox_min = [0.0, 0.0, 0.0]
        bbox_max = [0.0, 0.0, 0.0]

    manifest = {
        "source_ply": os.path.abspath(args.ply),
        "roi_pixels": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
        "pad": args.pad,
        "lowpass": args.lowpass,
        "filter_mode": args.filter_mode,
        "min_opacity": float(effective_min_opacity),
        "opacity_threshold": args.opacity_threshold,
        "camera": {
            "cam_id": args.cam_id,
            "img_name": entry.get("img_name"),
            "W": W, "H": H,
            "fx": cam["fx"], "fy": cam["fy"],
            "fovx_deg": math.degrees(cam["fovx"]),
            "fovy_deg": math.degrees(cam["fovy"]),
        },
        "n_total": int(N),
        "n_selected": n_sel,
        "n_culled_behind_camera": int(n_behind),
        "n_culled_low_opacity": int(n_low_op),
        "depth_pct": {"p5": _pct(sel_depth, 5), "p50": _pct(sel_depth, 50), "p95": _pct(sel_depth, 95)},
        "opacity_pct": {"p5": _pct(sel_op, 5), "p50": _pct(sel_op, 50), "p95": _pct(sel_op, 95)},
        "screen_center_x_pct_selected": {
            "p5": _pct(cx[sel], 5) if n_sel else float("nan"),
            "p50": _pct(cx[sel], 50) if n_sel else float("nan"),
            "p95": _pct(cx[sel], 95) if n_sel else float("nan"),
        },
        "screen_center_y_pct_selected": {
            "p5": _pct(cy[sel], 5) if n_sel else float("nan"),
            "p50": _pct(cy[sel], 50) if n_sel else float("nan"),
            "p95": _pct(cy[sel], 95) if n_sel else float("nan"),
        },
        "radius_pix_pct_selected": {
            "p5": _pct(radius_pix[sel], 5) if n_sel else float("nan"),
            "p50": _pct(radius_pix[sel], 50) if n_sel else float("nan"),
            "p95": _pct(radius_pix[sel], 95) if n_sel else float("nan"),
        },
        "world_bbox": {"min": bbox_min, "max": bbox_max},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
