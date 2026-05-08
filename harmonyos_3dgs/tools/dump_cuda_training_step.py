#!/usr/bin/env python3
"""Phase 0.3 — dump ONE CUDA forward+L1 step on basket-aaa.ply / cam0 for
VK-vs-CUDA first-loss parity testing.

Writes 18 artifacts to
`harmonyos_3dgs/tests/golden/basketball/cuda_ref/step_0001/`:

    1  view_matrix.npy                (4,4) float32 column-major W2C (VK layout)
    2  proj_matrix.npy                (4,4) float32 column-major full_proj (VK layout)
    3  cam_position.npy               (3,)  float32 world-space camera center
    4  tan_fov.npy                    (2,)  float32 [tan_fovx, tan_fovy]
    5  raw_params.npz                 raw_positions, raw_scales, raw_rotations,
                                      raw_opacities, raw_sh_coeffs (N,16,3)
    6  gt_image.npy                   (3,H,W) float32 in [0,1]
    7  rendered_image.npy             (3,H,W) float32 CUDA render output
    8  means2D.npy                    (N,2)  float32 screen-space
    9  conic_opacity.npy              (N,4)  float32 (a,b,c,opacity)
   10  rgb_colors.npy                 (N,3)  float32 SH-evaluated base colors
   11  radii.npy                      (N,)   int32 per-Gaussian screen radius
   12  tiles_touched.npy              (N,)   int32 per-Gaussian tile count
   13  depths.npy                     (N,)   float32 per-Gaussian sort depth
   14  T_final.npy                    (H,W)  float32 per-pixel final transmittance
   15  n_contrib.npy                  (H,W)  int32 per-pixel contribution count
   16  sorted_ids_per_tile.npy        (R,)   int32 flattened per-tile Gaussian IDs
       tile_offsets.npy               (num_tiles+1,) int32 start offsets + total
   17  l1_loss.npy                    (1,)   float32 scalar L1(render - gt)
   18  meta.json                      config, paths, layouts, N, W, H, etc.

Run with:
    conda run -n aaa-gs python harmonyos_3dgs/tools/dump_cuda_training_step.py

Parity config (forced, NOT aaa.json defaults):
    eval_3D=True, sort_mode=0 (GLOBAL, single-level), near_clipping=False,
    lambda_dssim=0.0, background=(0,0,0), sh_degree_max=3.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image


# ----------------------------------------------------------------------------
# Locate repo roots & import AAA-Gaussians
# ----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))               # harmonyos_3dgs
WORKTREE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, os.pardir))            # training
CLAUDE_ROOT = os.path.abspath(os.path.join(WORKTREE_ROOT, os.pardir, os.pardir))
AAA_ROOTS = [
    os.path.join(CLAUDE_ROOT, "AAA-Gaussians"),
    "/home/robota/h00813233/Graph/aaags-claude/AAA-Gaussians",
]
for p in AAA_ROOTS:
    if os.path.isdir(p):
        sys.path.insert(0, p)
        break

from diff_gaussian_rasterization import (  # noqa: E402
    GaussianRasterizationSettings,
    ExtendedSettings,
    _C,
)
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.cameras import MiniCam               # noqa: E402
from utils.graphics_utils import getWorld2View2, getProjectionMatrix  # noqa: E402


# ----------------------------------------------------------------------------
# Camera builder (mirrors dump_cuda_ncontrib.camera_from_json_entry).
# Returns MiniCam plus the intermediate matrices, so we can dump both VK and
# CUDA-native layouts for Gate I1.
# ----------------------------------------------------------------------------
def camera_from_json_entry(entry, device="cuda"):
    rot_c2w = np.array(entry["rotation"], dtype=np.float64)       # 3x3
    pos_c2w = np.array(entry["position"], dtype=np.float64)       # camera center
    C2W = np.eye(4, dtype=np.float64)
    C2W[:3, :3] = rot_c2w
    C2W[:3, 3] = pos_c2w
    W2C = np.linalg.inv(C2W)                                      # mathematical W2C (row-major)
    R = W2C[:3, :3].T.astype(np.float32)                          # R == R_c2w
    T = W2C[:3, 3].astype(np.float32)
    W, H, fx, fy = int(entry["width"]), int(entry["height"]), float(entry["fx"]), float(entry["fy"])
    fovx = 2.0 * math.atan(W / (2.0 * fx))
    fovy = 2.0 * math.atan(H / (2.0 * fy))

    # getWorld2View2(R, T) returns numpy W2C (row-major). .transpose(0,1) is the
    # DGR convention (matrix fed to the kernel is W2C.T in row-major storage,
    # which equals W2C in column-major storage — byte-identical to VK).
    world_view_transform = (
        torch.tensor(getWorld2View2(R, T), dtype=torch.float32).transpose(0, 1).to(device)
    )
    projection_matrix = (
        getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy)
        .transpose(0, 1).to(device)
    )
    full_proj = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)

    mini = MiniCam(
        width=W, height=H, fovy=fovy, fovx=fovx, znear=0.01, zfar=100.0,
        world_view_transform=world_view_transform, full_proj_transform=full_proj,
    )
    return mini, W2C.astype(np.float32)  # also return "canonical" W2C (row-major)


# ----------------------------------------------------------------------------
# Image load: COLMAP GT → (3,H,W) float32 in [0,1]
# ----------------------------------------------------------------------------
def load_gt(path, W, H):
    img = Image.open(path).convert("RGB")
    orig_size = img.size
    resized = False
    if img.size != (W, H):
        img = img.resize((W, H), Image.BILINEAR)
        resized = True
    arr = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3)
    arr = arr.transpose(2, 0, 1).copy()               # (3, H, W) CHW
    return arr, orig_size, resized


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default="/home/robota/h00813233/Graph/aaags-claude/basket-aaa.ply")
    ap.add_argument("--cameras",
                    default="/home/robota/Downloads/basketball/_sp0_dump_output/cameras.json")
    ap.add_argument("--images_dir", default="/home/robota/Downloads/basketball/images")
    ap.add_argument("--cam_id", type=int, default=0)
    ap.add_argument("--out_dir",
                    default=os.path.join(REPO_ROOT, "tests", "golden", "basketball",
                                         "cuda_ref", "step_0001"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[0/8] Output dir: {args.out_dir}")

    # -----------------------------------------------------------------------
    # 1. Load Gaussians
    # -----------------------------------------------------------------------
    print(f"[1/8] Loading PLY: {args.ply}")
    t0 = time.time()
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(args.ply)
    N = int(gaussians.get_xyz.shape[0])
    has_filter = gaussians.filter_3D is not None
    print(f"       N={N:,}  sh_degree={gaussians.active_sh_degree}  filter_3D={has_filter}"
          f"  ({time.time()-t0:.2f}s)")

    # -----------------------------------------------------------------------
    # 2. Load camera & GT image
    # -----------------------------------------------------------------------
    print(f"[2/8] Loading camera {args.cam_id} and GT image...")
    with open(args.cameras) as f:
        cams_j = json.load(f)
    entry = cams_j[args.cam_id]
    assert entry["id"] == args.cam_id, f"id mismatch: {entry['id']}"
    img_name = entry["img_name"]
    gt_path = os.path.join(args.images_dir, img_name + ".jpg")
    if not os.path.isfile(gt_path):
        # Try other extensions
        for ext in (".png", ".jpeg", ".JPG"):
            alt = os.path.join(args.images_dir, img_name + ext)
            if os.path.isfile(alt):
                gt_path = alt
                break
    cam, W2C_canon = camera_from_json_entry(entry)
    W, H = int(cam.image_width), int(cam.image_height)
    print(f"       cam{args.cam_id}: {W}x{H}  img_name={img_name}  gt={gt_path}")

    gt_chw, orig_size, resized = load_gt(gt_path, W, H)
    print(f"       GT original {orig_size} → ({W},{H}) resize={resized}")

    # -----------------------------------------------------------------------
    # 3. Build rasterizer settings (PARITY CONFIG — overrides aaa.json)
    # -----------------------------------------------------------------------
    parity_config_dict = {
        "eval_3D": True,
        "sort_settings": {
            "sort_mode": 0,   # GLOBAL
            "sort_order": 0,  # VIEWSPACE_Z
            "queue_sizes": {
                "tile_4x4": 64,
                "tile_2x2": 8,
                "per_pixel": 4,
            },
        },
        "culling_settings": {
            "rect_bounding": False,
            "tight_opacity_bounding": False,
            "tile_based_culling": False,
            "hierarchical_4x4_culling": False,
        },
        "load_balancing": False,
        "proper_ewa_scaling": False,
    }
    splat_args = ExtendedSettings.from_dict(parity_config_dict)
    splat_dict = splat_args.to_dict()
    # VK preprocess.comp hardcodes new_aabb=true (unconditional computeAABBView).
    # Match VK's production path; this is also the CUDA C++ default (rasterizer.h:140).
    splat_dict["new_aabb"] = True

    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    tan_fovx = math.tan(cam.FoVx * 0.5)
    tan_fovy = math.tan(cam.FoVy * 0.5)

    # -----------------------------------------------------------------------
    # 4. Forward pass via low-level _C.rasterize_gaussians (matches fwd_args
    #    tuple in _RasterizeGaussians.forward, so we can pull raw buffers).
    # -----------------------------------------------------------------------
    print("[3/8] Rasterizing (forward)...")
    means3D  = gaussians.get_xyz
    opacity  = gaussians.get_opacity
    scales   = gaussians.get_scaling
    rotations = gaussians.get_rotation
    shs      = gaussians.get_features            # (N,16,3) torch
    filter3D = gaussians.filter_3D if gaussians.filter_3D is not None else torch.Tensor([]).cuda()

    empty = torch.Tensor([]).to("cuda")
    colors_precomp = empty
    cov3Ds_precomp = empty

    campos = cam.world_view_transform.inverse()[3, :3]
    inv_proj = cam.full_proj_transform.inverse()

    fwd_args = (
        background,
        means3D,
        colors_precomp,
        opacity,
        scales,
        rotations,
        filter3D,
        1.0,                         # scale_modifier
        cov3Ds_precomp,
        cam.world_view_transform,
        cam.full_proj_transform,
        inv_proj,
        tan_fovx,
        tan_fovy,
        H, W,
        shs,
        int(gaussians.active_sh_degree),
        campos,
        False,                       # prefiltered
        splat_dict,
        False,                       # render_depth
        False,                       # debug
    )

    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.time()
        num_rendered, color, radii, geomBuf, binBuf, imgBuf = _C.rasterize_gaussians(*fwd_args)
        torch.cuda.synchronize()
    R = int(num_rendered)
    print(f"       forward: {time.time()-t0:.2f}s  R={R}  color.shape={tuple(color.shape)}")

    # -----------------------------------------------------------------------
    # 5. Materialize per-Gaussian and per-pixel buffers
    # -----------------------------------------------------------------------
    num_tiles_x = (W + 15) // 16
    num_tiles_y = (H + 15) // 16
    num_tiles = num_tiles_x * num_tiles_y
    print(f"[4/8] Materialize dump  P={N}  R={R}  tiles={num_tiles_x}x{num_tiles_y}={num_tiles}")

    # requires_cov3D_inv is true in this diagnostic build so the CUDA extension
    # can materialize eval_3D AABB trace scalars through the otherwise-unused buffer.
    requires_cov3D_inv = True
    requires_gauss2screen = True
    dump = _C.materialize_dump(
        geomBuf, binBuf, imgBuf,
        int(N), int(R), int(num_tiles), int(H), int(W),
        requires_cov3D_inv,
        requires_gauss2screen,
    )
    print(f"       materialize keys: {sorted(dump.keys())}")

    means2D        = dump["preprocess_means2D"].cpu().contiguous().numpy().astype(np.float32)       # (N,2)
    conic_opacity  = dump["preprocess_conic_opacity"].cpu().contiguous().numpy().astype(np.float32)  # (N,4)
    rgb_colors     = dump["preprocess_rgb"].cpu().contiguous().numpy().astype(np.float32)           # (N,3)
    tiles_touched  = dump["preprocess_tiles_touched"].cpu().contiguous().numpy().astype(np.int32)   # (N,)
    depths         = dump["preprocess_depths"].cpu().contiguous().numpy().astype(np.float32)         # (N,)
    rects2D        = (dump["preprocess_rects2D"].cpu().contiguous().numpy().astype(np.float32)
                      if "preprocess_rects2D" in dump else None)                                    # (N,2)
    gauss2screen   = (dump["preprocess_gauss2screen"].cpu().contiguous().numpy().astype(np.float32)
                      if "preprocess_gauss2screen" in dump else None)                               # (N,16)
    aabb_debug     = (dump["preprocess_aabb_debug"].cpu().contiguous().numpy().astype(np.float32)
                      if "preprocess_aabb_debug" in dump else None)                                 # (N,3,4)
    n_contrib      = dump["rasterize_n_contrib"].cpu().contiguous().numpy().astype(np.int32).reshape(H, W)
    T_final        = dump["rasterize_transmittance"].cpu().contiguous().numpy().astype(np.float32).reshape(H, W)

    tile_ranges    = dump["sort_tile_ranges"].cpu().contiguous().numpy().astype(np.int64)  # (num_tiles, 2)  [start,end)
    sorted_values  = (dump["sort_values_sorted"].cpu().contiguous().numpy().astype(np.int64)
                      if "sort_values_sorted" in dump else np.zeros((0,), dtype=np.int64))

    # Build tile_offsets: (num_tiles+1,) where offsets[i] = tile_ranges[i,0] and
    # offsets[num_tiles] = R. Validate contiguity.
    tile_offsets = np.zeros((num_tiles + 1,), dtype=np.int32)
    tile_offsets[:num_tiles] = tile_ranges[:, 0].astype(np.int32)
    tile_offsets[num_tiles] = R
    # quick sanity: tile_ranges[i,1] should match tile_offsets[i+1] for contiguous layout
    # (stopthepop GLOBAL layout: range [start,end) is contiguous across tiles).
    contig_ok = bool(np.all(tile_ranges[:-1, 1] == tile_ranges[1:, 0])) if num_tiles > 1 else True

    radii_np = radii.detach().cpu().contiguous().numpy().astype(np.int32)                           # (N,)
    rendered = color.detach().cpu().contiguous().numpy().astype(np.float32)                         # (3,H,W)

    # -----------------------------------------------------------------------
    # 6. L1 loss
    # -----------------------------------------------------------------------
    diff = rendered - gt_chw
    l1 = float(np.mean(np.abs(diff)))
    print(f"[5/8] L1 loss = {l1:.6f}")

    # -----------------------------------------------------------------------
    # 7. Raw (pre-activation) params
    # -----------------------------------------------------------------------
    print("[6/8] Collecting raw (pre-activation) params...")
    raw_positions = gaussians._xyz.detach().cpu().contiguous().numpy().astype(np.float32)           # (N,3)
    raw_scales    = gaussians._scaling.detach().cpu().contiguous().numpy().astype(np.float32)       # (N,3)
    raw_rotations = gaussians._rotation.detach().cpu().contiguous().numpy().astype(np.float32)      # (N,4) (r,x,y,z)
    raw_opacities = gaussians._opacity.detach().cpu().contiguous().numpy().astype(np.float32).reshape(-1)  # (N,)
    # get_features returns (N, 16, 3) in [gaussian, coeff, channel] order.
    raw_sh_coeffs = gaussians.get_features.detach().cpu().contiguous().numpy().astype(np.float32)   # (N,16,3)

    # -----------------------------------------------------------------------
    # 8. Matrix layouts — VK vs CUDA native
    # -----------------------------------------------------------------------
    # CUDA native: the tensor we hand to DGR is `world_view_transform` = W2C.T (row-major)
    # which is *byte-identical* to W2C in column-major storage (= VK layout).
    # We therefore dump BOTH under clearly different names.
    view_matrix_vk   = cam.world_view_transform.detach().cpu().contiguous().numpy().astype(np.float32)  # (4,4) — DGR-side numpy = column-major W2C
    proj_matrix_vk   = cam.full_proj_transform.detach().cpu().contiguous().numpy().astype(np.float32)   # (4,4) — column-major full_proj
    view_matrix_cuda = view_matrix_vk.copy()    # same bytes
    proj_matrix_cuda = proj_matrix_vk.copy()
    # Also provide the canonical mathematical W2C (row-major, [row,col]) for assertion helpers.
    view_matrix_canonical_row_major = W2C_canon.astype(np.float32)  # (4,4) W2C with [row,col] indexing

    cam_position = campos.detach().cpu().contiguous().numpy().astype(np.float32)   # (3,)
    tan_fov_np   = np.array([tan_fovx, tan_fovy], dtype=np.float32)

    # -----------------------------------------------------------------------
    # 9. Write outputs
    # -----------------------------------------------------------------------
    print(f"[7/8] Writing outputs to {args.out_dir}")
    out_files = {}

    def save(name, arr):
        p = os.path.join(args.out_dir, name)
        np.save(p, arr)
        out_files[name] = os.path.getsize(p)

    save("view_matrix.npy", view_matrix_vk)
    save("view_matrix_cuda_native.npy", view_matrix_cuda)
    save("view_matrix_canonical_row_major.npy", view_matrix_canonical_row_major)
    save("proj_matrix.npy", proj_matrix_vk)
    save("proj_matrix_cuda_native.npy", proj_matrix_cuda)
    save("cam_position.npy", cam_position)
    save("tan_fov.npy", tan_fov_np)

    # raw_params.npz
    p = os.path.join(args.out_dir, "raw_params.npz")
    np.savez(p,
             raw_positions=raw_positions,
             raw_scales=raw_scales,
             raw_rotations=raw_rotations,
             raw_opacities=raw_opacities,
             raw_sh_coeffs=raw_sh_coeffs)
    out_files["raw_params.npz"] = os.path.getsize(p)

    save("gt_image.npy", gt_chw)
    save("rendered_image.npy", rendered)
    save("means2D.npy", means2D)
    save("conic_opacity.npy", conic_opacity)
    save("rgb_colors.npy", rgb_colors)
    save("radii.npy", radii_np)
    save("tiles_touched.npy", tiles_touched)
    save("depths.npy", depths)
    if rects2D is not None:
        save("rects2D.npy", rects2D)
    if gauss2screen is not None:
        save("gauss2screen.npy", gauss2screen)
    if aabb_debug is not None:
        save("aabb_debug.npy", aabb_debug)
    save("T_final.npy", T_final)
    save("n_contrib.npy", n_contrib)
    save("sorted_ids_per_tile.npy", sorted_values.astype(np.int32))
    save("tile_offsets.npy", tile_offsets)
    save("l1_loss.npy", np.array([l1], dtype=np.float32))

    # meta.json (item 18)
    meta = {
        "phase": "0.3",
        "description": "CUDA reference dump for VK-vs-CUDA first-loss parity",
        "inputs": {
            "ply": os.path.abspath(args.ply),
            "cameras_json": os.path.abspath(args.cameras),
            "gt_image": os.path.abspath(gt_path),
            "gt_image_original_size": list(orig_size),
            "gt_image_resized_to": [W, H],
            "gt_image_resize_algo": "PIL.Image.BILINEAR" if resized else "none",
            "cam_id": args.cam_id,
            "img_name": img_name,
        },
        "dims": {
            "N": N,
            "R": R,
            "W": W,
            "H": H,
            "tile_w": 16,
            "tile_h": 16,
            "num_tiles_x": num_tiles_x,
            "num_tiles_y": num_tiles_y,
            "num_tiles": num_tiles,
            "sh_degree_active": int(gaussians.active_sh_degree),
            "sh_max_coeffs": 16,
        },
        "parity_config": {
            "eval_3D": True,
            "sort_mode": 0,
            "sort_order": 0,
            "near_clipping": False,
            "lambda_dssim": 0.0,
            "background": [0.0, 0.0, 0.0],
            "sh_degree_max": 3,
            "new_aabb": True,
            "scale_modifier": 1.0,
            "antialiasing": False,
            "note": ("Forces single-level GLOBAL sort (sort_mode=0). Uses "
                     "new_aabb=true (CUDA C++ default + VK production path). "
                     "Overrides aaa.json sort_mode=3. filter_3D passed through "
                     "if PLY had it."),
            "filter_3D_present": bool(has_filter),
        },
        "raster_inputs": {
            "tan_fovx": tan_fovx,
            "tan_fovy": tan_fovy,
            "znear": 0.01,
            "zfar": 100.0,
            "scale_modifier": 1.0,
            "prefiltered": False,
            "render_depth": False,
            "debug": False,
        },
        "matrix_layouts": {
            "view_matrix.npy": {
                "shape": [4, 4],
                "dtype": "float32",
                "convention": (
                    "row-major numpy storage of cam.world_view_transform "
                    "(the exact tensor CUDA receives). Because numpy row-major "
                    "storage of a matrix X is byte-identical to column-major "
                    "storage of X.T, and this tensor IS W2C.T, the file is "
                    "byte-identical to 'column-major W2C' — which is what VK's "
                    "loadCameraJson writes into Camera.view_matrix[16]. "
                    "Use np.load(...).reshape(-1) for VK vec4x4 comparison."
                ),
            },
            "view_matrix_cuda_native.npy": {
                "shape": [4, 4],
                "convention": "Same bytes as view_matrix.npy; alias for clarity.",
            },
            "view_matrix_canonical_row_major.npy": {
                "shape": [4, 4],
                "convention": (
                    "Mathematical W2C with standard [row, col] indexing "
                    "(row-major). Transpose of view_matrix.npy viewed as a "
                    "mathematical matrix. Use this if an assertion wants "
                    "W2C @ p_world to give p_view directly."
                ),
            },
            "proj_matrix.npy": {
                "shape": [4, 4],
                "convention": (
                    "row-major numpy storage of cam.full_proj_transform "
                    "(=world_view_transform @ projection_matrix, each "
                    "pre-transposed). Byte-identical to 'column-major "
                    "viewproj = proj * view' — matches VK Camera.viewproj_matrix."
                ),
            },
            "proj_matrix_cuda_native.npy": {
                "shape": [4, 4], "convention": "Alias of proj_matrix.npy."
            },
            "cam_position.npy": {"shape": [3], "convention": "World-space camera center."},
            "tan_fov.npy": {"shape": [2], "convention": "[tan_fovx, tan_fovy]"},
        },
        "outputs": {
            "view_matrix.npy": "col-major W2C",
            "proj_matrix.npy": "col-major full_proj",
            "cam_position.npy": "(3,) float32 cam center world-space",
            "tan_fov.npy": "(2,) [tan_fovx, tan_fovy]",
            "raw_params.npz": ("raw_positions(N,3), raw_scales(N,3), "
                               "raw_rotations(N,4) quat (r,x,y,z), "
                               "raw_opacities(N,), raw_sh_coeffs(N,16,3) "
                               "[gaussian, coeff, channel] (CUDA native; "
                               "note VK ply_loader uses channel-major-rest "
                               "so a transpose is needed on VK side)."),
            "gt_image.npy": "(3,H,W) float32 in [0,1]",
            "rendered_image.npy": "(3,H,W) float32 CUDA render output",
            "means2D.npy": "(N,2) pixel coordinates, center of image = (W/2, H/2)",
            "conic_opacity.npy": "(N,4) (conic.a, conic.b, conic.c, opacity)",
            "rgb_colors.npy": "(N,3) SH-evaluated RGB before clamp-min in kernel",
            "radii.npy": "(N,) int32 per-Gaussian screen-space radius",
            "tiles_touched.npy": "(N,) int32 per-Gaussian tile touch count",
            "depths.npy": "(N,) float32 per-Gaussian sort depth",
            "T_final.npy": "(H,W) float32 per-pixel final transmittance (accum_alpha)",
            "n_contrib.npy": "(H,W) int32 per-pixel contribution count",
            "sorted_ids_per_tile.npy": ("(R,) int32 flattened Gaussian IDs in "
                                        "render order, concatenated per tile."),
            "tile_offsets.npy": ("(num_tiles+1,) int32 start offsets; "
                                 "tile i's IDs are sorted_ids_per_tile[offsets[i]:offsets[i+1]]. "
                                 "Last entry = R."),
            "l1_loss.npy": "(1,) float32 L1(render - gt).mean() over all CHW pixels",
        },
        "sort_layout_sanity": {
            "tile_ranges_contiguous": contig_ok,
            "note": ("For sort_mode=0 (GLOBAL), stopthepop uses contiguous "
                     "per-tile ranges. Contiguity checked above. If False, "
                     "tile_offsets still accurately indexes each tile's "
                     "segment but gaps between tiles are possible."),
        },
        "l1_loss": l1,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    out_files["meta.json"] = os.path.getsize(os.path.join(args.out_dir, "meta.json"))

    # -----------------------------------------------------------------------
    print("[8/8] Done. Artifacts:")
    for k in sorted(out_files):
        print(f"       {k:40s} {out_files[k]:>14} bytes")
    print(f"\nL1 loss (VK parity target): {l1:.6f}")


if __name__ == "__main__":
    main()
