#!/usr/bin/env python3
"""render_cuda_basketball.py — Generate CUDA eval_3D golden for basket-aaa.ply @ cam 0.

Loads a trained 3DGS PLY, activates raw parameters (exp scales, sigmoid opacities,
normalized quaternions), loads per-Gaussian filter_3D from the baked PLY property,
renders with AAA-Gaussians' eval_3D=True rasterizer, and writes:
  - HWC float32 little-endian raw  -> <out>/cuda_image.raw
  - CHW float32 npy                -> <out>/cuda_image.npy
  - SHA256 hash of inputs          -> <out>/golden_hash.txt

Default paths match the design in
docs/superpowers/specs/2026-04-21-vk-cuda-parity-60db-design.md.

Usage:
    /home/robota/miniconda3/envs/aaa-gs/bin/python3.10 \\
        tools/render_cuda_basketball.py \\
        [--ply /home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/vulkan_3d/basket-aaa.ply] \\
        [--cameras /home/robota/Downloads/basketball/_sp0_dump_output/cameras.json] \\
        [--cam-id 0] \\
        [--out tests/golden/basketball/cam0] \\
        [--force]
"""

import argparse
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, '/home/robota/h00813233/Graph/AAA-Gaussians/submodules/diff-gaussian-rasterization')

import numpy as np
import torch

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings, GaussianRasterizer, ExtendedSettings
)

REPO = '/home/robota/h00813233/Graph/aaags-claude/.claude/worktrees/vulkan_3d'
DEFAULT_PLY = os.path.join(REPO, 'basket-aaa.ply')
DEFAULT_CAMS = '/home/robota/Downloads/basketball/_sp0_dump_output/cameras.json'
DEFAULT_OUT = os.path.join(REPO, 'harmonyos_3dgs/tests/golden/basketball/cam0')


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_ply_aaa(path: str) -> dict:
    """Load a trained AAA-Gaussians PLY. Returns raw (un-activated) tensors
    plus per-Gaussian filter_3D (AAA-variant PLYs bake filter_3D into the
    vertex properties as property `filter_3D`).
    """
    with open(path, 'rb') as f:
        props, N = [], 0
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if line.startswith('element vertex'):
                N = int(line.split()[-1])
            if line.startswith('property float'):
                props.append(line.split()[-1])
            if line == 'end_header':
                break
        data = np.frombuffer(f.read(N * len(props) * 4),
                             dtype=np.float32).reshape(N, len(props))
    idx = {p: i for i, p in enumerate(props)}
    positions = data[:, [idx['x'], idx['y'], idx['z']]].copy()
    scales_raw = data[:, [idx['scale_0'], idx['scale_1'], idx['scale_2']]].copy()
    rotations_raw = data[:, [idx['rot_0'], idx['rot_1'],
                             idx['rot_2'], idx['rot_3']]].copy()
    opacities_raw = data[:, idx['opacity']].copy()
    f_dc = data[:, [idx['f_dc_0'], idx['f_dc_1'], idx['f_dc_2']]]  # [N, 3]
    rest_names = sorted([p for p in props if p.startswith('f_rest_')],
                        key=lambda n: int(n.split('_')[-1]))
    n_rest = len(rest_names)
    max_coeffs = 1 + n_rest // 3
    sh_degree = int(round(math.sqrt(max_coeffs))) - 1
    f_rest = data[:, [idx[n] for n in rest_names]]  # [N, n_rest]
    sh = np.zeros((N, max_coeffs, 3), dtype=np.float32)
    sh[:, 0, :] = f_dc
    f_rest_r = f_rest.reshape(N, 3, max_coeffs - 1).transpose(0, 2, 1)
    sh[:, 1:, :] = f_rest_r
    if 'filter_3D' not in idx:
        raise RuntimeError(
            f'PLY {path} missing `filter_3D` property. This plan assumes the '
            'AAA-variant PLY with baked filter_3D. Either re-export the PLY '
            'with filter_3D or revise the harness to compute it.')
    filter_3D = data[:, idx['filter_3D']].copy()  # [N]
    return dict(N=N, sh_degree=sh_degree, max_coeffs=max_coeffs,
                positions=positions, scales_raw=scales_raw,
                rotations_raw=rotations_raw, opacities_raw=opacities_raw,
                sh=sh, filter_3D=filter_3D)


def activate(pply: dict) -> dict:
    """Apply standard 3DGS activations."""
    scales = np.exp(pply['scales_raw'])
    rot = pply['rotations_raw']
    rot_norm = rot / (np.linalg.norm(rot, axis=1, keepdims=True) + 1e-30)
    opacities = 1.0 / (1.0 + np.exp(-pply['opacities_raw']))
    return dict(positions=pply['positions'], scales=scales, rotations=rot_norm,
                opacities=opacities[:, None], sh=pply['sh'],
                sh_degree=pply['sh_degree'], N=pply['N'],
                filter_3D=pply['filter_3D'])


def load_camera(cam_path: str, cam_id: int) -> dict:
    with open(cam_path) as f:
        cams = json.load(f)
    cam = next(c for c in cams if c['id'] == cam_id)
    W, H = int(cam['width']), int(cam['height'])
    fx, fy = float(cam['fx']), float(cam['fy'])
    tan_fovx = W / (2.0 * fx)
    tan_fovy = H / (2.0 * fy)
    R = np.array(cam['rotation'], dtype=np.float32)            # [3,3]
    T = -R @ np.array(cam['position'], dtype=np.float32)       # [3]
    W2C = np.eye(4, dtype=np.float32)
    W2C[:3, :3] = R
    W2C[:3, 3] = T
    viewmatrix = W2C.T.astype(np.float32)
    znear, zfar = 0.01, 100.0
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 1.0 / tan_fovx
    proj[1, 1] = 1.0 / tan_fovy
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    proj[3, 2] = 1.0
    projmatrix = (viewmatrix @ proj).astype(np.float32)
    inv_viewprojmatrix = np.linalg.inv(projmatrix).astype(np.float32)
    campos = np.array(cam['position'], dtype=np.float32)
    return dict(W=W, H=H, tan_fovx=tan_fovx, tan_fovy=tan_fovy,
                viewmatrix=viewmatrix, projmatrix=projmatrix,
                inv_viewprojmatrix=inv_viewprojmatrix, campos=campos)


def render_eval3d(act: dict, cam: dict, filter_3D: np.ndarray) -> np.ndarray:
    """Call AAA-Gaussians with eval_3D=True. Returns [3, H, W] float32 CHW."""
    dev = 'cuda'
    means3D = torch.tensor(act['positions'], dtype=torch.float32, device=dev)
    scales = torch.tensor(act['scales'], dtype=torch.float32, device=dev)
    rotations = torch.tensor(act['rotations'], dtype=torch.float32, device=dev)
    opacities = torch.tensor(act['opacities'], dtype=torch.float32, device=dev)
    sh = torch.tensor(act['sh'], dtype=torch.float32, device=dev)
    f3d = torch.tensor(filter_3D.reshape(-1, 1), dtype=torch.float32, device=dev)
    vm = torch.tensor(cam['viewmatrix'], dtype=torch.float32, device=dev)
    pm = torch.tensor(cam['projmatrix'], dtype=torch.float32, device=dev)
    inv_vp = torch.tensor(cam['inv_viewprojmatrix'],
                          dtype=torch.float32, device=dev)
    campos = torch.tensor(cam['campos'], dtype=torch.float32, device=dev)
    bg = torch.zeros(3, dtype=torch.float32, device=dev)

    splat_args = ExtendedSettings()
    splat_args.eval_3D = True
    splat_args.proper_ewa_scaling = False  # matches VK preprocess.comp:959 comment

    raster_settings = GaussianRasterizationSettings(
        image_height=cam['H'], image_width=cam['W'],
        tanfovx=cam['tan_fovx'], tanfovy=cam['tan_fovy'],
        bg=bg, scale_modifier=1.0,
        viewmatrix=vm, projmatrix=pm, inv_viewprojmatrix=inv_vp,
        sh_degree=act['sh_degree'], campos=campos,
        prefiltered=False, settings=splat_args,
        render_depth=False, debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means2D = torch.zeros_like(means3D, requires_grad=True)

    with torch.no_grad():
        rendered, _radii = rasterizer(
            means3D=means3D, means2D=means2D, shs=sh,
            colors_precomp=None, opacities=opacities,
            scales=scales, rotations=rotations,
            filter3D=f3d, cov3D_precomp=None,
        )
    return rendered.detach().cpu().numpy().astype(np.float32)  # [3, H, W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', default=DEFAULT_PLY)
    ap.add_argument('--cameras', default=DEFAULT_CAMS)
    ap.add_argument('--cam-id', type=int, default=0)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    raw_path = os.path.join(args.out, 'cuda_image.raw')
    npy_path = os.path.join(args.out, 'cuda_image.npy')
    hash_path = os.path.join(args.out, 'golden_hash.txt')

    this_script = os.path.abspath(__file__)
    input_hash = hashlib.sha256(
        (sha256_of_file(args.ply) + sha256_of_file(args.cameras)
         + sha256_of_file(this_script) + str(args.cam_id)).encode()
    ).hexdigest()

    if (not args.force and os.path.exists(hash_path)
            and open(hash_path).read().strip() == input_hash):
        print(f'[cached] golden up-to-date: {raw_path}')
        return 0

    print(f'[load ] {args.ply}')
    pply = load_ply_aaa(args.ply)
    print(f'  N={pply["N"]}, sh_degree={pply["sh_degree"]}')
    act = activate(pply)
    cam = load_camera(args.cameras, args.cam_id)
    print(f'[cam  ] cam_id={args.cam_id} W={cam["W"]} H={cam["H"]}')
    filter_3D = act['filter_3D']
    print(f'[f3D  ] (from PLY) min={filter_3D.min():.6f} '
          f'mean={filter_3D.mean():.6f} max={filter_3D.max():.6f}')
    img_chw = render_eval3d(act, cam, filter_3D)  # [3, H, W]
    img_hwc = np.transpose(img_chw, (1, 2, 0)).copy()  # [H, W, 3]
    print(f'[img  ] shape CHW={img_chw.shape} HWC={img_hwc.shape} '
          f'range=[{img_hwc.min():.4f}, {img_hwc.max():.4f}]')

    img_hwc.astype('<f4').tofile(raw_path)
    np.save(npy_path, img_chw)
    with open(hash_path, 'w') as f:
        f.write(input_hash + '\n')
    print(f'[save ] {raw_path} ({os.path.getsize(raw_path)} bytes)')
    print(f'[save ] {npy_path}')
    print(f'[save ] {hash_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
