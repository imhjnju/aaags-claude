#!/usr/bin/env python3
"""Prepare flowers dataset with proper initial Gaussian parameters for C++ training."""

import numpy as np
import struct, json, os, subprocess
from scipy.spatial import KDTree

def read_points3D_binary(path):
    points = []
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            pt_id = struct.unpack('<Q', f.read(8))[0]
            xyz = struct.unpack('<3d', f.read(24))
            rgb = struct.unpack('<3B', f.read(3))
            error = struct.unpack('<d', f.read(8))[0]
            tl = struct.unpack('<Q', f.read(8))[0]
            f.read(tl * 8)
            points.append({'xyz': xyz, 'rgb': rgb, 'error': error})
    return points

def read_cameras_binary(path):
    cameras = {}
    with open(path, 'rb') as f:
        nc = struct.unpack('<Q', f.read(8))[0]
        for _ in range(nc):
            cid = struct.unpack('<i', f.read(4))[0]
            mid = struct.unpack('<i', f.read(4))[0]
            w = struct.unpack('<Q', f.read(8))[0]
            h = struct.unpack('<Q', f.read(8))[0]
            np_ = {0:3,1:4,2:4,3:5,4:8,5:8}.get(mid, 4)
            params = struct.unpack(f'<{np_}d', f.read(8*np_))
            cameras[cid] = {'model_id': mid, 'width': w, 'height': h, 'params': list(params)}
    return cameras

def read_images_binary(path):
    images = {}
    with open(path, 'rb') as f:
        ni = struct.unpack('<Q', f.read(8))[0]
        for _ in range(ni):
            iid = struct.unpack('<i', f.read(4))[0]
            qw,qx,qy,qz = struct.unpack('<4d', f.read(32))
            tx,ty,tz = struct.unpack('<3d', f.read(24))
            cid = struct.unpack('<i', f.read(4))[0]
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00': break
                name += c
            np2d = struct.unpack('<Q', f.read(8))[0]
            f.read(np2d * 24)
            images[iid] = {'qvec': [qw,qx,qy,qz], 'tvec': [tx,ty,tz], 'camera_id': cid, 'name': name.decode()}
    return images

def qvec2rotmat(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])

SH_C0 = 0.28209479177387814

def main():
    colmap_dir = os.path.expanduser('~/h00813233/360/flowers')
    out_dir = 'train/flowers_real'
    resolution = 8  # 1/8 of original
    os.makedirs(out_dir, exist_ok=True)

    sparse = os.path.join(colmap_dir, 'sparse', '0')
    print("Reading COLMAP data...")
    cameras = read_cameras_binary(os.path.join(sparse, 'cameras.bin'))
    images = read_images_binary(os.path.join(sparse, 'images.bin'))
    points = read_points3D_binary(os.path.join(sparse, 'points3D.bin'))
    print(f"  {len(cameras)} cameras, {len(images)} images, {len(points)} points")

    # Get target resolution
    cam0 = list(cameras.values())[0]
    tgt_w = cam0['width'] // resolution
    tgt_h = cam0['height'] // resolution
    print(f"  Target: {tgt_w}x{tgt_h}")

    # Compute per-point NN distance for initial scale
    print("Computing NN distances for initial scales...")
    positions = np.array([p['xyz'] for p in points], dtype=np.float64)
    N = len(positions)

    # Sample for KDTree speed
    if N > 10000:
        sample_idx = np.random.RandomState(42).choice(N, 10000, replace=False)
        tree = KDTree(positions[sample_idx])
        dists, _ = tree.query(positions, k=2)
        nn_dists = dists[:, 1]
    else:
        tree = KDTree(positions)
        dists, _ = tree.query(positions, k=2)
        nn_dists = dists[:, 1]

    # Clamp outliers
    nn_median = np.median(nn_dists)
    nn_dists = np.clip(nn_dists, nn_median * 0.01, nn_median * 10)
    print(f"  NN dist: median={nn_median:.4f}")

    # Write PLY with per-point scales
    ply_path = os.path.join(out_dir, 'init_points.ply')
    print(f"Writing PLY with {N} Gaussians...")

    header = f"""ply
format binary_little_endian 1.0
element vertex {N}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
    with open(ply_path, 'wb') as f:
        f.write(header.encode())
        for i, pt in enumerate(points):
            x,y,z = pt['xyz']
            r,g,b = [c/255.0 for c in pt['rgb']]
            f_dc = [(c - 0.5) / SH_C0 for c in [r,g,b]]

            # Position
            f.write(struct.pack('<3f', x, y, z))
            # Normals
            f.write(struct.pack('<3f', 0, 0, 0))
            # SH DC
            f.write(struct.pack('<3f', *f_dc))
            # Opacity: inverse_sigmoid(0.5) = 0
            f.write(struct.pack('<f', 0.0))
            # Scale: log(nn_dist * 0.5) — per-point adaptive
            s = float(nn_dists[i]) * 0.5
            log_s = float(np.log(max(s, 1e-6)))
            f.write(struct.pack('<3f', log_s, log_s, log_s))
            # Rotation: identity
            f.write(struct.pack('<4f', 1, 0, 0, 0))

    print(f"  Written to {ply_path}")

    # Write cameras JSON (C2W rotation for train_main.cpp)
    cam_list = []
    for iid, img in sorted(images.items()):
        ci = cameras[img['camera_id']]
        ow, oh = ci['width'], ci['height']
        params = ci['params']
        mid = ci['model_id']
        if mid in (0, 2):
            fx = fy = params[0]
        else:
            fx, fy = params[0], params[1]

        fx *= tgt_w / ow
        fy *= tgt_h / oh

        R_w2c = qvec2rotmat(img['qvec'])
        t_w2c = np.array(img['tvec'])
        cam_pos = -R_w2c.T @ t_w2c
        R_c2w = R_w2c.T

        cam_list.append({
            'id': iid,
            'img_name': img['name'].replace('.JPG', '.ppm'),
            'width': tgt_w, 'height': tgt_h,
            'fx': float(fx), 'fy': float(fy),
            'position': cam_pos.tolist(),
            'rotation': R_c2w.tolist(),
        })

    cam_path = os.path.join(out_dir, 'cameras.json')
    with open(cam_path, 'w') as f:
        json.dump(cam_list, f, indent=2)
    print(f"  {len(cam_list)} cameras → {cam_path}")

    # Convert GT images to PPM
    img_dir = os.path.join(colmap_dir, f'images_{resolution}')
    if not os.path.exists(img_dir):
        img_dir = os.path.join(colmap_dir, 'images')
    gt_dir = os.path.join(out_dir, 'gt_ppm')
    os.makedirs(gt_dir, exist_ok=True)

    print(f"Converting GT images to {tgt_w}x{tgt_h} PPM...")
    count = 0
    for img in sorted(images.values(), key=lambda x: x['name']):
        src = os.path.join(img_dir, img['name'])
        dst = os.path.join(gt_dir, img['name'].replace('.JPG', '.ppm'))
        if os.path.exists(src) and not os.path.exists(dst):
            subprocess.run(['convert', src, '-resize', f'{tgt_w}x{tgt_h}!', dst],
                         capture_output=True)
            count += 1
    print(f"  Converted {count} images")

    print(f"\nDone! Ready to train:")
    print(f"  ./build/gs3d_train \\")
    print(f"    --ply {ply_path} \\")
    print(f"    --cameras {cam_path} \\")
    print(f"    --gt_dir {gt_dir} \\")
    print(f"    --output {out_dir}/trained.ply \\")
    print(f"    --iterations 3000 --log_every 100 --lr_scale 2.0")

if __name__ == '__main__':
    main()
