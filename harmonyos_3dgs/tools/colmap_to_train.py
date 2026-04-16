#!/usr/bin/env python3
"""Convert COLMAP sparse reconstruction to training data for C++ gs3d_train.

Reads: sparse/0/{cameras.bin, images.bin, points3D.bin}
Outputs:
  - init_points.ply: Initial 3D Gaussian PLY (from COLMAP points)
  - cameras.json: Camera parameters for training
  - Copies/resizes ground truth images to gt_images/ directory

Usage:
  python3 colmap_to_train.py <colmap_dataset_dir> <output_dir> [--resolution 4]
"""

import sys
import os
import struct
import json
import numpy as np
from collections import namedtuple
import shutil

# ---- COLMAP binary readers ----

CameraModel = namedtuple('CameraModel', ['model_id', 'model_name', 'num_params'])
CAMERA_MODELS = {
    0: CameraModel(0, 'SIMPLE_PINHOLE', 3),
    1: CameraModel(1, 'PINHOLE', 4),
    2: CameraModel(2, 'SIMPLE_RADIAL', 4),
    3: CameraModel(3, 'RADIAL', 5),
    4: CameraModel(4, 'OPENCV', 8),
    5: CameraModel(5, 'OPENCV_FISHEYE', 8),
}

def read_cameras_binary(path):
    cameras = {}
    with open(path, 'rb') as f:
        num_cameras = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_cameras):
            cam_id = struct.unpack('<i', f.read(4))[0]
            model_id = struct.unpack('<i', f.read(4))[0]
            width = struct.unpack('<Q', f.read(8))[0]
            height = struct.unpack('<Q', f.read(8))[0]
            num_params = CAMERA_MODELS[model_id].num_params
            params = struct.unpack(f'<{num_params}d', f.read(8 * num_params))
            cameras[cam_id] = {
                'model': CAMERA_MODELS[model_id].model_name,
                'width': width,
                'height': height,
                'params': list(params)
            }
    return cameras

def read_images_binary(path):
    images = {}
    with open(path, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_images):
            img_id = struct.unpack('<i', f.read(4))[0]
            qw, qx, qy, qz = struct.unpack('<4d', f.read(32))
            tx, ty, tz = struct.unpack('<3d', f.read(24))
            cam_id = struct.unpack('<i', f.read(4))[0]
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name += c
            num_points2D = struct.unpack('<Q', f.read(8))[0]
            f.read(num_points2D * 24)  # skip 2D points
            images[img_id] = {
                'qvec': [qw, qx, qy, qz],
                'tvec': [tx, ty, tz],
                'camera_id': cam_id,
                'name': name.decode('utf-8')
            }
    return images

def read_points3D_binary(path):
    points = []
    with open(path, 'rb') as f:
        num_points = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_points):
            pt_id = struct.unpack('<Q', f.read(8))[0]
            xyz = struct.unpack('<3d', f.read(24))
            rgb = struct.unpack('<3B', f.read(3))
            error = struct.unpack('<d', f.read(8))[0]
            track_length = struct.unpack('<Q', f.read(8))[0]
            f.read(track_length * 8)  # skip track
            points.append({'xyz': xyz, 'rgb': rgb, 'error': error})
    return points

# ---- Quaternion to rotation matrix ----

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    return R

# ---- Write PLY with 3DGS format ----

def write_gaussian_ply(path, points, sh_degree=0):
    """Write initial Gaussians PLY from COLMAP points."""
    N = len(points)
    max_coeffs = (sh_degree + 1) ** 2

    # Properties matching 3DGS PLY format
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
"""
    # Add rest SH coefficients
    for i in range((max_coeffs - 1) * 3):
        header += f"property float f_rest_{i}\n"

    header += """property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

    SH_C0 = 0.28209479177387814

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        for pt in points:
            x, y, z = pt['xyz']
            r, g, b = [c / 255.0 for c in pt['rgb']]

            # Convert RGB to SH DC coefficients
            f_dc = [(c - 0.5) / SH_C0 for c in [r, g, b]]

            # Position
            f.write(struct.pack('<3f', x, y, z))
            # Normals (unused)
            f.write(struct.pack('<3f', 0, 0, 0))
            # SH DC
            f.write(struct.pack('<3f', *f_dc))
            # SH rest (zeros)
            for _ in range((max_coeffs - 1) * 3):
                f.write(struct.pack('<f', 0.0))
            # Opacity (inverse sigmoid of 0.1 -> small initial opacity)
            opacity = np.log(0.1 / (1.0 - 0.1))
            f.write(struct.pack('<f', opacity))
            # Scale (log-space, initial small)
            avg_dist = 0.5  # default initial scale
            log_scale = np.log(avg_dist)
            f.write(struct.pack('<3f', log_scale, log_scale, log_scale))
            # Rotation (identity quaternion)
            f.write(struct.pack('<4f', 1, 0, 0, 0))

    print(f"  Written {N} Gaussians to {path}")

# ---- Build camera JSON ----

def build_cameras_json(colmap_cameras, colmap_images, target_w, target_h):
    """Build camera list for C++ gs3d_train."""
    cam_list = []

    for img_id, img in sorted(colmap_images.items()):
        cam_info = colmap_cameras[img['camera_id']]
        orig_w, orig_h = cam_info['width'], cam_info['height']

        # Get focal lengths
        model = cam_info['model']
        params = cam_info['params']
        if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL'):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif model in ('PINHOLE', 'OPENCV'):
            fx, fy = params[0], params[1]
            cx, cy = params[2], params[3]
        else:
            fx = fy = params[0]
            cx, cy = orig_w / 2, orig_h / 2

        # Scale focal lengths to target resolution
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        fx *= scale_x
        fy *= scale_y

        # W2C rotation and translation
        R = qvec2rotmat(img['qvec'])  # world-to-camera rotation
        t = np.array(img['tvec'])      # world-to-camera translation

        # Camera position in world space
        cam_pos = -R.T @ t

        # Build 4x4 view matrix (column-major for OpenGL)
        view = np.eye(4)
        view[:3, :3] = R
        view[:3, 3] = t

        # Build projection matrix
        near, far = 0.01, 100.0
        tan_fovx = target_w / (2.0 * fx)
        tan_fovy = target_h / (2.0 * fy)

        # The C++ loader expects C2W rotation (it transposes internally to get W2C)
        R_c2w = R.T  # W2C^T = C2W

        cam_list.append({
            'id': img_id,
            'img_name': img['name'],
            'width': target_w,
            'height': target_h,
            'fx': float(fx),
            'fy': float(fy),
            'position': cam_pos.tolist(),
            'rotation': R_c2w.tolist(),  # 3x3 C2W rotation (code transposes to W2C)
        })

    return cam_list

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <colmap_dir> <output_dir> [--resolution 4]")
        sys.exit(1)

    colmap_dir = sys.argv[1]
    output_dir = sys.argv[2]
    resolution = 4  # default: use images_4

    for i, arg in enumerate(sys.argv):
        if arg == '--resolution' and i + 1 < len(sys.argv):
            resolution = int(sys.argv[i + 1])

    sparse_dir = os.path.join(colmap_dir, 'sparse', '0')
    if resolution == 1:
        images_dir = os.path.join(colmap_dir, 'images')
    else:
        images_dir = os.path.join(colmap_dir, f'images_{resolution}')
        if not os.path.exists(images_dir):
            images_dir = os.path.join(colmap_dir, 'images')
            print(f"  Warning: images_{resolution} not found, using full-res images")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading COLMAP data from {sparse_dir}...")
    cameras = read_cameras_binary(os.path.join(sparse_dir, 'cameras.bin'))
    images = read_images_binary(os.path.join(sparse_dir, 'images.bin'))
    points = read_points3D_binary(os.path.join(sparse_dir, 'points3D.bin'))

    print(f"  {len(cameras)} cameras, {len(images)} images, {len(points)} 3D points")

    # Determine target resolution from first image
    sample_img = sorted(images.values(), key=lambda x: x['name'])[0]
    sample_cam = cameras[sample_img['camera_id']]
    target_w = sample_cam['width'] // resolution
    target_h = sample_cam['height'] // resolution
    print(f"  Target resolution: {target_w}x{target_h} (1/{resolution} of original)")

    # 1. Write initial PLY
    ply_path = os.path.join(output_dir, 'init_points.ply')
    write_gaussian_ply(ply_path, points)

    # 2. Write cameras JSON
    cam_list = build_cameras_json(cameras, images, target_w, target_h)
    cam_json_path = os.path.join(output_dir, 'cameras.json')
    with open(cam_json_path, 'w') as f:
        json.dump(cam_list, f, indent=2)
    print(f"  Written {len(cam_list)} cameras to {cam_json_path}")

    # 3. Copy/link GT images
    gt_dir = os.path.join(output_dir, 'gt_images')
    os.makedirs(gt_dir, exist_ok=True)
    count = 0
    for img_info in sorted(images.values(), key=lambda x: x['name']):
        src = os.path.join(images_dir, img_info['name'])
        if os.path.exists(src):
            dst = os.path.join(gt_dir, img_info['name'])
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
            count += 1
    print(f"  Linked {count} GT images to {gt_dir}")

    print(f"\nDone! Output in {output_dir}")
    print(f"  PLY:     {ply_path} ({len(points)} points)")
    print(f"  Cameras: {cam_json_path} ({len(cam_list)} views)")
    print(f"  GT:      {gt_dir} ({count} images)")

if __name__ == '__main__':
    main()
