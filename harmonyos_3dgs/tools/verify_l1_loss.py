#!/usr/bin/env python3
"""Generate deterministic L1 loss test vectors and verify C++ matches Python.

Produces binary test data + expected values. Run C++ test afterwards to compare.
Usage:
    python harmonyos_3dgs/tools/verify_l1_loss.py
    ./build/gs3d_tests --gtest_filter="L1LossCrosslang.*"
"""

import numpy as np
import struct
import os
import json

def python_l1_loss_reference(rendered, gt):
    """Exact match of official 3DGS: torch.abs(output - gt).mean()
    Using float32 arithmetic to match C++."""
    r = rendered.astype(np.float32).flatten()
    g = gt.astype(np.float32).flatten()
    return float(np.mean(np.abs(r - g)))

def python_l1_loss_serial(rendered, gt):
    """Serial accumulation matching C++ loop order exactly."""
    r = rendered.astype(np.float32).flatten()
    g = gt.astype(np.float32).flatten()
    n = len(r)
    inv_n = np.float32(1.0 / n)
    total = np.float32(0.0)
    for i in range(n):
        diff = np.float32(r[i] - g[i])
        total = np.float32(total + np.float32(np.abs(diff)))
    return float(np.float32(total * inv_n))

def python_l1_gradient(rendered, gt):
    """Gradient of L1 loss w.r.t. rendered, matching C++ d_image."""
    r = rendered.astype(np.float32).flatten()
    g = gt.astype(np.float32).flatten()
    n = len(r)
    inv_n = np.float32(1.0 / n)
    d_image = np.zeros(n, dtype=np.float32)
    for i in range(n):
        diff = np.float32(r[i] - g[i])
        if diff > 0:
            d_image[i] = inv_n
        elif diff < 0:
            d_image[i] = -inv_n
        else:
            d_image[i] = 0.0
    return d_image

def generate_test_case(name, H, W, seed):
    """Generate a single test case with deterministic random data."""
    rng = np.random.RandomState(seed)
    rendered = rng.rand(H, W, 3).astype(np.float32)
    gt = rng.rand(H, W, 3).astype(np.float32)

    loss_serial = python_l1_loss_serial(rendered, gt)
    loss_numpy = python_l1_loss_reference(rendered, gt)
    d_image = python_l1_gradient(rendered, gt)

    return {
        'name': name,
        'H': H,
        'W': W,
        'seed': seed,
        'rendered': rendered,
        'gt': gt,
        'loss_serial': loss_serial,
        'loss_numpy': loss_numpy,
        'd_image': d_image,
    }

def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'tests', 'test_data', 'l1_verify')
    os.makedirs(out_dir, exist_ok=True)

    # Test cases: various sizes
    cases = [
        ('tiny_1x1',   1,   1, 42),
        ('small_4x4',  4,   4, 123),
        ('med_16x16', 16,  16, 456),
        ('med_64x64', 64,  64, 789),
        ('large_256x256', 256, 256, 1337),
    ]

    manifest = []

    print("=== Python L1 Loss Test Vector Generation ===\n")
    for name, H, W, seed in cases:
        tc = generate_test_case(name, H, W, seed)
        n = H * W * 3

        # Save binary data
        prefix = os.path.join(out_dir, name)
        tc['rendered'].tofile(f'{prefix}_rendered.bin')
        tc['gt'].tofile(f'{prefix}_gt.bin')
        tc['d_image'].tofile(f'{prefix}_d_image.bin')

        entry = {
            'name': name,
            'H': H,
            'W': W,
            'n_elements': n,
            'loss_serial': tc['loss_serial'],
            'loss_numpy': tc['loss_numpy'],
            'serial_numpy_diff': abs(tc['loss_serial'] - tc['loss_numpy']),
        }
        manifest.append(entry)

        print(f"  {name:20s}: {H:4d}x{W:4d}  n={n:8d}  "
              f"loss_serial={tc['loss_serial']:.10f}  "
              f"loss_numpy={tc['loss_numpy']:.10f}  "
              f"diff={abs(tc['loss_serial'] - tc['loss_numpy']):.2e}")

    # Save manifest
    manifest_path = os.path.join(out_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nTest data written to {out_dir}/")
    print(f"Manifest: {manifest_path}")
    print(f"\nNext: run ./build/gs3d_tests --gtest_filter='L1LossCrosslang.*'")

if __name__ == '__main__':
    main()
