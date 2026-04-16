#!/usr/bin/env python3
"""Generate a tiny PLY file with 3 Gaussians for unit testing the PLY loader.

Gaussian 0: pos=(0,0,5), raw_opacity=0, raw_scale=(0,0,0), rot=(1,0,0,0)
Gaussian 1: pos=(1,0,5), raw_opacity=2, raw_scale=(1,1,1), rot=(0,1,0,0)
Gaussian 2: pos=(-1,0,5), raw_opacity=-2, raw_scale=(-1,-1,-1), rot=(1,1,0,0)

All DC SH coefficients = 0.5 for all channels.
All rest SH coefficients = 0.0.

PLY binary little-endian format, 62 float properties per vertex.
"""

import struct
import os

def write_ply(path):
    gaussians = [
        {
            "pos": (0.0, 0.0, 5.0),
            "normal": (0.0, 0.0, 0.0),
            "f_dc": (0.5, 0.5, 0.5),
            "f_rest": [0.0] * 45,
            "opacity": 0.0,
            "scale": (0.0, 0.0, 0.0),
            "rot": (1.0, 0.0, 0.0, 0.0),
        },
        {
            "pos": (1.0, 0.0, 5.0),
            "normal": (0.0, 0.0, 0.0),
            "f_dc": (0.5, 0.5, 0.5),
            "f_rest": [0.0] * 45,
            "opacity": 2.0,
            "scale": (1.0, 1.0, 1.0),
            "rot": (0.0, 1.0, 0.0, 0.0),
        },
        {
            "pos": (-1.0, 0.0, 5.0),
            "normal": (0.0, 0.0, 0.0),
            "f_dc": (0.5, 0.5, 0.5),
            "f_rest": [0.0] * 45,
            "opacity": -2.0,
            "scale": (-1.0, -1.0, -1.0),
            "rot": (1.0, 1.0, 0.0, 0.0),
        },
    ]

    # Build header
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(gaussians)}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    for i in range(45):
        header_lines.append(f"property float f_rest_{i}")
    header_lines += [
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]
    header = "\n".join(header_lines) + "\n"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for g in gaussians:
            # x, y, z
            f.write(struct.pack("<3f", *g["pos"]))
            # nx, ny, nz
            f.write(struct.pack("<3f", *g["normal"]))
            # f_dc_0, f_dc_1, f_dc_2
            f.write(struct.pack("<3f", *g["f_dc"]))
            # f_rest_0 .. f_rest_44
            f.write(struct.pack(f"<{len(g['f_rest'])}f", *g["f_rest"]))
            # opacity
            f.write(struct.pack("<f", g["opacity"]))
            # scale_0, scale_1, scale_2
            f.write(struct.pack("<3f", *g["scale"]))
            # rot_0, rot_1, rot_2, rot_3
            f.write(struct.pack("<4f", *g["rot"]))

    print(f"Written {len(gaussians)} gaussians to {path}")
    # Verify: 62 floats * 4 bytes = 248 bytes per vertex
    file_size = os.path.getsize(path)
    header_size = len(header.encode("ascii"))
    data_size = file_size - header_size
    expected_data = len(gaussians) * 62 * 4
    print(f"Header: {header_size} bytes, Data: {data_size} bytes (expected {expected_data})")
    assert data_size == expected_data, f"Data size mismatch: {data_size} != {expected_data}"


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "test_data", "tiny_3gaussians.ply")
    write_ply(output_path)
