
import numpy as np
F = np.float32

def load_ply_raw(path):
    """Load PLY and return raw (pre-activation) parameters."""
    with open(path, "rb") as f:
        props = []
        N = 0
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                N = int(line.split()[-1])
            if line.startswith("property float"):
                props.append(line.split()[-1])
            if line == "end_header":
                break
        data = np.frombuffer(f.read(N * len(props) * 4), dtype=F).reshape(N, len(props))

    # Map property names to indices
    idx = {p: i for i, p in enumerate(props)}

    # Count SH rest coefficients
    sh_rest = [p for p in props if p.startswith("f_rest_")]
    n_rest = len(sh_rest)
    n_dc = 3  # f_dc_0, f_dc_1, f_dc_2
    total_sh = n_dc + n_rest
    max_coeffs = total_sh // 3
    sh_degree = int(max_coeffs ** 0.5) - 1

    # Extract params
    positions = data[:, [idx["x"], idx["y"], idx["z"]]].flatten().copy()
    scales = data[:, [idx.get(f"scale_{i}", 0) for i in range(3)]].flatten().copy()
    rotations = data[:, [idx.get(f"rot_{i}", 0) for i in range(4)]].flatten().copy()
    opacities = data[:, idx["opacity"]].flatten().copy()

    # SH: interleave DC + rest
    sh_coeffs = np.zeros((N, max_coeffs * 3), dtype=F)
    for ch in range(3):
        sh_coeffs[:, ch] = data[:, idx[f"f_dc_{ch}"]]
    for i in range(n_rest):
        sh_coeffs[:, 3 + i] = data[:, idx[f"f_rest_{i}"]]
    sh_coeffs = sh_coeffs.flatten().copy()

    return {
        "count": N,
        "sh_degree": sh_degree,
        "max_coeffs": max_coeffs,
        "positions": positions,
        "scales": scales,
        "rotations": rotations,
        "sh_coeffs": sh_coeffs,
        "opacities": opacities,
    }
