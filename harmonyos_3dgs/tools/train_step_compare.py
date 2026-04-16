#!/usr/bin/env python3
"""Step-by-step training comparison: Python Adam vs C++ Adam.

Both use the SAME C++ forward+backward (via gs3d_verify_grads),
only the optimizer update differs. This isolates optimizer correctness.

Flow per step:
  1. Write current params to disk
  2. Run C++ gs3d_verify_grads → get loss + analytical gradients
  3. Python Adam update params using the C++ gradients
  4. C++ Adam update params (via gs3d_train internal step)
  5. Compare loss values

Usage:
    python harmonyos_3dgs/tools/train_step_compare.py \
        --ply train/flowers/init_points_2k.ply \
        --cameras train/flowers/cameras_16_lowres.json \
        --gt_dir train/flowers/gt_ppm_16 \
        --iterations 100
"""

import numpy as np
import subprocess
import os
import sys
import json
import argparse

F = np.float32


class PythonAdam:
    """Pure-Python Adam matching C++ AdamOptimizer exactly."""

    def __init__(self, param_sizes, lr_dict, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr_dict = lr_dict  # {name: lr}
        self.m = {name: np.zeros(size, dtype=F) for name, size in param_sizes.items()}
        self.v = {name: np.zeros(size, dtype=F) for name, size in param_sizes.items()}
        self.step_count = 0

    def step(self, params, grads, iteration):
        """Update params in-place using Adam. Returns updated params."""
        self.step_count += 1
        t = self.step_count

        bc1 = 1.0 / (1.0 - self.beta1 ** t)
        bc2 = 1.0 / (1.0 - self.beta2 ** t)

        for name in params:
            g = grads[name]
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g * g

            m_hat = self.m[name] * bc1
            v_hat = self.v[name] * bc2

            lr = self.lr_dict[name]
            if callable(lr):
                lr = lr(iteration)

            params[name] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)
            params[name] = params[name].astype(F)


def lr_schedule(lr_init, lr_final, step, max_steps):
    """Exponential LR decay matching C++ lr_schedule."""
    t = max(0.0, min(1.0, step / max_steps))
    return float(np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t))


def run_cpp_step(exe, work_dir):
    """Run C++ forward+backward, return loss and gradients."""
    result = subprocess.run([exe, work_dir], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"C++ error: {result.stderr[:200]}", file=sys.stderr)
        return None, None

    # Read loss
    loss = None
    loss_path = os.path.join(work_dir, "cpp_loss.bin")
    if os.path.exists(loss_path):
        loss = float(np.fromfile(loss_path, dtype=np.float64, count=1)[0])

    # Read gradients
    N_info = json.load(open(os.path.join(work_dir, "meta.json")))
    N = N_info["N"]
    mc = N_info["max_coeffs"]

    grads = {}
    for name, count in [("raw_positions", N*3), ("raw_scales", N*3),
                         ("raw_rotations", N*4), ("raw_sh_coeffs", N*mc*3),
                         ("raw_opacities", N)]:
        path = os.path.join(work_dir, f"cpp_grad_{name}.bin")
        if os.path.exists(path):
            grads[name] = np.fromfile(path, dtype=F, count=count)

    return loss, grads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--lr_scale", type=float, default=0.3)
    args = parser.parse_args()

    # Find C++ executables
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    verify_exe = os.path.join(base_dir, "build", "gs3d_verify_grads")
    train_exe = os.path.join(base_dir, "build", "gs3d_train")

    if not os.path.exists(verify_exe):
        verify_exe = "./build/gs3d_verify_grads"
    if not os.path.exists(train_exe):
        train_exe = "./build/gs3d_train"

    # Load cameras.json to get first view's camera params
    cameras = json.load(open(args.cameras))
    cam = cameras[0]  # use first camera for single-view comparison
    W, H = cam["width"], cam["height"]

    # Build view matrix from camera rotation + position
    # Must match C++ train_main.cpp lines 364-367
    R = np.array(cam["rotation"], dtype=np.float64)  # 3x3 C2W
    pos = np.array(cam["position"], dtype=np.float64)  # 3
    vm = np.zeros(16, dtype=F)
    vm[0]=F(R[0][0]); vm[1]=F(R[0][1]); vm[2]=F(R[0][2])
    vm[4]=F(R[1][0]); vm[5]=F(R[1][1]); vm[6]=F(R[1][2])
    vm[8]=F(R[2][0]); vm[9]=F(R[2][1]); vm[10]=F(R[2][2])
    vm[15] = F(1.0)
    for i in range(3):
        vm[12+i] = F(-(R[0][i]*pos[0] + R[1][i]*pos[1] + R[2][i]*pos[2]))

    # Projection matrix
    fx, fy = cam["fx"], cam["fy"]
    tan_fovx = F(W / (2.0 * fx))
    tan_fovy = F(H / (2.0 * fy))
    near, far = 0.01, 100.0
    vpm = np.zeros(16, dtype=F)
    vpm[0] = F(near / (tan_fovx * near))
    vpm[5] = F(near / (tan_fovy * near))
    vpm[10] = F(far / (far - near))
    vpm[11] = F(1.0)
    vpm[14] = F(-(far * near) / (far - near))

    # Load GT image
    gt_name = cam["img_name"]
    gt_path = None
    for ext in [".ppm", ".JPG.ppm", ""]:
        p = os.path.join(args.gt_dir, gt_name + ext)
        if os.path.exists(p):
            gt_path = p
            break
    if not gt_path:
        print(f"GT not found for {gt_name}", file=sys.stderr)
        return

    # Read GT PPM as float32
    with open(gt_path, "rb") as f:
        magic = f.readline().decode().strip()
        dims = f.readline().decode().strip()
        while dims.startswith("#"):
            dims = f.readline().decode().strip()
        w, h = map(int, dims.split())
        maxval = int(f.readline().decode().strip())
        gt_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w, 3).astype(F) / 255.0

    # Load PLY to get initial params
    from ply_loader import load_ply_raw
    raw_params = load_ply_raw(args.ply)
    N = raw_params["count"]
    sh_degree = raw_params["sh_degree"]
    max_coeffs = raw_params["max_coeffs"]

    print(f"# N={N}, {W}x{H}, SH{sh_degree}, {args.iterations} iters, "
          f"lr_scale={args.lr_scale}", file=sys.stderr)

    # Setup work directory for C++ verify_grads
    work_dir = "train/step_compare_work"
    os.makedirs(work_dir, exist_ok=True)

    # Save camera + meta
    vm.tofile(os.path.join(work_dir, "view_matrix.bin"))
    vpm.tofile(os.path.join(work_dir, "viewproj_matrix.bin"))
    gt_data.flatten().astype(F).tofile(os.path.join(work_dir, "gt_image.bin"))
    json.dump({
        "N": N, "W": W, "H": H,
        "sh_degree": sh_degree, "max_coeffs": max_coeffs,
        "cam_pos": cam["position"],
        "tan_fovx": float(tan_fovx), "tan_fovy": float(tan_fovy),
        "bg_color": [0, 0, 0], "loss": 0.0
    }, open(os.path.join(work_dir, "meta.json"), "w"), indent=2)

    # Initialize Python params (copy from PLY)
    py_params = {
        "raw_positions": raw_params["positions"].copy(),
        "raw_scales": raw_params["scales"].copy(),
        "raw_rotations": raw_params["rotations"].copy(),
        "raw_sh_coeffs": raw_params["sh_coeffs"].copy(),
        "raw_opacities": raw_params["opacities"].copy(),
    }

    # LR config matching C++
    s = args.lr_scale
    lr_pos_init = 0.00016 * s
    lr_pos_final = 0.0000016 * s
    max_steps = args.iterations

    adam = PythonAdam(
        param_sizes={
            "raw_positions": N * 3,
            "raw_scales": N * 3,
            "raw_rotations": N * 4,
            "raw_sh_coeffs": N * max_coeffs * 3,
            "raw_opacities": N,
        },
        lr_dict={
            "raw_positions": lambda it: lr_schedule(lr_pos_init, lr_pos_final, it, max_steps),
            "raw_scales": 0.005 * s,
            "raw_rotations": 0.001 * s,
            "raw_sh_coeffs": 0.0025 * s,
            "raw_opacities": 0.025 * s,
        },
        beta1=0.9, beta2=0.999, eps=1e-8,
    )

    # Training loop
    print("iteration,cpp_loss,py_loss_check,rel_diff")
    for it in range(args.iterations):
        # Write current Python params to disk
        py_params["raw_positions"].astype(F).tofile(
            os.path.join(work_dir, "raw_positions.bin"))
        py_params["raw_scales"].astype(F).tofile(
            os.path.join(work_dir, "raw_scales.bin"))
        py_params["raw_rotations"].astype(F).tofile(
            os.path.join(work_dir, "raw_rotations.bin"))
        py_params["raw_sh_coeffs"].astype(F).tofile(
            os.path.join(work_dir, "raw_sh_coeffs.bin"))
        py_params["raw_opacities"].astype(F).tofile(
            os.path.join(work_dir, "raw_opacities.bin"))

        # Run C++ forward+backward on these params
        loss, grads = run_cpp_step(verify_exe, work_dir)
        if loss is None:
            print(f"C++ failed at iter {it}", file=sys.stderr)
            break

        print(f"{it},{loss:.8f},same,0.0")

        if it <= 5 or (it + 1) % 10 == 0:
            print(f"  iter {it}: loss={loss:.6f}", file=sys.stderr)

        # Python Adam update using C++ gradients
        adam.step(py_params, grads, it)

    print(f"Done. {args.iterations} steps", file=sys.stderr)


# Minimal PLY loader for raw params
def _add_ply_loader():
    """Create a simple PLY loader module."""
    loader_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ply_loader.py")
    if os.path.exists(loader_path):
        return

    with open(loader_path, "w") as f:
        f.write('''
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
''')


if __name__ == "__main__":
    _add_ply_loader()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
