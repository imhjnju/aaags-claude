#!/usr/bin/env python3
import argparse
import json
import math
import pathlib
import subprocess
import sys
import time

import numpy as np
from PIL import Image

SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[2] / "tools"
REPO_DIR = SCRIPT_DIR.parent
BUILD_DIR = REPO_DIR / "build"
AAA_ROOT = pathlib.Path("/home/robota/h00813233/Graph/aaags-claude/AAA-Gaussians")
INIT_PLY = pathlib.Path("/tmp/basketball_init_3dgs_filter3d.ply")
CAM_JSON = pathlib.Path("/home/robota/Downloads/basketball/_sp0_dump_output/cameras.json")
GT_JPG = pathlib.Path("/home/robota/Downloads/basketball/images/78899858295079.jpg")
CAM0_IDX = 0
SH_DEGREE = 3
LR_POS = 1.6e-4
LR_SH_DC = 2.5e-3
LR_SH_REST = 1.25e-4
LR_OP = 0.05
LR_SCA = 0.005
LR_ROT = 0.001
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-15


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def metrics(a, b):
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {
        "psnr": psnr(a, b),
        "mean_abs": float(diff.mean()),
        "p99_abs": float(np.quantile(diff, 0.99)),
        "max_abs": float(diff.max()),
    }


def save_png(arr, path):
    Image.fromarray((arr.clip(0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def write_ppm_u8(path, hwc):
    u8 = (hwc.clip(0, 1) * 255 + 0.5).astype(np.uint8)
    h, w = u8.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(u8.tobytes())


def load_cam0():
    cam_data = json.load(open(CAM_JSON))
    cam = cam_data[CAM0_IDX]
    assert cam["img_name"] == "78899858295079"
    gt = np.asarray(Image.open(GT_JPG).convert("RGB"), dtype=np.float32) / 255.0
    return cam, gt


def ply_count(path):
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[2])
            if line == "end_header":
                break
    raise RuntimeError(f"no element vertex in {path}")


def run_cuda(outdir, steps, densify_from_iter, densify_until_iter, densify_interval, cap_max, proper_ewa):
    sys.path.insert(0, str(AAA_ROOT))
    import torch
    from diff_gaussian_rasterization import ExtendedSettings
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene import GaussianModel
    from utils.loss_utils import l1_loss

    cam_info, gt_np = load_cam0()
    r_c2w = np.array(cam_info["rotation"], dtype=np.float32)
    cam_pos = np.array(cam_info["position"], dtype=np.float32)
    t_w2c = -(r_c2w.T @ cam_pos)
    w, h = cam_info["width"], cam_info["height"]
    fovx = 2.0 * math.atan(w / (2.0 * cam_info["fx"]))
    fovy = 2.0 * math.atan(h / (2.0 * cam_info["fy"]))
    gt_tensor = torch.from_numpy(gt_np).permute(2, 0, 1).float().cuda()
    cam = Camera(colmap_id=0, R=r_c2w, T=t_w2c.astype(np.float32), FoVx=fovx, FoVy=fovy,
                 image=gt_tensor, gt_alpha_mask=None, image_name="cam0", uid=0)

    gaussians = GaussianModel(sh_degree=SH_DEGREE)
    gaussians.load_ply(str(INIT_PLY))
    if gaussians.filter_3D is None:
        raise RuntimeError(f"{INIT_PLY} does not contain filter_3D")

    params = [
        {"params": [gaussians._xyz], "lr": LR_POS, "name": "xyz"},
        {"params": [gaussians._features_dc], "lr": LR_SH_DC, "name": "f_dc"},
        {"params": [gaussians._features_rest], "lr": LR_SH_REST, "name": "f_rest"},
        {"params": [gaussians._opacity], "lr": LR_OP, "name": "opacity"},
        {"params": [gaussians._scaling], "lr": LR_SCA, "name": "scaling"},
        {"params": [gaussians._rotation], "lr": LR_ROT, "name": "rotation"},
    ]
    optimizer = torch.optim.Adam(params, lr=0, betas=ADAM_BETAS, eps=ADAM_EPS)
    gaussians.optimizer = optimizer

    class PipeCfg:
        debug = False
        convert_SHs_python = False
        compute_cov3D_python = False

    bg = torch.zeros(3, device="cuda")

    def settings():
        es = ExtendedSettings()
        es.eval_3D = True
        es.proper_ewa_scaling = proper_ewa
        return es

    @torch.no_grad()
    def relocate_gs_with_filter(dead_mask):
        if dead_mask.sum() == 0:
            return 0
        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        if alive_indices.shape[0] <= 0:
            return 0
        probs = gaussians.get_opacity[alive_indices, 0]
        try:
            reinit_idx, ratio = gaussians._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])
        except Exception:
            return 0
        filter_src = gaussians.filter_3D[reinit_idx].clone()
        (
            gaussians._xyz[dead_indices],
            gaussians._features_dc[dead_indices],
            gaussians._features_rest[dead_indices],
            gaussians._opacity[dead_indices],
            gaussians._scaling[dead_indices],
            gaussians._rotation[dead_indices],
        ) = gaussians._update_params(reinit_idx, ratio=ratio)
        gaussians.filter_3D[dead_indices] = filter_src
        gaussians._opacity[reinit_idx] = gaussians._opacity[dead_indices]
        gaussians._scaling[reinit_idx] = gaussians._scaling[dead_indices]
        gaussians.replace_tensors_to_optimizer(inds=reinit_idx)
        return int(dead_indices.shape[0])

    @torch.no_grad()
    def add_new_gs_with_filter(cap):
        current_num_points = gaussians._opacity.shape[0]
        target_num = min(cap, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return 0
        probs = gaussians.get_opacity.squeeze(-1)
        add_idx, ratio = gaussians._sample_alives(probs=probs, num=num_gs)
        filter_src = gaussians.filter_3D[add_idx].clone()
        (
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        ) = gaussians._update_params(add_idx, ratio=ratio)
        gaussians._opacity[add_idx] = new_opacity
        gaussians._scaling[add_idx] = new_scaling
        old_filter = gaussians.filter_3D
        gaussians.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, reset_params=False)
        gaussians.filter_3D = torch.cat((old_filter, filter_src), dim=0)
        gaussians.replace_tensors_to_optimizer(inds=add_idx)
        return int(num_gs)

    losses = []
    events = []
    t0 = time.time()
    print(f"\n=== CUDA basketball eval_3D MCMC ({steps} steps, proper_ewa={proper_ewa}) ===")
    for step in range(1, steps + 1):
        image = render(cam, gaussians, PipeCfg(), bg, splat_args=settings())["render"]
        loss = l1_loss(image, gt_tensor)
        optimizer.zero_grad()
        loss.backward()
        if step < densify_until_iter and step > densify_from_iter and step % densify_interval == 0:
            before = int(gaussians.get_xyz.shape[0])
            dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
            relocated = relocate_gs_with_filter(dead_mask)
            added = add_new_gs_with_filter(cap_max)
            after = int(gaussians.get_xyz.shape[0])
            events.append((step, before, after, int(added), int(relocated)))
        if step < steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        lv = float(loss.item())
        losses.append(lv)
        if step <= 5 or step % 250 == 0:
            print(f"  [CUDA] step {step:5d}: loss={lv:.6f}, N={gaussians.get_xyz.shape[0]} ({time.time()-t0:.1f}s)")

    if gaussians.filter_3D.shape[0] != gaussians.get_xyz.shape[0]:
        raise RuntimeError(f"CUDA filter_3D length {gaussians.filter_3D.shape[0]} != Gaussian count {gaussians.get_xyz.shape[0]}")
    ply_path = outdir / "cuda_trained.ply"
    gaussians.save_ply(str(ply_path))
    with torch.no_grad():
        rendered = render(cam, gaussians, PipeCfg(), bg, splat_args=settings())["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    np.save(outdir / "cuda_render.npy", rendered)
    save_png(rendered, outdir / "cuda_render.png")
    return rendered, losses, int(gaussians.get_xyz.shape[0]), events


def run_vk(outdir, steps, vk_from_step, densify_until_step, densify_interval, cap_max, opacity_reset_interval, proper_ewa):
    cam_info, gt_np = load_cam0()
    cam0_json = outdir / "cam0_only.json"
    json.dump([cam_info], open(cam0_json, "w"), indent=2)
    ppm_dir = outdir / "ppm_cam0"
    ppm_dir.mkdir(exist_ok=True)
    write_ppm_u8(ppm_dir / (cam_info["img_name"] + ".ppm"), gt_np)
    vk_ply = outdir / "vk_trained.ply"
    cmd = [
        str(BUILD_DIR / "gs3d_vk_train"),
        "--ply", str(INIT_PLY),
        "--cameras", str(cam0_json),
        "--gt_dir", str(ppm_dir),
        "--output", str(vk_ply),
        "--iterations", str(steps),
        "--log_every", "250",
        "--eval_3d", "1",
        "--proper_ewa", "1" if proper_ewa else "0",
        "--densify", "1",
        "--densify_from_step", str(vk_from_step),
        "--densify_until_step", str(densify_until_step),
        "--densify_interval", str(densify_interval),
        "--cap_max", str(cap_max),
        "--opacity_reset_interval", str(opacity_reset_interval),
    ]
    print(f"\n=== VK basketball eval_3D MCMC ({steps} steps, proper_ewa={proper_ewa}) ===")
    print("  Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    render_wd = outdir / "vk_render_tmp"
    render_wd.mkdir(exist_ok=True)
    render_cmd = [
        str(BUILD_DIR / "gs3d_vk_render"),
        str(vk_ply.resolve()),
        str(cam0_json.resolve()),
        "0",
        "--eval_3d", "1",
        "--parity_mode", "1",
        "--proper_ewa", "1" if proper_ewa else "0",
    ]
    print("  Rendering saved PLY:", " ".join(render_cmd))
    subprocess.run(render_cmd, check=True, cwd=render_wd)
    h, w = cam_info["height"], cam_info["width"]
    raw = np.fromfile(render_wd / "vk_float.raw", dtype=np.float32)
    vk = raw.reshape(3, h, w).transpose(1, 2, 0).clip(0, 1)
    np.save(outdir / "vk_render.npy", vk)
    save_png(vk, outdir / "vk_render.png")

    train_ppm = pathlib.Path(str(vk_ply) + ".render.ppm")
    vk_train = np.asarray(Image.open(train_ppm).convert("RGB"), dtype=np.float32) / 255.0
    np.save(outdir / "vk_train_render.npy", vk_train)
    save_png(vk_train, outdir / "vk_train_render.png")
    return vk, vk_train, ply_count(vk_ply)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--cuda_densify_from_iter", type=int, default=500)
    ap.add_argument("--vk_densify_from_step", type=int, default=600)
    ap.add_argument("--densify_until", type=int, default=4999)
    ap.add_argument("--densify_interval", type=int, default=100)
    ap.add_argument("--cap_max", type=int, default=30000)
    ap.add_argument("--opacity_reset_interval", type=int, default=0)
    ap.add_argument("--proper_ewa", type=int, default=1, choices=[0, 1])
    args = ap.parse_args()

    if not INIT_PLY.exists():
        raise RuntimeError(f"missing filtered init PLY: {INIT_PLY}")
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cam_info, gt = load_cam0()
    save_png(gt, outdir / "gt_cam0.png")
    init_n = ply_count(INIT_PLY)
    proper_ewa = bool(args.proper_ewa)

    cuda, losses, cuda_n, events = run_cuda(outdir, args.steps, args.cuda_densify_from_iter,
                                            args.densify_until, args.densify_interval,
                                            args.cap_max, proper_ewa)
    vk, vk_train, vk_n = run_vk(outdir, args.steps, args.vk_densify_from_step,
                                args.densify_until, args.densify_interval,
                                args.cap_max, args.opacity_reset_interval, proper_ewa)
    save_png(np.concatenate([gt, cuda, vk], axis=1), outdir / "comparison.png")

    cuda_gt = metrics(cuda, gt)
    vk_gt = metrics(vk, gt)
    vk_cuda = metrics(vk, cuda)
    vk_train_gt = metrics(vk_train, gt)
    vk_train_cuda = metrics(vk_train, cuda)
    vk_saved_train = metrics(vk, vk_train)
    report = f"""
====================================================
  Basketball cam0 eval_3D/proper_ewa — VK vs CUDA MCMC
====================================================
  Init PLY    : {INIT_PLY}
  Initial N   : {init_n}
  cap_max     : {args.cap_max}
  Steps       : {args.steps}
  proper_ewa  : {int(proper_ewa)}
  CUDA gate   : iter > {args.cuda_densify_from_iter}, iter < {args.densify_until}, iter % {args.densify_interval} == 0
  VK gate     : step >= {args.vk_densify_from_step}, step <= {args.densify_until}, step % {args.densify_interval} == 0
  VK opacity reset interval: {args.opacity_reset_interval}
  CUDA final N: {cuda_n}
  VK final N  : {vk_n}
  CUDA final loss: {losses[-1]:.6f}
  CUDA events : {events[:20]}{' ...' if len(events) > 20 else ''}
====================================================
  CUDA vs GT       : PSNR={cuda_gt['psnr']:6.2f} dB  mean_abs={cuda_gt['mean_abs']:.6f}  p99={cuda_gt['p99_abs']:.6f}  max={cuda_gt['max_abs']:.6f}
  VK saved vs GT   : PSNR={vk_gt['psnr']:6.2f} dB  mean_abs={vk_gt['mean_abs']:.6f}  p99={vk_gt['p99_abs']:.6f}  max={vk_gt['max_abs']:.6f}
  VK saved vs CUDA : PSNR={vk_cuda['psnr']:6.2f} dB  mean_abs={vk_cuda['mean_abs']:.6f}  p99={vk_cuda['p99_abs']:.6f}  max={vk_cuda['max_abs']:.6f}
  VK train vs GT   : PSNR={vk_train_gt['psnr']:6.2f} dB  mean_abs={vk_train_gt['mean_abs']:.6f}  p99={vk_train_gt['p99_abs']:.6f}  max={vk_train_gt['max_abs']:.6f}
  VK train vs CUDA : PSNR={vk_train_cuda['psnr']:6.2f} dB  mean_abs={vk_train_cuda['mean_abs']:.6f}  p99={vk_train_cuda['p99_abs']:.6f}  max={vk_train_cuda['max_abs']:.6f}
  VK saved vs train: PSNR={vk_saved_train['psnr']:6.2f} dB  mean_abs={vk_saved_train['mean_abs']:.6f}  p99={vk_saved_train['p99_abs']:.6f}  max={vk_saved_train['max_abs']:.6f}
  Gap CUDA−VK(saved): {cuda_gt['psnr'] - vk_gt['psnr']:6.2f} dB
====================================================
"""
    print(report)
    (outdir / "report.txt").write_text(report)


if __name__ == "__main__":
    main()
