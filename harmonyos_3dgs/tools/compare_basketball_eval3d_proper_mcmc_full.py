#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
from PIL import Image

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
BUILD_DIR = REPO_DIR / "build"
AAA_ROOT = pathlib.Path("/home/robota/h00813233/Graph/aaags-claude/AAA-Gaussians")
DEFAULT_INIT_PLY = pathlib.Path("/tmp/basketball_init_3dgs_filter3d.ply")
DEFAULT_CAM_JSON = pathlib.Path("/home/robota/Downloads/basketball/_sp0_dump_output/cameras.json")
DEFAULT_IMAGE_DIR = pathlib.Path("/home/robota/Downloads/basketball/images")
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


def aggregate_metrics(lhs, rhs):
    a = np.concatenate([x.reshape(-1, 3) for x in lhs], axis=0)
    b = np.concatenate([x.reshape(-1, 3) for x in rhs], axis=0)
    per_view = [metrics(x, y)["psnr"] for x, y in zip(lhs, rhs)]
    m = metrics(a, b)
    m["mean_view_psnr"] = float(np.mean(per_view))
    m["min_view_psnr"] = float(np.min(per_view))
    return m


def save_png(arr, path):
    Image.fromarray((arr.clip(0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def write_ppm_u8(path, hwc):
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = (hwc.clip(0, 1) * 255 + 0.5).astype(np.uint8)
    h, w = u8.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(u8.tobytes())


def image_path(image_dir, img_name):
    candidates = [image_dir / img_name]
    if pathlib.Path(img_name).suffix == "":
        candidates.extend(image_dir / f"{img_name}{ext}" for ext in (".jpg", ".JPG", ".png", ".PNG"))
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing image for {img_name} in {image_dir}")


def load_dataset(cam_json, image_dir, max_views):
    cameras = json.load(open(cam_json))
    if max_views > 0:
        cameras = cameras[:max_views]
    if not cameras:
        raise RuntimeError("no cameras selected")
    out = []
    for idx, cam in enumerate(cameras):
        cam = dict(cam)
        cam["id"] = idx
        img = np.asarray(Image.open(image_path(image_dir, cam["img_name"])).convert("RGB"), dtype=np.float32) / 255.0
        expected = (cam["height"], cam["width"], 3)
        if img.shape != expected:
            raise RuntimeError(f"{cam['img_name']} shape {img.shape} != {expected}")
        out.append((cam, img))
    return out


def ply_count(path):
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[2])
            if line == "end_header":
                break
    raise RuntimeError(f"no element vertex in {path}")


def build_schedule(steps, n_views, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_views, size=steps, endpoint=False, dtype=np.int32)


def exp_lr(lr_init, lr_final, step, max_steps, spatial_lr_scale):
    t = max(0.0, min(1.0, float(step) / float(max_steps)))
    return spatial_lr_scale * math.exp(math.log(lr_init) * (1.0 - t) + math.log(lr_final) * t)


def final_forward_sh_degree(steps, sh_degree_warmup):
    if sh_degree_warmup <= 0:
        return SH_DEGREE
    return min((steps - 1) // sh_degree_warmup, SH_DEGREE)


def make_cuda_camera(cam_info, gt_np, idx):
    import torch
    from scene.cameras import Camera

    r_c2w = np.array(cam_info["rotation"], dtype=np.float32)
    cam_pos = np.array(cam_info["position"], dtype=np.float32)
    t_w2c = -(r_c2w.T @ cam_pos)
    w, h = cam_info["width"], cam_info["height"]
    fovx = 2.0 * math.atan(w / (2.0 * cam_info["fx"]))
    fovy = 2.0 * math.atan(h / (2.0 * cam_info["fy"]))
    gt_tensor = torch.from_numpy(gt_np).permute(2, 0, 1).float().cuda()
    cam = Camera(colmap_id=idx, R=r_c2w, T=t_w2c.astype(np.float32), FoVx=fovx, FoVy=fovy,
                 image=gt_tensor, gt_alpha_mask=None, image_name=cam_info["img_name"], uid=idx)
    return cam, gt_tensor


def run_cuda(outdir, init_ply, dataset, schedule, steps, densify_from_iter, densify_until_iter,
             densify_interval, cap_max, proper_ewa, seed, lambda_dssim,
             pos_lr_init, pos_lr_final, spatial_lr_scale, sh_degree_warmup,
             save_render_npy):
    sys.path.insert(0, str(AAA_ROOT))
    import torch
    from diff_gaussian_rasterization import ExtendedSettings
    from gaussian_renderer import render
    from scene import GaussianModel
    from utils.loss_utils import l1_loss
    if lambda_dssim > 0.0:
        from fused_ssim import fused_ssim
    else:
        fused_ssim = None

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cuda_views = [make_cuda_camera(cam, gt, idx) for idx, (cam, gt) in enumerate(dataset)]
    gt_images = [gt for _, gt in dataset]

    gaussians = GaussianModel(sh_degree=SH_DEGREE)
    gaussians.load_ply(str(init_ply))
    if sh_degree_warmup > 0:
        gaussians.active_sh_degree = 0
    if gaussians.filter_3D is None:
        raise RuntimeError(f"{init_ply} does not contain filter_3D")

    params = [
        {"params": [gaussians._xyz], "lr": pos_lr_init * spatial_lr_scale, "name": "xyz"},
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
    train_t0 = time.time()
    print(f"\n=== CUDA basketball full-dataset eval_3D MCMC ({steps} steps, views={len(dataset)}, proper_ewa={proper_ewa}) ===")
    for step in range(1, steps + 1):
        view_idx = int(schedule[step - 1])
        if sh_degree_warmup > 0:
            gaussians.active_sh_degree = min((step - 1) // sh_degree_warmup, SH_DEGREE)
        pos_lr = exp_lr(pos_lr_init, pos_lr_final, step, steps, spatial_lr_scale)
        optimizer.param_groups[0]["lr"] = pos_lr
        cam, gt_tensor = cuda_views[view_idx]
        image = render(cam, gaussians, PipeCfg(), bg, splat_args=settings())["render"]
        l1 = l1_loss(image, gt_tensor)
        if lambda_dssim > 0.0:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_tensor.unsqueeze(0))
            loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_value)
        else:
            loss = l1
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
        if step <= 5 or step % 250 == 0 or step == steps:
            print(f"  [CUDA] step {step:5d}: loss={lv:.6f}, view={view_idx}, N={gaussians.get_xyz.shape[0]} ({time.time()-t0:.1f}s)")

    train_seconds = time.time() - train_t0
    if gaussians.filter_3D.shape[0] != gaussians.get_xyz.shape[0]:
        raise RuntimeError(f"CUDA filter_3D length {gaussians.filter_3D.shape[0]} != Gaussian count {gaussians.get_xyz.shape[0]}")
    ply_path = outdir / "cuda_trained.ply"
    gaussians.save_ply(str(ply_path))

    render_t0 = time.time()
    renders = []
    with torch.no_grad():
        for idx, (cam, _) in enumerate(cuda_views):
            rendered = render(cam, gaussians, PipeCfg(), bg, splat_args=settings())["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            renders.append(rendered)
            if save_render_npy:
                np.save(outdir / f"cuda_render_view{idx:03d}.npy", rendered)
            if idx < 6:
                save_png(rendered, outdir / f"cuda_render_view{idx:03d}.png")
    render_seconds = time.time() - render_t0
    timing = {"train_seconds": train_seconds, "render_seconds": render_seconds}
    return renders, gt_images, losses, int(gaussians.get_xyz.shape[0]), events, timing


def run_vk(outdir, init_ply, dataset, schedule, steps, vk_from_step, densify_until_iter,
           densify_interval, cap_max, opacity_reset_interval, proper_ewa, lambda_dssim,
           pos_lr_init, pos_lr_final, spatial_lr_scale, sh_degree_warmup,
           save_render_npy):
    cam_json = outdir / "cameras_selected.json"
    json.dump([cam for cam, _ in dataset], open(cam_json, "w"), indent=2)

    ppm_dir = outdir / "gt_ppm"
    ppm_dir.mkdir(exist_ok=True)
    for cam, gt in dataset:
        write_ppm_u8(ppm_dir / f"{cam['img_name']}.ppm", gt)
        if pathlib.Path(cam["img_name"]).suffix:
            link = ppm_dir / pathlib.Path(cam["img_name"]).with_suffix(".ppm")
            if link != ppm_dir / f"{cam['img_name']}.ppm":
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    os.symlink(pathlib.Path(f"{cam['img_name']}.ppm").name, link)

    schedule_path = outdir / "view_schedule.txt"
    schedule_path.write_text("\n".join(str(int(v)) for v in schedule) + "\n")

    vk_ply = outdir / "vk_trained.ply"
    vk_until_step = densify_until_iter - 1
    vk_from_arg = vk_from_step
    vk_until_arg = vk_until_step
    if vk_until_arg < vk_from_arg:
        vk_from_arg = steps + 1
        vk_until_arg = steps + 1
    cmd = [
        str(BUILD_DIR / "gs3d_vk_train"),
        "--ply", str(init_ply),
        "--cameras", str(cam_json),
        "--gt_dir", str(ppm_dir),
        "--view_schedule", str(schedule_path),
        "--require_all_gt", "1",
        "--output", str(vk_ply),
        "--iterations", str(steps),
        "--log_every", "250",
        "--eval_3d", "1",
        "--proper_ewa", "1" if proper_ewa else "0",
        "--densify", "1",
        "--densify_from_step", str(vk_from_arg),
        "--densify_until_step", str(vk_until_arg),
        "--densify_interval", str(densify_interval),
        "--cap_max", str(cap_max),
        "--opacity_reset_interval", str(opacity_reset_interval),
        "--lambda_dssim", str(lambda_dssim),
        "--pos_lr_init", str(pos_lr_init),
        "--pos_lr_final", str(pos_lr_final),
        "--spatial_lr_scale", str(spatial_lr_scale),
        "--sh_degree_warmup", str(sh_degree_warmup),
    ]
    print(f"\n=== VK basketball full-dataset eval_3D MCMC ({steps} steps, views={len(dataset)}, proper_ewa={proper_ewa}) ===")
    print("  Running:", " ".join(cmd))
    train_t0 = time.time()
    subprocess.run(cmd, check=True)
    train_seconds = time.time() - train_t0

    train_ppm = pathlib.Path(str(vk_ply) + ".render.ppm")
    train_render = np.asarray(Image.open(train_ppm).convert("RGB"), dtype=np.float32) / 255.0

    render_t0 = time.time()
    renders = []
    render_sh_degree = final_forward_sh_degree(steps, sh_degree_warmup)
    for idx, (cam, _) in enumerate(dataset):
        render_wd = outdir / f"vk_render_view{idx:03d}"
        render_wd.mkdir(exist_ok=True)
        render_cmd = [
            str(BUILD_DIR / "gs3d_vk_render"),
            str(vk_ply.resolve()),
            str(cam_json.resolve()),
            str(idx),
            "--eval_3d", "1",
            "--parity_mode", "1",
            "--proper_ewa", "1" if proper_ewa else "0",
            "--sh_degree", str(render_sh_degree),
        ]
        subprocess.run(render_cmd, check=True, cwd=render_wd, stdout=subprocess.DEVNULL)
        h, w = cam["height"], cam["width"]
        raw = np.fromfile(render_wd / "vk_float.raw", dtype=np.float32)
        vk = raw.reshape(3, h, w).transpose(1, 2, 0).clip(0, 1)
        renders.append(vk)
        if save_render_npy:
            np.save(outdir / f"vk_render_view{idx:03d}.npy", vk)
        if idx < 6:
            save_png(vk, outdir / f"vk_render_view{idx:03d}.png")
    render_seconds = time.time() - render_t0
    timing = {"train_seconds": train_seconds, "render_seconds": render_seconds}
    return renders, train_render, ply_count(vk_ply), vk_from_arg, vk_until_arg, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--init_ply", type=pathlib.Path, default=DEFAULT_INIT_PLY)
    ap.add_argument("--cameras", type=pathlib.Path, default=DEFAULT_CAM_JSON)
    ap.add_argument("--image_dir", type=pathlib.Path, default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--max_views", type=int, default=0, help="0 means all cameras")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cuda_densify_from_iter", type=int, default=500)
    ap.add_argument("--vk_densify_from_step", type=int, default=600)
    ap.add_argument("--densify_until", type=int, default=4999)
    ap.add_argument("--densify_interval", type=int, default=100)
    ap.add_argument("--cap_max", type=int, default=30000)
    ap.add_argument("--opacity_reset_interval", type=int, default=0)
    ap.add_argument("--proper_ewa", type=int, default=1, choices=[0, 1])
    ap.add_argument("--lambda_dssim", type=float, default=0.0)
    ap.add_argument("--pos_lr_init", type=float, default=LR_POS)
    ap.add_argument("--pos_lr_final", type=float, default=LR_POS)
    ap.add_argument("--spatial_lr_scale", type=float, default=1.0)
    ap.add_argument("--sh_degree_warmup", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--save_render_npy", type=int, default=0, choices=[0, 1])
    ap.add_argument("--min_vk_cuda_psnr", type=float, default=20.0)
    ap.add_argument("--max_gt_psnr_gap", type=float, default=2.0)
    args = ap.parse_args()

    if not args.init_ply.exists():
        raise RuntimeError(f"missing filtered init PLY: {args.init_ply}")
    if not 0.0 <= args.lambda_dssim <= 1.0:
        raise RuntimeError("--lambda_dssim must be in [0, 1]")
    if args.pos_lr_init <= 0.0 or args.pos_lr_final <= 0.0:
        raise RuntimeError("--pos_lr_init and --pos_lr_final must be > 0")
    if args.spatial_lr_scale <= 0.0:
        raise RuntimeError("--spatial_lr_scale must be > 0")
    if args.sh_degree_warmup < 0:
        raise RuntimeError("--sh_degree_warmup must be >= 0")
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.cameras, args.image_dir, args.max_views)
    schedule = build_schedule(args.steps, len(dataset), args.seed)
    np.save(outdir / "view_schedule.npy", schedule)
    proper_ewa = bool(args.proper_ewa)
    init_n = ply_count(args.init_ply)
    render_sh_degree = final_forward_sh_degree(args.steps, args.sh_degree_warmup)
    config = {
        "label": args.label,
        "init_ply": str(args.init_ply),
        "cameras": str(args.cameras),
        "image_dir": str(args.image_dir),
        "views": len(dataset),
        "initial_n": init_n,
        "steps": args.steps,
        "seed": args.seed,
        "proper_ewa": int(proper_ewa),
        "lambda_dssim": args.lambda_dssim,
        "pos_lr_init": args.pos_lr_init,
        "pos_lr_final": args.pos_lr_final,
        "spatial_lr_scale": args.spatial_lr_scale,
        "sh_degree_warmup": args.sh_degree_warmup,
        "render_sh_degree": render_sh_degree,
        "cuda_densify_from_iter": args.cuda_densify_from_iter,
        "vk_densify_from_step": args.vk_densify_from_step,
        "densify_until": args.densify_until,
        "densify_interval": args.densify_interval,
        "cap_max": args.cap_max,
        "opacity_reset_interval": args.opacity_reset_interval,
        "save_render_npy": args.save_render_npy,
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    save_render_npy = bool(args.save_render_npy)
    cuda, gt_images, losses, cuda_n, events, cuda_timing = run_cuda(
        outdir, args.init_ply, dataset, schedule, args.steps,
        args.cuda_densify_from_iter, args.densify_until, args.densify_interval,
        args.cap_max, proper_ewa, args.seed, args.lambda_dssim,
        args.pos_lr_init, args.pos_lr_final, args.spatial_lr_scale,
        args.sh_degree_warmup, save_render_npy)
    vk, vk_train_final, vk_n, vk_from_arg, vk_until_arg, vk_timing = run_vk(
        outdir, args.init_ply, dataset, schedule, args.steps,
        args.vk_densify_from_step, args.densify_until, args.densify_interval,
        args.cap_max, args.opacity_reset_interval, proper_ewa, args.lambda_dssim,
        args.pos_lr_init, args.pos_lr_final, args.spatial_lr_scale,
        args.sh_degree_warmup, save_render_npy)

    if dataset:
        panels = []
        for idx in range(min(3, len(dataset))):
            panels.append(np.concatenate([gt_images[idx], cuda[idx], vk[idx]], axis=1))
        save_png(np.concatenate(panels, axis=0), outdir / "comparison.png")

    cuda_gt = aggregate_metrics(cuda, gt_images)
    vk_gt = aggregate_metrics(vk, gt_images)
    vk_cuda = aggregate_metrics(vk, cuda)
    final_view = int(schedule[args.steps - 1])
    vk_saved_train = metrics(vk[final_view], vk_train_final)
    failed = []
    if cuda_n != vk_n:
        failed.append(f"Gaussian count mismatch: CUDA={cuda_n}, VK={vk_n}")
    if vk_cuda["psnr"] < args.min_vk_cuda_psnr:
        failed.append(f"VK vs CUDA PSNR {vk_cuda['psnr']:.2f} < {args.min_vk_cuda_psnr:.2f}")
    if abs(cuda_gt["psnr"] - vk_gt["psnr"]) > args.max_gt_psnr_gap:
        failed.append(f"GT PSNR gap {cuda_gt['psnr'] - vk_gt['psnr']:.2f} exceeds ±{args.max_gt_psnr_gap:.2f}")
    cuda_total_seconds = cuda_timing["train_seconds"] + cuda_timing["render_seconds"]
    vk_total_seconds = vk_timing["train_seconds"] + vk_timing["render_seconds"]
    train_speedup = cuda_timing["train_seconds"] / vk_timing["train_seconds"] if vk_timing["train_seconds"] > 0 else float("inf")
    render_speedup = cuda_timing["render_seconds"] / vk_timing["render_seconds"] if vk_timing["render_seconds"] > 0 else float("inf")
    total_speedup = cuda_total_seconds / vk_total_seconds if vk_total_seconds > 0 else float("inf")
    report = f"""
====================================================
  Basketball full-dataset eval_3D/proper_ewa — VK vs CUDA MCMC
====================================================
  Init PLY    : {args.init_ply}
  Cameras     : {args.cameras}
  Views       : {len(dataset)}
  Initial N   : {init_n}
  cap_max     : {args.cap_max}
  Steps       : {args.steps}
  Seed        : {args.seed}
  proper_ewa  : {int(proper_ewa)}
  lambda_dssim: {args.lambda_dssim:.3f}
  pos LR      : {args.pos_lr_init:.8f} -> {args.pos_lr_final:.8f} (scale={args.spatial_lr_scale:.3f})
  SH warmup   : {args.sh_degree_warmup}
  render SH   : {render_sh_degree}
  CUDA gate   : iter > {args.cuda_densify_from_iter}, iter < {args.densify_until}, iter % {args.densify_interval} == 0
  VK gate     : step >= {vk_from_arg}, step <= {vk_until_arg}, step % {args.densify_interval} == 0
  VK opacity reset interval: {args.opacity_reset_interval}
  CUDA final N: {cuda_n}
  VK final N  : {vk_n}
  CUDA final loss: {losses[-1]:.6f}
  CUDA events : {events[:20]}{' ...' if len(events) > 20 else ''}
====================================================
  Timing CUDA train loop : {cuda_timing['train_seconds']:8.2f}s
  Timing VK train CLI    : {vk_timing['train_seconds']:8.2f}s  CUDA/VK={train_speedup:6.2f}x
  Timing CUDA all-view render: {cuda_timing['render_seconds']:8.2f}s
  Timing VK all-view render  : {vk_timing['render_seconds']:8.2f}s  CUDA/VK={render_speedup:6.2f}x
  Timing total train+render  : CUDA={cuda_total_seconds:8.2f}s  VK={vk_total_seconds:8.2f}s  CUDA/VK={total_speedup:6.2f}x
====================================================
  CUDA vs GT       : PSNR={cuda_gt['psnr']:6.2f} dB  mean_view={cuda_gt['mean_view_psnr']:6.2f}  min_view={cuda_gt['min_view_psnr']:6.2f}  mean_abs={cuda_gt['mean_abs']:.6f}  p99={cuda_gt['p99_abs']:.6f}  max={cuda_gt['max_abs']:.6f}
  VK saved vs GT   : PSNR={vk_gt['psnr']:6.2f} dB  mean_view={vk_gt['mean_view_psnr']:6.2f}  min_view={vk_gt['min_view_psnr']:6.2f}  mean_abs={vk_gt['mean_abs']:.6f}  p99={vk_gt['p99_abs']:.6f}  max={vk_gt['max_abs']:.6f}
  VK saved vs CUDA : PSNR={vk_cuda['psnr']:6.2f} dB  mean_view={vk_cuda['mean_view_psnr']:6.2f}  min_view={vk_cuda['min_view_psnr']:6.2f}  mean_abs={vk_cuda['mean_abs']:.6f}  p99={vk_cuda['p99_abs']:.6f}  max={vk_cuda['max_abs']:.6f}
  VK saved vs train final view {final_view}: PSNR={vk_saved_train['psnr']:6.2f} dB  mean_abs={vk_saved_train['mean_abs']:.6f}  p99={vk_saved_train['p99_abs']:.6f}  max={vk_saved_train['max_abs']:.6f}
  Gap CUDA−VK(saved): {cuda_gt['psnr'] - vk_gt['psnr']:6.2f} dB
  Gate status       : {'PASS' if not failed else 'FAIL — ' + '; '.join(failed)}
====================================================
"""
    metrics_row = {
        **config,
        "cuda_final_n": cuda_n,
        "vk_final_n": vk_n,
        "cuda_final_loss": losses[-1],
        "cuda_train_seconds": cuda_timing["train_seconds"],
        "vk_train_seconds": vk_timing["train_seconds"],
        "cuda_render_seconds": cuda_timing["render_seconds"],
        "vk_render_seconds": vk_timing["render_seconds"],
        "cuda_total_seconds": cuda_total_seconds,
        "vk_total_seconds": vk_total_seconds,
        "train_cuda_over_vk": train_speedup,
        "render_cuda_over_vk": render_speedup,
        "total_cuda_over_vk": total_speedup,
        "cuda_gt_psnr": cuda_gt["psnr"],
        "vk_gt_psnr": vk_gt["psnr"],
        "vk_cuda_psnr": vk_cuda["psnr"],
        "vk_saved_train_psnr": vk_saved_train["psnr"],
        "gt_psnr_gap": cuda_gt["psnr"] - vk_gt["psnr"],
        "final_view": final_view,
        "gate_status": "PASS" if not failed else "FAIL",
        "gate_failures": "; ".join(failed),
    }
    with open(outdir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_row.keys()))
        writer.writeheader()
        writer.writerow(metrics_row)
    print(report)
    (outdir / "report.txt").write_text(report)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
