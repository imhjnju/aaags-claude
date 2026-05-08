#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import pathlib
import re
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
    if not lhs or not rhs:
        raise RuntimeError("cannot aggregate metrics for an empty view set")
    a = np.concatenate([x.reshape(-1, 3) for x in lhs], axis=0)
    b = np.concatenate([x.reshape(-1, 3) for x in rhs], axis=0)
    per_view = [metrics(x, y)["psnr"] for x, y in zip(lhs, rhs)]
    m = metrics(a, b)
    m["mean_view_psnr"] = float(np.mean(per_view))
    m["min_view_psnr"] = float(np.min(per_view))
    return m


def select_views(values, indices):
    return [values[i] for i in indices]


def add_metric_columns(row, prefix, values):
    for key in ("psnr", "mean_view_psnr", "min_view_psnr", "mean_abs", "p99_abs", "max_abs"):
        row[f"{prefix}_{key}"] = values[key]


def format_metric_line(label, values):
    return (
        f"  {label:<18}: PSNR={values['psnr']:6.2f} dB  "
        f"mean_view={values['mean_view_psnr']:6.2f}  min_view={values['min_view_psnr']:6.2f}  "
        f"mean_abs={values['mean_abs']:.6f}  p99={values['p99_abs']:.6f}  max={values['max_abs']:.6f}"
    )


def finite_failure(name, value):
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return f"{name} is non-finite: {value}"
    return None


def array_finite_failure(name, arr):
    finite = np.isfinite(arr)
    if bool(finite.all()):
        return None
    bad = int(finite.size - finite.sum())
    return f"{name} contains {bad} non-finite value(s)"


def metric_finite_failures(label, values):
    out = []
    for key, value in values.items():
        failure = finite_failure(f"{label} {key}", value)
        if failure:
            out.append(failure)
    return out


def parse_vk_final_loss(log_text):
    final_loss = None
    number = r"([+-]?(?:nan|inf|infinity|\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    for match in re.finditer(rf"(?:Final loss:\s*|loss=){number}", log_text, re.IGNORECASE):
        final_loss = float(match.group(1))
    return final_loss


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


def camera_eval_sort_key(cam):
    # AAA-GS sorts COLMAP cameras by image_name before applying llffhold.
    # Exported JSONs in this repo may carry the same semantic name as img_name.
    for key in ("image_name", "name", "path", "img_name"):
        value = cam.get(key)
        if value:
            return str(value)
    return None


def split_eval_dataset(dataset, llffhold):
    if llffhold <= 0:
        raise RuntimeError("--llffhold must be > 0")
    keyed = []
    for idx, item in enumerate(dataset):
        cam, _ = item
        keyed.append((idx, camera_eval_sort_key(cam), item))
    if all(key is not None for _, key, _ in keyed):
        ordered = sorted(keyed, key=lambda item: item[1])
        sort_mode = "camera-name"
    else:
        # Clear fallback: if any camera lacks image_name/name/path/img_name, preserve the
        # stable input order rather than partially sorting incomparable camera records.
        ordered = keyed
        sort_mode = "stable-original-order"
    train = []
    test = []
    train_render_indices = []
    test_render_indices = []
    render_dataset = []
    split_rows = []
    for sorted_idx, (original_idx, key, item) in enumerate(ordered):
        render_idx = len(render_dataset)
        render_dataset.append(item)
        is_test = sorted_idx % llffhold == 0
        if is_test:
            test.append(item)
            test_render_indices.append(render_idx)
        else:
            train.append(item)
            train_render_indices.append(render_idx)
        cam, _ = item
        split_rows.append({
            "render_index": render_idx,
            "original_index": original_idx,
            "sorted_index": sorted_idx,
            "split": "test" if is_test else "train",
            "sort_key": key,
            "img_name": cam.get("img_name", ""),
        })
    if not train:
        raise RuntimeError(f"eval split produced no train cameras (views={len(dataset)}, llffhold={llffhold})")
    if not test:
        raise RuntimeError(f"eval split produced no test cameras (views={len(dataset)}, llffhold={llffhold})")
    return train, test, render_dataset, train_render_indices, test_render_indices, sort_mode, split_rows


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


def run_cuda(outdir, init_ply, train_dataset, render_dataset, schedule, steps, densify_from_iter, densify_until_iter,
             densify_interval, cap_max, proper_ewa, seed, lambda_dssim,
             pos_lr_init, pos_lr_final, spatial_lr_scale, sh_degree_warmup,
             opacity_reg, scale_reg, noise_lr, save_render_npy):
    sys.path.insert(0, str(AAA_ROOT))
    import torch
    from diff_gaussian_rasterization import ExtendedSettings
    from gaussian_renderer import render
    from scene import GaussianModel
    from scene.gaussian_model import build_scaling_rotation
    from utils.loss_utils import l1_loss
    if lambda_dssim > 0.0:
        from fused_ssim import fused_ssim
    else:
        fused_ssim = None

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cuda_train_views = [make_cuda_camera(cam, gt, idx) for idx, (cam, gt) in enumerate(train_dataset)]
    cuda_render_views = [make_cuda_camera(cam, gt, idx) for idx, (cam, gt) in enumerate(render_dataset)]
    gt_images = [gt for _, gt in render_dataset]

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
    print(f"\n=== CUDA basketball eval_3D MCMC ({steps} steps, train_views={len(train_dataset)}, render_views={len(render_dataset)}, proper_ewa={proper_ewa}) ===")
    for step in range(1, steps + 1):
        view_idx = int(schedule[step - 1])
        if sh_degree_warmup > 0:
            gaussians.active_sh_degree = min((step - 1) // sh_degree_warmup, SH_DEGREE)
        pos_lr = exp_lr(pos_lr_init, pos_lr_final, step, steps, spatial_lr_scale)
        optimizer.param_groups[0]["lr"] = pos_lr
        cam, gt_tensor = cuda_train_views[view_idx]
        image = render(cam, gaussians, PipeCfg(), bg, splat_args=settings())["render"]
        l1 = l1_loss(image, gt_tensor)
        if lambda_dssim > 0.0:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_tensor.unsqueeze(0))
            loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_value)
        else:
            loss = l1
        if opacity_reg != 0.0:
            loss = loss + opacity_reg * torch.abs(gaussians.get_opacity).mean()
        if scale_reg != 0.0:
            loss = loss + scale_reg * torch.abs(gaussians.get_scaling).mean()
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
            if noise_lr != 0.0:
                with torch.no_grad():
                    def op_sigmoid(x, k=100, x0=0.995):
                        return 1 / (1 + torch.exp(-k * (x - x0)))
                    L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                    actual_covariance = L @ L.transpose(1, 2)
                    noise = torch.randn_like(gaussians._xyz) * op_sigmoid(1 - gaussians.get_opacity) * noise_lr * pos_lr
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                    gaussians._xyz.add_(noise)
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
    render_failures = []
    with torch.no_grad():
        for idx, (cam, _) in enumerate(cuda_render_views):
            image = render(cam, gaussians, PipeCfg(), bg, splat_args=settings())["render"]
            finite = torch.isfinite(image)
            if not bool(finite.all().item()):
                bad = int(finite.numel() - finite.sum().item())
                render_failures.append(f"CUDA render view {idx} contains {bad} non-finite value(s) before clamp")
            rendered = image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            renders.append(rendered)
            if save_render_npy:
                np.save(outdir / f"cuda_render_view{idx:03d}.npy", rendered)
            if idx < 6:
                save_png(rendered, outdir / f"cuda_render_view{idx:03d}.png")
    render_seconds = time.time() - render_t0
    timing = {"train_seconds": train_seconds, "render_seconds": render_seconds}
    return renders, gt_images, losses, int(gaussians.get_xyz.shape[0]), events, timing, render_failures


def run_vk(outdir, init_ply, train_dataset, render_dataset, schedule, steps, vk_from_step, densify_until_iter,
           densify_interval, cap_max, opacity_reset_interval, proper_ewa, lambda_dssim,
           pos_lr_init, pos_lr_final, spatial_lr_scale, sh_degree_warmup,
           opacity_reg, scale_reg, noise_lr, vk_parity_mode, save_render_npy):
    train_cam_json = outdir / "cameras_train.json"
    render_cam_json = outdir / "cameras_render.json"
    json.dump([cam for cam, _ in train_dataset], open(train_cam_json, "w"), indent=2)
    json.dump([cam for cam, _ in render_dataset], open(render_cam_json, "w"), indent=2)

    ppm_dir = outdir / "gt_ppm"
    ppm_dir.mkdir(exist_ok=True)
    for cam, gt in train_dataset:
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
        "--cameras", str(train_cam_json),
        "--gt_dir", str(ppm_dir),
        "--view_schedule", str(schedule_path),
        "--require_all_gt", "1",
        "--output", str(vk_ply),
        "--iterations", str(steps),
        "--log_every", "250",
        "--eval_3d", "1",
        "--parity_mode", "1" if vk_parity_mode else "0",
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
        "--opacity_reg", str(opacity_reg),
        "--scale_reg", str(scale_reg),
        "--noise_lr", str(noise_lr),
    ]
    print(f"\n=== VK basketball eval_3D MCMC ({steps} steps, train_views={len(train_dataset)}, render_views={len(render_dataset)}, proper_ewa={proper_ewa}) ===")
    print("  Running:", " ".join(cmd))
    train_t0 = time.time()
    train_proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    train_seconds = time.time() - train_t0
    if train_proc.stdout:
        print(train_proc.stdout, end="" if train_proc.stdout.endswith("\n") else "\n")
    train_proc.check_returncode()
    vk_final_loss = parse_vk_final_loss(train_proc.stdout or "")

    train_ppm = pathlib.Path(str(vk_ply) + ".render.ppm")
    train_render = np.asarray(Image.open(train_ppm).convert("RGB"), dtype=np.float32) / 255.0

    render_t0 = time.time()
    renders = []
    render_failures = []
    render_sh_degree = final_forward_sh_degree(steps, sh_degree_warmup)
    for idx, (cam, _) in enumerate(render_dataset):
        render_wd = outdir / f"vk_render_view{idx:03d}"
        render_wd.mkdir(exist_ok=True)
        render_cmd = [
            str(BUILD_DIR / "gs3d_vk_render"),
            str(vk_ply.resolve()),
            str(render_cam_json.resolve()),
            str(idx),
            "--eval_3d", "1",
            "--parity_mode", "1" if vk_parity_mode else "0",
            "--proper_ewa", "1" if proper_ewa else "0",
            "--sh_degree", str(render_sh_degree),
        ]
        subprocess.run(render_cmd, check=True, cwd=render_wd, stdout=subprocess.DEVNULL)
        h, w = cam["height"], cam["width"]
        raw = np.fromfile(render_wd / "vk_float.raw", dtype=np.float32)
        failure = array_finite_failure(f"Vulkan render view {idx} raw", raw)
        if failure:
            render_failures.append(failure)
        vk = raw.reshape(3, h, w).transpose(1, 2, 0).clip(0, 1)
        renders.append(vk)
        if save_render_npy:
            np.save(outdir / f"vk_render_view{idx:03d}.npy", vk)
        if idx < 6:
            save_png(vk, outdir / f"vk_render_view{idx:03d}.png")
    render_seconds = time.time() - render_t0
    timing = {"train_seconds": train_seconds, "render_seconds": render_seconds}
    return renders, train_render, ply_count(vk_ply), vk_from_arg, vk_until_arg, timing, vk_final_loss, render_failures


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
    ap.add_argument("--vk_parity_mode", type=int, default=1, choices=[0, 1])
    ap.add_argument("--lambda_dssim", type=float, default=0.0)
    ap.add_argument("--pos_lr_init", type=float, default=LR_POS)
    ap.add_argument("--pos_lr_final", type=float, default=LR_POS)
    ap.add_argument("--spatial_lr_scale", type=float, default=1.0)
    ap.add_argument("--sh_degree_warmup", type=int, default=0)
    ap.add_argument("--opacity_reg", type=float, default=0.0)
    ap.add_argument("--scale_reg", type=float, default=0.0)
    ap.add_argument("--noise_lr", type=float, default=0.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--save_render_npy", type=int, default=0, choices=[0, 1])
    ap.add_argument("--eval_split", type=int, default=0, choices=[0, 1], help="Use AAA-GS llffhold train/test split")
    ap.add_argument("--llffhold", type=int, default=8, help="Hold out every Nth sorted camera when --eval_split=1")
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
    if args.opacity_reg < 0.0 or args.scale_reg < 0.0 or args.noise_lr < 0.0:
        raise RuntimeError("--opacity_reg, --scale_reg, and --noise_lr must be >= 0")
    if args.llffhold <= 0:
        raise RuntimeError("--llffhold must be > 0")
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.cameras, args.image_dir, args.max_views)
    if args.eval_split:
        train_dataset, test_dataset, render_dataset, train_indices, test_indices, split_sort_mode, split_rows = split_eval_dataset(dataset, args.llffhold)
    else:
        train_dataset = dataset
        test_dataset = []
        render_dataset = dataset
        train_indices = list(range(len(dataset)))
        test_indices = []
        split_sort_mode = "disabled"
        split_rows = []
    schedule = build_schedule(args.steps, len(train_dataset), args.seed)
    np.save(outdir / "view_schedule.npy", schedule)
    proper_ewa = bool(args.proper_ewa)
    init_n = ply_count(args.init_ply)
    render_sh_degree = final_forward_sh_degree(args.steps, args.sh_degree_warmup)
    config = {
        "label": args.label,
        "init_ply": str(args.init_ply),
        "cameras": str(args.cameras),
        "image_dir": str(args.image_dir),
        "views": len(render_dataset),
        "train_views": len(train_dataset),
        "test_views": len(test_dataset),
        "eval_split": args.eval_split,
        "llffhold": args.llffhold,
        "split_sort_mode": split_sort_mode,
        "initial_n": init_n,
        "steps": args.steps,
        "seed": args.seed,
        "proper_ewa": int(proper_ewa),
        "vk_parity_mode": args.vk_parity_mode,
        "lambda_dssim": args.lambda_dssim,
        "pos_lr_init": args.pos_lr_init,
        "pos_lr_final": args.pos_lr_final,
        "spatial_lr_scale": args.spatial_lr_scale,
        "sh_degree_warmup": args.sh_degree_warmup,
        "opacity_reg": args.opacity_reg,
        "scale_reg": args.scale_reg,
        "noise_lr": args.noise_lr,
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
    if split_rows:
        (outdir / "split_cameras.json").write_text(json.dumps(split_rows, indent=2) + "\n")

    save_render_npy = bool(args.save_render_npy)
    cuda, gt_images, losses, cuda_n, events, cuda_timing, cuda_render_failures = run_cuda(
        outdir, args.init_ply, train_dataset, render_dataset, schedule, args.steps,
        args.cuda_densify_from_iter, args.densify_until, args.densify_interval,
        args.cap_max, proper_ewa, args.seed, args.lambda_dssim,
        args.pos_lr_init, args.pos_lr_final, args.spatial_lr_scale,
        args.sh_degree_warmup, args.opacity_reg, args.scale_reg, args.noise_lr,
        save_render_npy)
    vk, vk_train_final, vk_n, vk_from_arg, vk_until_arg, vk_timing, vk_final_loss, vk_render_failures = run_vk(
        outdir, args.init_ply, train_dataset, render_dataset, schedule, args.steps,
        args.vk_densify_from_step, args.densify_until, args.densify_interval,
        args.cap_max, args.opacity_reset_interval, proper_ewa, args.lambda_dssim,
        args.pos_lr_init, args.pos_lr_final, args.spatial_lr_scale,
        args.sh_degree_warmup, args.opacity_reg, args.scale_reg, args.noise_lr,
        bool(args.vk_parity_mode), save_render_npy)

    if render_dataset:
        panels = []
        for idx in range(min(3, len(render_dataset))):
            panels.append(np.concatenate([gt_images[idx], cuda[idx], vk[idx]], axis=1))
        save_png(np.concatenate(panels, axis=0), outdir / "comparison.png")

    failed = []
    cuda_final_loss = losses[-1]
    for idx, value in enumerate(losses, start=1):
        failure = finite_failure(f"CUDA loss step {idx}", value)
        if failure:
            failed.append(failure)
    if vk_final_loss is None:
        failed.append("Vulkan final loss was not found in gs3d_vk_train output")
    else:
        failure = finite_failure("Vulkan final loss", vk_final_loss)
        if failure:
            failed.append(failure)
    failed.extend(cuda_render_failures)
    failed.extend(vk_render_failures)
    for idx, arr in enumerate(cuda):
        failure = array_finite_failure(f"CUDA render view {idx}", arr)
        if failure:
            failed.append(failure)
    for idx, arr in enumerate(vk):
        failure = array_finite_failure(f"Vulkan render view {idx}", arr)
        if failure:
            failed.append(failure)
    failure = array_finite_failure("Vulkan final training render", vk_train_final)
    if failure:
        failed.append(failure)

    cuda_gt = aggregate_metrics(cuda, gt_images)
    vk_gt = aggregate_metrics(vk, gt_images)
    vk_cuda = aggregate_metrics(vk, cuda)
    train_cuda_gt = aggregate_metrics(select_views(cuda, train_indices), select_views(gt_images, train_indices))
    train_vk_gt = aggregate_metrics(select_views(vk, train_indices), select_views(gt_images, train_indices))
    train_vk_cuda = aggregate_metrics(select_views(vk, train_indices), select_views(cuda, train_indices))
    test_cuda_gt = None
    test_vk_gt = None
    test_vk_cuda = None
    if test_indices:
        test_cuda_gt = aggregate_metrics(select_views(cuda, test_indices), select_views(gt_images, test_indices))
        test_vk_gt = aggregate_metrics(select_views(vk, test_indices), select_views(gt_images, test_indices))
        test_vk_cuda = aggregate_metrics(select_views(vk, test_indices), select_views(cuda, test_indices))
    final_train_view = int(schedule[args.steps - 1])
    final_render_view = train_indices[final_train_view]
    vk_saved_train = metrics(vk[final_render_view], vk_train_final)
    metric_sets = [
        ("Full CUDA vs GT", cuda_gt),
        ("Full VK saved vs GT", vk_gt),
        ("Full VK saved vs CUDA", vk_cuda),
        ("Train CUDA vs GT", train_cuda_gt),
        ("Train VK saved vs GT", train_vk_gt),
        ("Train VK saved vs CUDA", train_vk_cuda),
        ("VK saved vs train final", vk_saved_train),
    ]
    if test_indices:
        metric_sets.extend([
            ("Test CUDA vs GT", test_cuda_gt),
            ("Test VK saved vs GT", test_vk_gt),
            ("Test VK saved vs CUDA", test_vk_cuda),
        ])
    for label, values in metric_sets:
        failed.extend(metric_finite_failures(label, values))
    if cuda_n != vk_n:
        failed.append(f"Gaussian count mismatch: CUDA={cuda_n}, VK={vk_n}")
    gate_sets = [("full", cuda_gt, vk_gt, vk_cuda)] if not args.eval_split else [
        ("train", train_cuda_gt, train_vk_gt, train_vk_cuda),
        ("test", test_cuda_gt, test_vk_gt, test_vk_cuda),
    ]
    for split_name, split_cuda_gt, split_vk_gt, split_vk_cuda in gate_sets:
        if split_vk_cuda["psnr"] < args.min_vk_cuda_psnr:
            failed.append(f"{split_name} VK vs CUDA PSNR {split_vk_cuda['psnr']:.2f} < {args.min_vk_cuda_psnr:.2f}")
        gap = split_cuda_gt["psnr"] - split_vk_gt["psnr"]
        if abs(gap) > args.max_gt_psnr_gap:
            failed.append(f"{split_name} GT PSNR gap {gap:.2f} exceeds ±{args.max_gt_psnr_gap:.2f}")
    cuda_total_seconds = cuda_timing["train_seconds"] + cuda_timing["render_seconds"]
    vk_total_seconds = vk_timing["train_seconds"] + vk_timing["render_seconds"]
    train_speedup = cuda_timing["train_seconds"] / vk_timing["train_seconds"] if vk_timing["train_seconds"] > 0 else float("inf")
    render_speedup = cuda_timing["render_seconds"] / vk_timing["render_seconds"] if vk_timing["render_seconds"] > 0 else float("inf")
    total_speedup = cuda_total_seconds / vk_total_seconds if vk_total_seconds > 0 else float("inf")
    vk_final_loss_text = "unavailable" if vk_final_loss is None else f"{vk_final_loss:.6f}"
    if args.eval_split:
        metric_report = "\n".join([
            f"  Train split ({len(train_indices)} views)",
            format_metric_line("CUDA vs GT", train_cuda_gt),
            format_metric_line("VK saved vs GT", train_vk_gt),
            format_metric_line("VK saved vs CUDA", train_vk_cuda),
            f"  Train gap CUDA−VK(saved): {train_cuda_gt['psnr'] - train_vk_gt['psnr']:6.2f} dB",
            f"  Test split ({len(test_indices)} views)",
            format_metric_line("CUDA vs GT", test_cuda_gt),
            format_metric_line("VK saved vs GT", test_vk_gt),
            format_metric_line("VK saved vs CUDA", test_vk_cuda),
            f"  Test gap CUDA−VK(saved) : {test_cuda_gt['psnr'] - test_vk_gt['psnr']:6.2f} dB",
            "  Full rendered aggregate (train+test)",
            format_metric_line("CUDA vs GT", cuda_gt),
            format_metric_line("VK saved vs GT", vk_gt),
            format_metric_line("VK saved vs CUDA", vk_cuda),
        ])
    else:
        metric_report = "\n".join([
            format_metric_line("CUDA vs GT", cuda_gt),
            format_metric_line("VK saved vs GT", vk_gt),
            format_metric_line("VK saved vs CUDA", vk_cuda),
            f"  Gap CUDA−VK(saved): {cuda_gt['psnr'] - vk_gt['psnr']:6.2f} dB",
        ])
    report = f"""
====================================================
  Basketball eval_3D/proper_ewa — VK vs CUDA MCMC
====================================================
  Init PLY    : {args.init_ply}
  Cameras     : {args.cameras}
  Views       : {len(render_dataset)} (train={len(train_dataset)}, test={len(test_dataset)}, eval_split={args.eval_split}, llffhold={args.llffhold}, sort={split_sort_mode})
  Initial N   : {init_n}
  cap_max     : {args.cap_max}
  Steps       : {args.steps}
  Seed        : {args.seed}
  proper_ewa  : {int(proper_ewa)}
  VK parity   : {args.vk_parity_mode}
  lambda_dssim: {args.lambda_dssim:.3f}
  pos LR      : {args.pos_lr_init:.8f} -> {args.pos_lr_final:.8f} (scale={args.spatial_lr_scale:.3f})
  opacity_reg : {args.opacity_reg:g}
  scale_reg   : {args.scale_reg:g}
  noise_lr    : {args.noise_lr:g}
  SH warmup   : {args.sh_degree_warmup}
  render SH   : {render_sh_degree}
  CUDA gate   : iter > {args.cuda_densify_from_iter}, iter < {args.densify_until}, iter % {args.densify_interval} == 0
  VK gate     : step >= {vk_from_arg}, step <= {vk_until_arg}, step % {args.densify_interval} == 0
  VK opacity reset interval: {args.opacity_reset_interval}
  CUDA final N: {cuda_n}
  VK final N  : {vk_n}
  CUDA final loss: {cuda_final_loss:.6f}
  VK final loss  : {vk_final_loss_text}
  CUDA events : {events[:20]}{' ...' if len(events) > 20 else ''}
====================================================
  Timing CUDA train loop : {cuda_timing['train_seconds']:8.2f}s
  Timing VK train CLI    : {vk_timing['train_seconds']:8.2f}s  CUDA/VK={train_speedup:6.2f}x
  Timing CUDA all-view render: {cuda_timing['render_seconds']:8.2f}s
  Timing VK all-view render  : {vk_timing['render_seconds']:8.2f}s  CUDA/VK={render_speedup:6.2f}x
  Timing total train+render  : CUDA={cuda_total_seconds:8.2f}s  VK={vk_total_seconds:8.2f}s  CUDA/VK={total_speedup:6.2f}x
====================================================
{metric_report}
  VK saved vs train final train-view {final_train_view} (render-view {final_render_view}): PSNR={vk_saved_train['psnr']:6.2f} dB  mean_abs={vk_saved_train['mean_abs']:.6f}  p99={vk_saved_train['p99_abs']:.6f}  max={vk_saved_train['max_abs']:.6f}
  Gate status       : {'PASS' if not failed else 'FAIL — ' + '; '.join(failed)}
====================================================
"""
    metrics_row = {
        **config,
        "cuda_final_n": cuda_n,
        "vk_final_n": vk_n,
        "cuda_final_loss": cuda_final_loss,
        "vk_final_loss": vk_final_loss,
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
        "train_count": len(train_indices),
        "test_count": len(test_indices),
        "train_gt_psnr_gap": train_cuda_gt["psnr"] - train_vk_gt["psnr"],
        "test_gt_psnr_gap": None if test_cuda_gt is None else test_cuda_gt["psnr"] - test_vk_gt["psnr"],
        "final_train_view": final_train_view,
        "final_render_view": final_render_view,
        "gate_status": "PASS" if not failed else "FAIL",
        "gate_failures": "; ".join(failed),
    }
    add_metric_columns(metrics_row, "train_cuda_gt", train_cuda_gt)
    add_metric_columns(metrics_row, "train_vk_gt", train_vk_gt)
    add_metric_columns(metrics_row, "train_vk_cuda", train_vk_cuda)
    if test_indices:
        add_metric_columns(metrics_row, "test_cuda_gt", test_cuda_gt)
        add_metric_columns(metrics_row, "test_vk_gt", test_vk_gt)
        add_metric_columns(metrics_row, "test_vk_cuda", test_vk_cuda)
    else:
        for prefix in ("test_cuda_gt", "test_vk_gt", "test_vk_cuda"):
            for key in ("psnr", "mean_view_psnr", "min_view_psnr", "mean_abs", "p99_abs", "max_abs"):
                metrics_row[f"{prefix}_{key}"] = None
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
