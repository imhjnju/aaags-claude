#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np

W = 720
H = 960
BLOCK = 16
ALPHA_THRESHOLD = 1.0 / 255.0
T_MIN = 0.0001
Z_NEAR = -1.0
Z_FAR = 1.0


def load_f32(path):
    return np.fromfile(path, dtype=np.float32)


def load_u32(path):
    return np.fromfile(path, dtype=np.uint32)


def load_chw(path):
    return load_f32(path).reshape(3, H * W)


def load_cuda_hwc_as_chw(path):
    hwc = load_f32(path).reshape(H * W, 3)
    return hwc.T.copy()


def max_contrib_ray(g2s, x, y):
    plane_a = g2s[0] - g2s[3] * float(x)
    plane_b = g2s[1] - g2s[3] * float(y)
    result = max_contrib_ray_planes(plane_a, plane_b, allow_degenerate_zero=True)
    if result is None:
        return 0.0, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return result


def max_contrib_ray_planes(plane_a, plane_b, allow_degenerate_zero=False):
    dx = plane_a[1] * plane_b[2] - plane_a[2] * plane_b[1]
    dy = plane_a[2] * plane_b[0] - plane_a[0] * plane_b[2]
    dz = plane_a[0] * plane_b[1] - plane_a[1] * plane_b[0]
    mx = plane_a[3] * plane_b[0] - plane_a[0] * plane_b[3]
    my = plane_a[3] * plane_b[1] - plane_a[1] * plane_b[3]
    mz = plane_a[3] * plane_b[2] - plane_a[2] * plane_b[3]
    dd = dx * dx + dy * dy + dz * dz
    if dd < 1e-8:
        if allow_degenerate_zero:
            return 0.0, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return None
    m_div_dd = np.array([mx / dd, my / dd, mz / dd], dtype=np.float64)
    max_pos = np.empty(4, dtype=np.float64)
    max_pos[0] = dy * m_div_dd[2] - dz * m_div_dd[1]
    max_pos[1] = dz * m_div_dd[0] - dx * m_div_dd[2]
    max_pos[2] = dx * m_div_dd[1] - dy * m_div_dd[0]
    max_pos[3] = 1.0
    dist2 = mx * m_div_dd[0] + my * m_div_dd[1] + mz * m_div_dd[2]
    return dist2, max_pos


def in_screen_range(pos, range_from, extent, dim):
    d = pos[dim] - range_from[dim] * pos[3]
    return d > 0.0 and d < extent[dim] * pos[3], d


def max_contrib_plane_screen(plane, g2s):
    n2 = float(np.dot(plane[:3], plane[:3]))
    if n2 <= 0.0:
        return math.inf, None
    norm = plane[3] / n2
    pos_gauss = np.array([-plane[0] * norm, -plane[1] * norm, -plane[2] * norm, 1.0], dtype=np.float64)
    return plane[3] * norm, g2s @ pos_gauss


def max_contrib_ray_screen(plane_a, plane_b, g2s):
    result = max_contrib_ray_planes(plane_a, plane_b, allow_degenerate_zero=False)
    if result is None:
        return math.inf, None
    contrib, pos_gauss = result
    return contrib, g2s @ pos_gauss


def max_contrib_gaussian_frustum_3d(g2s, rect_min, rect_max):
    range_from = np.array([rect_min[0], rect_min[1], Z_NEAR], dtype=np.float64)
    extent = np.array([rect_max[0] - rect_min[0], rect_max[1] - rect_min[1], Z_FAR - Z_NEAR], dtype=np.float64)
    mean_screen = np.array([g2s[0, 3], g2s[1, 3], g2s[2, 3], g2s[3, 3]], dtype=np.float64)

    between_x, d_x = in_screen_range(mean_screen, range_from, extent, 0)
    between_y, d_y = in_screen_range(mean_screen, range_from, extent, 1)
    between_z, d_z = in_screen_range(mean_screen, range_from, extent, 2)
    if between_x and between_y and between_z:
        return 0.0, mean_screen[2] / mean_screen[3]

    d = np.array([d_x, d_y, d_z], dtype=np.float64)
    dx = math.copysign(extent[0] * 0.5, d[0] - extent[0] * 0.5 * mean_screen[3])
    dy = math.copysign(extent[1] * 0.5, d[1] - extent[1] * 0.5 * mean_screen[3])
    cx = range_from[0] + extent[0] * 0.5 + dx
    cy = range_from[1] + extent[1] * 0.5 + dy

    closer_plane_x = g2s[0] - g2s[3] * cx
    closer_plane_y = g2s[1] - g2s[3] * cy
    best = math.inf
    best_pos = None

    def consider(contrib, pos, dims):
        nonlocal best, best_pos
        if pos is None or not np.isfinite(contrib):
            return
        for dim in dims:
            ok, _ = in_screen_range(pos, range_from, extent, dim)
            if not ok:
                return
        if contrib < best:
            best = contrib
            best_pos = pos

    consider(*max_contrib_plane_screen(closer_plane_x, g2s), dims=(1, 2))
    consider(*max_contrib_plane_screen(closer_plane_y, g2s), dims=(0, 2))
    consider(*max_contrib_ray_screen(closer_plane_x, closer_plane_y, g2s), dims=(2,))

    other_plane_y = g2s[1] - g2s[3] * (range_from[1] + extent[1] * 0.5 - dy)
    consider(*max_contrib_ray_screen(closer_plane_x, other_plane_y, g2s), dims=(2,))

    other_plane_x = g2s[0] - g2s[3] * (range_from[0] + extent[0] * 0.5 - dx)
    consider(*max_contrib_ray_screen(other_plane_x, closer_plane_y, g2s), dims=(2,))

    if best_pos is None:
        return math.inf, math.inf
    return 0.5 * best, best_pos[2] / best_pos[3]


def passes_tail_frustum_alpha(g2s, opacity, rect_min, rect_max):
    factor, _ = max_contrib_gaussian_frustum_3d(g2s, rect_min, rect_max)
    if not np.isfinite(factor):
        return False
    alpha = min(0.99, float(opacity) * math.exp(-factor))
    return alpha >= ALPHA_THRESHOLD


class ReplayModel:
    def __init__(self, build_dir, cuda_path):
        self.build_dir = Path(build_dir)
        self.cuda = load_cuda_hwc_as_chw(cuda_path)
        self.vk = load_chw(self.build_dir / "vk_image.raw")
        self.g2s = load_f32(self.build_dir / "vk_pre_gauss2screen.raw").reshape(-1, 4, 4)
        self.opacity = load_f32(self.build_dir / "vk_pre_opacities_2d.raw")
        self.rgb = load_f32(self.build_dir / "vk_pre_rgb.raw").reshape(-1, 3)
        self.values = load_u32(self.build_dir / "vk_bin_values_sorted.raw")
        self.ranges = load_u32(self.build_dir / "vk_bin_tile_ranges.raw").reshape(-1, 2)
        self.replay_offsets = load_u32(self.build_dir / "vk_replay_order_offsets_cam0.raw")
        self.replay_gids = load_u32(self.build_dir / "vk_replay_order_gids_cam0.raw")

    def color_for_px(self, image, x, y):
        px = y * W + x
        return image[:, px].astype(np.float64)

    def pixel_eval(self, gid, x, y):
        g2s = self.g2s[int(gid)].astype(np.float64)
        dist2, max_pos = max_contrib_ray(g2s, x, y)
        power = -0.5 * dist2
        if power > 0.0:
            return None
        alpha = min(0.99, float(self.opacity[int(gid)]) * math.exp(power))
        if alpha < ALPHA_THRESHOLD:
            return None
        w_ndc = float(np.dot(g2s[3], max_pos))
        z_ndc = float(np.dot(g2s[2], max_pos))
        if z_ndc < -w_ndc or z_ndc > w_ndc:
            return None
        depth = z_ndc / w_ndc if w_ndc > 1e-6 else z_ndc
        return depth, alpha, self.rgb[int(gid)].astype(np.float64)

    def depth_at(self, gid, x, y):
        g2s = self.g2s[int(gid)].astype(np.float64)
        _, max_pos = max_contrib_ray(g2s, x, y)
        w_ndc = float(np.dot(g2s[3], max_pos))
        z_ndc = float(np.dot(g2s[2], max_pos))
        if z_ndc < -w_ndc or z_ndc > w_ndc:
            return None
        return z_ndc / w_ndc if w_ndc > 1e-6 else z_ndc

    def sort_depth_at(self, gid, x, y):
        g2s = self.g2s[int(gid)].astype(np.float64)
        _, max_pos = max_contrib_ray(g2s, x, y)
        w_ndc = float(np.dot(g2s[3], max_pos))
        z_ndc = float(np.dot(g2s[2], max_pos))
        return z_ndc / w_ndc if w_ndc > 1e-6 else z_ndc

    def tile_candidates(self, x, y):
        tile_x = x // BLOCK
        tile_y = y // BLOCK
        num_tiles_x = (W + BLOCK - 1) // BLOCK
        tile_id = tile_y * num_tiles_x + tile_x
        start, end = self.ranges[tile_id]
        return self.values[int(start):int(end)]

    def replay_sequence(self, x, y):
        px = y * W + x
        start = int(self.replay_offsets[px])
        end = int(self.replay_offsets[px + 1])
        return self.replay_gids[start:end]

    def blend_direct(self, gids, x, y):
        color = np.zeros(3, dtype=np.float64)
        T = 1.0
        order = []
        for gid in gids:
            ev = self.pixel_eval(int(gid), x, y)
            if ev is None:
                continue
            _, alpha, rgb = ev
            test_T = T * (1.0 - alpha)
            if test_T < T_MIN:
                break
            color += rgb * alpha * T
            T = test_T
            order.append(int(gid))
        return color, T, order

    def blend_head_stream(self, stream, x, y, head_window):
        color = np.zeros(3, dtype=np.float64)
        T = 1.0
        head = []
        order = []
        active = True

        def flush_one():
            nonlocal color, T, active
            if not head or not active:
                return
            depth, gid, alpha, rgb = head.pop(0)
            test_T = T * (1.0 - alpha)
            if test_T < T_MIN:
                active = False
                return
            color += rgb * alpha * T
            T = test_T
            order.append(int(gid))

        def push_gid(gid):
            if not active:
                return
            if len(head) == head_window:
                flush_one()
            if not active:
                return
            ev = self.pixel_eval(int(gid), x, y)
            if ev is None:
                return
            depth, alpha, rgb = ev
            item = (depth, int(gid), alpha, rgb)
            pos = len(head)
            for i, existing in enumerate(head):
                if depth < existing[0]:
                    pos = i
                    break
            head.insert(pos, item)

        for gid in stream:
            push_gid(int(gid))
            if not active:
                break
        while active and head:
            flush_one()
        return color, T, order

    def mid_stream(self, candidates, x, y, tail_alpha_cull=False):
        local_x = x % BLOCK
        local_y = y % BLOCK
        sub_x = local_x // 4
        sub_y = local_y // 4
        quad_x = (local_x % 4) // 2
        quad_y = (local_y % 4) // 2
        tile_x = x // BLOCK
        tile_y = y // BLOCK
        tail_base_x = tile_x * BLOCK + sub_x * 4
        tail_base_y = tile_y * BLOCK + sub_y * 4
        tail_px = tail_base_x + 2.0
        tail_py = tail_base_y + 2.0
        mid_px = tail_base_x + 1.0 + 2.0 * quad_x
        mid_py = tail_base_y + 1.0 + 2.0 * quad_y
        rect_min = (float(tail_base_x), float(tail_base_y))
        rect_max = (float(tail_base_x + 3), float(tail_base_y + 3))
        tail = []
        mid_backlog = []
        out = []

        def key_with_order(items):
            return sorted(items, key=lambda it: (it[0], it[2]))

        def drain_tail_16():
            nonlocal tail
            chunk = tail[:16]
            tail = tail[16:]
            for i in range(0, len(chunk), 4):
                group = []
                for _, gid, order_idx in chunk[i:i + 4]:
                    d = self.sort_depth_at(gid, mid_px, mid_py)
                    if np.isfinite(d):
                        group.append((d, int(gid), order_idx))
                push_mid_group(key_with_order(group))

        def push_mid_group(group):
            nonlocal mid_backlog
            if not group:
                return
            if not mid_backlog:
                mid_backlog = group
                return
            merged = key_with_order(mid_backlog + group)
            out.extend([gid for _, gid, _ in merged[:4]])
            mid_backlog = merged[4:]

        order_idx = 0
        for start in range(0, len(candidates), 32):
            new_items = []
            for gid in candidates[start:start + 32]:
                gid = int(gid)
                g2s = self.g2s[gid].astype(np.float64)
                if tail_alpha_cull and not passes_tail_frustum_alpha(g2s, self.opacity[gid], rect_min, rect_max):
                    order_idx += 1
                    continue
                d = self.sort_depth_at(gid, tail_px, tail_py)
                if np.isfinite(d):
                    new_items.append((d, gid, order_idx))
                order_idx += 1
            tail = key_with_order(tail + key_with_order(new_items))
            while len(tail) > 32:
                drain_tail_16()
        while tail:
            drain_tail_16()
        if mid_backlog:
            out.extend([gid for _, gid, _ in mid_backlog])
        return out

    def top_errors(self, count):
        diff = self.vk - self.cuda
        err = np.sqrt(np.mean(diff * diff, axis=0))
        top = np.argsort(err)[::-1][:count]
        return [(int(px % W), int(px // W), float(err[px])) for px in top]


def rmse(a, b):
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="harmonyos_3dgs/build")
    parser.add_argument("--cuda", default="harmonyos_3dgs/tests/golden/basketball/cam0/cuda_image.raw")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--tail-alpha-cull", action="store_true")
    args = parser.parse_args()

    model = ReplayModel(args.build_dir, args.cuda)
    wins = {"vk_replay": 0, "raw_tile": 0, "mid_h4": 0, "mid_h8": 0}
    sums = {"vk_replay": 0.0, "raw_tile": 0.0, "mid_h4": 0.0, "mid_h8": 0.0}
    exact = {"vk_replay": 0, "raw_tile": 0, "mid_h4": 0, "mid_h8": 0}
    replay_vk_errs = []

    print("# L1b MID replay diagnostic")
    print(f"top={args.top} build_dir={args.build_dir} tail_alpha_cull={args.tail_alpha_cull}")
    print("rank px err_vk_cuda  e_replay e_raw e_mid4 e_mid8  best counts(seq replay/raw/mid4/mid8)")

    for rank, (x, y, err0) in enumerate(model.top_errors(args.top), 1):
        cuda_color = model.color_for_px(model.cuda, x, y)
        vk_color = model.color_for_px(model.vk, x, y)
        replay_seq = model.replay_sequence(x, y)
        candidates = model.tile_candidates(x, y)
        mid_seq = model.mid_stream(candidates, x, y, tail_alpha_cull=args.tail_alpha_cull)

        replay_color, _, replay_order = model.blend_direct(replay_seq, x, y)
        raw_color, _, raw_order = model.blend_direct(candidates, x, y)
        mid4_color, _, mid4_order = model.blend_head_stream(mid_seq, x, y, 4)
        mid8_color, _, mid8_order = model.blend_head_stream(mid_seq, x, y, 8)

        errors = {
            "vk_replay": rmse(replay_color, cuda_color),
            "raw_tile": rmse(raw_color, cuda_color),
            "mid_h4": rmse(mid4_color, cuda_color),
            "mid_h8": rmse(mid8_color, cuda_color),
        }
        for name, value in errors.items():
            sums[name] += value
            if value < 1e-4:
                exact[name] += 1
        best = min(errors, key=errors.get)
        wins[best] += 1
        replay_vk_errs.append(rmse(replay_color, vk_color))

        print(
            f"{rank:2d} ({x:4d},{y:4d}) {err0:.6f}  "
            f"{errors['vk_replay']:.6f} {errors['raw_tile']:.6f} "
            f"{errors['mid_h4']:.6f} {errors['mid_h8']:.6f}  "
            f"{best:9s} {len(replay_order):3d}/{len(raw_order):3d}/{len(mid4_order):3d}/{len(mid8_order):3d}"
        )
        if rank == 1:
            print("  replay:", " ".join(map(str, replay_order[:32])))
            print("  raw   :", " ".join(map(str, raw_order[:32])))
            print("  mid_h4:", " ".join(map(str, mid4_order[:32])))
            print("  mid_h8:", " ".join(map(str, mid8_order[:32])))

    print("wins", " ".join(f"{k}={v}" for k, v in wins.items()))
    print("mean_error", " ".join(f"{k}={sums[k] / args.top:.6f}" for k in sums))
    print("exact_lt_1e-4", " ".join(f"{k}={v}" for k, v in exact.items()))
    print(f"replay_vs_vk_rmse max={max(replay_vk_errs):.9f} mean={np.mean(replay_vk_errs):.9f}")


if __name__ == "__main__":
    main()
