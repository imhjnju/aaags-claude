#include "cpu/tile_binner_cpu.h"
#include "math_utils.h"
#include <cmath>
#include <cstring>

BinningOutput TileBinnerCPU::bin(const PreprocessOutput& pre, int N,
                                  const Camera& cam, const RenderConfig& cfg,
                                  FrameAllocator& alloc) {
    BinningOutput out{};

    int grid_x = (cam.width + cfg.tile_w - 1) / cfg.tile_w;
    int grid_y = (cam.height + cfg.tile_h - 1) / cfg.tile_h;
    out.num_tiles = grid_x * grid_y;

    // Compute inverse viewproj for per-tile depth key
    float inverse_vp[16];
    bool have_inverse_vp = false;
    if (pre.eval_3D && pre.cov3D_inv) {
        have_inverse_vp = invertMatrix4x4(cam.viewproj_matrix, inverse_vp);
    }

    if (N == 0) {
        out.total_pairs = 0;
        out.keys_unsorted = nullptr;
        out.values_unsorted = nullptr;
        out.keys_sorted = nullptr;
        out.values_sorted = nullptr;
        out.tile_ranges = alloc.allocate_array<uint32_t>(out.num_tiles * 2);
        std::memset(out.tile_ranges, 0, out.num_tiles * 2 * sizeof(uint32_t));
        return out;
    }

    // Step 1: Compute inclusive prefix sum on tiles_touched
    int* offsets = alloc.allocate_array<int>(N);
    offsets[0] = pre.tiles_touched[0];
    for (int i = 1; i < N; i++) {
        offsets[i] = offsets[i - 1] + pre.tiles_touched[i];
    }

    // Step 2: Total pairs from last element of prefix sum
    int total_pairs = offsets[N - 1];
    out.total_pairs = total_pairs;

    // Step 3: Allocate output arrays
    if (total_pairs > 0) {
        out.keys_unsorted = alloc.allocate_array<uint64_t>(total_pairs);
        out.values_unsorted = alloc.allocate_array<uint32_t>(total_pairs);
        out.keys_sorted = alloc.allocate_array<uint64_t>(total_pairs);
        out.values_sorted = alloc.allocate_array<uint32_t>(total_pairs);
    } else {
        out.keys_unsorted = nullptr;
        out.values_unsorted = nullptr;
        out.keys_sorted = nullptr;
        out.values_sorted = nullptr;
    }
    out.tile_ranges = alloc.allocate_array<uint32_t>(out.num_tiles * 2);
    std::memset(out.tile_ranges, 0, out.num_tiles * 2 * sizeof(uint32_t));

    // Step 4: Generate unsorted keys and values
    // For eval_3D, also compute opacity_power_threshold for per-tile culling
    for (int i = 0; i < N; i++) {
        if (pre.radii[i] <= 0) continue;

        int off = (i == 0) ? 0 : offsets[i - 1];

        // Recompute tile rect
        float point[2] = {pre.means2D[i * 2], pre.means2D[i * 2 + 1]};
        int rect_min[2], rect_max[2];
        if (pre.radius_f) {
            float r = pre.radius_f[i];
            rect_min[0] = std::min(grid_x, std::max(0, static_cast<int>(std::floor((point[0] - r) / cfg.tile_w))));
            rect_min[1] = std::min(grid_y, std::max(0, static_cast<int>(std::floor((point[1] - r) / cfg.tile_h))));
            rect_max[0] = std::min(grid_x, std::max(0, static_cast<int>(std::ceil((point[0] + r) / cfg.tile_w))));
            rect_max[1] = std::min(grid_y, std::max(0, static_cast<int>(std::ceil((point[1] + r) / cfg.tile_h))));
        } else {
            getRect(point, pre.radii[i], grid_x, grid_y, cfg.tile_w, cfg.tile_h,
                    rect_min, rect_max);
        }

        // Encode depth as uint32
        uint32_t depth_bits;
        std::memcpy(&depth_bits, &pre.depths[i], sizeof(uint32_t));

        for (int y = rect_min[1]; y < rect_max[1]; y++) {
            for (int x = rect_min[0]; x < rect_max[0]; x++) {
                // Per-tile depth key via depthAlongRay (StopThePop corrects at pixel level)
                uint32_t tile_depth_bits = depth_bits;
                if (pre.eval_3D && !cfg.eval_3D_parity_mode && pre.cov3D_inv && have_inverse_vp) {
                    float tile_cx = (x + 0.5f) * cfg.tile_w;
                    float tile_cy = (y + 0.5f) * cfg.tile_h;
                    float viewdir[3];
                    pixelToWorldDir(tile_cx, tile_cy, cam.width, cam.height,
                                    inverse_vp, cam.cam_pos, viewdir);
                    float ptd = depthAlongRay(&pre.cov3D_inv[i * 6],
                                              &pre.mean_offset[i * 3], viewdir);
                    if (ptd > 0.0f)
                        std::memcpy(&tile_depth_bits, &ptd, sizeof(uint32_t));
                }

                uint32_t tile_id = static_cast<uint32_t>(y * grid_x + x);
                uint64_t key = (static_cast<uint64_t>(tile_id) << 32) | tile_depth_bits;
                out.keys_unsorted[off] = key;
                out.values_unsorted[off] = static_cast<uint32_t>(i);
                off++;
            }
        }
    }

    return out;
}
