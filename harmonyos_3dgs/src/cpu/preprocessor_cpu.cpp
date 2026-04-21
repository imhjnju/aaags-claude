#include "cpu/preprocessor_cpu.h"
#include "math_utils.h"
#include "sh_eval.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

static constexpr float ALPHA_THRESHOLD = 1.0f / 255.0f;

PreprocessOutput PreprocessorCPU::process(const GaussianData& g, const Camera& cam,
                                          const RenderConfig& cfg, FrameAllocator& alloc,
                                          ForwardCache* cache) {
    int N = g.count;
    PreprocessOutput out;
    out.means2D = alloc.allocate_array<float>(N * 2);
    out.depths = alloc.allocate_array<float>(N);
    out.conics = alloc.allocate_array<float>(N * 3);
    out.opacities_2d = alloc.allocate_array<float>(N);
    out.rgb = alloc.allocate_array<float>(N * 3);
    out.radii = alloc.allocate_array<int>(N);
    out.radius_f = alloc.allocate_array<float>(static_cast<std::size_t>(N));
    out.tiles_touched = alloc.allocate_array<int>(N);
    out.eval_3D = cfg.eval_3D;

    if (cfg.eval_3D) {
        out.gauss2screen = alloc.allocate_array<float>(N * 16);
        out.cov3D_inv = alloc.allocate_array<float>(N * 6);
        out.mean_offset = alloc.allocate_array<float>(N * 3);
    } else {
        out.gauss2screen = nullptr;
        out.cov3D_inv = nullptr;
        out.mean_offset = nullptr;
    }

    // Allocate ForwardCache arrays if cache is requested
    if (cache && !cfg.eval_3D) {
        cache->cov3D     = alloc.allocate_array<float>(N * 6);
        cache->p_view    = alloc.allocate_array<float>(N * 3);
        cache->p_hom_w   = alloc.allocate_array<float>(N);
        cache->cov2D     = alloc.allocate_array<float>(N * 3);
        cache->cov2D_det = alloc.allocate_array<float>(N);
        // Zero-initialize so culled Gaussians have known values
        std::memset(cache->cov3D,     0, N * 6 * sizeof(float));
        std::memset(cache->p_view,    0, N * 3 * sizeof(float));
        std::memset(cache->p_hom_w,   0, N * sizeof(float));
        std::memset(cache->cov2D,     0, N * 3 * sizeof(float));
        std::memset(cache->cov2D_det, 0, N * sizeof(float));
    }

    float focal_x = cam.width / (2.0f * cam.tan_fovx);
    float focal_y = cam.height / (2.0f * cam.tan_fovy);
    int grid_x = (cam.width + cfg.tile_w - 1) / cfg.tile_w;
    int grid_y = (cam.height + cfg.tile_h - 1) / cfg.tile_h;

    int sh_degree = std::min(cfg.sh_degree, g.sh_degree);

    for (int i = 0; i < N; i++) {
        out.radii[i] = 0;
        out.tiles_touched[i] = 0;

        float p_view[3];
        transformPoint4x3(&g.positions[i*3], cam.view_matrix, p_view);

        if (cfg.eval_3D) {
            // === AAA-Gaussians 3D evaluation path ===
            if (p_view[2] < 0.2f)
                continue;

            float opacity = g.opacities[i];
            float focal = std::max(focal_x, focal_y);
            float filter_3d = (g.filter_3D != nullptr) ? g.filter_3D[i] : 0.0f;

            float gauss2screen[16], gauss2view[16];
            float dilation_factor = computeGauss2Screen(
                &g.positions[i*3], &g.scales[i*3], &g.rotations[i*4],
                cfg.scale_modifier, cam.cam_pos,
                cam.viewproj_matrix, cam.view_matrix,
                focal, 0.3f, filter_3d,
                cam.width, cam.height,
                gauss2screen, gauss2view);

            opacity *= dilation_factor;
            if (opacity < ALPHA_THRESHOLD)
                continue;

            float opacity_power_threshold = std::log(opacity / ALPHA_THRESHOLD);
            float cutoff = std::min(11.11f, 2.0f * opacity_power_threshold);

            // Camera position in Gaussian space
            float R[3][3];
            quat2mat(&g.rotations[i*4], R);
            float sm = std::sqrt(cfg.scale_modifier);
            float sd[3];
            // Recompute scale_dilated for campos_gauss
            float scale_dilated[3] = {g.scales[i*3], g.scales[i*3+1], g.scales[i*3+2]};
            {
                float mean_view[3];
                transformPoint4x3(&g.positions[i*3], cam.view_matrix, mean_view);
                float scale_mip = (mean_view[2] / focal) * (mean_view[2] / focal) * 0.3f;
                scale_mip = std::max(filter_3d * filter_3d, scale_mip);
                for (int j = 0; j < 3; j++)
                    scale_dilated[j] = std::sqrt(g.scales[i*3+j]*g.scales[i*3+j] + scale_mip);
            }
            for (int j = 0; j < 3; j++)
                sd[j] = scale_dilated[j] * sm;

            // Compute inverse 3D covariance for per-tile depth key
            computeCov3DInv(scale_dilated, cfg.scale_modifier, &g.rotations[i*4],
                            &out.cov3D_inv[i * 6]);

            // Store mean_offset for per-tile depth
            out.mean_offset[i*3]   = g.positions[i*3]   - cam.cam_pos[0];
            out.mean_offset[i*3+1] = g.positions[i*3+1] - cam.cam_pos[1];
            out.mean_offset[i*3+2] = g.positions[i*3+2] - cam.cam_pos[2];

            // S_inv = diag(1/sd)
            // world2gauss = S_inv * R
            // campos_gauss = world2gauss * (cam_pos - mean3D)
            float diff[3] = {cam.cam_pos[0] - g.positions[i*3],
                             cam.cam_pos[1] - g.positions[i*3+1],
                             cam.cam_pos[2] - g.positions[i*3+2]};
            float campos_gauss[3];
            for (int j = 0; j < 3; j++) {
                // world2gauss row j = (1/sd[j]) * R[j]
                campos_gauss[j] = (R[j][0]*diff[0] + R[j][1]*diff[1] + R[j][2]*diff[2]) / sd[j];
            }

            float dist_sq = campos_gauss[0]*campos_gauss[0] +
                             campos_gauss[1]*campos_gauss[1] +
                             campos_gauss[2]*campos_gauss[2];
            if (dist_sq < cutoff)
                continue;  // Camera inside ellipsoid

            // 3D frustum culling
            float max_pos_depth;
            float max_contrib = maxContribGaussianFrustum3D(
                0.0f, 0.0f, (float)(cam.width-1), (float)(cam.height-1),
                gauss2screen, max_pos_depth);
            if (max_contrib > opacity_power_threshold)
                continue;

            // Use NDC-projected center for tile assignment (consistent with 2D path)
            // and AABB extent as radius for tile coverage
            float p_hom[4];
            transformPoint4x4(&g.positions[i*3], cam.viewproj_matrix, p_hom);
            float p_w_inv = 1.0f / (p_hom[3] + 1e-7f);
            float pixel_x = ndc2Pix(p_hom[0]*p_w_inv, cam.width);
            float pixel_y = ndc2Pix(p_hom[1]*p_w_inv, cam.height);

            // Screen-space AABB for extent estimation
            float mean2D_aabb[2], extent_aabb[2];
            if (!computeAABBScreen(gauss2screen, cutoff, mean2D_aabb, extent_aabb)) {
                // Fallback: view-space AABB (handles near-plane overflow)
                if (!computeAABBView(gauss2view, p_view, focal_x, focal_y,
                                     (float)cam.width, (float)cam.height, cutoff,
                                     mean2D_aabb, extent_aabb))
                    continue;
            }

            int my_radius = (int)std::ceil(std::max(extent_aabb[0], extent_aabb[1]));
            my_radius = std::max(my_radius, 1);

            float point_image[2] = {pixel_x, pixel_y};
            int rect_min_arr[2], rect_max_arr[2];
            getRect(point_image, my_radius, grid_x, grid_y, cfg.tile_w, cfg.tile_h, rect_min_arr, rect_max_arr);

            if ((rect_max_arr[0] - rect_min_arr[0]) * (rect_max_arr[1] - rect_min_arr[1]) == 0)
                continue;

            // SH color evaluation
            computeColorFromSH(sh_degree, g.max_coeffs,
                              &g.sh_coeffs[i * g.max_coeffs * 3],
                              &g.positions[i*3], cam.cam_pos,
                              &out.rgb[i*3], cfg.training);

            // Store outputs — use view-space z for depth sorting
            out.depths[i] = p_view[2];
            out.radii[i] = my_radius;
            out.means2D[i*2] = pixel_x;
            out.means2D[i*2+1] = pixel_y;
            out.opacities_2d[i] = opacity;
            out.tiles_touched[i] = (rect_max_arr[1] - rect_min_arr[1]) * (rect_max_arr[0] - rect_min_arr[0]);

            // Store gauss2screen matrix (row-major, 16 floats)
            for (int j = 0; j < 16; j++)
                out.gauss2screen[i*16+j] = gauss2screen[j];

        } else {
            // === Standard 2D 3DGS path (unchanged) ===
            if (p_view[2] <= 0.2f)
                continue;

            // NDC projection with epsilon
            float p_hom[4];
            transformPoint4x4(&g.positions[i*3], cam.viewproj_matrix, p_hom);
            float p_w = 1.0f / (p_hom[3] + 1e-7f);
            float p_ndc[3] = {p_hom[0]*p_w, p_hom[1]*p_w, p_hom[2]*p_w};

            float pixel_x = ndc2Pix(p_ndc[0], cam.width);
            float pixel_y = ndc2Pix(p_ndc[1], cam.height);

            float cov3d[6];
            computeCov3D(&g.scales[i*3], cfg.scale_modifier, &g.rotations[i*4], cov3d);

            float cov2d[3];
            computeCov2D(&g.positions[i*3], cov3d, cam.view_matrix,
                         focal_x, focal_y, cam.tan_fovx, cam.tan_fovy, cov2d);

            float det_cov = cov2d[0] * cov2d[2] - cov2d[1] * cov2d[1];
            cov2d[0] += 0.3f;
            cov2d[2] += 0.3f;
            float det_cov_plus_h = cov2d[0] * cov2d[2] - cov2d[1] * cov2d[1];
            float h_conv_scaling = 1.0f;
            if (cfg.antialiasing)
                h_conv_scaling = std::sqrt(std::max(0.000025f, det_cov / det_cov_plus_h));

            float det = det_cov_plus_h;
            if (det == 0.0f) continue;
            float det_inv = 1.0f / det;
            float conic[3] = {cov2d[2]*det_inv, -cov2d[1]*det_inv, cov2d[0]*det_inv};

            float mid = 0.5f * (cov2d[0] + cov2d[2]);
            float disc = std::max(0.01f, mid * mid - det);
            float lambda1 = mid + std::sqrt(disc);
            float lambda2 = mid - std::sqrt(disc);
            float radius_f_val = 3.33f * std::sqrt(std::max(lambda1, lambda2));
            int my_radius = static_cast<int>(std::ceil(radius_f_val));

            // Note: no max_screen_dim cull — matches CUDA which renders
            // large-radius Gaussians (near camera) without this check.

            float point_image[2] = {pixel_x, pixel_y};
            int rect_min[2], rect_max[2];
            {
                float r = radius_f_val;
                rect_min[0] = std::min(grid_x, std::max(0, static_cast<int>(std::floor((point_image[0] - r) / cfg.tile_w))));
                rect_min[1] = std::min(grid_y, std::max(0, static_cast<int>(std::floor((point_image[1] - r) / cfg.tile_h))));
                rect_max[0] = std::min(grid_x, std::max(0, static_cast<int>(std::ceil((point_image[0] + r) / cfg.tile_w))));
                rect_max[1] = std::min(grid_y, std::max(0, static_cast<int>(std::ceil((point_image[1] + r) / cfg.tile_h))));
            }

            if ((rect_max[0] - rect_min[0]) * (rect_max[1] - rect_min[1]) == 0)
                continue;

            computeColorFromSH(sh_degree, g.max_coeffs,
                              &g.sh_coeffs[i * g.max_coeffs * 3],
                              &g.positions[i*3], cam.cam_pos,
                              &out.rgb[i*3], cfg.training);

            out.depths[i] = p_view[2];
            out.radii[i] = my_radius;
            if (out.radius_f) out.radius_f[i] = radius_f_val;
            out.means2D[i*2] = pixel_x;
            out.means2D[i*2+1] = pixel_y;
            out.conics[i*3] = conic[0];
            out.conics[i*3+1] = conic[1];
            out.conics[i*3+2] = conic[2];
            out.opacities_2d[i] = g.opacities[i] * h_conv_scaling;
            out.tiles_touched[i] = (rect_max[1]-rect_min[1]) * (rect_max[0]-rect_min[0]);

            // Save intermediates for backward pass
            if (cache) {
                for (int j = 0; j < 6; j++) cache->cov3D[i*6+j] = cov3d[j];
                cache->p_view[i*3]   = p_view[0];
                cache->p_view[i*3+1] = p_view[1];
                cache->p_view[i*3+2] = p_view[2];
                cache->p_hom_w[i]    = p_hom[3];
                cache->cov2D[i*3]    = cov2d[0];  // after +0.3 filter
                cache->cov2D[i*3+1]  = cov2d[1];
                cache->cov2D[i*3+2]  = cov2d[2];
                cache->cov2D_det[i]  = det;       // det after +0.3 filter
            }
        }
    }
    return out;
}
