#include "cpu/rasterizer_backward_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

void RasterizerBackwardCPU::backward(const PreprocessOutput& pre, const BinningOutput& bin,
                                      const Camera& cam, const RenderConfig& cfg,
                                      const ForwardCache& cache, const float* d_image,
                                      RasterGradOutput& rgrad) {
    int grid_x = (cam.width + cfg.tile_w - 1) / cfg.tile_w;
    int grid_y = (cam.height + cfg.tile_h - 1) / cfg.tile_h;

    // Per-contributor info saved during forward replay
    struct ContribInfo {
        uint32_t gauss_idx;
        float alpha;
    };

    for (int ty = 0; ty < grid_y; ty++) {
        for (int tx = 0; tx < grid_x; tx++) {
            int tile_id = ty * grid_x + tx;
            uint32_t range_start = bin.tile_ranges[tile_id * 2];
            uint32_t range_end   = bin.tile_ranges[tile_id * 2 + 1];

            int px_min_x = tx * cfg.tile_w;
            int px_min_y = ty * cfg.tile_h;
            int px_max_x = std::min(px_min_x + cfg.tile_w, cam.width);
            int px_max_y = std::min(px_min_y + cfg.tile_h, cam.height);

            for (int py = px_min_y; py < px_max_y; py++) {
                for (int px = px_min_x; px < px_max_x; px++) {
                    int pix = py * cam.width + px;
                    int n_contrib = cache.n_contrib[pix];
                    if (n_contrib == 0) continue;

                    float T_final = cache.T_final[pix];
                    float d_C[3] = { d_image[pix*3], d_image[pix*3+1], d_image[pix*3+2] };

                    // --- Forward replay to collect contributors ---
                    std::vector<ContribInfo> contribs;
                    contribs.reserve(n_contrib);
                    {
                        float T = 1.0f;
                        int count = 0;
                        for (uint32_t j = range_start; j < range_end; j++) {
                            uint32_t idx = bin.values_sorted[j];

                            float dx = pre.means2D[idx*2]   - (float)px;
                            float dy = pre.means2D[idx*2+1] - (float)py;
                            float con_a = pre.conics[idx*3];
                            float con_b = pre.conics[idx*3+1];
                            float con_c = pre.conics[idx*3+2];
                            float power = -0.5f*(con_a*dx*dx + con_c*dy*dy) - con_b*dx*dy;
                            if (power > 0.0f) continue;

                            float alpha = std::min(0.99f, pre.opacities_2d[idx] * std::exp(power));
                            if (alpha < 1.0f/255.0f) continue;

                            float test_T = T * (1.0f - alpha);
                            if (test_T < 0.0001f) break;

                            contribs.push_back({idx, alpha});
                            T = test_T;
                            count++;
                        }
                    }

                    // --- Backward pass (back-to-front) ---
                    float T_accum = T_final;
                    // accum_rec: the expected color looking along the ray from behind.
                    // Initialized to bg (the color contribution behind all Gaussians).
                    float accum_rec[3] = {
                        cfg.bg_color[0],
                        cfg.bg_color[1],
                        cfg.bg_color[2]
                    };

                    for (int i = (int)contribs.size() - 1; i >= 0; i--) {
                        uint32_t gauss_idx = contribs[i].gauss_idx;
                        float alpha = contribs[i].alpha;

                        // Recover transmittance before this Gaussian
                        float one_minus_alpha = 1.0f - alpha;
                        float T_i = (one_minus_alpha > 1e-6f) ? T_accum / one_minus_alpha : 0.0f;

                        // Recompute spatial quantities
                        float dx = pre.means2D[gauss_idx*2]   - (float)px;
                        float dy = pre.means2D[gauss_idx*2+1] - (float)py;
                        float con_a = pre.conics[gauss_idx*3];
                        float con_b = pre.conics[gauss_idx*3+1];
                        float con_c = pre.conics[gauss_idx*3+2];

                        // RGB gradient
                        for (int ch = 0; ch < 3; ch++)
                            rgrad.d_rgb[gauss_idx*3+ch] += alpha * T_i * d_C[ch];

                        // Alpha gradient using the correct volumetric rendering formula:
                        //   dL/d(alpha_i) = T_i * (color_i - accum_rec) . dL/dC
                        // where accum_rec is the "expected future color" behind this Gaussian.
                        float d_alpha = 0.0f;
                        for (int ch = 0; ch < 3; ch++)
                            d_alpha += T_i * (pre.rgb[gauss_idx*3+ch] - accum_rec[ch]) * d_C[ch];

                        // Update accum_rec (moving back to front):
                        //   accum_rec = alpha_i * color_i + (1 - alpha_i) * accum_rec
                        for (int ch = 0; ch < 3; ch++)
                            accum_rec[ch] = alpha * pre.rgb[gauss_idx*3+ch] + one_minus_alpha * accum_rec[ch];
                        T_accum = T_i;

                        // Chain: alpha -> power, opacity_2d
                        float power = -0.5f*(con_a*dx*dx + con_c*dy*dy) - con_b*dx*dy;
                        float exp_power = std::exp(power);
                        float d_power, d_opacity_2d;
                        if (alpha < 0.99f) {
                            d_power = d_alpha * pre.opacities_2d[gauss_idx] * exp_power;
                            d_opacity_2d = d_alpha * exp_power;
                        } else {
                            d_power = 0.0f;
                            d_opacity_2d = 0.0f;
                        }

                        rgrad.d_opacities_2d[gauss_idx] += d_opacity_2d;

                        // Chain: power -> means2D
                        rgrad.d_means2D[gauss_idx*2]     += d_power * (-con_a * dx - con_b * dy);
                        rgrad.d_means2D[gauss_idx*2 + 1] += d_power * (-con_c * dy - con_b * dx);

                        // Chain: power -> conics
                        rgrad.d_conics[gauss_idx*3]     += d_power * (-0.5f * dx * dx);
                        rgrad.d_conics[gauss_idx*3 + 1] += d_power * (-dx * dy);
                        rgrad.d_conics[gauss_idx*3 + 2] += d_power * (-0.5f * dy * dy);
                    }
                }
            }
        }
    }
}
