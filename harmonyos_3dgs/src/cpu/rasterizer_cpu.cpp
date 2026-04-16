#include "cpu/rasterizer_cpu.h"
#include "math_utils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void RasterizerCPU::rasterize(const PreprocessOutput& pre, const BinningOutput& bin,
                               const Camera& cam, const RenderConfig& cfg,
                               float* out_img, float* out_depth,
                               ForwardCache* cache, FrameAllocator* allocator) {
    int grid_x = (cam.width + cfg.tile_w - 1) / cfg.tile_w;
    int grid_y = (cam.height + cfg.tile_h - 1) / cfg.tile_h;

    // Inverse viewproj for per-pixel ray (StopThePop)
    float inverse_vp[16];
    bool have_inverse_vp = false;
    if (pre.eval_3D && pre.cov3D_inv) {
        have_inverse_vp = invertMatrix4x4(cam.viewproj_matrix, inverse_vp);
    }

    // Allocate ForwardCache per-pixel arrays if requested
    if (cache && allocator) {
        int num_pixels = cam.width * cam.height;
        cache->T_final   = allocator->allocate_array<float>(num_pixels);
        cache->n_contrib = allocator->allocate_array<int>(num_pixels);
        std::memset(cache->T_final,   0, num_pixels * sizeof(float));
        std::memset(cache->n_contrib, 0, num_pixels * sizeof(int));
    }

    // Debug pixel from env vars
    int dbg_px = -1, dbg_py = -1;
    const char* dpx = getenv("DBG_PX");
    const char* dpy = getenv("DBG_PY");
    if (dpx && dpy) { dbg_px = atoi(dpx); dbg_py = atoi(dpy); }

    for (int ty = 0; ty < grid_y; ty++) {
        for (int tx = 0; tx < grid_x; tx++) {
            int tile_id = ty * grid_x + tx;
            uint32_t range_start = bin.tile_ranges[tile_id * 2];
            uint32_t range_end = bin.tile_ranges[tile_id * 2 + 1];

            int px_min_x = tx * cfg.tile_w;
            int px_min_y = ty * cfg.tile_h;
            int px_max_x = std::min(px_min_x + cfg.tile_w, cam.width);
            int px_max_y = std::min(px_min_y + cfg.tile_h, cam.height);

            for (int py = px_min_y; py < px_max_y; py++) {
                for (int px = px_min_x; px < px_max_x; px++) {
                    bool is_debug = (px == dbg_px && py == dbg_py);
                    float T = 1.0f;
                    float C[3] = {0, 0, 0};
                    float inv_depth = 0.0f;
                    int contrib_count = 0;

                    if (is_debug)
                        printf("=== DEBUG PIXEL (%d,%d) tile(%d,%d) range[%u..%u) (%u gaussians) eval_3D=%d ===\n",
                               px, py, tx, ty, range_start, range_end, range_end - range_start, pre.eval_3D ? 1 : 0);

                    if (pre.eval_3D) {
                        // === StopThePop kBuffer per-pixel sorting (K=16) ===
                        constexpr int KBUF_K = 16;
                        struct KEntry { float depth; uint32_t idx; float alpha; };
                        KEntry kbuf[KBUF_K];
                        int kcount = 0;
                        bool done_pixel = false;

                        // Per-pixel ray direction for depthAlongRay
                        float ray_dir[3] = {0, 0, 1};
                        if (pre.cov3D_inv) {
                            float fpx_r = (float)px + 0.5f;
                            float fpy_r = (float)py + 0.5f;
                            pixelToWorldDir(fpx_r, fpy_r, cam.width, cam.height,
                                            inverse_vp, cam.cam_pos, ray_dir);
                        }

                        for (uint32_t j = range_start; j < range_end && !done_pixel; j++) {
                            uint32_t idx = bin.values_sorted[j];
                            const float* g2s = &pre.gauss2screen[idx * 16];
                            float plane_x[4], plane_y[4];
                            float fpx = (float)px + 0.5f;
                            float fpy = (float)py + 0.5f;
                            for (int k = 0; k < 4; k++) {
                                plane_x[k] = g2s[0*4+k] - g2s[3*4+k] * fpx;
                                plane_y[k] = g2s[1*4+k] - g2s[3*4+k] * fpy;
                            }
                            float max_pos[4];
                            float power = -0.5f * maxContribRayPixel(plane_x, plane_y, max_pos);
                            if (power > 0.0f) continue;

                            float alpha = std::min(0.99f, pre.opacities_2d[idx] * std::exp(power));
                            if (alpha < 1.0f/255.0f) continue;

                            // StopThePop: per-pixel depth via depthAlongRay
                            float pix_depth = pre.depths[idx]; // fallback
                            if (pre.cov3D_inv) {
                                float ptd = depthAlongRay(&pre.cov3D_inv[idx * 6],
                                                          &pre.mean_offset[idx * 3], ray_dir);
                                if (ptd > 0.0f) pix_depth = ptd;
                            }

                            // Pop front if buffer full
                            if (kcount >= KBUF_K) {
                                float a = kbuf[0].alpha;
                                uint32_t gid = kbuf[0].idx;
                                float test_T = T * (1.0f - a);
                                if (test_T < 0.0001f) { done_pixel = true; continue; }
                                for (int ch = 0; ch < 3; ch++)
                                    C[ch] += pre.rgb[gid*3+ch] * a * T;
                                T = test_T;
                                contrib_count++;
                                for (int k = 0; k < KBUF_K - 1; k++) kbuf[k] = kbuf[k+1];
                                kcount = KBUF_K - 1;
                            }
                            // Sorted insertion
                            int pos = kcount;
                            for (int k = kcount - 1; k >= 0; k--) {
                                if (kbuf[k].depth > pix_depth) pos = k; else break;
                            }
                            for (int k = kcount; k > pos; k--) kbuf[k] = kbuf[k-1];
                            kbuf[pos] = {pix_depth, idx, alpha};
                            kcount++;
                        }
                        // Flush remaining
                        for (int k = 0; k < kcount && !done_pixel; k++) {
                            float a = kbuf[k].alpha;
                            uint32_t gid = kbuf[k].idx;
                            float test_T = T * (1.0f - a);
                            if (test_T < 0.0001f) break;
                            for (int ch = 0; ch < 3; ch++)
                                C[ch] += pre.rgb[gid*3+ch] * a * T;
                            T = test_T;
                            contrib_count++;
                        }
                    } else {
                    // === Standard 2D path ===
                    for (uint32_t j = range_start; j < range_end; j++) {
                        uint32_t idx = bin.values_sorted[j];

                        float dx = pre.means2D[idx*2] - (float)px;
                        float dy = pre.means2D[idx*2+1] - (float)py;
                        float con_a = pre.conics[idx*3];
                        float con_b = pre.conics[idx*3+1];
                        float con_c = pre.conics[idx*3+2];
                        float power = -0.5f*(con_a*dx*dx + con_c*dy*dy) - con_b*dx*dy;
                        if (power > 0.0f) continue;

                        float alpha = std::min(0.99f, pre.opacities_2d[idx] * std::exp(power));
                        if (alpha < 1.0f/255.0f) continue;

                        float test_T = T * (1.0f - alpha);

                        if (is_debug && contrib_count < 20) {
                            printf("  [%d] gauss=%u power=%.3f opa=%.4f alpha=%.4f T=%.4f->%.4f "
                                   "rgb=(%.3f,%.3f,%.3f)\n",
                                   contrib_count, idx,
                                   power, pre.opacities_2d[idx], alpha,
                                   T, test_T,
                                   pre.rgb[idx*3], pre.rgb[idx*3+1], pre.rgb[idx*3+2]);
                        }

                        if (test_T < 0.0001f) {
                            if (is_debug) printf("  T saturated at contrib %d\n", contrib_count);
                            break;
                        }

                        for (int ch = 0; ch < 3; ch++)
                            C[ch] += pre.rgb[idx*3+ch] * alpha * T;

                        if (out_depth)
                            inv_depth += (1.0f / pre.depths[idx]) * alpha * T;

                        T = test_T;
                        contrib_count++;
                    }
                    } // end else (2D path)

                    if (is_debug) {
                        printf("  FINAL: %d contributors, T=%.4f, C=(%.4f,%.4f,%.4f)\n",
                               contrib_count, T, C[0], C[1], C[2]);
                        printf("  OUTPUT: (%.4f,%.4f,%.4f) [bg=(%.1f,%.1f,%.1f)]\n",
                               C[0]+T*cfg.bg_color[0], C[1]+T*cfg.bg_color[1], C[2]+T*cfg.bg_color[2],
                               cfg.bg_color[0], cfg.bg_color[1], cfg.bg_color[2]);
                    }

                    int pix = py * cam.width + px;

                    // Save per-pixel cache for backward pass
                    if (cache) {
                        cache->T_final[pix]   = T;
                        cache->n_contrib[pix] = contrib_count;
                    }

                    for (int ch = 0; ch < 3; ch++)
                        out_img[pix*3+ch] = C[ch] + T * cfg.bg_color[ch];
                    if (out_depth)
                        out_depth[pix] = inv_depth;
                }
            }
        }
    }
}
